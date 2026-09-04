"""Device, dtype and model-loading helpers shared by training, eval and serving.

Isolated in one module because this project has to run in three very different
places — a CUDA box doing QLoRA on a 7B model, a Colab T4, and an Intel Mac
doing a CPU smoke run — and every stage needs to make the same choices. When
loading logic is duplicated per script, the base and fine-tuned models end up
loaded with different dtypes and the comparison silently stops being fair.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


def resolve_device(requested: str = "auto") -> str:
    """Pick a device.

    ``auto`` deliberately does **not** choose MPS. Apple's Metal backend
    reports as available on Intel Macs with a discrete GPU, but several ops
    used by 4-bit/quantised paths and by gradient checkpointing either fall
    back to CPU or produce NaNs there. Silently training on a backend that
    yields garbage is worse than being slow, so MPS must be asked for by name.
    """
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_dtype(requested: str, device: str) -> torch.dtype:
    if requested and requested != "auto":
        return getattr(torch, requested)
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # CPU bf16 exists but is glacial on pre-AVX512 hardware, and fp16 has no
    # CPU kernels for most ops. fp32 is the only sane CPU choice.
    return torch.float32


def resolve_attn(requested: str, device: str) -> str:
    if requested and requested != "auto":
        return requested
    if device == "cuda":
        try:
            import flash_attn  # noqa: F401

            return "flash_attention_2"
        except ImportError:
            return "sdpa"
    return "eager"


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any
    device: str
    dtype: torch.dtype
    adapter: str | None = None


def load_tokenizer(base_model: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=False)
    if tok.pad_token is None:
        # Most causal-LM checkpoints ship no pad token. Reusing EOS is the
        # standard fix; the collator masks pad positions out of the loss, so
        # the model never learns to predict padding.
        tok.pad_token = tok.eos_token
    # Left padding for generation: with right padding, the last real token of a
    # short sequence is no longer at position -1 and greedy decoding continues
    # from a pad. This single line is a common source of "the base model looks
    # terrible" artefacts in comparisons like this one.
    tok.padding_side = "left"
    return tok


def build_quant_config(quant, dtype: torch.dtype, device: str):
    """4-bit NF4 config, or ``None`` when quantisation is off or unsupported."""
    if not getattr(quant, "load_in_4bit", False):
        return None
    if device != "cuda":
        raise RuntimeError(
            "load_in_4bit requires CUDA — bitsandbytes has no CPU or MPS build. "
            "Run the QLoRA config on a GPU host, or drop quant.load_in_4bit for "
            "a local smoke run."
        )
    from transformers import BitsAndBytesConfig

    # bnb_4bit_compute_dtype must track actual hardware support, not the config
    # file's static default: bfloat16 has no native acceleration on pre-Ampere
    # GPUs (P100/Pascal has none at all; T4/Turing lacks bf16 tensor cores too,
    # that started at Ampere/cc8.0). Forcing bf16 there doesn't error — it just
    # runs the 4-bit dequant+matmul, which happens on every forward AND
    # backward pass, through an unaccelerated fallback path, making training
    # steps dramatically slower than eval (forward-only) steps. `dtype` here
    # already went through resolve_dtype()'s torch.cuda.is_bf16_supported()
    # check, so reuse it instead of reading quant.compute_dtype from the YAML.
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant.quant_type,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=quant.double_quant,
    )


def load_model(
    base_model: str,
    device: str = "auto",
    dtype: str = "auto",
    attn: str = "auto",
    quant=None,
    adapter: str | Path | None = None,
    for_training: bool = False,
) -> LoadedModel:
    """Load a base model, optionally attaching a LoRA adapter.

    The same function serves evaluation of the base model (``adapter=None``)
    and of the tuned model (``adapter=<dir>``), which is what guarantees the
    two sides of every comparison differ only by the adapter.
    """
    from transformers import AutoModelForCausalLM

    device = resolve_device(device)
    torch_dtype = resolve_dtype(dtype, device)
    quant_config = build_quant_config(quant, torch_dtype, device) if quant else None

    kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "attn_implementation": resolve_attn(attn, device),
        "low_cpu_mem_usage": True,
    }
    if quant_config is not None:
        kwargs["quantization_config"] = quant_config
        kwargs["device_map"] = {"": 0}
    elif device == "cuda":
        kwargs["device_map"] = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
    if quant_config is None and device != "cuda":
        model = model.to(device)

    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=for_training)

    if not for_training:
        model.eval()
        model.config.use_cache = True

    tok = load_tokenizer(base_model)
    return LoadedModel(model, tok, device, torch_dtype, str(adapter) if adapter else None)


# ---------------------------------------------------------------------------
# Environment capture — part of "reproducible from the config alone"
# ---------------------------------------------------------------------------


def environment_info() -> dict:
    """Everything about the host that could change a number in the report."""
    import transformers

    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "cpu_threads": torch.get_num_threads(),
    }
    try:
        import peft

        info["peft"] = peft.__version__
    except ImportError:
        pass
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_total_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 2
        )
        info["cuda_version"] = torch.version.cuda
    for var in ("SLURM_JOB_ID", "COLAB_GPU", "HOSTNAME"):
        if os.environ.get(var):
            info[var.lower()] = os.environ[var]
    return info


def peak_memory_gb() -> float | None:
    """Peak allocated GPU memory since the last reset, in GB."""
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / 1e9, 3)


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
