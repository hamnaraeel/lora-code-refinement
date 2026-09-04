"""LoRA / QLoRA fine-tuning.

Reproducibility contract
------------------------
``python -m coderefine train configs/<x>.yaml`` is the whole interface. The
config file fixes the model, the LoRA shape, the optimiser, the data and the
seed; the run directory afterwards contains the resolved config, its
fingerprint, the full metric history, the environment, and the adapter. Nothing
that affects the result is passed any other way.

Loss masking
------------
Only the assistant turn is trained on. Computing loss over the prompt would
spend most of the gradient budget teaching the model to reproduce code it was
already given — which on this task is 90% of the target text, and would produce
an adapter that has mostly learned to copy.

TRL's masking mechanism changed between releases in a way that is not just a
renamed argument: older TRL exposes ``DataCollatorForCompletionOnlyLM``, which
masks everything up to a literal response-template string matched against a
pre-rendered ``"text"`` column. Newer TRL removed that collator entirely in
favour of ``SFTConfig(assistant_only_loss=True)``, which requires a
conversational ``"messages"`` column and derives the mask from the tokenizer's
chat template (via ``apply_chat_template(..., return_assistant_tokens_mask=True)``),
inserting `{% generation %}` markers itself if the template lacks them. Both
paths are supported here, selected at runtime by whichever the installed TRL
actually provides — this project has to run against whatever TRL a Colab
runtime installs today *and* the older pinned stack on a CPU-only host, and a
silently-adopted argument rename is exactly the kind of drift that would
otherwise turn a run into an expensive no-op with no error.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .config import RunConfig
from .data.build import load_split
from .prompts import build_training_messages
from .runtime import (
    environment_info,
    load_tokenizer,
    peak_memory_gb,
    reset_peak_memory,
    resolve_attn,
    resolve_device,
    resolve_dtype,
)
from .tracking import build_tracker


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------


def render_dataset(records: list[dict], tokenizer) -> "Any":
    """Render curated examples into single training strings via the chat template.

    Used on the legacy TRL path (``DataCollatorForCompletionOnlyLM``). Using the
    tokenizer's own chat template rather than a hand-rolled string is what keeps
    training consistent with generation: at inference we call
    ``apply_chat_template(..., add_generation_prompt=True)``, so the tokens the
    model sees before the answer are byte-identical to the ones it was trained on.
    """
    from datasets import Dataset

    rows = []
    for rec in records:
        messages = build_training_messages(
            rec["old_code"], rec["comment"], rec["new_code"], rec["lang"]
        )
        text = _apply_template(tokenizer, messages)
        rows.append({"text": text, "id": rec["id"], "lang": rec["lang"], "category": rec["category"]})
    return Dataset.from_list(rows)


def render_conversational_dataset(records: list[dict]) -> "Any":
    """Render curated examples as a ``messages`` column, for the modern TRL path.

    No template is applied here — that is deliberate. ``assistant_only_loss``
    has TRL itself call ``apply_chat_template`` internally (inserting
    `{% generation %}` markers into the template if it lacks them), and doing
    it twice would double-apply the template on the second run.
    """
    from datasets import Dataset

    rows = []
    for rec in records:
        messages = build_training_messages(
            rec["old_code"], rec["comment"], rec["new_code"], rec["lang"]
        )
        rows.append(
            {"messages": messages, "id": rec["id"], "lang": rec["lang"], "category": rec["category"]}
        )
    return Dataset.from_list(rows)


def _apply_template(tokenizer, messages: list[dict[str, str]]) -> str:
    """Apply the chat template, folding the system turn in when unsupported.

    Several instruct checkpoints (Mistral v0.1/v0.2, Gemma) raise on a
    ``system`` role. Rather than dropping the system prompt — which would
    change the task definition between models — it is prepended to the first
    user turn, preserving the content.
    """
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False)
    except Exception:  # noqa: BLE001 - template errors are TemplateError, ValueError, ...
        merged: list[dict[str, str]] = []
        pending_system = ""
        for msg in messages:
            if msg["role"] == "system":
                pending_system = msg["content"]
                continue
            if pending_system and msg["role"] == "user":
                merged.append({"role": "user", "content": pending_system + "\n\n" + msg["content"]})
                pending_system = ""
            else:
                merged.append(dict(msg))
        return tokenizer.apply_chat_template(merged, tokenize=False)


def find_response_template(tokenizer) -> str:
    """Locate the literal string that marks the start of the assistant turn.

    Derived by diffing a rendered conversation against the same conversation
    with the assistant turn removed, rather than hard-coding ``[/INST]`` or
    ``<|start_header_id|>assistant<|end_header_id|>``. Hard-coding is how a
    pipeline silently breaks when someone swaps Mistral for Llama 3 — the
    collator finds no match, masks nothing, and the model trains on the prompt.
    """
    probe = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": "AAAA"},
    ]
    with_answer = _apply_template(tokenizer, probe)
    marker = with_answer.find("AAAA")
    if marker == -1:
        raise RuntimeError(
            "Could not locate the assistant turn in the chat template; "
            "completion-only loss masking would silently no-op."
        )
    prefix = with_answer[:marker]
    # Walk back to just after the user content to isolate the assistant header.
    user_end = prefix.rfind("U")
    template = prefix[user_end + 1 :] if user_end != -1 else prefix
    # Trailing whitespace in the template (e.g. the space in "[/INST] ") is not
    # a stable token boundary: SentencePiece merges it into the first token of
    # whatever follows (real targets start with a code fence, so "[/INST] ```"
    # tokenizes with that space fused into a "▁```" token, not as a standalone
    # space token). Matching on that fused token-id sequence then never occurs
    # in real data, so the collator silently fails to find the response key on
    # (in practice) every example. Stripping trailing whitespace keeps the
    # match anchored to the structural token(s) that are always split out on
    # their own, and lets the fused leading-space+content token land inside
    # the (correctly loss-visible) completion instead of the template.
    template = template.rstrip()
    if not template:
        raise RuntimeError("Empty response template derived from chat template.")
    return template


def _check_response_template_coverage(tokenizer, dataset, response_template: str, sample_size: int = 200) -> None:
    """Fail loudly if the response template's token ids don't actually occur in real examples.

    ``find_response_template`` derives the template from a synthetic probe; a
    real chat template can still tokenize it differently once real content
    follows, which is exactly the bug this guards against (see the trailing
    ` ` note above the caller). Silently masking every token as prompt would
    otherwise train on data with an empty loss and look like a normal run.
    """
    template_ids = tokenizer.encode(response_template, add_special_tokens=False)
    text_field = "text" if "text" in dataset.column_names else dataset.column_names[0]
    n = min(sample_size, len(dataset))
    misses = 0
    for i in range(n):
        ids = tokenizer.encode(dataset[i][text_field], add_special_tokens=False)
        found = any(ids[j : j + len(template_ids)] == template_ids for j in range(len(ids) - len(template_ids) + 1))
        if not found:
            misses += 1
    miss_rate = misses / n
    print(f"[train] response template coverage: {n - misses}/{n} examples matched ({100 * (1 - miss_rate):.1f}%)")
    if miss_rate > 0.02:
        raise RuntimeError(
            f"Response template {response_template!r} (ids={template_ids}) was not found in "
            f"{misses}/{n} sampled training examples ({100 * miss_rate:.1f}% miss rate). "
            "Completion-only loss masking would silently no-op (or near-no-op) on this data — "
            "refusing to train rather than produce a run with no real learning signal."
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# TRL compatibility
# ---------------------------------------------------------------------------
# TRL's SFT API has churned: `SFTTrainer(tokenizer=...)` became
# `processing_class=`, `SFTConfig.max_seq_length` became `max_length`, and
# `dataset_text_field` / `packing` moved around. This project has to run against
# whatever TRL a Colab runtime installs today *and* the pinned older stack that
# is the only thing available on an Intel Mac, so arguments are filtered against
# the installed signature instead of being pinned to one version. Anything the
# installed version does not accept is reported once rather than silently
# dropped — a quietly ignored `max_seq_length` would truncate targets and show
# up only as a slightly worse score.


def _supported_fields(cls) -> set[str]:
    """Field names the installed dataclass or __init__ signature accepts."""
    import dataclasses
    import inspect

    if dataclasses.is_dataclass(cls):
        return {f.name for f in dataclasses.fields(cls)}
    return set(inspect.signature(cls.__init__).parameters) - {"self"}


#: Renames applied when the modern name is absent. Maps our canonical name to
#: the older name that means the same thing.
_SFT_ALIASES = {"max_seq_length": ("max_length",)}


def _filter_kwargs(cls, kwargs: dict[str, Any], what: str) -> dict[str, Any]:
    supported = _supported_fields(cls)
    out: dict[str, Any] = {}
    dropped: list[str] = []
    for key, value in kwargs.items():
        if key in supported:
            out[key] = value
            continue
        alias = next((a for a in _SFT_ALIASES.get(key, ()) if a in supported), None)
        if alias is None:
            alias = next(
                (k for k, v in _SFT_ALIASES.items() if key in v and k in supported), None
            )
        if alias:
            out[alias] = value
        else:
            dropped.append(key)
    if dropped:
        print(f"[train] note: {what} does not accept {dropped} in this version; not set.")
    return out


def run_training(cfg: RunConfig) -> dict:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        EarlyStoppingCallback,
        TrainerCallback,
    )
    from trl import SFTConfig, SFTTrainer

    DataCollatorForCompletionOnlyLM = None
    try:
        from trl import DataCollatorForCompletionOnlyLM
    except ImportError:
        try:
            from trl.trainer.utils import DataCollatorForCompletionOnlyLM
        except ImportError:
            pass  # removed in this TRL release; fall back to assistant_only_loss below
    use_legacy_collator = DataCollatorForCompletionOnlyLM is not None
    print(
        f"[train] TRL masking mode: "
        f"{'legacy DataCollatorForCompletionOnlyLM' if use_legacy_collator else 'assistant_only_loss (modern TRL)'}"
    )

    from .runtime import build_quant_config

    torch.manual_seed(cfg.train.seed)
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "config.yaml")

    device = resolve_device(cfg.device)
    dtype = resolve_dtype(cfg.dtype, device)
    env = environment_info()

    tracker = build_tracker(
        cfg.tracker, cfg.project, cfg.name, {**cfg.to_dict(), "env": env}, run_dir
    )
    tracker.log_summary({"config_fingerprint": cfg.fingerprint(), **{f"env.{k}": v for k, v in env.items()}})

    # --- data ------------------------------------------------------------
    data_dir = Path(cfg.data_dir)
    train_records = load_split(data_dir / "train.jsonl")
    valid_records = load_split(data_dir / "valid.jsonl")
    if cfg.train.max_train_samples:
        train_records = train_records[: cfg.train.max_train_samples]
    if cfg.train.max_eval_samples:
        valid_records = valid_records[: cfg.train.max_eval_samples]

    tokenizer = load_tokenizer(cfg.base_model)
    # Right padding during training: the completion-only collator locates the
    # response template by scanning forward, and left padding shifts it.
    tokenizer.padding_side = "right"

    if use_legacy_collator:
        train_ds = render_dataset(train_records, tokenizer)
        valid_ds = render_dataset(valid_records, tokenizer)
    else:
        train_ds = render_conversational_dataset(train_records)
        valid_ds = render_conversational_dataset(valid_records)
    _report_length_stats(train_ds, tokenizer, cfg, tracker)

    # --- model -----------------------------------------------------------
    quant_config = build_quant_config(cfg.quant, dtype, device) if cfg.quant.load_in_4bit else None
    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "attn_implementation": resolve_attn(cfg.attn_implementation, device),
        "low_cpu_mem_usage": True,
    }
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
        model_kwargs["device_map"] = {"": 0}
    elif device == "cuda":
        model_kwargs["device_map"] = {"": 0}

    reset_peak_memory()
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, **model_kwargs)
    if quant_config is None and device != "cuda":
        model = model.to(device)
    model.config.use_cache = False  # incompatible with gradient checkpointing

    if quant_config is not None:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.train.gradient_checkpointing
        )

    lora_config = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        bias=cfg.lora.bias,
        task_type=cfg.lora.task_type,
        target_modules=_resolve_targets(cfg.lora.target_modules, model),
    )
    model = get_peft_model(model, lora_config)
    trainable, total = _count_params(model)
    print(f"[train] trainable {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")
    tracker.log_summary(
        {
            "trainable_params": trainable,
            "total_params": total,
            "trainable_pct": round(100 * trainable / total, 4),
            "device": device,
            "dtype": str(dtype),
            "n_train": len(train_ds),
            "n_valid": len(valid_ds),
        }
    )

    # --- schedule --------------------------------------------------------
    effective_batch = cfg.train.per_device_batch_size * cfg.train.gradient_accumulation_steps
    steps_per_epoch = max(1, math.ceil(len(train_ds) / effective_batch))
    eval_steps = max(1, steps_per_epoch // max(1, cfg.train.evals_per_epoch))

    collator = None
    if use_legacy_collator:
        response_template = find_response_template(tokenizer)
        collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template, tokenizer=tokenizer
        )
        _check_response_template_coverage(tokenizer, train_ds, response_template)

    sft_kwargs: dict[str, Any] = dict(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=cfg.train.num_epochs,
        per_device_train_batch_size=cfg.train.per_device_batch_size,
        per_device_eval_batch_size=cfg.train.per_device_batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        learning_rate=cfg.train.learning_rate,
        lr_scheduler_type=cfg.train.lr_scheduler,
        warmup_ratio=cfg.train.warmup_ratio,
        weight_decay=cfg.train.weight_decay,
        max_grad_norm=cfg.train.max_grad_norm,
        optim=cfg.train.optim,
        gradient_checkpointing=cfg.train.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_seq_length=cfg.train.max_seq_length,
        # Only meaningful on the legacy (flat "text" column) path; the modern
        # path uses assistant_only_loss against the "messages" column instead,
        # and _filter_kwargs drops assistant_only_loss on TRL versions too old
        # to accept it, so this dict works unmodified on either.
        dataset_text_field="text" if use_legacy_collator else None,
        assistant_only_loss=not use_legacy_collator,
        packing=False,  # packing would cross example boundaries and defeat masking
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        logging_steps=max(1, eval_steps // 4),
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],  # tracking is handled by our own callback, see below
        # Without this the Trainer picks its own device and will grab Apple MPS
        # whenever torch reports it available — overriding `device: cpu` in the
        # config, and on Intel Macs promptly hitting the ~3.4 GB Metal
        # watermark. The device decision belongs to resolve_device(), so it is
        # forced through here rather than left to the Trainer's autodetection.
        use_cpu=(device == "cpu"),
        seed=cfg.train.seed,
        data_seed=cfg.train.seed,
        dataloader_num_workers=0,
        remove_unused_columns=True,
    )
    sft_args = SFTConfig(**_filter_kwargs(SFTConfig, sft_kwargs, "SFTConfig"))

    history: list[dict] = []

    class TrackingCallback(TrainerCallback):
        """Bridge HF Trainer logs into the tracker, adding GPU memory."""

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            payload = dict(logs)
            mem = peak_memory_gb()
            if mem is not None:
                payload["gpu_peak_memory_gb"] = mem
            payload["epoch"] = state.epoch
            tracker.log(payload, step=state.global_step)
            history.append({"step": state.global_step, **payload})

    callbacks = [TrackingCallback()]
    if cfg.train.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=cfg.train.early_stopping_patience,
                early_stopping_threshold=cfg.train.early_stopping_threshold,
            )
        )

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": sft_args,
        "train_dataset": train_ds,
        "eval_dataset": valid_ds,
        "callbacks": callbacks,
    }
    if collator is not None:
        # Modern TRL derives assistant-only masking internally from the
        # "messages" column; passing no collator lets it use its own default
        # rather than one built for the legacy flat-text path.
        trainer_kwargs["data_collator"] = collator
    # `tokenizer=` was renamed to `processing_class=`; prefer whichever exists.
    trainer_fields = _supported_fields(SFTTrainer)
    trainer_kwargs["processing_class" if "processing_class" in trainer_fields else "tokenizer"] = tokenizer
    trainer = SFTTrainer(**trainer_kwargs)

    start = time.time()
    train_result = trainer.train()
    duration = time.time() - start

    # --- adapter export --------------------------------------------------
    adapter_dir = run_dir / "adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    best_step, best_loss, selection_reason = _describe_selection(trainer, history)
    final_eval = trainer.evaluate()

    summary = {
        "run_name": cfg.name,
        "config_fingerprint": cfg.fingerprint(),
        "base_model": cfg.base_model,
        "adapter_dir": str(adapter_dir),
        "adapter_size_mb": round(_dir_size(adapter_dir) / 1e6, 2),
        "train_runtime_s": round(duration, 1),
        "train_samples_per_second": round(len(train_ds) * cfg.train.num_epochs / max(duration, 1e-9), 3),
        "final_train_loss": train_result.metrics.get("train_loss"),
        "best_eval_loss": best_loss,
        "best_checkpoint_step": best_step,
        "checkpoint_selection_reason": selection_reason,
        "final_eval_loss": final_eval.get("eval_loss"),
        "eval_perplexity": round(math.exp(final_eval["eval_loss"]), 4)
        if final_eval.get("eval_loss") is not None and final_eval["eval_loss"] < 20
        else None,
        "steps_per_epoch": steps_per_epoch,
        "eval_every_n_steps": eval_steps,
        "effective_batch_size": effective_batch,
        "gpu_peak_memory_gb": peak_memory_gb(),
        "early_stopped": trainer.state.global_step
        < steps_per_epoch * cfg.train.num_epochs - 1,
        "trainable_params": trainable,
    }
    tracker.log_summary(summary)
    tracker.log_artifact(adapter_dir, name=f"{cfg.name}-adapter", kind="lora-adapter")
    tracker.finish()

    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (run_dir / "train_summary.json").write_text(
        json.dumps({**summary, "env": env, "config": cfg.to_dict()}, indent=2, default=str),
        encoding="utf-8",
    )
    return summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_targets(targets: list[str], model) -> list[str] | str:
    """Expand the ``all-linear`` shorthand and drop names the model lacks.

    Target module names differ across architectures (Mistral/Llama use
    ``q_proj``; GPT-NeoX uses ``query_key_value``). Passing a name the model
    does not have raises deep inside PEFT with an unhelpful message, so the
    check happens here where a clear error is possible.
    """
    if len(targets) == 1 and targets[0] == "all-linear":
        return "all-linear"
    present = {name.split(".")[-1] for name, _ in model.named_modules()}
    matched = [t for t in targets if t in present]
    missing = [t for t in targets if t not in present]
    if missing:
        candidates = sorted(n for n in present if "proj" in n or "attn" in n or "fc" in n)
        if not matched:
            raise ValueError(
                f"None of the requested LoRA target modules {targets} exist in this model. "
                f"Candidates: {candidates}"
            )
        print(f"[train] warning: target modules not found and skipped: {missing}")
    return matched


def _count_params(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())


def _describe_selection(trainer, history: list[dict]) -> tuple[int | None, float | None, str]:
    """Say which checkpoint was kept and why, in words the report can quote."""
    losses = [(h["step"], h["eval_loss"]) for h in history if "eval_loss" in h]
    if not losses:
        return None, None, "no evaluation ran; final weights kept"
    best_step, best_loss = min(losses, key=lambda x: x[1])
    last_step = losses[-1][0]
    if best_step == last_step:
        reason = (
            f"validation loss was still at its minimum ({best_loss:.4f}) at the final "
            f"evaluation (step {best_step}); training ended on the epoch budget, not early stopping"
        )
    else:
        worsened = len([s for s, _ in losses if s > best_step])
        reason = (
            f"validation loss bottomed at {best_loss:.4f} on step {best_step} and rose over the "
            f"following {worsened} evaluation(s); that checkpoint was restored "
            f"(load_best_model_at_end)"
        )
    return best_step, best_loss, reason


def _report_length_stats(dataset, tokenizer, cfg: RunConfig, tracker) -> None:
    """Warn loudly if max_seq_length is silently truncating targets.

    Truncation here is insidious: it removes the *end* of the assistant turn,
    so the model is trained to produce code that stops mid-statement, and the
    only symptom is a slightly worse exact-match score.
    """
    if "text" in dataset.column_names:
        lengths = [len(tokenizer(row["text"]).input_ids) for row in dataset]
    else:
        # apply_chat_template(tokenize=True) can return either a bare list of
        # ids or a BatchEncoding, depending on the installed transformers
        # version. len() on a BatchEncoding counts its keys (input_ids,
        # attention_mask, ...) rather than tokens, silently reporting every
        # sequence as length ~2 — normalise to a plain id list either way.
        def _n_tokens(messages: list[dict]) -> int:
            encoded = tokenizer.apply_chat_template(messages)
            ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
            return len(ids)

        lengths = [_n_tokens(row["messages"]) for row in dataset]
    lengths.sort()
    n = len(lengths)
    over = sum(1 for x in lengths if x > cfg.train.max_seq_length)
    stats = {
        "seq_len_mean": round(sum(lengths) / n, 1),
        "seq_len_p50": lengths[n // 2],
        "seq_len_p95": lengths[int(0.95 * n)],
        "seq_len_max": lengths[-1],
        "seq_len_truncated_pct": round(100 * over / n, 2),
    }
    print(f"[train] sequence lengths: {stats}")
    tracker.log_summary(stats)
    if over / n > 0.05:
        print(
            f"[train] WARNING: {stats['seq_len_truncated_pct']}% of examples exceed "
            f"max_seq_length={cfg.train.max_seq_length}; targets are being cut off. "
            f"Raise max_seq_length to at least {lengths[int(0.99 * n)]}."
        )
