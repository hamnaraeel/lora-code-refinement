"""Package a trained LoRA adapter for deployment.

The adapter is the deliverable, not the model. At rank 16 on q/v projections of
a 7B base it is a few megabytes against ~15 GB of base weights, and keeping the
two separate is what makes it possible to serve several domain-specific models
from one resident base — swap the adapter, keep the weights.

``--merge`` is offered for the cases that need a single artefact (a GGUF export
for Ollama, a runtime with no PEFT support), with the size cost stated plainly
rather than buried.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def export_adapter(
    adapter_dir: Path,
    out_dir: Path,
    merge: bool = False,
    base_model: str | None = None,
) -> dict:
    adapter_dir = Path(adapter_dir)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"No adapter at {adapter_dir}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    adapter_out = out_dir / "adapter"
    if adapter_out.exists():
        shutil.rmtree(adapter_out)
    shutil.copytree(adapter_dir, adapter_out)

    peft_config_path = adapter_out / "adapter_config.json"
    peft_config = json.loads(peft_config_path.read_text()) if peft_config_path.exists() else {}
    resolved_base = base_model or peft_config.get("base_model_name_or_path", "")

    # Checksums so a deployed adapter can be verified against the trained one.
    checksums = {
        f.relative_to(adapter_out).as_posix(): _sha256(f)
        for f in sorted(adapter_out.rglob("*"))
        if f.is_file() and f.suffix in {".safetensors", ".bin", ".json"}
    }

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": resolved_base,
        "adapter_dir": str(adapter_out),
        "adapter_size_mb": round(_dir_size(adapter_out) / 1e6, 3),
        "lora": {
            "r": peft_config.get("r"),
            "alpha": peft_config.get("lora_alpha"),
            "dropout": peft_config.get("lora_dropout"),
            "target_modules": peft_config.get("target_modules"),
        },
        "checksums_sha256": checksums,
        "task": "code_refinement_from_review_comment",
        "prompt_contract": (
            "Use coderefine.prompts.build_messages(old_code, comment, lang) to construct "
            "the chat turns. The adapter was trained on that exact template; a different "
            "system prompt or user layout will measurably degrade output quality."
        ),
    }

    _write_modelfile(out_dir, resolved_base)
    _write_vllm_snippet(out_dir, resolved_base, adapter_out)
    _write_readme(out_dir, manifest)

    if merge:
        merged_dir = out_dir / "merged"
        manifest["merged_model"] = merge_adapter(adapter_dir, merged_dir, resolved_base)

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def merge_adapter(adapter_dir: Path, out_dir: Path, base_model: str) -> dict:
    """Fold the adapter into the base weights and write a standalone model.

    Loaded in float16 on CPU: merging is a weight-space operation with no
    activations, so it does not need a GPU, but it does need enough RAM for the
    full model. The result is base-model-sized, which is the whole reason the
    adapter is the default artefact.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not base_model:
        raise ValueError("merge requires a base model; pass --base-model.")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model = model.merge_and_unload()
    model.save_pretrained(str(out_dir), safe_serialization=True)
    AutoTokenizer.from_pretrained(base_model).save_pretrained(str(out_dir))

    return {"path": str(out_dir), "size_mb": round(_dir_size(out_dir) / 1e6, 1)}


def _write_modelfile(out_dir: Path, base_model: str) -> None:
    """Ollama Modelfile. Ollama needs GGUF, so this documents the conversion."""
    (out_dir / "Modelfile").write_text(
        f"""# Ollama deployment for the code-refinement adapter.
#
# Ollama loads GGUF, so an adapter trained with PEFT has to be converted first:
#
#   1. Merge the adapter into the base weights:
#        coderefine export artifacts/runs/<run>/adapter --merge \\
#            --base-model {base_model}
#   2. Convert the merged model to GGUF with llama.cpp:
#        python llama.cpp/convert_hf_to_gguf.py artifacts/release/merged \\
#            --outfile coderefine.gguf --outtype q4_k_m
#   3. Build and run:
#        ollama create coderefine -f Modelfile
#        ollama run coderefine
#
# The SYSTEM block below must stay byte-identical to
# coderefine.prompts.SYSTEM_PROMPT — the adapter was trained against it.

FROM ./coderefine.gguf

PARAMETER temperature 0
PARAMETER top_p 1
PARAMETER num_ctx 4096
PARAMETER stop "<|im_end|>"

SYSTEM \"\"\"You are an expert software engineer applying code review feedback. \
You are given a snippet of code and a reviewer's comment about it. Rewrite the \
snippet so that it addresses the comment.

Rules:
1. Output ONLY the revised code inside a single fenced code block.
2. Change exactly what the comment asks for and nothing else. Preserve all \
unrelated lines, including their original indentation and whitespace.
3. Do not add explanations, commentary, or extra code fences.
4. The snippet is an excerpt from a larger file. It may be syntactically \
incomplete; keep it that way rather than closing braces or adding imports.\"\"\"
""",
        encoding="utf-8",
    )


def _write_vllm_snippet(out_dir: Path, base_model: str, adapter_out: Path) -> None:
    (out_dir / "serve_vllm.sh").write_text(
        f"""#!/usr/bin/env bash
# Production-grade batched serving with a runtime-attached LoRA adapter.
#
# vLLM keeps one copy of the base weights resident and applies the adapter per
# request, so base and fine-tuned traffic can be served from a single process —
# which is exactly what the /ab comparison endpoint needs.
set -euo pipefail

vllm serve "{base_model}" \\
  --enable-lora \\
  --lora-modules coderefine="{adapter_out.resolve()}" \\
  --max-lora-rank 32 \\
  --dtype bfloat16 \\
  --max-model-len 4096 \\
  --port 8000

# Then address either model by name against the OpenAI-compatible endpoint:
#   curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \\
#     -d '{{"model": "coderefine", "messages": [...], "temperature": 0}}'
#   curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \\
#     -d '{{"model": "{base_model}", "messages": [...], "temperature": 0}}'
""",
        encoding="utf-8",
    )
    (out_dir / "serve_vllm.sh").chmod(0o755)


def _write_readme(out_dir: Path, manifest: dict) -> None:
    lora = manifest["lora"]
    (out_dir / "README.md").write_text(
        f"""# Code-refinement LoRA adapter

Applies a code-review comment to a code snippet and returns the revised snippet.

| | |
|---|---|
| Base model | `{manifest['base_model']}` |
| Adapter size | {manifest['adapter_size_mb']} MB |
| LoRA rank / alpha | {lora['r']} / {lora['alpha']} |
| Target modules | `{lora['target_modules']}` |
| Built | {manifest['created_utc']} |

## Loading

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM
from coderefine.prompts import build_messages

model = AutoModelForCausalLM.from_pretrained("{manifest['base_model']}")
model = PeftModel.from_pretrained(model, "./adapter")

messages = build_messages(old_code, review_comment, "py")
```

## Prompt contract

{manifest['prompt_contract']}

## Serving

* `serve_vllm.sh` — vLLM with the adapter attached at runtime (recommended).
* `Modelfile` — Ollama, via a merge-then-GGUF conversion (steps in the file).
* `coderefine serve --adapter ./adapter` — the bundled FastAPI A/B server.
""",
        encoding="utf-8",
    )
