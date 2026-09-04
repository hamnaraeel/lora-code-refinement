"""Generate predictions and score them.

The same function evaluates the base model and the fine-tuned model; the only
difference between the two calls is ``adapter``. Everything else — prompt,
decoding parameters, parser, metrics — is shared, because a comparison where
the two sides differ in any other respect does not support a claim of
improvement.

Decoding is greedy (``do_sample=False``). Sampling would make every rerun
produce different numbers and would let a lucky seed masquerade as a better
model. Greedy is reproducible and is what a deployment of this kind would use.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from .metrics import aggregate, attach_scores, percentile
from .prompts import build_messages, parse_generation
from .quant import QuantSpec
from .runtime import environment_info, load_model, peak_memory_gb, reset_peak_memory

SPLIT_FILES = {
    "train": "data/processed/train.jsonl",
    "valid": "data/processed/valid.jsonl",
    "test": "data/processed/test.jsonl",
    "benchmark": "data/benchmark/benchmark.jsonl",
}


def load_eval_split(split: str, final: bool) -> list[dict]:
    """Load an evaluation split, guarding the test set.

    The test split is the honest performance measure and is only meaningful if
    it is looked at once, at the end. Requiring an explicit ``--final`` makes
    accidentally tuning against it something you have to do on purpose.
    """
    if split == "test" and not final:
        raise RuntimeError(
            "Refusing to evaluate on the test split without --final. "
            "Use --split valid for model selection and hyperparameter work; "
            "the test split is scored once, after the configuration is frozen."
        )
    path = Path(SPLIT_FILES.get(split, split))
    if not path.exists():
        raise FileNotFoundError(f"No such split: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


@torch.no_grad()
def generate_batch(
    model,
    tokenizer,
    records: list[dict],
    max_new_tokens: int,
    device: str,
) -> list[str]:
    """Greedy-decode one batch, returning only the newly generated text."""
    prompts = [
        tokenizer.apply_chat_template(
            build_messages(r["old_code"], r["comment"], r["lang"]),
            tokenize=False,
            add_generation_prompt=True,
        )
        for r in records
    ]
    encoded = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048,
        add_special_tokens=False,  # the chat template already inserted BOS
    ).to(device)

    output = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        temperature=None,
        top_p=None,
        top_k=None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    # Slice off the prompt by length: with left padding every sequence in the
    # batch has the same prompt width, so this is exact.
    prompt_len = encoded["input_ids"].shape[1]
    return tokenizer.batch_decode(output[:, prompt_len:], skip_special_tokens=True)


def run_evaluation(
    split: str = "valid",
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.3",
    adapter: Path | None = None,
    tag: str = "",
    limit: int | None = None,
    max_new_tokens: int = 512,
    batch_size: int = 4,
    device: str = "auto",
    load_in_4bit: bool = False,
    final: bool = False,
    out_dir: Path = Path("artifacts/eval"),
) -> dict:
    records = load_eval_split(split, final)
    if limit:
        records = records[:limit]

    tag = tag or (Path(adapter).name if adapter else "base")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reset_peak_memory()
    loaded = load_model(
        base_model=base_model,
        device=device,
        quant=QuantSpec(load_in_4bit=load_in_4bit),
        adapter=adapter,
    )
    model, tokenizer, resolved_device = loaded.model, loaded.tokenizer, loaded.device

    predictions: list[dict] = []
    latencies: list[float] = []
    start = time.time()

    for i in tqdm(range(0, len(records), batch_size), desc=f"generate[{tag}/{split}]"):
        batch = records[i : i + batch_size]
        t0 = time.time()
        try:
            raw_outputs = generate_batch(
                model, tokenizer, batch, max_new_tokens, resolved_device
            )
        except torch.cuda.OutOfMemoryError:  # pragma: no cover - GPU only
            torch.cuda.empty_cache()
            raw_outputs = [
                generate_batch(model, tokenizer, [r], max_new_tokens, resolved_device)[0]
                for r in batch
            ]
        elapsed = time.time() - t0
        latencies.extend([elapsed / len(batch)] * len(batch))

        for rec, raw in zip(batch, raw_outputs):
            parsed = parse_generation(raw)
            predictions.append(
                {
                    "id": rec["id"],
                    "lang": rec["lang"],
                    "category": rec["category"],
                    "repo": rec.get("repo", ""),
                    "origin": rec.get("origin", split),
                    "probe": rec.get("probe", ""),
                    "comment": rec["comment"],
                    "old_code": rec["old_code"],
                    "gold": rec["new_code"],
                    "prediction": parsed.code,
                    "parse_mode": parsed.how,
                    "raw_output": raw,
                }
            )

    duration = time.time() - start
    attach_scores(predictions)
    metrics = aggregate(predictions)

    latencies.sort()
    summary = {
        "tag": tag,
        "split": split,
        "base_model": base_model,
        "adapter": str(adapter) if adapter else None,
        "n": len(predictions),
        "metrics": metrics,
        "generation": {
            "decoding": "greedy",
            "max_new_tokens": max_new_tokens,
            "batch_size": batch_size,
            "device": resolved_device,
            "load_in_4bit": load_in_4bit,
            "wall_clock_s": round(duration, 2),
            "latency_p50_s": round(percentile(latencies, 0.50), 3) if latencies else None,
            "latency_p95_s": round(percentile(latencies, 0.95), 3) if latencies else None,
            "peak_gpu_memory_gb": peak_memory_gb(),
        },
        "env": environment_info(),
    }

    stem = f"{tag}__{split}"
    pred_path = out_dir / f"{stem}.predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as fh:
        for row in predictions:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out_dir / f"{stem}.summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    summary["predictions_path"] = str(pred_path)
    return summary
