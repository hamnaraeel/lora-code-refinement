"""Catastrophic forgetting analysis.

Fine-tuning on 2,000 narrow examples can teach a model that *every* prompt is a
code-refinement prompt. The failure is easy to miss: task metrics go up, the
demo looks great, and the model has quietly lost the ability to answer a
question without wrapping it in a fenced code block.

Two probes, measuring different things:

* **Multiple choice** — scored by length-normalised log-likelihood over the
  answer options rather than by generation. No decoding, no parsing, fully
  deterministic, and it measures what the model *knows* independently of whether
  it still follows instructions. This is the protocol lm-evaluation-harness uses.
* **Instruction following** — generated and rule-checked. This is the one that
  catches format collapse, which is the regression LoRA fine-tuning on a
  single output shape actually causes.

``--use-lm-eval`` additionally runs real lm-evaluation-harness tasks when the
package is installed; the bundled suite is the offline fallback so the analysis
is never skipped for want of a download.
"""

from __future__ import annotations

import json
import re
from importlib import resources
from pathlib import Path

import torch
from tqdm import tqdm

from .quant import QuantSpec
from .runtime import load_model


def load_probes() -> dict:
    with resources.files("coderefine.assets").joinpath("forgetting_probes.json").open(
        "r", encoding="utf-8"
    ) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Multiple choice via log-likelihood
# ---------------------------------------------------------------------------


@torch.no_grad()
def _option_logprob(model, tokenizer, context: str, option: str, device: str) -> tuple[float, int]:
    """Total log-probability of ``option`` continuing ``context``."""
    ctx_ids = tokenizer(context, return_tensors="pt", add_special_tokens=True).input_ids
    full_ids = tokenizer(context + option, return_tensors="pt", add_special_tokens=True).input_ids
    full_ids = full_ids.to(device)

    n_ctx = ctx_ids.shape[1]
    n_option = full_ids.shape[1] - n_ctx
    if n_option <= 0:
        return float("-inf"), 0

    logits = model(full_ids).logits
    # Position i predicts token i+1, so the option's tokens are scored by the
    # logits at positions n_ctx-1 .. end-1.
    log_probs = torch.log_softmax(logits[0, n_ctx - 1 : -1, :].float(), dim=-1)
    targets = full_ids[0, n_ctx:]
    picked = log_probs[torch.arange(targets.shape[0]), targets]
    return float(picked.sum().item()), int(n_option)


def run_multiple_choice(model, tokenizer, items: list[dict], device: str) -> dict:
    """Score each item by the highest length-normalised option log-likelihood.

    Normalising by token count matters: without it the model would
    systematically prefer the shortest option, and the probe would measure
    answer length rather than knowledge.
    """
    correct = 0
    per_domain: dict[str, list[int]] = {}
    details: list[dict] = []

    for item in tqdm(items, desc="probe[mcq]"):
        context = f"Question: {item['question']}\nAnswer:"
        scores = []
        for choice in item["choices"]:
            total, n_tokens = _option_logprob(model, tokenizer, context, " " + choice, device)
            scores.append(total / max(1, n_tokens))
        predicted = int(max(range(len(scores)), key=lambda i: scores[i]))
        hit = int(predicted == item["answer"])
        correct += hit
        per_domain.setdefault(item["domain"], []).append(hit)
        details.append(
            {
                "id": item["id"],
                "domain": item["domain"],
                "predicted": predicted,
                "answer": item["answer"],
                "correct": bool(hit),
                "normalised_logprobs": [round(s, 4) for s in scores],
            }
        )

    return {
        "n": len(items),
        "accuracy": round(correct / len(items), 4) if items else 0.0,
        "random_baseline": round(
            sum(1 / len(i["choices"]) for i in items) / len(items), 4
        )
        if items
        else 0.0,
        "by_domain": {
            d: {"n": len(v), "accuracy": round(sum(v) / len(v), 4)}
            for d, v in sorted(per_domain.items())
        },
        "details": details,
    }


# ---------------------------------------------------------------------------
# Instruction following
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"```")


def _check(kind: str, arg, text: str) -> bool:
    stripped = text.strip()
    if kind == "one_word":
        return len(stripped.split()) == 1
    if kind == "contains":
        return str(arg).lower() in stripped.lower()
    if kind == "yes_no":
        return stripped.upper().rstrip(".!").strip() in ("YES", "NO")
    if kind == "n_lines":
        return len([l for l in stripped.split("\n") if l.strip()]) == int(arg)
    if kind == "max_words":
        return 0 < len(stripped.split()) <= int(arg)
    if kind == "no_letter":
        return str(arg).lower() not in stripped.lower()
    if kind == "json_key":
        try:
            payload = json.loads(_strip_fence(stripped))
        except (json.JSONDecodeError, ValueError):
            return False
        return isinstance(payload, dict) and str(arg) in payload
    if kind == "json_array":
        try:
            payload = json.loads(_strip_fence(stripped))
        except (json.JSONDecodeError, ValueError):
            return False
        return isinstance(payload, list) and len(payload) == int(arg)
    return False


def _strip_fence(text: str) -> str:
    match = re.search(r"```[a-zA-Z]*\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text


@torch.no_grad()
def run_instruction_following(model, tokenizer, items: list[dict], device: str) -> dict:
    passed = 0
    fenced = 0
    details: list[dict] = []

    for item in tqdm(items, desc="probe[instr]"):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": item["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        output = model.generate(
            **encoded,
            max_new_tokens=96,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
        )
        text = tokenizer.decode(
            output[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True
        )
        ok = _check(item["check"], item.get("arg"), text)
        has_fence = bool(_FENCE.search(text))
        passed += ok
        fenced += has_fence
        details.append(
            {
                "id": item["id"],
                "prompt": item["prompt"],
                "response": text.strip()[:300],
                "passed": ok,
                "emitted_code_fence": has_fence,
            }
        )

    n = len(items) or 1
    return {
        "n": len(items),
        "pass_rate": round(passed / n, 4),
        # The tell-tale sign of task collapse: the model answering "what colour
        # is the sky" inside a code block, because that is all it saw in training.
        "spurious_code_fence_rate": round(fenced / n, 4),
        "details": details,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_probe(
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.3",
    adapter: Path | None = None,
    device: str = "auto",
    load_in_4bit: bool = False,
    limit: int | None = None,
    out_dir: Path = Path("artifacts/forgetting"),
    tag: str = "",
) -> dict:
    probes = load_probes()
    mcq = probes["multiple_choice"][:limit] if limit else probes["multiple_choice"]
    instr = probes["instruction_following"][:limit] if limit else probes["instruction_following"]

    loaded = load_model(
        base_model=base_model,
        device=device,
        quant=QuantSpec(load_in_4bit=load_in_4bit),
        adapter=adapter,
    )
    tag = tag or (Path(adapter).name if adapter else "base")

    result = {
        "tag": tag,
        "base_model": base_model,
        "adapter": str(adapter) if adapter else None,
        "multiple_choice": run_multiple_choice(loaded.model, loaded.tokenizer, mcq, loaded.device),
        "instruction_following": run_instruction_following(
            loaded.model, loaded.tokenizer, instr, loaded.device
        ),
    }
    result["headline"] = {
        "mcq_accuracy": result["multiple_choice"]["accuracy"],
        "instruction_pass_rate": result["instruction_following"]["pass_rate"],
        "spurious_code_fence_rate": result["instruction_following"]["spurious_code_fence_rate"],
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{tag}.forgetting.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def compare_forgetting(base_result: dict, tuned_result: dict) -> dict:
    """Turn two probe runs into a retention statement the report can quote."""
    b, t = base_result["headline"], tuned_result["headline"]
    mcq_ret = (t["mcq_accuracy"] / b["mcq_accuracy"] * 100) if b["mcq_accuracy"] else 100.0
    instr_ret = (
        t["instruction_pass_rate"] / b["instruction_pass_rate"] * 100
    ) if b["instruction_pass_rate"] else 100.0

    return {
        "mcq_accuracy_base": b["mcq_accuracy"],
        "mcq_accuracy_tuned": t["mcq_accuracy"],
        "mcq_retention_pct": round(mcq_ret, 1),
        "instruction_pass_base": b["instruction_pass_rate"],
        "instruction_pass_tuned": t["instruction_pass_rate"],
        "instruction_retention_pct": round(instr_ret, 1),
        "spurious_fence_base": b["spurious_code_fence_rate"],
        "spurious_fence_tuned": t["spurious_code_fence_rate"],
        "overall_retention_pct": round(min(mcq_ret, instr_ret), 1),
        "verdict": _forgetting_verdict(mcq_ret, instr_ret, b, t),
    }


def _forgetting_verdict(mcq_ret: float, instr_ret: float, b: dict, t: dict) -> str:
    fence_jump = t["spurious_code_fence_rate"] - b["spurious_code_fence_rate"]
    parts = []
    worst = min(mcq_ret, instr_ret)
    if worst >= 97:
        parts.append(
            f"General capability is essentially intact ({worst:.0f}% retained on the weaker of the two probes)."
        )
    elif worst >= 90:
        parts.append(f"Mild degradation: {worst:.0f}% of general capability retained.")
    else:
        parts.append(
            f"Material degradation: only {worst:.0f}% of general capability retained. "
            f"Consider fewer epochs or a lower LoRA rank."
        )
    if fence_jump > 0.1:
        parts.append(
            f"The tuned model wraps general answers in code fences "
            f"{fence_jump * 100:.0f} points more often than the base model, which is "
            f"task-format bleed from the fine-tuning objective."
        )
    return " ".join(parts)
