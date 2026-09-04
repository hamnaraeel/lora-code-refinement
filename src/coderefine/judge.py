"""LLM-as-judge: blind pairwise comparison of base vs fine-tuned outputs.

Automated metrics on this task are sharp but narrow. Exact match cannot tell
"semantically correct, formatted differently" from "wrong", and edit similarity
rewards a prediction for leaving the code alone. A strong model reading the
reviewer comment, the original code, the gold revision and both candidates
catches what those metrics miss.

Three controls, because an unblinded judge is worth very little:

1. **Blinding.** Candidates are presented as "Response A"/"Response B" and the
   assignment is randomised per example. The judge is never told which model
   produced which, or that fine-tuning is involved at all.
2. **Position-bias control.** Every pair is judged twice with the order
   swapped, and the two verdicts are combined. Pairs where the judge picks the
   same *slot* both times are counted as position bias and reported, not hidden.
3. **A reference answer.** The gold revision is supplied, so the judge grades
   against what the author actually did rather than its own taste in code.
"""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any, Callable

#: Default judges. `claude-opus-5` is the strongest available Anthropic model
#: and the judge should be at least as capable as the models being judged.
DEFAULT_MODELS = {
    "anthropic": "claude-opus-5",
    "openai": "gpt-4o",
}

JUDGE_SYSTEM = (
    "You are a meticulous staff engineer grading how well two candidate edits "
    "apply a code review comment.\n\n"
    "You will be shown the original code, the reviewer's comment, the revision the "
    "author actually merged (the reference), and two candidate revisions labelled A and B.\n\n"
    "Grade each candidate on this rubric, from 1 to 5:\n"
    "  5 - Fully addresses the comment and is equivalent to the reference; no unrelated changes.\n"
    "  4 - Addresses the comment correctly; trivial cosmetic divergence from the reference.\n"
    "  3 - Partially addresses the comment, or addresses it but also makes an unrequested change.\n"
    "  2 - Attempts the change but gets it wrong, or edits the wrong part of the snippet.\n"
    "  1 - Ignores the comment, returns the input unchanged, or produces broken/irrelevant code.\n\n"
    "Judge only how well the comment was applied. Do not reward a candidate for being "
    "longer, for adding error handling nobody asked for, or for reformatting untouched lines — "
    "an unrequested change is a defect, not a bonus.\n\n"
    "Respond with a single JSON object and nothing else:\n"
    '{"score_a": <1-5>, "score_b": <1-5>, "winner": "A" | "B" | "tie", '
    '"reason": "<one or two sentences>"}'
)


def _build_user_prompt(rec: dict, first: str, second: str) -> str:
    lang = rec.get("lang", "")
    return (
        f"Language: {lang}\n\n"
        f"Original code:\n```{lang}\n{rec['old_code']}\n```\n\n"
        f"Reviewer comment:\n{rec['comment']}\n\n"
        f"Reference revision (what the author merged):\n```{lang}\n{rec['gold']}\n```\n\n"
        f"Candidate A:\n```{lang}\n{first}\n```\n\n"
        f"Candidate B:\n```{lang}\n{second}\n```\n\n"
        f"Grade both candidates."
    )


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


def _anthropic_caller(model: str) -> Callable[[str], str]:
    import anthropic

    client = anthropic.Anthropic()

    def call(prompt: str) -> str:
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in response.content if b.type == "text")

    return call


def _openai_caller(model: str) -> Callable[[str], str]:
    from openai import OpenAI

    client = OpenAI()

    def call(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    return call


def _build_caller(provider: str, model: str) -> Callable[[str], str]:
    model = model or DEFAULT_MODELS[provider]
    if provider == "anthropic":
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            print("[judge] No ANTHROPIC_API_KEY set; relying on an `ant auth login` profile.")
        return _anthropic_caller(model)
    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set.")
        return _openai_caller(model)
    raise ValueError(f"Unknown judge provider: {provider!r}")


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_verdict(text: str) -> dict | None:
    match = _JSON_RE.search(text or "")
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not {"score_a", "score_b", "winner"} <= set(payload):
        return None
    return payload


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _read(path: Path) -> dict[str, dict]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return {row["id"]: row for row in (json.loads(l) for l in fh if l.strip())}


def run_judge(
    base_path: Path,
    tuned_path: Path,
    provider: str = "anthropic",
    model: str = "",
    limit: int | None = None,
    dual_order: bool = True,
    seed: int = 11,
    out_path: Path = Path("artifacts/eval/judge.json"),
) -> dict:
    base, tuned = _read(base_path), _read(tuned_path)
    ids = [i for i in base if i in tuned]
    if limit:
        ids = ids[:limit]
    if not ids:
        raise ValueError("No shared example ids between the two prediction files.")

    call = _build_caller(provider, model)
    rng = random.Random(seed)
    results: list[dict] = []
    failures = 0

    for example_id in ids:
        rec = base[example_id]
        base_pred = base[example_id]["prediction"]
        tuned_pred = tuned[example_id]["prediction"]

        # Pass 1: randomised assignment. Pass 2: the same pair, swapped.
        base_is_a = rng.random() < 0.5
        orders = [base_is_a, not base_is_a] if dual_order else [base_is_a]

        votes: list[dict] = []
        for is_a in orders:
            first, second = (base_pred, tuned_pred) if is_a else (tuned_pred, base_pred)
            raw = call(_build_user_prompt(rec, first, second))
            verdict = _parse_verdict(raw)
            if verdict is None:
                failures += 1
                continue
            # Translate slot-relative scores back to model-relative ones.
            votes.append(
                {
                    "base_score": verdict["score_a"] if is_a else verdict["score_b"],
                    "tuned_score": verdict["score_b"] if is_a else verdict["score_a"],
                    "winner": _slot_to_model(verdict["winner"], is_a),
                    "raw_winner_slot": verdict["winner"],
                    "base_was_slot_a": is_a,
                    "reason": verdict.get("reason", ""),
                }
            )

        if not votes:
            continue

        base_score = sum(v["base_score"] for v in votes) / len(votes)
        tuned_score = sum(v["tuned_score"] for v in votes) / len(votes)
        winners = [v["winner"] for v in votes]
        if len(set(winners)) == 1:
            winner, consistent = winners[0], True
        else:
            # Disagreement across orders means the judge followed position, not
            # content. Score it a tie rather than letting a coin flip decide.
            winner, consistent = "tie", False

        results.append(
            {
                "id": example_id,
                "lang": rec.get("lang", ""),
                "category": rec.get("category", ""),
                "origin": rec.get("origin", ""),
                "base_score": round(base_score, 3),
                "tuned_score": round(tuned_score, 3),
                "winner": winner,
                "order_consistent": consistent,
                "votes": votes,
            }
        )

    summary = _summarise(results, provider, model or DEFAULT_MODELS[provider], dual_order, failures)
    report = {"summary": summary, "results": results}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _slot_to_model(winner_slot: str, base_is_a: bool) -> str:
    if winner_slot not in ("A", "B"):
        return "tie"
    if winner_slot == "A":
        return "base" if base_is_a else "tuned"
    return "tuned" if base_is_a else "base"


def _summarise(
    results: list[dict], provider: str, model: str, dual_order: bool, failures: int
) -> dict:
    n = len(results)
    if n == 0:
        return {"n": 0, "note": "no examples judged"}

    wins = sum(1 for r in results if r["winner"] == "tuned")
    losses = sum(1 for r in results if r["winner"] == "base")
    ties = n - wins - losses
    inconsistent = sum(1 for r in results if not r["order_consistent"])

    by_category: dict[str, dict] = {}
    for r in results:
        entry = by_category.setdefault(
            r["category"], {"n": 0, "tuned_wins": 0, "base_wins": 0, "base_sum": 0.0, "tuned_sum": 0.0}
        )
        entry["n"] += 1
        entry["tuned_wins"] += r["winner"] == "tuned"
        entry["base_wins"] += r["winner"] == "base"
        entry["base_sum"] += r["base_score"]
        entry["tuned_sum"] += r["tuned_score"]
    for entry in by_category.values():
        entry["base_mean"] = round(entry.pop("base_sum") / entry["n"], 3)
        entry["tuned_mean"] = round(entry.pop("tuned_sum") / entry["n"], 3)

    return {
        "n": n,
        "judge_provider": provider,
        "judge_model": model,
        "dual_order": dual_order,
        "unparseable_responses": failures,
        "base_mean_score": round(sum(r["base_score"] for r in results) / n, 3),
        "tuned_mean_score": round(sum(r["tuned_score"] for r in results) / n, 3),
        "score_delta": round(
            (sum(r["tuned_score"] for r in results) - sum(r["base_score"] for r in results)) / n, 3
        ),
        "tuned_wins": wins,
        "base_wins": losses,
        "ties": ties,
        "tuned_win_rate": round(wins / n, 3),
        "position_bias_rate": round(inconsistent / n, 3),
        "position_bias_note": (
            f"{inconsistent}/{n} pairs flipped verdict when the candidate order was swapped; "
            "those are scored as ties. A high rate means the judge is reading position "
            "rather than content and the win rate should be treated as noise."
        ),
        "by_category": dict(sorted(by_category.items())),
    }
