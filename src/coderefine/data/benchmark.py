"""Assemble the hard evaluation benchmark.

The test split answers "how does the model do on average". The benchmark
answers a different and more useful question: "where does it break". It has two
halves, kept separable in every result table because they measure different things:

``handwritten`` (15 examples)
    Authored by hand, one per expected failure mode — over-editing a distractor
    identifier, an edit that is a deletion, an edit that is only a change of
    indentation, a comment phrased as a question, a near-miss typo next to an
    already-correct twin. These exist because the natural distribution contains
    almost none of them, and they are exactly the cases where a model that has
    learned "make a plausible-looking change" separates from one that has
    learned to read the comment.

``curated`` (25 by default)
    Drawn from the held-out test pool and stratified over feedback categories,
    biased toward the hard end: low comment-to-diff grounding, larger edits,
    and the categories where the required change is implied rather than stated.

Curated picks come from ``test.jsonl``, which no training or model-selection
step ever reads, so the benchmark inherits the test split's leak guarantees.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

#: Categories where the reviewer describes a problem rather than a fix. Weighted
#: up because they are where the interesting failures live.
_HARD_CATEGORIES = {"design_question", "logic_bug", "performance", "security", "other"}


def hardness(example: dict) -> float:
    """Heuristic difficulty, higher is harder.

    Three additive signals: weak lexical grounding between comment and diff
    (the model cannot copy its way to the answer), a larger edit (more chances
    to be wrong), and membership of a category whose comments imply rather than
    state the change.
    """
    grounding = example.get("grounding")
    grounding_term = 1.0 - float(grounding) if grounding is not None else 0.5
    size_term = min(1.0, (example.get("n_changed_lines") or 1) / 6.0)
    category_term = 1.0 if example.get("category") in _HARD_CATEGORIES else 0.0
    return round(0.5 * grounding_term + 0.2 * size_term + 0.3 * category_term, 4)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_benchmark(
    test_path: Path,
    handwritten_path: Path,
    out_path: Path,
    n_curated: int = 25,
    seed: int = 7,
) -> dict:
    """Write the merged benchmark and return its card."""
    pool = _read_jsonl(Path(test_path))
    handwritten = _read_jsonl(Path(handwritten_path))
    if not pool:
        raise FileNotFoundError(f"No test split at {test_path}; run build-data first.")

    rng = random.Random(seed)
    for ex in pool:
        ex["hardness"] = hardness(ex)

    # Stratify: walk the categories round-robin, taking the hardest unused
    # example from each. A flat "top-N hardest" would return 25 examples from
    # the two most-populous categories and tell us nothing about the rest.
    by_category: dict[str, list[dict]] = defaultdict(list)
    for ex in pool:
        by_category[ex["category"]].append(ex)
    for bucket in by_category.values():
        rng.shuffle(bucket)
        bucket.sort(key=lambda e: e["hardness"], reverse=True)

    categories = sorted(by_category)
    rng.shuffle(categories)
    curated: list[dict] = []
    cursor = 0
    while len(curated) < n_curated:
        progressed = False
        for category in categories:
            bucket = by_category[category]
            if cursor < len(bucket):
                curated.append(bucket[cursor])
                progressed = True
                if len(curated) >= n_curated:
                    break
        if not progressed:
            break
        cursor += 1

    for ex in curated:
        ex["origin"] = "curated"
        ex.setdefault("probe", "natural_distribution")
    for ex in handwritten:
        ex["origin"] = "handwritten"
        ex["hardness"] = None

    merged = handwritten + curated
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for ex in merged:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

    card = {
        "n_total": len(merged),
        "n_handwritten": len(handwritten),
        "n_curated": len(curated),
        "curated_source": str(test_path),
        "seed": seed,
        "languages": dict(Counter(e["lang"] for e in merged).most_common()),
        "categories": dict(Counter(e["category"] for e in merged).most_common()),
        "probes": sorted({e.get("probe", "") for e in handwritten}),
        "curated_mean_hardness": round(
            sum(e["hardness"] for e in curated) / len(curated), 4
        )
        if curated
        else None,
        "pool_mean_hardness": round(sum(e["hardness"] for e in pool) / len(pool), 4),
        "output": str(out_path),
    }
    (out_path.parent / "benchmark_card.json").write_text(
        json.dumps(card, indent=2), encoding="utf-8"
    )
    return card
