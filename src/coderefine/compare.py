"""Head-to-head comparison of two prediction sets.

Produces the numbers the report's headline claim rests on, and — equally
important — the ones that qualify it: which examples fine-tuning fixed, which
it broke, and whether the aggregate difference is large enough to be worth
believing at this sample size.

Statistical honesty
-------------------
The benchmark has 40 examples and the test split 250. At n=40, a 5-point
difference in exact match is two examples. So every headline delta is reported
with a bootstrap confidence interval and, for the paired binary metrics, an
exact McNemar test. A result whose interval straddles zero is labelled as such
rather than quoted as an improvement.
"""

from __future__ import annotations

import json
import random
from math import comb
from pathlib import Path
from typing import Sequence


def _read_predictions(path: Path) -> dict[str, dict]:
    with Path(path).open("r", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return {row["id"]: row for row in rows}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def bootstrap_ci(
    values_a: Sequence[float],
    values_b: Sequence[float],
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Percentile bootstrap CI for mean(b) - mean(a) over *paired* examples.

    Resampling example indices rather than the two sets independently keeps the
    pairing, which is the whole point: both models saw the same inputs, so the
    per-example difference is far less noisy than the two means separately.
    """
    n = len(values_a)
    if n == 0:
        return {"delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n_resamples": 0}
    diffs = [b - a for a, b in zip(values_a, values_b)]
    observed = sum(diffs) / n

    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_resamples):
        total = 0.0
        for _ in range(n):
            total += diffs[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    lo = means[int((alpha / 2) * n_resamples)]
    hi = means[min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))]
    return {
        "delta": round(observed, 4),
        "ci_low": round(lo, 4),
        "ci_high": round(hi, 4),
        "significant": bool(lo > 0 or hi < 0),
        "n_resamples": n_resamples,
    }


def mcnemar_exact(a_flags: Sequence[bool], b_flags: Sequence[bool]) -> dict:
    """Exact two-sided McNemar test on paired binary outcomes.

    Only the discordant pairs carry information: examples both models got right
    (or both wrong) say nothing about which is better. Under the null the count
    of "b fixed it" among discordant pairs is Binomial(n_discordant, 0.5).
    """
    b_only = sum(1 for a, b in zip(a_flags, b_flags) if b and not a)
    a_only = sum(1 for a, b in zip(a_flags, b_flags) if a and not b)
    n = b_only + a_only
    if n == 0:
        return {"fixed": 0, "broken": 0, "n_discordant": 0, "p_value": 1.0}

    k = min(b_only, a_only)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    p = min(1.0, 2 * tail)
    return {
        "fixed": b_only,
        "broken": a_only,
        "n_discordant": n,
        "p_value": round(p, 10),
        "significant_at_05": p < 0.05,
    }


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

#: Metrics compared example-by-example. Binary ones additionally get McNemar.
_PAIRED_METRICS = {
    "exact_match": True,
    "improved": True,
    "regressed": True,
    "copied": True,
    "edit_sim": False,
    "edit_line_f1": False,
    "token_f1": False,
    "delta_to_gold": False,
}


def compare_runs(base_path: Path, tuned_path: Path, out_path: Path | None = None) -> dict:
    base = _read_predictions(base_path)
    tuned = _read_predictions(tuned_path)

    shared = [i for i in base if i in tuned]
    missing_from_tuned = sorted(set(base) - set(tuned))
    missing_from_base = sorted(set(tuned) - set(base))
    if not shared:
        raise ValueError("The two prediction files share no example ids.")
    # Order by the base file so results are stable across reruns.
    order = [row["id"] for row in _iter_ordered(base_path) if row["id"] in tuned]

    paired: dict[str, dict] = {}
    for metric, is_binary in _PAIRED_METRICS.items():
        a = [float(base[i]["scores"][metric]) for i in order]
        b = [float(tuned[i]["scores"][metric]) for i in order]
        entry = {
            "base": round(sum(a) / len(a), 4),
            "tuned": round(sum(b) / len(b), 4),
            **bootstrap_ci(a, b),
        }
        if is_binary:
            entry["mcnemar"] = mcnemar_exact(
                [bool(base[i]["scores"][metric]) for i in order],
                [bool(tuned[i]["scores"][metric]) for i in order],
            )
        paired[metric] = entry

    # --- per-example outcomes -------------------------------------------
    wins, losses, ties = [], [], []
    for i in order:
        b_s, t_s = base[i]["scores"], tuned[i]["scores"]
        delta = t_s["edit_sim"] - b_s["edit_sim"]
        row = {
            "id": i,
            "lang": base[i]["lang"],
            "category": base[i]["category"],
            "origin": base[i].get("origin", ""),
            "probe": base[i].get("probe", ""),
            "comment": base[i]["comment"],
            "old_code": base[i]["old_code"],
            "gold": base[i]["gold"],
            "base_prediction": base[i]["prediction"],
            "tuned_prediction": tuned[i]["prediction"],
            "base_exact": b_s["exact_match"],
            "tuned_exact": t_s["exact_match"],
            "base_edit_sim": b_s["edit_sim"],
            "tuned_edit_sim": t_s["edit_sim"],
            "delta_edit_sim": round(delta, 4),
        }
        if t_s["exact_match"] and not b_s["exact_match"]:
            wins.append(row)
        elif b_s["exact_match"] and not t_s["exact_match"]:
            losses.append(row)
        elif abs(delta) > 0.02:
            (wins if delta > 0 else losses).append(row)
        else:
            ties.append(row)

    wins.sort(key=lambda r: -r["delta_edit_sim"])
    losses.sort(key=lambda r: r["delta_edit_sim"])

    # --- per-slice breakdown --------------------------------------------
    def slice_table(key: str) -> dict:
        buckets: dict[str, list[str]] = {}
        for i in order:
            buckets.setdefault(base[i].get(key) or "unknown", []).append(i)
        out = {}
        for name, ids in sorted(buckets.items()):
            b_em = sum(base[i]["scores"]["exact_match"] for i in ids) / len(ids)
            t_em = sum(tuned[i]["scores"]["exact_match"] for i in ids) / len(ids)
            b_es = sum(base[i]["scores"]["edit_sim"] for i in ids) / len(ids)
            t_es = sum(tuned[i]["scores"]["edit_sim"] for i in ids) / len(ids)
            out[name] = {
                "n": len(ids),
                "base_exact_match": round(b_em, 4),
                "tuned_exact_match": round(t_em, 4),
                "delta_exact_match": round(t_em - b_em, 4),
                "base_edit_sim": round(b_es, 4),
                "tuned_edit_sim": round(t_es, 4),
                "delta_edit_sim": round(t_es - b_es, 4),
            }
        return out

    em = paired["exact_match"]
    report = {
        "headline": {
            "n": len(order),
            "base_exact_match": em["base"],
            "tuned_exact_match": em["tuned"],
            "exact_match_delta": em["delta"],
            "exact_match_ci95": [em["ci_low"], em["ci_high"]],
            "exact_match_significant": em["significant"],
            "mcnemar_p": em["mcnemar"]["p_value"],
            "examples_fixed": em["mcnemar"]["fixed"],
            "examples_broken": em["mcnemar"]["broken"],
            "base_improved_rate": paired["improved"]["base"],
            "tuned_improved_rate": paired["improved"]["tuned"],
            "base_copy_rate": paired["copied"]["base"],
            "tuned_copy_rate": paired["copied"]["tuned"],
            "verdict": _verdict(em),
        },
        "paired_metrics": paired,
        "by_category": slice_table("category"),
        "by_lang": slice_table("lang"),
        "by_origin": slice_table("origin"),
        "wins": wins[:25],
        "losses": losses[:25],
        "n_wins": len(wins),
        "n_losses": len(losses),
        "n_ties": len(ties),
        "coverage": {
            "compared": len(order),
            "in_base_only": missing_from_tuned,
            "in_tuned_only": missing_from_base,
        },
        "sources": {"base": str(base_path), "tuned": str(tuned_path)},
    }

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def _verdict(entry: dict) -> str:
    """A one-line, quotable summary that will not overclaim."""
    delta = entry["delta"] * 100
    lo, hi = entry["ci_low"] * 100, entry["ci_high"] * 100
    p = entry["mcnemar"]["p_value"]
    direction = "improved" if delta > 0 else "reduced" if delta < 0 else "left unchanged"
    if entry["significant"] and p < 0.05:
        return (
            f"Fine-tuning {direction} exact match by {abs(delta):.1f} points "
            f"(95% CI [{lo:+.1f}, {hi:+.1f}], McNemar p={p:.4f}) — significant at this sample size."
        )
    return (
        f"Fine-tuning {direction} exact match by {abs(delta):.1f} points, but the 95% CI "
        f"[{lo:+.1f}, {hi:+.1f}] includes zero (McNemar p={p:.4f}); this sample is too small "
        f"to call the difference real."
    )


def _iter_ordered(path: Path):
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)
