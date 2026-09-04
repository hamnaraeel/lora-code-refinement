"""Task-specific scoring for code refinement.

A warning that shapes this whole module: **on this task the input is already
~90% of the correct output.** A model that echoes the code it was given, changing
nothing, scores above 80 BLEU. Reporting BLEU alone would make a model that does
nothing look excellent. So every headline metric here is either

* insensitive to the copy baseline (exact match), or
* reported *relative* to it (``copy_baseline_*`` fields, ``delta_to_gold``).

The metric that actually captures the task is ``improved_rate``: the fraction of
examples where the prediction is closer to the gold revision than the original
code was. It is 0 by construction for a model that copies.
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

import Levenshtein
import sacrebleu

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def normalize(code: str) -> str:
    """Canonical form for comparison.

    Trailing whitespace and line-ending style are not part of the task — no
    reviewer asked for them and the dataset is inconsistent about them — so
    they are normalised away. *Leading* whitespace is preserved: indentation is
    semantic in Python and is frequently the exact thing a reviewer asked to
    change elsewhere.
    """
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in code.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


_CODE_TOKEN = re.compile(r"[A-Za-z_]\w*|\d+\.\d+|\d+|[^\sA-Za-z0-9_]")


def tokenize(code: str) -> list[str]:
    """Split code into identifier / literal / punctuation tokens.

    Whitespace-splitting would treat ``foo(bar)`` as one token and make BLEU
    nearly blind to argument changes, which are a large share of review fixes.
    """
    return _CODE_TOKEN.findall(code)


def tokenized(code: str) -> str:
    return " ".join(tokenize(code))


# ---------------------------------------------------------------------------
# Per-example scoring
# ---------------------------------------------------------------------------


@dataclass
class ExampleScore:
    exact_match: bool
    token_f1: float
    edit_sim: float
    #: Normalised Levenshtein distance from prediction to gold, in [0, 1].
    dist_pred_gold: float
    #: Same, from the *original* code to gold. The bar the model must beat.
    dist_old_gold: float
    #: dist_old_gold - dist_pred_gold. Positive means the model moved the code
    #: toward the reviewed version; negative means it moved it further away.
    delta_to_gold: float
    improved: bool
    regressed: bool
    #: The model returned the input unchanged.
    copied: bool
    syntax_ok: bool | None
    changed_right_lines: float


def _norm_dist(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    return Levenshtein.distance(a, b) / max(len(a), len(b), 1)


def _token_f1(pred: str, gold: str) -> float:
    from collections import Counter

    p, g = Counter(tokenize(pred)), Counter(tokenize(gold))
    overlap = sum((p & g).values())
    if overlap == 0:
        return 0.0
    precision = overlap / max(1, sum(p.values()))
    recall = overlap / max(1, sum(g.values()))
    return 2 * precision * recall / (precision + recall)


def _syntax_ok(code: str, lang: str) -> bool | None:
    """Parseability check. Only meaningful where we have a parser in-process.

    Returns ``None`` for languages we cannot check, and — importantly — also for
    Python snippets that were already unparseable *before* the model touched
    them. The hunks are excerpts from larger files, so many are legitimately
    incomplete; penalising the model for that would be measuring the dataset,
    not the model.
    """
    if lang != "py":
        return None
    try:
        ast.parse(code)
        return True
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return False


def _changed_right_lines(old: str, pred: str, gold: str) -> float:
    """Line-level F1 over the *edit*, not over the whole snippet.

    Whole-snippet similarity is dominated by untouched context lines. This
    scores only the lines that gold actually adds or removes against the lines
    the prediction adds or removes, which is where the task lives.
    """
    old_lines = set(normalize(old).split("\n"))
    gold_set = set(normalize(gold).split("\n"))
    pred_set = set(normalize(pred).split("\n"))

    gold_edit = (gold_set - old_lines) | (old_lines - gold_set)
    pred_edit = (pred_set - old_lines) | (old_lines - pred_set)
    if not gold_edit:
        return 1.0 if not pred_edit else 0.0
    if not pred_edit:
        return 0.0
    hit = len(gold_edit & pred_edit)
    if hit == 0:
        return 0.0
    precision = hit / len(pred_edit)
    recall = hit / len(gold_edit)
    return 2 * precision * recall / (precision + recall)


def score_example(pred: str, gold: str, old: str, lang: str = "") -> ExampleScore:
    p, g, o = normalize(pred), normalize(gold), normalize(old)
    dist_pred_gold = _norm_dist(p, g)
    dist_old_gold = _norm_dist(o, g)

    # Only judge syntax when the original was itself parseable, so we measure
    # damage the model caused rather than pre-existing truncation.
    syntax = None
    if lang == "py" and _syntax_ok(o, lang):
        syntax = _syntax_ok(p, lang)

    return ExampleScore(
        exact_match=p == g,
        token_f1=_token_f1(p, g),
        edit_sim=1.0 - dist_pred_gold,
        dist_pred_gold=dist_pred_gold,
        dist_old_gold=dist_old_gold,
        delta_to_gold=dist_old_gold - dist_pred_gold,
        improved=dist_pred_gold < dist_old_gold - 1e-9,
        regressed=dist_pred_gold > dist_old_gold + 1e-9,
        copied=p == o,
        syntax_ok=syntax,
        changed_right_lines=_changed_right_lines(o, p, g),
    )


# ---------------------------------------------------------------------------
# Corpus scoring
# ---------------------------------------------------------------------------


def corpus_bleu(preds: Sequence[str], golds: Sequence[str]) -> float:
    """BLEU-4 over code-tokenized text. Read it next to ``copy_baseline_bleu``."""
    if not preds:
        return 0.0
    hyps = [tokenized(normalize(p)) for p in preds]
    refs = [[tokenized(normalize(g)) for g in golds]]
    return float(sacrebleu.corpus_bleu(hyps, refs, tokenize="none", force=True).score)


def percentile(sorted_values: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile, guaranteeing p50 <= p95 at any sample size.

    The obvious ``values[int(q * (n - 1))]`` truncates, and for small n that
    sends a high quantile *below* a low one — with two samples it returns the
    minimum for p95. Latency tables that claim a p95 faster than the median are
    the visible symptom; the ceiling-based nearest rank below is monotonic in q.
    """
    n = len(sorted_values)
    if n == 0:
        return None
    index = min(n - 1, max(0, math.ceil(q * n) - 1))
    return sorted_values[index]


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return round(sum(values) / len(values), 4) if values else 0.0


def aggregate(records: Sequence[dict]) -> dict:
    """Aggregate scored records into the metric block used everywhere downstream.

    Each record needs ``prediction``, ``gold``, ``old_code`` and optionally
    ``lang``, ``category`` and ``parse_mode``.
    """
    if not records:
        return {"n": 0}

    scores = [
        score_example(r["prediction"], r["gold"], r["old_code"], r.get("lang", ""))
        for r in records
    ]
    preds = [r["prediction"] for r in records]
    golds = [r["gold"] for r in records]
    olds = [r["old_code"] for r in records]

    syntax_checked = [s.syntax_ok for s in scores if s.syntax_ok is not None]
    fmt = [r.get("parse_mode", "") for r in records]

    metrics = {
        "n": len(records),
        # --- headline -------------------------------------------------------
        "exact_match": _mean(float(s.exact_match) for s in scores),
        "improved_rate": _mean(float(s.improved) for s in scores),
        "regressed_rate": _mean(float(s.regressed) for s in scores),
        "copy_rate": _mean(float(s.copied) for s in scores),
        # --- similarity -----------------------------------------------------
        "bleu": round(corpus_bleu(preds, golds), 3),
        "copy_baseline_bleu": round(corpus_bleu(olds, golds), 3),
        "edit_similarity": _mean(s.edit_sim for s in scores),
        "copy_baseline_edit_similarity": _mean(1.0 - s.dist_old_gold for s in scores),
        "token_f1": _mean(s.token_f1 for s in scores),
        "edit_line_f1": _mean(s.changed_right_lines for s in scores),
        "delta_to_gold": _mean(s.delta_to_gold for s in scores),
        # --- hygiene --------------------------------------------------------
        "format_compliance": _mean(float(m in ("fenced", "fenced_unterminated")) for m in fmt) if any(fmt) else None,
        "empty_rate": _mean(float(not r["prediction"].strip()) for r in records),
        "python_syntax_ok": _mean(float(bool(x)) for x in syntax_checked) if syntax_checked else None,
        "python_syntax_checked": len(syntax_checked),
    }

    # Per-slice breakdowns drive the "where did it help / hurt" section of the
    # report, so they carry the same headline metrics at smaller n.
    for key in ("category", "lang"):
        buckets: dict[str, list[int]] = {}
        for i, rec in enumerate(records):
            buckets.setdefault(rec.get(key, "unknown"), []).append(i)
        metrics[f"by_{key}"] = {
            name: {
                "n": len(idx),
                "exact_match": _mean(float(scores[i].exact_match) for i in idx),
                "improved_rate": _mean(float(scores[i].improved) for i in idx),
                "edit_line_f1": _mean(scores[i].changed_right_lines for i in idx),
            }
            for name, idx in sorted(buckets.items())
        }
    return metrics


def attach_scores(records: list[dict]) -> list[dict]:
    """Annotate each record in place with its per-example scores."""
    for rec in records:
        s = score_example(rec["prediction"], rec["gold"], rec["old_code"], rec.get("lang", ""))
        rec["scores"] = {
            "exact_match": s.exact_match,
            "improved": s.improved,
            "regressed": s.regressed,
            "copied": s.copied,
            "edit_sim": round(s.edit_sim, 4),
            "delta_to_gold": round(s.delta_to_gold, 4),
            "edit_line_f1": round(s.changed_right_lines, 4),
            "token_f1": round(s.token_f1, 4),
            "syntax_ok": s.syntax_ok,
        }
    return records
