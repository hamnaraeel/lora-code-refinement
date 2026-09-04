"""Turn the raw CodeReviewer refinement dumps into a curated instruction dataset.

Input  : Code_Refinement/ref-{train,valid,test}.jsonl  (~6 GB, ~150k records)
Output : data/processed/{train,valid,test}.jsonl       (a few thousand records)

The guiding principle is quality over quantity. A LoRA adapter trained on 2,000
examples where the reviewer comment genuinely determines the code change will
beat one trained on 50,000 examples where half the comments are "ditto" or
reference a conversation that happened somewhere else. Most of this module is
therefore filtering, and every filter records why it fired so the report can
show the funnel.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .taxonomy import classify

# ---------------------------------------------------------------------------
# Diff-marker handling
# ---------------------------------------------------------------------------
# The raw `old` / `new` fields are the two sides of a unified diff. Every line
# carries a marker in column 0:
#   ' '  context, present in both versions
#   '-'  present only in `old` (the reviewer wanted it gone)
#   '+'  present only in `new` (the author added it)
#   '\'  the literal "\ No newline at end of file" pseudo-line
# Reconstructing plain source means keeping the markers that apply to the side
# being built and dropping the pseudo-lines entirely.


def strip_markers(text: str, side: str) -> str:
    """Reconstruct plain source from one side of a unified diff.

    ``side`` is "old" (keep ' ' and '-') or "new" (keep ' ' and '+').
    """
    keep = {" ", "-"} if side == "old" else {" ", "+"}
    out: list[str] = []
    for line in text.split("\n"):
        if not line:
            out.append("")
            continue
        marker, rest = line[0], line[1:]
        if marker == "\\":  # "\ No newline at end of file"
            continue
        if marker in keep:
            out.append(rest)
        elif marker in {"+", "-"}:
            continue
        else:
            # Marker-less line: some records have a ragged final line. Keep it
            # verbatim rather than silently deleting a line of real code.
            out.append(line)
    return "\n".join(out)


def changed_line_count(old_raw: str, new_raw: str) -> int:
    """How many lines the refinement actually touches."""
    minus = sum(1 for l in old_raw.split("\n") if l.startswith("-"))
    plus = sum(1 for l in new_raw.split("\n") if l.startswith("+"))
    return minus + plus


# ---------------------------------------------------------------------------
# Comment quality heuristics
# ---------------------------------------------------------------------------

#: Comments that point outside the snippet. The change they ask for is not
#: recoverable from the inputs we give the model, so training on them teaches
#: the model to hallucinate. This is the single highest-value filter.
_UNGROUNDED = re.compile(
    r"\bas discussed\b|\bas (we|you) (said|mentioned|agreed)\b|\bsee (above|below|the )"
    r"|\bditto\b|\bsame (as|here|comment)\b|\bper (our|the) (chat|discussion|convo)"
    r"|\boffline\b|\bsee #\d+|\bthis pr\b|\bprevious (comment|review)\b|\brefer to\b"
    r"|\bmentioned (above|earlier|before)\b",
    re.IGNORECASE,
)

#: Pure acknowledgements with no requested change.
_NON_ACTIONABLE = re.compile(
    r"^\s*(lgtm|nice|thanks?|thx|ok(ay)?|done|\+1|good catch|agreed|sgtm|ack|👍|:\+1:)"
    r"[\s.!]*$",
    re.IGNORECASE,
)

_URL = re.compile(r"https?://\S+")
_CODE_SPAN = re.compile(r"`([^`]+)`")
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def _ascii_ratio(s: str) -> float:
    if not s:
        return 0.0
    return sum(1 for c in s if ord(c) < 128) / len(s)


def grounding_score(comment: str, old_code: str, new_code: str) -> float:
    """How strongly the comment predicts the change, in [0, 1].

    Built from three signals, because no single one is reliable:

    * ``span``    — identifiers the reviewer wrote in backticks that appear in
                    the new code but not the old. A reviewer who writes
                    "use `errors.Is` here" has literally handed over the answer.
    * ``added``   — overlap between the comment's word-like tokens and the
                    tokens on added lines.
    * ``removed`` — overlap with tokens on removed lines (the reviewer naming
                    the thing they object to).

    Examples scoring near zero are ones where the comment and the diff share no
    vocabulary at all, which in practice means the change came from context we
    are not showing the model.
    """
    old_tokens = set(_TOKEN.findall(old_code))
    new_tokens = set(_TOKEN.findall(new_code))
    added = new_tokens - old_tokens
    removed = old_tokens - new_tokens

    comment_tokens = set(_TOKEN.findall(comment))
    spans: set[str] = set()
    for span in _CODE_SPAN.findall(comment):
        spans.update(_TOKEN.findall(span))

    span_hit = 1.0 if spans & added else (0.5 if spans & (added | removed) else 0.0)
    added_hit = len(comment_tokens & added) / max(1, min(len(added), 8))
    removed_hit = len(comment_tokens & removed) / max(1, min(len(removed), 8))

    score = 0.5 * span_hit + 0.35 * min(1.0, added_hit) + 0.15 * min(1.0, removed_hit)
    return round(min(1.0, score), 4)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


@dataclass
class CurationConfig:
    """Every knob the funnel exposes. Serialised into the dataset card."""

    min_old_lines: int = 2
    max_old_lines: int = 18
    max_changed_lines: int = 8
    min_comment_chars: int = 20
    max_comment_chars: int = 400
    min_ascii_ratio: float = 0.95
    min_grounding: float = 0.10
    #: At most this many examples from any one GitHub repository, so that a
    #: handful of enormous monorepos cannot define the task.
    max_per_repo: int = 40
    #: At most this many examples sharing a normalised comment, so that
    #: "remove this" does not become 5% of the training set.
    max_per_comment: int = 3
    languages: tuple[str, ...] = ("py", "java", "go", "js", "php", "rb", "c", "cpp", ".cs")


@dataclass
class Example:
    """One curated, model-ready training example."""

    id: str
    repo: str
    lang: str
    old_code: str
    new_code: str
    comment: str
    category: str
    grounding: float
    n_old_lines: int
    n_changed_lines: int
    source_split: str

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)


@dataclass
class FunnelStats:
    """Counts of how many records each filter rejected, in order."""

    read: int = 0
    kept: int = 0
    rejected: Counter = field(default_factory=Counter)
    langs: Counter = field(default_factory=Counter)
    categories: Counter = field(default_factory=Counter)

    def as_dict(self) -> dict:
        return {
            "records_read": self.read,
            "records_kept": self.kept,
            "rejected_by_filter": dict(self.rejected.most_common()),
            "language_distribution": dict(self.langs.most_common()),
            "category_distribution": dict(self.categories.most_common()),
        }


def _norm_comment(comment: str) -> str:
    return re.sub(r"\s+", " ", _URL.sub("", comment)).strip().lower()


def _content_hash(old_code: str, comment: str) -> str:
    payload = re.sub(r"\s+", " ", old_code) + "||" + _norm_comment(comment)
    return hashlib.sha1(payload.encode("utf-8", "ignore")).hexdigest()


def iter_raw(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def curate_file(
    path: Path,
    cfg: CurationConfig,
    source_split: str,
    limit: int | None = None,
    seen_hashes: set[str] | None = None,
) -> tuple[list[Example], FunnelStats]:
    """Stream one raw ``.jsonl`` and return the examples that survive the funnel.

    ``seen_hashes`` is shared across files so that an example appearing in both
    the official train and test dumps is dropped from the second one rather
    than becoming a leak.
    """
    stats = FunnelStats()
    seen = seen_hashes if seen_hashes is not None else set()
    per_repo: Counter = Counter()
    per_comment: Counter = Counter()
    kept: list[Example] = []
    langs = set(cfg.languages)

    for rec in iter_raw(path):
        stats.read += 1
        if limit is not None and stats.read > limit:
            stats.read -= 1
            break

        lang = rec.get("lang", "")
        if lang not in langs:
            stats.rejected["language_not_selected"] += 1
            continue

        comment = (rec.get("comment") or "").strip()
        if not comment:
            stats.rejected["empty_comment"] += 1
            continue
        if _NON_ACTIONABLE.match(comment):
            stats.rejected["comment_non_actionable"] += 1
            continue
        if _UNGROUNDED.search(comment):
            stats.rejected["comment_references_outside_context"] += 1
            continue
        if not (cfg.min_comment_chars <= len(comment) <= cfg.max_comment_chars):
            stats.rejected["comment_length"] += 1
            continue
        if _ascii_ratio(comment) < cfg.min_ascii_ratio:
            stats.rejected["comment_not_ascii"] += 1
            continue

        old_raw, new_raw = rec.get("old") or "", rec.get("new") or ""
        old_code = strip_markers(old_raw, "old")
        new_code = strip_markers(new_raw, "new")

        if old_code == new_code:
            stats.rejected["no_actual_change"] += 1
            continue
        if not old_code.strip() or not new_code.strip():
            stats.rejected["empty_side"] += 1
            continue

        n_old = len(old_code.split("\n"))
        if not (cfg.min_old_lines <= n_old <= cfg.max_old_lines):
            stats.rejected["hunk_length"] += 1
            continue

        n_changed = changed_line_count(old_raw, new_raw)
        if n_changed == 0 or n_changed > cfg.max_changed_lines:
            stats.rejected["change_too_large"] += 1
            continue

        grounding = grounding_score(comment, old_code, new_code)
        if grounding < cfg.min_grounding:
            stats.rejected["ungrounded_change"] += 1
            continue

        h = _content_hash(old_code, comment)
        if h in seen:
            stats.rejected["duplicate"] += 1
            continue

        repo = rec.get("repo", "unknown")
        if per_repo[repo] >= cfg.max_per_repo:
            stats.rejected["repo_quota"] += 1
            continue

        cnorm = _norm_comment(comment)
        if per_comment[cnorm] >= cfg.max_per_comment:
            stats.rejected["comment_quota"] += 1
            continue

        seen.add(h)
        per_repo[repo] += 1
        per_comment[cnorm] += 1
        category = classify(comment)

        kept.append(
            Example(
                id=h[:16],
                repo=repo,
                lang=lang,
                old_code=old_code,
                new_code=new_code,
                comment=comment,
                category=category,
                grounding=grounding,
                n_old_lines=n_old,
                n_changed_lines=n_changed,
                source_split=source_split,
            )
        )
        stats.kept += 1
        stats.langs[lang] += 1
        stats.categories[category] += 1

    return kept, stats


# ---------------------------------------------------------------------------
# Balanced down-sampling
# ---------------------------------------------------------------------------


def balanced_sample(
    examples: list[Example], target: int, seed: int = 13
) -> list[Example]:
    """Take ``target`` examples spread evenly over (language, category) cells.

    Round-robin over cells rather than a flat random draw: the raw distribution
    is dominated by Go/Python style nits, and a flat sample would leave the
    benchmark unable to say anything about, say, C# error handling. Within a
    cell the highest-grounding examples are taken first, which is where the
    "quality over quantity" claim is actually cashed out.
    """
    if len(examples) <= target:
        return list(examples)

    rng = random.Random(seed)
    cells: dict[tuple[str, str], list[Example]] = defaultdict(list)
    for ex in examples:
        cells[(ex.lang, ex.category)].append(ex)

    for bucket in cells.values():
        # Shuffle first so ties in grounding do not always resolve to the same
        # repository, then sort by the quality signal.
        rng.shuffle(bucket)
        bucket.sort(key=lambda e: e.grounding, reverse=True)

    keys = sorted(cells)
    rng.shuffle(keys)
    out: list[Example] = []
    cursor = 0
    while len(out) < target:
        progressed = False
        for key in keys:
            bucket = cells[key]
            if cursor < len(bucket):
                out.append(bucket[cursor])
                progressed = True
                if len(out) >= target:
                    break
        if not progressed:
            break
        cursor += 1
    return out
