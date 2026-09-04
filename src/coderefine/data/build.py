"""Build leak-free train / validation / test splits plus a dataset card.

Leakage discipline
------------------
Three independent guards, because each catches something the others miss:

1. **Provenance.** train/valid/test are curated from the *separate* upstream
   dumps (``ref-train``, ``ref-valid``, ``ref-test``). The CodeReviewer authors
   already split those by project and time.
2. **Content hashing.** A single ``seen`` set spans all three files, processed
   train-first. Any example whose (code, comment) fingerprint already appeared
   is dropped, so a snippet vendored into two repos cannot cross the boundary.
3. **Repository disjointness.** Any validation or test example whose repository
   appears anywhere in train is dropped. This is the strict one: examples
   derived from the same source file — the exact failure mode called out in the
   brief — share a repo even when their content differs, so cutting at the repo
   boundary is the only guard that actually stops them.

The test split is written once and never read by training or by hyperparameter
selection. Model selection uses validation only; ``evaluate.py`` refuses to
score the test split unless ``--final`` is passed.
"""

from __future__ import annotations

import json
import hashlib
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from .curate import CurationConfig, Example, balanced_sample, curate_file


def _assign_side(repo: str) -> str:
    """Deterministically send a repository to either validation or test.

    A hash of the repository name rather than a random draw, so that rebuilding
    the dataset — on another machine, with another seed — puts the same
    projects on the same side. Reproducibility of the *split boundary* matters
    more here than of the sampling within it.
    """
    digest = hashlib.sha1(repo.encode("utf-8")).hexdigest()
    return "valid" if int(digest, 16) % 2 == 0 else "test"


def _write_jsonl(path: Path, examples: list[Example]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(ex.to_json() + "\n")


def build_splits(
    raw_dir: Path,
    out_dir: Path,
    cfg: CurationConfig,
    n_train: int = 2000,
    n_valid: int = 250,
    n_test: int = 250,
    scan_limit: int | None = None,
    seed: int = 13,
) -> dict:
    """Run the full curation funnel and write the three splits + a dataset card.

    ``scan_limit`` caps how many raw records are read per file. The train dump
    is 5.3 GB; a few hundred thousand records is far more than enough to fill a
    2,000-example budget, and capping keeps a full rebuild to a couple of
    minutes instead of half an hour.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    funnels: dict[str, dict] = {}
    pools: dict[str, list[Example]] = {}

    # Order matters: train claims content hashes first, so any collision is
    # resolved by dropping the *evaluation* copy. The reverse would silently
    # shrink training data while leaving a leak in place.
    plan = [
        ("train", raw_dir / "ref-train.jsonl", n_train),
        ("valid", raw_dir / "ref-valid.jsonl", n_valid),
        ("test", raw_dir / "ref-test.jsonl", n_test),
    ]

    for split, path, _ in plan:
        if not path.exists():
            raise FileNotFoundError(f"Missing raw dump: {path}")
        pool, stats = curate_file(path, cfg, split, limit=scan_limit, seen_hashes=seen)
        pools[split] = pool
        funnels[split] = stats.as_dict()

    # --- Guard 3: repository disjointness -----------------------------------
    # 3a. Nothing in valid or test may come from a repository seen in training.
    train_repos = {ex.repo for ex in pools["train"]}
    repo_drops: dict[str, int] = {}
    for split in ("valid", "test"):
        before = len(pools[split])
        pools[split] = [ex for ex in pools[split] if ex.repo not in train_repos]
        repo_drops[split] = before - len(pools[split])

    # 3b. Valid and test must not share repositories either, so that tuning
    #     against validation cannot indirectly tune against test. Upstream does
    #     not guarantee this — its valid/test cut is by pull request, so the two
    #     dumps draw on largely the same projects. *Dropping* the overlap would
    #     destroy the test split (it removes ~99% of it), so instead each shared
    #     repository is assigned to exactly one side by a stable hash of its
    #     name. Deterministic, seed-independent, and roughly an even split, so
    #     both halves stay large and neither can see the other's projects.
    valid_repos = {ex.repo for ex in pools["valid"]}
    test_repos = {ex.repo for ex in pools["test"]}
    contested = valid_repos & test_repos
    assignment = {repo: _assign_side(repo) for repo in contested}

    before_valid, before_test = len(pools["valid"]), len(pools["test"])
    pools["valid"] = [
        ex for ex in pools["valid"] if assignment.get(ex.repo, "valid") == "valid"
    ]
    pools["test"] = [
        ex for ex in pools["test"] if assignment.get(ex.repo, "test") == "test"
    ]
    repo_drops["valid_lost_to_test"] = before_valid - len(pools["valid"])
    repo_drops["test_lost_to_valid"] = before_test - len(pools["test"])
    repo_drops["contested_repos"] = len(contested)

    # --- Balanced down-sampling to the target budget ------------------------
    splits: dict[str, list[Example]] = {}
    for split, _, target in plan:
        splits[split] = balanced_sample(pools[split], target, seed=seed)
        _write_jsonl(out_dir / f"{split}.jsonl", splits[split])

    card = _dataset_card(splits, funnels, repo_drops, cfg, scan_limit, seed)
    (out_dir / "dataset_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    return card


def _dataset_card(
    splits: dict[str, list[Example]],
    funnels: dict[str, dict],
    repo_drops: dict[str, int],
    cfg: CurationConfig,
    scan_limit: int | None,
    seed: int,
) -> dict:
    """Everything a reader needs to judge whether the splits are trustworthy."""

    def describe(examples: list[Example]) -> dict:
        if not examples:
            return {"n": 0}
        grounding = sorted(e.grounding for e in examples)
        old_lines = sorted(e.n_old_lines for e in examples)
        return {
            "n": len(examples),
            "languages": dict(Counter(e.lang for e in examples).most_common()),
            "categories": dict(Counter(e.category for e in examples).most_common()),
            "n_repos": len({e.repo for e in examples}),
            "grounding": {
                "mean": round(sum(grounding) / len(grounding), 4),
                "median": grounding[len(grounding) // 2],
                "min": grounding[0],
            },
            "hunk_lines": {
                "mean": round(sum(old_lines) / len(old_lines), 2),
                "median": old_lines[len(old_lines) // 2],
                "max": old_lines[-1],
            },
            "mean_comment_chars": round(
                sum(len(e.comment) for e in examples) / len(examples), 1
            ),
        }

    total = sum(len(v) for v in splits.values()) or 1
    repos = {k: {e.repo for e in v} for k, v in splits.items()}
    ids = {k: {e.id for e in v} for k, v in splits.items()}

    return {
        "task": "code_refinement_from_review_comment",
        "source": "Microsoft CodeReviewer refinement dumps (ref-train/valid/test.jsonl)",
        "curation_config": asdict(cfg),
        "scan_limit_per_file": scan_limit,
        "seed": seed,
        "splits": {k: describe(v) for k, v in splits.items()},
        "split_ratio": {k: round(len(v) / total, 3) for k, v in splits.items()},
        "leakage_checks": {
            "content_hash_collisions_dropped": {
                k: funnels[k]["rejected_by_filter"].get("duplicate", 0) for k in funnels
            },
            "dropped_for_repo_overlap": repo_drops,
            "repo_overlap_train_valid": sorted(repos["train"] & repos["valid"]),
            "repo_overlap_train_test": sorted(repos["train"] & repos["test"]),
            "repo_overlap_valid_test": sorted(repos["valid"] & repos["test"]),
            "id_overlap_any": sorted(
                (ids["train"] & ids["valid"])
                | (ids["train"] & ids["test"])
                | (ids["valid"] & ids["test"])
            ),
        },
        "funnel": funnels,
    }


def load_split(path: Path) -> list[dict]:
    """Read a curated split back as plain dicts."""
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
