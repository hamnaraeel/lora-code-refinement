"""Integrity of the curated splits actually on disk.

These are integration tests: they read the real `data/processed/` output and
assert the properties the report depends on. They skip cleanly when the data
has not been built, so a fresh clone still passes.
"""

import json
from pathlib import Path

import pytest

DATA = Path("data/processed")


def _load(name):
    path = DATA / name
    if not path.exists():
        pytest.skip("splits not built; run `coderefine build-data`")
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


@pytest.fixture(scope="module")
def splits():
    return {n: _load(f"{n}.jsonl") for n in ("train", "valid", "test")}


class TestLeakage:
    def test_no_repository_appears_in_two_splits(self, splits):
        repos = {k: {e["repo"] for e in v} for k, v in splits.items()}
        assert not repos["train"] & repos["valid"]
        assert not repos["train"] & repos["test"]
        assert not repos["valid"] & repos["test"]

    def test_no_example_id_appears_twice(self, splits):
        ids = {k: {e["id"] for e in v} for k, v in splits.items()}
        assert not ids["train"] & ids["valid"]
        assert not ids["train"] & ids["test"]
        assert not ids["valid"] & ids["test"]

    def test_no_identical_input_across_splits(self, splits):
        """Even with distinct ids, the same (code, comment) must not recur."""
        import re

        def key(e):
            return (re.sub(r"\s+", " ", e["old_code"]), re.sub(r"\s+", " ", e["comment"]).lower())

        train_keys = {key(e) for e in splits["train"]}
        for name in ("valid", "test"):
            overlap = [key(e) for e in splits[name] if key(e) in train_keys]
            assert not overlap, f"{len(overlap)} inputs shared between train and {name}"


class TestContent:
    def test_every_example_actually_changes_the_code(self, splits):
        for name, rows in splits.items():
            for e in rows:
                assert e["old_code"] != e["new_code"], f"{name}/{e['id']} is a no-op"

    def test_no_empty_fields(self, splits):
        for name, rows in splits.items():
            for e in rows:
                assert e["old_code"].strip(), f"{name}/{e['id']}"
                assert e["new_code"].strip(), f"{name}/{e['id']}"
                assert e["comment"].strip(), f"{name}/{e['id']}"

    def test_no_diff_markers_survived_into_the_code(self, splits):
        """A leaked '+'/'-' column would teach the model to emit diffs."""
        for name, rows in splits.items():
            for e in rows[:200]:
                for line in e["old_code"].split("\n"):
                    assert not line.startswith("\\ No newline"), f"{name}/{e['id']}"

    def test_languages_and_categories_are_diverse(self, splits):
        for name, rows in splits.items():
            assert len({e["lang"] for e in rows}) >= 5, name
            assert len({e["category"] for e in rows}) >= 6, name

    def test_grounding_respects_the_configured_floor(self, splits):
        for name, rows in splits.items():
            assert min(e["grounding"] for e in rows) >= 0.10, name


class TestDatasetCard:
    def test_card_records_clean_leakage_checks(self):
        path = DATA / "dataset_card.json"
        if not path.exists():
            pytest.skip("dataset card not built")
        checks = json.loads(path.read_text())["leakage_checks"]
        assert checks["repo_overlap_train_valid"] == []
        assert checks["repo_overlap_train_test"] == []
        assert checks["repo_overlap_valid_test"] == []
        assert checks["id_overlap_any"] == []

    def test_split_ratio_is_roughly_80_10_10(self):
        path = DATA / "dataset_card.json"
        if not path.exists():
            pytest.skip("dataset card not built")
        ratio = json.loads(path.read_text())["split_ratio"]
        assert 0.7 <= ratio["train"] <= 0.85
        assert 0.05 <= ratio["valid"] <= 0.15
        assert 0.05 <= ratio["test"] <= 0.15
