"""Benchmark assembly and the forgetting probe suite."""

import json
from pathlib import Path

import pytest

from coderefine.data.benchmark import build_benchmark, hardness
from coderefine.forgetting import _check, compare_forgetting, load_probes


class TestHardness:
    def test_low_grounding_is_harder(self):
        easy = {"grounding": 0.95, "n_changed_lines": 1, "category": "typo_or_docs"}
        hard = {"grounding": 0.05, "n_changed_lines": 1, "category": "typo_or_docs"}
        assert hardness(hard) > hardness(easy)

    def test_implied_edit_categories_are_harder(self):
        stated = {"grounding": 0.5, "n_changed_lines": 1, "category": "typo_or_docs"}
        implied = {"grounding": 0.5, "n_changed_lines": 1, "category": "design_question"}
        assert hardness(implied) > hardness(stated)

    def test_larger_edits_are_harder(self):
        small = {"grounding": 0.5, "n_changed_lines": 1, "category": "naming"}
        large = {"grounding": 0.5, "n_changed_lines": 6, "category": "naming"}
        assert hardness(large) > hardness(small)

    def test_missing_grounding_does_not_crash(self):
        assert 0.0 <= hardness({"category": "naming"}) <= 1.0


def _pool_row(i, category, grounding):
    return {
        "id": f"p{i}", "repo": f"r{i}", "lang": "py", "old_code": "a", "new_code": "b",
        "comment": "c", "category": category, "grounding": grounding,
        "n_old_lines": 3, "n_changed_lines": 2, "source_split": "test",
    }


class TestBuildBenchmark:
    def _setup(self, tmp_path, n_pool=60):
        test_path = tmp_path / "test.jsonl"
        cats = ["naming", "logic_bug", "security", "design_question", "typo_or_docs"]
        rows = [_pool_row(i, cats[i % len(cats)], (i % 10) / 10) for i in range(n_pool)]
        test_path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

        hw_path = tmp_path / "hw.jsonl"
        hw = [{**_pool_row(1000 + i, "naming", None), "probe": f"probe_{i}"} for i in range(4)]
        hw_path.write_text("\n".join(json.dumps(r) for r in hw), encoding="utf-8")
        return test_path, hw_path

    def test_combines_both_halves(self, tmp_path):
        test_path, hw_path = self._setup(tmp_path)
        card = build_benchmark(test_path, hw_path, tmp_path / "bench.jsonl", n_curated=10)
        assert card["n_total"] == 14
        assert card["n_handwritten"] == 4 and card["n_curated"] == 10

    def test_curated_half_is_harder_than_the_pool(self, tmp_path):
        test_path, hw_path = self._setup(tmp_path)
        card = build_benchmark(test_path, hw_path, tmp_path / "bench.jsonl", n_curated=10)
        assert card["curated_mean_hardness"] > card["pool_mean_hardness"]

    def test_origin_is_tagged_on_every_row(self, tmp_path):
        test_path, hw_path = self._setup(tmp_path)
        out = tmp_path / "bench.jsonl"
        build_benchmark(test_path, hw_path, out, n_curated=10)
        rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
        assert {r["origin"] for r in rows} == {"handwritten", "curated"}

    def test_stratifies_across_categories(self, tmp_path):
        # A flat top-N-hardest pick would collapse onto one or two categories.
        test_path, hw_path = self._setup(tmp_path)
        card = build_benchmark(test_path, hw_path, tmp_path / "bench.jsonl", n_curated=10)
        assert len(card["categories"]) >= 4

    def test_is_deterministic_for_a_seed(self, tmp_path):
        test_path, hw_path = self._setup(tmp_path)
        a = tmp_path / "a.jsonl"; b = tmp_path / "b.jsonl"
        build_benchmark(test_path, hw_path, a, n_curated=10, seed=4)
        build_benchmark(test_path, hw_path, b, n_curated=10, seed=4)
        assert a.read_text() == b.read_text()

    def test_missing_pool_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_benchmark(tmp_path / "nope.jsonl", tmp_path / "hw.jsonl", tmp_path / "o.jsonl")


class TestHandwrittenBenchmarkFile:
    """The hand-written half is source, not output — it must stay well-formed."""

    def test_file_is_valid_and_complete(self):
        path = Path("data/benchmark/handwritten.jsonl")
        if not path.exists():
            pytest.skip("handwritten benchmark not present")
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        assert len(rows) >= 15
        for r in rows:
            assert r["old_code"].strip() and r["new_code"].strip()
            assert r["old_code"] != r["new_code"], f"{r['id']} has no actual change"
            assert r["comment"].strip() and r["lang"] and r["category"]
            assert r["probe"], f"{r['id']} is missing its failure-mode label"

    def test_every_example_probes_a_distinct_failure_mode(self):
        path = Path("data/benchmark/handwritten.jsonl")
        if not path.exists():
            pytest.skip("handwritten benchmark not present")
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        probes = [r["probe"] for r in rows]
        assert len(set(probes)) == len(probes)


class TestForgettingChecks:
    @pytest.mark.parametrize("kind,arg,text,expected", [
        ("one_word", None, "blue", True),
        ("one_word", None, "the sky is blue", False),
        ("contains", "42", "The answer is 42.", True),
        ("contains", "42", "The answer is 41.", False),
        ("yes_no", "YES", "Yes", True),
        ("yes_no", "YES", "Yes, definitely, because...", False),
        ("n_lines", 3, "a\nb\nc", True),
        ("n_lines", 3, "a\nb", False),
        ("max_words", 5, "one two three", True),
        ("max_words", 5, "one two three four five six", False),
        ("no_letter", "e", "rain on tin", True),
        ("no_letter", "e", "gentle rain", False),
        ("json_key", "answer", '{"answer": "Paris"}', True),
        ("json_key", "answer", '{"result": "Paris"}', False),
        ("json_key", "answer", "not json at all", False),
        ("json_array", 3, "[1, 2, 3]", True),
        ("json_array", 3, "[1, 2]", False),
    ])
    def test_rule_checks(self, kind, arg, text, expected):
        assert _check(kind, arg, text) is expected

    def test_json_inside_a_code_fence_still_validates(self):
        # A tuned model may wrap everything in a fence; that is measured
        # separately as spurious_code_fence_rate, not double-counted as failure.
        assert _check("json_key", "answer", '```json\n{"answer": 1}\n```')


class TestProbeSuite:
    def test_bundled_suite_loads_and_is_well_formed(self):
        probes = load_probes()
        assert len(probes["multiple_choice"]) >= 20
        assert len(probes["instruction_following"]) >= 15
        for item in probes["multiple_choice"]:
            assert 0 <= item["answer"] < len(item["choices"])
            assert len(item["choices"]) == len(set(item["choices"])), item["id"]
        for item in probes["instruction_following"]:
            assert item["check"] and item["prompt"]


class TestForgettingComparison:
    def _result(self, mcq, instr, fence=0.0):
        return {"headline": {
            "mcq_accuracy": mcq, "instruction_pass_rate": instr,
            "spurious_code_fence_rate": fence,
        }}

    def test_no_degradation_reads_as_intact(self):
        c = compare_forgetting(self._result(0.8, 0.9), self._result(0.8, 0.9))
        assert c["overall_retention_pct"] == 100.0
        assert "intact" in c["verdict"]

    def test_large_drop_is_called_out(self):
        c = compare_forgetting(self._result(0.8, 0.9), self._result(0.4, 0.5))
        assert c["overall_retention_pct"] < 60
        assert "Material degradation" in c["verdict"]

    def test_format_bleed_is_flagged(self):
        c = compare_forgetting(
            self._result(0.8, 0.9, fence=0.0), self._result(0.8, 0.9, fence=0.6)
        )
        assert "code fences" in c["verdict"]

    def test_zero_base_does_not_divide_by_zero(self):
        c = compare_forgetting(self._result(0.0, 0.0), self._result(0.5, 0.5))
        assert c["overall_retention_pct"] == 100.0
