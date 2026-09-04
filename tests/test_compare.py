"""Statistics and leakage tests.

The report quotes confidence intervals and p-values; if those are wrong the
project overclaims, which is the worst failure mode available to it.
"""

import json
import math

import pytest

from coderefine.compare import bootstrap_ci, compare_runs, mcnemar_exact
from coderefine.data.taxonomy import CATEGORIES, classify


class TestMcNemar:
    def test_no_discordant_pairs_gives_p_one(self):
        r = mcnemar_exact([True, False], [True, False])
        assert r["n_discordant"] == 0 and r["p_value"] == 1.0

    def test_counts_fixes_and_breaks(self):
        base = [False, False, True, True]
        tuned = [True, True, False, True]
        r = mcnemar_exact(base, tuned)
        assert r["fixed"] == 2 and r["broken"] == 1

    def test_all_one_sided_with_enough_pairs_is_significant(self):
        base = [False] * 10
        tuned = [True] * 10
        r = mcnemar_exact(base, tuned)
        assert r["p_value"] == pytest.approx(2 / 1024, abs=1e-9)
        assert r["significant_at_05"]

    def test_small_one_sided_sample_is_not_significant(self):
        # Three fixes and no breaks: p = 2 * (1/8) = 0.25. Correctly unconvincing.
        r = mcnemar_exact([False] * 3, [True] * 3)
        assert r["p_value"] == pytest.approx(0.25)
        assert not r["significant_at_05"]

    def test_p_value_never_exceeds_one(self):
        r = mcnemar_exact([False, True], [True, False])
        assert r["p_value"] <= 1.0


class TestBootstrap:
    def test_delta_matches_the_paired_mean(self):
        r = bootstrap_ci([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], n_resamples=500)
        assert r["delta"] == pytest.approx(1.0)

    def test_identical_inputs_give_a_zero_interval(self):
        r = bootstrap_ci([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], n_resamples=500)
        assert r["delta"] == 0.0 and r["ci_low"] == 0.0 and r["ci_high"] == 0.0
        assert not r["significant"]

    def test_interval_brackets_the_estimate(self):
        a = [0.0] * 40
        b = [1.0] * 20 + [0.0] * 20
        r = bootstrap_ci(a, b, n_resamples=2000, seed=1)
        assert r["ci_low"] <= r["delta"] <= r["ci_high"]

    def test_noisy_difference_is_not_called_significant(self):
        a = [0.0, 1.0] * 10
        b = [1.0, 0.0] * 10
        assert not bootstrap_ci(a, b, n_resamples=2000, seed=2)["significant"]

    def test_deterministic_for_a_seed(self):
        a, b = [0.0] * 20, [1.0] * 10 + [0.0] * 10
        assert bootstrap_ci(a, b, n_resamples=800, seed=5) == bootstrap_ci(a, b, n_resamples=800, seed=5)

    def test_empty_input_is_handled(self):
        assert bootstrap_ci([], [], n_resamples=10)["n_resamples"] == 0


def _pred(i, prediction, gold="G", old="O"):
    from coderefine.metrics import score_example
    s = score_example(prediction, gold, old)
    return {
        "id": f"e{i}", "lang": "py", "category": "naming", "origin": "curated",
        "probe": "", "comment": "c", "old_code": old, "gold": gold,
        "prediction": prediction, "parse_mode": "fenced",
        "scores": {
            "exact_match": s.exact_match, "improved": s.improved,
            "regressed": s.regressed, "copied": s.copied,
            "edit_sim": s.edit_sim, "delta_to_gold": s.delta_to_gold,
            "edit_line_f1": s.changed_right_lines, "token_f1": s.token_f1,
            "syntax_ok": s.syntax_ok,
        },
    }


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


class TestCompareRuns:
    def test_detects_a_clean_improvement(self, tmp_path):
        base = tmp_path / "b.jsonl"; tuned = tmp_path / "t.jsonl"
        _write(base, [_pred(i, "O") for i in range(8)])     # base copies the input
        _write(tuned, [_pred(i, "G") for i in range(8)])    # tuned is perfect
        r = compare_runs(base, tuned, tmp_path / "out.json")
        h = r["headline"]
        assert h["base_exact_match"] == 0.0 and h["tuned_exact_match"] == 1.0
        assert h["examples_fixed"] == 8 and h["examples_broken"] == 0
        assert h["base_copy_rate"] == 1.0 and h["tuned_copy_rate"] == 0.0
        assert "improved" in h["verdict"]

    def test_verdict_hedges_when_the_interval_includes_zero(self, tmp_path):
        base = tmp_path / "b.jsonl"; tuned = tmp_path / "t.jsonl"
        _write(base, [_pred(i, "G" if i % 2 else "O") for i in range(6)])
        _write(tuned, [_pred(i, "O" if i % 2 else "G") for i in range(6)])
        r = compare_runs(base, tuned, tmp_path / "out.json")
        assert "too small" in r["headline"]["verdict"]

    def test_reports_examples_present_in_only_one_file(self, tmp_path):
        base = tmp_path / "b.jsonl"; tuned = tmp_path / "t.jsonl"
        _write(base, [_pred(i, "O") for i in range(4)])
        _write(tuned, [_pred(i, "G") for i in range(3)])
        r = compare_runs(base, tuned, tmp_path / "out.json")
        assert r["coverage"]["compared"] == 3
        assert r["coverage"]["in_base_only"] == ["e3"]

    def test_disjoint_files_raise(self, tmp_path):
        base = tmp_path / "b.jsonl"; tuned = tmp_path / "t.jsonl"
        _write(base, [_pred(1, "O")])
        _write(tuned, [_pred(99, "G")])
        with pytest.raises(ValueError, match="share no example ids"):
            compare_runs(base, tuned, tmp_path / "out.json")

    def test_output_file_is_written(self, tmp_path):
        base = tmp_path / "b.jsonl"; tuned = tmp_path / "t.jsonl"
        _write(base, [_pred(i, "O") for i in range(3)])
        _write(tuned, [_pred(i, "G") for i in range(3)])
        out = tmp_path / "nested" / "out.json"
        compare_runs(base, tuned, out)
        assert json.loads(out.read_text())["headline"]["n"] == 3


class TestTaxonomy:
    def test_suggestion_blocks_win_over_everything(self):
        # Must be isolated: the reviewer supplied the answer verbatim.
        assert classify("rename this\n```suggestion\nfoo = 1\n```") == "suggested_edit"

    @pytest.mark.parametrize("comment,expected", [
        ("typo: recieve -> receive", "typo_or_docs"),
        ("this can be null here", "null_check"),
        ("nit: extra blank line", "style_format"),
        ("Why is this method public?", "design_question"),
        ("this is a SQL injection risk", "security"),
        ("please add a unit test for this", "testing"),
    ])
    def test_representative_comments(self, comment, expected):
        assert classify(comment) == expected

    def test_unmatched_comment_falls_through_to_other(self):
        assert classify("hmm") == "other"

    def test_every_returned_label_is_declared(self):
        for c in ["typo here", "null", "why?", "zzz", "```suggestion\nx\n```"]:
            assert classify(c) in CATEGORIES
