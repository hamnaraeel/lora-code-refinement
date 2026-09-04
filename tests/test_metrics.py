"""Metric correctness. The central claim of the project is a difference between
two numbers produced here, so these need to be right."""

import pytest

from coderefine.metrics import (
    aggregate,
    corpus_bleu,
    normalize,
    score_example,
    tokenize,
)

OLD = "def f(x):\n    return x + 1"
GOLD = "def f(x):\n    return x + 2"


class TestNormalize:
    def test_strips_trailing_whitespace(self):
        assert normalize("a   \nb\t") == "a\nb"

    def test_preserves_leading_whitespace(self):
        assert normalize("    indented") == "    indented"

    def test_normalises_line_endings(self):
        assert normalize("a\r\nb") == "a\nb"

    def test_trims_surrounding_blank_lines(self):
        assert normalize("\n\na\n\n") == "a"


class TestTokenize:
    def test_splits_punctuation_from_identifiers(self):
        assert tokenize("foo(bar)") == ["foo", "(", "bar", ")"]

    def test_keeps_underscored_identifiers_whole(self):
        assert tokenize("my_var = 1") == ["my_var", "=", "1"]


class TestScoreExample:
    def test_perfect_prediction(self):
        s = score_example(GOLD, GOLD, OLD)
        assert s.exact_match and s.improved and not s.regressed and not s.copied

    def test_copying_the_input_is_never_an_improvement(self):
        s = score_example(OLD, GOLD, OLD)
        assert s.copied
        assert not s.improved
        assert not s.exact_match
        assert s.delta_to_gold == pytest.approx(0.0)

    def test_moving_away_from_gold_is_a_regression(self):
        s = score_example("completely unrelated text here", GOLD, OLD)
        assert s.regressed and not s.improved

    def test_whitespace_only_difference_still_matches(self):
        assert score_example(GOLD + "   ", GOLD, OLD).exact_match

    def test_indentation_difference_does_not_match(self):
        assert not score_example("def f(x):\n        return x + 2", GOLD, OLD).exact_match

    def test_syntax_check_skipped_when_input_was_already_unparseable(self):
        # Hunks are excerpts and are often incomplete; penalising the model for
        # the dataset's truncation would measure the wrong thing.
        s = score_example("if x:", "if y:", "if x:", lang="py")
        assert s.syntax_ok is None

    def test_syntax_check_runs_when_input_was_parseable(self):
        assert score_example("x = 1", "x = 2", "x = 0", lang="py").syntax_ok is True
        assert score_example("x = (", "x = 2", "x = 0", lang="py").syntax_ok is False

    def test_syntax_check_is_none_for_non_python(self):
        assert score_example("a", "b", "c", lang="go").syntax_ok is None

    def test_edit_line_f1_rewards_changing_the_right_line(self):
        right = score_example(GOLD, GOLD, OLD).changed_right_lines
        wrong = score_example("def g(x):\n    return x + 1", GOLD, OLD).changed_right_lines
        assert right == 1.0
        assert wrong < right


class TestAggregate:
    def _records(self, preds):
        return [
            {"prediction": p, "gold": GOLD, "old_code": OLD, "lang": "py",
             "category": "logic_bug", "parse_mode": "fenced"}
            for p in preds
        ]

    def test_copy_baseline_is_reported_and_beats_naive_bleu(self):
        # The point of the whole module: a do-nothing model looks great on BLEU.
        m = aggregate(self._records([OLD, OLD, OLD]))
        assert m["exact_match"] == 0.0
        assert m["improved_rate"] == 0.0
        assert m["copy_rate"] == 1.0
        assert m["bleu"] > 50  # a copier scores high...
        assert m["bleu"] == pytest.approx(m["copy_baseline_bleu"], abs=1e-6)  # ...and the floor says so

    def test_perfect_predictions(self):
        m = aggregate(self._records([GOLD, GOLD]))
        assert m["exact_match"] == 1.0
        assert m["improved_rate"] == 1.0
        assert m["copy_rate"] == 0.0

    def test_empty_input(self):
        assert aggregate([])["n"] == 0

    def test_slices_are_present(self):
        m = aggregate(self._records([GOLD, OLD]))
        assert m["by_category"]["logic_bug"]["n"] == 2
        assert m["by_lang"]["py"]["n"] == 2

    def test_format_compliance_tracks_parse_mode(self):
        recs = self._records([GOLD, GOLD])
        recs[0]["parse_mode"] = "raw"
        assert aggregate(recs)["format_compliance"] == pytest.approx(0.5)


class TestCorpusBleu:
    def test_identical_corpora_score_100(self):
        assert corpus_bleu([GOLD, OLD], [GOLD, OLD]) == pytest.approx(100.0, abs=1e-6)

    def test_empty_is_zero(self):
        assert corpus_bleu([], []) == 0.0


class TestPercentile:
    """p95 must never fall below p50, at any sample size."""

    def test_monotonic_in_q_for_tiny_samples(self):
        from coderefine.metrics import percentile

        for n in range(1, 12):
            values = sorted(float(i) for i in range(n))
            p50 = percentile(values, 0.50)
            p95 = percentile(values, 0.95)
            assert p50 <= p95, f"n={n}: p50={p50} > p95={p95}"

    def test_two_sample_case_that_regressed(self):
        # The exact shape that produced "p95 faster than p50" in /metrics.
        from coderefine.metrics import percentile

        assert percentile([18.473, 24.849], 0.95) == 24.849
        assert percentile([18.473, 24.849], 0.50) == 18.473

    def test_single_sample(self):
        from coderefine.metrics import percentile

        assert percentile([7.0], 0.95) == 7.0

    def test_empty(self):
        from coderefine.metrics import percentile

        assert percentile([], 0.5) is None

    def test_matches_nearest_rank_on_a_round_sample(self):
        from coderefine.metrics import percentile

        values = [float(i) for i in range(1, 101)]
        assert percentile(values, 0.50) == 50.0
        assert percentile(values, 0.95) == 95.0
