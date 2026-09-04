"""Curation correctness. These filters decide what the model learns, and a bug
here is invisible downstream — the model just gets quietly worse."""

from coderefine.data.curate import (
    CurationConfig,
    balanced_sample,
    changed_line_count,
    grounding_score,
    strip_markers,
)
from coderefine.data.curate import Example


class TestStripMarkers:
    def test_old_side_keeps_context_and_removals(self):
        raw = " keep\n-gone\n context"
        assert strip_markers(raw, "old") == "keep\ngone\ncontext"

    def test_new_side_keeps_context_and_additions(self):
        raw = " keep\n+added\n context"
        assert strip_markers(raw, "new") == "keep\nadded\ncontext"

    def test_old_side_drops_additions(self):
        assert strip_markers(" a\n+new\n b", "old") == "a\nb"

    def test_new_side_drops_removals(self):
        assert strip_markers(" a\n-old\n b", "new") == "a\nb"

    def test_no_newline_pseudo_line_is_dropped(self):
        assert strip_markers(" a\n\\ No newline at end of file", "old") == "a"

    def test_indentation_after_marker_is_preserved(self):
        # The marker occupies column 0; everything after it is real source and
        # indentation is semantic.
        assert strip_markers("     indented", "old") == "    indented"

    def test_blank_lines_survive(self):
        assert strip_markers(" a\n\n b", "old") == "a\n\nb"


class TestChangedLineCount:
    def test_counts_both_sides(self):
        assert changed_line_count(" a\n-b\n-c", " a\n+d") == 3

    def test_zero_when_nothing_marked(self):
        assert changed_line_count(" a\n b", " a\n b") == 0


class TestGroundingScore:
    def test_backticked_identifier_in_added_code_scores_high(self):
        score = grounding_score(
            "use `errors.Is` here instead", "if err == target {", "if errors.Is(err, target) {"
        )
        assert score >= 0.5

    def test_unrelated_comment_scores_low(self):
        score = grounding_score(
            "please rebase onto main when you get a chance",
            "alpha = compute(beta)",
            "gamma = evaluate(delta)",
        )
        assert score < 0.25

    def test_score_is_bounded(self):
        for s in (
            grounding_score("`x`", "y = 1", "x = 1"),
            grounding_score("", "a", "b"),
            grounding_score("a b c", "", ""),
        ):
            assert 0.0 <= s <= 1.0


def _example(i, lang, category, grounding=0.5):
    return Example(
        id=f"id{i}", repo=f"r{i % 7}", lang=lang, old_code="a", new_code="b",
        comment="c", category=category, grounding=grounding, n_old_lines=3,
        n_changed_lines=1, source_split="train",
    )


class TestBalancedSample:
    def test_returns_everything_when_pool_is_small(self):
        pool = [_example(i, "py", "naming") for i in range(5)]
        assert len(balanced_sample(pool, 10)) == 5

    def test_respects_the_target_size(self):
        pool = [_example(i, "py", "naming") for i in range(100)]
        assert len(balanced_sample(pool, 30)) == 30

    def test_spreads_across_cells_rather_than_taking_the_biggest(self):
        # 90 Python + 10 Go. A flat sample of 20 would take ~2 Go examples;
        # round-robin over cells should take far more.
        pool = [_example(i, "py", "naming") for i in range(90)]
        pool += [_example(100 + i, "go", "naming") for i in range(10)]
        picked = balanced_sample(pool, 20)
        assert sum(1 for e in picked if e.lang == "go") >= 8

    def test_prefers_higher_grounding_within_a_cell(self):
        pool = [_example(i, "py", "naming", grounding=i / 100) for i in range(100)]
        picked = balanced_sample(pool, 10)
        assert min(e.grounding for e in picked) > 0.5

    def test_is_deterministic_for_a_given_seed(self):
        pool = [_example(i, "py" if i % 2 else "go", "naming", i / 100) for i in range(60)]
        a = [e.id for e in balanced_sample(pool, 20, seed=3)]
        b = [e.id for e in balanced_sample(pool, 20, seed=3)]
        assert a == b


class TestCurationConfigDefaults:
    def test_all_nine_languages_are_selected(self):
        assert len(CurationConfig().languages) == 9
