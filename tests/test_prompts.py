"""Prompt construction and output parsing.

The fairness of every comparison in this project rests on base and fine-tuned
models receiving identical prompts, so the builders are tested for stability.
"""

from coderefine.prompts import (
    SYSTEM_PROMPT,
    build_messages,
    build_target,
    build_training_messages,
    build_user_prompt,
    lang_fence,
    lang_name,
    parse_generation,
)


class TestPromptBuilders:
    def test_user_prompt_contains_all_three_inputs(self):
        p = build_user_prompt("x = 1", "rename x to count", "py")
        assert "x = 1" in p and "rename x to count" in p and "Python" in p

    def test_prompt_is_deterministic(self):
        a = build_user_prompt("x = 1", "c", "py")
        b = build_user_prompt("x = 1", "c", "py")
        assert a == b

    def test_messages_have_no_assistant_turn(self):
        msgs = build_messages("x", "c", "py")
        assert [m["role"] for m in msgs] == ["system", "user"]

    def test_training_messages_append_the_gold_turn(self):
        msgs = build_training_messages("x", "c", "y", "py")
        assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
        assert "y" in msgs[-1]["content"]

    def test_generation_and_training_share_the_same_prefix(self):
        # If these ever diverge, the model is trained on one prompt and
        # evaluated on another, and every reported delta becomes meaningless.
        gen = build_messages("x", "c", "py")
        train = build_training_messages("x", "c", "y", "py")
        assert train[: len(gen)] == gen

    def test_system_prompt_forbids_unrelated_changes(self):
        assert "nothing else" in SYSTEM_PROMPT

    def test_language_names_and_fences(self):
        assert lang_name("py") == "Python" and lang_fence("py") == "python"
        assert lang_name(".cs") == "C#" and lang_fence(".cs") == "csharp"
        assert lang_name("unknown-lang") == "unknown-lang"

    def test_target_is_a_fenced_block(self):
        assert build_target("x = 1", "py").startswith("```python")


class TestParseGeneration:
    def test_extracts_a_fenced_block(self):
        p = parse_generation("Sure!\n```python\nx = 1\n```\nHope that helps.")
        assert p.code == "x = 1" and p.how == "fenced"

    def test_takes_the_first_block_only(self):
        # A rambling model often follows a correct answer with an "or you could
        # ..." variant; only the first should be credited.
        p = parse_generation("```py\nfirst\n```\ntext\n```py\nsecond\n```")
        assert p.code == "first"

    def test_bare_code_is_still_recovered(self):
        p = parse_generation("x = 1\ny = 2")
        assert p.code == "x = 1\ny = 2" and p.how == "raw"

    def test_conversational_preamble_is_stripped(self):
        assert parse_generation("Here's the revised code:\nx = 1").code == "x = 1"

    def test_empty_generation(self):
        assert parse_generation("").how == "empty"
        assert parse_generation("   \n  ").how == "empty"

    def test_unterminated_block_is_recovered_and_flagged(self):
        p = parse_generation("```python\nx = 1\ny = 2")
        assert p.code == "x = 1\ny = 2"
        assert p.how == "fenced_unterminated"

    def test_internal_indentation_is_preserved(self):
        p = parse_generation("```python\ndef f():\n    return 1\n```")
        assert p.code == "def f():\n    return 1"

    def test_edge_blank_lines_are_trimmed(self):
        assert parse_generation("```py\n\n\nx = 1\n\n\n```").code == "x = 1"
