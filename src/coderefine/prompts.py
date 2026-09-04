"""Prompt construction — the single source of truth for how the task is posed.

Every stage of the pipeline (training, base-model evaluation, fine-tuned
evaluation, the A/B server) imports the builders here. That is deliberate: the
central claim of this project is "fine-tuning improved the model", and that
claim is only honest if the base and fine-tuned models are shown byte-identical
prompts. Duplicating the template in the training script and the eval script is
the single easiest way to accidentally cheat, so there is exactly one copy.

The task
--------
Given a code hunk as it stood *before* review, plus the reviewer's
natural-language comment, produce the hunk as it stood *after* the author
addressed that comment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Language display names. The dataset stores terse extensions; spelling the
# language out gives the model a token sequence it has actually seen a lot of
# during pre-training ("Python" >> "py").
# ---------------------------------------------------------------------------
LANG_NAMES: dict[str, str] = {
    "py": "Python",
    "java": "Java",
    "go": "Go",
    "js": "JavaScript",
    "php": "PHP",
    "rb": "Ruby",
    "c": "C",
    "cpp": "C++",
    ".cs": "C#",
    "cs": "C#",
}

# Markdown fence hints, so the model emits a fenced block we can parse back out.
LANG_FENCE: dict[str, str] = {
    "py": "python",
    "java": "java",
    "go": "go",
    "js": "javascript",
    "php": "php",
    "rb": "ruby",
    "c": "c",
    "cpp": "cpp",
    ".cs": "csharp",
    "cs": "csharp",
}


def lang_name(lang: str) -> str:
    return LANG_NAMES.get(lang, lang)


def lang_fence(lang: str) -> str:
    return LANG_FENCE.get(lang, "")


SYSTEM_PROMPT = (
    "You are an expert software engineer applying code review feedback. "
    "You are given a snippet of code and a reviewer's comment about it. "
    "Rewrite the snippet so that it addresses the comment.\n\n"
    "Rules:\n"
    "1. Output ONLY the revised code inside a single fenced code block.\n"
    "2. Change exactly what the comment asks for and nothing else. Preserve all "
    "unrelated lines, including their original indentation and whitespace.\n"
    "3. Do not add explanations, commentary, or extra code fences.\n"
    "4. The snippet is an excerpt from a larger file. It may be syntactically "
    "incomplete; keep it that way rather than closing braces or adding imports."
)


def build_user_prompt(old_code: str, comment: str, lang: str) -> str:
    """Render the user turn. Used identically for base and fine-tuned models."""
    fence = lang_fence(lang)
    return (
        f"Language: {lang_name(lang)}\n\n"
        f"Code under review:\n"
        f"```{fence}\n{old_code}\n```\n\n"
        f"Reviewer comment:\n"
        f"{comment.strip()}\n\n"
        f"Revised code:"
    )


def build_target(new_code: str, lang: str) -> str:
    """Render the assistant turn used as the supervised training target."""
    fence = lang_fence(lang)
    return f"```{fence}\n{new_code}\n```"


def build_messages(old_code: str, comment: str, lang: str) -> list[dict[str, str]]:
    """Chat-format prompt (no assistant turn) for generation."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(old_code, comment, lang)},
    ]


def build_training_messages(
    old_code: str, comment: str, new_code: str, lang: str
) -> list[dict[str, str]]:
    """Full chat-format conversation including the gold assistant turn."""
    return build_messages(old_code, comment, lang) + [
        {"role": "assistant", "content": build_target(new_code, lang)}
    ]


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+#.-]*\n(.*?)(?:\n```|\Z)", re.DOTALL)


@dataclass
class ParsedGeneration:
    """A model completion decomposed into the code we will score and how we got it."""

    code: str
    #: "fenced" when a proper code block was found, "fenced_unterminated" when
    #: the block ran past the generation budget, "raw" when the model emitted
    #: bare code, "empty" when nothing usable came back. Tracked because base
    #: models fail format compliance far more often than tuned ones, and that
    #: is itself a headline result worth reporting rather than silently fixing.
    how: str


_PREAMBLE_RE = re.compile(
    r"^\s*(here(?:'s| is)[^\n:]*:|revised code:|sure[,!][^\n]*|certainly[,!][^\n]*)\s*",
    re.IGNORECASE,
)


def parse_generation(text: str) -> ParsedGeneration:
    """Pull the revised code out of a raw model completion.

    Lenient on purpose. A base model that wraps correct code in three
    paragraphs of chat should be credited for the code; the format failure is
    recorded separately in ``how`` so the report can quantify it instead of
    conflating "wrong answer" with "chatty answer".
    """
    if not text:
        return ParsedGeneration("", "empty")

    match = _FENCE_RE.search(text)
    if match:
        # Take the first block: models that ramble often emit a correct answer
        # and then an "or alternatively..." variant we should not credit.
        #
        # Termination is read off the match itself rather than by searching the
        # text again for a closing fence — re-searching misidentifies the block
        # whenever the generated code happens to contain a backtick run, and
        # this flag feeds the format-compliance metric.
        terminated = match.group(0).rstrip().endswith("```")
        return ParsedGeneration(
            _strip_trailing_blank(match.group(1)),
            "fenced" if terminated else "fenced_unterminated",
        )

    # No fence at all: strip a conversational preamble and take what is left.
    stripped = _PREAMBLE_RE.sub("", text)
    stripped = _strip_trailing_blank(stripped)
    if not stripped.strip():
        return ParsedGeneration("", "empty")
    return ParsedGeneration(stripped, "raw")


def _strip_trailing_blank(code: str) -> str:
    """Drop leading/trailing blank lines but never touch intra-line whitespace.

    Indentation is semantically load-bearing in this task (Python especially,
    and exact-match scoring everywhere), so we only remove wholly empty lines
    at the edges.
    """
    lines = code.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)
