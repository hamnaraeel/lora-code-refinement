"""Rule-based taxonomy for code-review comments.

Why this exists
---------------
Two parts of the pipeline need to know *what kind* of feedback a comment is:

1. Benchmark construction, so the 40-example evaluation set is stratified
   across feedback types rather than being 30 "fix the typo" comments.
2. Reporting, so "fine-tuning improved exact match from X% to Y%" can be
   broken down into where it helped and where it regressed.

The classifier is keyword rules rather than a model. That is a deliberate
trade: rules are deterministic, auditable, cost nothing to run over 100k
comments, and — most importantly — cannot leak information from a model that
also touches the evaluation. Precision matters more than recall here, so
ordering is specific-to-general and the first match wins.
"""

from __future__ import annotations

import re

#: Ordered most-specific first. Each entry is (category, compiled pattern).
#: Ordering is load-bearing: "fix the typo in this error message" is a typo
#: comment, not an error-handling comment, so TYPO is tested first.
_RULES: list[tuple[str, str]] = [
    (
        # Tested first, and reported separately in every results table. GitHub's
        # ```suggestion blocks contain the replacement code verbatim, so these
        # are a fundamentally easier sub-task. Folding them into the headline
        # number would inflate it; a reader is entitled to see them split out.
        "suggested_edit",
        r"```suggestion",
    ),
    (
        "typo_or_docs",
        r"\btypos?\b|\bspell(ing|ed)?\b|\bgrammar\b|\bmisspell|\bcomment (here|says|should)"
        r"|\bdocstring\b|\bjavadoc\b|\bdocs?\b\s|\bwording\b|\bcapitali[sz]",
    ),
    (
        "null_check",
        r"\bnull\b|\bnil\b|\bNone\b|\bnullptr\b|\bnpe\b|null[- ]?check|\bisset\b"
        r"|\bundefined\b|\bexists?\b.*\bcheck\b|\bcheck\b.*\bexists?\b",
    ),
    (
        "error_handling",
        r"\berror\b|\bexception\b|\bexcept\b|\braise\b|\bthrow(s|n)?\b|\bpanic\b"
        r"|\btry\b|\bcatch\b|\bfail(ure|s)?\b|\bhandle\b.*\berr|\berr\b|\blog\b.*\berror",
    ),
    (
        "naming",
        r"\bnam(e|ing|ed)\b|\brename\b|\bcall(ed|ing) it\b|\bbetter name\b|\bvariable name\b"
        r"|\bmethod name\b|\bfunction name\b|\bmisleading name\b|\bcamel[ -]?case\b|\bsnake[ -]?case\b",
    ),
    (
        "simplification",
        r"\bsimplif|\bredundant\b|\bunnecessary\b|\bnot needed\b|\bremove\b|\bdead code\b"
        r"|\bduplicat|\bextract\b|\brefactor\b|\binline\b|\bcan be\b.*\bshorter\b|\bone[- ]liner\b"
        r"|\bdrop\b.*\bthis\b|\bno need\b",
    ),
    (
        "api_usage",
        r"\buse\b|\bprefer\b|\binstead of\b|\bshould (be|use|call)\b|\breplace\b|\bswitch to\b"
        r"|\bthere('s| is) (a|an)\b.*\b(helper|method|function|util)|\bdeprecat",
    ),
    (
        "logic_bug",
        r"\boff[- ]by[- ]one\b|\bshould be\b.*[<>=]|\bwrong\b|\bincorrect\b|\bbug\b|\bbroken\b"
        r"|\bcondition\b|\bboundary\b|\bedge case\b|\brace\b|\bdeadlock\b|\bleak\b"
        r"|\b(>=|<=|!=|==)\b|\breversed\b|\binverted\b",
    ),
    (
        "performance",
        r"\bperformance\b|\bslow\b|\befficien|\bO\(n|\bcache\b|\ballocat|\bcopy\b.*\bavoid"
        r"|\bexpensive\b|\bloop\b.*\bevery\b|\bn\+1\b",
    ),
    (
        "security",
        r"\bsecurity\b|\binjection\b|\bsanitiz|\bescape\b|\bxss\b|\bcsrf\b|\bauth\b"
        r"|\bsecret\b|\bpassword\b|\btoken\b.*\blog|\bvalidate\b.*\binput\b|\bpermission\b",
    ),
    (
        "testing",
        r"\btests?\b|\bunit test\b|\bassert\b|\bmock\b|\bcoverage\b|\bfixture\b",
    ),
    (
        "style_format",
        r"\bstyle\b|\bformat(ting)?\b|\bindent|\bwhitespace\b|\bblank line\b|\bnewline\b"
        r"|\blint\b|\bspaces?\b|\btabs?\b|\bbrace\b|\bsemicolon\b|\bimport order\b|\bnit\b",
    ),
    (
        # Deliberately last of the real rules: a comment phrased as an open
        # question ("why is this public?", "shouldn't we check X?") only lands
        # here if no concrete rule matched. These are the hardest examples in
        # the set, because the required edit is implied rather than stated, and
        # they are worth isolating for exactly that reason.
        "design_question",
        r"^\s*(why|what|how|should(n't)?|shouldn|could|can|would|do|does|did|is|are|any reason)\b"
        r"|\?\s*$|\bi wonder\b|\bwondering\b|\bthoughts\?|\bwdyt\b",
    ),
]

_COMPILED: list[tuple[str, re.Pattern[str]]] = [
    (name, re.compile(pat, re.IGNORECASE)) for name, pat in _RULES
]

CATEGORIES: list[str] = [name for name, _ in _RULES] + ["other"]


def classify(comment: str) -> str:
    """Assign a review comment to exactly one feedback category."""
    for name, pattern in _COMPILED:
        if pattern.search(comment):
            return name
    return "other"


def classify_all(comment: str) -> list[str]:
    """Every category the comment matches — useful for auditing rule overlap."""
    hits = [name for name, pattern in _COMPILED if pattern.search(comment)]
    return hits or ["other"]
