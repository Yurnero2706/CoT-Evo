"""
Regression tests for the DiscourseMT answer parser.

The failure these guard against is specific and nasty: reading the *wrong*
option letter out of a prose answer. A refusal (None) only costs accuracy and
is visible in the run's parse-mode histogram as "unparsed". A misread letter
silently records the opposite verdict and reports itself as "declared", which
looks like a successful parse in the audit trail.

Both directions are therefore tested: the parser must find real commitments,
and it must refuse everything that merely mentions an option.
"""

import pytest

from src.evaluation.discourse_mt import (
    critical_span_verdict,
    extract_choice,
    unwrap_answer,
)


# (answer text, expected letter) - phrasings that do commit to a choice.
COMMITTED = [
    # The shapes a format-compliant model produces. "[ANSWER_START]B" with no
    # closing marker is the normal case: [ANSWER_END] is also a stop sequence,
    # so the API strips it.
    ("[ANSWER_START]B", "B"),
    ("[ANSWER_START]B[ANSWER_END]", "B"),
    ("B", "B"),
    ("b", "B"),
    ("A.", "A"),
    ("**B**", "B"),
    ("[ANSWER_START]A[ANSWER_END]\n\nNote: B was tempting.", "A"),
    # Prose, which is what a smaller or non-instruction-tuned model gives you.
    ("The answer is A.", "A"),
    ("Final answer: B", "B"),
    ("My choice is A", "A"),
    ("I choose A", "A"),
    ("I'll go with B.", "B"),
    ("Select B.", "B"),
    ("A is correct. B is not.", "A"),
    ("Between A and B, the correct translation is A.", "A"),
    ("The correct option is B, not A.", "B"),
    ("Neither A nor B is great, but I pick A.", "A"),
    ("Option A.", "A"),
    ("Looking at the idiom, B captures it.", "B"),
    # A commitment stated before a justification that names the other option.
    # Taking the last letter *mentioned* reads this as B; it is A.
    ("The answer is A because option B would require a plural subject.", "A"),
    # Everything else ruled out.
    ("Option B is wrong, so A.", "A"),
    # The model changing its mind: the last real commitment wins.
    ("The answer is B.\nWait, no - the answer is A.", "A"),
]

# Phrasings that commit to nothing. The parser must return None rather than
# guess, so the caller scores them wrong and the run reports them as unparsed.
UNCOMMITTED = [
    "",
    "   ",
    "Let me compare A and B.",
    "Both A and B are plausible.",
    "Neither is right.",
    "A is not correct.",
    "A seems right... actually B.",
    # "a" is the English indefinite article. A case-insensitive letter match
    # turns almost any sentence into a vote for option A.
    "The answer is a plural form.",
    "a cat is correct here",
]


@pytest.mark.parametrize("text,expected", COMMITTED)
def test_extract_choice_finds_real_commitments(text, expected):
    choice, how = extract_choice(text)
    assert choice == expected, f"{text!r} -> {choice!r} via {how!r}"
    assert how, "a successful parse must report how it was found"


@pytest.mark.parametrize("text", UNCOMMITTED)
def test_extract_choice_refuses_mere_mentions(text):
    choice, how = extract_choice(text)
    assert choice is None, f"{text!r} was read as {choice!r} via {how!r}"
    assert how == ""


def test_parse_mode_is_reported_for_auditing():
    assert extract_choice("B")[1] == "exact"
    assert extract_choice("The answer is B.")[1] == "declared"
    assert extract_choice("Option A is wrong, so B.")[1] == "eliminated"
    assert extract_choice("...\n\nB")[1] == "trailing"
    assert extract_choice("B captures the idiom.")[1] == "unique"


def test_unwrap_answer_strips_framework_markers():
    assert unwrap_answer("[ANSWER_START]B[ANSWER_END]") == "B"
    assert unwrap_answer("[ANSWER_START]B") == "B"
    assert unwrap_answer("<|think|>reasoning<|answer|>\nB") == "reasoning\nB"
    # Newlines survive: extract_choice's trailing-line heuristic needs them.
    assert "\n" in unwrap_answer("[ANSWER_START]line one\nline two[ANSWER_END]")


class TestCriticalSpan:
    meta = {"critical_correct": "she", "critical_incorrect": "you"}

    def test_correct_span_only(self):
        verdict, _ = critical_span_verdict("[ANSWER_START]She went home.", self.meta)
        assert verdict is True

    def test_wrong_span_only(self):
        verdict, _ = critical_span_verdict("You went home.", self.meta)
        assert verdict is False

    def test_hedging_with_both_is_not_a_success(self):
        verdict, _ = critical_span_verdict("She (or you) went home.", self.meta)
        assert verdict is False

    def test_neither_span(self):
        verdict, _ = critical_span_verdict("They went home.", self.meta)
        assert verdict is False

    def test_unjudgeable_item_is_flagged_not_guessed(self):
        verdict, reason = critical_span_verdict("anything", {})
        assert verdict is None
        assert "no critical span" in reason

    def test_word_boundaries(self):
        # "toe" must not match inside "toes".
        verdict, _ = critical_span_verdict(
            "He wiggled his toes.", {"critical_correct": "toe"}
        )
        assert verdict is False
