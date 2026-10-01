"""
Scoring for the DiscourseMT free-translation task.

Exact match against a reference translation is useless for free generation:
there are many correct ways to translate a sentence, so string equality scores
~0 regardless of whether the model got the discourse dependency right.

What actually matters in this test set is one span. Each item is built from a
minimal pair whose two translations differ only in the discourse-sensitive
expression - "she" vs "you", "martin" vs "swallow", "fingers" vs "toes". So a
free translation can be judged on exactly that: does it contain the contextually
correct rendering, and not the contextually wrong one? That check is mechanical,
needs no LLM judge, and measures the phenomenon the test set was built for
rather than surface overlap with one reference string.

This matters because ranking two given candidates and producing a translation
unprompted are different computations. A model can reliably pick the right
candidate from a list and still drop or botch the same distinction when
translating freely, so a contrastive score is not evidence about generation.
Scoring generation needs its own instrument; this is it.

scripts/prepare_discourse_mt.py records the two spans as meta.critical_correct
and meta.critical_incorrect for every item where they could be localised.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Contrastive (A / B) scoring
# ---------------------------------------------------------------------------

def unwrap_answer(text: str) -> str:
    """
    Strip the framework's answer markers, leaving the answer content.

    A well-formed reply to these prompts puts the answer inside
    [ANSWER_START]...[ANSWER_END], and the closing marker doubles as a stop
    sequence, so what actually reaches a scorer is "[ANSWER_START]B". Any
    scorer that inspects the answer text has to peel that off first - reading
    it literally is how a perfectly correct "B" gets scored wrong.

    Newlines inside the answer are preserved, because the heuristics below use
    line structure. (clean_answer() in utils collapses whitespace, which is
    fine for equality testing but would destroy that signal.)

    Args:
        text: Raw answer field

    Returns:
        The answer content, or the original text if no markers are present
    """
    if not text:
        return ""

    out = text.replace("<|think|>", "").replace("<|answer|>", "").strip()

    marker = re.search(r"\[ANSWER_START\](.*?)(?:\[ANSWER_END\]|$)", out, re.DOTALL)
    if marker:
        inner = marker.group(1).strip()
        if inner:
            return inner
        # Empty markers: fall through to whatever else the reply contained.
        out = (out[:marker.start()] + " " + out[marker.end():]).strip()

    return out


# Phrasings that express an actual commitment to an option, with the letter
# *after* the phrase. The commitment word is mandatory: "option B" merely names
# a candidate, while "the answer is B" picks one. Dropping that requirement is
# how "The answer is A because option B would need a plural subject" gets read
# as B - a correct answer silently recorded as the wrong letter.
_COMMIT_BEFORE = [
    r"\b(?:final\s+)?(?:answer|choice|selection)\s*(?:is|:|=|would\s+be)\s*",
    r"\b(?:the\s+)?(?:correct|right|better|best|appropriate)\s+"
    r"(?:answer|option|choice|candidate|translation|rendering|one)\s*"
    r"(?:is|:|=|would\s+be)\s*",
    r"\b(?:i|we)\s*(?:'ll|\s+will|\s+would|\s+shall)?\s*"
    r"(?:choose|pick|select|go\s+with|prefer|answer)\s*:?\s*",
    r"\b(?:choose|select|pick|go\s+with)\s+",
    r"\bmy\s+(?:answer|choice)\s*(?:is|:|=)\s*",
]

# ...and with the letter *before* the phrase ("A is correct"). Only assertions
# count: "A seems right" is the model thinking aloud, not choosing, and reading
# it as a choice overrides whatever it concluded afterwards.
_COMMIT_AFTER = (
    r"\s+(?:is|would\s+be)\s+(?:the\s+)?"
    r"(?:correct|right|better|best|more\s+accurate|appropriate|preferable)\b"
)

# Phrasings that explicitly reject an option. Used only to narrow the field
# when nothing was committed to outright, never to pick a letter by itself.
_REJECT = (
    r"\s+(?:is|seems|would\s+be)\s+(?:the\s+)?"
    r"(?:wrong|incorrect|inaccurate|not\b|unnatural|implausible)"
)


def extract_choice(
    text: str, options: Sequence[str] = ("A", "B")
) -> Tuple[Optional[str], str]:
    """
    Pull the chosen option letter out of a model answer.

    Comparing the raw answer to "B" by string equality fails the moment a model
    writes "Therefore, the correct translation is B.\\n\\nB" instead of a bare
    letter. That is a formatting deviation, not a wrong answer, and scoring it
    as wrong understates accuracy on the task being measured.

    Leniency has a limit though, and it cuts both ways: refusing a clear answer
    understates accuracy, but reading the wrong letter out of prose *fabricates*
    a verdict, which is worse because it looks legitimate in the audit trail. So
    a letter is only accepted when the sentence actually commits to it, and the
    route is reported so a run can be audited:

        "exact"       the answer was already just the letter
        "declared"    an explicit commitment ("the answer is B", "A is correct")
        "eliminated"  every other option was explicitly rejected
        "trailing"    the answer ends on a standalone letter
        "unique"      exactly one option letter is mentioned, none are rejected
        ""            no confident reading; caller should score it wrong

    Naming an option is never enough on its own: in "B is wrong, so A" the only
    mention of B must not be read as a choice of B.

    Args:
        text: The model's answer field
        options: Valid option letters

    Returns:
        (letter or None, how it was found)
    """
    if not text or not text.strip():
        return None, ""

    # Peel off [ANSWER_START]...[ANSWER_END] first. Without this the normal,
    # fully compliant answer "[ANSWER_START]B" parses as nothing at all.
    cleaned = unwrap_answer(text)
    if not cleaned:
        return None, ""

    valid = {opt.upper() for opt in options}

    # 1. Already exactly the letter, optionally with trailing punctuation.
    bare = re.fullmatch(r"\s*([A-Za-z])\s*[.)]?\s*", cleaned)
    if bare and bare.group(1).upper() in valid:
        return bare.group(1).upper(), "exact"

    letters = "".join(sorted(valid))
    letter_group = r"[\"'`*(\[]*\s*([" + letters + r"])\b"

    # 2. An explicit commitment. Take the LAST one: models rehearse ("A would
    #    mean...") before committing, and may revise themselves.
    #
    #    The commitment *phrase* is matched case-insensitively, but the option
    #    letter never is. "A" is also the English indefinite article, so an
    #    ignore-case match reads "the answer is a plural form" as a vote for A.
    #    A model that writes its letter in lowercase is already handled by the
    #    bare-letter branch above.
    commitments: List[Tuple[int, str]] = []
    for prefix in _COMMIT_BEFORE:
        for m in re.finditer(r"(?i:" + prefix + r")" + letter_group, cleaned):
            commitments.append((m.start(), m.group(1)))
    for m in re.finditer(r"\b([" + letters + r"])(?i:" + _COMMIT_AFTER + r")", cleaned):
        commitments.append((m.start(), m.group(1)))

    if commitments:
        commitments.sort()
        return commitments[-1][1], "declared"

    # Same rule for mentions and rejections: the letter itself is case-sensitive.
    mentioned = {m.group(1) for m in re.finditer(r"\b([" + letters + r"])\b", cleaned)}
    rejected = {
        m.group(1)
        for m in re.finditer(r"\b([" + letters + r"])(?i:" + _REJECT + r")", cleaned)
    }
    rejected |= {
        m.group(1)
        for m in re.finditer(r"(?i:\bnot\s+)[\"'`*(\[]*([" + letters + r"])\b", cleaned)
    }

    # 3. Everything but one option was ruled out, and the survivor is named.
    #    ("Option B is wrong, so A.")
    survivors = (valid - rejected) & mentioned
    if rejected and len(survivors) == 1:
        return survivors.pop(), "eliminated"

    # 4. The answer ends on a standalone letter line, e.g. "...\n\nB".
    tail = re.search(
        r"(?:^|\n)\s*[\"'`*(\[]*\s*([" + letters + r"])\s*[.)\]\"'`*]*\s*$",
        cleaned,
    )
    if tail:
        return tail.group(1).upper(), "trailing"

    # 5. Only one option is named anywhere and nothing was rejected, so there is
    #    nothing for it to be confused with. ("B captures the idiom.")
    if len(mentioned) == 1 and not rejected:
        return mentioned.pop(), "unique"

    return None, ""


class ChoiceEvaluator:
    """
    Exact-match replacement for contrastive (A / B) items.

    Exposes the same ``await match(predicted, ground_truth)`` surface as
    ExactMatchEvaluator, so it can be passed to
    FitnessEvaluator.set_exact_match_evaluator() for an evolution run.

    ``last_parse`` records how the last answer was read, so a run can report
    how much it leaned on lenient parsing rather than hiding it.
    """

    def __init__(self, options: Sequence[str] = ("A", "B")):
        self.options = options
        self.last_parse: str = ""

    async def match(self, predicted: str, ground_truth: str) -> bool:
        """Compare the chosen option letter against the ground-truth letter."""
        choice, how = extract_choice(predicted or "", self.options)
        self.last_parse = how or "unparsed"

        if choice is None:
            logger.debug(f"No option letter found in answer: {(predicted or '')[:120]!r}")
            return False

        expected, _ = extract_choice(ground_truth or "", self.options)
        if expected is None:
            expected = (ground_truth or "").strip().upper()

        return choice == expected


def _contains_span(haystack: str, span: str) -> bool:
    """
    Case-insensitive whole-word search for a span inside a translation.

    Word boundaries stop "toe" from matching inside "toes" and, more
    importantly, stop a short pronoun like "I" from matching inside unrelated
    words. \\b is only meaningful next to word characters, so spans that begin
    or end with punctuation fall back to a plain substring test on that side.
    """
    span = span.strip()
    if not span:
        return False

    left = r"\b" if span[0].isalnum() or span[0] == "_" else ""
    right = r"\b" if span[-1].isalnum() or span[-1] == "_" else ""
    pattern = left + re.escape(span) + right

    return re.search(pattern, haystack, re.IGNORECASE) is not None


def critical_span_verdict(
    hypothesis: str, meta: Dict[str, Any]
) -> Tuple[Optional[bool], str]:
    """
    Judge a free translation on the discourse-sensitive span alone.

    Args:
        hypothesis: The model's translation of the sentence
        meta: Sample meta from prepare_discourse_mt.py, carrying
            critical_correct and critical_incorrect

    Returns:
        (verdict, reason) where verdict is
            True  - the correct rendering is present and the wrong one is not
            False - the wrong rendering is present, both are, or neither is
            None  - not judgeable, because the item has no recorded spans
    """
    correct = (meta or {}).get("critical_correct")
    incorrect = (meta or {}).get("critical_incorrect")

    if not correct:
        return None, "no critical span recorded for this item"

    # Same marker peeling as extract_choice: searching inside a literal
    # "[ANSWER_START]" could also match a span by accident.
    hypothesis = unwrap_answer(hypothesis or "")

    if not hypothesis.strip():
        return False, "empty translation"

    has_correct = _contains_span(hypothesis, correct)
    has_incorrect = bool(incorrect) and _contains_span(hypothesis, incorrect)

    if has_correct and not has_incorrect:
        return True, f"contains {correct!r}, not {incorrect!r}"
    if has_incorrect and not has_correct:
        return False, f"contains the context-violating {incorrect!r}"
    if has_correct and has_incorrect:
        # Both renderings present: the model hedged, or produced a gloss. Not a
        # clean success, so it does not count as one.
        return False, f"contains both {correct!r} and {incorrect!r}"

    return False, f"contains neither {correct!r} nor {incorrect!r}"


class CriticalSpanEvaluator:
    """
    Per-sample evaluator wrapping critical_span_verdict.

    Constructed with one sample's meta so it can expose the same
    ``await match(predicted, ground_truth)`` surface as ExactMatchEvaluator,
    and therefore be dropped into FitnessEvaluator.set_exact_match_evaluator()
    for a free-translation evolution run.

    Items with no recorded span are scored False rather than skipped: treating
    unjudgeable items as successes would inflate the score, and silently
    dropping them would change the denominator without saying so. Filter them
    out of the dataset up front if you want them excluded.
    """

    def __init__(self, meta: Dict[str, Any]):
        self.meta = meta or {}
        self.last_reason: str = ""

    async def match(self, predicted: str, ground_truth: str) -> bool:
        """Score a translation on its discourse-sensitive span."""
        verdict, reason = critical_span_verdict(predicted or "", self.meta)
        self.last_reason = reason

        if verdict is None:
            logger.debug(f"Unjudgeable DiscourseMT item: {reason}")
            return False

        return verdict
