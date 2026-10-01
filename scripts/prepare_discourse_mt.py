#!/usr/bin/env python3
"""
Prepare the NTT Japanese-to-English discourse translation test set for CoT-Evo.

Source: https://github.com/nttcslab-nlp/discourse-mt-test-sets
Paper:  Nagata & Morishita (LREC 2020), "A Test Set for Discourse Translation
        from Japanese to English"

The upstream data is a *contrastive* test set, not a QA dataset. Each item is a
two-sentence Japanese passage plus two English translations that differ only in
the second sentence:

    {
      "type": "coreference",
      "examples": [{
        "src": ["あの彼女、今度うちの学校に来るらしいよ。", "けど、いつ来るか、わからない。"],
        "trg": {
          "correct":   ["That girl seems to come to our school soon.",
                        "But I don't know when she comes."],
          "incorrect": ["That girl seems to come to our school soon.",
                        "But I don't know when you come."]
        }
      }]
    }

Sentence 1 is always the context (identical in both translations); sentence 2
carries the discourse-sensitive choice. Three phenomena are covered:

    coreference  (250 items) - a pronoun / zero pronoun must pick up the
                               referent introduced in the context
    disamb       (378 items) - an ambiguous Japanese word must be disambiguated
                               by the context (2 examples per item: a minimal
                               pair with opposite contexts)
    repet        ( 73 items) - a word repeated in the source must be rendered
                               consistently with the context (2 examples per item)

CoT-Evo's fitness is driven by exact match against a single `answer` string
(see FitnessEvaluator / ExactMatchEvaluator in src/core/fitness.py), so this
script reframes each contrastive pair into an exact-matchable task. Three modes:

  contrastive (default)
      The two candidate second-sentence translations are shown as A / B in a
      seeded random order. `answer` is the letter "A" or "B", which gives a
      clean, unambiguous exact match.

      IMPORTANT: this is NOT the paper's protocol, and scores from it are not
      comparable to the numbers in the paper. Nagata & Morishita use forced
      decoding: each candidate is scored independently under the model and the
      test is whether log P(correct|source) > log P(incorrect|source), with the
      model never seeing the two candidates together. Putting both in the prompt
      turns generation into discrimination, which is a much easier task - an
      LLM only has to spot what differs between two sentences it can see. Expect
      high accuracy here, and do not read it as translation quality. Use it to
      harvest CoT traces with a checkable answer, not as a benchmark result.

  lexical
      The discourse-sensitive span is blanked out of the correct translation
      and the model must supply it. `answer` is the correct span (e.g. "she",
      "demons", "fingers"). Generative rather than multiple-choice, and still
      exact-matchable. Items whose span cannot be localised, or whose span runs
      longer than --max-answer-words, are skipped.

  translate
      The model translates sentence 2 in full. `answer` is the reference
      translation. NOTE: exact match will almost always be 0 here, so only use
      this mode with --lambda-length / a judge carrying the signal, or after
      swapping in a semantic-similarity evaluator.

Usage:
    # default: contrastive mode, all three phenomena
    python scripts/prepare_discourse_mt.py

    # lexical mode, coreference only, everything in train.json
    python scripts/prepare_discourse_mt.py --mode lexical --phenomena coreference --test-ratio 0

Outputs data/DiscourseMT/train.json and data/DiscourseMT/test.json in the
format expected by src/data/dataset_loader.py (id / query / answer / task /
subtask / knowledge / meta).
"""

import argparse
import difflib
import json
import logging
import random
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Windows consoles default to a legacy codepage, which cannot encode the
# Japanese source text. Files on disk are always written as UTF-8 regardless.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError, ValueError):
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

RAW_BASE = "https://raw.githubusercontent.com/nttcslab-nlp/discourse-mt-test-sets/main"

# Upstream filename -> phenomenon key used as the CoT-Evo `task` field.
SOURCE_FILES = {
    "coreference_2018_0327.json": "coreference",
    "disamb.json": "disamb",
    "repet.json": "repet",
}

# What makes the second sentence context-dependent, per phenomenon. Shown to the
# model so it knows which cue to reason about, without revealing the answer.
PHENOMENON_HINT = {
    "coreference": (
        "the sentence contains a pronoun or a dropped (zero) pronoun whose referent "
        "is only recoverable from the context sentence, so the English pronoun must "
        "agree with the entity introduced there"
    ),
    "disamb": (
        "the sentence contains a Japanese word that is ambiguous in isolation and can "
        "only be disambiguated by the context sentence"
    ),
    "repet": (
        "the sentence repeats a Japanese word that already appeared in the context "
        "sentence, so it must be translated consistently with how it was rendered there"
    ),
}

# Gold reference knowledge, written into the `knowledge` field. The engine passes
# this to KnowledgeJudgeEvaluator only (it is never shown to the thinker models),
# so it may safely name the correct rendering.
KNOWLEDGE_TEMPLATE = {
    "coreference": (
        'In this passage the discourse-sensitive expression must be rendered as "{correct}" '
        'and not "{incorrect}", because its referent is the entity introduced in the '
        "preceding context sentence rather than the addressee or a new entity. Pronoun "
        "choice in Japanese-to-English translation must be resolved from the antecedent "
        "in the preceding sentence, since Japanese routinely drops the subject."
    ),
    "disamb": (
        'In this passage the ambiguous Japanese expression must be rendered as "{correct}" '
        'and not "{incorrect}", because the preceding context sentence fixes which of its '
        "senses is intended. A word that is ambiguous sentence-internally must be "
        "disambiguated using the surrounding discourse, not by its most frequent sense."
    ),
    "repet": (
        'In this passage the repeated Japanese expression must be rendered as "{correct}" '
        'and not "{incorrect}", because the same word was already translated that way in '
        "the preceding context sentence. Lexical choice must stay consistent across "
        "sentences for a repeated source word, even when several synonyms are acceptable "
        "in isolation."
    ),
}

# Tokens keep their character offsets so the critical span can be cut out of the
# original string without re-joining tokens (which would mangle punctuation).
TOKEN_RE = re.compile(r"\w+|[^\w\s]")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_source_files(cache_dir: Path, download: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Load the three upstream JSON files, downloading them into cache_dir if needed.

    Returns:
        Mapping of phenomenon -> {item_id: item}
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Dict[str, Any]] = {}

    for filename, phenomenon in SOURCE_FILES.items():
        path = cache_dir / filename

        if not path.exists():
            if not download:
                raise FileNotFoundError(
                    f"{path} not found and --no-download was given. "
                    f"Fetch it manually from {RAW_BASE}/{filename}"
                )
            url = f"{RAW_BASE}/{filename}"
            logger.info(f"Downloading {url} -> {path}")
            urllib.request.urlretrieve(url, path)

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # Upstream wraps everything in a single-element list holding one dict
        # keyed by stringified item numbers.
        if isinstance(data, list):
            if len(data) != 1 or not isinstance(data[0], dict):
                raise ValueError(f"Unexpected top-level structure in {path}")
            items = data[0]
        elif isinstance(data, dict):
            items = data
        else:
            raise ValueError(f"Unexpected top-level structure in {path}")

        out[phenomenon] = items
        logger.info(f"Loaded {len(items)} {phenomenon} items from {path.name}")

    return out


# ---------------------------------------------------------------------------
# Critical span extraction
# ---------------------------------------------------------------------------

def critical_span(correct: str, incorrect: str) -> Optional[Tuple[int, int, str, str, int]]:
    """
    Locate the span where two translations of the same sentence disagree.

    Diffs are computed over word/punctuation tokens. When a pair disagrees in
    several places (~14% of items), the returned span is the union from the
    first to the last differing token, so it is always contiguous.

    Args:
        correct: The context-appropriate translation
        incorrect: The contrastive (context-violating) translation

    Returns:
        (start, end, correct_span, incorrect_span, n_diff_blocks) where start/end
        are character offsets into `correct`, or None if the correct side of the
        diff is empty (i.e. the contrast is a pure insertion).
    """
    a_tokens = list(TOKEN_RE.finditer(correct))
    b_tokens = list(TOKEN_RE.finditer(incorrect))
    a_words = [t.group(0) for t in a_tokens]
    b_words = [t.group(0) for t in b_tokens]

    blocks = [
        op for op in difflib.SequenceMatcher(a=a_words, b=b_words).get_opcodes()
        if op[0] != "equal"
    ]
    if not blocks:
        return None

    a_lo = min(op[1] for op in blocks)
    a_hi = max(op[2] for op in blocks)
    b_lo = min(op[3] for op in blocks)
    b_hi = max(op[4] for op in blocks)

    if a_lo >= a_hi:
        # Pure insertion: nothing in `correct` to blank out.
        return None

    start = a_tokens[a_lo].start()
    end = a_tokens[a_hi - 1].end()

    correct_span = correct[start:end]
    incorrect_span = (
        incorrect[b_tokens[b_lo].start():b_tokens[b_hi - 1].end()]
        if b_lo < b_hi else ""
    )

    return start, end, correct_span, incorrect_span, len(blocks)


# ---------------------------------------------------------------------------
# Query construction
# ---------------------------------------------------------------------------

def _preamble(
    src_ctx: str, ctx_translation: str, src_cur: str, ablate_context: bool = False
) -> str:
    """
    Build the shared header: the context sentence, then the sentence to translate.

    With ablate_context the context is withheld entirely, which is the control
    condition. The test set is designed so that a context-blind system scores
    50%, but that design goal is known to have failed for at least one category
    in the original paper (2020-era systems reached 0.76-0.78 on articles
    without needing context), so the blind score has to be measured rather than
    assumed. Whatever it comes out at is the real floor your headline accuracy
    should be read against.
    """
    if ablate_context:
        return f"""You are translating a Japanese sentence into English.

[Sentence to translate]
Japanese: {src_cur}"""

    return f"""You are translating a Japanese passage into English one sentence at a time. The previous sentence has already been translated and is fixed.

[Context sentence - already translated]
Japanese: {src_ctx}
English:  {ctx_translation}

[Sentence to translate]
Japanese: {src_cur}"""


def _contrast_instruction(phenomenon: str, ablate_context: bool) -> str:
    """Task wording for contrastive mode, with and without the context sentence."""
    if ablate_context:
        return (
            "Exactly one candidate is the correct translation of this sentence. "
            "The other is a fluent, plausible English sentence, but it is the wrong "
            "rendering here.\n\n"
            "Decide which candidate is correct from the Japanese source alone."
        )
    return (
        "Exactly one candidate is the correct translation in this context. The other "
        "is a fluent, plausible English sentence on its own, but it breaks the "
        f"discourse link with the context sentence: {PHENOMENON_HINT[phenomenon]}.\n\n"
        "Work out which candidate is correct by reasoning about the Japanese source "
        "and the context, not by which English sentence sounds better in isolation."
    )


def _lexical_instruction(phenomenon: str, ablate_context: bool) -> str:
    """Task wording for lexical mode, with and without the context sentence."""
    if ablate_context:
        return (
            "Supply the English expression that belongs in the blank marked ___.\n\n"
            "Work it out from the Japanese source alone."
        )
    return (
        "The blank marked ___ is the discourse-sensitive part of this sentence: "
        f"{PHENOMENON_HINT[phenomenon]}. Its correct filling cannot be determined "
        "from the sentence alone; you must use the context sentence above.\n\n"
        "Work out what belongs in the blank by reasoning about the Japanese source "
        "and the context."
    )


def _translate_instruction(phenomenon: str, ablate_context: bool) -> str:
    """Task wording for translate mode, with and without the context sentence."""
    if ablate_context:
        return (
            "Translate the sentence above into English.\n\n"
            "Work out the translation from the Japanese source alone."
        )
    return (
        "Translate the sentence above into English so that it is consistent with the "
        "context sentence. Pay particular attention to the discourse dependency: "
        f"{PHENOMENON_HINT[phenomenon]}.\n\n"
        "Work out the translation by reasoning about the Japanese source and the context."
    )


def build_contrastive_query(
    src_ctx: str,
    src_cur: str,
    ctx_translation: str,
    option_a: str,
    option_b: str,
    phenomenon: str,
    ablate_context: bool = False,
) -> str:
    """Build a two-choice discourse-consistency query. Answer is 'A' or 'B'."""
    return f"""{_preamble(src_ctx, ctx_translation, src_cur, ablate_context)}

[Candidate translations]
A. {option_a}
B. {option_b}

{_contrast_instruction(phenomenon, ablate_context)}

Your final answer must be exactly one letter: A or B."""


def build_lexical_query(
    src_ctx: str,
    src_cur: str,
    ctx_translation: str,
    masked_translation: str,
    phenomenon: str,
    ablate_context: bool = False,
) -> str:
    """Build a fill-the-blank query. Answer is the correct span itself."""
    return f"""{_preamble(src_ctx, ctx_translation, src_cur, ablate_context)}

[Partial English translation]
{masked_translation}

{_lexical_instruction(phenomenon, ablate_context)}

Your final answer must be only the English expression that fills the blank, with no surrounding words and no punctuation."""


def build_translate_query(
    src_ctx: str,
    src_cur: str,
    ctx_translation: str,
    phenomenon: str,
    ablate_context: bool = False,
) -> str:
    """Build a full-translation query. Answer is the reference translation."""
    return f"""{_preamble(src_ctx, ctx_translation, src_cur, ablate_context)}

{_translate_instruction(phenomenon, ablate_context)}

Your final answer must be only the English translation of the sentence to translate - a single sentence, nothing else."""


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def convert_example(
    phenomenon: str,
    item_id: str,
    ex_index: int,
    example: Dict[str, Any],
    mode: str,
    rng: random.Random,
    include_reference_knowledge: bool,
    max_answer_words: int = 0,
    ablate_context: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Convert one upstream example into a CoT-Evo sample.

    Returns None if the example is malformed or unusable in the chosen mode.
    """
    src = example.get("src") or []
    trg = example.get("trg") or {}
    correct = trg.get("correct") or []
    incorrect = trg.get("incorrect") or []

    if len(src) < 2 or len(correct) < 2 or len(incorrect) < 2:
        logger.warning(f"Skipping {phenomenon}-{item_id}-{ex_index}: expected 2 sentences per side")
        return None

    src_ctx, src_cur = src[0].strip(), src[1].strip()
    ctx_translation = correct[0].strip()
    correct_cur, incorrect_cur = correct[1].strip(), incorrect[1].strip()

    if correct_cur == incorrect_cur:
        logger.warning(f"Skipping {phenomenon}-{item_id}-{ex_index}: candidates are identical")
        return None

    span = critical_span(correct_cur, incorrect_cur)

    sample_id = f"discourse-{phenomenon}-{item_id}-{ex_index}"
    meta: Dict[str, Any] = {
        "phenomenon": phenomenon,
        "upstream_item": item_id,
        "upstream_example": ex_index,
        "mode": mode,
        "context_ablated": ablate_context,
        "src_context": src_ctx,
        "src_current": src_cur,
        "context_translation": ctx_translation,
        "correct_translation": correct_cur,
        "incorrect_translation": incorrect_cur,
    }

    if span is not None:
        _, _, correct_span, incorrect_span, n_blocks = span
        meta["critical_correct"] = correct_span
        meta["critical_incorrect"] = incorrect_span
        meta["n_diff_blocks"] = n_blocks

    # --- mode-specific query / answer -------------------------------------
    if mode == "contrastive":
        correct_first = rng.random() < 0.5
        option_a = correct_cur if correct_first else incorrect_cur
        option_b = incorrect_cur if correct_first else correct_cur
        answer = "A" if correct_first else "B"

        query = build_contrastive_query(
            src_ctx, src_cur, ctx_translation, option_a, option_b, phenomenon,
            ablate_context,
        )
        meta["option_a"] = option_a
        meta["option_b"] = option_b
        meta["correct_option_text"] = correct_cur

    elif mode == "lexical":
        if span is None:
            logger.debug(f"Skipping {sample_id}: critical span not localisable (pure insertion)")
            return None
        start, end, correct_span, _, _ = span
        # A pair that disagrees in several places yields a union span covering
        # the equal tokens in between, which is too long to be hit by exact
        # match. Drop those rather than seeding the population with items no
        # trajectory can score on.
        if max_answer_words and len(correct_span.split()) > max_answer_words:
            logger.debug(
                f"Skipping {sample_id}: critical span is {len(correct_span.split())} words "
                f"(> --max-answer-words {max_answer_words})"
            )
            return None
        query = build_lexical_query(
            src_ctx,
            src_cur,
            ctx_translation,
            correct_cur[:start] + "___" + correct_cur[end:],
            phenomenon,
            ablate_context,
        )
        answer = correct_span

    elif mode == "translate":
        query = build_translate_query(
            src_ctx, src_cur, ctx_translation, phenomenon, ablate_context
        )
        answer = correct_cur

    else:
        raise ValueError(f"Unknown mode: {mode}")

    knowledge = None
    if include_reference_knowledge and span is not None:
        knowledge = KNOWLEDGE_TEMPLATE[phenomenon].format(
            correct=span[2], incorrect=span[3]
        )

    return {
        "id": sample_id,
        "query": query,
        "answer": answer,
        "task": phenomenon,
        "subtask": mode,
        "knowledge": knowledge,
        "meta": meta,
    }


def convert_all(
    sources: Dict[str, Dict[str, Any]],
    phenomena: List[str],
    mode: str,
    seed: int,
    include_reference_knowledge: bool,
    max_answer_words: int = 0,
    ablate_context: bool = False,
) -> List[Dict[str, Any]]:
    """Convert every example of the selected phenomena into CoT-Evo samples."""
    rng = random.Random(seed)
    samples: List[Dict[str, Any]] = []
    skipped = 0

    for phenomenon in phenomena:
        items = sources[phenomenon]
        before = len(samples)

        # Sort by numeric item id so output order is stable across runs.
        for item_id in sorted(items, key=lambda k: (len(k), k)):
            item = items[item_id]
            examples = item.get("examples") or []
            if not examples:
                # coreference item 90 upstream has an empty examples list.
                logger.debug(f"{phenomenon}-{item_id}: no examples, skipping")
                continue

            for ex_index, example in enumerate(examples):
                sample = convert_example(
                    phenomenon, item_id, ex_index, example, mode, rng,
                    include_reference_knowledge, max_answer_words, ablate_context,
                )
                if sample is None:
                    skipped += 1
                else:
                    samples.append(sample)

        logger.info(f"  {phenomenon}: {len(samples) - before} samples")

    if skipped:
        logger.warning(f"Skipped {skipped} unusable examples in mode '{mode}'")

    return samples


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert the NTT discourse MT test set into CoT-Evo format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["contrastive", "lexical", "translate"],
        default="contrastive",
        help="Task framing (default: contrastive, the only one with reliable exact match)",
    )
    parser.add_argument(
        "--phenomena",
        nargs="+",
        choices=list(SOURCE_FILES.values()),
        default=list(SOURCE_FILES.values()),
        help="Which discourse phenomena to include (default: all)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/DiscourseMT"),
        help="Output directory (default: data/DiscourseMT)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/DiscourseMT/raw"),
        help="Where to keep the downloaded upstream JSON files",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Fraction held out into test.json (default: 0.1; use 0 to put everything in train)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for option shuffling and the split"
    )
    parser.add_argument(
        "--max-answer-words",
        type=int,
        default=4,
        help="lexical mode only: drop items whose critical span is longer than this "
             "many words, since exact match cannot realistically hit them "
             "(default: 4; 0 disables the filter)",
    )
    parser.add_argument(
        "--ablate-context",
        action="store_true",
        help="Control condition: build the same items with the context sentence "
             "withheld. Accuracy here is the context-blind floor your real score "
             "must be read against - the 50%% the test set was designed around is "
             "an assumption, not a measurement. Write it somewhere separate, e.g. "
             "--out-dir data/DiscourseMT_nocontext",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Fail instead of fetching missing upstream files",
    )
    parser.add_argument(
        "--no-reference-knowledge",
        action="store_true",
        help="Leave the `knowledge` field empty so the judge has no gold reference",
    )
    args = parser.parse_args()

    if not 0.0 <= args.test_ratio < 1.0:
        logger.error("--test-ratio must be in [0, 1)")
        return 1

    logger.info("=" * 70)
    logger.info("Preparing DiscourseMT (Nagata & Morishita, LREC 2020) for CoT-Evo")
    logger.info("=" * 70)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Phenomena: {', '.join(args.phenomena)}")

    sources = load_source_files(args.cache_dir, download=not args.no_download)

    samples = convert_all(
        sources,
        args.phenomena,
        args.mode,
        args.seed,
        include_reference_knowledge=not args.no_reference_knowledge,
        max_answer_words=args.max_answer_words if args.mode == "lexical" else 0,
        ablate_context=args.ablate_context,
    )
    if not samples:
        logger.error("No samples produced")
        return 1

    # Split by upstream item, never by example. disamb and repet ship two
    # examples per item that form a minimal pair: the same sentence to
    # translate, with opposite contexts and opposite correct answers. Splitting
    # those across train and test puts near-identical inputs on both sides,
    # which leaks and also breaks the independence any significance test over
    # the results would assume. Grouping keeps each pair whole.
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for sample in samples:
        key = (sample["task"], sample["meta"]["upstream_item"])
        groups.setdefault(key, []).append(sample)

    group_keys = sorted(groups)
    random.Random(args.seed).shuffle(group_keys)

    n_test_groups = int(round(len(group_keys) * args.test_ratio))
    test = [s for key in group_keys[:n_test_groups] for s in groups[key]]
    train = [s for key in group_keys[n_test_groups:] for s in groups[key]]

    logger.info(
        f"Split by upstream item: {len(group_keys)} items -> "
        f"{len(group_keys) - n_test_groups} train / {n_test_groups} test"
    )

    # Shuffle within each split so that --max-samples N takes a mix of
    # phenomena (dataset_loader slices data[:max_samples] before anything else).
    random.Random(args.seed).shuffle(train)
    random.Random(args.seed + 1).shuffle(test)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, split in (("train", train), ("test", test)):
        path = args.out_dir / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(split, f, ensure_ascii=False, indent=2)
        logger.info(f"Wrote {len(split)} samples -> {path}")

    if args.mode == "translate":
        logger.warning(
            "Mode 'translate' has free-form reference answers; ExactMatchEvaluator "
            "will score ~0 for almost every trajectory. Use --use-knowledge-judge "
            "or swap in a semantic-similarity evaluator."
        )

    logger.info("Done. Run with:")
    logger.info("  python run_evolution.py --dataset DiscourseMT --max-samples 10 --use-knowledge-judge")
    return 0


if __name__ == "__main__":
    sys.exit(main())
