#!/usr/bin/env python3
"""
Single-thinker CoT generation - no evolution.

A preview tool: it runs *one* teacher model over a dataset and dumps the raw
chains of thought, so you can read what the teachers actually produce before
paying for a full evolution run (selection, crossover, mutation, judging).

It deliberately reuses MultiThinkerGenerator from the real pipeline, so the
system prompt, the user prompt, the stop sequences and the answer extraction are
byte-for-byte what `run_evolution.py` would send. The only difference is that the
model registry is replaced by a shim that always returns the single model you
name, instead of sampling randomly from config/models.yaml's thinkers.

What it does NOT do: no NSLC selection, no crossover, no mutation, no LLM judge.
Fitness is reported as exact match + the length score only, purely as a readout.

Usage:
    # see the exact prompts without spending a single token
    python scripts/generate_cot_single.py --dataset DiscourseMT --max-samples 2 --dry-run

    # generate one CoT per sample with DeepSeek, reasoning turned on, and print
    # the prompt sent and the answer returned side by side
    python scripts/generate_cot_single.py --dataset DiscourseMT --model deepseek-flash \
        --thinking --max-samples 5 --show

    # three CoTs per sample, to see how much the model varies
    python scripts/generate_cot_single.py --dataset DiscourseMT --model deepseek-flash \
        --thinking --max-samples 5 --n-vanilla 3

    # offline, against a local vLLM server (see scripts/serve_qwen3_local.sh)
    python scripts/generate_cot_single.py --dataset DiscourseMT \
        --models config/models.vllm.yaml --model qwen3-8b-local \
        --thinking --temperature 0.6 --top-p 0.95 --top-k 20 \
        --max-samples -1 --concurrency 32

`--model` takes a `name` from the `thinkers` list of whichever file `--models`
points at - config/models.yaml by default (deepseek-flash, deepseek-v4-pro,
qwen-max, gemini-pro as shipped), or config/models.vllm.yaml for a local
server. Fill in the API_KEY placeholders there first; a local vLLM endpoint
needs no real key, just a non-empty one.

Reasoning mode is requested differently by each model family, and the
difference is not symmetric:

  DeepSeek  the old always-reasoning `deepseek-r1` is retired, and on the
            current models reasoning is per-request and OFF by default. Pass
            --thinking to turn it on.
  Qwen3     reasoning is ON by default in the chat template. Not passing
            --thinking is not enough to get a non-reasoning run; the script
            sends enable_thinking=false explicitly for you.

--thinking-style picks the dialect and defaults to inferring it from the model
name. Either way the trace comes back in a separate `reasoning_content` field,
which this script folds into the <|think|> block for you.

Writes to outputs/{dataset}_single/{model}_{timestamp}/:
    cots.md        - human-readable CoTs, this is the thing to actually read
    sample_NN.json - one file per sample, full text plus scores
    summary.json   - per-sample scores and run metadata
"""

import argparse
import asyncio
import collections
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Add the project root and src to path (mirrors run_evolution.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.core.fitness import ExactMatchEvaluator, LengthEvaluator
from src.data.dataset_loader import DatasetLoader
from src.evaluation.discourse_mt import ChoiceEvaluator, CriticalSpanEvaluator
from src.initialization.generators import MultiThinkerGenerator
from src.initialization.prompts import (
    DISCOURSE_KNOWLEDGE_GENERATION_PROMPT,
    DISCOURSE_MT_DIRECT_SYSTEM,
    DISCOURSE_MT_DIRECT_TEMPLATE,
)
from src.knowledge.generation import KnowledgeGenerator
from src.knowledge.hybrid import HybridKnowledgeAugmenter
from src.models.base import GenerationConfig, LLMProvider
from src.models.registry import ModelRegistry
from src.utils.answer_extractor import extract_cot_and_answer


def _force_utf8_console() -> None:
    """
    Print Japanese without crashing.

    Windows consoles default to a legacy codepage (cp1252 here), so writing the
    Japanese source text raises UnicodeEncodeError. `errors="replace"` keeps a
    terminal that genuinely cannot render the glyphs from killing the run - the
    files on disk are always full UTF-8 regardless.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass


_force_utf8_console()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Datasets whose knowledge augmentation needs the linguistic prompt rather than
# the default scientific one (kept in sync with run_evolution.py).
LANGUAGE_DATASETS = {"DiscourseMT", "DiscourseMT_NoContext"}


# ---------------------------------------------------------------------------
# Model plumbing
# ---------------------------------------------------------------------------

THINKING_STYLES = ("auto", "deepseek", "qwen3", "none")


def infer_thinking_style(api_model_name: str) -> str:
    """
    Guess which reasoning-mode dialect an endpoint speaks, from its model name.

    There is no standard for this. Each family invented its own field, and
    sending the wrong one is silent: the endpoint ignores an unknown key and
    answers in whatever mode it defaults to, so a run that looks like a
    thinking run may not be one.

    Args:
        api_model_name: The model name sent to the API (or the config alias)

    Returns:
        One of "deepseek", "qwen3", "none"
    """
    name = (api_model_name or "").lower()
    if "qwen" in name:
        return "qwen3"
    if "deepseek" in name:
        return "deepseek"
    return "none"


def build_thinking_params(style: str, enabled: bool, effort: str) -> Dict[str, Any]:
    """
    Request parameters that put a reasoning endpoint into the mode we want.

    Two dialects, and they differ in a way that matters for the baseline arm:

    DeepSeek retired the standalone `deepseek-r1` model; on `deepseek-flash` /
    `deepseek-v4-pro` reasoning is a per-request mode that is **off** unless
    asked for, via the vendor `thinking` field plus `reasoning_effort`.

    Qwen3 is the opposite. Its chat template enables thinking **by default**,
    and it is turned off by passing `chat_template_kwargs.enable_thinking =
    false`. So "do not pass --thinking" does not mean "no reasoning" on Qwen3 -
    it means the model reasons anyway and the --no-cot arm silently stops being
    a no-reasoning condition. The off state therefore has to be sent
    explicitly, which is why this returns params even when disabled.

    Everything goes inside `extra_body` rather than as named kwargs.
    `reasoning_effort` only became a typed parameter in newer openai SDKs -
    requirements.txt allows >=1.12.0, and passing it by name raises TypeError
    on older ones. `extra_body` is merged into the raw JSON body by every
    version, so the endpoint sees identical fields either way.

    Args:
        style: Dialect to speak ("deepseek", "qwen3" or "none")
        enabled: Whether thinking mode was requested
        effort: Reasoning effort level to ask for (DeepSeek only)

    Returns:
        An `extra_body` dict to merge into every generation call

    Raises:
        ValueError: If style is not a known dialect
    """
    if style == "deepseek":
        if not enabled:
            return {}
        return {"thinking": {"type": "enabled"}, "reasoning_effort": effort}

    if style == "qwen3":
        # Sent in both directions on purpose - see the docstring. Qwen3 has no
        # reasoning_effort knob; depth is controlled by the token budget.
        return {"chat_template_kwargs": {"enable_thinking": bool(enabled)}}

    if style == "none":
        return {}

    raise ValueError(f"Unknown thinking style: {style!r}")


def build_extra_params(
    style: str, thinking: bool, effort: str, top_k: Optional[int]
) -> Dict[str, Any]:
    """
    Assemble the single `extra_body` payload for a run.

    Everything non-standard has to travel in one dict: a second `extra_body`
    key would overwrite the first rather than merge with it.

    Args:
        style: Thinking dialect, already resolved (never "auto")
        thinking: Whether thinking mode was requested
        effort: Reasoning effort (DeepSeek only)
        top_k: Top-k cutoff, or None to leave it to the server

    Returns:
        Kwargs to merge into every generation call (empty when there is nothing
        to send)
    """
    body = build_thinking_params(style, thinking, effort)

    if top_k is not None:
        # Not an OpenAI-API field, so it only reaches the sampler via extra_body.
        body["top_k"] = top_k

    return {"extra_body": body} if body else {}


def resolve_thinking_style(requested: str, model_name: str, source: str) -> str:
    """Resolve --thinking-style, inferring from the model name when 'auto'."""
    if requested != "auto":
        return requested

    style = infer_thinking_style(model_name)
    if style == "none":
        logger.warning(
            f"Could not infer a thinking dialect from {source} {model_name!r}. "
            f"No reasoning-mode parameters will be sent, so the endpoint will "
            f"answer in whatever mode it defaults to. Pass --thinking-style "
            f"explicitly if that is not what you want."
        )
    return style


class TunedProvider(LLMProvider):
    """
    Wraps a provider to inject per-run generation defaults.

    ModelRegistry drops the `temperature` and `max_tokens` fields from
    models.yaml when it builds a provider, so every call falls back to
    GenerationConfig's max_tokens=2048. That silently truncates long reasoning
    traces, which is exactly what we are here to inspect. This wrapper supplies
    the defaults - and any thinking-mode parameters - that
    MultiThinkerGenerator has no way to pass through itself.
    """

    def __init__(
        self,
        inner: LLMProvider,
        temperature: float,
        max_tokens: int,
        extra_params: Optional[Dict[str, Any]] = None,
        top_p: float = 1.0,
    ):
        super().__init__(inner.model_name, inner.base_url, inner.api_key)
        self._inner = inner
        self._config = GenerationConfig(
            temperature=temperature, max_tokens=max_tokens, top_p=top_p
        )
        self._extra_params = dict(extra_params or {})

    def _with_extras(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Add the run's extra params without overriding an explicit caller value."""
        for key, value in self._extra_params.items():
            kwargs.setdefault(key, value)
        return kwargs

    def generate(self, prompt: str, config: Optional[GenerationConfig] = None, **kwargs) -> str:
        return self._inner.generate(prompt, config or self._config, **self._with_extras(kwargs))

    async def generate_async(
        self, prompt: str, config: Optional[GenerationConfig] = None, **kwargs
    ) -> str:
        return await self._inner.generate_async(
            prompt, config or self._config, **self._with_extras(kwargs)
        )

    @property
    def provider_type(self) -> str:
        return self._inner.provider_type


class PromptRecorder(LLMProvider):
    """
    Wraps a provider to remember the exact messages sent alongside each reply.

    MultiThinkerGenerator hands back Trajectory objects carrying the dataset
    query, not the templated prompt that was actually sent, so --show would
    otherwise have no prompt to display. One recorder is created per sample, so
    its log never interleaves with another sample's concurrent calls.
    """

    def __init__(self, inner: LLMProvider):
        super().__init__(inner.model_name, inner.base_url, inner.api_key)
        self._inner = inner
        self.calls: List[Dict[str, Any]] = []
        self.errors: List[str] = []

    def generate(self, prompt: str, config: Optional[GenerationConfig] = None, **kwargs) -> str:
        return self._inner.generate(prompt, config, **kwargs)

    async def generate_async(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> str:
        try:
            output = await self._inner.generate_async(prompt, config, messages=messages, **kwargs)
        except Exception as exc:
            # generate_initial_pool gathers with return_exceptions=True and only
            # logs a warning, so a failed generation would otherwise vanish and
            # leave the sample with an empty trajectory list and no explanation.
            self.errors.append(f"{type(exc).__name__}: {exc}")
            raise

        if messages is not None:
            # messages is None only on the knowledge-extraction call, which does
            # not correspond to any trajectory.
            self.calls.append({"messages": messages, "output": output})
        return output

    @property
    def provider_type(self) -> str:
        return self._inner.provider_type


def take_prompt_for(
    calls: List[Dict[str, Any]], reasoning: str, answer: str
) -> Optional[List[Dict[str, str]]]:
    """
    Find the recorded call that produced a given trajectory, and consume it.

    Trajectories come back from asyncio.gather in task order while calls are
    logged in completion order, so the two lists cannot simply be zipped.
    Instead each recorded output is re-split the same way the generator split
    it, and matched on the result. Matches are popped, so trajectories that
    happen to be identical still map one-to-one.

    Args:
        calls: Recorded (messages, output) pairs, mutated in place
        reasoning: Trajectory reasoning to match
        answer: Trajectory answer to match

    Returns:
        The messages that produced this trajectory, or None if unmatched
    """
    for i, call in enumerate(calls):
        call_reasoning, call_answer = extract_cot_and_answer(call["output"])
        if call_reasoning == reasoning and call_answer == answer:
            return calls.pop(i)["messages"]
    return None


class RecordingProvider(LLMProvider):
    """A stand-in that records prompts and returns a canned reply (--dry-run)."""

    def __init__(self, model_name: str = "dry-run"):
        super().__init__(model_name)
        self.calls: List[Dict[str, Any]] = []

    def generate(self, prompt: str, config: Optional[GenerationConfig] = None, **kwargs) -> str:
        raise NotImplementedError

    async def generate_async(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> str:
        self.calls.append({"prompt": prompt, "messages": messages, "kwargs": dict(kwargs)})
        if messages is None:
            # Knowledge generation path.
            return "[KNOWLEDGE_START]\n(dry run: no knowledge generated)\n[KNOWLEDGE_END]"
        return "<|think|>\n(dry run: no reasoning generated)\n<|answer|>\n[ANSWER_START]?[ANSWER_END]"

    @property
    def provider_type(self) -> str:
        return "dry-run"


class SingleThinkerRegistry:
    """
    Registry shim exposing exactly one thinker.

    MultiThinkerGenerator only ever asks for `get_random_thinker()`, so pinning
    that to one model turns the multi-thinker initialisation into a
    single-thinker one without touching the generator itself.
    """

    def __init__(self, model: LLMProvider):
        self._model = model
        self.thinkers = [model.model_name]

    def get_random_thinker(self) -> LLMProvider:
        return self._model

    def get_thinker_models(self) -> List[LLMProvider]:
        return [self._model]

    def get_judge_model(self) -> Optional[LLMProvider]:
        return None

    def get_knowledge_generator_model(self) -> LLMProvider:
        return self._model


def lookup_model(registry: ModelRegistry, requested: str) -> LLMProvider:
    """
    Look up a thinker by its models.yaml `name` and validate its credentials.

    Returns the raw provider rather than a wrapped one, because the thinking
    dialect is inferred from the resolved API model name - which is only known
    after the lookup, and may differ from the config alias.
    """
    inner = registry.get_model(requested)

    if inner is None:
        # Fall back to matching on the API-level model_name, since `name` and
        # `model_name` are allowed to differ in models.yaml.
        for name in registry.thinkers:
            candidate = registry.get_model(name)
            if candidate is not None and candidate.model_name == requested:
                inner = candidate
                break

    if inner is None:
        available = ", ".join(registry.list_models()) or "(none)"
        raise SystemExit(
            f"Model '{requested}' not found in the registry.\n"
            f"Available: {available}\n"
            f"Use one of the `name` values under `models.thinkers` in your models.yaml."
        )

    if inner.base_url in (None, "", "BASE_URL") or inner.api_key in (None, "", "API_KEY"):
        raise SystemExit(
            f"Model '{requested}' still has placeholder credentials "
            f"(base_url={inner.base_url!r}, api_key={'set' if inner.api_key else 'unset'}).\n"
            f"Fill in base_url and api_key in config/models.yaml, or use --dry-run "
            f"to inspect the prompts without calling an API."
        )

    # Reasoning endpoints return the trace in `reasoning_content` rather than in
    # `content`; ask the provider to fold it back into the <|think|> block.
    # This is opt-in per instance, so run_evolution.py is unaffected.
    inner.capture_reasoning = True

    return inner


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def build_generator(
    model: LLMProvider, dataset_name: str, needs_knowledge: bool
) -> MultiThinkerGenerator:
    """Build the real MultiThinkerGenerator, pinned to a single thinker."""
    augmenter = HybridKnowledgeAugmenter(generator=None)

    if needs_knowledge:
        # Single-model run: the same thinker also extracts the knowledge.
        prompt = (
            DISCOURSE_KNOWLEDGE_GENERATION_PROMPT
            if dataset_name in LANGUAGE_DATASETS
            else None
        )
        augmenter = HybridKnowledgeAugmenter(
            generator=KnowledgeGenerator(model=model, prompt_template=prompt)
        )

    return MultiThinkerGenerator(
        model_registry=SingleThinkerRegistry(model),
        knowledge_augmenter=augmenter,
        dataset_name=dataset_name,
    )


async def run_sample(
    idx: int,
    sample: Dict[str, Any],
    make_generator: Callable[[], Tuple[PromptRecorder, MultiThinkerGenerator]],
    exact_match: ExactMatchEvaluator,
    length: LengthEvaluator,
    n_vanilla: int,
    n_knowledge: int,
    no_cot: bool = False,
) -> Dict[str, Any]:
    """Generate CoTs for one sample and score them."""
    sample_id = sample.get("id", f"sample_{idx}")
    ground_truth = sample["answer"]

    # A generator per sample, so each gets its own prompt log rather than one
    # shared log interleaved by concurrent samples.
    recorder, generator = make_generator()

    # Free translation cannot be scored by string equality against one
    # reference; score the discourse-sensitive span instead. See
    # src/evaluation/discourse_mt.py.
    scorer = exact_match
    if sample.get("subtask") == "translate":
        scorer = CriticalSpanEvaluator(sample.get("meta") or {})
    elif sample.get("subtask") == "contrastive":
        # A / B items: read the chosen letter rather than demanding the answer
        # field be nothing but that letter. See src/evaluation/discourse_mt.py.
        scorer = ChoiceEvaluator()

    population = await generator.generate_initial_pool(
        query=sample["query"],
        ground_truth=ground_truth,
        n_vanilla=n_vanilla,
        n_knowledge_augmented=n_knowledge,
        # MultiThinkerGenerator already accepts a template override, so the
        # no-CoT arm needs no change to the generator itself.
        prompt_template=DISCOURSE_MT_DIRECT_TEMPLATE if no_cot else None,
    )

    trajectories = []
    for traj in population.trajectories:
        is_match = await scorer.match(traj.answer, ground_truth)
        messages = take_prompt_for(recorder.calls, traj.reasoning or "", traj.answer or "")
        by_role = {m["role"]: m["content"] for m in (messages or [])}
        trajectories.append(
            {
                "generation_method": traj.generation_method,
                "source_model": traj.source_model,
                "system_prompt": by_role.get("system"),
                "prompt": by_role.get("user"),
                "reasoning": traj.reasoning,
                "reasoning_words": len(traj.reasoning.split()) if traj.reasoning else 0,
                "answer": traj.answer,
                "exact_match": is_match,
                "answer_parse": getattr(scorer, "last_parse", "exact"),
                "length_score": length.score(traj.reasoning or ""),
                "knowledge": traj.knowledge,
            }
        )

    n_correct = sum(1 for t in trajectories if t["exact_match"])
    n_requested = n_vanilla + n_knowledge

    for message in recorder.errors:
        logger.error(f"[{idx}] {sample_id}: generation failed: {message}")
    if not recorder.errors and len(trajectories) < n_requested:
        logger.warning(
            f"[{idx}] {sample_id}: only {len(trajectories)}/{n_requested} trajectories "
            f"came back, with no error recorded"
        )

    logger.info(
        f"[{idx}] {sample_id}: {n_correct}/{len(trajectories)} correct, "
        f"mean {int(sum(t['reasoning_words'] for t in trajectories) / max(len(trajectories), 1))} words"
    )

    return {
        "sample_id": sample_id,
        "task": sample.get("task", ""),
        "subtask": sample.get("subtask", ""),
        "query": sample["query"],
        "ground_truth": ground_truth,
        "reference_knowledge": sample.get("knowledge"),
        "trajectories": trajectories,
        "n_correct": n_correct,
        "n_trajectories": len(trajectories),
        "n_requested": n_requested,
        "generation_errors": recorder.errors,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_markdown(path: Path, results: List[Dict[str, Any]], meta: Dict[str, Any]) -> None:
    """Write the human-readable CoT dump."""
    lines = [
        f"# Single-thinker CoTs - {meta['dataset']} / {meta['model']}",
        "",
        f"- Generated: {meta['timestamp']}",
        f"- Samples: {len(results)}",
        f"- Trajectories per sample: {meta['n_vanilla']} vanilla + {meta['n_knowledge']} knowledge-augmented",
        f"- temperature={meta['temperature']}, max_tokens={meta['max_tokens']}",
        f"- thinking mode: {'on, effort=' + str(meta['reasoning_effort']) if meta['thinking'] else 'off'}",
        f"- CoT scaffold: {'OFF (direct-answer baseline)' if meta.get('no_cot') else 'on'}",
        "",
    ]

    scored = [r for r in results if r["n_trajectories"]]
    if scored:
        total = sum(r["n_trajectories"] for r in scored)
        correct = sum(r["n_correct"] for r in scored)
        lines += [f"- Exact match: {correct}/{total} trajectories ({correct / total:.1%})", ""]

    for i, r in enumerate(results, 1):
        lines += [
            "---",
            "",
            f"## {i}. `{r['sample_id']}` ({r['task']})",
            "",
            f"**Ground truth:** `{r['ground_truth']}`",
            "",
            "<details><summary>Query</summary>",
            "",
            "```",
            r["query"],
            "```",
            "",
            "</details>",
            "",
        ]

        if r.get("reference_knowledge"):
            lines += [f"**Reference knowledge:** {r['reference_knowledge']}", ""]

        if not r["trajectories"]:
            reason = r.get("error") or "; ".join(r.get("generation_errors") or []) or "unknown"
            lines += [f"**No trajectories generated.** Reason: {reason}", ""]
            continue

        if r.get("generation_errors"):
            lines += [
                f"**{len(r['generation_errors'])} of {r.get('n_requested')} generations failed:** "
                + "; ".join(r["generation_errors"]),
                "",
            ]

        for j, t in enumerate(r["trajectories"], 1):
            mark = "correct" if t["exact_match"] else "WRONG"
            lines += [
                f"### CoT {j} - {t['generation_method']} - {mark}",
                "",
                f"*{t['reasoning_words']} words, length_score={t['length_score']}*",
                "",
            ]
            if t.get("knowledge"):
                lines += [f"> Knowledge given to the model: {t['knowledge']}", ""]
            if t.get("prompt"):
                lines += [
                    "<details><summary>Exact prompt sent</summary>",
                    "",
                    "```",
                    t["prompt"],
                    "```",
                    "",
                    "</details>",
                    "",
                ]
            lines += [
                t["reasoning"] or "*(empty reasoning)*",
                "",
                f"**Answer:** `{t['answer']}`",
                "",
            ]

    path.write_text("\n".join(lines), encoding="utf-8")


def print_summary(results: List[Dict[str, Any]]) -> None:
    """Print a compact per-task readout."""
    by_task: Dict[str, List[int]] = {}
    for r in results:
        by_task.setdefault(r["task"] or "(none)", []).append((r["n_correct"], r["n_trajectories"]))

    logger.info("=" * 70)

    # Surface incomplete generation before the accuracy numbers: a rate computed
    # over trajectories that never came back is not the rate it looks like.
    empty = [r for r in results if not r["n_trajectories"]]
    short = [
        r for r in results
        if r["n_trajectories"] and r.get("n_requested")
        and r["n_trajectories"] < r["n_requested"]
    ]
    if empty or short:
        logger.warning(
            f"INCOMPLETE: {len(empty)} sample(s) produced no trajectory at all, "
            f"{len(short)} produced fewer than requested"
        )
        for r in (empty + short):
            reason = r.get("error") or "; ".join(r.get("generation_errors") or []) or "no error recorded"
            logger.warning(f"  {r['sample_id']}: {r['n_trajectories']}/{r.get('n_requested')} - {reason}")

    logger.info("Exact match by task:")
    for task, pairs in sorted(by_task.items()):
        correct = sum(c for c, _ in pairs)
        total = sum(n for _, n in pairs)
        rate = f"{correct / total:.1%}" if total else "n/a"
        logger.info(f"  {task:<14} {correct:>4}/{total:<4} {rate}")

    parses = collections.Counter(
        t.get("answer_parse") or "exact" for r in results for t in r["trajectories"]
    )
    lenient = sum(n for k, n in parses.items() if k not in ("exact", ""))
    if lenient:
        logger.info(
            f"Answer parsing: {dict(parses)} - {lenient} answer(s) needed lenient "
            f"parsing (the model did not emit a bare answer)"
        )

    all_words = [
        t["reasoning_words"] for r in results for t in r["trajectories"] if t["reasoning_words"]
    ]
    if all_words:
        all_words.sort()
        n = len(all_words)
        logger.info(
            f"Reasoning length (words): min={all_words[0]} "
            f"p15={all_words[int(0.15 * (n - 1))]} median={all_words[n // 2]} "
            f"p85={all_words[int(0.85 * (n - 1))]} max={all_words[-1]}"
        )
        logger.info(
            "  ^ use the p15/p85 values as `length_percentiles` for this dataset "
            "in config/datasets.yaml"
        )
    logger.info("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CoTs with a single teacher model (no evolution)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", default="DiscourseMT",
                        help="Dataset name from config/datasets.yaml (default: DiscourseMT)")
    parser.add_argument("--model", default="deepseek-flash",
                        help="Thinker `name` from config/models.yaml (default: deepseek-flash)")
    parser.add_argument("--max-samples", type=int, default=5,
                        help="Samples to generate for, -1 for all (default: 5)")
    parser.add_argument("--split", default="train", choices=["train", "test"],
                        help="Dataset split (default: train)")
    parser.add_argument("--n-vanilla", type=int, default=1,
                        help="Vanilla CoTs per sample (default: 1)")
    parser.add_argument("--n-knowledge", type=int, default=0,
                        help="Knowledge-augmented CoTs per sample; the same model "
                             "extracts the knowledge (default: 0)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7)")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="Nucleus sampling cutoff (default: 1.0). Qwen3 "
                             "recommends 0.95 in thinking mode, 0.8 without it.")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Top-k cutoff, sent via extra_body since it is not "
                             "an OpenAI-API field. Most local servers (vLLM, SGLang) "
                             "accept it; hosted OpenAI endpoints reject it. "
                             "Qwen3 recommends 20 (default: not sent)")
    parser.add_argument("--max-tokens", type=int, default=16384,
                        help="Max tokens per generation; the registry otherwise "
                             "defaults to 2048 and truncates long CoTs (default: 16384)")
    parser.add_argument("--no-cot", action="store_true",
                        help="Baseline arm: ask for the answer directly, with no reasoning "
                             "scaffold and no <|think|> block. This is the cell that "
                             "isolates the contribution of reasoning - every other arm "
                             "scaffolds CoT, so comparing them measures something else. "
                             "Run it with thinking mode OFF.")
    parser.add_argument("--thinking", action="store_true",
                        help="Turn on the endpoint's thinking mode. Needed on DeepSeek's "
                             "current models (deepseek-flash / deepseek-v4-pro), where "
                             "reasoning is a per-request mode rather than a separate "
                             "model - the retired deepseek-r1 had it always on.")
    parser.add_argument("--reasoning-effort", default="high",
                        choices=["minimal", "low", "medium", "high"],
                        help="Reasoning effort to request with --thinking. "
                             "DeepSeek only; Qwen3 has no such knob (default: high)")
    parser.add_argument("--thinking-style", default="auto", choices=list(THINKING_STYLES),
                        help="Which reasoning-mode dialect the endpoint speaks. "
                             "'auto' infers it from the model name. This matters in "
                             "both directions: DeepSeek reasons only when asked, "
                             "Qwen3 reasons unless told not to, so the dialect "
                             "decides whether --no-cot is really a no-reasoning "
                             "arm (default: auto)")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="Samples in flight at once (default: 5)")
    parser.add_argument("--models", default="config/models.yaml",
                        help="Path to models.yaml (default: config/models.yaml)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: outputs/{dataset}_single/{model}_{timestamp})")
    parser.add_argument("--no-stop", action="store_true",
                        help="Drop the dataset's stop sequences. Try this if a reasoning "
                             "endpoint rejects them or cuts off mid-thought.")
    parser.add_argument("--show", nargs="?", const="full", default="off",
                        choices=["off", "answer", "prompt", "full"],
                        help="Print results to the terminal as well as to cots.md. "
                             "A bare --show means --show full: the exact prompt that was "
                             "sent, then the reasoning and the answer side by side. "
                             "--show answer prints only the reasoning and answer; "
                             "--show prompt prints only the prompt (default: off)")
    parser.add_argument("--show-system", action="store_true",
                        help="Include the system prompt when --show displays prompts. "
                             "It is identical for every sample, so it is hidden by default.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the exact prompts that would be sent, without calling any API")
    return parser.parse_args()


async def main() -> int:
    args = parse_arguments()
    timestamp = datetime.now()

    logger.info("=" * 70)
    logger.info("Single-thinker CoT generation (no evolution)")
    logger.info("=" * 70)
    logger.info(f"Dataset: {args.dataset} ({args.split})")
    logger.info(f"Model: {args.model}{' [DRY RUN]' if args.dry_run else ''}")
    logger.info(f"Per sample: {args.n_vanilla} vanilla + {args.n_knowledge} knowledge-augmented")
    logger.info(
        f"Thinking mode: {'on (effort=' + args.reasoning_effort + ')' if args.thinking else 'off'}"
    )

    if args.no_cot:
        logger.info("CoT scaffold: OFF (direct-answer baseline)")
        if args.thinking:
            logger.error(
                "--no-cot with --thinking is not a no-reasoning condition: the endpoint "
                "still reasons internally, the trace is just hidden from the transcript. "
                "Re-run without --thinking."
            )
            return 1
        if args.n_knowledge:
            logger.error(
                "--no-cot with --n-knowledge is contradictory: knowledge augmentation "
                "injects reasoning material into the prompt. Use --n-knowledge 0."
            )
            return 1

    loader = DatasetLoader()
    if args.dataset not in loader.list_datasets():
        logger.error(f"Dataset '{args.dataset}' not found. "
                     f"Available: {', '.join(loader.list_datasets())}")
        return 1

    data = loader.load_dataset_data(args.dataset, max_samples=args.max_samples, split=args.split)
    if not data:
        logger.error(f"No samples loaded from the '{args.split}' split")
        return 1

    # ---- dry run: show the real prompts, spend nothing --------------------
    if args.dry_run:
        # Wrap the recorder exactly as a real model would be wrapped, so the
        # recorded kwargs show the thinking-mode params too.
        recorder = RecordingProvider()
        # No registry lookup here (a dry run must work without credentials), so
        # "auto" is inferred from the config alias rather than the API model
        # name. The real run re-infers from the resolved name.
        style = resolve_thinking_style(args.thinking_style, args.model, "--model")
        tuned = TunedProvider(
            recorder,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            extra_params=build_extra_params(
                style, args.thinking, args.reasoning_effort, args.top_k
            ),
            top_p=args.top_p,
        )
        generator = build_generator(tuned, args.dataset, needs_knowledge=args.n_knowledge > 0)
        # Mirror every override the real path applies, or the dry run shows a
        # prompt that is not the one that would be sent.
        if args.no_stop:
            generator.stop_sequences = []
        if args.no_cot:
            generator.system_prompt = DISCOURSE_MT_DIRECT_SYSTEM
        await generator.generate_initial_pool(
            query=data[0]["query"],
            ground_truth=data[0]["answer"],
            n_vanilla=max(args.n_vanilla, 1),
            n_knowledge_augmented=args.n_knowledge,
            prompt_template=DISCOURSE_MT_DIRECT_TEMPLATE if args.no_cot else None,
        )
        for call in recorder.calls:
            print("\n" + "=" * 70)
            if call["messages"] is None:
                print("KNOWLEDGE GENERATION PROMPT")
                print("=" * 70)
                print(call["prompt"])
            else:
                print(f"CoT GENERATION - request kwargs: {call['kwargs']}")
                for message in call["messages"]:
                    print("=" * 70)
                    print(f"[{message['role']}]")
                    print(message["content"])
        print("\n" + "=" * 70)
        print(f"Dry run: {len(recorder.calls)} request(s) would be sent per sample, "
              f"{len(recorder.calls) * len(data)} in total for {len(data)} sample(s).")
        return 0

    # ---- real run --------------------------------------------------------
    registry = ModelRegistry(config_path=args.models)
    inner = lookup_model(registry, args.model)
    style = resolve_thinking_style(args.thinking_style, inner.model_name, "model name")
    extra_params = build_extra_params(
        style, args.thinking, args.reasoning_effort, args.top_k
    )
    model = TunedProvider(
        inner,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        extra_params=extra_params,
        top_p=args.top_p,
    )
    logger.info(f"Resolved to API model_name={model.model_name!r} at {model.base_url}")
    logger.info(
        f"Thinking style: {style}"
        + (f" (inferred)" if args.thinking_style == "auto" else " (explicit)")
    )
    logger.info(f"extra_body sent with every request: {extra_params.get('extra_body', {})}")

    if style == "qwen3" and not args.thinking:
        # Worth stating, because it is the opposite of the DeepSeek default and
        # it is what makes --no-cot an honest baseline on this family.
        logger.info(
            "Qwen3 reasons by default; sending enable_thinking=false to turn it off."
        )
    elif style == "none" and args.thinking:
        logger.warning(
            "--thinking was passed but no dialect is known for this endpoint, so "
            "nothing was sent to enable it. The run will not be a thinking run."
        )

    def make_generator() -> Tuple[PromptRecorder, MultiThinkerGenerator]:
        """Fresh recorder + generator per sample, so prompt logs stay separated."""
        recorder = PromptRecorder(model)
        generator = build_generator(
            recorder, args.dataset, needs_knowledge=args.n_knowledge > 0
        )
        if args.no_stop:
            generator.stop_sequences = []
        if args.no_cot:
            # The dataset system prompt tells the model to reason explicitly,
            # which would contradict the direct template.
            generator.system_prompt = DISCOURSE_MT_DIRECT_SYSTEM
        return recorder, generator

    # One reference instance purely to report the settings actually in force.
    _, reference_generator = make_generator()
    logger.info(f"Stop sequences: {reference_generator.stop_sequences or 'none'}")

    percentiles = loader.get_dataset_config(args.dataset).length_percentiles
    exact_match = ExactMatchEvaluator(strict=True)
    length = LengthEvaluator(percentiles["lower"], percentiles["upper"])

    output_dir = Path(args.output_dir) if args.output_dir else (
        Path("outputs") / f"{args.dataset}_single"
        / f"{args.model}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Generating for {len(data)} sample(s), {args.concurrency} concurrent...")

    semaphore = asyncio.Semaphore(args.concurrency)

    async def guarded(idx: int, sample: Dict[str, Any]) -> Dict[str, Any]:
        async with semaphore:
            try:
                return await run_sample(
                    idx, sample, make_generator, exact_match, length,
                    args.n_vanilla, args.n_knowledge, args.no_cot,
                )
            except Exception as exc:
                logger.error(f"[{idx}] {sample.get('id')}: failed: {exc}")
                return {
                    "sample_id": sample.get("id", f"sample_{idx}"),
                    "task": sample.get("task", ""),
                    "subtask": sample.get("subtask", ""),
                    "query": sample["query"],
                    "ground_truth": sample["answer"],
                    "reference_knowledge": sample.get("knowledge"),
                    "trajectories": [],
                    "n_correct": 0,
                    "n_trajectories": 0,
                    "error": str(exc),
                }

    results = await asyncio.gather(
        *(guarded(i, s) for i, s in enumerate(data, 1))
    )

    for i, result in enumerate(results, 1):
        path = output_dir / f"sample_{i:02d}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    meta = {
        "dataset": args.dataset,
        "split": args.split,
        "model": args.model,
        "api_model_name": model.model_name,
        "n_vanilla": args.n_vanilla,
        "n_knowledge": args.n_knowledge,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "no_cot": args.no_cot,
        "thinking": args.thinking,
        "thinking_style": style,
        "extra_body": extra_params.get("extra_body", {}),
        "reasoning_effort": args.reasoning_effort if args.thinking else None,
        "stop_sequences": reference_generator.stop_sequences,
        "length_percentiles": percentiles,
        "timestamp": timestamp.isoformat(),
    }

    markdown_path = output_dir / "cots.md"
    write_markdown(markdown_path, list(results), meta)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": meta,
                "samples": [
                    {
                        "sample_id": r["sample_id"],
                        "task": r["task"],
                        "ground_truth": r["ground_truth"],
                        "n_correct": r["n_correct"],
                        "n_trajectories": r["n_trajectories"],
                        "n_requested": r.get("n_requested"),
                        "answers": [t["answer"] for t in r["trajectories"]],
                        "reasoning_words": [t["reasoning_words"] for t in r["trajectories"]],
                        "answer_parse": [t.get("answer_parse") for t in r["trajectories"]],
                        "generation_errors": r.get("generation_errors", []),
                        "error": r.get("error"),
                    }
                    for r in results
                ],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    if args.show != "off":
        show_prompt = args.show in ("prompt", "full")
        show_answer = args.show in ("answer", "full")

        for r in results:
            for t in r["trajectories"]:
                print("\n" + "=" * 70)
                print(f"{r['sample_id']} | {t['generation_method']} | "
                      f"gt={r['ground_truth']!r} | {'correct' if t['exact_match'] else 'WRONG'}")
                print("=" * 70)

                if show_prompt:
                    if args.show_system and t.get("system_prompt"):
                        print("--- SYSTEM PROMPT " + "-" * 51)
                        print(t["system_prompt"])
                    print("--- PROMPT SENT " + "-" * 53)
                    print(t["prompt"] or "(prompt could not be matched to this trajectory)")

                if show_answer:
                    print(f"--- REASONING ({t['reasoning_words']} words) " + "-" * 40)
                    print(t["reasoning"] or "(empty reasoning)")
                    print("--- ANSWER " + "-" * 58)
                    print(f"{t['answer']!r}   [ground truth: {r['ground_truth']!r}]")

    print_summary(list(results))
    logger.info(f"Read the CoTs here: {markdown_path}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
