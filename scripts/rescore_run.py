#!/usr/bin/env python3
"""
Re-score a finished generate_cot_single.py run, without calling any API.

Generation and scoring are separate concerns, but they were coupled: a bug in
the scorer meant throwing away correct model output and paying to generate it
again. This decouples them. The trajectories on disk are the expensive part;
re-reading them through the current evaluators costs nothing.

Use it whenever an evaluator changes, or whenever a run's accuracy looks wrong
and you want to check the scoring before blaming the model.

Usage:
    # report what the run scores under the current evaluators
    python scripts/rescore_run.py outputs/DiscourseMT_single/deepseek-flash_20260919_114854

    # and write the corrected verdicts back into the sample files
    python scripts/rescore_run.py outputs/.../deepseek-flash_20260919_114854 --in-place
"""

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError, ValueError):
        pass

from src.core.fitness import ExactMatchEvaluator, LengthEvaluator
from src.data.dataset_loader import DatasetLoader
from src.evaluation.discourse_mt import ChoiceEvaluator, CriticalSpanEvaluator


def load_dataset_index(meta: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Index the run's source dataset by sample id.

    Needed because the per-sample output files do not carry `meta`, and the
    translate-mode scorer needs the critical spans from it.
    """
    dataset = meta.get("dataset")
    split = meta.get("split", "train")
    if not dataset:
        return {}

    try:
        rows = DatasetLoader().load_dataset_data(dataset, max_samples=-1, split=split)
    except Exception as exc:
        print(f"  (could not load dataset {dataset}/{split}: {exc})")
        return {}

    return {r["id"]: r for r in rows}


def pick_scorer(subtask: str, source: Optional[Dict[str, Any]]):
    """Choose the evaluator the same way generate_cot_single.py does."""
    if subtask == "contrastive":
        return ChoiceEvaluator()
    if subtask == "translate":
        return CriticalSpanEvaluator((source or {}).get("meta") or {})
    return ExactMatchEvaluator(strict=True)


async def rescore(run_dir: Path, in_place: bool) -> int:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        print(f"No summary.json in {run_dir}")
        return 1

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    meta = summary.get("meta", {})
    index = load_dataset_index(meta)

    percentiles = meta.get("length_percentiles") or {}
    length = LengthEvaluator(
        percentiles.get("lower", 0), percentiles.get("upper", 10 ** 9)
    )

    files = sorted(run_dir.glob("sample_*.json"))
    if not files:
        print(f"No sample_*.json in {run_dir}")
        return 1

    by_task: Dict[str, list] = collections.defaultdict(lambda: [0, 0])
    parses: collections.Counter = collections.Counter()
    changed = before = after = total = 0
    words = []

    for path in files:
        sample = json.loads(path.read_text(encoding="utf-8"))
        source = index.get(sample.get("sample_id"))
        subtask = sample.get("subtask") or (source or {}).get("subtask", "")

        n_correct = 0
        for traj in sample["trajectories"]:
            total += 1
            was = bool(traj.get("exact_match"))
            before += was

            scorer = pick_scorer(subtask, source)
            now = await scorer.match(traj.get("answer") or "", sample["ground_truth"])
            parses[getattr(scorer, "last_parse", "exact") or "exact"] += 1

            after += now
            n_correct += now
            changed += was != now

            if traj.get("reasoning_words"):
                words.append(traj["reasoning_words"])

            traj["exact_match"] = now
            traj["answer_parse"] = getattr(scorer, "last_parse", "exact")
            traj["length_score"] = length.score(traj.get("reasoning") or "")

        sample["n_correct"] = n_correct
        by_task[sample.get("task", "")][0] += n_correct
        by_task[sample.get("task", "")][1] += len(sample["trajectories"])

        if in_place:
            path.write_text(
                json.dumps(sample, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    print(f"Run: {run_dir}")
    print(f"  dataset={meta.get('dataset')} model={meta.get('model')} "
          f"thinking={meta.get('thinking')}")
    print(f"  trajectories: {total}")
    print(f"  stored score: {before}/{total} = {before / total:.1%}")
    print(f"  re-scored   : {after}/{total} = {after / total:.1%}   ({changed} verdict(s) changed)")
    print("  by task:")
    for task, (ok, n) in sorted(by_task.items()):
        print(f"    {task:<14} {ok:>5}/{n:<5} {ok / n:.1%}" if n else f"    {task}: none")
    print(f"  parse modes: {dict(parses)}")

    if words:
        words.sort()
        n = len(words)
        print(f"  reasoning words: p15={words[int(0.15 * (n - 1))]} "
              f"median={words[n // 2]} p85={words[int(0.85 * (n - 1))]} max={words[-1]}")

    if in_place:
        summary["rescored"] = {
            "n_correct": after,
            "n_trajectories": total,
            "verdicts_changed": changed,
            "parse_modes": dict(parses),
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  wrote updated verdicts into {len(files)} sample file(s) and summary.json")
    else:
        print("  (report only - pass --in-place to write the corrected verdicts back)")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-score a finished run with the current evaluators (no API calls)"
    )
    parser.add_argument("run_dir", nargs="+", type=Path,
                        help="One or more generate_cot_single.py output directories")
    parser.add_argument("--in-place", action="store_true",
                        help="Write corrected verdicts back into the sample files "
                             "and summary.json (default: report only)")
    args = parser.parse_args()

    import asyncio
    status = 0
    for d in args.run_dir:
        status |= asyncio.run(rescore(d, args.in_place))
        print()
    return status


if __name__ == "__main__":
    sys.exit(main())
