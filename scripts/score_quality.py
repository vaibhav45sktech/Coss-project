"""
Apply quality scoring to a trajectory JSONL file.

Reads trajectories, runs the heuristic scorer on each, writes per-trajectory
quality reports and a summary distribution. Optionally uses the LLM judge
when USE_LLM is set.

Run:
    python -m scripts.score_quality \
        --input data/processed/trajectories_redacted.jsonl \
        --output data/processed/quality_reports.jsonl \
        --summary data/processed/quality_summary.json
"""

import argparse
import json
import logging
import os
from collections import Counter
from pathlib import Path

from src.quality.scorer import HeuristicScorer
from src.quality.llm_judge import LLMJudge

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("score_quality")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score trajectory quality.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path,
                        default=Path("data/processed/quality_summary.json"))
    parser.add_argument("--persona", type=Path, default=Path("config/persona.md"))
    parser.add_argument("--use-llm-judge", action="store_true",
                        help="Also run LLMJudge if USE_LLM is set")
    parser.add_argument("--tag-trajectories", type=Path, default=None,
                        help="Write a copy of input with quality labels stamped on assistant events")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    heuristic = HeuristicScorer.load(args.persona)
    llm_judge = LLMJudge(args.persona) if args.use_llm_judge else None

    label_counts: Counter = Counter()
    score_buckets: Counter = Counter()
    total = 0
    llm_judged = 0

    with args.input.open(encoding="utf-8") as fin, args.output.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            traj = json.loads(line)
            report = heuristic.score(traj)
            out_record = {"heuristic": report.to_dict()}

            if llm_judge and llm_judge.enabled:
                llm_report = llm_judge.score(traj)
                if llm_report:
                    out_record["llm_judge"] = llm_report.to_dict()
                    llm_judged += 1

            fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            total += 1
            label_counts[report.quality_label] += 1

            # Histogram buckets
            bucket = round(report.composite_score * 10) / 10
            score_buckets[f"{bucket:.1f}"] += 1

            if total % 50 == 0:
                logger.info(f"  scored {total} trajectories...")

    summary = {
        "total_trajectories": total,
        "label_distribution": dict(label_counts),
        "score_histogram": dict(sorted(score_buckets.items())),
        "scorer": "heuristic",
        "llm_judged": llm_judged,
        "weights": heuristic.weights,
    }
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("  Quality scoring complete")
    print("=" * 60)
    print(json.dumps(summary, indent=2))

    if args.tag_trajectories:
        args.tag_trajectories.parent.mkdir(parents=True, exist_ok=True)
        # Re-read input + reports together
        reports_by_id = {}
        with args.output.open(encoding="utf-8") as fr:
            for line in fr:
                rec = json.loads(line)
                tid = rec["heuristic"]["trajectory_id"]
                reports_by_id[tid] = rec["heuristic"]

        with args.input.open(encoding="utf-8") as fin, args.tag_trajectories.open("w", encoding="utf-8") as ftag:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                traj = json.loads(line)
                tid = traj.get("trajectory_id")
                report = reports_by_id.get(tid)
                if report:
                    # Stash on the final assistant message metadata for easy retrieval
                    for ev in reversed(traj.get("events", [])):
                        if ev.get("role") == "assistant":
                            ev.setdefault("metadata", {})["quality"] = {
                                "label": report["quality_label"],
                                "composite_score": report["composite_score"],
                            }
                            break
                ftag.write(json.dumps(traj, ensure_ascii=False) + "\n")
        print(f"\nTagged trajectories written to: {args.tag_trajectories}")


if __name__ == "__main__":
    main()
