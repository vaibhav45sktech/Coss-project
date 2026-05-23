"""
Run the full PII pipeline over ingested trajectories.

Input:  data/processed/trajectories.jsonl       (from src/ingest.py)
Output:
  data/processed/trajectories_redacted.jsonl    redacted trajectories
  data/processed/audit_sample.json              5% sample for human review
  data/processed/pii_run_stats.json             summary statistics

Each trajectory gets its own PIIPipeline session (fresh placeholder map),
so placeholders are consistent within a session but isolated across sessions.

Run from repo root:
    python scripts/run_pii.py
    python scripts/run_pii.py --sample-rate 0.10   # increase audit coverage
"""

import argparse
import json
import logging
import random
from collections import Counter
from pathlib import Path

from src.pii import PIIPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_pii")


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply PII redaction to ingested trajectories.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/trajectories.jsonl"),
        help="Input trajectories JSONL (from ingest).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/trajectories_redacted.jsonl"),
        help="Where to write redacted trajectories.",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("data/processed/audit_sample.json"),
        help="Where to write the audit sample for human review.",
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=Path("data/processed/pii_run_stats.json"),
        help="Where to write the run stats summary.",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=0.05,
        help="Fraction of trajectories sampled into the audit file (default: 0.05 = 5%%).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for audit sampling reproducibility.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    logger.info("Initializing PII pipeline (this loads Presidio + spaCy)...")
    pipeline = PIIPipeline()

    # Stats
    total_trajectories = 0
    trajectories_with_pii = 0
    total_redactions = 0
    type_counter: Counter = Counter()
    source_counter: Counter = Counter()
    confidence_buckets = {"high (>=0.9)": 0, "medium (0.7-0.9)": 0, "low (<0.7)": 0}

    audit_records: list[dict] = []

    logger.info(f"Reading trajectories from {args.input}")
    with args.input.open(encoding="utf-8") as fin, args.output.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            trajectory = json.loads(line)
            total_trajectories += 1

            redacted_traj, traj_audits = pipeline.redact_trajectory(trajectory)
            fout.write(json.dumps(redacted_traj) + "\n")

            if traj_audits:
                trajectories_with_pii += 1
                total_redactions += sum(len(a["matches"]) for a in traj_audits)

                for audit in traj_audits:
                    for m in audit["matches"]:
                        type_counter[m["type"]] += 1
                        source_counter[m["source"]] += 1
                        conf = m["confidence"]
                        if conf >= 0.9:
                            confidence_buckets["high (>=0.9)"] += 1
                        elif conf >= 0.7:
                            confidence_buckets["medium (0.7-0.9)"] += 1
                        else:
                            confidence_buckets["low (<0.7)"] += 1

                # Sample for audit
                if random.random() < args.sample_rate:
                    audit_records.append({
                        "trajectory_id": redacted_traj["trajectory_id"],
                        "session_id": redacted_traj["session_id"],
                        "domain": redacted_traj.get("domain_tag"),
                        "redactions": traj_audits,
                    })

            if total_trajectories % 50 == 0:
                logger.info(f"  processed {total_trajectories} trajectories...")

    # Write audit file
    args.audit.write_text(
        json.dumps({
            "description": (
                "Sampled trajectories for human PII review. Each record shows "
                "the original text, redacted text, and the matches that produced "
                "the redaction. Reviewers should flag false positives, false "
                "negatives, and document any patterns in residual risk."
            ),
            "sample_rate": args.sample_rate,
            "total_audit_records": len(audit_records),
            "records": audit_records,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Write stats summary
    stats = {
        "total_trajectories": total_trajectories,
        "trajectories_with_pii": trajectories_with_pii,
        "trajectories_clean": total_trajectories - trajectories_with_pii,
        "total_redactions": total_redactions,
        "pii_type_distribution": dict(type_counter.most_common()),
        "detection_source_distribution": dict(source_counter),
        "confidence_distribution": confidence_buckets,
        "audit_sample_rate": args.sample_rate,
        "audit_records_written": len(audit_records),
    }
    args.stats.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    # Pretty print
    print("\n" + "=" * 60)
    print("  PII redaction complete")
    print("=" * 60)
    print(json.dumps(stats, indent=2))
    print(f"\n  Redacted trajectories : {args.output}")
    print(f"  Audit sample          : {args.audit}")
    print(f"  Stats                 : {args.stats}\n")


if __name__ == "__main__":
    main()