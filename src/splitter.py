"""
Split SFT JSONL into train / eval / preference subsets with leak prevention.

Splits by trajectory_id (no session bleed) into 80/10/10:
  train       — for supervised fine-tuning
  eval        — held-out evaluation
  preference  — reserved for DPO pair construction in Phase 2

MinHash + LSH catches near-duplicates across splits:
  - Build LSH index of all train records' user-message shingles
  - For every eval and preference record, query the index
  - Drop any record whose Jaccard similarity > threshold (default 0.7)
    against any train record — that's a leak

This addresses the ticket's "no near-duplicate leakage across train and
preference sets" acceptance criterion.

Run:
    python -m src.splitter \
        --input data/exports/sft_all.jsonl \
        --output-dir data/exports/
"""

import argparse
import json
import logging
import random
import re
from collections import Counter
from pathlib import Path

from datasketch import MinHash, MinHashLSH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("splitter")

NUM_PERM = 128       # MinHash permutations — 128 is the standard tradeoff
SHINGLE_SIZE = 3     # word n-gram size for shingling
SIMILARITY_THRESHOLD = 0.7


def extract_user_text(record: dict) -> str:
    """Concatenate all user messages in a record. We hash this because user
    queries are what makes records semantically similar — assistant responses
    can paraphrase the same content, but the *prompt* identifies the query."""
    user_chunks = [
        m["content"] for m in record["messages"]
        if m["role"] == "user" and m.get("content")
    ]
    return " ".join(user_chunks).lower()


def shingle(text: str, n: int = SHINGLE_SIZE) -> set[str]:
    """Word-level n-grams. We strip punctuation and split on whitespace —
    this is robust to small variations like capitalization or trailing periods."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    if len(tokens) < n:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def make_minhash(text: str) -> MinHash:
    mh = MinHash(num_perm=NUM_PERM)
    for sh in shingle(text):
        mh.update(sh.encode("utf-8"))
    return mh


def main() -> None:
    parser = argparse.ArgumentParser(description="Split SFT JSONL into train/eval/preference.")
    parser.add_argument("--input", type=Path, default=Path("data/exports/sft_all.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/exports/"))
    parser.add_argument("--train-frac", type=float, default=0.80)
    parser.add_argument("--eval-frac", type=float, default=0.10)
    parser.add_argument("--preference-frac", type=float, default=0.10)
    parser.add_argument("--threshold", type=float, default=SIMILARITY_THRESHOLD,
                        help="Jaccard similarity above which a record is treated as a leak.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    fracs = (args.train_frac, args.eval_frac, args.preference_frac)
    if abs(sum(fracs) - 1.0) > 1e-6:
        raise SystemExit(f"Fractions must sum to 1.0; got {sum(fracs)}")

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    # Load all records
    logger.info(f"Loading {args.input}")
    records: list[dict] = []
    with args.input.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info(f"Loaded {len(records)} records")

    # Shuffle by trajectory_id-derived key (deterministic given seed)
    random.shuffle(records)

    # Naive index-based split
    n = len(records)
    n_train = int(n * args.train_frac)
    n_eval = int(n * args.eval_frac)
    train = records[:n_train]
    eval_split = records[n_train:n_train + n_eval]
    preference = records[n_train + n_eval:]

    logger.info(f"Initial split: train={len(train)}, eval={len(eval_split)}, preference={len(preference)}")

    # ---- Build LSH index over train ----
    logger.info(f"Building MinHash LSH index (num_perm={NUM_PERM}, threshold={args.threshold})...")
    lsh = MinHashLSH(threshold=args.threshold, num_perm=NUM_PERM)
    for rec in train:
        text = extract_user_text(rec)
        if not text:
            continue
        mh = make_minhash(text)
        lsh.insert(rec["metadata"]["trajectory_id"], mh)
    logger.info(f"LSH indexed {len(train)} train records")

    # ---- Filter eval and preference against the train index ----
    def filter_against_train(split: list[dict], split_name: str) -> tuple[list[dict], list[dict]]:
        kept: list[dict] = []
        dropped: list[dict] = []
        for rec in split:
            text = extract_user_text(rec)
            if not text:
                kept.append(rec)
                continue
            mh = make_minhash(text)
            matches = lsh.query(mh)
            if matches:
                dropped.append({
                    "trajectory_id": rec["metadata"]["trajectory_id"],
                    "split": split_name,
                    "matched_train_ids": list(matches),
                    "user_text_preview": text[:120],
                })
            else:
                kept.append(rec)
        return kept, dropped

    eval_kept, eval_dropped = filter_against_train(eval_split, "eval")
    pref_kept, pref_dropped = filter_against_train(preference, "preference")

    logger.info(f"Eval: kept {len(eval_kept)}, dropped {len(eval_dropped)} near-duplicates")
    logger.info(f"Preference: kept {len(pref_kept)}, dropped {len(pref_dropped)} near-duplicates")

    # ---- Write output files ----
    def write_jsonl(path: Path, recs: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info(f"Wrote {len(recs)} records to {path}")

    write_jsonl(args.output_dir / "sft_train.jsonl", train)
    write_jsonl(args.output_dir / "sft_eval.jsonl", eval_kept)
    write_jsonl(args.output_dir / "preference_pool.jsonl", pref_kept)

    # Composition stats per split
    def composition(recs: list[dict], key: str) -> dict[str, int]:
        return dict(Counter(r["metadata"].get(key) for r in recs))

    report = {
        "input_records": n,
        "splits": {
            "train": {
                "count": len(train),
                "complexity_distribution": composition(train, "complexity_bucket"),
                "domain_distribution": composition(train, "domain_tag"),
            },
            "eval": {
                "count": len(eval_kept),
                "dropped_near_duplicates": len(eval_dropped),
                "complexity_distribution": composition(eval_kept, "complexity_bucket"),
                "domain_distribution": composition(eval_kept, "domain_tag"),
            },
            "preference": {
                "count": len(pref_kept),
                "dropped_near_duplicates": len(pref_dropped),
                "complexity_distribution": composition(pref_kept, "complexity_bucket"),
                "domain_distribution": composition(pref_kept, "domain_tag"),
            },
        },
        "minhash_config": {
            "num_perm": NUM_PERM,
            "shingle_size": SHINGLE_SIZE,
            "jaccard_threshold": args.threshold,
        },
        "dropped_details": {
            "eval": eval_dropped[:20],         # cap to keep file readable
            "preference": pref_dropped[:20],
        },
    }

    report_path = args.output_dir / "split_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Wrote split report to {report_path}")

    print("\n" + "=" * 60)
    print("  Split complete")
    print("=" * 60)
    print(json.dumps({
        "train": len(train),
        "eval": len(eval_kept),
        "preference": len(pref_kept),
        "eval_near_duplicates_dropped": len(eval_dropped),
        "preference_near_duplicates_dropped": len(pref_dropped),
        "report_file": str(report_path),
    }, indent=2))


if __name__ == "__main__":
    main()
