"""
Build DPO preference pairs from quality-tagged trajectories.

Three pair sources, each tagged in metadata so trainers can weight or filter:

  1. repair_pair: (repaired version, original low-quality version) — same
     trajectory, before/after rule-based repair
  2. quality_contrast: (high-quality, low-quality) trajectories sharing the
     same domain — different trajectories, similar query intent
  3. persona_synthetic: deliberately generated persona-violating reply paired
     with the actual high-quality reply on the same trajectory

DPO JSONL format (TRL-compatible):
{
  "prompt": "<rendered user query + any clarification turns>",
  "chosen": "<good assistant reply>",
  "rejected": "<bad assistant reply>",
  "metadata": {
    "source": "repair_pair" | "quality_contrast" | "persona_synthetic",
    "chosen_traj_id": "...",
    "rejected_traj_id": "...",
    "fixes_applied": [...],
    "domain_tag": "...",
    "complexity_bucket": "...",
    "language_mix": "..."
  }
}

Run:
    python -m scripts.export_dpo \
        --input data/processed/trajectories_quality_tagged.jsonl \
        --output data/exports/dpo_pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path

from src.repair.rule_based import repair_trajectory

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("export_dpo")


def _quality_label(traj: dict) -> str | None:
    """Extract quality label from final-assistant metadata (set by Module 8)."""
    for ev in reversed(traj.get("events", [])):
        if ev.get("role") == "assistant":
            q = ev.get("metadata", {}).get("quality", {})
            return q.get("label")
    return None


def _final_reply(traj: dict) -> str:
    for ev in reversed(traj.get("events", [])):
        if ev.get("role") == "assistant":
            return ev.get("content", "")
    return ""


def _user_prompt(traj: dict) -> str:
    """Concatenate user turns + any assistant clarifications into a prompt string."""
    parts = []
    for ev in traj.get("events", []):
        role = ev.get("role")
        if role == "user":
            parts.append(f"User: {ev.get('content', '')}")
        elif role == "assistant" and any(
            "clarif" in (next_ev.get("content", "") or "").lower()
            for next_ev in traj.get("events", [])
            if next_ev != ev and next_ev.get("role") == "user"
        ):
            # Clarification turn — include
            parts.append(f"Assistant: {ev.get('content', '')}")
        elif role == "assistant":
            break  # stop at the first non-clarification assistant reply
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Pair builders
# ---------------------------------------------------------------------------

def build_repair_pairs(trajectories: list[dict]) -> list[dict]:
    """For every low-quality trajectory, try to repair it. If repair succeeds,
    emit a DPO pair: chosen=repaired, rejected=original."""
    pairs = []
    for traj in trajectories:
        if _quality_label(traj) != "low":
            continue
        result = repair_trajectory(traj)
        if not result.repaired:
            continue
        chosen_reply = _final_reply(result.trajectory)
        rejected_reply = _final_reply(traj)
        if chosen_reply.strip() == rejected_reply.strip():
            continue  # repair was a no-op
        pairs.append({
            "prompt": _user_prompt(traj),
            "chosen": chosen_reply,
            "rejected": rejected_reply,
            "metadata": {
                "source": "repair_pair",
                "chosen_traj_id": traj.get("trajectory_id") + "_repaired",
                "rejected_traj_id": traj.get("trajectory_id"),
                "fixes_applied": result.fixes_applied,
                "domain_tag": traj.get("domain_tag"),
                "complexity_bucket": traj.get("complexity_bucket"),
                "language_mix": traj.get("language_mix"),
            },
        })
    return pairs


def build_quality_contrast_pairs(trajectories: list[dict], rng: random.Random) -> list[dict]:
    """Pair high-quality and low-quality trajectories sharing the same domain.

    Both trajectories asked questions in the same domain (price, irrigation,
    etc.) but one got a good reply and the other got a bad one. The user
    prompts differ slightly but they sample the same prompt distribution,
    so this gives the trainer useful preference signal at the response level.
    """
    by_domain: dict[str, dict[str, list[dict]]] = defaultdict(lambda: {"high": [], "low": []})
    for traj in trajectories:
        label = _quality_label(traj)
        domain = traj.get("domain_tag")
        if label in ("high", "low") and domain:
            by_domain[domain][label].append(traj)

    pairs = []
    for domain, buckets in by_domain.items():
        highs = buckets["high"]
        lows = buckets["low"]
        if not highs or not lows:
            continue
        # Pair each low with a randomly-chosen high in the same domain
        for low in lows:
            high = rng.choice(highs)
            pairs.append({
                "prompt": _user_prompt(low),  # use the bad-trajectory's prompt
                "chosen": _final_reply(high),
                "rejected": _final_reply(low),
                "metadata": {
                    "source": "quality_contrast",
                    "chosen_traj_id": high.get("trajectory_id"),
                    "rejected_traj_id": low.get("trajectory_id"),
                    "domain_tag": domain,
                    "complexity_bucket": low.get("complexity_bucket"),
                    "language_mix": low.get("language_mix"),
                    "note": "Chosen and rejected come from different trajectories in the same domain",
                },
            })
    return pairs


def build_persona_synthetic_pairs(trajectories: list[dict], rng: random.Random, max_pairs: int = 30) -> list[dict]:
    """Synthesize persona-violating replies for high-quality trajectories.

    The chosen reply is the actual (high-quality) one. The rejected is a
    deliberately persona-violating variant (USD currency, no action, etc.).
    Cheap to generate, gives the trainer direct persona signal.
    """
    high_trajs = [t for t in trajectories if _quality_label(t) == "high"]
    if not high_trajs:
        return []
    sample = rng.sample(high_trajs, min(max_pairs, len(high_trajs)))

    persona_violations = [
        ("USD currency", lambda r: r.replace("₹", "$").replace("INR", "USD").replace("per quintal", "per bushel")),
        ("Generic non-action", lambda r: "I'm not sure about the specifics, you'll have to check yourself."),
        ("No tool citation", lambda r: r.replace("Based on the", "").replace("According to", "")),
    ]

    pairs = []
    for traj in sample:
        good_reply = _final_reply(traj)
        if not good_reply or len(good_reply) < 20:
            continue
        violation_name, transform = rng.choice(persona_violations)
        bad_reply = transform(good_reply)
        if bad_reply.strip() == good_reply.strip():
            continue
        pairs.append({
            "prompt": _user_prompt(traj),
            "chosen": good_reply,
            "rejected": bad_reply,
            "metadata": {
                "source": "persona_synthetic",
                "violation_type": violation_name,
                "chosen_traj_id": traj.get("trajectory_id"),
                "rejected_traj_id": traj.get("trajectory_id") + "_persona_violated",
                "domain_tag": traj.get("domain_tag"),
                "complexity_bucket": traj.get("complexity_bucket"),
                "language_mix": traj.get("language_mix"),
            },
        })
    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build DPO preference pairs.")
    parser.add_argument("--input", type=Path, required=True,
                        help="Quality-tagged trajectory JSONL (from Module 8).")
    parser.add_argument("--output", type=Path, default=Path("data/exports/dpo_pairs.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("data/exports/dpo_summary.json"))
    parser.add_argument("--max-persona-pairs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # Load
    trajectories = []
    with args.input.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                trajectories.append(json.loads(line))
    logger.info(f"Loaded {len(trajectories)} trajectories from {args.input}")

    # Quality label distribution
    label_counts = Counter(_quality_label(t) for t in trajectories)
    logger.info(f"Quality label distribution: {dict(label_counts)}")

    # Build pairs from each source
    repair_pairs = build_repair_pairs(trajectories)
    logger.info(f"Repair pairs: {len(repair_pairs)}")

    contrast_pairs = build_quality_contrast_pairs(trajectories, rng)
    logger.info(f"Quality-contrast pairs: {len(contrast_pairs)}")

    persona_pairs = build_persona_synthetic_pairs(trajectories, rng, max_pairs=args.max_persona_pairs)
    logger.info(f"Persona-synthetic pairs: {len(persona_pairs)}")

    all_pairs = repair_pairs + contrast_pairs + persona_pairs
    rng.shuffle(all_pairs)

    # Write
    with args.output.open("w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    summary = {
        "total_pairs": len(all_pairs),
        "by_source": {
            "repair_pair": len(repair_pairs),
            "quality_contrast": len(contrast_pairs),
            "persona_synthetic": len(persona_pairs),
        },
        "quality_label_distribution": dict(label_counts),
        "output_file": str(args.output),
    }
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("  DPO pair export complete")
    print("=" * 60)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()