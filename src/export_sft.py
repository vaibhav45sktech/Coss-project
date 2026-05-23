"""
Convert redacted trajectories into SFT-ready JSONL formatted for the Qwen2.5
chat template.

Each input trajectory becomes one SFT record:
  {
    "messages": [...],      # chat-template-formatted conversation
    "metadata": {...}       # complexity, domain, etc. for curriculum/sampling
  }

The metadata is preserved so downstream training can do stratified sampling
or curriculum scheduling without recomputing tags from scratch — this is the
"complexity tags map to training schedules" requirement from the ticket.

Tool calls use Hermes-style format which Qwen2.5-Instruct understands natively.

Run:
    python -m src.export_sft \
        --input data/processed/trajectories_redacted.jsonl \
        --output data/exports/sft_all.jsonl \
        --model Qwen/Qwen2.5-0.5B-Instruct
"""

import argparse
import json
import logging
import uuid
from collections import Counter
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("export_sft")


def trajectory_to_sft_messages(trajectory: dict) -> list[dict]:
    """Convert a trajectory's event list into chat-template messages.

    Mapping rules:
      role=system    -> {"role": "system", "content": ...}
      role=user      -> {"role": "user", "content": ...}
      role=assistant -> {"role": "assistant", "content": ...}
      role=tool_call -> attached to the *next* assistant message as tool_calls,
                        OR emitted as a standalone assistant message with
                        empty content + tool_calls if no assistant text follows
      role=tool_result -> {"role": "tool", "tool_call_id": ..., "content": json}
      role=error     -> skipped (errors aren't part of the model's target output;
                        the recovery is captured by the subsequent tool_call)

    We pair tool_call events with the *immediately following* tool_result by
    matching tool_name. If the pairing fails, we still emit both — the model
    will see the call but no result, which mirrors real production logs.
    """
    messages: list[dict] = []
    pending_tool_calls: list[dict] = []
    tool_call_id_map: dict[str, str] = {}  # tool_name -> call_id (most recent)

    for event in trajectory["events"]:
        role = event["role"]

        if role == "system":
            messages.append({"role": "system", "content": event["content"] or ""})

        elif role == "user":
            # If we have unflushed tool_calls, that's a malformed trajectory —
            # emit them as a standalone assistant turn first.
            if pending_tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": pending_tool_calls,
                })
                pending_tool_calls = []
            messages.append({"role": "user", "content": event["content"] or ""})

        elif role == "tool_call":
            call_id = f"call_{uuid.uuid4().hex[:12]}"
            tool_call_id_map[event["tool_name"]] = call_id
            pending_tool_calls.append({
                "id": call_id,
                "type": "function",
                "function": {
                    "name": event["tool_name"],
                    "arguments": json.dumps(event["tool_args"] or {}, ensure_ascii=False),
                },
            })

        elif role == "tool_result":
            # Flush pending tool_calls into an assistant message first
            if pending_tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": pending_tool_calls,
                })
                pending_tool_calls = []
            tool_name = event["tool_name"]
            call_id = tool_call_id_map.get(tool_name, f"call_{uuid.uuid4().hex[:12]}")
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(event["tool_output"] or {}, ensure_ascii=False),
            })

        elif role == "assistant":
            # Flush any pending tool_calls into this assistant turn
            msg = {"role": "assistant", "content": event["content"] or ""}
            if pending_tool_calls:
                msg["tool_calls"] = pending_tool_calls
                pending_tool_calls = []
            messages.append(msg)

        elif role == "error":
            # Skip — errors are part of the log but not part of the model's
            # target trajectory. The recovery (next tool_call) carries the signal.
            continue

    # Flush any leftover tool_calls at the very end
    if pending_tool_calls:
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": pending_tool_calls,
        })

    return messages


def validate_with_chat_template(messages: list[dict], tokenizer) -> bool:
    """Try rendering through tokenizer.apply_chat_template. Returns True if
    the messages list is template-compatible.

    This is the key acceptance test: if the trainer can't render this record,
    it won't train on it. Validating here keeps bad records out of the export.
    """
    try:
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return True
    except Exception as e:
        logger.debug(f"Chat template failed: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Export trajectories as SFT JSONL.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/trajectories_redacted.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/exports/sft_all.jsonl"),
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HuggingFace model whose chat template we render against.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading tokenizer for chat-template validation: {args.model}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    exported = 0
    skipped = 0
    complexity_counter: Counter = Counter()
    domain_counter: Counter = Counter()

    logger.info(f"Reading trajectories from {args.input}")
    with args.input.open(encoding="utf-8") as fin, args.output.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            trajectory = json.loads(line)

            messages = trajectory_to_sft_messages(trajectory)
            if not validate_with_chat_template(messages, tokenizer):
                skipped += 1
                logger.warning(f"Skipped trajectory {trajectory['trajectory_id']}: chat template failed")
                continue

            sft_record = {
                "messages": messages,
                "metadata": {
                    "trajectory_id": trajectory["trajectory_id"],
                    "session_id": trajectory["session_id"],
                    "domain_tag": trajectory.get("domain_tag"),
                    "complexity_bucket": trajectory.get("complexity_bucket"),
                    "language_mix": trajectory.get("language_mix"),
                    "step_count": trajectory.get("step_count"),
                    "had_error_recovery": trajectory.get("had_error_recovery"),
                    "tools_used": trajectory.get("tools_used", []),
                },
            }
            fout.write(json.dumps(sft_record, ensure_ascii=False) + "\n")
            exported += 1
            complexity_counter[trajectory.get("complexity_bucket")] += 1
            domain_counter[trajectory.get("domain_tag")] += 1

    print("\n" + "=" * 60)
    print("  SFT export complete")
    print("=" * 60)
    print(json.dumps({
        "exported": exported,
        "skipped": skipped,
        "complexity_distribution": dict(complexity_counter),
        "domain_distribution": dict(domain_counter),
        "output_file": str(args.output),
        "chat_template_model": args.model,
    }, indent=2))


if __name__ == "__main__":
    main()
