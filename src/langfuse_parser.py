"""
Convert Langfuse/Pydantic-AI traces into our canonical Trajectory schema.

This is the boundary adapter: production logs go in, our internal schema
comes out. The rest of the pipeline (PII, SFT export, splitter) operates
on Trajectory objects, so this adapter is the only piece that needs to
know about the Langfuse wire format.

Mapping rules:
  user_question        → RawEvent(role="user", content=...)
  bot_response         → RawEvent(role="assistant", content=...) at end
  agent_turn.tool-call → RawEvent(role="tool_call", tool_name, tool_args)
  agent_turn.tool-return → RawEvent(role="tool_result", tool_name, tool_output)
  agent_turn.text      → RawEvent(role="assistant", content=...)
  finish_reason="error" or non-success → RawEvent(role="error", error_message)

Run:
    python -m src.langfuse_parser \
        --input data/synthetic_logs/langfuse_traces.jsonl \
        --output data/processed/trajectories_from_langfuse.jsonl
"""

import argparse
import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.langfuse_schema import LangfuseTrace
from src.schemas import EventRole, RawEvent
from src.ingest import session_to_trajectory

logger = logging.getLogger(__name__)


def _now_iso() -> datetime:
    return datetime.now(timezone.utc)


def _make_event(
    session_id: str,
    role: EventRole,
    ts: datetime,
    **kwargs: Any,
) -> RawEvent:
    return RawEvent(
        event_id=str(uuid.uuid4()),
        session_id=session_id,
        timestamp=ts,
        role=role,
        **kwargs,
    )


def parse_langfuse_trace(raw: dict) -> list[RawEvent]:
    """Convert one Langfuse trace dict into a list of RawEvent objects.

    Returns events in chronological order ready for session_to_trajectory.
    """
    trace = LangfuseTrace(**raw)
    session_id = trace.session_id or trace.trace_id or str(uuid.uuid4())
    base_ts = trace.timestamp or _now_iso()

    events: list[RawEvent] = []
    tick = 0  # incremented seconds for ordering

    def next_ts() -> datetime:
        nonlocal tick
        tick += 1
        return base_ts + timedelta(seconds=tick)

    # 1. user_question
    if trace.user_question:
        events.append(_make_event(
            session_id=session_id,
            role=EventRole.USER,
            ts=next_ts(),
            content=trace.user_question,
        ))

    # 2. walk agent_turns
    for turn in trace.agent_turns:
        for part in turn.parts:
            kind = part.get("part_kind")

            if kind == "tool-call":
                events.append(_make_event(
                    session_id=session_id,
                    role=EventRole.TOOL_CALL,
                    ts=next_ts(),
                    tool_name=part.get("tool_name", "unknown"),
                    tool_args=part.get("args") or {},
                ))
            elif kind == "tool-return":
                # Tool return content can be a string or already-structured dict
                content = part.get("content")
                if isinstance(content, str):
                    tool_output = {"text": content}
                elif isinstance(content, dict):
                    tool_output = content
                else:
                    tool_output = {"raw": str(content)}
                events.append(_make_event(
                    session_id=session_id,
                    role=EventRole.TOOL_RESULT,
                    ts=next_ts(),
                    tool_name=part.get("tool_name", "unknown"),
                    tool_output=tool_output,
                ))
            elif kind == "text":
                # An intermediate assistant text inside a turn — emit as assistant event
                events.append(_make_event(
                    session_id=session_id,
                    role=EventRole.ASSISTANT,
                    ts=next_ts(),
                    content=part.get("content", ""),
                ))

        # If the turn has finish_reason indicating error, emit an ERROR event
        if turn.finish_reason and turn.finish_reason.lower() in {"error", "failed", "timeout"}:
            events.append(_make_event(
                session_id=session_id,
                role=EventRole.ERROR,
                ts=next_ts(),
                error_message=f"Turn finished with reason: {turn.finish_reason}",
            ))

    # 3. final bot_response
    if trace.bot_response:
        events.append(_make_event(
            session_id=session_id,
            role=EventRole.ASSISTANT,
            ts=next_ts(),
            content=trace.bot_response,
        ))

    return events


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse Langfuse traces into Trajectory JSONL.")
    parser.add_argument("--input", type=Path, required=True,
                        help="JSONL file of Langfuse trace objects (one per line).")
    parser.add_argument("--output", type=Path, required=True,
                        help="Where to write canonical Trajectory JSONL.")
    parser.add_argument("--quarantine", type=Path, default=None,
                        help="Where to write malformed trace records.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    quarantine_path = args.quarantine or args.output.parent / "langfuse_quarantine.jsonl"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    quarantine_path.parent.mkdir(parents=True, exist_ok=True)

    parsed = 0
    quarantined = 0

    with args.input.open(encoding="utf-8") as fin, \
         args.output.open("w", encoding="utf-8") as fout, \
         quarantine_path.open("w", encoding="utf-8") as fq:

        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                events = parse_langfuse_trace(raw)
                trajectory = session_to_trajectory(events[0].session_id, events)
                fout.write(trajectory.model_dump_json() + "\n")
                parsed += 1
            except (json.JSONDecodeError, ValidationError, Exception) as e:
                fq.write(json.dumps({
                    "line_number": line_no,
                    "reason": str(e),
                    "original_line_preview": line[:200],
                }) + "\n")
                quarantined += 1
                logger.warning(f"Quarantined line {line_no}: {e}")

    print("\n" + "=" * 60)
    print("  Langfuse parsing complete")
    print("=" * 60)
    print(json.dumps({
        "parsed": parsed,
        "quarantined": quarantined,
        "output": str(args.output),
        "quarantine": str(quarantine_path),
    }, indent=2))


if __name__ == "__main__":
    main()
