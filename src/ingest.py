import argparse
import json
import logging
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

from pydantic import ValidationError

from src.schemas import EventRole, RawEvent, Trajectory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain classification map
# ---------------------------------------------------------------------------
_TOOL_DOMAIN: dict[str, str] = {
    "get_weather": "irrigation",
    "get_soil_moisture": "irrigation",
    "irrigation_recommendation": "irrigation",
    "identify_pest": "pest",
    "recommend_treatment": "pest",
    "get_mandi_price": "price",
    "get_price_trend": "price",
    "check_landholding": "scheme",
    "verify_pm_kisan_eligibility": "scheme",
    "get_agro_climatic_zone": "calendar",
    "crop_calendar": "calendar",
}

# Hinglish markers used for language detection (matched as whole words)
_HINGLISH_MARKERS = {
    "mere", "kab", "hai", "kya", "paani", "field", "fasal",
    "kheti", "kisan", "mandi", "msp", "ka", "ki", "ke", "bhai",
    "baarish", "ganne", "tamatar", "patti", "keede",
}


# ---------------------------------------------------------------------------
# Function 1: load_raw_events
# ---------------------------------------------------------------------------

def load_raw_events(input_path: Path, quarantine_path: Path) -> Iterator[RawEvent]:
    """
    Stream-parse a JSONL file into validated RawEvent objects.

    Lines that fail JSON parsing or Pydantic validation are written to
    quarantine_path (appended) and skipped. A summary is logged after the
    file is exhausted.
    """
    # Clear quarantine file before this run
    quarantine_path.open("w").close()

    total = 0
    yielded = 0
    quarantined = 0

    with input_path.open("r", encoding="utf-8") as src, \
         quarantine_path.open("a", encoding="utf-8") as qf:

        for line_number, raw_line in enumerate(src, start=1):
            line = raw_line.strip()
            if not line:
                continue
            total += 1

            try:
                data = json.loads(line)
                event = RawEvent(**data)
                yielded += 1
                yield event

            except json.JSONDecodeError as exc:
                reason = f"JSONDecodeError: {exc}"
                logger.warning("Line %d: %s", line_number, reason)
                qf.write(json.dumps({
                    "original_line": raw_line,
                    "reason": reason,
                    "line_number": line_number,
                }) + "\n")
                quarantined += 1

            except ValidationError as exc:
                reason = f"ValidationError: {exc}"
                logger.warning("Line %d: %s", line_number, reason)
                qf.write(json.dumps({
                    "original_line": raw_line,
                    "reason": reason,
                    "line_number": line_number,
                }) + "\n")
                quarantined += 1

    logger.info(
        "load_raw_events finished — total=%d  yielded=%d  quarantined=%d",
        total, yielded, quarantined,
    )


# ---------------------------------------------------------------------------
# Function 2: group_into_sessions
# ---------------------------------------------------------------------------

def group_into_sessions(events: Iterable[RawEvent]) -> dict[str, list[RawEvent]]:
    """
    Partition events by session_id and sort each session by timestamp ascending.
    """
    sessions: dict[str, list[RawEvent]] = defaultdict(list)
    for event in events:
        sessions[event.session_id].append(event)
    for session_events in sessions.values():
        session_events.sort(key=lambda e: e.timestamp)
    return dict(sessions)


# ---------------------------------------------------------------------------
# Function 3: _classify_domain
# ---------------------------------------------------------------------------

def _classify_domain(tools_used: list[str]) -> str:
    """
    Map tool names to a domain label. Returns the domain with the most tools,
    breaking ties alphabetically. Returns 'general_qa' when no tools were used.
    """
    if not tools_used:
        return "general_qa"

    counts: dict[str, int] = defaultdict(int)
    for tool in tools_used:
        domain = _TOOL_DOMAIN.get(tool)
        if domain:
            counts[domain] += 1

    if not counts:
        return "general_qa"

    # Sort by (-count, name) so the highest count wins and ties go alphabetically
    return min(counts, key=lambda d: (-counts[d], d))


# ---------------------------------------------------------------------------
# Function 4: _detect_language
# ---------------------------------------------------------------------------

def _detect_language(events: list[RawEvent]) -> str:
    """
    Inspect user-role event content to classify the language mix.

    Priority: Devanagari characters > Hinglish markers > English.
    """
    user_contents = [
        e.content for e in events
        if e.role is EventRole.USER and e.content
    ]

    for text in user_contents:
        if any("ऀ" <= ch <= "ॿ" for ch in text):
            return "hi"

    for text in user_contents:
        words = set(text.lower().split())
        if words & _HINGLISH_MARKERS:
            return "code_mixed"

    return "en"


# ---------------------------------------------------------------------------
# Function 5: _compute_complexity
# ---------------------------------------------------------------------------

def _compute_complexity(step_count: int, had_error_recovery: bool) -> str:
    """
    Classify session complexity from tool-call count and error-recovery flag.
    """
    if step_count >= 4 or had_error_recovery:
        return "complex"
    if step_count <= 1 and not had_error_recovery:
        return "simple"
    return "moderate"


# ---------------------------------------------------------------------------
# Function 6: _detect_error_recovery
# ---------------------------------------------------------------------------

def _detect_error_recovery(events: list[RawEvent]) -> bool:
    """
    Return True if an error indicator (ERROR event or TOOL_RESULT with
    tool_output status == 'error') is followed by at least one TOOL_CALL
    later in the timeline.
    """
    for i, event in enumerate(events):
        is_error_indicator = (
            event.role is EventRole.ERROR
            or (
                event.role is EventRole.TOOL_RESULT
                and isinstance(event.tool_output, dict)
                and event.tool_output.get("status") == "error"
            )
        )
        if is_error_indicator:
            if any(e.role is EventRole.TOOL_CALL for e in events[i + 1:]):
                return True
    return False


# ---------------------------------------------------------------------------
# Function 7: session_to_trajectory
# ---------------------------------------------------------------------------

def session_to_trajectory(session_id: str, events: list[RawEvent]) -> Trajectory:
    """
    Derive a Trajectory from a sorted list of RawEvents belonging to one session.
    """
    tool_call_events = [e for e in events if e.role is EventRole.TOOL_CALL]

    # Preserve first-appearance order
    tools_used: list[str] = list(dict.fromkeys(
        e.tool_name for e in tool_call_events if e.tool_name
    ))
    step_count = len(tool_call_events)
    had_error_recovery = _detect_error_recovery(events)

    return Trajectory(
        trajectory_id=f"traj_{uuid.uuid4().hex[:12]}",
        session_id=session_id,
        events=events,
        tools_used=tools_used,
        step_count=step_count,
        had_error_recovery=had_error_recovery,
        complexity_bucket=_compute_complexity(step_count, had_error_recovery),
        domain_tag=_classify_domain(tools_used),
        language_mix=_detect_language(events),
    )


# ---------------------------------------------------------------------------
# Function 8: ingest
# ---------------------------------------------------------------------------

def ingest(
    input_path: Path,
    output_path: Path,
    quarantine_path: Path | None = None,
) -> dict:
    """
    Full ingest pipeline: parse → validate → group → build trajectories → write.

    Returns a stats dict summarising the run.
    """
    if quarantine_path is None:
        quarantine_path = output_path.parent / "quarantine.jsonl"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    quarantine_path.parent.mkdir(parents=True, exist_ok=True)

    raw_events = load_raw_events(input_path, quarantine_path)
    sessions = group_into_sessions(raw_events)

    complexity_dist: dict[str, int] = {"simple": 0, "moderate": 0, "complex": 0}
    domain_dist: dict[str, int] = defaultdict(int)
    language_dist: dict[str, int] = {"en": 0, "code_mixed": 0, "hi": 0}
    events_processed = 0

    with output_path.open("w", encoding="utf-8") as out:
        for session_id, session_events in sessions.items():
            trajectory = session_to_trajectory(session_id, session_events)
            out.write(trajectory.model_dump_json() + "\n")

            events_processed += len(session_events)
            complexity_dist[trajectory.complexity_bucket] += 1
            domain_dist[trajectory.domain_tag or "general_qa"] += 1
            language_dist[trajectory.language_mix] += 1

    quarantined_count = 0
    if quarantine_path.exists():
        with quarantine_path.open("r", encoding="utf-8") as qf:
            quarantined_count = sum(1 for line in qf if line.strip())

    stats = {
        "sessions_processed": len(sessions),
        "events_processed": events_processed,
        "quarantined": quarantined_count,
        "complexity_distribution": complexity_dist,
        "domain_distribution": dict(domain_dist),
        "language_distribution": language_dist,
    }

    logger.info("Ingest complete: %s", stats)
    return stats


# ---------------------------------------------------------------------------
# Function 9: main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Ingest raw event JSONL into trajectory JSONL."
    )
    parser.add_argument("--input", required=True, type=Path, metavar="PATH",
                        help="Path to the raw_logs.jsonl input file.")
    parser.add_argument("--output", required=True, type=Path, metavar="PATH",
                        help="Path to write trajectories.jsonl.")
    parser.add_argument("--quarantine", type=Path, default=None, metavar="PATH",
                        help="Path to write quarantined lines (default: <output-dir>/quarantine.jsonl).")
    args = parser.parse_args()

    stats = ingest(args.input, args.output, args.quarantine)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
