"""
Rule-based trajectory repair.

Takes a low-quality trajectory and applies targeted fixes based on the
quality scorer's rationale. Produces a "repaired" copy of the trajectory
suitable for use as the "chosen" half of a DPO pair (with the original
as "rejected").

Fix strategies:
  - 'unable_to_X' / 'i don't know' reply → synthesize a substantive reply
    from available tool outputs
  - Missing tool citation → prepend "Based on tool data..." framing
  - Retry loop (same tool >2× consecutively) → trim to first successful call
  - Language mismatch (Devanagari user → English reply) → add Devanagari
    prefix acknowledging the query language
  - Trajectory with zero tool calls but data-heavy query → flagged as
    unrepairable by rules; teacher LLM needed (Module 9.2 stub)

Each repair returns the modified trajectory AND a list of fixes applied,
so DPO export can include the rationale as preference metadata.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class RepairResult:
    """Outcome of an attempt to repair a trajectory."""
    repaired: bool                 # True if any fix was applied
    trajectory: dict               # The (possibly modified) trajectory
    fixes_applied: list[str]       # Human-readable list of what was fixed
    unrepairable_reasons: list[str]  # Why some failures couldn't be fixed
    repair_kind: str = "rule_based"


# ---------------------------------------------------------------------------
# Repair patterns
# ---------------------------------------------------------------------------

DUNNO_PATTERNS = [
    re.compile(r"i don't know\.?", re.IGNORECASE),
    re.compile(r"i cannot help", re.IGNORECASE),
    re.compile(r"unable to process", re.IGNORECASE),
    re.compile(r"unable to", re.IGNORECASE),
    re.compile(r"no information available", re.IGNORECASE),
]

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


def _final_assistant_event(events: list[dict]) -> dict | None:
    """Return the last assistant event in a trajectory (mutable reference)."""
    for ev in reversed(events):
        if ev.get("role") == "assistant":
            return ev
    return None


def _first_user_event(events: list[dict]) -> dict | None:
    for ev in events:
        if ev.get("role") == "user":
            return ev
    return None


def _successful_tool_results(events: list[dict]) -> list[dict]:
    """Tool result events where the tool succeeded (status='ok' or no status)."""
    out = []
    for ev in events:
        if ev.get("role") != "tool_result":
            continue
        output = ev.get("tool_output") or {}
        status = output.get("status", "ok")
        if status == "ok":
            out.append(ev)
    return out


# ---------------------------------------------------------------------------
# Individual repair strategies
# ---------------------------------------------------------------------------

def _repair_dunno_reply(trajectory: dict) -> tuple[bool, str]:
    """If the final reply is 'I don't know' but tool data exists, synthesize
    a substantive reply from the successful tool outputs."""
    events = trajectory["events"]
    final = _final_assistant_event(events)
    if not final:
        return False, ""

    content = final.get("content", "")
    if not any(p.search(content) for p in DUNNO_PATTERNS):
        return False, ""

    successes = _successful_tool_results(events)
    if not successes:
        # Can't repair — no tool data to draw from
        return False, "Reply is 'don't know' AND no successful tool calls"

    # Synthesize a reply from the tool results
    parts = ["Based on the available data:"]
    for r in successes:
        tool_name = r.get("tool_name", "")
        output = r.get("tool_output") or {}
        if tool_name == "mandi_prices" and output.get("price_per_quintal"):
            parts.append(
                f"{output.get('crop', 'the crop')} in {output.get('market', 'the market')} "
                f"is priced at ₹{output['price_per_quintal']}/quintal."
            )
        elif tool_name == "weather_forecast" and output.get("forecast"):
            total = output.get("total_rain_mm", 0)
            days = len(output.get("forecast", []))
            parts.append(f"The {days}-day forecast shows {total}mm total rain.")
        elif tool_name == "soil_test_report" and output.get("ph"):
            parts.append(f"Soil pH is {output['ph']}. {output.get('recommendation', '')}")
        elif tool_name == "govt_scheme_lookup" and "eligible" in output:
            verdict = "eligible" if output["eligible"] else "not eligible"
            parts.append(f"You are {verdict} for {output.get('scheme', 'this scheme')}.")
        elif tool_name == "fetch_agristack_data" and output.get("district"):
            parts.append(
                f"Agristack confirms {output.get('land_area_hectares', 'N/A')} hectares "
                f"in {output.get('district', '')}."
            )

    if len(parts) == 1:
        # Nothing useful to assemble — unrepairable
        return False, "Tool data available but no actionable fields recognized"

    parts.append("Recommend acting on these specifics for your situation.")
    final["content"] = " ".join(parts)
    return True, "Synthesized substantive reply from tool outputs (was 'don't know')"


def _repair_missing_citation(trajectory: dict) -> tuple[bool, str]:
    """If tools were called but reply doesn't cite them, prepend citation framing."""
    events = trajectory["events"]
    final = _final_assistant_event(events)
    if not final:
        return False, ""
    content = final.get("content", "")
    if not content:
        return False, ""

    cite_markers = ["based on", "according to", "the data shows", "tool data"]
    if any(m in content.lower() for m in cite_markers):
        return False, ""

    tool_calls = [e for e in events if e.get("role") == "tool_call"]
    if not tool_calls:
        return False, ""

    # Prepend citation framing
    final["content"] = "Based on the tool data: " + content
    return True, "Added explicit tool citation framing"


def _repair_retry_loop(trajectory: dict) -> tuple[bool, str]:
    """If the same tool is called 3+ times in a row, trim to first successful call.

    This rewrites the events list, which means the trajectory's metadata
    (tools_used, step_count) becomes stale; downstream re-derivation is needed.
    We flag the trajectory with `_metadata_stale: True` so callers know.
    """
    events = trajectory["events"]
    tool_calls = [(i, e) for i, e in enumerate(events) if e.get("role") == "tool_call"]
    if len(tool_calls) < 3:
        return False, ""

    # Find consecutive runs of same tool_name
    runs = []
    start = 0
    for i in range(1, len(tool_calls)):
        if tool_calls[i][1].get("tool_name") != tool_calls[i - 1][1].get("tool_name"):
            runs.append((start, i - 1))
            start = i
    runs.append((start, len(tool_calls) - 1))

    long_runs = [r for r in runs if r[1] - r[0] + 1 >= 3]
    if not long_runs:
        return False, ""

    # For each long run, find the first successful call's index in events
    indices_to_drop = set()
    for run_start, run_end in long_runs:
        # First successful tool_result for this tool name within the run window
        tool_name = tool_calls[run_start][1].get("tool_name")
        first_success_event_idx = None
        for idx in range(tool_calls[run_start][0], tool_calls[run_end][0] + 2):
            if idx >= len(events):
                break
            ev = events[idx]
            if (ev.get("role") == "tool_result"
                and ev.get("tool_name") == tool_name
                and (ev.get("tool_output") or {}).get("status", "ok") == "ok"):
                first_success_event_idx = idx
                break
        # If we found a success, drop everything after first success up to last in run
        if first_success_event_idx is not None:
            last_event_idx_in_run = tool_calls[run_end][0]
            for i in range(first_success_event_idx + 1, last_event_idx_in_run + 1):
                indices_to_drop.add(i)

    if not indices_to_drop:
        return False, ""

    new_events = [e for i, e in enumerate(events) if i not in indices_to_drop]
    trajectory["events"] = new_events
    trajectory["_metadata_stale"] = True
    dropped = len(events) - len(new_events)
    return True, f"Trimmed {dropped} events from retry loop (kept first successful call)"


def _repair_language_mismatch(trajectory: dict) -> tuple[bool, str]:
    """If user used Devanagari but reply is English, add Devanagari acknowledgment prefix."""
    events = trajectory["events"]
    user = _first_user_event(events)
    final = _final_assistant_event(events)
    if not (user and final):
        return False, ""

    user_text = user.get("content", "") or ""
    reply = final.get("content", "") or ""
    if not user_text or not reply:
        return False, ""

    if not DEVANAGARI_RE.search(user_text):
        return False, ""
    if DEVANAGARI_RE.search(reply):
        return False, ""

    # Add a brief Devanagari opener acknowledging the question
    final["content"] = "आपके प्रश्न के अनुसार: " + reply
    return True, "Added Devanagari opener to address language mismatch"


def _refresh_metadata(trajectory: dict) -> None:
    """Recompute trajectory metadata if events were modified.

    Lightweight version that only updates the fields downstream actually
    reads: tools_used, step_count, had_error_recovery.
    """
    if not trajectory.pop("_metadata_stale", False):
        return

    events = trajectory["events"]
    tool_calls = [e for e in events if e.get("role") == "tool_call"]

    # Tools used, order-preserving dedup
    seen = []
    for tc in tool_calls:
        name = tc.get("tool_name")
        if name and name not in seen:
            seen.append(name)
    trajectory["tools_used"] = seen
    trajectory["step_count"] = len(tool_calls)

    # Error recovery: an ERROR event with a later tool_call
    error_indices = [i for i, e in enumerate(events) if e.get("role") == "error"]
    later_tool_call = any(
        any(e.get("role") == "tool_call" for e in events[idx + 1:])
        for idx in error_indices
    )
    trajectory["had_error_recovery"] = bool(error_indices) and later_tool_call


# ---------------------------------------------------------------------------
# Top-level repair function
# ---------------------------------------------------------------------------

def repair_trajectory(trajectory: dict) -> RepairResult:
    """Apply all rule-based repairs to a trajectory.

    Returns a deep copy with fixes applied. Tracks what was fixed and what
    couldn't be repaired (for the DPO export to record as metadata).
    """
    copy_traj = copy.deepcopy(trajectory)
    fixes: list[str] = []
    unrepairable: list[str] = []

    repairs = [
        _repair_dunno_reply,
        _repair_retry_loop,
        _repair_missing_citation,
        _repair_language_mismatch,
    ]

    for repair_fn in repairs:
        changed, message = repair_fn(copy_traj)
        if changed:
            fixes.append(message)
        elif message:
            unrepairable.append(message)

    _refresh_metadata(copy_traj)

    return RepairResult(
        repaired=bool(fixes),
        trajectory=copy_traj,
        fixes_applied=fixes,
        unrepairable_reasons=unrepairable,
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    bad_trajectory = {
        "trajectory_id": "test_bad",
        "tools_used": ["weather_forecast"],
        "step_count": 3,
        "had_error_recovery": False,
        "events": [
            {"role": "user", "content": "मेरे खेत में पानी की ज़रूरत है क्या?"},
            {"role": "tool_call", "tool_name": "weather_forecast", "tool_args": {"latitude": 19.0, "longitude": 73.0, "days": 3}},
            {"role": "tool_result", "tool_name": "weather_forecast", "tool_output": {"status": "ok", "forecast": [{"day": 1, "rain_mm": 5}], "total_rain_mm": 5}},
            {"role": "tool_call", "tool_name": "weather_forecast", "tool_args": {}},
            {"role": "tool_result", "tool_name": "weather_forecast", "tool_output": {"status": "ok", "total_rain_mm": 5, "forecast": [{"day": 1, "rain_mm": 5}]}},
            {"role": "tool_call", "tool_name": "weather_forecast", "tool_args": {}},
            {"role": "tool_result", "tool_name": "weather_forecast", "tool_output": {"status": "ok", "total_rain_mm": 5, "forecast": [{"day": 1, "rain_mm": 5}]}},
            {"role": "assistant", "content": "I don't know."},
        ],
    }

    result = repair_trajectory(bad_trajectory)
    print(f"Repaired: {result.repaired}")
    print(f"Fixes applied: {result.fixes_applied}")
    print(f"Unrepairable: {result.unrepairable_reasons}")
    print(f"New step_count: {result.trajectory['step_count']}")
    print(f"Final reply: {result.trajectory['events'][-1]['content']}")