"""
Trajectory builder primitives.

Takes a high-level recipe (query + tool sequence + outcome) and produces a
fully-formed Trajectory object — the same schema your existing pipeline
consumes downstream. Used by both templated and LLM-guided generators.

Three building blocks:
  - QueryRecipe: a dict describing what user asked, what tools to call,
                 what difficulty/failure modes to inject
  - build_trajectory(): turns one recipe into one Trajectory using the
                        Mock Tool Environment for grounded responses
  - DifficultyKnobs: controlled escalation primitives (extra clarification,
                     forced tool failure with recovery, multi-step chains)
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Literal

from src.schemas import EventRole, RawEvent
from src.ingest import session_to_trajectory
from src.mock_tools import MockToolExecutor, ToolResult


# ---------------------------------------------------------------------------
# Recipe + knobs
# ---------------------------------------------------------------------------

@dataclass
class ToolStep:
    """One tool call in a planned trajectory."""
    tool_name: str
    args: dict
    force_failure: str | None = None  # 'timeout' | 'no_data' | 'invalid_args' | 'upstream_error'


@dataclass
class QueryRecipe:
    """High-level plan for one trajectory.

    fields:
      user_query: the user-facing question (can contain Hinglish, etc.)
      tool_steps: ordered tool calls to execute (after which we synthesize
                  an assistant reply)
      workflow: domain tag for stratification (irrigation/pest/price/scheme/calendar)
      add_clarification: if True, inject an assistant clarify-question and a user follow-up
                         between the initial question and the first tool call
      add_recovery: if True, the FIRST tool step is forced to fail, then retried (creates
                    a recover-from-error trajectory)
      language: "en" | "hi" | "code_mixed"
      assistant_template: optional final answer template (overrides default)
    """
    user_query: str
    tool_steps: list[ToolStep]
    workflow: str
    add_clarification: bool = False
    clarification_question: str | None = None
    user_clarification_reply: str | None = None
    add_recovery: bool = False
    language: str = "en"
    assistant_template: str | None = None
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _next_event(session_id: str, role: EventRole, ts: datetime, **kwargs) -> RawEvent:
    return RawEvent(
        event_id=str(uuid.uuid4()),
        session_id=session_id,
        timestamp=ts,
        role=role,
        **kwargs,
    )


def _format_assistant_reply(recipe: QueryRecipe, tool_results: list[ToolResult]) -> str:
    """Build a plausible assistant reply from the tool results.

    If the recipe provides an `assistant_template`, use it (with simple
    string formatting). Otherwise synthesize one based on tool outputs.
    """
    if recipe.assistant_template:
        # Best-effort template fill; ignore missing keys
        try:
            ctx = {}
            for r in tool_results:
                if r.data:
                    ctx.update(r.data)
            return recipe.assistant_template.format(**ctx)
        except (KeyError, IndexError):
            pass

    # Default synthesized reply
    successful = [r for r in tool_results if r.status == "ok"]
    if not successful:
        return "I wasn't able to fetch the data needed to answer that. Please try again later."

    parts = []
    for r in successful:
        if r.tool_name == "mandi_prices" and r.data:
            parts.append(
                f"In {r.data['market']}, {r.data['crop']} is currently "
                f"₹{r.data['price_per_quintal']}/quintal ({r.data.get('trend_7d', 'stable')} trend)."
            )
        elif r.tool_name == "weather_forecast" and r.data:
            total_rain = r.data.get("total_rain_mm", 0)
            parts.append(
                f"The forecast shows {total_rain}mm total rain over the next "
                f"{len(r.data.get('forecast', []))} days."
            )
        elif r.tool_name == "soil_test_report" and r.data:
            parts.append(f"Soil pH is {r.data['ph']}. Recommendation: {r.data['recommendation']}")
        elif r.tool_name == "govt_scheme_lookup" and r.data:
            verdict = "eligible" if r.data.get("eligible") else "not eligible"
            parts.append(
                f"You appear {verdict} for {r.data['scheme']} — {r.data.get('reason', '')}."
            )
        elif r.tool_name == "fetch_agristack_data" and r.data:
            parts.append(
                f"Found Agristack record: {r.data.get('district', 'N/A')} district, "
                f"{r.data.get('land_area_hectares', 'N/A')} hectares."
            )

    return " ".join(parts) if parts else "Here are the details based on the tool results."


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_trajectory(
    recipe: QueryRecipe,
    executor: MockToolExecutor,
    base_timestamp: datetime | None = None,
):
    """Build one Trajectory from a recipe by executing tool steps through the mock executor.

    Returns a Trajectory object ready to write as JSONL.
    """
    session_id = str(uuid.uuid4())
    ts = base_timestamp or datetime.now(timezone.utc)
    tick = 0

    def next_ts() -> datetime:
        nonlocal tick
        tick += 1
        return ts + timedelta(seconds=tick * 3)

    events: list[RawEvent] = []
    tool_results: list[ToolResult] = []

    # System prompt
    events.append(_next_event(
        session_id, EventRole.SYSTEM, next_ts(),
        content=(
            "You are a helpful agricultural assistant for Indian farmers. "
            f"Current workflow: {recipe.workflow}. "
            "Answer in simple language and recommend practical actions."
        ),
    ))

    # User question
    events.append(_next_event(
        session_id, EventRole.USER, next_ts(),
        content=recipe.user_query,
    ))

    # Optional clarification round
    if recipe.add_clarification:
        events.append(_next_event(
            session_id, EventRole.ASSISTANT, next_ts(),
            content=recipe.clarification_question or "Could you give me a bit more detail?",
        ))
        events.append(_next_event(
            session_id, EventRole.USER, next_ts(),
            content=recipe.user_clarification_reply or "Sure, here's more context.",
        ))

    # Tool steps
    steps = list(recipe.tool_steps)

    # If recovery requested, force-fail the first step then retry it successfully
    if recipe.add_recovery and steps:
        first = steps[0]
        # Failed attempt
        events.append(_next_event(
            session_id, EventRole.TOOL_CALL, next_ts(),
            tool_name=first.tool_name,
            tool_args=first.args,
        ))
        failure_result = executor.execute(
            first.tool_name, first.args, force_failure="upstream_error"
        )
        tool_results.append(failure_result)
        events.append(_next_event(
            session_id, EventRole.TOOL_RESULT, next_ts(),
            tool_name=first.tool_name,
            tool_output=failure_result.to_dict(),
        ))
        # Explicit ERROR marker so domain classifier sees recovery pattern
        events.append(_next_event(
            session_id, EventRole.ERROR, next_ts(),
            error_message=f"Tool '{first.tool_name}' failed: {failure_result.error_message}",
        ))
        # Retry the same tool successfully
        events.append(_next_event(
            session_id, EventRole.TOOL_CALL, next_ts(),
            tool_name=first.tool_name,
            tool_args={**first.args, "_retry": True},
        ))
        retry_result = executor.execute(first.tool_name, first.args)
        tool_results.append(retry_result)
        events.append(_next_event(
            session_id, EventRole.TOOL_RESULT, next_ts(),
            tool_name=first.tool_name,
            tool_output=retry_result.to_dict(),
        ))
        # Remaining tool steps execute normally
        remaining = steps[1:]
    else:
        remaining = steps

    for step in remaining:
        events.append(_next_event(
            session_id, EventRole.TOOL_CALL, next_ts(),
            tool_name=step.tool_name,
            tool_args=step.args,
        ))
        result = executor.execute(
            step.tool_name, step.args, force_failure=step.force_failure
        )
        tool_results.append(result)
        events.append(_next_event(
            session_id, EventRole.TOOL_RESULT, next_ts(),
            tool_name=step.tool_name,
            tool_output=result.to_dict(),
        ))

    # Final assistant reply
    reply = _format_assistant_reply(recipe, tool_results)
    events.append(_next_event(
        session_id, EventRole.ASSISTANT, next_ts(),
        content=reply,
        metadata={**recipe.metadata, "language_mix": recipe.language},
    ))

    trajectory = session_to_trajectory(session_id, events)
    # Stamp synthesis source so the dataset can be filtered by origin
    trajectory.events[-1].metadata["synthesis_source"] = recipe.metadata.get(
        "synthesis_source", "templated"
    )
    return trajectory