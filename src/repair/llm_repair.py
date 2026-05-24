"""
LLM-based trajectory repair (design + stub).

When a teacher LLM is available, this repair strategy handles cases the
rule-based repair can't:
  - Trajectories with zero tool calls answering a data-heavy question
    (no tool data to draw from for synthesis)
  - Reply structurally correct but persona/style wrong
  - Hard tool-selection errors (right query, wrong tool chosen)

The teacher gets the persona spec + the user query + available tool outputs,
and produces a corrected final reply. Optionally, it can also re-plan the
tool sequence and re-execute through the mock environment (Module 5).

Env gates (same as other LLM stubs):
    USE_LLM=1
    OPENAI_BASE_URL=http://localhost:8000/v1
    LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

from src.repair.rule_based import RepairResult


REPAIR_PROMPT = """You are an agricultural assistant persona expert.
Below is a trajectory that scored as low-quality. Your job is to produce
ONLY the corrected final assistant reply — not to re-explain the trajectory.

PERSONA SPEC:
{persona}

USER QUERY:
{user_query}

TOOL CALLS AND RESULTS:
{tool_log}

CURRENT (BAD) ASSISTANT REPLY:
{current_reply}

WHAT'S WRONG (from quality scorer):
{rationale}

Produce a corrected reply that:
  - Matches the persona's communication style and language
  - Cites tool data where available
  - Ends with a concrete actionable recommendation
  - Matches the language of the user query
  - Stays under 80 words

Respond with EXACTLY this JSON:
{{"repaired_reply": "<the new assistant reply text>"}}
"""


def _tool_log(events: list[dict]) -> str:
    lines = []
    for ev in events:
        role = ev.get("role")
        if role == "tool_call":
            args = json.dumps(ev.get("tool_args", {}), ensure_ascii=False)
            lines.append(f"  CALL: {ev.get('tool_name')}({args})")
        elif role == "tool_result":
            out = json.dumps(ev.get("tool_output", {}), ensure_ascii=False)[:200]
            lines.append(f"  RESULT: {ev.get('tool_name')} -> {out}")
        elif role == "error":
            lines.append(f"  ERROR: {ev.get('error_message', '')}")
    return "\n".join(lines) if lines else "  (no tool calls)"


class LLMRepairer:
    """Teacher-model repair. Env-gated; no-ops without LLM."""

    def __init__(self, persona_path: Path = Path("config/persona.md")):
        self.persona = persona_path.read_text(encoding="utf-8") if persona_path.exists() else ""
        self.enabled = bool(os.environ.get("USE_LLM"))

    def repair(self, trajectory: dict, rationale: str = "") -> RepairResult | None:
        if not self.enabled:
            return None

        from openai import OpenAI
        client = OpenAI(
            base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
            api_key=os.environ.get("OPENAI_API_KEY", "local"),
        )

        events = trajectory.get("events", [])
        user_event = next((e for e in events if e.get("role") == "user"), {})
        final_event = next((e for e in reversed(events) if e.get("role") == "assistant"), {})

        prompt = REPAIR_PROMPT.format(
            persona=self.persona,
            user_query=user_event.get("content", ""),
            tool_log=_tool_log(events),
            current_reply=final_event.get("content", ""),
            rationale=rationale,
        )

        response = client.chat.completions.create(
            model=os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        parsed = json.loads(response.choices[0].message.content)
        new_reply = parsed.get("repaired_reply", "")

        repaired_traj = copy.deepcopy(trajectory)
        for ev in reversed(repaired_traj["events"]):
            if ev.get("role") == "assistant":
                ev["content"] = new_reply
                break

        return RepairResult(
            repaired=True,
            trajectory=repaired_traj,
            fixes_applied=["Teacher LLM regenerated final reply"],
            unrepairable_reasons=[],
            repair_kind="llm_teacher",
        )


if __name__ == "__main__":
    print("LLMRepairer requires USE_LLM=1 and a configured LLM endpoint.")
    print("Designed and tested for import. Enable with USE_LLM=1 + endpoint config.")