"""
LLM-judge quality scorer for trajectories (design + stub).

When a local LLM is available, this judge:
  1. Reads the persona spec from config/persona.md
  2. Builds a structured prompt with the trajectory + persona
  3. Asks the LLM to score persona / efficiency / completion (0-1 each)
     and produce a 1-sentence rationale per dimension
  4. Parses the JSON response into a QualityReport

Higher fidelity than heuristics on subtle persona issues but slower and
costlier. Use in batches over the heuristic scorer's "medium" tier — the
high and low tiers are usually obvious; the medium tier needs judgment.

Env gates (same as src/synthesis/llm_guided):
    USE_LLM=1
    OPENAI_BASE_URL=http://localhost:8000/v1
    LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from src.quality.scorer import DimensionScore, QualityReport

logger = logging.getLogger("quality_llm_judge")


JUDGE_PROMPT = """You are scoring an agricultural assistant trajectory against a persona spec.

PERSONA SPEC:
{persona}

TRAJECTORY (user query, tool calls, final reply):
{trajectory_text}

Score the trajectory on three dimensions, each 0.0-1.0:
  - persona_adherence: Does the reply match the persona's communication style,
    language matching, currency conventions, and hard rules?
  - tool_efficiency: Were tools used minimally and appropriately?
  - goal_completion: Does the final reply actually answer the user's question?

Respond with EXACTLY this JSON shape, no other text:
{{
  "persona_adherence": {{"score": <0-1>, "rationale": "<1 sentence>"}},
  "tool_efficiency": {{"score": <0-1>, "rationale": "<1 sentence>"}},
  "goal_completion": {{"score": <0-1>, "rationale": "<1 sentence>"}}
}}
"""


def _trajectory_to_text(trajectory: dict) -> str:
    """Render a trajectory as compact prompt-ready text."""
    lines = []
    for ev in trajectory.get("events", []):
        role = ev.get("role")
        if role == "user":
            lines.append(f"USER: {ev.get('content', '')}")
        elif role == "assistant":
            lines.append(f"ASSISTANT: {ev.get('content', '')}")
        elif role == "tool_call":
            args = json.dumps(ev.get("tool_args", {}), ensure_ascii=False)
            lines.append(f"TOOL_CALL: {ev.get('tool_name')}({args})")
        elif role == "tool_result":
            out = json.dumps(ev.get("tool_output", {}), ensure_ascii=False)[:200]
            lines.append(f"TOOL_RESULT: {ev.get('tool_name')} -> {out}")
        elif role == "error":
            lines.append(f"ERROR: {ev.get('error_message', '')}")
    return "\n".join(lines)


class LLMJudge:
    """LLM-based quality scorer. Env-gated; gracefully no-ops without LLM."""

    def __init__(self, persona_path: Path = Path("config/persona.md")):
        self.persona = persona_path.read_text(encoding="utf-8")
        self.enabled = bool(os.environ.get("USE_LLM"))

    def score(self, trajectory: dict) -> QualityReport | None:
        if not self.enabled:
            return None

        from openai import OpenAI
        client = OpenAI(
            base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
            api_key=os.environ.get("OPENAI_API_KEY", "local"),
        )
        prompt = JUDGE_PROMPT.format(
            persona=self.persona,
            trajectory_text=_trajectory_to_text(trajectory),
        )
        response = client.chat.completions.create(
            model=os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        parsed = json.loads(response.choices[0].message.content)

        def dim(key: str) -> DimensionScore:
            block = parsed.get(key, {"score": 0.5, "rationale": "no response"})
            return DimensionScore(
                score=float(block["score"]),
                rationale=[block.get("rationale", "")],
            )

        persona = dim("persona_adherence")
        efficiency = dim("tool_efficiency")
        completion = dim("goal_completion")
        weights = {"persona_adherence": 0.4, "tool_efficiency": 0.3, "goal_completion": 0.3}
        composite = (
            persona.score * weights["persona_adherence"]
            + efficiency.score * weights["tool_efficiency"]
            + completion.score * weights["goal_completion"]
        )
        label = "high" if composite >= 0.75 else ("medium" if composite >= 0.5 else "low")
        return QualityReport(
            trajectory_id=trajectory.get("trajectory_id", "unknown"),
            persona_adherence=persona,
            tool_efficiency=efficiency,
            goal_completion=completion,
            composite_score=composite,
            quality_label=label,
            weights=weights,
            scorer_kind="llm_judge",
        )


if __name__ == "__main__":
    print("LLMJudge requires USE_LLM=1 and a configured endpoint to run.")
    print("Module imports cleanly without GPU; integration is one env var away.")
