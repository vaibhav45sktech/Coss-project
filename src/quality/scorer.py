"""
Heuristic quality scorer for trajectories.

Scores three dimensions, each 0.0-1.0:
  - persona_adherence: rules-based check against config/persona.md
  - tool_efficiency:   step count + retry patterns + tool diversity
  - goal_completion:   final reply addresses the user query and uses tool data

Each score has a rationale list explaining why it landed where it did.
The composite score is a weighted average; weights are tunable.

Heuristics intentionally err on the side of being noisy but FAST and
EXPLAINABLE. Use the LLMJudge in src/quality/llm_judge.py for higher-fidelity
scoring when a local model is available.

Usage:
    from src.quality.scorer import HeuristicScorer
    scorer = HeuristicScorer.load()
    report = scorer.score(trajectory_dict)
    print(report.composite_score, report.persona_adherence.score)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class DimensionScore:
    """One dimension's score with rationale."""
    score: float
    rationale: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"score": round(self.score, 3), "rationale": self.rationale}


@dataclass
class QualityReport:
    """Composite quality assessment for one trajectory."""
    trajectory_id: str
    persona_adherence: DimensionScore
    tool_efficiency: DimensionScore
    goal_completion: DimensionScore
    composite_score: float
    quality_label: Literal["high", "medium", "low"]
    weights: dict
    scorer_kind: str = "heuristic"

    def to_dict(self) -> dict:
        return {
            "trajectory_id": self.trajectory_id,
            "persona_adherence": self.persona_adherence.to_dict(),
            "tool_efficiency": self.tool_efficiency.to_dict(),
            "goal_completion": self.goal_completion.to_dict(),
            "composite_score": round(self.composite_score, 3),
            "quality_label": self.quality_label,
            "weights": self.weights,
            "scorer_kind": self.scorer_kind,
        }


# ---------------------------------------------------------------------------
# Heuristic patterns
# ---------------------------------------------------------------------------

# Hard rule violation detectors — match against the final assistant reply
USD_PATTERN = re.compile(r"\$\s?\d|USD\b|dollars?\b", re.IGNORECASE)
INVENTED_NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\s*(?:rupees|₹/quintal|₹/kg|inr/quintal)\b", re.IGNORECASE)

# Persona-positive markers
CITES_TOOL_MARKERS = [
    "based on", "according to", "the data shows", "mandi data",
    "weather forecast", "soil report", "agristack", "scheme database",
]
ACTION_MARKERS = [
    "recommend", "you should", "you can", "try", "apply", "irrigate",
    "spray", "register", "sow", "plant", "harvest", "consult",
]

# Anti-pattern: "I don't know" replies
DUNNO_MARKERS = [
    "i don't know", "i cannot help", "i am not sure", "unable to assist",
    "no information available", "i can't answer",
]

# Language-mismatch heuristic: Hindi script user vs Latin-only reply
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


# ---------------------------------------------------------------------------
# Heuristic scorer
# ---------------------------------------------------------------------------

class HeuristicScorer:
    """Fast rules-based scorer with explainable dimension scores."""

    DEFAULT_WEIGHTS = {
        "persona_adherence": 0.4,
        "tool_efficiency": 0.3,
        "goal_completion": 0.3,
    }

    def __init__(
        self,
        persona_text: str,
        weights: dict | None = None,
        ideal_step_range: tuple[int, int] = (1, 4),
    ):
        self.persona_text = persona_text
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.ideal_min, self.ideal_max = ideal_step_range

    @classmethod
    def load(cls, persona_path: Path = Path("config/persona.md")) -> "HeuristicScorer":
        if not persona_path.exists():
            raise SystemExit(f"Persona file not found: {persona_path}. Create it first.")
        return cls(persona_text=persona_path.read_text(encoding="utf-8"))

    # ---- per-dimension scoring ----

    def score_persona_adherence(self, trajectory: dict) -> DimensionScore:
        """Check the final assistant reply against persona rules.

        Penalize: USD currency, language mismatch, missing tool citation,
                  missing action, "I don't know" replies.
        Reward:   proper currency, action verbs, tool citation, language match.
        """
        rationale: list[str] = []
        score = 1.0

        events = trajectory.get("events", [])
        user_events = [e for e in events if e.get("role") == "user" and e.get("content")]
        assistant_events = [e for e in events if e.get("role") == "assistant" and e.get("content")]

        if not assistant_events:
            return DimensionScore(0.0, ["No assistant reply present"])

        final_reply = assistant_events[-1].get("content", "")
        user_query = user_events[0].get("content", "") if user_events else ""

        # Penalty: uses USD or wrong currency
        if USD_PATTERN.search(final_reply):
            score -= 0.3
            rationale.append("Uses USD / dollar phrasing instead of ₹")

        # Penalty: "I don't know" without trying
        if any(m in final_reply.lower() for m in DUNNO_MARKERS):
            tool_calls_count = sum(1 for e in events if e.get("role") == "tool_call")
            if tool_calls_count == 0:
                score -= 0.5
                rationale.append("Says 'don't know' without attempting any tool call")
            else:
                score -= 0.2
                rationale.append("Concedes ignorance after partial tool use")

        # Penalty: language mismatch (Devanagari user, no Devanagari in reply)
        user_has_devanagari = bool(DEVANAGARI_RE.search(user_query))
        reply_has_devanagari = bool(DEVANAGARI_RE.search(final_reply))
        if user_has_devanagari and not reply_has_devanagari:
            # Always penalize language mismatch; the magnitude scales with reply length
            # so a very short reply isn't dinged as hard as a long English wall of text.
            penalty = 0.4 if len(final_reply) > 50 else 0.2
            score -= penalty
            rationale.append(
                f"User asked in Devanagari, reply is English-only (penalty {penalty})"
            )

        # Bonus: cites a tool source
        if any(m in final_reply.lower() for m in CITES_TOOL_MARKERS):
            rationale.append("Cites tool source explicitly")
        else:
            # Only penalize if tool calls actually happened
            if any(e.get("role") == "tool_call" for e in events):
                score -= 0.15
                rationale.append("Used tools but did not cite source in reply")

        # Bonus: contains actionable language
        if any(m in final_reply.lower() for m in ACTION_MARKERS):
            rationale.append("Contains actionable recommendation")
        elif len(final_reply) > 100:
            score -= 0.1
            rationale.append("Long reply without clear action verb")

        return DimensionScore(max(0.0, min(1.0, score)), rationale)

    def score_tool_efficiency(self, trajectory: dict) -> DimensionScore:
        """Score how efficiently tools were used to answer the query."""
        rationale: list[str] = []
        events = trajectory.get("events", [])
        tool_calls = [e for e in events if e.get("role") == "tool_call"]
        step_count = len(tool_calls)

        if step_count == 0:
            rationale.append("Zero tool calls — likely hallucinated answer")
            return DimensionScore(0.2, rationale)

        # Ideal range scoring (1-4 tools, per persona)
        if self.ideal_min <= step_count <= self.ideal_max:
            score = 1.0
            rationale.append(f"Step count {step_count} within ideal range ({self.ideal_min}-{self.ideal_max})")
        elif step_count <= self.ideal_max + 2:
            score = 0.7
            rationale.append(f"Step count {step_count} slightly above ideal max")
        else:
            score = 0.4
            rationale.append(f"Step count {step_count} significantly above ideal — likely inefficient")

        # Penalty: same tool called >2 times in immediate succession (retry loop)
        consecutive_same = 1
        max_consecutive = 1
        for i in range(1, len(tool_calls)):
            if tool_calls[i].get("tool_name") == tool_calls[i - 1].get("tool_name"):
                consecutive_same += 1
                max_consecutive = max(max_consecutive, consecutive_same)
            else:
                consecutive_same = 1
        if max_consecutive >= 3:
            # Retry loop is a strong signal of broken planning — heavy penalty
            score -= 0.5
            rationale.append(f"Same tool called {max_consecutive}× in a row (retry loop)")
        elif max_consecutive == 2:
            # Two in a row could be legitimate (retry after no-data), small penalty
            score -= 0.1
            rationale.append("Same tool called twice consecutively (possible retry)")

        # Penalty: error events not followed by recovery
        had_error = any(e.get("role") == "error" for e in events)
        had_recovery = trajectory.get("had_error_recovery", False)
        if had_error and not had_recovery:
            score -= 0.2
            rationale.append("Errors occurred without recovery attempts")

        # Bonus: error recovery shows resilience
        if had_recovery:
            rationale.append("Successfully recovered from tool error")

        # Penalty: unique tool diversity vs total calls (high duplication)
        unique_tools = len({tc.get("tool_name") for tc in tool_calls})
        if step_count > 3 and unique_tools / step_count < 0.5:
            score -= 0.15
            rationale.append(f"Low tool diversity: {unique_tools} unique / {step_count} calls")

        return DimensionScore(max(0.0, min(1.0, score)), rationale)

    def score_goal_completion(self, trajectory: dict) -> DimensionScore:
        """Score whether the final reply actually addresses the user's question.

        Heuristic proxy: extract key nouns from the user query, check whether
        the assistant reply mentions them or references tool data that would
        have answered them.
        """
        rationale: list[str] = []
        events = trajectory.get("events", [])
        user_events = [e for e in events if e.get("role") == "user" and e.get("content")]
        assistant_events = [e for e in events if e.get("role") == "assistant" and e.get("content")]
        tool_results = [e for e in events if e.get("role") == "tool_result"]

        if not user_events:
            return DimensionScore(0.5, ["No user query to compare against"])
        if not assistant_events:
            return DimensionScore(0.0, ["No assistant reply"])

        user_query = user_events[0].get("content", "").lower()
        final_reply = assistant_events[-1].get("content", "").lower()

        # Extract noun-like tokens from the user query (>3 letters, not stopwords)
        STOPWORDS = {
            "what", "when", "where", "why", "how", "can", "should",
            "the", "and", "for", "with", "from", "have", "this", "that",
            "today", "tomorrow", "please", "tell", "give", "need",
            "are", "you", "your", "i", "me", "my", "is",
        }
        query_tokens = {
            t for t in re.findall(r"\b[a-z]{4,}\b", user_query)
            if t not in STOPWORDS
        }

        if not query_tokens:
            # No English content tokens in query — likely Devanagari/Indic.
            # Fall back to reply substance: short generic reply = bad, substantive = okay.
            word_count = len(final_reply.split())
            has_dunno = any(m in final_reply.lower() for m in DUNNO_MARKERS)
            if has_dunno:
                score = 0.1
                rationale.append("Non-Latin query met with 'don't know' reply")
            elif word_count < 8:
                score = 0.3
                rationale.append("Non-Latin query with very short reply (likely incomplete)")
            else:
                score = 0.7
                rationale.append("Non-Latin query with substantive reply (cannot verify token overlap)")
            return DimensionScore(score, rationale)

        # Token overlap with reply
        reply_tokens = set(re.findall(r"\b[a-z]{4,}\b", final_reply))
        overlap = query_tokens & reply_tokens
        overlap_ratio = len(overlap) / len(query_tokens)

        score = min(1.0, 0.3 + overlap_ratio * 0.7)
        rationale.append(
            f"Reply covers {len(overlap)}/{len(query_tokens)} content tokens from query"
        )

        # Penalty: tool returned data but reply is suspiciously short
        if tool_results and len(final_reply.split()) < 10:
            score -= 0.2
            rationale.append("Tools returned data but reply is very short")

        # Bonus: reply length proportional to query complexity
        if len(final_reply.split()) >= 15 and overlap_ratio >= 0.3:
            rationale.append("Reply length and coverage look proportional to query")

        return DimensionScore(max(0.0, min(1.0, score)), rationale)

    # ---- composite ----

    def _composite(self, persona: DimensionScore, efficiency: DimensionScore, completion: DimensionScore) -> float:
        return (
            persona.score * self.weights["persona_adherence"]
            + efficiency.score * self.weights["tool_efficiency"]
            + completion.score * self.weights["goal_completion"]
        )

    @staticmethod
    def _label(composite: float) -> Literal["high", "medium", "low"]:
        if composite >= 0.75:
            return "high"
        if composite >= 0.5:
            return "medium"
        return "low"

    def score(self, trajectory: dict) -> QualityReport:
        persona = self.score_persona_adherence(trajectory)
        efficiency = self.score_tool_efficiency(trajectory)
        completion = self.score_goal_completion(trajectory)
        composite = self._composite(persona, efficiency, completion)
        return QualityReport(
            trajectory_id=trajectory.get("trajectory_id", "unknown"),
            persona_adherence=persona,
            tool_efficiency=efficiency,
            goal_completion=completion,
            composite_score=composite,
            quality_label=self._label(composite),
            weights=self.weights,
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    scorer = HeuristicScorer.load()

    # Build a fake good trajectory
    good = {
        "trajectory_id": "test_good",
        "events": [
            {"role": "user", "content": "What's the wheat price in Nashik today?"},
            {"role": "tool_call", "tool_name": "mandi_prices"},
            {"role": "tool_result", "tool_name": "mandi_prices"},
            {"role": "assistant", "content": "Based on mandi data, wheat in Nashik is ₹2150/quintal. You should sell today if prices are rising — recommend checking trend before final decision."},
        ],
        "had_error_recovery": False,
    }

    # Bad trajectory: too many tools, language mismatch, no action
    bad = {
        "trajectory_id": "test_bad",
        "events": [
            {"role": "user", "content": "मेरे खेत में पानी की ज़रूरत है क्या?"},
            {"role": "tool_call", "tool_name": "weather_forecast"},
            {"role": "tool_result", "tool_name": "weather_forecast"},
            {"role": "tool_call", "tool_name": "weather_forecast"},
            {"role": "tool_result", "tool_name": "weather_forecast"},
            {"role": "tool_call", "tool_name": "weather_forecast"},
            {"role": "tool_result", "tool_name": "weather_forecast"},
            {"role": "tool_call", "tool_name": "soil_test_report"},
            {"role": "tool_result", "tool_name": "soil_test_report"},
            {"role": "assistant", "content": "I don't know."},
        ],
        "had_error_recovery": False,
    }

    print("=" * 60)
    print("GOOD trajectory:")
    print(json.dumps(scorer.score(good).to_dict(), indent=2, ensure_ascii=False))
    print()
    print("BAD trajectory:")
    print(json.dumps(scorer.score(bad).to_dict(), indent=2, ensure_ascii=False))