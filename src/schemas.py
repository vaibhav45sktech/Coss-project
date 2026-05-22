from datetime import datetime
from enum import Enum
from typing import Optional, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class EventRole(str, Enum):
    """Enumerates the possible roles for a single event in a session transcript."""

    USER = "user"
    ASSISTANT = "assistant"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    SYSTEM = "system"
    ERROR = "error"


class RawEvent(BaseModel):
    """
    Represents a single turn or action captured during a live agent session.

    Lives at the ingestion layer — raw events are written by the session recorder
    and consumed by the trajectory builder before any filtering or transformation.
    """

    event_id: str
    session_id: str
    timestamp: datetime
    role: EventRole
    content: Optional[str] = None
    tool_name: Optional[str] = None
    tool_args: Optional[dict] = None
    tool_output: Optional[dict] = None
    error_message: Optional[str] = None
    metadata: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_role_consistency(self) -> "RawEvent":
        """Enforce that each role has the fields it requires and no more."""
        role = self.role

        if role in (EventRole.USER, EventRole.ASSISTANT, EventRole.SYSTEM):
            if not self.content:
                raise ValueError(
                    f"role '{role.value}' requires a non-empty 'content' field"
                )

        elif role is EventRole.TOOL_CALL:
            if not self.tool_name or self.tool_args is None:
                raise ValueError(
                    "role 'tool_call' requires both 'tool_name' and 'tool_args'"
                )

        elif role is EventRole.TOOL_RESULT:
            if not self.tool_name or self.tool_output is None:
                raise ValueError(
                    "role 'tool_result' requires both 'tool_name' and 'tool_output'"
                )

        elif role is EventRole.ERROR:
            if not self.error_message:
                raise ValueError(
                    "role 'error' requires a non-empty 'error_message' field"
                )

        return self


class Trajectory(BaseModel):
    """
    A complete, structured sequence of RawEvents belonging to one agent session.

    Lives at the curation layer — trajectories are assembled from raw events,
    tagged with domain/complexity metadata, and stored before SFT conversion.
    """

    trajectory_id: str
    session_id: str
    events: list[RawEvent]
    tools_used: list[str]
    step_count: int = Field(ge=0)
    had_error_recovery: bool
    domain_tag: Optional[str] = None
    complexity_bucket: Literal["simple", "moderate", "complex"]
    language_mix: Literal["en", "hi", "code_mixed"]
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SFTMessage(BaseModel):
    """
    A single chat message formatted for supervised fine-tuning.

    Mirrors the OpenAI / Anthropic messages API shape so that SFTRecord objects
    can be serialised directly into training JSONL without further transformation.
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_calls: Optional[list[dict]] = None
    name: Optional[str] = None  # tool name, required when role == "tool"


class SFTRecord(BaseModel):
    """
    A fully formatted training example ready for SFT ingestion.

    Lives at the export layer — SFTRecords are the final artefact written to the
    training JSONL files consumed by the fine-tuning job.
    """

    messages: list[SFTMessage]
    metadata: dict


if __name__ == "__main__":
    # Smoke-test: construct one of each model and print it.

    raw_user = RawEvent(
        event_id="evt-001",
        session_id="sess-abc",
        timestamp=datetime(2026, 5, 22, 10, 0, 0),
        role=EventRole.USER,
        content="Search the web for the latest LLM benchmarks.",
    )
    print("RawEvent (user):")
    print(raw_user.model_dump_json(indent=2))

    raw_tool_call = RawEvent(
        event_id="evt-002",
        session_id="sess-abc",
        timestamp=datetime(2026, 5, 22, 10, 0, 1),
        role=EventRole.TOOL_CALL,
        tool_name="web_search",
        tool_args={"query": "latest LLM benchmarks 2026"},
    )

    raw_tool_result = RawEvent(
        event_id="evt-003",
        session_id="sess-abc",
        timestamp=datetime(2026, 5, 22, 10, 0, 2),
        role=EventRole.TOOL_RESULT,
        tool_name="web_search",
        tool_output={"results": ["result1", "result2"]},
    )

    raw_assistant = RawEvent(
        event_id="evt-004",
        session_id="sess-abc",
        timestamp=datetime(2026, 5, 22, 10, 0, 3),
        role=EventRole.ASSISTANT,
        content="Here are the latest LLM benchmarks I found.",
    )

    trajectory = Trajectory(
        trajectory_id="traj-001",
        session_id="sess-abc",
        events=[raw_user, raw_tool_call, raw_tool_result, raw_assistant],
        tools_used=["web_search"],
        step_count=4,
        had_error_recovery=False,
        domain_tag="research",
        complexity_bucket="moderate",
        language_mix="en",
    )
    print("\nTrajectory:")
    print(trajectory.model_dump_json(indent=2))

    sft_record = SFTRecord(
        messages=[
            SFTMessage(role="system", content="You are a helpful research assistant."),
            SFTMessage(role="user", content="Search the web for the latest LLM benchmarks."),
            SFTMessage(
                role="assistant",
                content="",
                tool_calls=[{"name": "web_search", "args": {"query": "latest LLM benchmarks 2026"}}],
            ),
            SFTMessage(
                role="tool",
                content='{"results": ["result1", "result2"]}',
                name="web_search",
            ),
            SFTMessage(role="assistant", content="Here are the latest LLM benchmarks I found."),
        ],
        metadata={"trajectory_id": "traj-001", "complexity_bucket": "moderate"},
    )
    print("\nSFTRecord:")
    print(sft_record.model_dump_json(indent=2))
