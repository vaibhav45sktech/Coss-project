"""
Pydantic models matching the Langfuse/Pydantic-AI trace format used by
the mentor's production OpenAgriNet bot.

Reference schema (from the mentor's GitHub comment):
{
    "user_question": str,
    "bot_response": str,
    "agent_turns": [
        {
            "parts": [
                {"tool_name": ..., "args": {...}, "tool_call_id": ..., "part_kind": "tool-call"},
                OR
                {"tool_name": ..., "content": ..., "tool_call_id": ..., "part_kind": "tool-return"}
            ],
            "usage": {...},
            "model_name": str,
            "timestamp": str,
            "kind": "response",
            "provider_name": str,
            "finish_reason": str,
            "run_id": str,
            ...
        }
    ]
}

These models are the "wire format" — the on-disk format we receive from production.
They get converted to our canonical Trajectory schema via langfuse_parser.py.
"""

from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field


class ToolCallPart(BaseModel):
    """A single tool-call inside an agent turn's parts list."""
    part_kind: Literal["tool-call"]
    tool_name: str
    args: dict = Field(default_factory=dict)
    tool_call_id: str
    id: Optional[str] = None
    provider_details: Optional[dict] = None


class ToolReturnPart(BaseModel):
    """A single tool-return (result) inside an agent turn's parts list."""
    part_kind: Literal["tool-return"]
    tool_name: str
    content: str  # tool return is typically a string in Langfuse format
    tool_call_id: str
    timestamp: Optional[datetime] = None
    metadata: Optional[dict] = None


class TextPart(BaseModel):
    """An assistant text reply part (some Langfuse turns mix text + tool calls)."""
    part_kind: Literal["text"]
    content: str


class AgentTurn(BaseModel):
    """One LLM-side turn in a Langfuse trace. May contain multiple parts
    (e.g., reasoning text + a tool call)."""
    model_config = {"protected_namespaces": ()}

    parts: list[dict]  # union of ToolCallPart | ToolReturnPart | TextPart; we keep flexible
    usage: Optional[dict] = None
    model_name: Optional[str] = None
    timestamp: Optional[datetime] = None
    kind: Optional[str] = None  # "response" or "request"
    provider_name: Optional[str] = None
    provider_url: Optional[str] = None
    provider_details: Optional[dict] = None
    provider_response_id: Optional[str] = None
    finish_reason: Optional[str] = None
    run_id: Optional[str] = None
    metadata: Optional[dict] = None


class LangfuseTrace(BaseModel):
    """A complete Langfuse-format trace for one user-bot interaction."""
    user_question: str
    bot_response: str = ""
    agent_turns: list[AgentTurn] = Field(default_factory=list)
    # Optional metadata fields commonly present in Langfuse exports
    session_id: Optional[str] = None
    trace_id: Optional[str] = None
    user_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    metadata: Optional[dict] = Field(default_factory=dict)
