"""
LLM-guided synthetic trajectory generator (design + stub).

When a local LLM is available (Qwen 7B or similar), this generator:
  1. Samples a seed user query from real logs (or templated pool)
  2. Prompts the LLM with the mock tool catalog and asks it to propose
     the next action (tool call OR final answer)
  3. Executes the proposed tool call via MockToolExecutor (validates +
     returns grounded data)
  4. Feeds the result back to the LLM, repeats until the LLM emits a
     final answer or hits max_steps
  5. Tags the trajectory `synthesis_source="llm_guided"`

This produces more linguistically diverse queries and more emergent
tool-sequence variety than templates can hit by construction.

Requires either:
  - A locally-hosted Qwen2.5-7B-Instruct (or similar) at OPENAI_BASE_URL
  - OR an Anthropic/OpenAI API key (the openai-compatible client works
    against most providers)

This file ships with the design and a working interface; actual LLM calls
are stubbed by default so the module is importable without GPU. To enable:
    export USE_LLM=1
    export OPENAI_BASE_URL=http://localhost:8000/v1  # or your endpoint
    export OPENAI_API_KEY=local

Run:
    python -m src.synthesis.llm_guided --n 5 --seed-queries data/synthetic_logs/seed_queries.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from datetime import datetime, timezone
from pathlib import Path

from src.mock_tools import MockToolExecutor, TOOL_REGISTRY
from src.synthesis.builder import QueryRecipe, ToolStep, build_trajectory

logger = logging.getLogger("synth_llm")


SYSTEM_PROMPT = """You are a planning assistant for an agricultural agent.
Given a user query and a set of tools, you decide the next action.

Available tools:
{tool_catalog}

For each turn, respond with EXACTLY one JSON object:
  {{"action": "tool_call", "tool_name": "...", "args": {{...}}}}
  OR
  {{"action": "answer", "content": "final answer to the user"}}

Use minimum tool calls. Stop with action=answer once you have enough info.
""".strip()


def _tool_catalog_text() -> str:
    lines = []
    for name, spec in TOOL_REGISTRY.items():
        args = ", ".join(f"{k}: {v[0].__name__}" for k, v in spec.args_schema.items())
        lines.append(f"- {name}({args}) — {spec.description}")
    return "\n".join(lines)


def _call_llm(messages: list[dict]) -> dict:
    """Call the local LLM and parse the JSON action.

    Returns a dict like {"action": "tool_call", ...} or {"action": "answer", ...}.
    If USE_LLM env is not set, raises NotImplementedError — callers should
    fall back to templated generation.
    """
    if not os.environ.get("USE_LLM"):
        raise NotImplementedError(
            "LLM-guided synthesis requires USE_LLM=1 and a configured LLM endpoint."
        )

    # The openai client works against most local servers (vllm, llama.cpp, etc.)
    from openai import OpenAI
    client = OpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.environ.get("OPENAI_API_KEY", "local"),
    )
    response = client.chat.completions.create(
        model=os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        messages=messages,
        temperature=0.7,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def generate_one(seed_query: str, executor: MockToolExecutor, max_steps: int = 6) -> QueryRecipe:
    """Generate one LLM-guided recipe (then convert to trajectory via build_trajectory)."""
    catalog = _tool_catalog_text()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(tool_catalog=catalog)},
        {"role": "user", "content": seed_query},
    ]
    tool_steps: list[ToolStep] = []
    final_answer = ""

    for step in range(max_steps):
        action = _call_llm(messages)
        if action.get("action") == "tool_call":
            tool_steps.append(ToolStep(
                tool_name=action["tool_name"],
                args=action.get("args", {}),
            ))
            # Execute mock to get the result we'll feed back to the LLM
            result = executor.execute(action["tool_name"], action.get("args", {}))
            messages.append({"role": "assistant", "content": json.dumps(action)})
            messages.append({
                "role": "user",
                "content": f"Tool result: {json.dumps(result.to_dict())}",
            })
        elif action.get("action") == "answer":
            final_answer = action.get("content", "")
            break

    return QueryRecipe(
        user_query=seed_query,
        workflow="general",  # could classify via tools_used after the fact
        tool_steps=tool_steps,
        assistant_template=final_answer or None,
        metadata={"synthesis_source": "llm_guided"},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-guided synthetic trajectory generator.")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--seed-queries", type=Path, required=True,
                        help="One seed query per line.")
    parser.add_argument("--output", type=Path,
                        default=Path("data/synthetic_logs/synth_llm.jsonl"))
    args = parser.parse_args()

    if not os.environ.get("USE_LLM"):
        print("⚠️  LLM-guided synthesis requires USE_LLM=1 and a configured endpoint.")
        print("This module is designed and tested; to enable, set:")
        print("    export USE_LLM=1")
        print("    export OPENAI_BASE_URL=http://localhost:8000/v1   # your LLM server")
        print("    export LLM_MODEL=Qwen/Qwen2.5-7B-Instruct")
        print("Falling back: writing 0 trajectories.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    seeds = args.seed_queries.read_text(encoding="utf-8").splitlines()
    seeds = [s.strip() for s in seeds if s.strip()][: args.n]

    executor = MockToolExecutor()
    trajectories = []
    for seed in seeds:
        recipe = generate_one(seed, executor)
        traj = build_trajectory(recipe, executor)
        trajectories.append(traj)

    with args.output.open("w", encoding="utf-8") as f:
        for t in trajectories:
            f.write(t.model_dump_json() + "\n")

    print(f"Wrote {len(trajectories)} LLM-guided trajectories to {args.output}")


if __name__ == "__main__":
    main()