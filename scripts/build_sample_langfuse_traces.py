"""
Build a small sample file of Langfuse-format traces for testing the parser.

Output: data/synthetic_logs/langfuse_traces.jsonl

These mirror the EXACT structure the mentor showed in his GitHub comment,
so the parser proves it can handle production-shaped input.

Run:
    python scripts/build_sample_langfuse_traces.py
"""

import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path


def build_trace(
    user_question: str,
    tool_sequence: list[tuple[str, dict, dict | str]],
    bot_response: str,
) -> dict:
    """Build one Langfuse trace.

    tool_sequence: list of (tool_name, args, return_content) tuples.
    """
    base_ts = datetime(2026, 1, 20, 5, 54, 15, tzinfo=timezone.utc)
    agent_turns = []

    for i, (tool_name, args, ret_content) in enumerate(tool_sequence):
        # Tool call turn
        agent_turns.append({
            "parts": [{
                "tool_name": tool_name,
                "args": args,
                "tool_call_id": f"toolu_{uuid.uuid4().hex[:20]}",
                "id": None,
                "provider_details": None,
                "part_kind": "tool-call",
            }],
            "usage": {
                "input_tokens": 11000 + i * 200,
                "output_tokens": 40,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
            },
            "model_name": "claude-3-5-sonnet-20241022",
            "timestamp": (base_ts + timedelta(seconds=i * 2)).isoformat(),
            "kind": "response",
            "provider_name": "anthropic",
            "finish_reason": "tool_call",
            "run_id": str(uuid.uuid4()),
        })
        # Tool return turn
        agent_turns.append({
            "parts": [{
                "tool_name": tool_name,
                "content": ret_content if isinstance(ret_content, str) else json.dumps(ret_content),
                "tool_call_id": f"toolu_{uuid.uuid4().hex[:20]}",
                "metadata": None,
                "timestamp": (base_ts + timedelta(seconds=i * 2 + 1)).isoformat(),
                "part_kind": "tool-return",
            }],
        })

    return {
        "user_question": user_question,
        "bot_response": bot_response,
        "agent_turns": agent_turns,
        "session_id": f"session_{uuid.uuid4().hex[:12]}",
        "trace_id": f"trace_{uuid.uuid4().hex[:12]}",
        "timestamp": base_ts.isoformat(),
    }


SAMPLE_TRACES = [
    build_trace(
        user_question="What's the weather forecast for my farm in Loha, Maharashtra?",
        tool_sequence=[
            (
                "fetch_agristack_data",
                {"farmer_id": "f_demo_001"},
                "> Farmer Information (Agristack)\nResponses:\n      Provider: Agristack\n      Farmer Details:\n        Name: Ramesh Patil\n        Mobile: 87***2\n        Village Name: Loha\n        District Name: Nanded\n        Total Plot Area: 2.5 hectares\n      Locations:\n        Loha, Maharashtra, India (19.066, 77.174)",
            ),
            (
                "weather_forecast",
                {"latitude": 19.066, "longitude": 77.174, "days": 5},
                {
                    "forecast": [
                        {"day": 1, "rain_mm": 0, "temp_max": 34},
                        {"day": 2, "rain_mm": 12, "temp_max": 31},
                        {"day": 3, "rain_mm": 4, "temp_max": 32},
                        {"day": 4, "rain_mm": 0, "temp_max": 35},
                        {"day": 5, "rain_mm": 8, "temp_max": 30},
                    ],
                    "location": "Loha, Maharashtra",
                },
            ),
        ],
        bot_response="Based on your location in Loha, Maharashtra, the 5-day forecast shows light rain on day 2 (12mm) and day 5 (8mm), with temperatures between 30-35°C. Plan irrigation for days 1, 3, and 4 when no rain is expected.",
    ),
    build_trace(
        user_question="aaj mandi mein gehun ka kya bhav hai Nashik mein?",
        tool_sequence=[
            (
                "mandi_prices",
                {"crop": "wheat", "market": "Nashik", "date": "2026-01-20"},
                {"crop": "wheat", "market": "Nashik", "price_per_quintal": 2185, "date": "2026-01-20", "unit": "INR/quintal"},
            ),
        ],
        bot_response="Nashik mandi mein aaj gehun ka bhav ₹2185 per quintal hai. Yeh achha rate hai bechne ke liye.",
    ),
    build_trace(
        user_question="Am I eligible for PM-Kisan? My Aadhaar is 234567890123",
        tool_sequence=[
            (
                "fetch_agristack_data",
                {"farmer_id": "f_demo_002"},
                "> Farmer Information (Agristack)\n      Name: Suresh Kumar\n      Mobile: 98***7\n      Total Plot Area: 1.8 hectares\n      Locations: Belagavi, Karnataka",
            ),
            (
                "govt_scheme_lookup",
                {"scheme": "pm_kisan", "aadhaar_last4": "0123", "land_area_hectares": 1.8},
                {"eligible": True, "scheme": "pm_kisan", "next_installment": "Feb 2026", "amount": 2000},
            ),
        ],
        bot_response="Yes, you are eligible for PM-Kisan. Your next installment of ₹2000 should arrive by February 2026.",
    ),
    # Add a deliberately-broken trajectory (wrong tool, no recovery) — for repair module later
    build_trace(
        user_question="What's the soil moisture in my field today?",
        tool_sequence=[
            (
                "mandi_prices",  # WRONG tool — should be soil_test_report
                {"crop": "tomato", "market": "Pune", "date": "2026-01-20"},
                {"crop": "tomato", "market": "Pune", "price_per_quintal": 3200},
            ),
        ],
        bot_response="The tomato price in Pune is ₹3200 per quintal.",  # Doesn't answer soil question
    ),
]


def main() -> None:
    out_dir = Path("data/synthetic_logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "langfuse_traces.jsonl"

    with out_path.open("w", encoding="utf-8") as f:
        for trace in SAMPLE_TRACES:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")

    print(f"Wrote {len(SAMPLE_TRACES)} sample Langfuse traces to {out_path}")


if __name__ == "__main__":
    main()
