"""
Templated synthetic trajectory generator.

Composes a large diverse dataset from:
  - Query templates per workflow (English + Hinglish)
  - Crop / location / scheme variations
  - Difficulty knobs (simple / moderate / complex)
  - Failure modes (error recovery, no-data, ambiguous queries)

Produces grounded trajectories by executing planned tool sequences through
the Mock Tool Environment. No LLM required — runs instantly, deterministic
under a seed.

Run:
    python -m src.synthesis.templated --n 200 --output data/synthetic_logs/synth_templated.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.mock_tools import MockToolExecutor
from src.synthesis.builder import QueryRecipe, ToolStep, build_trajectory

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("synth_templated")


# ---------------------------------------------------------------------------
# Variation pools
# ---------------------------------------------------------------------------

CROPS = ["wheat", "paddy", "cotton", "soybean", "onion", "tomato", "sugarcane", "maize", "groundnut", "ragi"]
MARKETS = ["Nashik", "Pune", "Ludhiana", "Hisar", "Kolar", "Mysuru", "Coimbatore", "Guntur", "Vijayawada", "Belagavi"]
SCHEMES = ["pm_kisan", "pm_fasal_bima", "kcc"]
COORDS = {
    "Nashik": (19.997, 73.789),
    "Pune": (18.520, 73.856),
    "Ludhiana": (30.901, 75.857),
    "Hisar": (29.151, 75.722),
    "Kolar": (13.137, 78.129),
    "Mysuru": (12.295, 76.639),
    "Coimbatore": (11.016, 76.956),
    "Guntur": (16.306, 80.439),
    "Belagavi": (15.852, 74.504),
}


# ---------------------------------------------------------------------------
# Workflow templates
# ---------------------------------------------------------------------------

def template_mandi_simple(rng: random.Random) -> QueryRecipe:
    crop = rng.choice(CROPS)
    market = rng.choice(MARKETS)
    query_en = f"What's {crop} selling for in {market} mandi today?"
    query_hi = f"aaj {market} mandi mein {crop} ka kya bhav hai?"
    lang = rng.choice(["en", "en", "en", "code_mixed"])
    return QueryRecipe(
        user_query=query_hi if lang == "code_mixed" else query_en,
        workflow="price",
        language=lang,
        tool_steps=[
            ToolStep("mandi_prices", {"crop": crop, "market": market}),
        ],
        assistant_template=(
            f"In {market}, {crop} is currently priced at "
            "₹{price_per_quintal}/quintal with a {trend_7d} trend over the last 7 days."
        ),
    )


def template_weather_with_agristack(rng: random.Random) -> QueryRecipe:
    farmer_id = f"f_{rng.randint(100, 999)}"
    return QueryRecipe(
        user_query="What's the weather forecast for my farm this week?",
        workflow="irrigation",
        language="en",
        tool_steps=[
            ToolStep("fetch_agristack_data", {"farmer_id": farmer_id}),
            # weather coords will be plugged via a real district later if available
            ToolStep("weather_forecast", {
                "latitude": rng.uniform(12.0, 30.0),
                "longitude": rng.uniform(73.0, 82.0),
                "days": rng.choice([3, 5, 7]),
            }),
        ],
    )


def template_pm_kisan_eligibility(rng: random.Random) -> QueryRecipe:
    land = round(rng.uniform(0.4, 6.0), 1)
    query_variants = [
        f"Am I eligible for PM-Kisan? I have {land} hectares of land.",
        f"PM Kisan ke liye apply karna hai, meri zameen {land} hectare hai.",
        f"Will I get PM-Kisan benefit with {land} ha?",
    ]
    lang = rng.choice(["en", "code_mixed", "en"])
    return QueryRecipe(
        user_query=rng.choice(query_variants),
        workflow="scheme",
        language=lang,
        tool_steps=[
            ToolStep("govt_scheme_lookup", {
                "scheme": "pm_kisan",
                "land_area_hectares": land,
            }),
        ],
        assistant_template=(
            "Based on your landholding of " + str(land) + " hectares, you appear {eligible} "
            "for PM-Kisan. Next installment: {next_installment_estimated}."
        ),
    )


def template_soil_advice(rng: random.Random) -> QueryRecipe:
    crop = rng.choice(CROPS)
    field_id = f"FLD{rng.randint(100, 999)}"
    return QueryRecipe(
        user_query=f"Can you check my soil report for the {crop} season?",
        workflow="calendar",
        language="en",
        tool_steps=[
            ToolStep("soil_test_report", {"field_id": field_id, "crop": crop}),
        ],
        assistant_template=(
            "Your soil pH is {ph}. Recommendation: {recommendation}"
        ),
    )


def template_multi_tool_complex(rng: random.Random) -> QueryRecipe:
    """3-tool chain: agristack → soil test → weather. Stresses the pipeline."""
    farmer_id = f"f_{rng.randint(100, 999)}"
    field_id = f"FLD{rng.randint(100, 999)}"
    crop = rng.choice(CROPS)
    return QueryRecipe(
        user_query=f"Plan my {crop} season — soil status and rain outlook for the next 5 days?",
        workflow="calendar",
        language="en",
        tool_steps=[
            ToolStep("fetch_agristack_data", {"farmer_id": farmer_id}),
            ToolStep("soil_test_report", {"field_id": field_id, "crop": crop}),
            ToolStep("weather_forecast", {
                "latitude": rng.uniform(12.0, 30.0),
                "longitude": rng.uniform(73.0, 82.0),
                "days": 5,
            }),
        ],
    )


def template_ambiguous_query(rng: random.Random) -> QueryRecipe:
    """Ambiguous user query that requires clarification."""
    return QueryRecipe(
        user_query="I need help with my farm.",
        workflow="general",
        language="en",
        add_clarification=True,
        clarification_question="Sure — could you tell me what specifically: weather, prices, soil, or a government scheme?",
        user_clarification_reply="Yes, I want to check the mandi price for onions in Nashik.",
        tool_steps=[
            ToolStep("mandi_prices", {"crop": "onion", "market": "Nashik"}),
        ],
    )


def template_with_recovery(rng: random.Random) -> QueryRecipe:
    """First tool call fails, then retries successfully. Trains error-recovery behavior."""
    crop = rng.choice(CROPS)
    market = rng.choice(MARKETS)
    return QueryRecipe(
        user_query=f"What's the price of {crop} in {market}?",
        workflow="price",
        language="en",
        add_recovery=True,
        tool_steps=[
            ToolStep("mandi_prices", {"crop": crop, "market": market}),
        ],
    )


def template_invalid_args_dropped(rng: random.Random) -> QueryRecipe:
    """User asks something that causes invalid args (e.g. coordinates outside India).
    Tests how the model handles a tool that rejects its args."""
    return QueryRecipe(
        user_query="What's the weather forecast for my field at 40.7, -74.0?",
        workflow="irrigation",
        language="en",
        tool_steps=[
            ToolStep("weather_forecast", {
                "latitude": 40.7,
                "longitude": -74.0,
                "days": 3,
            }),
        ],
    )


TEMPLATES = [
    template_mandi_simple,
    template_weather_with_agristack,
    template_pm_kisan_eligibility,
    template_soil_advice,
    template_multi_tool_complex,
    template_ambiguous_query,
    template_with_recovery,
    template_invalid_args_dropped,
]


# Weight the templates so the distribution reflects realistic agri-bot usage
TEMPLATE_WEIGHTS = [
    0.20,  # mandi_simple — common
    0.15,  # weather + agristack
    0.15,  # pm_kisan
    0.10,  # soil
    0.15,  # multi-tool complex
    0.10,  # ambiguous
    0.10,  # recovery
    0.05,  # invalid_args edge case
]


# ---------------------------------------------------------------------------
# Generator orchestrator
# ---------------------------------------------------------------------------

def generate(n: int, seed: int = 42) -> tuple[list, dict]:
    rng = random.Random(seed)
    executor = MockToolExecutor(seed=seed)

    trajectories = []
    stats: Counter = Counter()
    base_time = datetime.now(timezone.utc) - timedelta(days=30)

    for i in range(n):
        template = rng.choices(TEMPLATES, weights=TEMPLATE_WEIGHTS, k=1)[0]
        recipe = template(rng)
        recipe.metadata["synthesis_source"] = "templated"
        recipe.metadata["template"] = template.__name__

        ts = base_time + timedelta(minutes=rng.randint(0, 60 * 24 * 30))
        traj = build_trajectory(recipe, executor, base_timestamp=ts)
        trajectories.append(traj)

        stats[("template", template.__name__)] += 1
        stats[("workflow", recipe.workflow)] += 1
        stats[("language", recipe.language)] += 1
        stats[("complexity", traj.complexity_bucket)] += 1
        if recipe.add_recovery:
            stats[("recovery", "yes")] += 1
        if recipe.add_clarification:
            stats[("clarification", "yes")] += 1

    # Flatten stats for printing
    summary = {
        "total": n,
        "by_template": {k[1]: v for k, v in stats.items() if k[0] == "template"},
        "by_workflow": {k[1]: v for k, v in stats.items() if k[0] == "workflow"},
        "by_language": {k[1]: v for k, v in stats.items() if k[0] == "language"},
        "by_complexity": {k[1]: v for k, v in stats.items() if k[0] == "complexity"},
        "with_recovery": stats[("recovery", "yes")],
        "with_clarification": stats[("clarification", "yes")],
    }
    return trajectories, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate templated synthetic trajectories.")
    parser.add_argument("--n", type=int, default=200, help="Number of trajectories to generate")
    parser.add_argument("--output", type=Path, default=Path("data/synthetic_logs/synth_templated.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating {args.n} templated trajectories (seed={args.seed})")
    trajectories, summary = generate(args.n, seed=args.seed)

    with args.output.open("w", encoding="utf-8") as f:
        for t in trajectories:
            f.write(t.model_dump_json() + "\n")

    logger.info(f"Wrote {len(trajectories)} trajectories to {args.output}")
    print("\n" + "=" * 60)
    print("  Templated synthesis complete")
    print("=" * 60)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()