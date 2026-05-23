"""
Mock Tool Environment for synthetic trajectory generation.

Provides a registry of 5 mock tools matching the OpenAgriNet production
toolset (inferred from the mentor's sample trace and ticket context).
Each tool:
  - Validates incoming args against a schema (rejects malformed calls)
  - Returns realistic synthetic data with India-specific plausibility
  - Can be forced to fail (timeout, no_data, invalid_args) for generating
    error-recovery training trajectories

This is the foundation for Module 6's synthetic trajectory generator:
LLM proposes a tool call → MockToolExecutor validates + returns → LLM
continues. Grounding via real tool schemas prevents hallucinated outputs.

Run standalone:
    python -m src.mock_tools --tool weather_forecast --args '{"latitude": 19.066, "longitude": 77.174, "days": 3}'
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool result type
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    """Result of a mock tool invocation.

    `status` is 'ok' on success, or one of {timeout, no_data, invalid_args,
    upstream_error} on failure. Failure results have `data=None` and a
    populated `error_message`. Successful results have `data` populated.
    """
    tool_name: str
    status: str  # "ok" | "timeout" | "no_data" | "invalid_args" | "upstream_error"
    data: dict | None = None
    error_message: str | None = None
    latency_ms: int = 0

    def to_dict(self) -> dict:
        out = {"status": self.status, "latency_ms": self.latency_ms}
        if self.data is not None:
            out.update(self.data)
        if self.error_message:
            out["error"] = self.error_message
        return out


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------

def _validate_args(args: dict, schema: dict) -> tuple[bool, str]:
    """Lightweight schema validator: checks required keys exist with correct types.

    schema format: {"key": (type, required), ...}
    """
    for key, spec in schema.items():
        expected_type, required = spec
        if key not in args:
            if required:
                return False, f"missing required arg: {key}"
            continue
        value = args[key]
        # Allow int where float expected
        if expected_type is float and isinstance(value, int):
            continue
        if not isinstance(value, expected_type):
            return False, f"arg '{key}' has type {type(value).__name__}, expected {expected_type.__name__}"
    return True, ""


# ---------------------------------------------------------------------------
# Realistic-data helpers
# ---------------------------------------------------------------------------

INDIAN_DISTRICTS = [
    "Nashik", "Pune", "Aurangabad", "Solapur", "Nanded", "Kolhapur",
    "Ludhiana", "Amritsar", "Patiala", "Hisar", "Karnal",
    "Kolar", "Mysuru", "Hassan", "Belagavi", "Tumkur",
    "Coimbatore", "Salem", "Madurai", "Vellore",
    "Guntur", "Vijayawada", "Tirupati", "Anantapur",
]

CROP_PRICE_RANGES = {
    "wheat": (1800, 2500),
    "paddy": (1900, 2700),
    "cotton": (5500, 7200),
    "soybean": (3500, 4800),
    "onion": (1200, 3500),
    "tomato": (800, 2800),
    "sugarcane": (250, 380),
    "maize": (1700, 2300),
    "groundnut": (4800, 6500),
    "ragi": (2800, 3600),
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Individual tool implementations
# ---------------------------------------------------------------------------

def mandi_prices(args: dict) -> ToolResult:
    crop = args["crop"].lower()
    market = args["market"]
    date = args.get("date", _now_iso())

    if crop not in CROP_PRICE_RANGES:
        return ToolResult(
            tool_name="mandi_prices",
            status="no_data",
            error_message=f"No mandi data available for crop '{crop}'",
            latency_ms=random.randint(80, 200),
        )

    low, high = CROP_PRICE_RANGES[crop]
    price = random.randint(low, high)
    return ToolResult(
        tool_name="mandi_prices",
        status="ok",
        data={
            "crop": crop,
            "market": market,
            "date": date,
            "price_per_quintal": price,
            "unit": "INR/quintal",
            "trend_7d": random.choice(["rising", "falling", "stable"]),
            "msp": round(price * random.uniform(0.85, 1.05)),
        },
        latency_ms=random.randint(120, 350),
    )


def weather_forecast(args: dict) -> ToolResult:
    lat = args["latitude"]
    lng = args["longitude"]
    days = args.get("days", 3)

    # Bounding-box check — outside India returns no_data
    if not (6.0 <= lat <= 37.0 and 68.0 <= lng <= 97.0):
        return ToolResult(
            tool_name="weather_forecast",
            status="no_data",
            error_message=f"Coordinates ({lat}, {lng}) outside India service area",
            latency_ms=random.randint(50, 150),
        )

    forecast = [
        {
            "day": i + 1,
            "rain_mm": random.choice([0, 0, 0, 2, 5, 8, 12, 18, 25]),
            "temp_max": random.randint(26, 42),
            "temp_min": random.randint(18, 28),
            "humidity_pct": random.randint(40, 88),
        }
        for i in range(days)
    ]
    return ToolResult(
        tool_name="weather_forecast",
        status="ok",
        data={
            "latitude": lat,
            "longitude": lng,
            "forecast": forecast,
            "total_rain_mm": sum(d["rain_mm"] for d in forecast),
        },
        latency_ms=random.randint(180, 450),
    )


def fetch_agristack_data(args: dict) -> ToolResult:
    farmer_id = args["farmer_id"]

    # Simulate ~10% miss rate (no record found)
    if random.random() < 0.10:
        return ToolResult(
            tool_name="fetch_agristack_data",
            status="no_data",
            error_message=f"No Agristack record found for farmer_id '{farmer_id}'",
            latency_ms=random.randint(200, 500),
        )

    district = random.choice(INDIAN_DISTRICTS)
    # Real-looking lat/lng for a few key districts; otherwise fallback
    coords = {
        "Nashik": (19.997, 73.789),
        "Pune": (18.520, 73.856),
        "Kolar": (13.137, 78.129),
        "Ludhiana": (30.901, 75.857),
        "Nanded": (19.155, 77.310),
    }.get(district, (round(random.uniform(8.0, 32.0), 3), round(random.uniform(72.0, 88.0), 3)))

    return ToolResult(
        tool_name="fetch_agristack_data",
        status="ok",
        data={
            "farmer_id": farmer_id,
            "name_masked": "R**** P****",  # production already masks
            "mobile_masked": f"{random.randint(70, 99)}***{random.randint(0, 9)}",
            "village": random.choice(["Loha", "Khedgaon", "Pimpalgaon", "Bhadgaon"]),
            "district": district,
            "state": random.choice(["Maharashtra", "Karnataka", "Punjab", "Andhra Pradesh"]),
            "land_area_hectares": round(random.uniform(0.5, 5.0), 2),
            "latitude": coords[0],
            "longitude": coords[1],
            "verified": True,
        },
        latency_ms=random.randint(300, 800),
    )


def soil_test_report(args: dict) -> ToolResult:
    field_id = args["field_id"]
    crop = args.get("crop")

    return ToolResult(
        tool_name="soil_test_report",
        status="ok",
        data={
            "field_id": field_id,
            "crop_planned": crop,
            "ph": round(random.uniform(5.8, 8.2), 1),
            "nitrogen_kg_per_ha": random.randint(120, 320),
            "phosphorus_kg_per_ha": random.randint(10, 60),
            "potassium_kg_per_ha": random.randint(80, 280),
            "organic_carbon_pct": round(random.uniform(0.3, 1.2), 2),
            "moisture_pct": random.randint(15, 65),
            "recommendation": random.choice([
                "Apply 50kg urea per acre before sowing",
                "Soil pH high; apply gypsum @ 200kg/acre",
                "Adequate NPK; focus on micronutrient supplementation",
                "Low organic carbon; add 5 tonnes FYM per acre",
            ]),
        },
        latency_ms=random.randint(150, 400),
    )


def govt_scheme_lookup(args: dict) -> ToolResult:
    scheme = args["scheme"].lower()
    land_area = args.get("land_area_hectares", 0)

    scheme_db = {
        "pm_kisan": {
            "eligibility_max_hectares": 2.0,
            "amount_per_installment": 2000,
            "installments_per_year": 3,
        },
        "pm_fasal_bima": {
            "eligibility_max_hectares": 5.0,
            "premium_pct": 2.0,
            "coverage": "kharif and rabi crops",
        },
        "kcc": {
            "eligibility_max_hectares": 10.0,
            "interest_rate_pct": 7.0,
            "max_credit_inr": 300000,
        },
    }

    if scheme not in scheme_db:
        return ToolResult(
            tool_name="govt_scheme_lookup",
            status="no_data",
            error_message=f"Scheme '{scheme}' not in database",
            latency_ms=random.randint(80, 180),
        )

    info = scheme_db[scheme]
    eligible = land_area <= info["eligibility_max_hectares"]
    return ToolResult(
        tool_name="govt_scheme_lookup",
        status="ok",
        data={
            "scheme": scheme,
            "eligible": eligible,
            "reason": "meets criteria" if eligible else f"landholding > {info['eligibility_max_hectares']} ha cap",
            **info,
            "next_installment_estimated": "Feb 2026",
        },
        latency_ms=random.randint(220, 500),
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@dataclass
class ToolSpec:
    name: str
    description: str
    args_schema: dict  # {"key": (type, required)}
    impl: Callable[[dict], ToolResult]


TOOL_REGISTRY: dict[str, ToolSpec] = {
    "mandi_prices": ToolSpec(
        name="mandi_prices",
        description="Fetch current mandi price for a crop in a specific market",
        args_schema={
            "crop": (str, True),
            "market": (str, True),
            "date": (str, False),
        },
        impl=mandi_prices,
    ),
    "weather_forecast": ToolSpec(
        name="weather_forecast",
        description="N-day weather forecast for a lat/lng (India bbox only)",
        args_schema={
            "latitude": (float, True),
            "longitude": (float, True),
            "days": (int, False),
        },
        impl=weather_forecast,
    ),
    "fetch_agristack_data": ToolSpec(
        name="fetch_agristack_data",
        description="Look up farmer details from Agristack by farmer_id",
        args_schema={
            "farmer_id": (str, True),
        },
        impl=fetch_agristack_data,
    ),
    "soil_test_report": ToolSpec(
        name="soil_test_report",
        description="Soil test results for a specific field, optionally with planned crop",
        args_schema={
            "field_id": (str, True),
            "crop": (str, False),
        },
        impl=soil_test_report,
    ),
    "govt_scheme_lookup": ToolSpec(
        name="govt_scheme_lookup",
        description="Check eligibility and details for a government scheme (pm_kisan, pm_fasal_bima, kcc)",
        args_schema={
            "scheme": (str, True),
            "land_area_hectares": (float, False),
            "aadhaar_last4": (str, False),
        },
        impl=govt_scheme_lookup,
    ),
}


# ---------------------------------------------------------------------------
# The executor
# ---------------------------------------------------------------------------

class MockToolExecutor:
    """Executes mock tools with schema validation and controlled failure injection.

    Used by the synthetic trajectory generator (Module 6) to ground LLM-proposed
    tool calls against realistic, validated mock responses.

    Usage:
        executor = MockToolExecutor(seed=42)
        result = executor.execute("weather_forecast", {"latitude": 19.0, "longitude": 73.0, "days": 3})
        if result.status == "ok":
            ... use result.data
    """

    def __init__(self, seed: int | None = None, failure_rate: float = 0.0):
        """
        seed: if set, makes mock responses deterministic for reproducibility
        failure_rate: 0.0-1.0, probability of injecting an upstream_error
                      on each call (useful for generating error-recovery data)
        """
        if seed is not None:
            random.seed(seed)
        self.failure_rate = failure_rate
        self.call_log: list[dict] = []

    def execute(self, tool_name: str, args: dict, force_failure: str | None = None) -> ToolResult:
        """Execute a mock tool call.

        force_failure: if set, bypass success and return that failure type.
                       Values: 'timeout', 'no_data', 'invalid_args', 'upstream_error'
        """
        # Log the attempt
        self.call_log.append({"tool": tool_name, "args": args, "forced_failure": force_failure})

        # Unknown tool
        if tool_name not in TOOL_REGISTRY:
            return ToolResult(
                tool_name=tool_name,
                status="invalid_args",
                error_message=f"Unknown tool '{tool_name}'. Available: {list(TOOL_REGISTRY.keys())}",
            )

        spec = TOOL_REGISTRY[tool_name]

        # Forced failure path
        if force_failure:
            return self._make_failure(tool_name, force_failure)

        # Schema validation
        ok, err = _validate_args(args, spec.args_schema)
        if not ok:
            return ToolResult(
                tool_name=tool_name,
                status="invalid_args",
                error_message=err,
                latency_ms=random.randint(20, 80),
            )

        # Random failure injection
        if self.failure_rate > 0 and random.random() < self.failure_rate:
            return self._make_failure(tool_name, "upstream_error")

        # Execute the real mock
        try:
            return spec.impl(args)
        except Exception as e:
            logger.exception(f"Mock tool {tool_name} raised unexpectedly")
            return ToolResult(
                tool_name=tool_name,
                status="upstream_error",
                error_message=f"Mock tool crashed: {e}",
            )

    def _make_failure(self, tool_name: str, kind: str) -> ToolResult:
        messages = {
            "timeout": "Tool timed out after 30s",
            "no_data": "No data found for the requested parameters",
            "invalid_args": "One or more arguments failed validation",
            "upstream_error": "Upstream service temporarily unavailable",
        }
        return ToolResult(
            tool_name=tool_name,
            status=kind,
            error_message=messages.get(kind, "unknown failure"),
            latency_ms=random.randint(50, 30_000) if kind == "timeout" else random.randint(100, 400),
        )

    # Convenience for the synthetic generator
    def list_tools(self) -> list[dict]:
        """Return tool catalog formatted for an LLM system prompt."""
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "args_schema": {
                    k: {"type": v[0].__name__, "required": v[1]}
                    for k, v in spec.args_schema.items()
                },
            }
            for spec in TOOL_REGISTRY.values()
        ]


# ---------------------------------------------------------------------------
# CLI runner — lets you verify any tool manually
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Invoke a mock tool from the CLI.")
    parser.add_argument("--tool", required=False, help="Tool name (see --list)")
    parser.add_argument("--args", default="{}", help="JSON dict of args")
    parser.add_argument("--force-failure", default=None,
                        help="Force a failure mode: timeout|no_data|invalid_args|upstream_error")
    parser.add_argument("--list", action="store_true", help="List available tools and exit")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    executor = MockToolExecutor(seed=args.seed)

    if args.list:
        print(json.dumps(executor.list_tools(), indent=2))
        return

    if not args.tool:
        raise SystemExit("--tool is required unless --list is given")

    try:
        tool_args = json.loads(args.args)
    except json.JSONDecodeError as e:
        raise SystemExit(f"--args must be valid JSON: {e}")

    result = executor.execute(args.tool, tool_args, force_failure=args.force_failure)
    print(json.dumps({
        "tool": result.tool_name,
        "status": result.status,
        "latency_ms": result.latency_ms,
        "data": result.data,
        "error": result.error_message,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
