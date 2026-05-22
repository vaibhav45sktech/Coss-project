"""
Synthetic log generator for the agricultural LLM training pipeline.

Produces data/synthetic_logs/raw_logs.jsonl: one RawEvent JSON object per line.
300 sessions × varying depth → ~2 000–2 500 events total.

Run from the repo root:
    python -m scripts.generate_synthetic_logs
"""

import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from faker import Faker

from src.schemas import EventRole, RawEvent

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
fake = Faker("en_IN")
random.seed(42)
Faker.seed(42)


def _uuid() -> str:
    """Seeded UUID so every run produces identical IDs."""
    return str(uuid.UUID(int=random.getrandbits(128)))


# ---------------------------------------------------------------------------
# Workflow definitions
# ---------------------------------------------------------------------------
WORKFLOWS: dict[str, list[str]] = {
    "irrigation_advisory": ["get_weather", "get_soil_moisture", "irrigation_recommendation"],
    "pest_diagnosis": ["identify_pest", "recommend_treatment"],
    "mandi_price": ["get_mandi_price", "get_price_trend"],
    "pm_kisan_eligibility": ["check_landholding", "verify_pm_kisan_eligibility"],
    "crop_calendar": ["get_agro_climatic_zone", "crop_calendar"],
}

DISTRICTS = [
    "Kolar", "Mysuru", "Anantapur", "Nashik", "Ludhiana",
    "Coimbatore", "Guntur", "Belagavi", "Salem", "Hassan",
]

# ---------------------------------------------------------------------------
# User-message templates
# ---------------------------------------------------------------------------
USER_TEMPLATES: dict[str, list[str]] = {
    "irrigation_advisory": [
        "When should I water my tomato field this week?",
        "Soil feels dry, should I irrigate today?",
        "How much water does my wheat crop need right now?",
        "It hasn't rained in 5 days, should I switch on the pump?",
        "Best irrigation schedule for sugarcane in summer?",
        "My drip system is set to 30 min daily — is that enough for brinjal?",
    ],
    "pest_diagnosis": [
        "Yellow spots on my brinjal leaves, what is it?",
        "Leaves curling up on chilli plants, help",
        "White powder on mango leaves since yesterday",
        "Small black insects on my cotton crop, very worried",
        "Holes in the leaves of my tomato plants, please advise",
        "My paddy has brown sheath at the base, disease or pest?",
    ],
    "mandi_price": [
        "What's wheat selling for in Nashik today?",
        "What is the current price of onion in the market?",
        "Is the price of soybean going up or down this month?",
        "Should I sell my maize now or wait for better price?",
        "What is the MSP for paddy this season?",
        "Is tomato price recovering after last week's dip?",
    ],
    "pm_kisan_eligibility": [
        "Am I eligible for PM-Kisan? My Aadhaar is 234567890123",
        "How do I check my PM Kisan installment status?",
        "What documents do I need for PM Kisan registration?",
        "My father died — can I transfer PM-Kisan to my name?",
        "I have 3 acres of land, will I get PM-Kisan benefit?",
        "PM Kisan amount not received for last 2 installments, what to do?",
    ],
    "crop_calendar": [
        "When should I sow paddy in Mysuru?",
        "Best time to plant ragi this season?",
        "What crops are best for rabi season in Punjab?",
        "Can I grow sunflower in December in Andhra?",
        "Seasonal crop rotation advice for my 2 acre farm",
        "Which variety of groundnut should I plant in June in Karnataka?",
    ],
}

HINGLISH_TEMPLATES: dict[str, list[str]] = {
    "irrigation_advisory": [
        "bhai mere khet mein pani ki zaroorat hai kya abhi?",
        "soil moisture kam lag rahi hai, drip chalao ya nahi?",
        "kal baarish hogi kya mere area mein, weather batao",
        "ganne ki fasal ke liye kitna paani chahiye is hafte?",
    ],
    "pest_diagnosis": [
        "mere tamatar mein kuch keede lage hain, kya spray karun?",
        "patti mein patela rog aa gaya hai, koi solution batao",
        "fungicide ya pesticide, kya use karna chahiye brinjal ke liye?",
        "kapas ke patte mein safed macchhar dikh rahe hain, kya hai yeh?",
    ],
    "mandi_price": [
        "aaj gehun ka bhav Nashik mandi mein kya hai?",
        "pyaz ka rate badhega ya ghatega next week?",
        "MSP se zyada daam milega kya is baar cotton ka?",
        "soybean bechun abhi ya ruk jaun, price trend kya hai?",
    ],
    "pm_kisan_eligibility": [
        "PM Kisan ka paisa kab aayega mere account mein?",
        "eligibility check karna hai, kaise pata chalega?",
        "meri zamin 2.5 acre hai, PM Kisan milega kya?",
        "document kya chahiye PM Kisan ke liye?",
    ],
    "crop_calendar": [
        "is baar kharif mein chawal lagaun ya makka?",
        "rabi ke liye konsi fasal sabse aachi rahegi mere area mein?",
        "monsoon ke baad kya boona chahiye Mysuru mein?",
        "ragi ki bawaai kab karni chahiye Hassan district mein?",
    ],
}

CLARIFYING_QUESTIONS: dict[str, list[str]] = {
    "irrigation_advisory": [
        "Which crop are you growing and what is your field size?",
        "Can you tell me your irrigation method — drip, sprinkler, or flood?",
    ],
    "pest_diagnosis": [
        "Which crop is affected and since when have you noticed the symptoms?",
        "Can you describe the colour and pattern of the spots more clearly?",
    ],
    "mandi_price": [
        "Which crop and which market are you asking about?",
        "Are you looking for today's spot price or the 30-day trend?",
    ],
    "pm_kisan_eligibility": [
        "Can you share your total landholding in acres and your state?",
        "Are you already registered in the land records of your state?",
    ],
    "crop_calendar": [
        "Which district and state is your farm located in?",
        "What is your current water availability — rainfed or irrigated?",
    ],
}

USER_CLARIFICATIONS: dict[str, list[str]] = {
    "irrigation_advisory": [
        "I grow tomatoes, about 1 acre, I use drip irrigation",
        "2 acre wheat field, flood irrigation system",
    ],
    "pest_diagnosis": [
        "It is brinjal, I noticed it 3 days ago, yellow patches on lower leaves",
        "Chilli plants, since last week, leaves curling inward and sticky",
    ],
    "mandi_price": [
        "Wheat at Ludhiana mandi, today's price please",
        "Onion price trend for Nashik over last 30 days",
    ],
    "pm_kisan_eligibility": [
        "I have 2.5 acres in Kolar district Karnataka, yes registered in RTC",
        "1.8 acres in Belagavi, Karnataka, name is in 7/12 extract",
    ],
    "crop_calendar": [
        "My farm is in Hassan district Karnataka, red loamy soil, well irrigation",
        "Guntur district Andhra Pradesh, canal irrigation available",
    ],
}

ASSISTANT_ANSWERS: dict[str, list[str]] = {
    "irrigation_advisory": [
        "Based on the current soil moisture and weather forecast, I recommend irrigating your field tomorrow morning with approximately 500 litres per acre.",
        "Your soil moisture is adequate for now. No irrigation is needed for the next 2 days — check again after the expected rainfall.",
        "The 3-day forecast shows 18 mm of rain. Hold off on irrigation until after the rain to avoid waterlogging.",
    ],
    "pest_diagnosis": [
        "The symptoms indicate an aphid infestation. Spray neem oil solution at 5 ml per litre of water every 7 days for 3 weeks.",
        "This is fungal blight. Apply copper-based fungicide every 7 days for 3 weeks. Remove and burn heavily infected leaves.",
        "Your crop has whitefly attack. Use yellow sticky traps and apply imidacloprid at 0.5 ml per litre.",
    ],
    "mandi_price": [
        "Current market price is around ₹2 150 per quintal. The 7-day trend is slightly rising — good time to sell if storage is not an issue.",
        "Today's mandi rate is ₹1 890 per quintal. The 30-day average is ₹1 980, so prices are slightly below average. Consider holding for 2–3 more days.",
        "Price has been stable at ₹2 300 per quintal for the past week. If you need liquidity, this is a reasonable time to sell.",
    ],
    "pm_kisan_eligibility": [
        "Based on your landholding details, you appear eligible for PM-Kisan. Register at pmkisan.gov.in with your Aadhaar and bank account details.",
        "You meet all eligibility criteria. Your next installment of ₹2 000 should arrive by December 2024 in your linked bank account.",
        "Unfortunately your land area exceeds the 5-acre smallholder limit. You are not eligible for PM-Kisan under current guidelines.",
    ],
    "crop_calendar": [
        "In your agro-climatic zone, the best time to sow paddy is between 15 June and 15 July. Use a short-duration variety like BPT 5204 for your region.",
        "For your district, ragi sowing should begin by 15 October for the rabi season to get optimal yield before the dry spell.",
        "You can grow maize this kharif season. Sow in the first week of June and expect harvest by late October.",
    ],
}

INCONSISTENT_ANSWERS: dict[str, str] = {
    "irrigation_advisory": "Your soil has plenty of moisture and no irrigation is needed for at least 10 days.",
    "pest_diagnosis": "Your crop looks completely healthy. No pest or disease treatment is required at this time.",
    "mandi_price": "Prices have crashed below ₹500 per quintal — sell all your stock immediately before it drops further.",
    "pm_kisan_eligibility": "You are definitely not eligible for PM-Kisan under any circumstances, regardless of your landholding.",
    "crop_calendar": "This is the worst possible season to sow any crop in your region. Please wait at least 6 months before attempting any planting.",
}

# ---------------------------------------------------------------------------
# Tool arg / output factories
# ---------------------------------------------------------------------------

def _tool_args(tool: str) -> dict:
    return {
        "get_weather": {
            "location": random.choice(DISTRICTS),
            "days": random.randint(3, 7),
        },
        "get_soil_moisture": {
            "field_id": _uuid()[:8],
            "depth_cm": random.choice([10, 20, 30]),
        },
        "irrigation_recommendation": {
            "crop": random.choice(["tomato", "wheat", "sugarcane", "rice", "brinjal"]),
            "moisture_level": round(random.uniform(20, 60), 1),
        },
        "identify_pest": {
            "crop": random.choice(["brinjal", "chilli", "tomato", "cotton", "paddy"]),
            "symptom": random.choice(["yellow spots", "curling leaves", "white powder", "black insects", "brown sheath"]),
        },
        "recommend_treatment": {
            "pest": random.choice(["aphid", "whitefly", "fungal blight", "leaf miner", "stem borer"]),
            "crop": random.choice(["brinjal", "chilli", "tomato", "cotton"]),
        },
        "get_mandi_price": {
            "commodity": random.choice(["wheat", "onion", "cotton", "paddy", "soybean", "maize"]),
            "market": random.choice(DISTRICTS),
        },
        "get_price_trend": {
            "commodity": random.choice(["wheat", "onion", "cotton", "paddy", "soybean"]),
            "days": random.choice([7, 14, 30]),
        },
        "check_landholding": {
            "aadhaar_last4": str(random.randint(1000, 9999)),
            "state": random.choice(["Karnataka", "Maharashtra", "Punjab", "Andhra Pradesh", "Telangana"]),
        },
        "verify_pm_kisan_eligibility": {
            "land_area_acres": round(random.uniform(0.5, 5.0), 1),
            "state": random.choice(["Karnataka", "Maharashtra", "Punjab", "Andhra Pradesh"]),
        },
        "get_agro_climatic_zone": {
            "district": random.choice(DISTRICTS),
            "state": random.choice(["Karnataka", "Maharashtra", "Punjab", "Andhra Pradesh"]),
        },
        "crop_calendar": {
            "zone": random.choice(["Southern Dry Zone", "Northern Plains", "Deccan Plateau", "Eastern Coastal"]),
            "season": random.choice(["kharif", "rabi", "zaid"]),
        },
    }.get(tool, {"query": tool})


def _tool_output(tool: str, error: bool = False) -> dict:
    if error:
        return {
            "status": "error",
            "message": random.choice([
                "Service temporarily unavailable — please retry",
                "Invalid input parameters for the given region",
                "Data not found for the specified location and date",
            ]),
        }
    return {
        "get_weather": {
            "forecast": [
                {"day": i + 1, "rain_mm": random.randint(0, 25), "temp_max": random.randint(28, 42)}
                for i in range(3)
            ],
            "rain_mm": random.randint(0, 30),
            "temp_max": random.randint(28, 42),
            "humidity_pct": random.randint(40, 90),
        },
        "get_soil_moisture": {
            "moisture_percent": random.randint(15, 80),
            "status": random.choice(["dry", "moderate", "wet"]),
            "depth_cm": 20,
        },
        "irrigation_recommendation": {
            "recommend": random.choice(["irrigate_now", "wait_2_days", "no_irrigation_needed"]),
            "amount_liters_per_acre": random.randint(300, 800),
            "next_check_days": random.randint(1, 4),
        },
        "identify_pest": {
            "pest": random.choice(["aphid", "whitefly", "leaf miner", "fungal blight", "stem borer"]),
            "confidence": round(random.uniform(0.72, 0.97), 2),
            "crop_affected": random.choice(["brinjal", "chilli", "tomato", "cotton"]),
            "severity": random.choice(["mild", "moderate", "severe"]),
        },
        "recommend_treatment": {
            "treatment": random.choice([
                "spray neem oil 5ml/L",
                "apply imidacloprid 0.5ml/L",
                "use copper fungicide 3g/L",
                "spray chlorpyrifos 2ml/L",
            ]),
            "dosage": f"{random.randint(3, 10)} ml/L",
            "frequency": random.choice(["once a week", "every 3 days", "twice a month"]),
            "safety_period_days": random.randint(7, 21),
        },
        "get_mandi_price": {
            "price_per_quintal": random.randint(1200, 4500),
            "market": random.choice(DISTRICTS),
            "date": (datetime.utcnow() - timedelta(days=random.randint(0, 2))).strftime("%Y-%m-%d"),
            "unit": "INR/quintal",
        },
        "get_price_trend": {
            "trend": random.choice(["rising", "falling", "stable"]),
            "7day_avg": random.randint(1800, 3500),
            "30day_avg": random.randint(1700, 3400),
            "change_pct": round(random.uniform(-10.0, 15.0), 1),
        },
        "check_landholding": {
            "area_acres": round(random.uniform(0.5, 5.0), 1),
            "verified": random.choice([True, True, False]),
            "land_type": random.choice(["agricultural", "dry land", "irrigated"]),
        },
        "verify_pm_kisan_eligibility": {
            "eligible": random.choice([True, True, False]),
            "reason": random.choice([
                "meets all eligibility criteria",
                "landholding within 5-acre smallholder limit",
                "not registered in state land records",
            ]),
            "next_installment": "December 2024",
        },
        "get_agro_climatic_zone": {
            "zone": random.choice(["Southern Dry Zone", "Northern Plains", "Deccan Plateau", "Eastern Coastal"]),
            "state": random.choice(["Karnataka", "Maharashtra", "Punjab"]),
            "annual_rainfall_mm": random.randint(500, 1200),
        },
        "crop_calendar": {
            "crop": random.choice(["paddy", "ragi", "maize", "groundnut", "sunflower"]),
            "sow_start": random.choice(["1 June", "15 June", "1 July", "15 October"]),
            "sow_end": random.choice(["15 July", "1 August", "30 November"]),
            "harvest": random.choice(["October", "November", "February", "March"]),
            "water_requirement": random.choice(["low", "medium", "high"]),
        },
    }.get(tool, {"result": "data unavailable"})


# ---------------------------------------------------------------------------
# PII injection
# ---------------------------------------------------------------------------

def _inject_pii(text: str) -> tuple[str, dict[str, bool]]:
    """
    Randomly sprinkle PII into a user message using the seeded global random.
    Returns the (possibly modified) text and a dict of which PII types were injected.
    """
    injected: dict[str, bool] = {
        "name": False, "phone": False, "aadhaar": False, "district": False, "gps": False,
    }

    if random.random() < 0.30:
        name = fake.first_name()
        text = f"My name is {name} and I have a question. " + text
        injected["name"] = True

    if random.random() < 0.15:
        phone = random.choice([
            f"+91 {random.randint(7000000000, 9999999999)}",
            f"+91-{random.randint(7000000000, 9999999999)}",
            f"{random.randint(7000000000, 9999999999)}",
            f"{random.randint(70000, 99999)} {random.randint(10000, 99999)}",
        ])
        text += f" You can reach me at {phone}."
        injected["phone"] = True

    if random.random() < 0.05:
        raw = random.randint(100000000000, 999999999999)
        if random.random() < 0.5:
            aadhaar = f"{str(raw)[:4]} {str(raw)[4:8]} {str(raw)[8:]}"
        else:
            aadhaar = str(raw)
        text += f" My Aadhaar number is {aadhaar}."
        injected["aadhaar"] = True

    if random.random() < 0.20:
        district = random.choice(DISTRICTS)
        text += f" I am from {district} district."
        injected["district"] = True

    if random.random() < 0.03:
        lat = round(random.uniform(6.0, 37.0), 4)
        lng = round(random.uniform(68.0, 97.0), 4)
        text += f" My field is at {lat}, {lng}."
        injected["gps"] = True

    return text, injected


# ---------------------------------------------------------------------------
# Session generation
# ---------------------------------------------------------------------------

def generate_session(
    workflow: str,
    session_type: str,
    session_start: datetime,
) -> tuple[list[RawEvent], dict[str, bool]]:
    """
    Build a list of RawEvents for one session.

    Returns (events, pii_flags) where pii_flags records which PII types appeared.
    Each RawEvent is validated at construction time — any schema mismatch crashes.
    """
    session_id = _uuid()
    tools = WORKFLOWS[workflow]
    events: list[RawEvent] = []
    ts = session_start
    all_pii: dict[str, bool] = {k: False for k in ("name", "phone", "aadhaar", "district", "gps")}

    def tick() -> datetime:
        nonlocal ts
        ts += timedelta(seconds=random.randint(5, 45))
        return ts

    def evt(**kwargs) -> RawEvent:
        return RawEvent(session_id=session_id, event_id=_uuid(), **kwargs)

    is_codemix = session_type == "codemix"
    is_inconsistent = session_type == "inconsistent"

    # ---- system prompt ----
    events.append(evt(
        timestamp=ts,
        role=EventRole.SYSTEM,
        content=(
            f"You are a helpful agricultural assistant for Indian farmers. "
            f"Current workflow: {workflow}. "
            "Answer in simple language and recommend practical actions."
        ),
    ))

    # ---- initial user message ----
    templates = HINGLISH_TEMPLATES[workflow] if is_codemix else USER_TEMPLATES[workflow]
    user_text, pii = _inject_pii(random.choice(templates))
    for k, v in pii.items():
        all_pii[k] = all_pii[k] or v

    events.append(evt(timestamp=tick(), role=EventRole.USER, content=user_text))

    # ---- clarification sub-turn ----
    if session_type == "clarify":
        events.append(evt(
            timestamp=tick(),
            role=EventRole.ASSISTANT,
            content=random.choice(CLARIFYING_QUESTIONS[workflow]),
        ))
        clarify_text, pii = _inject_pii(random.choice(USER_CLARIFICATIONS[workflow]))
        for k, v in pii.items():
            all_pii[k] = all_pii[k] or v
        events.append(evt(timestamp=tick(), role=EventRole.USER, content=clarify_text))

    # ---- error-recovery first tool attempt ----
    remaining_tools = list(tools)
    if session_type == "recovery":
        first_tool = remaining_tools[0]
        remaining_tools = remaining_tools[1:]

        events.append(evt(
            timestamp=tick(),
            role=EventRole.TOOL_CALL,
            tool_name=first_tool,
            tool_args=_tool_args(first_tool),
        ))

        if random.random() < 0.5:
            # hard ERROR event
            events.append(evt(
                timestamp=tick(),
                role=EventRole.ERROR,
                error_message=f"Tool '{first_tool}' failed: service timeout after 30 s",
            ))
        else:
            # tool_result carrying an error payload
            events.append(evt(
                timestamp=tick(),
                role=EventRole.TOOL_RESULT,
                tool_name=first_tool,
                tool_output=_tool_output(first_tool, error=True),
            ))

        # retry with adjusted args
        retry_args = _tool_args(first_tool)
        retry_args["retry"] = True
        events.append(evt(
            timestamp=tick(),
            role=EventRole.TOOL_CALL,
            tool_name=first_tool,
            tool_args=retry_args,
        ))
        events.append(evt(
            timestamp=tick(),
            role=EventRole.TOOL_RESULT,
            tool_name=first_tool,
            tool_output=_tool_output(first_tool, error=False),
        ))

    # ---- normal tool calls for remaining tools ----
    tool_outputs: list[dict] = []
    for tool in remaining_tools:
        events.append(evt(
            timestamp=tick(),
            role=EventRole.TOOL_CALL,
            tool_name=tool,
            tool_args=_tool_args(tool),
        ))
        output = _tool_output(tool)
        tool_outputs.append(output)
        events.append(evt(
            timestamp=tick(),
            role=EventRole.TOOL_RESULT,
            tool_name=tool,
            tool_output=output,
        ))

    # ---- final assistant answer ----
    if is_inconsistent:
        answer = INCONSISTENT_ANSWERS[workflow]
        meta = {"inject_inconsistency": True}
    else:
        answer = random.choice(ASSISTANT_ANSWERS[workflow])
        meta = {}

    events.append(evt(
        timestamp=tick(),
        role=EventRole.ASSISTANT,
        content=answer,
        metadata=meta,
    ))

    return events, all_pii


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    output_dir = Path("data/synthetic_logs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "raw_logs.jsonl"

    # Build balanced session spec: 60 per workflow × 5 workflows = 300
    # Type distribution per workflow: 36 happy / 9 clarify / 6 recovery / 6 codemix / 3 inconsistent
    type_counts = {"happy": 36, "clarify": 9, "recovery": 6, "codemix": 6, "inconsistent": 3}
    session_specs: list[tuple[str, str]] = []
    for workflow in WORKFLOWS:
        types: list[str] = []
        for stype, count in type_counts.items():
            types.extend([stype] * count)
        random.shuffle(types)
        for stype in types:
            session_specs.append((workflow, stype))
    random.shuffle(session_specs)

    # Spread session start times across last 30 days
    now = datetime.utcnow()
    window_secs = int(timedelta(days=30).total_seconds())

    # Stats accumulators
    total_events = 0
    workflow_dist: dict[str, int] = {w: 0 for w in WORKFLOWS}
    type_dist: dict[str, int] = {t: 0 for t in type_counts}
    pii_totals: dict[str, int] = {"name": 0, "phone": 0, "aadhaar": 0, "district": 0, "gps": 0}
    sessions_with_any_pii = 0

    with output_path.open("w", encoding="utf-8") as fh:
        for workflow, session_type in session_specs:
            session_start = now - timedelta(seconds=random.randint(0, window_secs))
            events, pii_flags = generate_session(workflow, session_type, session_start)

            for event in events:
                fh.write(event.model_dump_json() + "\n")

            total_events += len(events)
            workflow_dist[workflow] += 1
            type_dist[session_type] += 1

            if any(pii_flags.values()):
                sessions_with_any_pii += 1
            for k, v in pii_flags.items():
                if v:
                    pii_totals[k] += 1

    total_sessions = len(session_specs)
    avg_events = total_events / total_sessions

    print(f"\n{'=' * 56}")
    print("  Synthetic log generation complete")
    print(f"{'=' * 56}")
    print(f"  Output file     : {output_path.resolve()}")
    print(f"  Total sessions  : {total_sessions}")
    print(f"  Total events    : {total_events}")
    print(f"  Avg events/sess : {avg_events:.1f}")

    print(f"\n  Workflow distribution:")
    for w, c in workflow_dist.items():
        print(f"    {w:<30}  {c:>3}  ({c / total_sessions * 100:.0f}%)")

    print(f"\n  Session-type distribution:")
    for t, c in type_dist.items():
        print(f"    {t:<15}  {c:>3}  ({c / total_sessions * 100:.0f}%)")

    print(f"\n  PII injection (sessions that contained each type):")
    print(f"    Any PII        : {sessions_with_any_pii:>3}  ({sessions_with_any_pii / total_sessions * 100:.0f}%)")
    for k, c in pii_totals.items():
        print(f"    {k:<14} : {c:>3}  ({c / total_sessions * 100:.0f}%)")
    print()


if __name__ == "__main__":
    main()
