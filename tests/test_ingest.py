import json
import uuid
import pytest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.ingest import (
    load_raw_events,
    group_into_sessions,
    session_to_trajectory,
    ingest,
    _classify_domain,
    _detect_language,
    _compute_complexity,
    _detect_error_recovery,
)
from src.schemas import RawEvent, EventRole


@pytest.fixture
def make_event():
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)

    def _make(role, session_id="s1", offset_sec=0, **kwargs):
        return RawEvent(
            event_id=str(uuid.uuid4()),
            session_id=session_id,
            timestamp=base + timedelta(seconds=offset_sec),
            role=role,
            **kwargs,
        )

    return _make


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, events: list[RawEvent]) -> None:
    path.write_text("\n".join(e.model_dump_json() for e in events) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# TestLoadRawEvents
# ---------------------------------------------------------------------------

class TestLoadRawEvents:

    def test_loads_valid_events(self, tmp_path, make_event):
        events = [make_event(EventRole.USER, content="hello", offset_sec=i) for i in range(3)]
        input_path = tmp_path / "in.jsonl"
        _write_jsonl(input_path, events)

        result = list(load_raw_events(input_path, tmp_path / "quarantine.jsonl"))
        assert len(result) == 3

    def test_quarantines_malformed_json(self, tmp_path, make_event):
        events = [make_event(EventRole.USER, content="ok", offset_sec=i) for i in range(2)]
        input_path = tmp_path / "in.jsonl"
        with input_path.open("w", encoding="utf-8") as fh:
            for e in events:
                fh.write(e.model_dump_json() + "\n")
            fh.write("this is not json at all\n")

        quarantine = tmp_path / "quarantine.jsonl"
        result = list(load_raw_events(input_path, quarantine))

        assert len(result) == 2
        assert quarantine.exists()
        lines = [l for l in quarantine.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 1

    def test_quarantines_schema_violations(self, tmp_path, make_event):
        valid = make_event(EventRole.USER, content="valid")
        bad = json.dumps({
            "event_id": "x",
            "session_id": "s1",
            "timestamp": "2024-01-01T00:00:00",
            "role": "tool_call",
            "tool_args": {"a": 1},
            # tool_name intentionally omitted → ValidationError
        })
        input_path = tmp_path / "in.jsonl"
        input_path.write_text(valid.model_dump_json() + "\n" + bad + "\n", encoding="utf-8")

        quarantine = tmp_path / "quarantine.jsonl"
        result = list(load_raw_events(input_path, quarantine))

        assert len(result) == 1
        lines = [l for l in quarantine.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 1

    def test_empty_file_yields_nothing(self, tmp_path):
        input_path = tmp_path / "in.jsonl"
        input_path.write_text("", encoding="utf-8")

        result = list(load_raw_events(input_path, tmp_path / "quarantine.jsonl"))
        assert result == []


# ---------------------------------------------------------------------------
# TestGroupIntoSessions
# ---------------------------------------------------------------------------

class TestGroupIntoSessions:

    def test_groups_by_session_id(self, make_event):
        events = (
            [make_event(EventRole.USER, session_id="s1", content="hi", offset_sec=i) for i in range(3)]
            + [make_event(EventRole.USER, session_id="s2", content="hi", offset_sec=i) for i in range(3)]
        )
        sessions = group_into_sessions(events)

        assert set(sessions.keys()) == {"s1", "s2"}
        assert len(sessions["s1"]) == 3
        assert len(sessions["s2"]) == 3

    def test_sorts_by_timestamp(self, make_event):
        events = [
            make_event(EventRole.USER, content="c", offset_sec=20),
            make_event(EventRole.USER, content="a", offset_sec=5),
            make_event(EventRole.USER, content="b", offset_sec=10),
        ]
        sessions = group_into_sessions(events)
        timestamps = [e.timestamp for e in sessions["s1"]]

        assert timestamps == sorted(timestamps)


# ---------------------------------------------------------------------------
# TestComplexityComputation
# ---------------------------------------------------------------------------

class TestComplexityComputation:

    def test_simple_zero_steps(self):
        assert _compute_complexity(0, False) == "simple"

    def test_simple_one_step(self):
        assert _compute_complexity(1, False) == "simple"

    def test_moderate_two_steps(self):
        assert _compute_complexity(2, False) == "moderate"

    def test_moderate_three_steps(self):
        assert _compute_complexity(3, False) == "moderate"

    def test_complex_four_steps(self):
        assert _compute_complexity(4, False) == "complex"

    def test_complex_via_recovery_with_low_steps(self):
        assert _compute_complexity(2, True) == "complex"

    def test_complex_via_recovery_with_zero_steps(self):
        assert _compute_complexity(0, True) == "complex"


# ---------------------------------------------------------------------------
# TestDomainClassification
# ---------------------------------------------------------------------------

class TestDomainClassification:

    def test_irrigation(self):
        assert _classify_domain(["get_weather", "get_soil_moisture"]) == "irrigation"

    def test_pest(self):
        assert _classify_domain(["identify_pest", "recommend_treatment"]) == "pest"

    def test_price(self):
        assert _classify_domain(["get_mandi_price"]) == "price"

    def test_scheme(self):
        assert _classify_domain(["verify_pm_kisan_eligibility"]) == "scheme"

    def test_calendar(self):
        assert _classify_domain(["crop_calendar"]) == "calendar"

    def test_mixed_picks_majority(self):
        assert _classify_domain(["get_weather", "get_weather", "get_mandi_price"]) == "irrigation"

    def test_empty_returns_general_qa(self):
        assert _classify_domain([]) == "general_qa"

    def test_unknown_tool_returns_general_qa(self):
        assert _classify_domain(["some_unknown_tool"]) == "general_qa"


# ---------------------------------------------------------------------------
# TestLanguageDetection
# ---------------------------------------------------------------------------

class TestLanguageDetection:

    def test_pure_english(self, make_event):
        events = [make_event(EventRole.USER, content="What is the weather today")]
        assert _detect_language(events) == "en"

    def test_code_mixed_hinglish(self, make_event):
        events = [make_event(EventRole.USER, content="mere field mein paani kab dena hai")]
        assert _detect_language(events) == "code_mixed"

    def test_devanagari_returns_hi(self, make_event):
        events = [make_event(EventRole.USER, content="मेरे खेत में पानी की ज़रूरत है")]
        assert _detect_language(events) == "hi"

    def test_devanagari_beats_codemix(self, make_event):
        events = [make_event(EventRole.USER, content="mere खेत mein paani")]
        assert _detect_language(events) == "hi"

    def test_no_user_events_returns_en(self, make_event):
        events = [make_event(EventRole.SYSTEM, content="You are a helpful assistant.")]
        assert _detect_language(events) == "en"


# ---------------------------------------------------------------------------
# TestErrorRecoveryDetection
# ---------------------------------------------------------------------------

class TestErrorRecoveryDetection:

    def test_no_error_returns_false(self, make_event):
        events = [
            make_event(EventRole.TOOL_CALL, offset_sec=0, tool_name="x", tool_args={}),
            make_event(EventRole.TOOL_RESULT, offset_sec=10, tool_name="x", tool_output={"ok": True}),
        ]
        assert _detect_error_recovery(events) is False

    def test_error_followed_by_retry_returns_true(self, make_event):
        events = [
            make_event(EventRole.TOOL_CALL, offset_sec=0, tool_name="x", tool_args={}),
            make_event(EventRole.ERROR, offset_sec=10, error_message="failed"),
            make_event(EventRole.TOOL_CALL, offset_sec=20, tool_name="x", tool_args={}),
        ]
        assert _detect_error_recovery(events) is True

    def test_error_at_end_returns_false(self, make_event):
        events = [
            make_event(EventRole.TOOL_CALL, offset_sec=0, tool_name="x", tool_args={}),
            make_event(EventRole.ERROR, offset_sec=10, error_message="failed"),
        ]
        assert _detect_error_recovery(events) is False

    def test_error_status_in_tool_result_then_retry(self, make_event):
        events = [
            make_event(EventRole.TOOL_CALL, offset_sec=0, tool_name="x", tool_args={}),
            make_event(EventRole.TOOL_RESULT, offset_sec=10, tool_name="x", tool_output={"status": "error"}),
            make_event(EventRole.TOOL_CALL, offset_sec=20, tool_name="x", tool_args={}),
        ]
        assert _detect_error_recovery(events) is True


# ---------------------------------------------------------------------------
# TestTrajectoryConstruction
# ---------------------------------------------------------------------------

class TestTrajectoryConstruction:

    def test_tools_used_deduplicated(self, make_event):
        events = [
            make_event(EventRole.TOOL_CALL, offset_sec=i, tool_name="get_weather", tool_args={"x": 1})
            for i in range(3)
        ]
        traj = session_to_trajectory("s1", events)
        assert traj.tools_used == ["get_weather"]

    def test_step_count_counts_only_tool_calls(self, make_event):
        events = [
            make_event(EventRole.USER, offset_sec=0, content="question"),
            make_event(EventRole.TOOL_CALL, offset_sec=1, tool_name="get_weather", tool_args={}),
            make_event(EventRole.TOOL_RESULT, offset_sec=2, tool_name="get_weather", tool_output={"temp": 30}),
            make_event(EventRole.TOOL_CALL, offset_sec=3, tool_name="get_soil_moisture", tool_args={}),
            make_event(EventRole.TOOL_RESULT, offset_sec=4, tool_name="get_soil_moisture", tool_output={"pct": 40}),
            make_event(EventRole.ASSISTANT, offset_sec=5, content="done"),
        ]
        traj = session_to_trajectory("s1", events)
        assert traj.step_count == 2

    def test_trajectory_id_format(self, make_event):
        events = [make_event(EventRole.USER, content="hello")]
        traj = session_to_trajectory("s1", events)
        assert traj.trajectory_id.startswith("traj_")


# ---------------------------------------------------------------------------
# TestIngestEndToEnd
# ---------------------------------------------------------------------------

class TestIngestEndToEnd:

    def test_full_ingest_creates_trajectories(self, tmp_path, make_event):
        s1_events = [make_event(EventRole.USER, session_id="s1", content="q", offset_sec=i) for i in range(3)]
        s2_events = [make_event(EventRole.USER, session_id="s2", content="q", offset_sec=i) for i in range(3)]

        input_path = tmp_path / "raw.jsonl"
        _write_jsonl(input_path, s1_events + s2_events)

        output_path = tmp_path / "out" / "trajectories.jsonl"
        stats = ingest(input_path, output_path)

        lines = [l for l in output_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 2
        assert stats["sessions_processed"] == 2
        assert stats["events_processed"] == 6
        assert stats["quarantined"] == 0
