import pytest
from datetime import datetime
from pydantic import ValidationError
from src.schemas import RawEvent, EventRole, Trajectory


@pytest.fixture
def valid_user_event():
    return RawEvent(
        event_id="e1",
        session_id="s1",
        timestamp=datetime(2024, 1, 1),
        role=EventRole.USER,
        content="hello",
    )


def _base(**kwargs):
    """Minimal required fields for RawEvent, overridable via kwargs."""
    return {
        "event_id": "e1",
        "session_id": "s1",
        "timestamp": datetime(2024, 1, 1),
        **kwargs,
    }


def _base_trajectory(valid_user_event, **kwargs):
    """Minimal required fields for Trajectory, overridable via kwargs."""
    return {
        "trajectory_id": "t1",
        "session_id": "s1",
        "events": [valid_user_event],
        "tools_used": [],
        "step_count": 0,
        "had_error_recovery": False,
        "complexity_bucket": "simple",
        "language_mix": "en",
        **kwargs,
    }


class TestRawEventValidation:

    def test_valid_user_event(self, valid_user_event):
        assert valid_user_event.role == EventRole.USER
        assert valid_user_event.content == "hello"

    def test_valid_tool_call(self):
        event = RawEvent(**_base(role=EventRole.TOOL_CALL, tool_name="get_weather", tool_args={"city": "Mysuru"}))
        assert event.role == EventRole.TOOL_CALL

    def test_valid_tool_result(self):
        event = RawEvent(**_base(role=EventRole.TOOL_RESULT, tool_name="get_weather", tool_output={"temp": 30}))
        assert event.role == EventRole.TOOL_RESULT

    def test_valid_error_event(self):
        event = RawEvent(**_base(role=EventRole.ERROR, error_message="timeout after 30s"))
        assert event.error_message == "timeout after 30s"

    def test_user_without_content_fails(self):
        with pytest.raises(ValidationError):
            RawEvent(**_base(role=EventRole.USER, content=None))

    def test_tool_call_without_name_fails(self):
        with pytest.raises(ValidationError):
            RawEvent(**_base(role=EventRole.TOOL_CALL, tool_args={"x": 1}))

    def test_tool_call_without_args_fails(self):
        with pytest.raises(ValidationError):
            RawEvent(**_base(role=EventRole.TOOL_CALL, tool_name="get_weather"))

    def test_tool_result_without_output_fails(self):
        with pytest.raises(ValidationError):
            RawEvent(**_base(role=EventRole.TOOL_RESULT, tool_name="get_weather"))

    def test_error_without_message_fails(self):
        with pytest.raises(ValidationError):
            RawEvent(**_base(role=EventRole.ERROR))

    def test_invalid_role_string_fails(self):
        with pytest.raises(ValidationError):
            RawEvent(**_base(role="bogus_role"))


class TestTrajectoryConstruction:

    def test_valid_trajectory(self, valid_user_event):
        traj = Trajectory(**_base_trajectory(valid_user_event))
        assert traj.trajectory_id == "t1"
        assert traj.complexity_bucket == "simple"

    def test_invalid_complexity_bucket_fails(self, valid_user_event):
        with pytest.raises(ValidationError):
            Trajectory(**_base_trajectory(valid_user_event, complexity_bucket="trivial"))

    def test_invalid_language_mix_fails(self, valid_user_event):
        with pytest.raises(ValidationError):
            Trajectory(**_base_trajectory(valid_user_event, language_mix="french"))

    def test_negative_step_count_fails(self, valid_user_event):
        with pytest.raises(ValidationError):
            Trajectory(**_base_trajectory(valid_user_event, step_count=-1))
