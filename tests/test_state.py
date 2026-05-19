import pytest
from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from memgpt.state import MemGPTState


def test_default_state_has_assistant_and_human_blocks():
    s = MemGPTState()
    assert s.messages == []
    assert s.recursive_summary is None
    assert s.core_memory.has("assistant")
    assert s.core_memory.has("human")
    assert s.step_count == 0
    assert s.memory_pressure_alerted is False
    assert s.evicted_count == 0


def test_step_count_must_be_non_negative():
    with pytest.raises(ValidationError):
        MemGPTState(step_count=-1)


def test_evicted_count_must_be_non_negative():
    with pytest.raises(ValidationError):
        MemGPTState(evicted_count=-5)


def test_messages_accept_langchain_message_objects():
    s = MemGPTState(messages=[HumanMessage(content="hi")])
    assert len(s.messages) == 1
    assert s.messages[0].content == "hi"
