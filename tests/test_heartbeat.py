"""Tests unitarios de HeartbeatConfig + helpers (Fase 5)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from memgpt.heartbeat import (
    DEFAULT_AUTO_CONTINUE_TOOLS,
    HeartbeatConfig,
    HeartbeatMode,
    extract_tool_call_keys,
    loop_repetition_count,
    tool_call_key,
)


def test_default_config_is_native_with_safety_net():
    cfg = HeartbeatConfig()
    assert cfg.mode == HeartbeatMode.NATIVE
    assert cfg.max_chained_heartbeats == 50
    assert cfg.turn_timeout_seconds == 300
    assert cfg.loop_detection_threshold == 3
    assert "core_memory_append" in cfg.auto_continue_tools
    assert "conversation_search" in cfg.auto_continue_tools


def test_positive_int_validators():
    with pytest.raises(ValidationError):
        HeartbeatConfig(max_chained_heartbeats=0)
    with pytest.raises(ValidationError):
        HeartbeatConfig(turn_timeout_seconds=-1)
    with pytest.raises(ValidationError):
        HeartbeatConfig(loop_detection_threshold=0)


def test_auto_continue_tools_accepts_set_list_or_frozenset():
    cfg1 = HeartbeatConfig(auto_continue_tools={"a", "b"})
    cfg2 = HeartbeatConfig(auto_continue_tools=["a", "b"])
    cfg3 = HeartbeatConfig(auto_continue_tools=frozenset({"a", "b"}))
    assert cfg1.auto_continue_tools == cfg2.auto_continue_tools == cfg3.auto_continue_tools


def test_auto_continue_tools_rejects_garbage():
    with pytest.raises(ValidationError):
        HeartbeatConfig(auto_continue_tools=42)


def test_tool_call_key_is_stable_across_arg_order():
    k1 = tool_call_key("search", {"q": "x", "limit": 5})
    k2 = tool_call_key("search", {"limit": 5, "q": "x"})
    assert k1 == k2


def test_tool_call_key_distinguishes_different_args():
    assert tool_call_key("search", {"q": "x"}) != tool_call_key("search", {"q": "y"})
    assert tool_call_key("a", {}) != tool_call_key("b", {})


def test_tool_call_key_handles_unserializable_args():
    class NotJsonable:
        def __str__(self):
            return "<obj>"

    k = tool_call_key("t", {"x": NotJsonable()})
    assert k.startswith("t::")


def test_extract_tool_call_keys_for_parallel_calls():
    calls = [
        {"name": "a", "args": {"x": 1}},
        {"name": "b", "args": {"y": 2}},
    ]
    keys = extract_tool_call_keys(calls)
    assert len(keys) == 2
    assert keys[0].startswith("a::")
    assert keys[1].startswith("b::")


def test_loop_repetition_count_simple_buffer():
    buf = ["a", "b", "a", "a"]
    assert loop_repetition_count(buf, "a") == 3
    assert loop_repetition_count(buf, "c") == 0


def test_default_constants_match():
    cfg = HeartbeatConfig()
    assert cfg.auto_continue_tools == DEFAULT_AUTO_CONTINUE_TOOLS
