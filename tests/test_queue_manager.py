"""Unit tests del Queue Manager: config, atomic blocks, eviction selection."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from memgpt.queue_manager import (
    QueueManagerConfig,
    count_messages_tokens,
    count_state_tokens,
    group_into_atomic_blocks,
    select_blocks_to_evict,
)


def _ai_with_tools(content: str, calls: list[tuple[str, str]]) -> AIMessage:
    """Build an AIMessage carrying tool_calls. `calls` is [(id, name)]."""
    return AIMessage(
        content=content,
        tool_calls=[
            {"id": cid, "name": name, "args": {}, "type": "tool_call"}
            for cid, name in calls
        ],
    )


def test_thresholds_must_be_in_unit_interval():
    with pytest.raises(ValueError):
        QueueManagerConfig(warning_threshold=0)
    with pytest.raises(ValueError):
        QueueManagerConfig(flush_threshold=1.5)
    with pytest.raises(ValueError):
        QueueManagerConfig(flush_eviction_ratio=-0.1)


def test_warning_must_be_strictly_below_flush():
    with pytest.raises(ValueError):
        QueueManagerConfig(warning_threshold=0.9, flush_threshold=0.9)
    with pytest.raises(ValueError):
        QueueManagerConfig(warning_threshold=0.95, flush_threshold=0.5)


def test_threshold_helpers_are_consistent():
    cfg = QueueManagerConfig(
        warning_threshold=0.7,
        flush_threshold=1.0,
        flush_eviction_ratio=0.5,
        context_window_tokens=1000,
    )
    assert cfg.warning_tokens == 700
    assert cfg.flush_tokens == 1000
    assert cfg.target_eviction_tokens == 500


def test_group_keeps_tool_call_pair_atomic():
    msgs = [
        HumanMessage(content="hello", id="h1"),
        _ai_with_tools("", [("c1", "search")]),
        ToolMessage(content="result", tool_call_id="c1", id="t1"),
        AIMessage(content="final", id="a1"),
    ]
    blocks = group_into_atomic_blocks(msgs)
    assert len(blocks) == 3
    assert isinstance(blocks[0][0], HumanMessage)
    assert isinstance(blocks[1][0], AIMessage) and blocks[1][0].tool_calls
    assert isinstance(blocks[1][1], ToolMessage)
    assert isinstance(blocks[2][0], AIMessage) and not blocks[2][0].tool_calls


def test_group_keeps_parallel_tool_calls_atomic():
    msgs = [
        HumanMessage(content="hi", id="h1"),
        _ai_with_tools("", [("c1", "f"), ("c2", "g"), ("c3", "h")]),
        ToolMessage(content="r1", tool_call_id="c1", id="t1"),
        ToolMessage(content="r2", tool_call_id="c2", id="t2"),
        ToolMessage(content="r3", tool_call_id="c3", id="t3"),
        AIMessage(content="ok", id="a1"),
    ]
    blocks = group_into_atomic_blocks(msgs)
    assert len(blocks) == 3
    parallel_block = blocks[1]
    assert len(parallel_block) == 4  # 1 AI + 3 ToolMessages
    assert {m.tool_call_id for m in parallel_block if isinstance(m, ToolMessage)} == {
        "c1",
        "c2",
        "c3",
    }


def test_orphan_tool_message_is_its_own_block():
    msgs = [
        HumanMessage(content="hi", id="h1"),
        ToolMessage(content="orphan", tool_call_id="ghost", id="t1"),
    ]
    blocks = group_into_atomic_blocks(msgs)
    assert len(blocks) == 2


def test_system_messages_are_individual_blocks():
    msgs = [
        SystemMessage(content="alert", id="s1"),
        HumanMessage(content="hi", id="h1"),
    ]
    blocks = group_into_atomic_blocks(msgs)
    assert len(blocks) == 2


def test_select_blocks_to_evict_never_splits_pair():
    msgs = [
        HumanMessage(content="word " * 50, id="h1"),
        _ai_with_tools("", [("c1", "search")]),
        ToolMessage(content="word " * 50, tool_call_id="c1", id="t1"),
        AIMessage(content="word " * 50, id="a1"),
        HumanMessage(content="word " * 50, id="h2"),
    ]
    evicted, kept = select_blocks_to_evict(msgs, target_tokens=80)

    evicted_ids = {m.id for m in evicted}
    if "c1" in {tc["id"] for m in evicted if isinstance(m, AIMessage) for tc in m.tool_calls}:
        assert "t1" in evicted_ids
    if "t1" in evicted_ids:
        assert any(
            isinstance(m, AIMessage) and any(tc["id"] == "c1" for tc in m.tool_calls)
            for m in evicted
        )
    assert {m.id for m in kept}.isdisjoint(evicted_ids)
    assert len(evicted) + len(kept) == len(msgs)


def test_select_blocks_to_evict_returns_empty_for_zero_target():
    msgs = [HumanMessage(content="hi", id="h1")]
    evicted, kept = select_blocks_to_evict(msgs, target_tokens=0)
    assert evicted == []
    assert [m.id for m in kept] == ["h1"]


def test_count_state_tokens_is_monotonic_with_messages():
    base = count_state_tokens(
        system_prompt="sys",
        core_memory=None,
        recursive_summary=None,
        messages=[],
    )
    more = count_state_tokens(
        system_prompt="sys",
        core_memory=None,
        recursive_summary=None,
        messages=[HumanMessage(content="word " * 100, id="h1")],
    )
    assert more > base


def test_count_messages_tokens_sees_tool_call_payload():
    plain = AIMessage(content="hi", id="a0")
    with_call = _ai_with_tools("hi", [("c1", "search_documents_for_user_questions")])
    assert count_messages_tokens([with_call]) > count_messages_tokens([plain])
