"""Tests E2E de la Fase 3 sobre el grafo: pressure alert + flush + summary."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from memgpt.agent import build_agent
from memgpt.queue_manager import MEMORY_PRESSURE_ALERT_TEXT, QueueManagerConfig


class EchoLLM:
    """Returns an AIMessage that echoes the latest HumanMessage. Records calls."""

    def __init__(self) -> None:
        self.calls: list[list[Any]] = []

    def bind_tools(self, _tools):
        return self

    def invoke(self, messages):
        self.calls.append(list(messages))
        last_human = next(
            (m for m in reversed(messages) if isinstance(m, HumanMessage)), None
        )
        text = last_human.content if last_human else ""
        return AIMessage(content=f"echo:{text}")


def _summary_stub(label: str):
    def _impl(_msgs):
        return label

    return _impl


def _seed_long_history(prefix: str = "msg") -> list[HumanMessage]:
    return [HumanMessage(content=f"{prefix}-{i}: " + "word " * 20, id=f"h-{i}") for i in range(8)]


def _agent_with_small_window(summary_label: str = "FRESH SUMMARY"):
    qcfg = QueueManagerConfig(
        warning_threshold=0.50,
        flush_threshold=0.80,
        flush_eviction_ratio=0.40,
        context_window_tokens=400,
    )
    return build_agent(
        llm=EchoLLM(),
        checkpointer=MemorySaver(),
        queue_config=qcfg,
        summarizer_callable=_summary_stub(summary_label),
    )


def test_warning_alert_injected_once_per_episode():
    qcfg = QueueManagerConfig(
        warning_threshold=0.40,
        flush_threshold=0.95,
        flush_eviction_ratio=0.50,
        context_window_tokens=400,
    )
    captured: list[list] = []

    def stub(_msgs):
        captured.append(_msgs)
        return "S"

    agent = build_agent(
        llm=EchoLLM(),
        checkpointer=MemorySaver(),
        queue_config=qcfg,
        summarizer_callable=stub,
    )
    cfg = {"configurable": {"thread_id": "t-warn"}}

    seed = {"messages": _seed_long_history()}
    state1 = agent.invoke(seed, config=cfg)
    alerts1 = [
        m for m in state1["messages"]
        if isinstance(m, SystemMessage) and m.content == MEMORY_PRESSURE_ALERT_TEXT
    ]
    assert len(alerts1) == 1, "alert should be injected exactly once"

    state2 = agent.invoke({"messages": [HumanMessage(content="follow-up")]}, config=cfg)
    alerts2 = [
        m for m in state2["messages"]
        if isinstance(m, SystemMessage) and m.content == MEMORY_PRESSURE_ALERT_TEXT
    ]
    assert len(alerts2) == 1, "alert must NOT be re-injected within the same episode"


def test_flush_evicts_messages_and_writes_summary():
    agent = _agent_with_small_window("FRESH SUMMARY")
    cfg = {"configurable": {"thread_id": "t-flush"}}

    state = agent.invoke({"messages": _seed_long_history()}, config=cfg)

    snap = agent.get_state(cfg).values
    assert snap["recursive_summary"] == "FRESH SUMMARY"
    assert snap["evicted_count"] >= 1
    assert snap["memory_pressure_alerted"] is False

    surviving_seeded = [
        m for m in state["messages"]
        if isinstance(m, HumanMessage) and m.content.startswith(("msg-",))
    ]
    assert len(surviving_seeded) < 8, "FIFO should be smaller after flush"


def test_flush_preserves_prior_summary_via_summariser_input():
    qcfg = QueueManagerConfig(
        warning_threshold=0.50,
        flush_threshold=0.80,
        flush_eviction_ratio=0.40,
        context_window_tokens=400,
    )
    captured_prompts: list[list[dict]] = []

    def stub(messages):
        captured_prompts.append(messages)
        return f"SUMMARY-{len(captured_prompts)}"

    agent = build_agent(
        llm=EchoLLM(),
        checkpointer=MemorySaver(),
        queue_config=qcfg,
        summarizer_callable=stub,
    )
    cfg = {"configurable": {"thread_id": "t-twoflush"}}

    agent.invoke({"messages": _seed_long_history("first")}, config=cfg)
    agent.invoke({"messages": _seed_long_history("second")}, config=cfg)

    assert len(captured_prompts) >= 2, "expected at least two flushes"
    second_user_text = captured_prompts[-1][-1]["content"]
    assert "SUMMARY-1" in second_user_text, (
        "second flush prompt must include the previous summary so it gets folded in"
    )

    snap = agent.get_state(cfg).values
    assert snap["recursive_summary"].startswith("SUMMARY-")


def test_flush_does_not_orphan_tool_call_pairs():
    qcfg = QueueManagerConfig(
        warning_threshold=0.50,
        flush_threshold=0.80,
        flush_eviction_ratio=0.50,
        context_window_tokens=400,
    )
    agent = build_agent(
        llm=EchoLLM(),
        checkpointer=MemorySaver(),
        queue_config=qcfg,
        summarizer_callable=_summary_stub("S"),
    )
    cfg = {"configurable": {"thread_id": "t-pairs"}}

    seeded = [
        HumanMessage(content="word " * 30, id="h1"),
        AIMessage(
            content="",
            id="a1",
            tool_calls=[{"id": "c1", "name": "search", "args": {}, "type": "tool_call"}],
        ),
        ToolMessage(content="word " * 30, tool_call_id="c1", id="t1", name="search"),
        AIMessage(content="word " * 30, id="a2"),
        HumanMessage(content="word " * 30, id="h2"),
        HumanMessage(content="word " * 30, id="h3"),
    ]
    agent.invoke({"messages": seeded}, config=cfg)

    snap = agent.get_state(cfg).values
    surviving_ids = {m.id for m in snap["messages"] if getattr(m, "id", None)}
    assert ("a1" in surviving_ids) == ("t1" in surviving_ids), (
        "the AIMessage with tool_calls and its ToolMessage must be evicted together "
        "or kept together — never split"
    )


def test_no_flush_when_below_threshold():
    qcfg = QueueManagerConfig(
        warning_threshold=0.95,
        flush_threshold=1.00,
        flush_eviction_ratio=0.50,
        context_window_tokens=200_000,
    )
    agent = build_agent(
        llm=EchoLLM(),
        checkpointer=MemorySaver(),
        queue_config=qcfg,
        summarizer_callable=_summary_stub("SHOULD-NOT-RUN"),
    )
    cfg = {"configurable": {"thread_id": "t-none"}}
    agent.invoke({"messages": [HumanMessage(content="hi")]}, config=cfg)

    snap = agent.get_state(cfg).values
    assert snap.get("recursive_summary") in (None, "")
    assert snap.get("evicted_count", 0) == 0
    assert snap.get("memory_pressure_alerted", False) is False
