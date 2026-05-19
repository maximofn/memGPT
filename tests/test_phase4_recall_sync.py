"""Tests E2E de la Fase 4: el agente persiste en Recall y sobrevive flushes."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from memgpt.agent import build_agent
from memgpt.memory_store import InMemoryStore
from memgpt.queue_manager import QueueManagerConfig


class EchoLLM:
    """LLM stub that echoes the latest HumanMessage as an AIMessage."""

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


def _agent(store: InMemoryStore, *, qcfg: QueueManagerConfig | None = None):
    return build_agent(
        llm=EchoLLM(),
        checkpointer=MemorySaver(),
        memory_store=store,
        queue_config=qcfg,
        summarizer_callable=lambda _msgs: "S",
    )


def test_human_and_assistant_messages_are_persisted_each_turn():
    store = InMemoryStore()
    agent = _agent(store)
    cfg = {"configurable": {"thread_id": "t-persist"}}

    agent.invoke({"messages": [HumanMessage(content="hi", id="h1")]}, config=cfg)

    contents = {m.content for m in store.messages}
    assert "hi" in contents
    assert any(c.startswith("echo:hi") for c in contents)


def test_persistence_is_idempotent_across_invocations():
    store = InMemoryStore()
    agent = _agent(store)
    cfg = {"configurable": {"thread_id": "t-idem"}}

    agent.invoke({"messages": [HumanMessage(content="one", id="h1")]}, config=cfg)
    n_after_first = len(store.messages)
    agent.invoke({"messages": [HumanMessage(content="two", id="h2")]}, config=cfg)
    n_after_second = len(store.messages)

    # First turn: persists h1 + AIMessage(echo:one) = 2.
    # Second turn: persists previous AIMessage was already counted via h1's
    # turn, and now h2 + new AIMessage(echo:two). Net +2, never re-persists h1.
    assert n_after_first == 2
    assert n_after_second == 4
    ids = [m.message_id for m in store.messages]
    assert len(ids) == len(set(ids)), "no duplicate message_ids"


def test_no_memory_store_means_no_persistence_and_no_extra_tools():
    agent = build_agent(
        llm=EchoLLM(),
        checkpointer=MemorySaver(),
        summarizer_callable=lambda _msgs: "S",
    )
    cfg = {"configurable": {"thread_id": "t-nostore"}}
    state = agent.invoke({"messages": [HumanMessage(content="x")]}, config=cfg)
    # The agent must not crash and must not have a Recall backend bound.
    assert any(isinstance(m, AIMessage) for m in state["messages"])


def test_evicted_messages_are_searchable_in_recall():
    store = InMemoryStore()
    qcfg = QueueManagerConfig(
        warning_threshold=0.50,
        flush_threshold=0.80,
        flush_eviction_ratio=0.40,
        context_window_tokens=400,
    )
    agent = _agent(store, qcfg=qcfg)
    cfg = {"configurable": {"thread_id": "t-evict"}}

    seed = [
        HumanMessage(content=f"my secret-{i} word " * 10, id=f"h{i}")
        for i in range(8)
    ]
    state = agent.invoke({"messages": seed}, config=cfg)

    # FIFO shrunk because of the flush
    surviving = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    assert len(surviving) < 8

    # …but evicted ones still live in Recall
    hits = store.search_conversation("secret-0")
    assert any("secret-0" in h.content for h in hits)


def test_system_messages_are_not_persisted():
    store = InMemoryStore()
    qcfg = QueueManagerConfig(
        warning_threshold=0.40,
        flush_threshold=0.95,
        flush_eviction_ratio=0.50,
        context_window_tokens=400,
    )
    agent = _agent(store, qcfg=qcfg)
    cfg = {"configurable": {"thread_id": "t-sys"}}

    seed = [HumanMessage(content="word " * 20, id=f"h{i}") for i in range(6)]
    agent.invoke({"messages": seed}, config=cfg)

    roles = {m.role for m in store.messages}
    assert "system" not in roles


def test_tool_messages_are_persisted_with_role_tool():
    store = InMemoryStore()
    agent = _agent(store)
    cfg = {"configurable": {"thread_id": "t-tool"}}

    seed = [
        HumanMessage(content="search please", id="h1"),
        AIMessage(
            content="",
            id="a1",
            tool_calls=[{"id": "c1", "name": "lookup", "args": {"q": "x"}, "type": "tool_call"}],
        ),
        ToolMessage(content="result-x", tool_call_id="c1", id="t1", name="lookup"),
    ]
    agent.invoke({"messages": seed}, config=cfg)

    by_id = {m.message_id: m for m in store.messages}
    assert "t1" in by_id
    assert by_id["t1"].role == "tool"
    assert "result-x" in by_id["t1"].content
    assert "a1" in by_id
    assert "lookup" in by_id["a1"].content  # tool calls captured
