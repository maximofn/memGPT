"""Tests E2E de la Fase 7: persistencia robusta entre sesiones.

Simulamos "kill -9 + restart" construyendo una **segunda** instancia
del agente que comparte únicamente los backends de persistencia
(checkpointer, memory_store, event_store) con la primera. Si la
arquitectura es correcta, el segundo agente recupera el estado entero
sin intervención adicional. Esto valida el contrato sin depender de
infra real (Postgres / Neo4j) — los `MemorySaver`, `InMemoryStore` e
`InMemoryEventStore` cumplen el mismo contrato que sus equivalentes
Postgres/Graphiti.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langgraph.checkpoint.memory import MemorySaver

from memgpt.agent import build_agent
from memgpt.events import (
    EventRegistry,
    InMemoryEventStore,
    WallClockEvent,
    default_wallclock_dispatcher,
)
from memgpt.memory_store import InMemoryStore
from memgpt.persistence import build_persistent_agent
from memgpt.queue_manager import QueueManagerConfig
from memgpt.state import MemGPTState


# --- LLM stubs ----------------------------------------------------------


class TextLLM:
    def __init__(self, text: str = "ok") -> None:
        self.text = text
        self.calls = 0

    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        self.calls += 1
        return AIMessage(content=self.text, id=f"ai-{self.calls}")


class ScriptedLLM:
    def __init__(self, responses: list[BaseMessage]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        idx = min(self.calls, len(self.responses) - 1)
        resp = self.responses[idx]
        self.calls += 1
        return resp


def _ai_with_tool(name: str, args: dict, *, tc_id: str = "c1", msg_id: str = "ai") -> AIMessage:
    return AIMessage(
        content="",
        id=msg_id,
        tool_calls=[{"id": tc_id, "name": name, "args": args, "type": "tool_call"}],
    )


# --- helpers ------------------------------------------------------------


def _agent(*, llm, checkpointer, store, qcfg=None):
    return build_agent(
        llm=llm,
        checkpointer=checkpointer,
        memory_store=store,
        queue_config=qcfg,
        summarizer_callable=lambda _msgs: "S",
    )


# --- 1. State round-trip ------------------------------------------------


def test_memgpt_state_round_trips_through_pydantic():
    """Toda la suma de campos de Fases 0-6 debe sobrevivir a la
    serialización Pydantic — esto es lo que hace el checkpointer
    bajo el capó (el `JsonPlusSerializer` de LangGraph)."""
    from datetime import datetime, timezone

    state = MemGPTState(
        messages=[HumanMessage(content="hi", id="h1")],
        recursive_summary="prior summary",
        step_count=7,
        memory_pressure_alerted=True,
        evicted_count=3,
        persisted_message_ids=["h1", "ai1"],
        chained_heartbeats=2,
        turn_started_at=datetime.now(timezone.utc),
        recent_tool_call_keys=["lookup::{}", "get_time::{}"],
        last_processed_human_id="h1",
    )
    state.core_memory = state.core_memory.with_appended("human", "loves coffee")

    raw = state.model_dump_json()
    restored = MemGPTState.model_validate_json(raw)

    assert restored.recursive_summary == "prior summary"
    assert restored.step_count == 7
    assert restored.evicted_count == 3
    assert restored.persisted_message_ids == ["h1", "ai1"]
    assert restored.chained_heartbeats == 2
    assert restored.recent_tool_call_keys == ["lookup::{}", "get_time::{}"]
    assert restored.core_memory.blocks["human"].value.endswith("loves coffee")


# --- 2. Kill+restart: Core Memory + recursive_summary survive -----------


def test_kill_restart_preserves_core_memory_and_summary():
    saver = MemorySaver()
    store = InMemoryStore()
    cfg = {"configurable": {"thread_id": "t-restart"}}

    # ---- Session 1 ----
    llm1 = TextLLM(text="acknowledged")
    agent1 = _agent(llm=llm1, checkpointer=saver, store=store)
    agent1.invoke(
        {"messages": [HumanMessage(content="remember I like Rust", id="h1")]},
        config=cfg,
    )
    # Simulate a Core Memory write that the LLM would have made. The
    # snapshot dict omits fields still at their default factory; reading
    # the live state object is the supported way.
    snap1 = agent1.get_state(cfg).values
    base_core = snap1.get("core_memory") or MemGPTState().core_memory
    new_core = base_core.with_appended("human", "likes Rust")
    agent1.update_state(cfg, {"core_memory": new_core, "recursive_summary": "Phase-7 conv"})

    # Drop every reference to the first agent — gone, "killed".
    del agent1, llm1

    # ---- Session 2: brand-new agent, only the backends are shared ----
    llm2 = TextLLM(text="welcome back")
    agent2 = _agent(llm=llm2, checkpointer=saver, store=store)
    snap = agent2.get_state(cfg).values
    assert snap["recursive_summary"] == "Phase-7 conv"
    assert "likes Rust" in snap["core_memory"].blocks["human"].value


# --- 3. Kill+restart: Recall remembers evicted messages ----------------


def test_kill_restart_preserves_recall_after_flush():
    """Tras un flush en sesión 1, en sesión 2 los mensajes evictados
    siguen recuperables vía el `MemoryStore` compartido."""
    saver = MemorySaver()
    store = InMemoryStore()
    cfg = {"configurable": {"thread_id": "t-recall"}}

    # Tiny window so the flush trips immediately (warning at 70 of 100,
    # flush at 100, evict 50%).
    qcfg = QueueManagerConfig(context_window_tokens=100)
    llm1 = TextLLM(text="ack")
    agent1 = _agent(llm=llm1, checkpointer=saver, store=store, qcfg=qcfg)

    # Push enough text to force a flush.
    big_payload = "the secret pizza topping is anchovies " * 5
    agent1.invoke(
        {"messages": [HumanMessage(content=big_payload, id="h-secret")]},
        config=cfg,
    )
    for i in range(5):
        agent1.invoke(
            {"messages": [HumanMessage(content=f"filler msg {i}", id=f"h-f{i}")]},
            config=cfg,
        )

    snap_after_session1 = agent1.get_state(cfg).values
    # Crash & restart: the backends survive, the agent does not.
    del agent1, llm1

    llm2 = TextLLM(text="back")
    agent2 = _agent(llm=llm2, checkpointer=saver, store=store, qcfg=qcfg)

    # Recall via the store directly (the agent uses this through tools).
    hits = store.search_conversation("anchovies", limit=5)
    assert hits, "the evicted secret should still be retrievable from Recall"
    assert any("anchovies" in h.content for h in hits)

    # And the checkpointer's view of state survives too.
    snap_after_restart = agent2.get_state(cfg).values
    assert snap_after_restart["evicted_count"] == snap_after_session1["evicted_count"]
    assert snap_after_restart["evicted_count"] >= 1


# --- 4. Kill+restart: scheduled events restore -------------------------


def test_kill_restart_restores_wallclock_events():
    event_store = InMemoryEventStore()
    saver = MemorySaver()
    mem_store = InMemoryStore()

    # ---- Session 1: register a wall-clock event ----
    agent1, registry1 = build_persistent_agent(
        checkpointer=saver,
        memory_store=mem_store,
        event_registry=EventRegistry(store=event_store),
        llm=TextLLM(text="acknowledged"),
        summarizer_callable=lambda _m: "S",
    )
    ev = WallClockEvent(
        name="hourly-check",
        agent_id="agent-restart",
        trigger_type="interval",
        trigger_kwargs={"hours": 1},
        payload="hourly ping",
    )
    registry1.register_wallclock(ev)

    # ---- Crash ----
    del agent1, registry1

    # ---- Session 2: brand-new agent + brand-new registry, same store ----
    agent2, registry2 = build_persistent_agent(
        checkpointer=saver,
        memory_store=mem_store,
        event_registry=EventRegistry(store=event_store),
        llm=TextLLM(text="back"),
        summarizer_callable=lambda _m: "S",
    )

    # build_persistent_agent calls restore() internally → the spec is back.
    names = [e.name for e in registry2.list_wallclock()]
    assert names == ["hourly-check"]
    job = registry2._scheduler.get_job("hourly-check")
    assert job is not None
    assert job.args == ("agent-restart", "hourly ping")


# --- 5. Atomicity: core_memory_replace is a single checkpoint write ---


def test_core_memory_replace_is_one_atomic_step():
    """Un fallo a mitad del replace no puede dejar el bloque vacío.

    `core_memory_replace` devuelve **un único** `Command(update={...})`
    que LangGraph escribe como un único checkpoint. La verificación
    operativa: tras un replace el block contiene `new` y nunca un estado
    intermedio sin `old` ni `new`.
    """
    from memgpt.core_memory import default_core_memory

    saver = MemorySaver()
    store = InMemoryStore()
    cfg = {"configurable": {"thread_id": "t-atomic"}}

    llm = ScriptedLLM(
        [
            _ai_with_tool(
                "core_memory_replace",
                {"label": "human", "old": "Likes JS.", "new": "Loves Python."},
                msg_id="ai-1",
            ),
            AIMessage(content="done", id="ai-2"),
        ]
    )
    agent = _agent(llm=llm, checkpointer=saver, store=store)
    # Seed the human block with a non-empty value so `old` matches.
    seeded = default_core_memory(human="Likes JS.")
    agent.update_state(cfg, {"core_memory": seeded})

    agent.invoke(
        {"messages": [HumanMessage(content="please update", id="h-1")]}, config=cfg,
    )

    # Walk the checkpoint history and check no intermediate snapshot
    # has the human block in an inconsistent (partial) state.
    history = list(agent.get_state_history(cfg))
    assert history, "checkpointer should have written at least one snapshot"
    seen_values: set[str] = set()
    for h in history:
        cm = h.values.get("core_memory")
        if cm is None:
            continue
        seen_values.add(cm.blocks["human"].value)
    # Only two legal values exist across history: the original ("Likes JS.")
    # and the post-replace ("Loves Python."). No frankenstein in between.
    assert seen_values == {"Likes JS.", "Loves Python."}


# --- 6. Persistent agent factory wiring --------------------------------


def test_build_persistent_agent_wires_dispatcher_and_restore():
    """`build_persistent_agent` debe instalar el default dispatcher e
    invocar `restore()` automáticamente."""
    event_store = InMemoryEventStore()
    pre_existing = WallClockEvent(
        name="pre-existing",
        agent_id="a",
        trigger_type="interval",
        trigger_kwargs={"minutes": 30},
        payload="ping",
    )
    event_store.save_wallclock(pre_existing)

    agent, registry = build_persistent_agent(
        checkpointer=MemorySaver(),
        memory_store=InMemoryStore(),
        event_registry=EventRegistry(store=event_store),
        llm=TextLLM(),
        summarizer_callable=lambda _m: "S",
    )

    # restore() ran → the pre-existing event is in the registry.
    assert [e.name for e in registry.list_wallclock()] == ["pre-existing"]

    # Dispatcher is installed → calling it routes to the real agent.
    # Manually fire the registered job (avoiding scheduler timing).
    job = registry._scheduler.get_job("pre-existing")
    assert job is not None
    job.func(*job.args)

    # The dispatcher invoked the agent under thread_id="a".
    snap = agent.get_state({"configurable": {"thread_id": "a"}}).values
    contents = [m.content for m in snap["messages"]]
    assert "ping" in contents


def test_build_persistent_agent_rejects_both_registry_and_dsn():
    with pytest.raises(ValueError, match="not both"):
        build_persistent_agent(
            checkpointer=MemorySaver(),
            memory_store=InMemoryStore(),
            event_registry=EventRegistry(),
            event_store_dsn="postgresql://x",
            llm=TextLLM(),
            summarizer_callable=lambda _m: "S",
        )


# pytest import lives at module level so the assertion above type-checks.
import pytest  # noqa: E402
