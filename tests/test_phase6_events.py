"""Tests E2E de la Fase 6: integración EventRegistry ↔ agent loop."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver

from memgpt.agent import build_agent
from memgpt.events import EventRegistry, IterationEvent, WallClockEvent
from memgpt.heartbeat import HeartbeatConfig


# --- LLM stubs ---------------------------------------------------------


class TextLLM:
    """Stub: siempre devuelve un AIMessage de texto, nunca tool calls."""

    def __init__(self, text: str = "ok") -> None:
        self.text = text
        self.calls = 0

    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        self.calls += 1
        return AIMessage(content=self.text, id=f"ai-{self.calls}")


def _agent(*, llm, registry: EventRegistry | None = None):
    return build_agent(
        llm=llm,
        checkpointer=MemorySaver(),
        summarizer_callable=lambda _msgs: "S",
        event_registry=registry,
    )


# --- step_count ----------------------------------------------------------


def test_step_count_increments_per_llm_call():
    reg = EventRegistry()
    agent = _agent(llm=TextLLM(), registry=reg)
    cfg = {"configurable": {"thread_id": "t-step"}}

    agent.invoke({"messages": [HumanMessage(content="hi", id="h1")]}, config=cfg)
    snap1 = agent.get_state(cfg).values
    assert snap1["step_count"] == 1

    agent.invoke({"messages": [HumanMessage(content="hi2", id="h2")]}, config=cfg)
    snap2 = agent.get_state(cfg).values
    assert snap2["step_count"] == 2


def test_no_step_tick_when_registry_omitted():
    """Sin registry, step_count se queda en 0 (compatibilidad con Fase 5)."""
    agent = _agent(llm=TextLLM())
    cfg = {"configurable": {"thread_id": "t-nostep"}}
    agent.invoke({"messages": [HumanMessage(content="hi", id="h1")]}, config=cfg)
    snap = agent.get_state(cfg).values
    # When the registry is None, the step_tick node never runs and step_count
    # stays at its default. Pydantic-typed states omit unset defaults from
    # the snapshot, so use .get(...).
    assert snap.get("step_count", 0) == 0


# --- iteration callbacks -------------------------------------------------


def test_iteration_callback_fires_every_n_steps():
    fired_at: list[int] = []

    def cb(state):
        fired_at.append(state.step_count)
        return None

    reg = EventRegistry()
    reg.register_iteration(IterationEvent(name="every2", every_n_steps=2, callback=cb))

    agent = _agent(llm=TextLLM(), registry=reg)
    cfg = {"configurable": {"thread_id": "t-iter"}}
    for i in range(1, 6):
        agent.invoke(
            {"messages": [HumanMessage(content=f"q{i}", id=f"h{i}")]}, config=cfg,
        )
    # 5 LLM calls → step_count goes 1..5 → callback fires at 2 and 4.
    assert fired_at == [2, 4]


def test_iteration_callback_can_inject_messages():
    """Una callback que devuelve `messages` los appendea al state vía el reducer."""

    def consolidator(_state):
        return {"messages": [SystemMessage(content="[sleep-time tick]")]}

    reg = EventRegistry()
    reg.register_iteration(IterationEvent(
        name="sleeper", every_n_steps=1, callback=consolidator,
    ))

    agent = _agent(llm=TextLLM(), registry=reg)
    cfg = {"configurable": {"thread_id": "t-inject"}}
    state = agent.invoke(
        {"messages": [HumanMessage(content="hi", id="h1")]}, config=cfg,
    )

    sleep_msgs = [
        m for m in state["messages"]
        if isinstance(m, SystemMessage) and "sleep-time tick" in m.content
    ]
    assert len(sleep_msgs) == 1


# --- wall-clock dispatcher integrated with the agent --------------------


def test_wallclock_dispatcher_drives_agent_via_systemmessage():
    """Simulamos APScheduler invocando al dispatcher: el agente debe
    procesar el SystemMessage como un nuevo turno."""
    from memgpt.events import default_wallclock_dispatcher

    llm = TextLLM(text="acknowledged")
    agent = _agent(llm=llm)
    dispatcher = default_wallclock_dispatcher(agent)

    # Equivalente a APScheduler disparando el job ``("scheduled-agent", "wake up")``.
    dispatcher("scheduled-agent", "wake up")

    cfg = {"configurable": {"thread_id": "scheduled-agent"}}
    snap = agent.get_state(cfg).values
    assert llm.calls == 1
    # The triggering SystemMessage and the AIMessage are both in state.
    contents = [m.content for m in snap["messages"]]
    assert "wake up" in contents
    assert "acknowledged" in contents


def test_registered_wallclock_job_invokes_dispatcher_with_correct_args():
    """Verifica el cableado APScheduler → dispatcher sin esperar al cron:
    accedemos al `Job` registrado y lo disparamos manualmente."""
    received: list[tuple[str, str]] = []

    reg = EventRegistry(wallclock_dispatcher=lambda a, p: received.append((a, p)))
    ev = WallClockEvent(
        name="hourly-check",
        agent_id="agent-x",
        trigger_type="interval",
        trigger_kwargs={"hours": 1},
        payload="hourly ping",
    )
    reg.register_wallclock(ev, persist=False)

    job = reg._scheduler.get_job("hourly-check")
    assert job is not None
    job.func(*job.args)

    assert received == [("agent-x", "hourly ping")]
