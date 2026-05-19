"""Unit tests del módulo `events` (registry, store, configs)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from memgpt.events import (
    EventRegistry,
    InMemoryEventStore,
    IterationEvent,
    WallClockEvent,
    default_wallclock_dispatcher,
)
from memgpt.state import MemGPTState


# --- WallClockEvent ----------------------------------------------------


def test_wallclock_event_builds_each_trigger_type():
    cron = WallClockEvent(
        name="c",
        agent_id="a",
        trigger_type="cron",
        trigger_kwargs={"hour": 9, "minute": 0},
        payload="ping",
    )
    interval = WallClockEvent(
        name="i",
        agent_id="a",
        trigger_type="interval",
        trigger_kwargs={"seconds": 10},
        payload="ping",
    )
    one_shot = WallClockEvent(
        name="d",
        agent_id="a",
        trigger_type="date",
        trigger_kwargs={"run_date": datetime(2030, 1, 1, tzinfo=timezone.utc)},
        payload="ping",
    )

    assert isinstance(cron.build_trigger(), CronTrigger)
    assert isinstance(interval.build_trigger(), IntervalTrigger)
    assert isinstance(one_shot.build_trigger(), DateTrigger)


def test_wallclock_event_rejects_unknown_trigger_type():
    bad = WallClockEvent(
        name="bad", agent_id="a", trigger_type="solar", payload="x"
    )
    with pytest.raises(ValueError, match="unknown trigger_type"):
        bad.build_trigger()


def test_wallclock_event_round_trips_through_json():
    ev = WallClockEvent(
        name="c",
        agent_id="agent-42",
        trigger_type="cron",
        trigger_kwargs={"hour": 9},
        payload="check inbox",
    )
    restored = WallClockEvent.model_validate_json(ev.model_dump_json())
    assert restored == ev


# --- IterationEvent ----------------------------------------------------


def test_iteration_event_requires_positive_n():
    with pytest.raises(Exception):
        IterationEvent(name="x", every_n_steps=0, callback=lambda s: None)


# --- InMemoryEventStore ------------------------------------------------


def test_in_memory_store_save_list_delete():
    store = InMemoryEventStore()
    a = WallClockEvent(name="a", agent_id="x", trigger_type="interval",
                       trigger_kwargs={"seconds": 1}, payload="p")
    b = WallClockEvent(name="b", agent_id="x", trigger_type="interval",
                       trigger_kwargs={"seconds": 2}, payload="q")
    store.save_wallclock(a)
    store.save_wallclock(b)
    names = {e.name for e in store.list_wallclock()}
    assert names == {"a", "b"}

    store.delete_wallclock("a")
    assert {e.name for e in store.list_wallclock()} == {"b"}


def test_in_memory_store_save_overwrites_same_name():
    store = InMemoryEventStore()
    v1 = WallClockEvent(name="x", agent_id="a", trigger_type="interval",
                        trigger_kwargs={"seconds": 1}, payload="v1")
    v2 = WallClockEvent(name="x", agent_id="a", trigger_type="interval",
                        trigger_kwargs={"seconds": 1}, payload="v2")
    store.save_wallclock(v1)
    store.save_wallclock(v2)
    [only] = store.list_wallclock()
    assert only.payload == "v2"


# --- EventRegistry: iteration -------------------------------------------


def _state(step_count: int = 0) -> MemGPTState:
    return MemGPTState(messages=[HumanMessage(content="hi", id="h1")],
                       step_count=step_count)


def test_dispatch_iteration_skips_when_step_count_zero():
    reg = EventRegistry()
    seen: list[int] = []
    reg.register_iteration(IterationEvent(
        name="t", every_n_steps=1, callback=lambda s: seen.append(s.step_count) or None,
    ))
    assert reg.dispatch_iteration(_state(0)) == {}
    assert seen == []


def test_dispatch_iteration_fires_only_on_multiples():
    reg = EventRegistry()
    fired: list[int] = []
    reg.register_iteration(IterationEvent(
        name="every3", every_n_steps=3,
        callback=lambda s: fired.append(s.step_count) or None,
    ))
    for n in range(1, 8):
        reg.dispatch_iteration(_state(n))
    assert fired == [3, 6]


def test_dispatch_iteration_merges_messages_and_overwrites_other_keys():
    reg = EventRegistry()

    def cb_a(_s):
        return {"messages": [SystemMessage(content="from a")], "evicted_count": 1}

    def cb_b(_s):
        return {"messages": [SystemMessage(content="from b")], "evicted_count": 5}

    reg.register_iteration(IterationEvent(name="a", every_n_steps=1, callback=cb_a))
    reg.register_iteration(IterationEvent(name="b", every_n_steps=1, callback=cb_b))

    update = reg.dispatch_iteration(_state(1))
    msgs = update["messages"]
    assert [m.content for m in msgs] == ["from a", "from b"]
    # Last write wins for non-messages keys.
    assert update["evicted_count"] == 5


def test_register_iteration_rejects_duplicates():
    reg = EventRegistry()
    cb = lambda s: None  # noqa: E731
    reg.register_iteration(IterationEvent(name="x", every_n_steps=1, callback=cb))
    with pytest.raises(ValueError, match="already registered"):
        reg.register_iteration(IterationEvent(name="x", every_n_steps=1, callback=cb))


def test_unregister_iteration_is_idempotent():
    reg = EventRegistry()
    reg.register_iteration(IterationEvent(
        name="x", every_n_steps=1, callback=lambda s: None,
    ))
    reg.unregister_iteration("x")
    reg.unregister_iteration("x")  # no error
    assert reg.list_iteration() == []


# --- EventRegistry: wall-clock ------------------------------------------


def _make_registry(*, with_dispatcher: bool = True, with_store: bool = True):
    sched = BackgroundScheduler()
    store = InMemoryEventStore() if with_store else None
    calls: list[tuple[str, str]] = []

    def dispatch(agent_id: str, payload: str) -> None:
        calls.append((agent_id, payload))

    reg = EventRegistry(
        scheduler=sched,
        wallclock_dispatcher=dispatch if with_dispatcher else None,
        store=store,
    )
    return reg, sched, store, calls


def test_register_wallclock_requires_dispatcher():
    reg, _sched, _store, _calls = _make_registry(with_dispatcher=False)
    ev = WallClockEvent(name="x", agent_id="a", trigger_type="interval",
                        trigger_kwargs={"seconds": 60}, payload="p")
    with pytest.raises(RuntimeError, match="wallclock_dispatcher"):
        reg.register_wallclock(ev)


def test_register_wallclock_adds_job_and_persists():
    reg, sched, store, _calls = _make_registry()
    ev = WallClockEvent(name="x", agent_id="agent-1", trigger_type="interval",
                        trigger_kwargs={"hours": 1}, payload="hello")
    reg.register_wallclock(ev)

    job = sched.get_job("x")
    assert job is not None
    assert job.args == ("agent-1", "hello")
    assert [e.name for e in store.list_wallclock()] == ["x"]


def test_register_wallclock_rejects_duplicate_name():
    reg, _sched, _store, _calls = _make_registry()
    ev = WallClockEvent(name="x", agent_id="a", trigger_type="interval",
                        trigger_kwargs={"seconds": 60}, payload="p")
    reg.register_wallclock(ev)
    with pytest.raises(ValueError, match="already registered"):
        reg.register_wallclock(ev)


def test_unregister_wallclock_removes_job_and_store_entry():
    reg, sched, store, _calls = _make_registry()
    ev = WallClockEvent(name="x", agent_id="a", trigger_type="interval",
                        trigger_kwargs={"seconds": 60}, payload="p")
    reg.register_wallclock(ev)
    reg.unregister_wallclock("x")
    assert sched.get_job("x") is None
    assert store.list_wallclock() == []


def test_restore_reloads_persisted_events_into_fresh_registry():
    # Persist via one registry…
    reg1, _sched1, store, _calls1 = _make_registry()
    ev = WallClockEvent(name="alpha", agent_id="a", trigger_type="interval",
                        trigger_kwargs={"minutes": 30}, payload="ping")
    reg1.register_wallclock(ev)

    # …then start a fresh registry with the same store.
    sched2 = BackgroundScheduler()
    calls2: list[tuple[str, str]] = []
    reg2 = EventRegistry(
        scheduler=sched2,
        wallclock_dispatcher=lambda a, p: calls2.append((a, p)),
        store=store,
    )
    restored = reg2.restore()
    assert [e.name for e in restored] == ["alpha"]
    assert sched2.get_job("alpha") is not None


def test_restore_skips_already_registered_events():
    reg, _sched, store, _calls = _make_registry()
    ev = WallClockEvent(name="x", agent_id="a", trigger_type="interval",
                        trigger_kwargs={"hours": 1}, payload="p")
    reg.register_wallclock(ev)
    # Calling restore() again with the spec already in the store should not
    # raise (we already have it registered) and should not duplicate.
    restored = reg.restore()
    assert restored == []
    assert [e.name for e in reg.list_wallclock()] == ["x"]


def test_set_wallclock_dispatcher_after_construction():
    reg = EventRegistry(scheduler=BackgroundScheduler())
    received: list[tuple[str, str]] = []
    reg.set_wallclock_dispatcher(lambda a, p: received.append((a, p)))
    ev = WallClockEvent(name="x", agent_id="a", trigger_type="interval",
                        trigger_kwargs={"hours": 1}, payload="hi")
    reg.register_wallclock(ev, persist=False)
    job = reg._scheduler.get_job("x")
    assert job is not None
    # Manually fire the dispatcher to validate the wiring (avoids scheduler timing).
    job.func(*job.args)
    assert received == [("a", "hi")]


# --- default_wallclock_dispatcher --------------------------------------


class _RecorderAgent:
    """Pretende ser un CompiledStateGraph: solo captura llamadas a invoke."""

    def __init__(self) -> None:
        self.invocations: list[tuple[Any, dict[str, Any]]] = []

    def invoke(self, payload, config):
        self.invocations.append((payload, config))


def test_default_dispatcher_invokes_with_system_message_and_thread_id():
    agent = _RecorderAgent()
    dispatcher = default_wallclock_dispatcher(agent)
    dispatcher("agent-7", "time to consolidate")

    [(payload, config)] = agent.invocations
    [msg] = payload["messages"]
    assert isinstance(msg, SystemMessage)
    assert msg.content == "time to consolidate"
    assert config == {"configurable": {"thread_id": "agent-7"}}


def test_default_dispatcher_accepts_custom_payload_to_message():
    agent = _RecorderAgent()
    dispatcher = default_wallclock_dispatcher(
        agent, payload_to_message=lambda p: HumanMessage(content=f"[evt] {p}"),
    )
    dispatcher("a1", "noon ping")

    [(payload, _config)] = agent.invocations
    [msg] = payload["messages"]
    assert isinstance(msg, HumanMessage)
    assert msg.content == "[evt] noon ping"


# --- start / shutdown lifecycle ----------------------------------------


def test_start_and_shutdown_are_idempotent():
    reg, sched, _store, _calls = _make_registry()
    reg.start()
    assert sched.running
    reg.start()  # no-op when already running
    reg.shutdown(wait=False)
    assert not sched.running
    reg.shutdown(wait=False)  # no-op when stopped
