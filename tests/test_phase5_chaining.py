"""Tests E2E de la Fase 5: function chaining + red de seguridad."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from memgpt.agent import build_agent
from memgpt.heartbeat import HeartbeatConfig, HeartbeatMode


# --- Test tools ----------------------------------------------------------


@tool
def lookup(q: str, request_heartbeat: bool = False) -> str:
    """Test tool: includes ``request_heartbeat`` in its schema so the LLM
    stub can emit it."""
    return f"result for {q}"


@tool
def auto_thing(x: int = 0) -> str:
    """Test tool used to validate ``auto_continue_tools`` without the explicit flag."""
    return f"auto:{x}"


# --- LLM stubs -----------------------------------------------------------


def _ai_with_tool(name: str, args: dict, *, tc_id: str = "c1", msg_id: str = "ai") -> AIMessage:
    return AIMessage(
        content="",
        id=msg_id,
        tool_calls=[{"id": tc_id, "name": name, "args": args, "type": "tool_call"}],
    )


def _ai_text(text: str, *, msg_id: str = "ai") -> AIMessage:
    return AIMessage(content=text, id=msg_id)


class ScriptedLLM:
    """Returns a pre-scripted sequence of ``AIMessage``s on each ``invoke``."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        idx = min(self.calls, len(self.responses) - 1)
        resp = self.responses[idx]
        self.calls += 1
        return resp


class InfiniteToolLLM:
    """Stub that always returns a tool call (used to stress safety nets)."""

    def __init__(self, *, name: str = "lookup", args_factory=None):
        self.calls = 0
        self.name = name
        self.args_factory = args_factory or (lambda i: {"q": f"q{i}"})

    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        self.calls += 1
        return _ai_with_tool(
            self.name, self.args_factory(self.calls), msg_id=f"ai-inf-{self.calls}"
        )


def _agent(llm, *, tools_list=(lookup,), heartbeat_config=None, summary_label="S"):
    return build_agent(
        llm=llm,
        tools=list(tools_list),
        checkpointer=MemorySaver(),
        summarizer_callable=lambda _msgs: summary_label,
        heartbeat_config=heartbeat_config,
    )


# --- NATIVE mode --------------------------------------------------------


def test_native_chains_until_yield():
    llm = ScriptedLLM(
        [
            _ai_with_tool("lookup", {"q": "first"}, msg_id="ai1"),
            _ai_with_tool("lookup", {"q": "second"}, msg_id="ai2"),
            _ai_text("done", msg_id="ai3"),
        ]
    )
    agent = _agent(llm)
    cfg = {"configurable": {"thread_id": "t-native"}}
    state = agent.invoke({"messages": [HumanMessage(content="go", id="h1")]}, config=cfg)

    assert llm.calls == 3
    final = state["messages"][-1]
    assert isinstance(final, AIMessage)
    assert final.content == "done"


def test_native_max_chained_heartbeats_caps_loop():
    llm = InfiniteToolLLM()
    agent = _agent(
        llm, heartbeat_config=HeartbeatConfig(max_chained_heartbeats=3)
    )
    cfg = {"configurable": {"thread_id": "t-max"}}
    agent.invoke({"messages": [HumanMessage(content="go", id="h1")]}, config=cfg)

    snap = agent.get_state(cfg).values
    assert snap["chained_heartbeats"] == 3
    assert llm.calls == 3


def test_native_turn_timeout_forces_end():
    llm = InfiniteToolLLM()
    agent = _agent(
        llm,
        heartbeat_config=HeartbeatConfig(turn_timeout_seconds=1, max_chained_heartbeats=100),
    )
    cfg = {"configurable": {"thread_id": "t-timeout"}}

    # Pre-seed last_processed_human_id matching the new HumanMessage id so
    # turn_init does NOT reset turn_started_at (which we want stale).
    agent.invoke(
        {
            "messages": [HumanMessage(content="go", id="h1")],
            "turn_started_at": datetime.now(timezone.utc) - timedelta(minutes=10),
            "last_processed_human_id": "h1",
        },
        config=cfg,
    )
    assert llm.calls == 1


# --- LEGACY mode --------------------------------------------------------


def test_legacy_ends_without_flag_or_auto_tool():
    llm = ScriptedLLM(
        [
            _ai_with_tool("lookup", {"q": "x"}, msg_id="ai1"),
            _ai_text("never reached", msg_id="ai2"),
        ]
    )
    agent = _agent(
        llm,
        heartbeat_config=HeartbeatConfig(
            mode=HeartbeatMode.LEGACY, auto_continue_tools=set()
        ),
    )
    cfg = {"configurable": {"thread_id": "t-legacy-noflag"}}
    agent.invoke({"messages": [HumanMessage(content="go", id="h1")]}, config=cfg)

    assert llm.calls == 1


def test_legacy_continues_with_request_heartbeat_true():
    llm = ScriptedLLM(
        [
            _ai_with_tool(
                "lookup", {"q": "x", "request_heartbeat": True}, msg_id="ai1"
            ),
            _ai_text("done", msg_id="ai2"),
        ]
    )
    agent = _agent(
        llm,
        heartbeat_config=HeartbeatConfig(
            mode=HeartbeatMode.LEGACY, auto_continue_tools=set()
        ),
    )
    cfg = {"configurable": {"thread_id": "t-legacy-flag"}}
    agent.invoke({"messages": [HumanMessage(content="go", id="h1")]}, config=cfg)

    assert llm.calls == 2


def test_legacy_continues_via_auto_continue_tools():
    llm = ScriptedLLM(
        [
            _ai_with_tool("auto_thing", {"x": 1}, msg_id="ai1"),
            _ai_text("done", msg_id="ai2"),
        ]
    )
    agent = _agent(
        llm,
        tools_list=(auto_thing,),
        heartbeat_config=HeartbeatConfig(
            mode=HeartbeatMode.LEGACY, auto_continue_tools={"auto_thing"}
        ),
    )
    cfg = {"configurable": {"thread_id": "t-legacy-auto"}}
    agent.invoke({"messages": [HumanMessage(content="go", id="h1")]}, config=cfg)

    assert llm.calls == 2


# --- Loop detection -----------------------------------------------------


def test_loop_detection_injects_warning_then_ends():
    llm = InfiniteToolLLM(args_factory=lambda _i: {"q": "same"})
    agent = _agent(
        llm,
        heartbeat_config=HeartbeatConfig(
            loop_detection_threshold=3, max_chained_heartbeats=100
        ),
    )
    cfg = {"configurable": {"thread_id": "t-loop"}}
    state = agent.invoke({"messages": [HumanMessage(content="go", id="h1")]}, config=cfg)

    warnings = [
        m
        for m in state["messages"]
        if isinstance(m, SystemMessage) and "Loop detected" in m.content
    ]
    assert len(warnings) == 1
    # 3 calls before warning + 1 call after warning → 4 total LLM calls.
    assert llm.calls == 4


# --- Turn boundary ------------------------------------------------------


def test_new_human_message_resets_per_turn_counters():
    llm = ScriptedLLM(
        [
            _ai_with_tool("lookup", {"q": "x"}, msg_id="ai1"),
            _ai_text("done1", msg_id="ai2"),
            _ai_with_tool("lookup", {"q": "y"}, msg_id="ai3"),
            _ai_text("done2", msg_id="ai4"),
        ]
    )
    agent = _agent(llm)
    cfg = {"configurable": {"thread_id": "t-reset"}}

    agent.invoke({"messages": [HumanMessage(content="first", id="h1")]}, config=cfg)
    snap1 = agent.get_state(cfg).values
    assert snap1["last_processed_human_id"] == "h1"
    assert snap1["chained_heartbeats"] == 1

    agent.invoke({"messages": [HumanMessage(content="second", id="h2")]}, config=cfg)
    snap2 = agent.get_state(cfg).values

    assert snap2["last_processed_human_id"] == "h2"
    # If the counter had not reset, it would be 2; the second turn ran exactly
    # one tool call, so 1 confirms the reset happened.
    assert snap2["chained_heartbeats"] == 1
