"""Unit tests del grafo: estructura, persistencia y reducer de mensajes.

Usan un LLM stub para no depender de proveedores externos.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from memgpt.agent import build_agent
from memgpt.tools import get_current_time


class EchoLLM:
    """Devuelve un AIMessage que ecoa el último HumanMessage del prompt."""

    def __init__(self) -> None:
        self.calls: list[list[Any]] = []

    def bind_tools(self, _tools):
        return self

    def invoke(self, messages):
        self.calls.append(list(messages))
        last_human = next(
            (m for m in reversed(messages) if isinstance(m, HumanMessage)),
            None,
        )
        text = last_human.content if last_human else ""
        return AIMessage(content=f"echo:{text}")


def test_graph_has_expected_nodes():
    agent = build_agent(llm=EchoLLM(), tools=[get_current_time])
    nodes = set(agent.get_graph().nodes)
    assert {"agent", "tools"} <= nodes


def test_basic_invocation_returns_response():
    agent = build_agent(llm=EchoLLM(), tools=[get_current_time])
    out = agent.invoke(
        {"messages": [HumanMessage(content="hola")]},
        config={"configurable": {"thread_id": "t1"}},
    )
    assert any(isinstance(m, AIMessage) and m.content == "echo:hola" for m in out["messages"])


def test_state_persists_across_invocations_with_same_thread_id():
    saver = MemorySaver()
    agent = build_agent(llm=EchoLLM(), tools=[get_current_time], checkpointer=saver)
    cfg = {"configurable": {"thread_id": "t-persist"}}

    agent.invoke({"messages": [HumanMessage(content="msg-1")]}, config=cfg)
    second = agent.invoke({"messages": [HumanMessage(content="msg-2")]}, config=cfg)

    contents = [m.content for m in second["messages"]]
    assert "msg-1" in contents
    assert "msg-2" in contents
    assert any(c == "echo:msg-1" for c in contents)
    assert any(c == "echo:msg-2" for c in contents)


def test_recursive_summary_is_injected_as_system_message():
    llm = EchoLLM()
    agent = build_agent(llm=llm, tools=[get_current_time])
    agent.invoke(
        {
            "messages": [HumanMessage(content="q")],
            "recursive_summary": "Earlier the user mentioned project X.",
        },
        config={"configurable": {"thread_id": "t-sum"}},
    )

    prompt = llm.calls[0]
    sys_contents = [m.content for m in prompt if m.__class__.__name__ == "SystemMessage"]
    assert any("project X" in c for c in sys_contents)


def test_get_current_time_returns_iso_string():
    out = get_current_time.invoke({})
    assert "T" in out
    assert out.endswith("+00:00") or out.endswith("Z")
