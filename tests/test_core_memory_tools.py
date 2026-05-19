"""Tests de las tools que mutan Core Memory devolviendo Command(...)."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from memgpt.agent import build_agent
from memgpt.core_memory import CoreMemory, MemoryBlock, default_core_memory
from memgpt.state import MemGPTState


class ScriptedLLM:
    """LLM stub que reproduce una secuencia de respuestas predeterminadas."""

    def __init__(self, responses: list[AIMessage]) -> None:
        self.responses = list(responses)
        self.calls: list[list] = []

    def bind_tools(self, _tools):
        return self

    def invoke(self, messages):
        self.calls.append(list(messages))
        if not self.responses:
            return AIMessage(content="done")
        return self.responses.pop(0)


def _tool_call(name: str, args: dict, call_id: str = "call_1") -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def test_core_memory_append_updates_state_via_command():
    llm = ScriptedLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        "core_memory_append",
                        {"label": "human", "content": "name=Maximo"},
                    )
                ],
            ),
            AIMessage(content="updated"),
        ]
    )
    saver = MemorySaver()
    agent = build_agent(llm=llm, checkpointer=saver)
    cfg = {"configurable": {"thread_id": "t-append"}}

    agent.invoke({"messages": [HumanMessage(content="remember me")]}, config=cfg)

    snapshot = agent.get_state(cfg).values
    assert snapshot["core_memory"].get("human").value == "name=Maximo"

    msgs = snapshot["messages"]
    assert any(isinstance(m, ToolMessage) and "Appended" in (m.content or "") for m in msgs)


def test_core_memory_replace_updates_block():
    llm = ScriptedLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        "core_memory_replace",
                        {"label": "human", "old": "Bob", "new": "Alice"},
                    )
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    initial_state = {
        "messages": [HumanMessage(content="rename me")],
        "core_memory": default_core_memory(human="name=Bob"),
    }
    agent = build_agent(llm=llm, checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "t-replace"}}
    agent.invoke(initial_state, config=cfg)

    snapshot = agent.get_state(cfg).values
    assert snapshot["core_memory"].get("human").value == "name=Alice"


def test_core_memory_create_block_adds_block():
    llm = ScriptedLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        "core_memory_create_block",
                        {
                            "label": "project_alpha",
                            "initial_content": "deadline=2026-06-01",
                            "limit": 500,
                        },
                    )
                ],
            ),
            AIMessage(content="created"),
        ]
    )
    agent = build_agent(llm=llm, checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "t-create"}}
    agent.invoke({"messages": [HumanMessage(content="track project")]}, config=cfg)

    cm: CoreMemory = agent.get_state(cfg).values["core_memory"]
    assert cm.has("project_alpha")
    assert cm.get("project_alpha").value == "deadline=2026-06-01"


def test_core_memory_delete_block_removes_block():
    llm = ScriptedLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        "core_memory_delete_block",
                        {"label": "human"},
                    )
                ],
            ),
            AIMessage(content="deleted"),
        ]
    )
    agent = build_agent(llm=llm, checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "t-delete"}}
    agent.invoke({"messages": [HumanMessage(content="forget me")]}, config=cfg)

    cm: CoreMemory = agent.get_state(cfg).values["core_memory"]
    assert not cm.has("human")
    assert cm.has("assistant")


def test_invalid_label_in_create_returns_tool_error():
    llm = ScriptedLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        "core_memory_create_block",
                        {"label": "Bad-Label", "initial_content": "x", "limit": 100},
                    )
                ],
            ),
            AIMessage(content="acknowledged"),
        ]
    )
    agent = build_agent(llm=llm, checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "t-bad"}}
    agent.invoke({"messages": [HumanMessage(content="try invalid")]}, config=cfg)

    msgs = agent.get_state(cfg).values["messages"]
    err = next(m for m in msgs if isinstance(m, ToolMessage))
    assert err.status == "error"
    assert "label" in (err.content or "").lower()


def test_core_memory_text_is_injected_into_prompt():
    llm = ScriptedLLM([AIMessage(content="ok")])
    agent = build_agent(llm=llm, checkpointer=MemorySaver())
    state = MemGPTState(
        messages=[HumanMessage(content="hi")],
        core_memory=default_core_memory(human="favorite_color=blue"),
    )
    agent.invoke(state.model_dump(), config={"configurable": {"thread_id": "t-prompt"}})

    prompt = llm.calls[0]
    sys_contents = [m.content for m in prompt if m.__class__.__name__ == "SystemMessage"]
    joined = "\n".join(sys_contents)
    assert "Core Memory" in joined
    assert "favorite_color=blue" in joined
