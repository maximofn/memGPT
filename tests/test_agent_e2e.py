"""Tests E2E contra un LLM real. Skippean si no hay credenciales.

Cubren los 3 criterios de "definición de hecho" de la Fase 1.
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from memgpt.agent import build_agent

load_dotenv()

pytestmark = pytest.mark.skipif(
    not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")),
    reason="No LLM provider key configured",
)


def test_agent_responds_to_basic_question():
    agent = build_agent()
    out = agent.invoke(
        {"messages": [HumanMessage(content="Reply with the single word: pong")]},
        config={"configurable": {"thread_id": "e2e-basic"}},
    )
    final = out["messages"][-1]
    assert isinstance(final, AIMessage)
    assert "pong" in (final.content or "").lower()


def test_agent_uses_tool_and_consumes_result():
    agent = build_agent()
    out = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Call the get_current_time tool and tell me the year "
                        "from the returned ISO timestamp."
                    )
                )
            ]
        },
        config={"configurable": {"thread_id": "e2e-tool"}},
    )

    tool_messages = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert tool_messages, "expected the agent to invoke the tool"

    final = out["messages"][-1]
    assert isinstance(final, AIMessage)
    assert any(str(year) in (final.content or "") for year in range(2024, 2031))


def test_state_persists_across_invocations_same_thread():
    agent = build_agent()
    cfg = {"configurable": {"thread_id": "e2e-persist"}}

    agent.invoke(
        {"messages": [HumanMessage(content="My name is Maximo. Just acknowledge.")]},
        config=cfg,
    )
    out = agent.invoke(
        {"messages": [HumanMessage(content="What is my name?")]},
        config=cfg,
    )

    final = out["messages"][-1]
    assert isinstance(final, AIMessage)
    assert "maximo" in (final.content or "").lower()


def test_agent_writes_to_core_memory_on_durable_fact():
    agent = build_agent()
    cfg = {"configurable": {"thread_id": "e2e-core-memory"}}

    agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Important durable fact about me: my favorite "
                        "programming language is Rust. Please update your "
                        "core memory accordingly."
                    )
                )
            ]
        },
        config=cfg,
    )

    snapshot = agent.get_state(cfg).values
    human_block = snapshot["core_memory"].get("human").value
    assert "rust" in human_block.lower()
