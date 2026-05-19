"""Unit tests del summarizer custom (Plan B)."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from memgpt.summarizer import build_summariser_prompt, regenerate_recursive_summary


def test_prompt_includes_previous_summary_when_present():
    prompt = build_summariser_prompt(
        old_summary="The user is named Maximo.",
        evicted_messages=[HumanMessage(content="hi", id="h1")],
        core_memory_text="",
    )
    user = prompt[-1]["content"]
    assert "Maximo" in user
    assert "FOLD" in user.upper()


def test_prompt_marks_first_flush_when_no_previous_summary():
    prompt = build_summariser_prompt(
        old_summary=None,
        evicted_messages=[HumanMessage(content="hi", id="h1")],
        core_memory_text="",
    )
    user = prompt[-1]["content"]
    assert "first flush" in user.lower() or "(none" in user.lower()


def test_prompt_includes_core_memory_as_anti_hint():
    prompt = build_summariser_prompt(
        old_summary=None,
        evicted_messages=[HumanMessage(content="hi", id="h1")],
        core_memory_text="[human] name=Maximo",
    )
    user = prompt[-1]["content"]
    assert "name=Maximo" in user
    assert "DO NOT" in user.upper()


def test_prompt_serializes_tool_calls_and_results():
    msgs = [
        HumanMessage(content="search docs", id="h1"),
        AIMessage(
            content="",
            tool_calls=[
                {"id": "c1", "name": "search", "args": {"q": "memgpt"}, "type": "tool_call"}
            ],
        ),
        ToolMessage(content="hit-1, hit-2", tool_call_id="c1", name="search"),
        AIMessage(content="found two hits"),
    ]
    prompt = build_summariser_prompt(
        old_summary=None,
        evicted_messages=msgs,
        core_memory_text="",
    )
    user = prompt[-1]["content"]
    assert "search" in user
    assert "memgpt" in user
    assert "hit-1" in user
    assert "found two hits" in user


def test_regenerate_returns_old_summary_when_no_evicted():
    out = regenerate_recursive_summary(
        old_summary="prior",
        evicted_messages=[],
        llm_callable=lambda _msgs: "should-not-be-called",
    )
    assert out == "prior"


def test_regenerate_calls_llm_with_built_prompt():
    seen: list[list[dict]] = []

    def stub(messages):
        seen.append(messages)
        return "NEW SUMMARY"

    out = regenerate_recursive_summary(
        old_summary="prior summary",
        evicted_messages=[HumanMessage(content="evicted text", id="h1")],
        core_memory_text="[human] x",
        llm_callable=stub,
    )
    assert out == "NEW SUMMARY"
    assert seen, "llm callable should have been invoked"
    user = seen[0][-1]["content"]
    assert "prior summary" in user
    assert "evicted text" in user
    assert "[human] x" in user


def test_regenerate_falls_back_to_old_summary_when_llm_returns_empty():
    out = regenerate_recursive_summary(
        old_summary="prior",
        evicted_messages=[HumanMessage(content="x", id="h1")],
        llm_callable=lambda _: "   ",
    )
    assert out == "prior"
