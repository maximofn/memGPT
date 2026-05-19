"""Tests de las tools de Recall y Archival (Fase 4)."""

from __future__ import annotations

from datetime import datetime, timezone

from memgpt.memory_store import InMemoryStore
from memgpt.recall_archival_tools import make_recall_archival_tools


def test_factory_returns_three_tools():
    store = InMemoryStore()
    tools = make_recall_archival_tools(store)
    names = {t.name for t in tools}
    assert names == {
        "conversation_search",
        "archival_memory_insert",
        "archival_memory_search",
    }


def test_archival_insert_and_search_via_tools():
    store = InMemoryStore()
    insert, search = (
        next(t for t in make_recall_archival_tools(store) if t.name == "archival_memory_insert"),
        next(t for t in make_recall_archival_tools(store) if t.name == "archival_memory_search"),
    )
    out = insert.invoke({"content": "Pluto was reclassified in 2006."})
    assert "Inserted into archival memory" in out
    found = search.invoke({"query": "Pluto"})
    assert "Pluto was reclassified" in found


def test_conversation_search_no_match_response():
    store = InMemoryStore()
    search = next(
        t for t in make_recall_archival_tools(store) if t.name == "conversation_search"
    )
    out = search.invoke({"query": "anything"})
    assert out == "(no matches)"


def test_conversation_search_filters_by_date():
    store = InMemoryStore()
    store.persist_message(
        content="early",
        role="user",
        occurred_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        message_id="m-early",
    )
    store.persist_message(
        content="late",
        role="user",
        occurred_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        message_id="m-late",
    )
    search = next(
        t for t in make_recall_archival_tools(store) if t.name == "conversation_search"
    )
    out = search.invoke(
        {
            "query": "early",
            "start_date": "2025-12-01T00:00:00+00:00",
            "end_date": "2026-02-01T00:00:00+00:00",
        }
    )
    assert "[user] early" in out
    out = search.invoke(
        {
            "query": "late",
            "start_date": "2025-12-01T00:00:00+00:00",
            "end_date": "2026-02-01T00:00:00+00:00",
        }
    )
    assert out == "(no matches)"


def test_conversation_search_invalid_date_returns_error():
    store = InMemoryStore()
    search = next(
        t for t in make_recall_archival_tools(store) if t.name == "conversation_search"
    )
    out = search.invoke({"query": "x", "start_date": "not-a-date"})
    assert out.startswith("ERROR:")
