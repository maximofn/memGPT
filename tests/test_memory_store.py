"""Tests del backend en memoria de Recall + Archival (Fase 4)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from memgpt.memory_store import InMemoryStore


def _utc(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=timezone.utc)


def test_persist_message_and_search_roundtrip():
    store = InMemoryStore()
    store.persist_message(
        content="The user's birthday is March 5",
        role="user",
        occurred_at=_utc(2026, 1, 1),
        message_id="m1",
    )
    results = store.search_conversation("birthday")
    assert len(results) == 1
    assert results[0].content == "The user's birthday is March 5"
    assert results[0].role == "user"
    assert results[0].message_id == "m1"


def test_persist_message_is_idempotent_by_id():
    store = InMemoryStore()
    store.persist_message(
        content="hello", role="user", occurred_at=_utc(2026, 1, 1), message_id="m1"
    )
    store.persist_message(
        content="hello again",
        role="user",
        occurred_at=_utc(2026, 1, 2),
        message_id="m1",
    )
    assert len(store.messages) == 1


def test_search_conversation_filters_by_date_range():
    store = InMemoryStore()
    for i in range(5):
        store.persist_message(
            content=f"event {i}",
            role="user",
            occurred_at=_utc(2026, 1, 1) + timedelta(days=i),
            message_id=f"m{i}",
        )
    res = store.search_conversation(
        "event", start_date=_utc(2026, 1, 2), end_date=_utc(2026, 1, 4)
    )
    assert {r.content for r in res} == {"event 1", "event 2", "event 3"}


def test_search_conversation_respects_limit():
    store = InMemoryStore()
    for i in range(5):
        store.persist_message(
            content=f"event {i}",
            role="user",
            occurred_at=_utc(2026, 1, 1) + timedelta(days=i),
            message_id=f"m{i}",
        )
    res = store.search_conversation("event", limit=2)
    assert len(res) == 2


def test_archival_insert_and_search_roundtrip():
    store = InMemoryStore()
    eid = store.insert_archival(content="Cape Cod is a peninsula in Massachusetts.")
    assert eid.startswith("arc-")
    res = store.search_archival("Cape Cod")
    assert len(res) == 1
    assert res[0].episode_id == eid


def test_archival_search_no_match_returns_empty():
    store = InMemoryStore()
    store.insert_archival(content="something")
    assert store.search_archival("absent") == []


def test_recall_and_archival_are_independent():
    store = InMemoryStore()
    store.persist_message(
        content="foo recall", role="user", occurred_at=_utc(2026, 1, 1), message_id="m1"
    )
    store.insert_archival(content="foo archival")

    recall_hits = store.search_conversation("foo")
    archival_hits = store.search_archival("foo")
    assert len(recall_hits) == 1
    assert len(archival_hits) == 1
    assert recall_hits[0].content == "foo recall"
    assert archival_hits[0].content == "foo archival"


# ---------------------------------------------------------------------------
# Búsqueda semántica con embedder inyectado.
# ---------------------------------------------------------------------------


def _word_overlap_embedder(text: str) -> list[float]:
    """Embedder de juguete determinista para tests.

    Vectoriza por presencia/ausencia de palabras del vocabulario fijo. La
    cosine similarity entre dos textos termina siendo proporcional al
    overlap de palabras — semánticamente trivial pero suficiente para
    comprobar que el ranking por cosine se ejecuta.
    """
    vocab = ["cape", "cod", "peninsula", "massachusetts", "hallelujah", "leonard", "cohen", "song"]
    words = set(text.lower().split())
    return [1.0 if w in words else 0.0 for w in vocab]


def test_archival_search_uses_cosine_when_embedder_present():
    store = InMemoryStore(embedder=_word_overlap_embedder)
    store.insert_archival(content="Cape Cod is a peninsula in Massachusetts")
    store.insert_archival(content="Leonard Cohen wrote the song Hallelujah")

    # Query sin overlap léxico con doc1 ("written" no está en el doc),
    # pero comparte "song hallelujah" con el doc2 → cosine alto solo con doc2.
    results = store.search_archival("who wrote the song hallelujah")
    assert len(results) >= 1
    assert "Leonard Cohen" in results[0].content


def test_archival_search_substring_fallback_when_no_embedder():
    """Sin embedder, el fallback substring devuelve solo matches literales."""
    store = InMemoryStore()  # embedder=None por default
    store.insert_archival(content="Cape Cod is a peninsula")
    # "wrote" no está literal en el doc → 0 resultados (esto es exactamente
    # el problema que motivó añadir embedders al benchmark).
    assert store.search_archival("who wrote about peninsulas") == []


def test_insert_archival_calls_embedder_once_and_stores_vector():
    calls: list[str] = []

    def tracking_embedder(text: str) -> list[float]:
        calls.append(text)
        return [1.0, 0.0]

    store = InMemoryStore(embedder=tracking_embedder)
    store.insert_archival(content="hello")
    store.insert_archival(content="world")

    # Una llamada por insert.
    assert calls == ["hello", "world"]
    assert all(ep.embedding is not None for ep in store.archival)


def test_archival_search_skips_unembedded_episodes_in_semantic_mode():
    """Si insertaste docs antes de configurar embedder, salen del ranking."""
    store_no_embed = InMemoryStore()
    store_no_embed.insert_archival(content="legacy doc without embedding")

    # Promovemos el mismo store a modo semántico in-place para reproducir
    # el caso "configuré embedder tarde". Los docs sin embedding no deben
    # romper el ranking.
    store_no_embed._embedder = _word_overlap_embedder  # type: ignore[attr-defined]
    store_no_embed.insert_archival(content="Cape Cod peninsula Massachusetts")

    results = store_no_embed.search_archival("Cape Cod")
    contents = [r.content for r in results]
    assert "Cape Cod peninsula Massachusetts" in contents
    assert "legacy doc without embedding" not in contents


def test_archival_search_ranks_by_similarity_descending():
    store = InMemoryStore(embedder=_word_overlap_embedder)
    store.insert_archival(content="cape cod song")  # 3 palabras del vocab que querremos
    store.insert_archival(content="hallelujah leonard cohen song")  # 4 palabras
    store.insert_archival(content="random unrelated text")  # 0 palabras

    results = store.search_archival("hallelujah leonard cohen song", limit=3)
    # El doc con 4 palabras del vocab matchea exacto → cosine = 1.0; gana.
    assert results[0].content == "hallelujah leonard cohen song"
    # El "random unrelated" tiene cosine 0 con la query — debería quedar
    # último (o fuera si sort estable lo desempata).
    assert results[-1].content != "hallelujah leonard cohen song"
