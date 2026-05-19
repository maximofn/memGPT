"""Tests del módulo embedders.

NO ejecutamos los embedders reales (sentence-transformers carga ~80MB de
modelo, OpenAI requiere API key). Los tests cubren:

- ``make_embedder("none")`` devuelve None.
- ``make_embedder`` con provider inválido lanza ValueError.
- ``Embedder.embed_batch`` por defecto delega en ``__call__``.
- Validación de errores de configuración (sin API key, sin la dep opcional).
"""

from __future__ import annotations

import os

import pytest

from memgpt.embedders import (
    Embedder,
    LocalSentenceTransformersEmbedder,
    OpenAIEmbedder,
    make_embedder,
)


class _Stub(Embedder):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, text: str) -> list[float]:
        self.calls.append(text)
        return [float(len(text))]


def test_embed_batch_default_delegates_to_call():
    stub = _Stub()
    out = stub.embed_batch(["a", "bb", "ccc"])
    assert out == [[1.0], [2.0], [3.0]]
    assert stub.calls == ["a", "bb", "ccc"]


def test_embed_batch_empty_returns_empty():
    stub = _Stub()
    assert stub.embed_batch([]) == []
    assert stub.calls == []


def test_make_embedder_none_returns_none():
    assert make_embedder("none") is None


def test_make_embedder_invalid_raises():
    with pytest.raises(ValueError, match="provider desconocido"):
        make_embedder("inexistente")


def test_make_embedder_case_insensitive():
    # "NONE" debería funcionar igual que "none".
    assert make_embedder("NONE") is None


def test_openai_embedder_without_api_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIEmbedder()


def test_local_embedder_class_attribute_default_model():
    """Sanity: el default apunta a un modelo razonable. Es un check
    barato — no carga el modelo."""
    assert "all-MiniLM" in LocalSentenceTransformersEmbedder.DEFAULT_MODEL


def test_openai_embedder_class_attribute_default_model():
    assert OpenAIEmbedder.DEFAULT_MODEL == "text-embedding-3-small"
