"""Embedders para Archival Memory (cf. memGPT-resumen.md §"¿Archival Storage es lo mismo que RAG?").

El paper usa Contriever-MS-MARCO. Aquí exponemos dos backends:

- ``LocalSentenceTransformersEmbedder`` (default): corre en CPU/GPU sin red.
  Por defecto carga ``all-MiniLM-L6-v2`` (rápido, 384 dims, calidad razonable).
  Para reproducir más fielmente al paper se puede pasar
  ``model_name="facebook/contriever-msmarco"``.
- ``OpenAIEmbedder``: usa la API de OpenAI (``text-embedding-3-small`` por
  defecto, 1536 dims). Requiere ``OPENAI_API_KEY``.

Interfaz: cada embedder es callable ``(text: str) -> list[float]`` y expone
``embed_batch(texts: list[str]) -> list[list[float]]`` para minimizar
overhead cuando se ingieren muchos docs de golpe.

El embedder se inyecta en ``InMemoryStore`` en el constructor; si no hay
embedder el store cae al matching por substring (útil para tests).
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    pass


class Embedder(ABC):
    """Protocolo común. Cualquier callable ``(str) -> list[float]`` también
    sirve; esta clase añade ``embed_batch`` para batch eficiente.
    """

    @abstractmethod
    def __call__(self, text: str) -> list[float]: ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Default: llamada serial. Subclases que tengan API batch lo overriden."""
        return [self(t) for t in texts]


class LocalSentenceTransformersEmbedder(Embedder):
    """Embedder local con sentence-transformers.

    Carga el modelo en construcción (warm start cuesta ~1-3s la primera vez).
    Reusa el mismo modelo entre llamadas. Para reproducir el paper:
    ``LocalSentenceTransformersEmbedder("facebook/contriever-msmarco")``.
    """

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        device: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depende del entorno
            raise ImportError(
                "sentence-transformers no está instalado. "
                "Instálalo con: uv sync --extra embeddings-local"
            ) from exc

        self._model = SentenceTransformer(model_name, device=device)
        self._model_name = model_name

    def __call__(self, text: str) -> list[float]:
        vec = self._model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        mat = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [row.tolist() for row in mat]


class OpenAIEmbedder(Embedder):
    """Embedder vía API de OpenAI.

    Por defecto ``text-embedding-3-small`` (1536 dims, $0.02 / 1M tokens).
    Para más calidad: ``text-embedding-3-large`` (3072 dims, $0.13 / 1M).
    """

    DEFAULT_MODEL = "text-embedding-3-small"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        api_key: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "openai no está instalado. Está en las deps del proyecto, "
                "verifica que `uv sync` se ejecutó correctamente."
            ) from exc

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY no está definida. Añádela a .env o pasa "
                "api_key= explícitamente."
            )
        self._client = OpenAI(api_key=key)
        self._model = model

    def __call__(self, text: str) -> list[float]:
        response = self._client.embeddings.create(model=self._model, input=text)
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # OpenAI acepta hasta 2048 inputs por request, pero el límite real
        # suele venir por tokens (8192 totales). Troceamos en bloques de
        # 256 docs como compromiso seguro.
        out: list[list[float]] = []
        for i in range(0, len(texts), 256):
            chunk = texts[i : i + 256]
            response = self._client.embeddings.create(model=self._model, input=chunk)
            out.extend(item.embedding for item in response.data)
        return out


def make_embedder(
    provider: str,
    *,
    model_name: str | None = None,
) -> Embedder | None:
    """Factory para uso desde CLI.

    Args:
        provider: ``"local"``, ``"openai"`` o ``"none"`` (devuelve ``None``
            para que el store caiga al substring matching).
        model_name: override del modelo por defecto. Para ``"local"``
            cualquier id de HF; para ``"openai"`` un id de embedding model.

    Returns:
        Una instancia de ``Embedder`` o ``None`` si ``provider == "none"``.
    """
    p = provider.lower()
    if p == "none":
        return None
    if p == "local":
        return LocalSentenceTransformersEmbedder(
            model_name or LocalSentenceTransformersEmbedder.DEFAULT_MODEL
        )
    if p == "openai":
        return OpenAIEmbedder(model_name or OpenAIEmbedder.DEFAULT_MODEL)
    raise ValueError(
        f"provider desconocido: {provider!r} (esperaba 'local', 'openai' o 'none')"
    )


__all__ = [
    "Embedder",
    "LocalSentenceTransformersEmbedder",
    "OpenAIEmbedder",
    "make_embedder",
]
