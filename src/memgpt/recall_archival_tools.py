"""Tools de Recall y Archival Memory expuestas al LLM (Fase 4).

Patrón **factory + closure**: las tools necesitan acceso al `MemoryStore`
inyectado en runtime. No podemos meter el store en el `MemGPTState` (no es
serializable) ni recurrir a un singleton global (tests poco aislados).
`make_recall_archival_tools(store)` devuelve las 3 tools cerradas sobre la
instancia que se les pase.

Tools (siguen el catálogo del paper, sección 2.3 + apéndice 4):

- ``conversation_search(query, limit, start_date, end_date)``: busca en
  Recall (mensajes pasados ya volcados al backend). El rango temporal es
  opcional y se expresa en ISO 8601.
- ``archival_memory_insert(content)``: añade texto arbitrario al Archival.
- ``archival_memory_search(query, limit)``: búsqueda semántica en Archival.

El formato de salida es **texto plano** orientado a que el LLM lo lea de
nuevo en el siguiente turno: timestamp + role + contenido por línea, o un
mensaje explícito de "no matches" para no confundirlo con un fallo silencioso.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

from langchain_core.tools import tool

from .memory_store import MemoryStore


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


def _format_recall(results: list[Any]) -> str:
    if not results:
        return "(no matches)"
    return "\n".join(
        f"[{r.occurred_at.isoformat()}] [{r.role}] {r.content}" for r in results
    )


def _format_archival(results: list[Any]) -> str:
    if not results:
        return "(no matches)"
    return "\n".join(f"[{r.occurred_at.isoformat()}] {r.content}" for r in results)


def make_recall_archival_tools(store: MemoryStore) -> list[Callable[..., Any]]:
    """Build the three Recall/Archival tools bound to ``store``.

    Returns a list of LangChain tools (decorated with ``@tool``) ready to
    pass to ``build_agent(tools=...)`` or to a ``ToolNode``.
    """

    @tool
    def conversation_search(
        query: str,
        limit: int = 5,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> str:
        """Search past conversation messages stored in Recall Memory.

        Use this when the user references something they said earlier that is
        no longer in your immediate context window, or when you need to
        recover details from previous sessions.

        Args:
            query: Free-form search string.
            limit: Maximum number of messages to return (default 5).
            start_date: Optional ISO 8601 lower bound on the message time.
            end_date: Optional ISO 8601 upper bound on the message time.

        Returns:
            One result per line as ``[timestamp] [role] content`` or
            ``(no matches)`` if nothing was found.
        """
        try:
            sd = _parse_dt(start_date)
            ed = _parse_dt(end_date)
        except ValueError as exc:
            return f"ERROR: invalid ISO 8601 date — {exc}"
        results = store.search_conversation(
            query, limit=limit, start_date=sd, end_date=ed
        )
        return _format_recall(results)

    @tool
    def archival_memory_insert(content: str) -> str:
        """Insert a piece of text into Archival Memory for later semantic retrieval.

        Use this for durable knowledge that should survive context flushes:
        learned facts, summaries, structured notes. Prefer Core Memory for
        small, always-visible facts and Archival for larger, searchable text.

        Args:
            content: Text to store. Can be any length.

        Returns:
            A short confirmation including the resulting episode id.
        """
        episode_id = store.insert_archival(content=content)
        suffix = f" (id={episode_id})" if episode_id else ""
        return f"Inserted into archival memory{suffix}."

    @tool
    def archival_memory_search(query: str, limit: int = 5) -> str:
        """Search Archival Memory for content semantically related to the query.

        Args:
            query: Free-form search string.
            limit: Maximum number of results (default 5).

        Returns:
            One result per line as ``[timestamp] content`` or
            ``(no matches)`` if nothing was found.
        """
        results = store.search_archival(query, limit=limit)
        return _format_archival(results)

    return [conversation_search, archival_memory_insert, archival_memory_search]
