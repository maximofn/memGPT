"""Backend de Recall + Archival Memory.

Decisiones (cf. `posts/papers/memGPT-plan.md` §5):

- **Interfaz síncrona**. El grafo de LangGraph se invoca con `.invoke()`
  síncrono en los tests (mezclar nodos `async` con `.invoke()` falla con
  `TypeError: No synchronous function provided`). La capa cliente expone
  métodos `def` y, cuando el backend real (Graphiti) sea asíncrono, lo
  envolvemos en un event loop dedicado en un hilo de fondo.
- **Una sola instancia por agente**: namespacing de la sección 5 del plan.
  Un `group_id` constante (default `"default"`) o un `Graphiti` con su
  propia BD aísla los datos. Si en el futuro se añaden más agentes, se
  pasa un `group_id` por agente sin cambiar nada del runtime.
- **Misma instancia para Recall y Archival**: el paper los trata como
  dos almacenes lógicos pero la búsqueda de Graphiti es la misma. Los
  diferenciamos con `source_description` (`"conversation:<role>"` vs
  `"archival"`) para poder filtrar.
- **Modelo bi-temporal**: cada episodio guarda `occurred_at` (cuándo
  pasó) y `learned_at` (cuándo lo aprendimos). Para mensajes de
  conversación coinciden; para inserts en Archival pueden diferir si la
  información se refiere al pasado.
- **Dedupe por `message_id`**: cada mensaje tiene un id LangChain;
  guardamos el conjunto persistido para no volver a insertar.
"""

from __future__ import annotations

import asyncio
import math
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable

from pydantic import BaseModel, Field

if TYPE_CHECKING:  # pragma: no cover
    from graphiti_core import Graphiti


# Tipo público: cualquier callable (str) -> list[float] sirve como embedder.
EmbedderFn = Callable[[str], list[float]]


class RecallEpisode(BaseModel):
    """Mensaje de conversación recuperado de Recall Memory."""

    content: str
    role: str  # "user" / "assistant" / "tool" / "system"
    occurred_at: datetime
    learned_at: datetime
    message_id: str | None = None


class ArchivalEpisode(BaseModel):
    """Texto arbitrario recuperado de Archival Memory."""

    content: str
    occurred_at: datetime
    learned_at: datetime
    episode_id: str | None = None
    # Vector pre-calculado en inserción. Excluido de la API pública del
    # tool: el agente solo ve content/timestamp.
    embedding: list[float] | None = Field(default=None, exclude=True, repr=False)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity entre dos vectores. Asume dimensiones iguales.

    Si alguno tiene norma cero (raro: embedder roto) devuelve 0.0 en lugar
    de NaN para no romper el ranking.
    """
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class MemoryStore(ABC):
    """Backend abstracto para Recall + Archival.

    Todos los métodos son síncronos para encajar con `agent.invoke()`.
    Los backends que envuelven APIs async (Graphiti) deben gestionar su
    propio event loop internamente.
    """

    @abstractmethod
    def persist_message(
        self,
        *,
        content: str,
        role: str,
        occurred_at: datetime,
        message_id: str,
    ) -> None: ...

    @abstractmethod
    def insert_archival(
        self,
        *,
        content: str,
        occurred_at: datetime | None = None,
    ) -> str: ...

    @abstractmethod
    def search_conversation(
        self,
        query: str,
        *,
        limit: int = 10,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[RecallEpisode]: ...

    @abstractmethod
    def search_archival(
        self,
        query: str,
        *,
        limit: int = 10,
    ) -> list[ArchivalEpisode]: ...

    def close(self) -> None:  # pragma: no cover - default no-op
        """Liberar recursos (event loops, drivers). Override si aplica."""
        return None


class InMemoryStore(MemoryStore):
    """Backend en memoria. Para tests, benchmarks y experimentación local.

    Búsqueda en archival:

    - **Sin embedder** (default): ``query.lower() in content.lower()`` —
      determinista, sin red, suficiente para validar la lógica del agente
      en tests unitarios.
    - **Con embedder**: cosine similarity entre el embedding de la query
      y los embeddings pre-calculados al insertar. Esto hace que el store
      sea un retriever semántico real, comparable al pgvector que usa el
      paper. Es lo que el benchmark Document QA necesita para reproducir
      los números de la Figura 5 — sin embeddings reales, el agente no
      puede recuperar docs cuyo texto no contenga literalmente la query.

    Args:
        embedder: callable ``(text) -> vector``. Si se pasa, ``insert_archival``
            embebe cada doc en la inserción y ``search_archival`` ranquea
            por cosine. Si es ``None``, fallback substring.
    """

    def __init__(self, embedder: EmbedderFn | None = None) -> None:
        self._messages: list[RecallEpisode] = []
        self._archival: list[ArchivalEpisode] = []
        self._persisted_message_ids: set[str] = set()
        self._archival_seq: int = 0
        self._embedder = embedder

    def persist_message(
        self,
        *,
        content: str,
        role: str,
        occurred_at: datetime,
        message_id: str,
    ) -> None:
        if message_id in self._persisted_message_ids:
            return
        self._persisted_message_ids.add(message_id)
        self._messages.append(
            RecallEpisode(
                content=content,
                role=role,
                occurred_at=occurred_at,
                learned_at=datetime.now(timezone.utc),
                message_id=message_id,
            )
        )

    def insert_archival(
        self,
        *,
        content: str,
        occurred_at: datetime | None = None,
    ) -> str:
        now = datetime.now(timezone.utc)
        self._archival_seq += 1
        episode_id = f"arc-{self._archival_seq}"
        embedding = self._embedder(content) if self._embedder is not None else None
        self._archival.append(
            ArchivalEpisode(
                content=content,
                occurred_at=occurred_at or now,
                learned_at=now,
                episode_id=episode_id,
                embedding=embedding,
            )
        )
        return episode_id

    def search_conversation(
        self,
        query: str,
        *,
        limit: int = 10,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[RecallEpisode]:
        q = query.lower()
        results: list[RecallEpisode] = []
        for ep in self._messages:
            if start_date is not None and ep.occurred_at < start_date:
                continue
            if end_date is not None and ep.occurred_at > end_date:
                continue
            if q in ep.content.lower():
                results.append(ep)
        return results[:limit]

    def search_archival(
        self,
        query: str,
        *,
        limit: int = 10,
    ) -> list[ArchivalEpisode]:
        if self._embedder is None:
            q = query.lower()
            results = [ep for ep in self._archival if q in ep.content.lower()]
            return results[:limit]

        # Modo semántico: rank by cosine similarity entre query y los
        # embeddings pre-calculados en insert_archival.
        q_vec = self._embedder(query)
        scored: list[tuple[float, ArchivalEpisode]] = []
        for ep in self._archival:
            if ep.embedding is None:
                # Doc insertado antes de configurar el embedder. Lo dejamos
                # fuera del ranking semántico — el caller debería
                # re-ingerir si quiere búsqueda completa.
                continue
            scored.append((_cosine_similarity(q_vec, ep.embedding), ep))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [ep for _, ep in scored[:limit]]

    # Helpers de inspección para tests
    @property
    def messages(self) -> list[RecallEpisode]:
        return list(self._messages)

    @property
    def archival(self) -> list[ArchivalEpisode]:
        return list(self._archival)


class GraphitiStore(MemoryStore):
    """Backend Graphiti. Conecta a Neo4j vía un cliente `Graphiti`.

    Como Graphiti expone una API `async`, mantenemos un event loop dedicado
    en un hilo de fondo y enviamos cada coroutine con
    `run_coroutine_threadsafe`. Ventajas:

    - Funciona con `agent.invoke()` sin requerir `ainvoke()`.
    - Una única conexión / driver vive todo el ciclo (no se abre y cierra
      en cada llamada).
    - El bloqueo del hilo principal es exactamente el de la latencia de
      Graphiti, que es lo esperable.
    """

    def __init__(
        self,
        client: "Graphiti",
        *,
        group_id: str = "default",
    ) -> None:
        self._client = client
        self._group_id = group_id
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever,
            name="memgpt-graphiti-loop",
            daemon=True,
        )
        self._thread.start()
        self._persisted_message_ids: set[str] = set()

        # El driver Neo4j de Graphiti solo programa build_indices_and_constraints
        # si encuentra un event loop corriendo en el momento de la construcción
        # (neo4j_driver.py:96). Como aquí lo construimos desde código síncrono,
        # nunca se ejecuta — y la primera query fulltext (p. ej. edge_name_and_fact)
        # revienta. Lo disparamos a mano. Es idempotente.
        try:
            self._run(self._client.build_indices_and_constraints())
        except Exception:  # noqa: BLE001
            # No bloqueamos la construcción del store si la BD aún no responde:
            # el primer add_episode reintentará y dará un error más informativo.
            pass

    def _run(self, coro: Any) -> Any:
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def persist_message(
        self,
        *,
        content: str,
        role: str,
        occurred_at: datetime,
        message_id: str,
    ) -> None:
        if message_id in self._persisted_message_ids:
            return
        from graphiti_core.nodes import EpisodeType  # local import (heavy module)

        body = f"[{role}] {content}"
        self._run(
            self._client.add_episode(
                name=f"msg-{message_id}",
                episode_body=body,
                source_description=f"conversation:{role}",
                reference_time=occurred_at,
                source=EpisodeType.message,
                group_id=self._group_id,
            )
        )
        self._persisted_message_ids.add(message_id)

    def insert_archival(
        self,
        *,
        content: str,
        occurred_at: datetime | None = None,
    ) -> str:
        from graphiti_core.nodes import EpisodeType

        ts = occurred_at or datetime.now(timezone.utc)
        result = self._run(
            self._client.add_episode(
                name=f"archival-{ts.isoformat()}",
                episode_body=content,
                source_description="archival",
                reference_time=ts,
                source=EpisodeType.text,
                group_id=self._group_id,
            )
        )
        episode = getattr(result, "episode", None)
        if episode is not None and getattr(episode, "uuid", None):
            return episode.uuid
        return ""

    def _search_episodes(
        self,
        query: str,
        *,
        limit: int,
    ) -> list[Any]:
        """Run a search restricted to episodes via `search_`."""
        from graphiti_core.search.search_config_recipes import (
            COMBINED_HYBRID_SEARCH_RRF,
        )

        config = COMBINED_HYBRID_SEARCH_RRF.model_copy(deep=True)
        config.limit = limit
        results = self._run(
            self._client.search_(
                query=query,
                config=config,
                group_ids=[self._group_id],
            )
        )
        return list(getattr(results, "episodes", []) or [])

    def search_conversation(
        self,
        query: str,
        *,
        limit: int = 10,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[RecallEpisode]:
        episodes = self._search_episodes(query, limit=limit * 3)
        out: list[RecallEpisode] = []
        for ep in episodes:
            sd: str = getattr(ep, "source_description", "") or ""
            if not sd.startswith("conversation:"):
                continue
            occurred = getattr(ep, "valid_at", None) or getattr(ep, "reference_time", None)
            learned = getattr(ep, "created_at", occurred)
            if occurred is None:
                continue
            if start_date is not None and occurred < start_date:
                continue
            if end_date is not None and occurred > end_date:
                continue
            role = sd.split(":", 1)[1] if ":" in sd else "unknown"
            out.append(
                RecallEpisode(
                    content=getattr(ep, "content", "") or "",
                    role=role,
                    occurred_at=occurred,
                    learned_at=learned or occurred,
                    message_id=getattr(ep, "uuid", None),
                )
            )
            if len(out) >= limit:
                break
        return out

    def search_archival(
        self,
        query: str,
        *,
        limit: int = 10,
    ) -> list[ArchivalEpisode]:
        episodes = self._search_episodes(query, limit=limit * 3)
        out: list[ArchivalEpisode] = []
        for ep in episodes:
            sd: str = getattr(ep, "source_description", "") or ""
            if sd != "archival":
                continue
            occurred = getattr(ep, "valid_at", None) or getattr(ep, "reference_time", None)
            learned = getattr(ep, "created_at", occurred)
            if occurred is None:
                continue
            out.append(
                ArchivalEpisode(
                    content=getattr(ep, "content", "") or "",
                    occurred_at=occurred,
                    learned_at=learned or occurred,
                    episode_id=getattr(ep, "uuid", None),
                )
            )
            if len(out) >= limit:
                break
        return out

    def close(self) -> None:
        try:
            self._run(self._client.close())
        except Exception:  # pragma: no cover - best effort
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
