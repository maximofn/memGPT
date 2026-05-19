"""Phase 7 — persistencia robusta entre sesiones.

Cierra el ciclo de los componentes que ya tienen contrato de persistencia:

- **Checkpointer LangGraph**: pasa de `MemorySaver` a `PostgresSaver`. Cada
  step del grafo flushea el `MemGPTState` entero (Core Memory + FIFO +
  recursive_summary + contadores de heartbeat + step_count + ids
  persistidos) a Postgres dentro de su propia transacción → atomicidad
  por step "gratis".
- **Recall + Archival**: ya viven en Graphiti (Neo4j), persistencia
  garantizada por construcción desde Fase 4.
- **Wall-clock events**: ya tienen `PostgresEventStore` desde Fase 6;
  ``EventRegistry.restore()`` los recarga al arrancar.

`build_persistent_agent(...)` cablea los tres en una sola llamada para
que el código cliente no tenga que re-derivar la receta.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from langgraph.checkpoint.base import BaseCheckpointSaver

from .agent import build_agent
from .config import get_settings
from .events import (
    EventRegistry,
    PostgresEventStore,
    default_wallclock_dispatcher,
)
from .heartbeat import HeartbeatConfig
from .memory_store import MemoryStore
from .queue_manager import QueueManagerConfig
from .summarizer import LLMCallable


@contextmanager
def postgres_checkpointer(dsn: str) -> Iterator[BaseCheckpointSaver]:
    """Context manager para un `PostgresSaver` listo para usar.

    Llama a `setup()` la primera vez (idempotente: crea las tablas
    `checkpoints*` si no existen). Devuelve el saver dentro del with;
    al salir cierra la conexión.

    Lo exponemos como context manager porque la API oficial de
    `PostgresSaver.from_conn_string` lo es: forzarla a un constructor
    plano filtraría el manejo de la pool.

    Ejemplo:

        with postgres_checkpointer(dsn) as saver:
            agent = build_agent(checkpointer=saver, ...)
            agent.invoke(...)
    """
    from langgraph.checkpoint.postgres import PostgresSaver
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    with PostgresSaver.from_conn_string(dsn) as saver:
        # Registrar tipos custom para que el deserializador no caiga al
        # pickle fallback (bloqueado en versiones futuras de langgraph y
        # entretanto emite un warning ruidoso en cada lectura). El saver
        # expone `serde` como atributo de instancia, así que basta con
        # sustituirlo tras instanciarlo.
        saver.serde = JsonPlusSerializer(
            allowed_msgpack_modules=[("memgpt.core_memory", "CoreMemory")],
        )
        saver.setup()
        yield saver


def build_persistent_agent(
    *,
    checkpointer: BaseCheckpointSaver,
    memory_store: MemoryStore,
    event_store_dsn: str | None = None,
    event_registry: EventRegistry | None = None,
    install_default_dispatcher: bool = True,
    queue_config: QueueManagerConfig | None = None,
    heartbeat_config: HeartbeatConfig | None = None,
    summarizer_callable: LLMCallable | None = None,
    **build_agent_kwargs: Any,
) -> tuple[Any, EventRegistry]:
    """Construye el agente con todas las capas de persistencia activas.

    Devuelve `(agent, event_registry)` para que el cliente pueda
    registrar/listar eventos sin re-importar el módulo.

    Parameters
    ----------
    checkpointer:
        Cualquier `BaseCheckpointSaver` (típicamente el `PostgresSaver`
        que devuelve `postgres_checkpointer`). Se pasa intacto a
        `build_agent`.
    memory_store:
        Backend Recall + Archival. Producción → `GraphitiStore` (Fase 4);
        tests → `InMemoryStore`.
    event_store_dsn:
        Si se pasa, crea un `PostgresEventStore(dsn)` para los wall-clock
        events. Si es `None`, los eventos solo viven en memoria
        (útil cuando el agente solo usa iteration events).
    event_registry:
        Permite reusar un registry pre-existente (p. ej. compartido entre
        varios agentes). Si es `None` se construye uno nuevo.
    install_default_dispatcher:
        Si `True` (default), instala `default_wallclock_dispatcher(agent)`
        en el registry tras construir el agente, cerrando el ciclo
        ``registry → agent → registry`` que de otro modo requeriría que
        el cliente lo hiciera a mano.
    """
    if event_registry is None:
        store = PostgresEventStore(event_store_dsn) if event_store_dsn else None
        event_registry = EventRegistry(store=store)
    elif event_store_dsn is not None:
        # No tocamos el store del registry pre-existente: avisar de la
        # ambigüedad ayuda a detectar configs inconsistentes.
        raise ValueError(
            "pass either event_registry or event_store_dsn, not both"
        )

    agent = build_agent(
        checkpointer=checkpointer,
        memory_store=memory_store,
        event_registry=event_registry,
        queue_config=queue_config,
        heartbeat_config=heartbeat_config,
        summarizer_callable=summarizer_callable,
        **build_agent_kwargs,
    )

    if install_default_dispatcher:
        event_registry.set_wallclock_dispatcher(default_wallclock_dispatcher(agent))

    # Recargar wall-clock events persistidos antes de que el cliente arranque
    # el scheduler. Si el store es None es no-op.
    event_registry.restore()

    return agent, event_registry


def default_persistent_agent(
    *,
    memory_store: MemoryStore,
    **kwargs: Any,
) -> tuple[Any, EventRegistry]:
    """Atajo: usa los DSN de `Settings` y abre el checkpointer en el sitio.

    Pensado para scripts y la app principal. **No** es un context manager:
    la pool del PostgresSaver se mantiene viva mientras el proceso. Para
    scripts cortos o tests prefiere `postgres_checkpointer(...)` + un
    `build_persistent_agent(...)` explícito dentro del `with`.
    """
    from langgraph.checkpoint.postgres import PostgresSaver
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    settings = get_settings()
    saver = PostgresSaver.from_conn_string(settings.postgres_dsn).__enter__()
    saver.serde = JsonPlusSerializer(
        allowed_msgpack_modules=[("memgpt.core_memory", "CoreMemory")],
    )
    saver.setup()
    return build_persistent_agent(
        checkpointer=saver,
        memory_store=memory_store,
        event_store_dsn=settings.postgres_dsn,
        **kwargs,
    )
