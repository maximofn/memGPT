"""Phase 6 — eventos automáticos.

Dos tipos de evento que cubren el paper:

- **Iteration events** ("sleep-time agents"): callbacks que se ejecutan cada
  N pasos del agent loop. Viven en proceso (no se persisten en BD): una
  callback es un `Callable` Python no serializable, así que el código
  cliente las re-registra al arrancar.

- **Wall-clock events**: jobs de APScheduler que disparan un mensaje
  (normalmente `SystemMessage`) al agente cuando vence un cron / interval /
  fecha concreta. Los specs (sin la callable) sí se persisten en un
  ``EventStore`` para sobrevivir a reinicios; al rearrancar,
  ``EventRegistry.restore()`` los vuelve a registrar contra el dispatcher.

La separación entre `EventRegistry` (orquestador) y `EventStore`
(persistencia) sigue el patrón de los almacenes de Fase 4: la API tiene
una sola entrada y el backend es intercambiable (`InMemoryEventStore`
para tests, `PostgresEventStore` para producción).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Protocol

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.base import BaseScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger
from langchain_core.messages import AnyMessage
from pydantic import BaseModel, Field, PositiveInt

from .state import MemGPTState

IterationCallback = Callable[[MemGPTState], "dict[str, Any] | None"]
"""Callback ejecutada cada N steps. Devuelve un update parcial del estado
(mismo formato que devuelven los nodos LangGraph) o ``None`` si no hay
nada que cambiar."""

WallClockDispatcher = Callable[[str, str], None]
"""Función disparada por APScheduler: ``(agent_id, payload) -> None``.
El default que provee este módulo construye un ``SystemMessage`` y
invoca el grafo con ``thread_id=agent_id`` — pero se inyecta para que
los tests puedan capturar las llamadas sin levantar un agente real."""


class IterationEvent(BaseModel):
    """Sleep-time agent: callback que corre cada `every_n_steps` pasos."""

    name: str
    every_n_steps: PositiveInt
    callback: IterationCallback

    model_config = {"arbitrary_types_allowed": True}


_TRIGGER_BUILDERS: dict[str, Callable[..., Any]] = {
    "cron": CronTrigger,
    "interval": IntervalTrigger,
    "date": DateTrigger,
}


class WallClockEvent(BaseModel):
    """Spec serializable de un job APScheduler.

    `trigger_type` + `trigger_kwargs` se traducen a un trigger de
    APScheduler en `build_trigger()`. El payload es la cadena que el
    dispatcher envuelve en un `SystemMessage` antes de invocar el grafo.
    """

    name: str
    agent_id: str
    trigger_type: str
    trigger_kwargs: dict[str, Any] = Field(default_factory=dict)
    payload: str

    def build_trigger(self) -> Any:
        builder = _TRIGGER_BUILDERS.get(self.trigger_type)
        if builder is None:
            raise ValueError(
                f"unknown trigger_type {self.trigger_type!r} "
                f"(expected one of {sorted(_TRIGGER_BUILDERS)})"
            )
        return builder(**self.trigger_kwargs)


class EventStore(Protocol):
    """Backend de persistencia de wall-clock events."""

    def save_wallclock(self, event: WallClockEvent) -> None: ...
    def delete_wallclock(self, name: str) -> None: ...
    def list_wallclock(self) -> list[WallClockEvent]: ...


class InMemoryEventStore:
    """Implementación in-memory para tests y agentes single-process."""

    def __init__(self) -> None:
        self._events: dict[str, WallClockEvent] = {}

    def save_wallclock(self, event: WallClockEvent) -> None:
        self._events[event.name] = event

    def delete_wallclock(self, name: str) -> None:
        self._events.pop(name, None)

    def list_wallclock(self) -> list[WallClockEvent]:
        return list(self._events.values())


class PostgresEventStore:
    """Postgres-backed event store usando una única tabla JSONB.

    Se mantiene deliberadamente plano: el spec se serializa entero en
    una columna JSONB indexada por `name`. Ideal para volúmenes bajos
    (decenas-cientos de eventos por agente). Para volúmenes altos o
    queries complejas, migrar a columnas tipadas.
    """

    DEFAULT_TABLE = "memgpt_wallclock_events"

    def __init__(self, dsn: str, *, table: str = DEFAULT_TABLE) -> None:
        # Importamos psycopg perezosamente: el resto del módulo no debería
        # exigir psycopg si solo se usa el InMemoryEventStore.
        import psycopg

        self._psycopg = psycopg
        self._dsn = dsn
        self._table = table
        self._ensure_table()

    def _ensure_table(self) -> None:
        ddl = (
            f"CREATE TABLE IF NOT EXISTS {self._table} ("
            " name TEXT PRIMARY KEY,"
            " spec JSONB NOT NULL"
            ")"
        )
        with self._psycopg.connect(self._dsn) as conn, conn.cursor() as cur:
            cur.execute(ddl)
            conn.commit()

    def save_wallclock(self, event: WallClockEvent) -> None:
        sql = (
            f"INSERT INTO {self._table} (name, spec) VALUES (%s, %s) "
            "ON CONFLICT (name) DO UPDATE SET spec = EXCLUDED.spec"
        )
        spec_json = event.model_dump_json()
        with self._psycopg.connect(self._dsn) as conn, conn.cursor() as cur:
            cur.execute(sql, (event.name, spec_json))
            conn.commit()

    def delete_wallclock(self, name: str) -> None:
        sql = f"DELETE FROM {self._table} WHERE name = %s"
        with self._psycopg.connect(self._dsn) as conn, conn.cursor() as cur:
            cur.execute(sql, (name,))
            conn.commit()

    def list_wallclock(self) -> list[WallClockEvent]:
        sql = f"SELECT spec FROM {self._table} ORDER BY name"
        out: list[WallClockEvent] = []
        with self._psycopg.connect(self._dsn) as conn, conn.cursor() as cur:
            cur.execute(sql)
            for (spec,) in cur.fetchall():
                # psycopg3 devuelve JSONB ya parseado; si es str lo cargamos.
                data = json.loads(spec) if isinstance(spec, str) else spec
                out.append(WallClockEvent.model_validate(data))
        return out


class EventRegistry:
    """API unificada para registrar y disparar eventos automáticos.

    Wall-clock events se registran como jobs en un `BaseScheduler` (default
    `BackgroundScheduler`). Iteration events viven en un dict en memoria;
    el agente los consume desde su nodo `step_tick`.

    El dispatcher es inyectable: por defecto se llama a
    ``default_wallclock_dispatcher(agent)`` para construir uno ligado a un
    `CompiledStateGraph` concreto, pero los tests inyectan callables que
    solo registran las llamadas.
    """

    def __init__(
        self,
        *,
        scheduler: BaseScheduler | None = None,
        wallclock_dispatcher: WallClockDispatcher | None = None,
        store: EventStore | None = None,
    ) -> None:
        self._scheduler: BaseScheduler = scheduler or BackgroundScheduler()
        self._dispatcher = wallclock_dispatcher
        self._store = store
        self._iteration: dict[str, IterationEvent] = {}
        self._wallclock_specs: dict[str, WallClockEvent] = {}

    # ---- Iteration events ----

    def register_iteration(self, event: IterationEvent) -> None:
        if event.name in self._iteration:
            raise ValueError(f"iteration event {event.name!r} already registered")
        self._iteration[event.name] = event

    def unregister_iteration(self, name: str) -> None:
        self._iteration.pop(name, None)

    def list_iteration(self) -> list[IterationEvent]:
        return list(self._iteration.values())

    def dispatch_iteration(self, state: MemGPTState) -> dict[str, Any]:
        """Fusiona los updates de todas las callbacks aplicables.

        Solo dispara cuando ``state.step_count > 0`` y
        ``step_count % every_n_steps == 0``. Si varios callbacks devuelven
        ``messages``, se concatenan (compatible con el reducer
        ``add_messages``). Cualquier otra clave la sobrescribe el último
        callback que la modifique — el orden es de inserción.
        """
        if state.step_count <= 0:
            return {}
        merged: dict[str, Any] = {}
        merged_messages: list[AnyMessage] = []
        for ev in self._iteration.values():
            if state.step_count % ev.every_n_steps != 0:
                continue
            update = ev.callback(state)
            if not update:
                continue
            for k, v in update.items():
                if k == "messages":
                    merged_messages.extend(v)
                else:
                    merged[k] = v
        if merged_messages:
            merged["messages"] = merged_messages
        return merged

    # ---- Wall-clock events ----

    def set_wallclock_dispatcher(self, dispatcher: WallClockDispatcher) -> None:
        """Inyectar el dispatcher después de construir el registry.

        Útil cuando el dispatcher necesita capturar el agente compilado:
        primero se crea el registry, luego se compila el grafo (que puede
        tener referencias al registry), y finalmente se cierra el ciclo
        instalando el dispatcher.
        """
        self._dispatcher = dispatcher

    def register_wallclock(
        self, event: WallClockEvent, *, persist: bool = True
    ) -> None:
        if event.name in self._wallclock_specs:
            raise ValueError(f"wallclock event {event.name!r} already registered")
        if self._dispatcher is None:
            raise RuntimeError(
                "wallclock_dispatcher is not set; call set_wallclock_dispatcher() "
                "or pass it to the constructor before registering wall-clock events"
            )
        self._scheduler.add_job(
            self._dispatcher,
            event.build_trigger(),
            args=(event.agent_id, event.payload),
            id=event.name,
            replace_existing=True,
        )
        self._wallclock_specs[event.name] = event
        if persist and self._store is not None:
            self._store.save_wallclock(event)

    def unregister_wallclock(self, name: str) -> None:
        if name in self._wallclock_specs:
            try:
                self._scheduler.remove_job(name)
            except Exception:
                # APScheduler lanza JobLookupError si el job ya no existe
                # (p. ej. tras un fallo o un shutdown previo). Lo ignoramos
                # para que unregister sea idempotente.
                pass
            self._wallclock_specs.pop(name, None)
        if self._store is not None:
            self._store.delete_wallclock(name)

    def list_wallclock(self) -> list[WallClockEvent]:
        return list(self._wallclock_specs.values())

    def restore(self) -> list[WallClockEvent]:
        """Recargar los wall-clock events persistidos en `EventStore`.

        Devuelve la lista de eventos registrados. Los iteration events son
        in-process y no se persisten: el código cliente debe re-registrarlos
        en cada arranque (típicamente dentro del módulo de inicialización
        del agente).
        """
        if self._store is None:
            return []
        restored: list[WallClockEvent] = []
        for ev in self._store.list_wallclock():
            if ev.name in self._wallclock_specs:
                continue
            self.register_wallclock(ev, persist=False)
            restored.append(ev)
        return restored

    # ---- Lifecycle ----

    def start(self) -> None:
        if not self._scheduler.running:
            self._scheduler.start()

    def shutdown(self, *, wait: bool = False) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=wait)


def default_wallclock_dispatcher(
    agent: Any,
    *,
    payload_to_message: Callable[[str], AnyMessage] | None = None,
) -> WallClockDispatcher:
    """Construir un dispatcher que invoca `agent` con el payload como SystemMessage.

    `agent` es un `CompiledStateGraph` (lo que devuelve `build_agent`).
    `payload_to_message` permite cambiar el wrapping (por ejemplo, para
    enviar el evento como `HumanMessage` y que entre en Recall).
    """
    from langchain_core.messages import SystemMessage

    if payload_to_message is None:
        def payload_to_message(p: str) -> AnyMessage:  # type: ignore[misc]
            return SystemMessage(content=p)

    def _dispatch(agent_id: str, payload: str) -> None:
        msg = payload_to_message(payload)
        agent.invoke(
            {"messages": [msg]},
            config={"configurable": {"thread_id": agent_id}},
        )

    return _dispatch
