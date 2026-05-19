"""Configuración y utilidades para `request_heartbeat` + function chaining.

Decisiones (cf. `posts/papers/memGPT-plan.md` §6):

- **Dos modos coexistiendo**: ``NATIVE`` (LLMs modernos: el reasoning
  multi-step decide implícitamente cuando seguir o ceder) y ``LEGACY``
  (LLMs pequeños: necesitan flag explícito ``request_heartbeat=True`` o
  estar en la lista ``auto_continue_tools``).
- **Red de seguridad común a los dos modos**: contador de heartbeats,
  timeout por turno, detección de loops idénticos. La red previene loops
  triviales y costes descontrolados independientemente de si el LLM se
  porta bien.
- **`auto_continue_tools` para puentear el desajuste de schema en LEGACY**:
  envolver tools con `InjectedState` (los `core_memory_*`) para añadirles
  un parámetro `request_heartbeat` rompe la inyección de estado de
  LangChain. En lugar de eso, mantenemos el schema intacto y declaramos
  por configuración qué tools señalan continuación implícita.
  Default = los 4 `core_memory_*` y los 3 de Recall/Archival, que son
  pasos preparatorios cuyo único propósito es que el agente razone
  después con la información actualizada.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, PositiveInt, field_validator

DEFAULT_AUTO_CONTINUE_TOOLS: frozenset[str] = frozenset(
    {
        # Core memory edits — siempre quieres razonar después de actualizar.
        "core_memory_append",
        "core_memory_replace",
        "core_memory_create_block",
        "core_memory_delete_block",
        # Recall + Archival lookups — sirven para alimentar el siguiente paso.
        "conversation_search",
        "archival_memory_insert",
        "archival_memory_search",
    }
)


class HeartbeatMode(str, Enum):
    """Modo de decisión sobre si encadenar otra inferencia."""

    NATIVE = "native"   # confía en el reasoning multi-step nativo del LLM
    LEGACY = "legacy"   # exige flag explícito o tool en auto_continue_tools


class HeartbeatConfig(BaseModel):
    """Parámetros del heartbeat + red de seguridad por turno."""

    mode: HeartbeatMode = HeartbeatMode.NATIVE

    max_chained_heartbeats: PositiveInt = 50
    """Máximo de iteraciones tools→agent encadenadas dentro del mismo turno."""

    turn_timeout_seconds: PositiveInt = 300
    """Tiempo máximo (s) por turno antes de forzar un yield."""

    loop_detection_threshold: PositiveInt = 3
    """Repeticiones idénticas tolerables. La N-ésima inyecta warning; la (N+1)-ésima fuerza END."""

    auto_continue_tools: frozenset[str] = Field(
        default_factory=lambda: frozenset(DEFAULT_AUTO_CONTINUE_TOOLS)
    )
    """Tools cuya invocación cuenta como heartbeat implícito en modo LEGACY."""

    recent_keys_buffer: PositiveInt = 30
    """Tamaño del buffer FIFO de claves de tool calls usadas para loop detection."""

    @field_validator("auto_continue_tools", mode="before")
    @classmethod
    def _coerce_auto_continue(cls, v: Any) -> frozenset[str]:
        if v is None:
            return frozenset()
        if isinstance(v, frozenset):
            return v
        if isinstance(v, (set, list, tuple)):
            return frozenset(v)
        raise ValueError(
            f"auto_continue_tools must be set-like, got {type(v).__name__}"
        )


def tool_call_key(name: str, args: Mapping[str, Any] | None) -> str:
    """Hash estable nombre + args para loop detection.

    Determinista (sort_keys + default=str) para que dos llamadas con los
    mismos kwargs en orden distinto generen la misma clave. Caer a
    ``str(args)`` si la serialización JSON falla — peor que JSON pero
    siempre se obtiene una clave.
    """
    args = args or {}
    try:
        serialized = json.dumps(args, sort_keys=True, default=str)
    except (TypeError, ValueError):
        serialized = str(args)
    return f"{name}::{serialized}"


def extract_tool_call_keys(tool_calls: Iterable[Mapping[str, Any]]) -> list[str]:
    """Build keys for every tool call in a single AIMessage."""
    return [tool_call_key(tc.get("name", ""), tc.get("args")) for tc in tool_calls]


def loop_repetition_count(buffer: list[str], key: str) -> int:
    """Count occurrences of `key` in `buffer` (post-update)."""
    return buffer.count(key)


LOOP_DETECTION_WARNING_TEMPLATE = (
    "Loop detected: tool call {tool_name} has been issued {count} times in "
    "a row with the same arguments without observable progress. Either "
    "change strategy (different tool, different args, additional reasoning) "
    "or respond to the user. The next identical call will end this turn."
)
