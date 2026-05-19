"""Queue Manager: contabilidad de tokens y selección de mensajes a expulsar.

Decisiones (cf. `posts/papers/memGPT-plan.md` §4):

- **Umbrales configurables** vía `QueueManagerConfig` (70 / 100 / 50 % por
  defecto). El paper los menciona con "e.g." — modelarlos como config evita
  hardcodearlos.
- **Bloques atómicos**: agrupamos `(AIMessage con tool_calls + sus
  ToolMessages)` como una unidad indivisible. Al expulsar nunca rompemos un
  par tool_call ↔ tool_message (workaround de los bugs #111 y #126 de
  `langmem.SummarizationNode`, descritos en el plan; aquí los evitamos al
  hacer el summarizer custom — Plan B).
- **Conteo**: usamos `count_tokens` (que envuelve `litellm.token_counter`)
  sobre system prompt + Core Memory + recursive_summary + texto de cada
  mensaje. No es exacto al 100 % (no contabiliza la serialización JSON de
  los tool calls del provider) pero está dentro del orden correcto y es
  determinista para los tests.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from pydantic import BaseModel, Field, PositiveInt, field_validator, model_validator

from .tokens import count_tokens

if TYPE_CHECKING:
    from .core_memory import CoreMemory


class QueueManagerConfig(BaseModel):
    """Umbrales del Queue Manager. Todos en (0, 1] excepto el tamaño absoluto.

    - `warning_threshold`: dispara la Memory Pressure Alert (default 70 %).
    - `flush_threshold`: dispara el flush con resumen recursivo (default 100 %).
    - `flush_eviction_ratio`: fracción de la ventana a expulsar al hacer flush
      (default 50 %).
    - `context_window_tokens`: tamaño total de la ventana del LLM principal.
    """

    warning_threshold: float = 0.70
    flush_threshold: float = 1.00
    flush_eviction_ratio: float = 0.50
    context_window_tokens: PositiveInt = 200_000

    @field_validator("warning_threshold", "flush_threshold", "flush_eviction_ratio")
    @classmethod
    def _ratio_in_unit_interval(cls, v: float) -> float:
        if not 0 < v <= 1:
            raise ValueError("threshold must be in (0, 1]")
        return v

    @model_validator(mode="after")
    def _validate_order(self) -> "QueueManagerConfig":
        if self.warning_threshold >= self.flush_threshold:
            raise ValueError("warning_threshold must be < flush_threshold")
        return self

    @classmethod
    def from_settings(cls) -> "QueueManagerConfig":
        """Lee overrides desde `Settings` (`.env`) y cae a los defaults.

        Pensado para que scripts y app principal compartan una sola fuente
        de verdad sin duplicar el cableado.
        """
        from .config import get_settings

        s = get_settings()
        overrides: dict[str, Any] = {}
        if s.context_window_tokens is not None:
            overrides["context_window_tokens"] = s.context_window_tokens
        if s.warning_threshold is not None:
            overrides["warning_threshold"] = s.warning_threshold
        if s.flush_threshold is not None:
            overrides["flush_threshold"] = s.flush_threshold
        if s.flush_eviction_ratio is not None:
            overrides["flush_eviction_ratio"] = s.flush_eviction_ratio
        return cls(**overrides)

    @property
    def warning_tokens(self) -> int:
        return int(self.warning_threshold * self.context_window_tokens)

    @property
    def flush_tokens(self) -> int:
        return int(self.flush_threshold * self.context_window_tokens)

    @property
    def target_eviction_tokens(self) -> int:
        return int(self.flush_eviction_ratio * self.context_window_tokens)


def _message_text(m: BaseMessage) -> str:
    """Best-effort string view of a message for token counting."""
    content = m.content
    if isinstance(content, str):
        text = content
    else:
        text = str(content)
    if isinstance(m, AIMessage) and m.tool_calls:
        for tc in m.tool_calls:
            text += "\n" + str(tc.get("name", "")) + str(tc.get("args", ""))
    return text


def count_messages_tokens(messages: Iterable[BaseMessage], model: str | None = None) -> int:
    return count_tokens("\n".join(_message_text(m) for m in messages), model=model)


def count_state_tokens(
    *,
    system_prompt: str,
    core_memory: "CoreMemory | None",
    recursive_summary: str | None,
    messages: Sequence[BaseMessage],
    model: str | None = None,
) -> int:
    """Count tokens of everything we'd send to the LLM in `agent_node`."""
    parts: list[str] = [system_prompt]
    if core_memory is not None:
        cm_text = core_memory.to_prompt_text()
        if cm_text:
            parts.append(cm_text)
    if recursive_summary:
        parts.append(recursive_summary)
    for m in messages:
        parts.append(_message_text(m))
    return count_tokens("\n".join(parts), model=model)


def group_into_atomic_blocks(messages: Sequence[BaseMessage]) -> list[list[BaseMessage]]:
    """Group messages so that no `(AIMessage with tool_calls, ToolMessage)` pair is split.

    Rules:
    - An `AIMessage` with `tool_calls` opens a block; the matching
      `ToolMessage`s (by `tool_call_id`) are appended to it. The block closes
      when all expected tool_call_ids are seen.
    - Any other message (`HumanMessage`, plain `AIMessage`, `SystemMessage`,
      orphan `ToolMessage`) is its own block.
    """
    blocks: list[list[BaseMessage]] = []
    current: list[BaseMessage] = []
    pending: set[str] = set()

    def flush_current() -> None:
        nonlocal current, pending
        if current:
            blocks.append(current)
        current = []
        pending = set()

    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            flush_current()
            current = [m]
            pending = {tc["id"] for tc in m.tool_calls if tc.get("id")}
            if not pending:
                flush_current()
        elif isinstance(m, ToolMessage) and pending and m.tool_call_id in pending:
            current.append(m)
            pending.discard(m.tool_call_id)
            if not pending:
                flush_current()
        else:
            flush_current()
            blocks.append([m])
    flush_current()
    return blocks


def select_blocks_to_evict(
    messages: Sequence[BaseMessage],
    *,
    target_tokens: int,
    model: str | None = None,
) -> tuple[list[BaseMessage], list[BaseMessage]]:
    """Pick the oldest atomic blocks whose cumulative tokens reach `target_tokens`.

    Returns `(evicted, kept)`. Eviction always operates at block granularity:
    if including the next block would exceed `target_tokens` we still include
    it (we'd rather over-evict by one block than split a tool-call pair).
    """
    if target_tokens <= 0 or not messages:
        return [], list(messages)

    blocks = group_into_atomic_blocks(messages)
    evicted: list[BaseMessage] = []
    accumulated = 0
    cut_index = 0
    for i, block in enumerate(blocks):
        block_tokens = count_messages_tokens(block, model=model)
        evicted.extend(block)
        accumulated += block_tokens
        cut_index = i + 1
        if accumulated >= target_tokens:
            break

    kept_blocks = blocks[cut_index:]
    kept = [m for blk in kept_blocks for m in blk]
    return evicted, kept


MEMORY_PRESSURE_ALERT_TEXT = (
    "Memory Pressure Alert: the working context is approaching its limit. "
    "Consolidate any durable facts from the recent conversation into Core "
    "Memory using the core_memory_* tools so they survive the upcoming "
    "flush. Older messages will be summarised and evicted shortly."
)
