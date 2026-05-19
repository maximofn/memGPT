"""Core Memory: bloques editables tipo Letta (`assistant`, `human`, custom).

Decisiones de diseño (cf. `posts/papers/memGPT-plan.md` §3):

- `CoreMemory` es un contenedor de `dict[str, MemoryBlock]`, no una clase con
  campos fijos: permite añadir bloques arbitrarios en runtime sin tocar el
  schema, y persiste automáticamente vía Pydantic + checkpointer.
- Métodos mutativos son **inmutables**: devuelven una nueva `CoreMemory`,
  apta para `Command(update={"core_memory": ...})`.
- Validaciones: formato de label snake_case, presupuesto de tokens por bloque,
  presupuesto total, número máximo de bloques, idempotencia en creación.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, PositiveInt, field_validator, model_validator

from .tokens import count_tokens

LABEL_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")


class MemoryBlock(BaseModel):
    label: str
    value: str = ""
    limit: PositiveInt = 2000

    @field_validator("label")
    @classmethod
    def _validate_label(cls, v: str) -> str:
        if not LABEL_RE.match(v):
            raise ValueError(
                f"label {v!r} must match {LABEL_RE.pattern} "
                "(snake_case, ≤32 chars, starts with letter)"
            )
        return v

    @model_validator(mode="after")
    def _validate_token_budget(self) -> "MemoryBlock":
        used = count_tokens(self.value)
        if used > self.limit:
            raise ValueError(
                f"block {self.label!r} value uses {used} tokens > limit {self.limit}"
            )
        return self


class CoreMemory(BaseModel):
    blocks: dict[str, MemoryBlock] = Field(default_factory=dict)
    max_blocks: PositiveInt = 10
    total_token_budget: PositiveInt = 8000

    @model_validator(mode="after")
    def _validate(self) -> "CoreMemory":
        for key, block in self.blocks.items():
            if key != block.label:
                raise ValueError(
                    f"dict key {key!r} does not match block.label {block.label!r}"
                )
        if len(self.blocks) > self.max_blocks:
            raise ValueError(
                f"too many blocks: {len(self.blocks)} > max_blocks={self.max_blocks}"
            )
        total_limit = sum(b.limit for b in self.blocks.values())
        if total_limit > self.total_token_budget:
            raise ValueError(
                f"sum of block limits {total_limit} > total_token_budget "
                f"{self.total_token_budget}"
            )
        return self

    def has(self, label: str) -> bool:
        return label in self.blocks

    def get(self, label: str) -> MemoryBlock:
        if label not in self.blocks:
            raise KeyError(f"core memory has no block {label!r}")
        return self.blocks[label]

    def _replace(self, new_blocks: dict[str, MemoryBlock]) -> "CoreMemory":
        return CoreMemory(
            blocks=new_blocks,
            max_blocks=self.max_blocks,
            total_token_budget=self.total_token_budget,
        )

    def with_block(self, block: MemoryBlock) -> "CoreMemory":
        if block.label in self.blocks:
            raise ValueError(f"block {block.label!r} already exists")
        if len(self.blocks) + 1 > self.max_blocks:
            raise ValueError(
                f"adding block would exceed max_blocks={self.max_blocks}"
            )
        new_blocks = {**self.blocks, block.label: block}
        return self._replace(new_blocks)

    def without_block(self, label: str) -> "CoreMemory":
        if label not in self.blocks:
            raise KeyError(label)
        new_blocks = {k: v for k, v in self.blocks.items() if k != label}
        return self._replace(new_blocks)

    def with_appended(self, label: str, content: str) -> "CoreMemory":
        block = self.get(label)
        sep = "\n" if block.value else ""
        new_block = MemoryBlock(
            label=block.label,
            value=block.value + sep + content,
            limit=block.limit,
        )
        return self._replace({**self.blocks, label: new_block})

    def with_replaced(self, label: str, old: str, new: str) -> "CoreMemory":
        block = self.get(label)
        if old not in block.value:
            raise ValueError(f"text {old!r} not found in block {label!r}")
        new_block = MemoryBlock(
            label=block.label,
            value=block.value.replace(old, new, 1),
            limit=block.limit,
        )
        return self._replace({**self.blocks, label: new_block})

    def to_prompt_text(self) -> str:
        if not self.blocks:
            return ""
        parts: list[str] = ["=== Core Memory ==="]
        for label, block in self.blocks.items():
            parts.append("")
            parts.append(f"[{label}] (limit: {block.limit} tokens)")
            parts.append(block.value if block.value else "(empty)")
        return "\n".join(parts)

    def total_tokens_used(self) -> int:
        return sum(count_tokens(b.value) for b in self.blocks.values())


def default_core_memory(
    assistant: str = "I am MemGPT, a helpful assistant with persistent memory.",
    human: str = "",
    *,
    extra_blocks: dict[str, MemoryBlock] | None = None,
    block_limit: int = 2000,
    max_blocks: int = 10,
    total_token_budget: int = 8000,
) -> CoreMemory:
    """Build the default Core Memory with `assistant` and `human` blocks."""
    blocks: dict[str, MemoryBlock] = {
        "assistant": MemoryBlock(label="assistant", value=assistant, limit=block_limit),
        "human": MemoryBlock(label="human", value=human, limit=block_limit),
    }
    if extra_blocks:
        for label, block in extra_blocks.items():
            if label != block.label:
                raise ValueError(f"extra_blocks key {label!r} != block.label {block.label!r}")
            blocks[label] = block
    return CoreMemory(
        blocks=blocks,
        max_blocks=max_blocks,
        total_token_budget=total_token_budget,
    )
