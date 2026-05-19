from datetime import datetime, timezone
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from .core_memory import MemoryBlock
from .state import MemGPTState


@tool
def get_current_time() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _err(tool_call_id: str, msg: str) -> Command:
    return Command(
        update={
            "messages": [
                ToolMessage(content=f"ERROR: {msg}", tool_call_id=tool_call_id, status="error")
            ]
        }
    )


def _ok(tool_call_id: str, msg: str, *, core_memory) -> Command:
    return Command(
        update={
            "core_memory": core_memory,
            "messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)],
        }
    )


@tool
def core_memory_append(
    label: str,
    content: str,
    state: Annotated[MemGPTState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Append `content` to an existing core-memory block.

    A newline separator is inserted automatically when the block is non-empty.
    Fails if the block does not exist or if the new value would exceed the
    block's token limit.
    """
    try:
        new_core = state.core_memory.with_appended(label, content)
    except (KeyError, ValueError) as exc:
        return _err(tool_call_id, str(exc))
    return _ok(tool_call_id, f"Appended to '{label}'.", core_memory=new_core)


@tool
def core_memory_replace(
    label: str,
    old: str,
    new: str,
    state: Annotated[MemGPTState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Replace the first occurrence of `old` with `new` in a core-memory block.

    Fails if the block does not exist, the substring `old` is not found, or
    the resulting value would exceed the block's token limit.
    """
    try:
        new_core = state.core_memory.with_replaced(label, old, new)
    except (KeyError, ValueError) as exc:
        return _err(tool_call_id, str(exc))
    return _ok(tool_call_id, f"Replaced text in '{label}'.", core_memory=new_core)


@tool
def core_memory_create_block(
    label: str,
    initial_content: str,
    limit: int,
    state: Annotated[MemGPTState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Create a new core-memory block.

    The label must be snake_case and ≤32 chars. Fails if a block with the
    same label exists, if `max_blocks` would be exceeded, or if the total
    token budget would be exceeded.
    """
    try:
        block = MemoryBlock(label=label, value=initial_content, limit=limit)
        new_core = state.core_memory.with_block(block)
    except (KeyError, ValueError) as exc:
        return _err(tool_call_id, str(exc))
    return _ok(tool_call_id, f"Created block '{label}'.", core_memory=new_core)


@tool
def core_memory_delete_block(
    label: str,
    state: Annotated[MemGPTState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Delete a core-memory block entirely (different from emptying it)."""
    try:
        new_core = state.core_memory.without_block(label)
    except KeyError as exc:
        return _err(tool_call_id, f"block {exc} not found")
    return _ok(tool_call_id, f"Deleted block '{label}'.", core_memory=new_core)


CORE_MEMORY_TOOLS = [
    core_memory_append,
    core_memory_replace,
    core_memory_create_block,
    core_memory_delete_block,
]
