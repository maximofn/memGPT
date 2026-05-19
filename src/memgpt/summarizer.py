"""Summarizer recursivo (Plan B): función propia, sin `langmem`.

Por qué Plan B y no `langmem.SummarizationNode`: los bugs activos #118
(no fusiona el resumen previo), #111 (parte HumanMessage con tool calls)
y #126 (parallel tool calls) tocan exactamente nuestro caso de uso. Los
workarounds suman más código frágil que escribir el summarizer desde cero,
y además queremos pasar el Working Context al prompt como anti-hint —
mejora explícita sobre el paper.

API:
    new_summary = regenerate_recursive_summary(
        old_summary, evicted_messages, core_memory_text,
        llm_callable=call_summarizer,  # inyectable en tests
    )
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from .llm import call_summarizer

LLMMessages = list[dict[str, Any]]
LLMCallable = Callable[[LLMMessages], str]


SUMMARIZER_SYSTEM_PROMPT = (
    "You are the summariser of a memory-augmented agent. Your output will "
    "REPLACE the evicted portion of the conversation in the agent's working "
    "context, so it must preserve every detail that may be relevant later: "
    "facts, decisions, intents, open questions, named entities, numbers, "
    "dates, and the ordering of events when relevant.\n"
    "\n"
    "Output ONLY the new summary text. No preamble, no headings, no markdown."
)


def _format_message(m: BaseMessage) -> str:
    role = type(m).__name__.replace("Message", "").lower() or "message"
    if isinstance(m, AIMessage) and m.tool_calls:
        calls = "; ".join(
            f"{tc.get('name')}({tc.get('args')})" for tc in m.tool_calls
        )
        body = (m.content or "").strip()
        suffix = f" [tool_calls: {calls}]"
        return f"[{role}] {body}{suffix}".rstrip()
    if isinstance(m, ToolMessage):
        return f"[tool_result name={m.name or '?'}] {m.content}"
    if isinstance(m, HumanMessage):
        return f"[user] {m.content}"
    if isinstance(m, AIMessage):
        return f"[assistant] {m.content}"
    return f"[{role}] {m.content}"


def _format_evicted(messages: Sequence[BaseMessage]) -> str:
    return "\n".join(_format_message(m) for m in messages) or "(no messages)"


def build_summariser_prompt(
    *,
    old_summary: str | None,
    evicted_messages: Sequence[BaseMessage],
    core_memory_text: str,
) -> LLMMessages:
    """Build the prompt sent to the summariser LLM.

    Includes:
    - The previous recursive summary (if any) — the new summary MUST fold it in.
    - The Working Context (Core Memory) as anti-hint — facts already there
      should NOT be duplicated in the summary.
    - The evicted messages, one per line, role-tagged.
    """
    user_parts: list[str] = []

    if old_summary:
        user_parts.append(
            "Previous recursive summary (FOLD INTO THE NEW ONE — do not lose any "
            "detail from here):\n"
            f"{old_summary.strip()}"
        )
    else:
        user_parts.append(
            "Previous recursive summary: (none — this is the first flush)"
        )

    if core_memory_text:
        user_parts.append(
            "Working Context (already saved as durable knowledge — DO NOT "
            "duplicate these facts in the summary):\n"
            f"{core_memory_text.strip()}"
        )

    user_parts.append(
        "Messages being evicted from the working context (oldest first):\n"
        f"{_format_evicted(evicted_messages)}"
    )

    user_parts.append(
        "Produce the updated recursive summary now. Be concise but complete. "
        "Output ONLY the summary text."
    )

    return [
        {"role": "system", "content": SUMMARIZER_SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def regenerate_recursive_summary(
    *,
    old_summary: str | None,
    evicted_messages: Sequence[BaseMessage],
    core_memory_text: str = "",
    llm_callable: LLMCallable | None = None,
) -> str:
    """Generate the new recursive summary after a flush.

    `llm_callable` defaults to `litellm`-backed `call_summarizer`. Tests
    inject a stub.
    """
    if not evicted_messages:
        return old_summary or ""

    prompt = build_summariser_prompt(
        old_summary=old_summary,
        evicted_messages=evicted_messages,
        core_memory_text=core_memory_text,
    )
    fn = llm_callable if llm_callable is not None else call_summarizer
    summary = fn(prompt).strip()
    return summary or (old_summary or "")
