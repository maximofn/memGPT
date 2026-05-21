"""Tests de regresión de ThinkingChatDeepSeek.

La clase reinyecta el `reasoning_content` que DeepSeek V4 exige reenviar en el
tool-chaining (ver `src/memgpt/deepseek.py`). Se apoya en dos internals de
`langchain-deepseek` que no son API pública: que `ChatDeepSeek` capture el
`reasoning_content` en ``additional_kwargs`` y que ``_get_request_payload``
devuelva ``payload["messages"]`` con ``role == "assistant"``. Estos tests son
el canario: si una futura versión cambia ese contrato, fallan al instante
—sin gastar una llamada a la API— en vez de degradar a un 400 en producción.

No tocan la red: construimos el cliente con una key dummy y llamamos a
``_get_request_payload`` directamente (no ``.invoke()``).
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from memgpt.deepseek import ThinkingChatDeepSeek


def _client() -> ThinkingChatDeepSeek:
    # api_key dummy: la construcción no dispara ninguna llamada.
    return ThinkingChatDeepSeek(model="deepseek-v4-flash", api_key="test-key")


def _ai_tool_call(reasoning: str | None) -> AIMessage:
    extra = {"reasoning_content": reasoning} if reasoning is not None else {}
    return AIMessage(
        content="",
        tool_calls=[
            {"id": "call_1", "name": "get_weather", "args": {"city": "Madrid"}, "type": "tool_call"}
        ],
        additional_kwargs=extra,
    )


def _assistant_messages(payload: dict) -> list[dict]:
    return [m for m in payload["messages"] if m["role"] == "assistant"]


def test_reinjects_reasoning_content_into_assistant_message():
    """El reasoning_content del AIMessage acaba en el mensaje assistant."""
    llm = _client()
    messages = [
        HumanMessage(content="¿Tiempo en Madrid?"),
        _ai_tool_call("Necesito llamar a la tool del tiempo."),
        ToolMessage(content="22 grados y sol", tool_call_id="call_1"),
    ]

    payload = llm._get_request_payload(messages)
    assistants = _assistant_messages(payload)

    assert len(assistants) == 1
    assert assistants[0]["reasoning_content"] == "Necesito llamar a la tool del tiempo."


def test_no_reasoning_is_noop():
    """Sin reasoning_content (thinking off), no se añade la clave."""
    llm = _client()
    messages = [
        HumanMessage(content="¿Tiempo en Madrid?"),
        _ai_tool_call(None),
        ToolMessage(content="22 grados y sol", tool_call_id="call_1"),
    ]

    payload = llm._get_request_payload(messages)
    assistants = _assistant_messages(payload)

    assert len(assistants) == 1
    assert "reasoning_content" not in assistants[0]


def test_multiple_assistant_turns_map_in_order():
    """Cada reasoning_content va a su assistant correcto, sin cruzarse."""
    llm = _client()
    first = AIMessage(
        content="",
        tool_calls=[
            {"id": "call_1", "name": "search", "args": {"q": "a"}, "type": "tool_call"}
        ],
        additional_kwargs={"reasoning_content": "razonamiento-1"},
    )
    second = AIMessage(
        content="",
        tool_calls=[
            {"id": "call_2", "name": "search", "args": {"q": "b"}, "type": "tool_call"}
        ],
        additional_kwargs={"reasoning_content": "razonamiento-2"},
    )
    messages = [
        HumanMessage(content="busca"),
        first,
        ToolMessage(content="r1", tool_call_id="call_1"),
        second,
        ToolMessage(content="r2", tool_call_id="call_2"),
    ]

    payload = llm._get_request_payload(messages)
    assistants = _assistant_messages(payload)

    assert len(assistants) == 2
    assert assistants[0]["reasoning_content"] == "razonamiento-1"
    assert assistants[1]["reasoning_content"] == "razonamiento-2"
