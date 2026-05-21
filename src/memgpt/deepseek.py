"""Integración de DeepSeek V4 con thinking mode dentro del loop del agente.

DeepSeek V4 (deepseek-v4-flash / -pro) razona por defecto y devuelve el
`reasoning_content` (cadena de pensamiento) separado del `content`. A partir
de V4, cuando hay **tool calls**, la API **exige** que ese `reasoning_content`
de cada turno asistente previo se **reenvíe** en las llamadas siguientes; si
no, responde ``400 - The reasoning_content in the thinking mode must be
passed back to the API`` en el segundo paso del tool-chaining.

El ecosistema aún no cubre esto:
- ``langchain-openai`` ni siquiera captura el `reasoning_content` (lo tira a la
  entrada).
- ``langchain-deepseek`` (``ChatDeepSeek``) sí lo captura en
  ``additional_kwargs["reasoning_content"]``, pero **no lo reinyecta** en el
  payload de salida → el 400 persiste.

``ThinkingChatDeepSeek`` cierra ese hueco: extiende el override que
``ChatDeepSeek`` ya hace de ``_get_request_payload`` y re-mapea, en orden, el
`reasoning_content` capturado de cada ``AIMessage`` sobre los mensajes
``role=assistant`` del payload. Con thinking desactivado no hay
`reasoning_content` que capturar y la reinyección es un no-op, así que la
clase es segura tanto en modo razonamiento como sin él.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_deepseek import ChatDeepSeek


class ThinkingChatDeepSeek(ChatDeepSeek):
    """``ChatDeepSeek`` que reenvía el `reasoning_content` en tool-chaining."""

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        # `reasoning_content` de cada AIMessage previo, en orden de aparición.
        msgs = self._convert_input(input_).to_messages()
        reasonings = [
            m.additional_kwargs.get("reasoning_content")
            for m in msgs
            if isinstance(m, AIMessage)
        ]
        # Los mensajes role=assistant del payload corresponden 1:1, en orden,
        # con esos AIMessages. Re-pegamos el reasoning_content que la API exige.
        idx = 0
        for message in payload["messages"]:
            if message["role"] != "assistant":
                continue
            rc = reasonings[idx] if idx < len(reasonings) else None
            idx += 1
            if rc:
                message["reasoning_content"] = rc
        return payload
