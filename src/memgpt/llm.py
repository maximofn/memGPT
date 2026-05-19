from typing import Any

from litellm import acompletion, completion

from .config import get_settings

Message = dict[str, Any]


def to_litellm_model(model_id: str) -> str:
    # litellm usa "provider/model"; langchain usa "provider:model". Aceptamos
    # ambos en .env para no tener dos variables paralelas.
    if "/" in model_id or ":" not in model_id:
        return model_id
    provider, _, name = model_id.partition(":")
    return f"{provider}/{name}"


def _endpoint_kwargs(role: str) -> dict[str, Any]:
    """`api_base` + `api_key` overrides para litellm.

    Solo se añaden si el modelo es openai-compatible y hay overrides en
    Settings. Evita pisar los defaults de Anthropic/OpenAI canónico y
    permite apuntar a un llama-server local sin tocar OPENAI_API_KEY.
    """
    settings = get_settings()
    model = getattr(settings, f"{role}_llm_model")
    if not (model.startswith("openai:") or model.startswith("openai/")):
        return {}
    out: dict[str, Any] = {}
    base = getattr(settings, f"{role}_llm_base_url", None)
    key = getattr(settings, f"{role}_llm_api_key", None)
    if base:
        out["api_base"] = base
    if key:
        out["api_key"] = key
    return out


async def acall_primary(messages: list[Message], **kwargs: Any) -> str:
    settings = get_settings()
    response = await acompletion(
        model=to_litellm_model(settings.primary_llm_model),
        messages=messages,
        **_endpoint_kwargs("primary"),
        **kwargs,
    )
    return response.choices[0].message.content or ""


async def acall_summarizer(messages: list[Message], **kwargs: Any) -> str:
    settings = get_settings()
    response = await acompletion(
        model=to_litellm_model(settings.summarizer_llm_model),
        messages=messages,
        **_endpoint_kwargs("summarizer"),
        **kwargs,
    )
    return response.choices[0].message.content or ""


def call_primary(messages: list[Message], **kwargs: Any) -> str:
    settings = get_settings()
    response = completion(
        model=to_litellm_model(settings.primary_llm_model),
        messages=messages,
        **_endpoint_kwargs("primary"),
        **kwargs,
    )
    return response.choices[0].message.content or ""


def call_summarizer(messages: list[Message], **kwargs: Any) -> str:
    settings = get_settings()
    response = completion(
        model=to_litellm_model(settings.summarizer_llm_model),
        messages=messages,
        **_endpoint_kwargs("summarizer"),
        **kwargs,
    )
    return response.choices[0].message.content or ""
