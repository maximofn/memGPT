import json
from functools import lru_cache
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    primary_llm_model: str = Field(default="claude-sonnet-4-6")
    summarizer_llm_model: str = Field(default="claude-haiku-4-5")

    # Overrides para apuntar el LLM primario / summarizer a un endpoint
    # OpenAI-compatible distinto del global (p. ej. un llama-server local).
    # Si se quedan a None se usan las credenciales globales (OPENAI_API_KEY /
    # ANTHROPIC_API_KEY) — necesario para que Graphiti, que reusa el SDK de
    # OpenAI a través de las env vars globales, no se redirija sin querer al
    # endpoint local cuando solo el agente debe usarlo.
    primary_llm_base_url: str | None = None
    primary_llm_api_key: str | None = None
    summarizer_llm_base_url: str | None = None
    summarizer_llm_api_key: str | None = None

    # `extra_body` crudo (JSON) que se reenvía al endpoint OpenAI-compatible.
    # Pensado para flags propietarios del provider que no están en el spec de
    # OpenAI, p. ej. desactivar el thinking de DeepSeek v4:
    #   PRIMARY_LLM_EXTRA_BODY='{"thinking": {"type": "disabled"}}'
    # (DeepSeek v4 razona por defecto y devuelve `reasoning_content`, que
    # langchain-openai no reenvía en el tool-chaining → 400 en el 2º paso del
    # loop). Solo se aplica a modelos `openai:` para no contaminar a Anthropic
    # ni a OpenAI canónico. Ver `extra_body_dict()`.
    primary_llm_extra_body: str | None = None
    summarizer_llm_extra_body: str | None = None

    # Queue Manager: si no se setean, se usan los defaults del paper
    # (200k window, 70 % warning, 100 % flush, 50 % evicción). Útil para
    # ajustar a la ventana real del LLM primario (p. ej. 262144 para
    # Qwen3.6 con -c 262144).
    context_window_tokens: int | None = None
    warning_threshold: float | None = None
    flush_threshold: float | None = None
    flush_eviction_ratio: float | None = None

    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    postgres_dsn: str = "postgresql://memgpt:memgpt@localhost:5433/memgpt"

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "memgptmemgpt"

    def extra_body_dict(self, role: str) -> dict[str, Any]:
        """Parsea `{role}_llm_extra_body` (JSON) a dict; ``{}`` si no está.

        `role` es ``"primary"`` o ``"summarizer"``. Falla ruidosamente si el
        JSON es inválido — un extra_body mal escrito debe verse, no tragarse.
        """
        raw = getattr(self, f"{role}_llm_extra_body", None)
        if not raw:
            return {}
        return json.loads(raw)


@lru_cache
def get_settings() -> Settings:
    return Settings()
