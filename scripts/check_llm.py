"""Smoke test: invoca el LLM principal y el summarizer una vez."""

from __future__ import annotations

import sys

from dotenv import load_dotenv

from memgpt.config import get_settings
from memgpt.llm import call_primary, call_summarizer


def main() -> int:
    load_dotenv()
    settings = get_settings()

    prompt = [{"role": "user", "content": "Reply with the single word: OK"}]

    print(f"[primary] model={settings.primary_llm_model}")
    try:
        primary_out = call_primary(prompt, max_tokens=10)
        print(f"[primary] response={primary_out!r}")
    except Exception as exc:
        print(f"[primary] FAILED: {exc}")
        return 1

    print(f"[summarizer] model={settings.summarizer_llm_model}")
    try:
        summarizer_out = call_summarizer(prompt, max_tokens=10)
        print(f"[summarizer] response={summarizer_out!r}")
    except Exception as exc:
        print(f"[summarizer] FAILED: {exc}")
        return 1

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
