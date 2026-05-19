from .config import get_settings


def count_tokens(text: str, model: str | None = None) -> int:
    """Count tokens of `text` using litellm's tokenizer for the given model.

    Falls back to a coarse word-count estimate if litellm cannot resolve a
    tokenizer for the model — keeps validation usable on exotic model ids.
    """
    from litellm import token_counter

    target = model or get_settings().primary_llm_model
    try:
        return int(token_counter(model=target, text=text))
    except Exception:
        return max(1, int(len(text.split()) * 1.3))
