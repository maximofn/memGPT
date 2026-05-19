"""Nested Key-Value retrieval benchmark (Fase 8, §3.2.2 del paper).

El paper original no liberó la generación del dataset junto con el código de
evaluación, así que reproducimos la tarea siguiendo la descripción de la
sección 3.2.2 y el apéndice 6.1.6:

- Por configuración: 140 pares clave-valor con UUIDs de 128 bits.
- Cada configuración contiene una *cadena guía* de 5 saltos (k0 → k1 → … → k4
  → terminal_no_clave) y 135 pares "distractor" cuyas claves no participan en
  la cadena.
- Por configuración se generan 5 queries — una por nivel de anidamiento
  (0..4). El nivel ``L`` parte de ``k_{4-L}`` y exige ``L`` resoluciones
  encadenadas hasta el terminal.
- 30 configuraciones × 5 niveles = 150 queries.

El asistente del agente y el prompt de query reproducen literalmente el
apéndice 6.1.6 (en inglés, manteniendo el "DO NOT STOP SEARCHING" como ancla
para el LLM).
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from random import Random
from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph

from ..agent import build_agent
from ..memory_store import InMemoryStore, MemoryStore

CHAIN_LENGTH = 5  # niveles 0..4 ⇒ 5 nodos en la cadena.
PAIRS_PER_CONFIG = 140
LEVELS = (0, 1, 2, 3, 4)

NESTED_KV_ASSISTANT = (
    "You are the MemGPT DOC-QA bot. Your job is to answer questions about "
    "documents that are stored in your archival memory. The answer to the "
    "user's question will ALWAYS be in your archival memory, so remember to "
    "keep searching if you don't find the answer.\n"
    "DO NOT STOP SEARCHING UNTIL YOU VERIFY THAT THE VALUE IS NOT A KEY. Do "
    "not stop doing nested lookups until this condition is met."
)

QUERY_TEMPLATE = (
    "Your archival memory contains key-value pairs where each key and each "
    "value is a 128-bit UUID. The pairs are stored as `key=<uuid> value=<uuid>` "
    "entries.\n"
    "Return the value associated with the key `{start_key}`. If the value is "
    "itself a key, follow the nested lookup until you reach a value that is "
    "NOT a key in archival memory.\n"
    "When you have the final value, reply with ONLY the UUID — no extra text."
)

# --- Baseline (LLM sin memoria, 140 pares en el system prompt) ---------------
#
# Reproduce el prompt del apéndice 6.1.6 ("references"): pares como objeto
# JSON dentro del system prompt, sin tools de archival. Sirve de control para
# comparar contra MemGPT y reproducir la curva de la Figura 7 del paper, que
# cae a 0% en niveles 3-4.

BASELINE_SYSTEM_PROMPT_HEADER = (
    "Below is a JSON object that contains key-value pairs, all keys and values "
    "are 128-bit UUIDs, and your task is to return the value associated with "
    "the specified key. If a value is itself a key, return the value of that "
    "key (do a nested lookup). For example, if the value of 'x' is 'y', but "
    "'y' is also a key, return the value of the 'y' key."
)

BASELINE_QUERY_TEMPLATE = (
    "Return the final value for the key `{start_key}`. Follow nested lookups "
    "until the value is not also a key in the JSON above. Reply with ONLY the "
    "final UUID — no extra text."
)

UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)


@dataclass(frozen=True)
class KVPair:
    key: str
    value: str

    def as_archival_text(self) -> str:
        return f"key={self.key} value={self.value}"


@dataclass(frozen=True)
class NestedKVQuery:
    start_key: str
    expected_value: str
    nesting_level: int  # número de resoluciones encadenadas requeridas (0..4).


@dataclass
class NestedKVConfig:
    config_id: int
    pairs: tuple[KVPair, ...]
    queries: tuple[NestedKVQuery, ...]


def _uuid(rng: Random) -> str:
    """UUID v4 derivado del RNG para hacer el dataset determinista."""
    return str(uuid.UUID(int=rng.getrandbits(128), version=4))


def generate_config(
    config_id: int,
    *,
    seed: int,
    n_pairs: int = PAIRS_PER_CONFIG,
    chain_length: int = CHAIN_LENGTH,
) -> NestedKVConfig:
    """Genera una configuración independiente.

    Estructura de los ``n_pairs`` pares:
    - ``chain_length`` pares forman la cadena guía (k0..k_{n-1} → terminal).
    - El resto son distractores: claves frescas con valores que NO aparecen
      como clave en este config.

    Las queries (una por nivel) se construyen al vuelo a partir de la cadena.
    """
    if n_pairs < chain_length:
        raise ValueError("n_pairs must be >= chain_length")

    rng = Random(seed)

    chain_keys = [_uuid(rng) for _ in range(chain_length)]
    terminal = _uuid(rng)
    chain_pairs: list[KVPair] = []
    for i, k in enumerate(chain_keys):
        v = chain_keys[i + 1] if i + 1 < chain_length else terminal
        chain_pairs.append(KVPair(key=k, value=v))

    chain_key_set = set(chain_keys)

    distractor_pairs: list[KVPair] = []
    while len(distractor_pairs) < n_pairs - chain_length:
        k = _uuid(rng)
        if k in chain_key_set:
            continue
        v = _uuid(rng)  # extremadamente improbable colisión con chain_keys.
        if v in chain_key_set:
            continue
        chain_key_set.add(k)
        distractor_pairs.append(KVPair(key=k, value=v))

    pairs = chain_pairs + distractor_pairs
    rng.shuffle(pairs)  # "diferentes configuraciones de ordenación" (§3.2.2).

    queries: list[NestedKVQuery] = []
    for level in LEVELS:
        # nivel L ⇒ partimos de k_{(chain_length - 1) - L} y resolvemos L saltos.
        start_idx = (chain_length - 1) - level
        queries.append(
            NestedKVQuery(
                start_key=chain_keys[start_idx],
                expected_value=terminal,
                nesting_level=level,
            )
        )

    return NestedKVConfig(
        config_id=config_id,
        pairs=tuple(pairs),
        queries=tuple(queries),
    )


def generate_dataset(
    *,
    seed: int = 42,
    n_configs: int = 30,
    n_pairs: int = PAIRS_PER_CONFIG,
    chain_length: int = CHAIN_LENGTH,
) -> list[NestedKVConfig]:
    """Genera ``n_configs`` configuraciones independientes y deterministas."""
    return [
        generate_config(
            i,
            seed=seed + i,
            n_pairs=n_pairs,
            chain_length=chain_length,
        )
        for i in range(n_configs)
    ]


def extract_uuid(text: str) -> str | None:
    """Devuelve el último UUID que aparece en ``text`` o ``None``.

    Tomamos el último porque la respuesta del agente típicamente incluye
    pasos intermedios y el UUID final aparece al cierre.
    """
    matches = UUID_RE.findall(text or "")
    return matches[-1].lower() if matches else None


@dataclass
class QueryResult:
    config_id: int
    nesting_level: int
    start_key: str
    expected: str
    predicted: str | None
    correct: bool
    elapsed_seconds: float
    final_text: str
    chained_heartbeats: int
    archival_search_calls: int


@dataclass
class BenchmarkSummary:
    total: int
    correct: int
    accuracy: float
    accuracy_by_level: dict[int, float]
    mean_elapsed_seconds: float
    results: list[QueryResult] = field(default_factory=list)


def populate_archival(store: MemoryStore, pairs: tuple[KVPair, ...]) -> None:
    """Inserta los 140 pares en Archival Memory tal cual.

    Forzamos el formato ``key=<uuid> value=<uuid>`` para que el ``InMemoryStore``
    (búsqueda por substring) recupere por la UUID exacta. Graphiti usa búsqueda
    semántica/BM25, también compatible.
    """
    for p in pairs:
        store.insert_archival(content=p.as_archival_text())


def _count_archival_calls(messages: list[Any]) -> int:
    return sum(
        1
        for m in messages
        if isinstance(m, AIMessage)
        for tc in (m.tool_calls or [])
        if tc.get("name") == "archival_memory_search"
    )


_RETRY_AFTER_RE = re.compile(r"try again in ([\d.]+)\s*s", re.IGNORECASE)


def _is_rate_limit(exc: BaseException) -> bool:
    """Detección laxa: cubre OpenAI/Anthropic/etc. sin importar sus clases.

    Excluye explícitamente ``insufficient_quota`` y ``billing``: esos son
    errores permanentes (saldo agotado, plan caducado) y reintentar no
    sirve — solo gasta tiempo y dispara backoffs largos. El runner los
    propaga inmediatamente para que el caller los gestione.
    """
    msg = str(exc).lower()
    if "insufficient_quota" in msg or "exceeded your current quota" in msg:
        return False
    return "rate_limit" in msg or "rate limit" in msg or " 429" in msg or "429 " in msg


def _suggested_backoff(exc: BaseException, attempt: int) -> float:
    """Usa el "try again in Xs" del provider si está; si no, exponencial."""
    m = _RETRY_AFTER_RE.search(str(exc))
    if m:
        return float(m.group(1)) + 1.0  # margen de seguridad.
    return min(60.0, 2.0 ** attempt)


def _invoke_with_retry(
    agent: CompiledStateGraph,
    payload: dict,
    config: dict,
    *,
    max_retries: int = 6,
) -> Any:
    """Reintenta ante 429 con el delay sugerido por el provider o exp backoff.

    Cualquier otra excepción se propaga sin tocar.
    """
    for attempt in range(max_retries):
        try:
            return agent.invoke(payload, config=config)
        except Exception as exc:
            if not _is_rate_limit(exc) or attempt == max_retries - 1:
                raise
            delay = _suggested_backoff(exc, attempt)
            print(
                f"[rate-limit] sleeping {delay:.1f}s "
                f"(attempt {attempt + 1}/{max_retries})"
            )
            time.sleep(delay)
    # Inalcanzable: el último intento o devuelve o relanza.
    raise RuntimeError("retry loop exited without returning")


def run_query(
    agent: CompiledStateGraph,
    query: NestedKVQuery,
    *,
    config_id: int,
    thread_id: str,
    query_template: str = QUERY_TEMPLATE,
) -> QueryResult:
    start = time.perf_counter()
    out = _invoke_with_retry(
        agent,
        {"messages": [HumanMessage(content=query_template.format(start_key=query.start_key))]},
        {"configurable": {"thread_id": thread_id}},
    )
    elapsed = time.perf_counter() - start

    messages = out.get("messages", []) if isinstance(out, dict) else []
    final = messages[-1] if messages else None
    final_text = ""
    if isinstance(final, AIMessage):
        raw = final.content
        final_text = raw if isinstance(raw, str) else str(raw or "")

    predicted = extract_uuid(final_text)
    correct = predicted == query.expected_value.lower()

    return QueryResult(
        config_id=config_id,
        nesting_level=query.nesting_level,
        start_key=query.start_key,
        expected=query.expected_value,
        predicted=predicted,
        correct=correct,
        elapsed_seconds=elapsed,
        final_text=final_text,
        chained_heartbeats=int(out.get("chained_heartbeats", 0)) if isinstance(out, dict) else 0,
        archival_search_calls=_count_archival_calls(messages),
    )


AgentFactory = Callable[[MemoryStore], CompiledStateGraph]
StoreFactory = Callable[[int], MemoryStore]


def default_store_factory(_config_id: int) -> MemoryStore:
    return InMemoryStore()


def default_agent_factory(store: MemoryStore) -> CompiledStateGraph:
    return build_agent(
        system_prompt=NESTED_KV_ASSISTANT,
        memory_store=store,
    )


def run_benchmark(
    configs: list[NestedKVConfig],
    *,
    levels: tuple[int, ...] = LEVELS,
    agent_factory: AgentFactory = default_agent_factory,
    store_factory: StoreFactory = default_store_factory,
    on_result: Callable[[QueryResult], None] | None = None,
    sleep_between_seconds: float = 0.0,
) -> BenchmarkSummary:
    """Ejecuta el benchmark sobre ``configs`` y devuelve el resumen.

    Por cada config se construye un ``MemoryStore`` y un agente nuevos para
    aislar la archival entre configs (las cadenas son ortogonales y un agente
    contaminado con la cadena de otro config falsea la métrica).

    ``sleep_between_seconds`` espera ese tiempo tras cada query para evitar
    saturar el TPM de proveedores con cuotas bajas (p.ej. tier 1 GPT-4).
    """
    results: list[QueryResult] = []
    for cfg in configs:
        store = store_factory(cfg.config_id)
        try:
            populate_archival(store, cfg.pairs)
            agent = agent_factory(store)
            for q in cfg.queries:
                if q.nesting_level not in levels:
                    continue
                thread_id = f"nested-kv-cfg{cfg.config_id}-lvl{q.nesting_level}"
                r = run_query(agent, q, config_id=cfg.config_id, thread_id=thread_id)
                results.append(r)
                if on_result is not None:
                    on_result(r)
                if sleep_between_seconds > 0:
                    time.sleep(sleep_between_seconds)
        finally:
            store.close()

    total = len(results)
    correct = sum(1 for r in results if r.correct)
    accuracy = correct / total if total else 0.0

    by_level: dict[int, float] = {}
    for level in levels:
        level_results = [r for r in results if r.nesting_level == level]
        if not level_results:
            continue
        by_level[level] = sum(1 for r in level_results if r.correct) / len(level_results)

    mean_elapsed = (
        sum(r.elapsed_seconds for r in results) / total if total else 0.0
    )

    return BenchmarkSummary(
        total=total,
        correct=correct,
        accuracy=accuracy,
        accuracy_by_level=by_level,
        mean_elapsed_seconds=mean_elapsed,
        results=results,
    )


# ---------------------------------------------------------------------------
# Baseline (sin memoria): los 140 pares viven en el system prompt
# ---------------------------------------------------------------------------


def build_baseline_system_prompt(pairs: tuple[KVPair, ...]) -> str:
    """Construye el system prompt del baseline con los pares como JSON.

    El paper (apéndice 6.1.6) describe el formato exacto: una introducción
    pidiendo búsqueda anidada seguida del objeto JSON con todos los pares.
    Sin tools de archival, el LLM tiene que resolver la cadena leyendo el
    contexto — ese es justo el punto de fallo que MemGPT busca evitar.
    """
    payload = json.dumps({p.key: p.value for p in pairs}, indent=2)
    return f"{BASELINE_SYSTEM_PROMPT_HEADER}\n\n{payload}"


BaselineAgentBuilder = Callable[[tuple[KVPair, ...]], CompiledStateGraph]


def default_baseline_agent_builder(pairs: tuple[KVPair, ...]) -> CompiledStateGraph:
    return build_agent(
        system_prompt=build_baseline_system_prompt(pairs),
        memory_store=None,  # sin recall ni archival.
    )


def make_baseline_agent_builder(model_id: str | None) -> BaselineAgentBuilder:
    """Builder que cierra sobre ``model_id`` para usarlo en el CLI."""

    def _builder(pairs: tuple[KVPair, ...]) -> CompiledStateGraph:
        return build_agent(
            system_prompt=build_baseline_system_prompt(pairs),
            memory_store=None,
            model_id=model_id,
        )

    return _builder


def run_baseline_benchmark(
    configs: list[NestedKVConfig],
    *,
    levels: tuple[int, ...] = LEVELS,
    agent_builder: BaselineAgentBuilder = default_baseline_agent_builder,
    on_result: Callable[[QueryResult], None] | None = None,
    sleep_between_seconds: float = 0.0,
) -> BenchmarkSummary:
    """Ejecuta el benchmark contra un agente sin memoria.

    Por cada config:
    - Construye un agente fresco con los 140 pares incrustados en el system
      prompt como JSON.
    - Lanza las 5 queries (filtradas por ``levels``) usando el prompt del
      apéndice ("references"), que NO menciona archival.

    El ``QueryResult.archival_search_calls`` será siempre 0 — el baseline no
    tiene esa tool. Útil para constatar que las accuracies por nivel se
    desploman donde MemGPT no.
    """
    results: list[QueryResult] = []
    for cfg in configs:
        agent = agent_builder(cfg.pairs)
        for q in cfg.queries:
            if q.nesting_level not in levels:
                continue
            thread_id = f"nested-kv-baseline-cfg{cfg.config_id}-lvl{q.nesting_level}"
            r = run_query(
                agent,
                q,
                config_id=cfg.config_id,
                thread_id=thread_id,
                query_template=BASELINE_QUERY_TEMPLATE,
            )
            results.append(r)
            if on_result is not None:
                on_result(r)
            if sleep_between_seconds > 0:
                time.sleep(sleep_between_seconds)

    total = len(results)
    correct = sum(1 for r in results if r.correct)
    accuracy = correct / total if total else 0.0

    by_level: dict[int, float] = {}
    for level in levels:
        level_results = [r for r in results if r.nesting_level == level]
        if not level_results:
            continue
        by_level[level] = sum(1 for r in level_results if r.correct) / len(level_results)

    mean_elapsed = (
        sum(r.elapsed_seconds for r in results) / total if total else 0.0
    )

    return BenchmarkSummary(
        total=total,
        correct=correct,
        accuracy=accuracy,
        accuracy_by_level=by_level,
        mean_elapsed_seconds=mean_elapsed,
        results=results,
    )
