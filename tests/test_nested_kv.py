"""Tests del benchmark Nested KV (Fase 8).

Cubren la generación del dataset y el scoring sin llamar al LLM. Para los
tests con LLM real está ``test_nested_kv_e2e.py`` (no incluido aquí: depende
de credenciales y se ejecuta on-demand).
"""

from __future__ import annotations

import uuid

from langchain_core.messages import AIMessage, HumanMessage

from memgpt.benchmarks.nested_kv import (
    BASELINE_QUERY_TEMPLATE,
    BASELINE_SYSTEM_PROMPT_HEADER,
    CHAIN_LENGTH,
    LEVELS,
    NESTED_KV_ASSISTANT,
    PAIRS_PER_CONFIG,
    QUERY_TEMPLATE,
    KVPair,
    NestedKVQuery,
    QueryResult,
    build_baseline_system_prompt,
    default_store_factory,
    extract_uuid,
    generate_config,
    generate_dataset,
    populate_archival,
    run_baseline_benchmark,
    run_benchmark,
    run_query,
)
from memgpt.memory_store import InMemoryStore


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------


def test_generate_config_has_140_unique_pairs():
    cfg = generate_config(0, seed=1)
    assert len(cfg.pairs) == PAIRS_PER_CONFIG
    keys = [p.key for p in cfg.pairs]
    assert len(set(keys)) == PAIRS_PER_CONFIG, "keys must be unique within a config"


def test_generate_config_chain_resolves_for_every_level():
    cfg = generate_config(7, seed=99)
    pair_by_key = {p.key: p.value for p in cfg.pairs}
    assert len(cfg.queries) == len(LEVELS)

    for q in cfg.queries:
        # Resolver L saltos partiendo de start_key debe llegar al expected.
        cur = q.start_key
        for _ in range(q.nesting_level + 1):
            assert cur in pair_by_key, f"chain broken at {cur}"
            cur = pair_by_key[cur]
        assert cur == q.expected_value
        # El terminal NO debe ser una clave del dataset.
        assert q.expected_value not in pair_by_key


def test_generate_config_distractor_values_are_not_chain_keys():
    cfg = generate_config(3, seed=11)
    chain_keys: set[str] = set()
    cur = cfg.queries[-1].start_key  # nivel 4 ⇒ k0 de la cadena.
    pair_by_key = {p.key: p.value for p in cfg.pairs}
    for _ in range(CHAIN_LENGTH):
        chain_keys.add(cur)
        cur = pair_by_key[cur]
    distractor_pairs = [p for p in cfg.pairs if p.key not in chain_keys]
    assert len(distractor_pairs) == PAIRS_PER_CONFIG - CHAIN_LENGTH
    for p in distractor_pairs:
        assert p.value not in chain_keys, (
            "distractor value must not collide with chain keys "
            "(would create unintended nesting)"
        )


def test_generate_config_is_deterministic():
    a = generate_config(2, seed=42)
    b = generate_config(2, seed=42)
    assert a.pairs == b.pairs
    assert a.queries == b.queries


def test_generate_dataset_returns_30_independent_configs():
    ds = generate_dataset(seed=42, n_configs=30)
    assert len(ds) == 30
    # Configs distintas deben tener cadenas distintas.
    starts = {cfg.queries[0].start_key for cfg in ds}
    assert len(starts) == 30


def test_query_count_matches_paper():
    ds = generate_dataset(seed=42, n_configs=30)
    total_queries = sum(len(cfg.queries) for cfg in ds)
    assert total_queries == 30 * 5 == 150


# ---------------------------------------------------------------------------
# UUID extraction / scoring
# ---------------------------------------------------------------------------


def test_extract_uuid_picks_last_match():
    text = (
        "I searched and found 11111111-2222-3333-4444-555555555555 first, "
        "then resolved to aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee."
    )
    out = extract_uuid(text)
    assert out == "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


def test_extract_uuid_returns_none_when_absent():
    assert extract_uuid("no uuid here, just words") is None
    assert extract_uuid("") is None
    assert extract_uuid(None) is None  # type: ignore[arg-type]


def test_extract_uuid_normalises_case():
    upper = "FFFFFFFF-FFFF-FFFF-FFFF-FFFFFFFFFFFF"
    assert extract_uuid(upper) == upper.lower()


# ---------------------------------------------------------------------------
# Archival population (sin LLM)
# ---------------------------------------------------------------------------


def test_populate_archival_inserts_every_pair():
    cfg = generate_config(0, seed=1)
    store = InMemoryStore()
    populate_archival(store, cfg.pairs)
    assert len(store.archival) == PAIRS_PER_CONFIG


def test_archival_search_finds_chain_value_by_key():
    cfg = generate_config(0, seed=1)
    store = InMemoryStore()
    populate_archival(store, cfg.pairs)
    chain_start = cfg.queries[-1].start_key  # nivel 4 ⇒ k0.
    hits = store.search_archival(chain_start, limit=5)
    # Solo el par cuya clave es chain_start debería contenerlo.
    matching = [h for h in hits if h.content.startswith(f"key={chain_start}")]
    assert matching, "search must return the pair indexed by start_key"


# ---------------------------------------------------------------------------
# Asistente / prompt sanity
# ---------------------------------------------------------------------------


def test_assistant_contains_paper_anchor_phrase():
    # Apéndice 6.1.6: la frase clave es la traducción de "DO NOT STOP SEARCHING".
    assert "DO NOT STOP SEARCHING" in NESTED_KV_ASSISTANT


def test_query_template_contains_start_key_placeholder():
    assert "{start_key}" in QUERY_TEMPLATE


# ---------------------------------------------------------------------------
# Runner integration con un agent factory falso (sin LLM)
# ---------------------------------------------------------------------------


class _StubAgent:
    """Agente falso que devuelve siempre el ``terminal`` correcto del config.

    Resuelve la cadena buscando linealmente en archival como referencia. Se
    usa para validar que ``run_benchmark`` cablea bien dataset → store →
    agent → scoring, sin tocar la red ni a un LLM.
    """

    def __init__(self, store: InMemoryStore) -> None:
        self._store = store

    def invoke(self, payload: dict, *, config: dict) -> dict:
        msg = payload["messages"][0]
        assert isinstance(msg, HumanMessage)
        # Extraemos el start_key del prompt y resolvemos la cadena.
        start = extract_uuid(msg.content or "")
        assert start is not None
        cur = start
        pair_by_key: dict[str, str] = {}
        for ep in self._store.archival:
            # contenido: "key=<uuid> value=<uuid>"
            text = ep.content
            if not text.startswith("key="):
                continue
            k = text.split(" ", 1)[0].removeprefix("key=")
            v = text.split("value=", 1)[1].strip()
            pair_by_key[k] = v
        for _ in range(10):
            nxt = pair_by_key.get(cur)
            if nxt is None:
                break
            cur = nxt
        ai = AIMessage(content=cur)
        return {"messages": [msg, ai], "chained_heartbeats": 1}


def test_run_query_scores_correctly_with_stub():
    cfg = generate_config(0, seed=5)
    store = InMemoryStore()
    populate_archival(store, cfg.pairs)
    agent = _StubAgent(store)

    for q in cfg.queries:
        r = run_query(agent, q, config_id=cfg.config_id, thread_id="test")
        assert isinstance(r, QueryResult)
        assert r.correct, (
            f"stub agent should resolve every chain (level={q.nesting_level} "
            f"start={q.start_key} predicted={r.predicted} expected={q.expected_value})"
        )


def test_run_benchmark_summary_is_consistent_with_stub():
    configs = generate_dataset(seed=42, n_configs=3)

    summary = run_benchmark(
        configs,
        agent_factory=lambda store: _StubAgent(store),
        store_factory=default_store_factory,
    )

    assert summary.total == 3 * len(LEVELS) == 15
    assert summary.correct == summary.total
    assert summary.accuracy == 1.0
    # Una entrada por nivel.
    assert sorted(summary.accuracy_by_level) == list(LEVELS)
    for lvl in LEVELS:
        assert summary.accuracy_by_level[lvl] == 1.0


def test_run_benchmark_filters_levels():
    configs = generate_dataset(seed=42, n_configs=2)
    summary = run_benchmark(
        configs,
        levels=(0, 4),
        agent_factory=lambda store: _StubAgent(store),
    )
    assert summary.total == 2 * 2  # 2 configs × 2 levels.
    assert sorted(summary.accuracy_by_level) == [0, 4]


def test_kv_pair_archival_text_is_round_trippable():
    k = str(uuid.uuid4())
    v = str(uuid.uuid4())
    p = KVPair(key=k, value=v)
    text = p.as_archival_text()
    assert k in text and v in text
    # El parseo del stub debe recuperar k → v exactamente.
    parsed_key = text.split(" ", 1)[0].removeprefix("key=")
    parsed_value = text.split("value=", 1)[1].strip()
    assert parsed_key == k
    assert parsed_value == v


def test_run_benchmark_reports_progress_callback():
    configs = generate_dataset(seed=42, n_configs=1)
    seen: list[int] = []

    def cb(r: QueryResult) -> None:
        seen.append(r.nesting_level)

    run_benchmark(
        configs,
        agent_factory=lambda store: _StubAgent(store),
        on_result=cb,
    )
    assert sorted(seen) == list(LEVELS)


def test_nested_kv_query_dataclass_round_trip():
    q = NestedKVQuery(start_key="a", expected_value="b", nesting_level=0)
    assert q.nesting_level == 0
    assert q.start_key == "a"
    assert q.expected_value == "b"


# ---------------------------------------------------------------------------
# Baseline (sin memoria, pares en system prompt)
# ---------------------------------------------------------------------------


class _BaselineStubAgent:
    """Resuelve la cadena leyendo el system prompt JSON.

    Simula un LLM "perfecto" sobre el baseline: parsea el JSON de pares y
    encadena lookups. Sirve para validar el cableado, no la curva real
    (que requiere LLM auténtico).
    """

    def __init__(self, system_prompt: str) -> None:
        import json as _json

        # El system prompt es: header + "\n\n" + json.
        json_blob = system_prompt.split("\n\n", 1)[1]
        self._pairs: dict[str, str] = _json.loads(json_blob)

    def invoke(self, payload: dict, *, config: dict) -> dict:
        msg = payload["messages"][0]
        start = extract_uuid(msg.content or "")
        assert start is not None
        cur = start
        for _ in range(10):
            nxt = self._pairs.get(cur)
            if nxt is None:
                break
            cur = nxt
        return {
            "messages": [msg, AIMessage(content=cur)],
            "chained_heartbeats": 0,
        }


def test_build_baseline_system_prompt_embeds_all_pairs_as_json():
    cfg = generate_config(0, seed=1)
    prompt = build_baseline_system_prompt(cfg.pairs)
    assert BASELINE_SYSTEM_PROMPT_HEADER in prompt
    # Cada UUID debe aparecer literal en el prompt.
    for p in cfg.pairs:
        assert p.key in prompt
        assert p.value in prompt


def test_baseline_query_template_does_not_mention_archival():
    # El baseline NO tiene tools de archival; el prompt no debe inducir el uso.
    assert "archival" not in BASELINE_QUERY_TEMPLATE.lower()


def test_run_baseline_benchmark_with_perfect_stub_hits_100_percent():
    configs = generate_dataset(seed=42, n_configs=2)

    def builder(pairs):
        return _BaselineStubAgent(build_baseline_system_prompt(pairs))

    summary = run_baseline_benchmark(configs, agent_builder=builder)
    assert summary.total == 2 * len(LEVELS) == 10
    assert summary.correct == summary.total
    assert summary.accuracy == 1.0
    # El baseline nunca debe registrar archival_search_calls (no hay tool).
    assert all(r.archival_search_calls == 0 for r in summary.results)


def test_run_baseline_benchmark_filters_levels():
    configs = generate_dataset(seed=42, n_configs=1)

    def builder(pairs):
        return _BaselineStubAgent(build_baseline_system_prompt(pairs))

    summary = run_baseline_benchmark(configs, levels=(0, 4), agent_builder=builder)
    assert summary.total == 2
    assert sorted(summary.accuracy_by_level) == [0, 4]


class _BrokenBaselineAgent:
    """Devuelve siempre el primer valor sin seguir la cadena.

    Reproduce el modo de fallo típico del baseline GPT-3.5 descrito en el
    paper: "su principal modo de fallo es simplemente devolver el valor
    original".
    """

    def __init__(self, system_prompt: str) -> None:
        import json as _json

        self._pairs: dict[str, str] = _json.loads(system_prompt.split("\n\n", 1)[1])

    def invoke(self, payload: dict, *, config: dict) -> dict:
        msg = payload["messages"][0]
        start = extract_uuid(msg.content or "")
        assert start is not None
        first_value = self._pairs[start]  # 1 lookup, no nesting.
        return {"messages": [msg, AIMessage(content=first_value)], "chained_heartbeats": 0}


def test_baseline_failure_mode_only_passes_level_0():
    """Sanity: un agente que solo hace 1 lookup acierta nivel 0 y falla 1..4.

    Es el comportamiento que vamos a observar en GPT-3.5 real y que
    justifica la existencia de MemGPT.
    """
    configs = generate_dataset(seed=42, n_configs=3)

    def builder(pairs):
        return _BrokenBaselineAgent(build_baseline_system_prompt(pairs))

    summary = run_baseline_benchmark(configs, agent_builder=builder)
    assert summary.accuracy_by_level[0] == 1.0
    for lvl in (1, 2, 3, 4):
        assert summary.accuracy_by_level[lvl] == 0.0
