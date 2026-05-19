"""Tests del benchmark DMR (Fase 9).

Cubren parsing del dataset, métricas, ingesta en Recall y el runner sin
llamar al LLM real. La validación contra LLM auténtico vive en el runner CLI
y se ejecuta on-demand.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from memgpt.benchmarks.dmr import (
    BASELINE_PREPROMPT_TEMPLATE,
    DMR_ASSISTANT_PROMPT,
    DialogTurn,
    DMRResult,
    DMRSample,
    Session,
    build_baseline_summary,
    build_baseline_system_prompt,
    build_dmr_initial_state,
    default_store_factory,
    load_dataset,
    parse_judge_verdict,
    parse_record,
    populate_recall,
    rouge_l_recall,
    run_baseline_benchmark,
    run_benchmark,
    run_sample,
)
from memgpt.memory_store import InMemoryStore


# ---------------------------------------------------------------------------
# Fixture sintético: un solo registro con la misma estructura del dataset real
# ---------------------------------------------------------------------------


def _synthetic_record() -> dict:
    return {
        "personas": [
            [
                "I like Taylor Swift.",
                "I work as a wrestler.",
                "I volunteer at a shelter.",
            ],
            [
                "I have two dogs called Baron and Spike.",
                "I work on Mustangs.",
                "I am a dog trainer.",
            ],
        ],
        "previous_dialogs": [
            {
                "personas": [["a"], ["b"]],
                "dialog": [
                    {"text": "Hi! I'm into country music."},
                    {"text": "Cool. What artists do you like?"},
                    {"text": "Mostly Taylor Swift."},
                    {"text": "Nice."},
                ],
                "time_num": 5,
                "time_unit": "days",
                "time_back": "5 days ago",
            },
            {
                "personas": [["a"], ["b"]],
                "dialog": [
                    {"text": "How are your dogs?"},
                    {"text": "Great, working on the Mustangs."},
                ],
                "time_num": 2,
                "time_unit": "days",
                "time_back": "2 days ago",
            },
        ],
        "dialog": [
            {"id": "Speaker 1", "text": "I won my fight today."},
            {"id": "Speaker 2", "text": "Awesome!"},
        ],
        "metadata": {"initial_data_id": "valid_synth", "session_id": 4},
        "self_instruct": {
            "B": "Hey, what artist did you say you could get into?",
            "A": "Taylor Swift!",
        },
        "summary_speaker_1": [["I like Taylor Swift.", "I am a wrestler."]],
        "summary_speaker_2": [["I have two dogs."]],
        "init_personas": [],
        "personas_update1": [],
        "personas_update2": [],
    }


# ---------------------------------------------------------------------------
# Dataset parsing
# ---------------------------------------------------------------------------


def test_parse_record_assigns_speakers_correctly():
    sample = parse_record(_synthetic_record(), sample_id=0)
    assert sample.agent_speaker == "Speaker 1"
    assert sample.other_speaker == "Speaker 2"
    assert "Taylor Swift" in "\n".join(sample.persona_agent)
    assert "two dogs" in "\n".join(sample.persona_other)


def test_parse_record_preserves_session_order_and_count():
    sample = parse_record(_synthetic_record(), sample_id=0)
    # 2 prev dialogs + 1 current = 3 sesiones.
    assert len(sample.sessions) == 3
    # La última sesión es la "actual" (days_ago = 0).
    assert sample.sessions[-1].days_ago == 0.0
    # Las anteriores están más en el pasado, en orden cronológico creciente.
    assert sample.sessions[0].days_ago > sample.sessions[1].days_ago > 0


def test_parse_record_alternates_speakers_when_id_missing():
    sample = parse_record(_synthetic_record(), sample_id=0)
    prev = sample.sessions[0]
    assert prev.turns[0].speaker == "Speaker 1"
    assert prev.turns[1].speaker == "Speaker 2"
    assert prev.turns[2].speaker == "Speaker 1"


def test_parse_record_uses_explicit_speaker_for_session5():
    sample = parse_record(_synthetic_record(), sample_id=0)
    last = sample.sessions[-1]
    assert last.turns[0].speaker == "Speaker 1"
    assert last.turns[1].speaker == "Speaker 2"


def test_parse_record_extracts_dmr_question_and_gold():
    sample = parse_record(_synthetic_record(), sample_id=0)
    assert sample.question.startswith("Hey, what artist")
    assert sample.gold_answer == "Taylor Swift!"


def test_parse_record_swap_agent_speaker_swaps_personas():
    sample = parse_record(_synthetic_record(), sample_id=0, agent_speaker="Speaker 2")
    assert sample.agent_speaker == "Speaker 2"
    assert "two dogs" in "\n".join(sample.persona_agent)
    assert "Taylor Swift" in "\n".join(sample.persona_other)


def test_load_dataset_with_limit(tmp_path: Path):
    target = tmp_path / "msc.jsonl"
    rec = _synthetic_record()
    with target.open("w") as fp:
        for _ in range(3):
            fp.write(json.dumps(rec) + "\n")
    samples = load_dataset(target, limit=2)
    assert len(samples) == 2
    assert all(isinstance(s, DMRSample) for s in samples)


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------


def test_rouge_l_recall_perfect_match():
    assert rouge_l_recall("Taylor Swift", "Taylor Swift") == 1.0


def test_rouge_l_recall_handles_verbose_candidate():
    # El paper usa recall específicamente para no penalizar respuestas largas.
    r = rouge_l_recall(
        "shell necklace",
        "Oh yeah, I got a shell necklace and a surfboard there.",
    )
    assert r == 1.0


def test_rouge_l_recall_partial_match():
    r = rouge_l_recall("Taylor Swift", "I like Taylor")  # 1 of 2 tokens matched.
    assert r == 0.5


def test_rouge_l_recall_returns_zero_for_empty_inputs():
    assert rouge_l_recall("", "anything") == 0.0
    assert rouge_l_recall("anything", "") == 0.0


def test_rouge_l_recall_ignores_punctuation_and_case():
    # Deben tokenizarse igual.
    a = rouge_l_recall("Taylor Swift!", "taylor swift")
    b = rouge_l_recall("taylor swift", "Taylor Swift!")
    assert a == 1.0
    assert b == 1.0


def test_parse_judge_verdict_recognises_correct():
    text = "The generated answer mentions the same artist. CORRECT"
    assert parse_judge_verdict(text) is True


def test_parse_judge_verdict_recognises_wrong():
    text = "The answer is unrelated to the topic. WRONG"
    assert parse_judge_verdict(text) is False


def test_parse_judge_verdict_returns_none_on_ambiguous():
    text = "It could be CORRECT or WRONG, hard to tell."
    assert parse_judge_verdict(text) is None


def test_parse_judge_verdict_returns_none_on_missing_verdict():
    assert parse_judge_verdict("no verdict here") is None
    assert parse_judge_verdict("") is None


def test_parse_judge_verdict_is_case_insensitive():
    assert parse_judge_verdict("correct") is True
    assert parse_judge_verdict("WRONG") is False


# ---------------------------------------------------------------------------
# Recall ingestion
# ---------------------------------------------------------------------------


def test_populate_recall_inserts_every_turn():
    sample = parse_record(_synthetic_record(), sample_id=0)
    store = InMemoryStore()
    populate_recall(store, sample)
    expected = sum(len(s.turns) for s in sample.sessions)
    assert len(store.messages) == expected


def test_populate_recall_assigns_assistant_role_to_agent_turns():
    sample = parse_record(_synthetic_record(), sample_id=0)
    store = InMemoryStore()
    populate_recall(store, sample)

    assistant_count = sum(1 for ep in store.messages if ep.role == "assistant")
    user_count = sum(1 for ep in store.messages if ep.role == "user")
    # En el fixture: las prev sesiones alternan (Speaker 1 abre) y la sesión 5
    # tiene 1 + 1.
    expected_assistant = 0
    expected_user = 0
    for s in sample.sessions:
        for t in s.turns:
            if t.speaker == sample.agent_speaker:
                expected_assistant += 1
            else:
                expected_user += 1
    assert assistant_count == expected_assistant
    assert user_count == expected_user


def test_populate_recall_orders_sessions_chronologically():
    sample = parse_record(_synthetic_record(), sample_id=0)
    store = InMemoryStore()
    anchor = datetime(2025, 1, 1, tzinfo=timezone.utc)
    populate_recall(store, sample, anchor=anchor)
    # El primer turno de la sesión 5 debe ser más reciente que el primer turno
    # de cualquier sesión previa.
    msgs = store.messages
    last_session_first = next(
        m for m in msgs if m.content.startswith("I won my fight")
    )
    earliest_prev = min(msgs, key=lambda m: m.occurred_at)
    assert last_session_first.occurred_at > earliest_prev.occurred_at


def test_populate_recall_search_finds_relevant_fact():
    sample = parse_record(_synthetic_record(), sample_id=0)
    store = InMemoryStore()
    populate_recall(store, sample)
    hits = store.search_conversation("Taylor Swift")
    assert hits, "search must find the Taylor Swift fact in recall"
    assert any("Taylor Swift" in h.content for h in hits)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def test_assistant_prompt_matches_appendix_6_1_1():
    # Anclas literales del apéndice 6.1.1.
    assert "completely immerse myself in this role" in DMR_ASSISTANT_PROMPT
    assert "conversation_search" in DMR_ASSISTANT_PROMPT


def test_baseline_preprompt_contains_no_answer_contract():
    assert "{conversation_summary}" in BASELINE_PREPROMPT_TEMPLATE
    assert "NO ANSWER" in BASELINE_PREPROMPT_TEMPLATE


def test_baseline_summary_includes_both_personas():
    sample = parse_record(_synthetic_record(), sample_id=0)
    summary = build_baseline_summary(sample)
    assert "Taylor Swift" in summary
    assert "two dogs" in summary


def test_baseline_system_prompt_substitutes_summary():
    sample = parse_record(_synthetic_record(), sample_id=0)
    prompt = build_baseline_system_prompt(sample)
    assert "Taylor Swift" in prompt
    # No deben quedar placeholders sin sustituir.
    assert "{conversation_summary}" not in prompt


# ---------------------------------------------------------------------------
# Runner con stub
# ---------------------------------------------------------------------------


class _RecallReadingStub:
    """Stub que responde leyendo Recall directamente.

    Imita un agente perfecto: busca la pregunta en Recall y devuelve el
    contenido del primer match. Sirve para probar el cableado completo
    sample → store → agent → scoring.
    """

    def __init__(self, store: InMemoryStore, *, query_term: str) -> None:
        self._store = store
        self._term = query_term

    def invoke(self, payload: dict, *, config: dict) -> dict:
        question = next(
            (m for m in payload.get("messages", []) if isinstance(m, HumanMessage)),
            None,
        )
        assert question is not None
        hits = self._store.search_conversation(self._term, limit=1)
        answer = hits[0].content if hits else "I don't know"
        return {
            "messages": [question, AIMessage(content=answer)],
            "chained_heartbeats": 1,
        }


def _stub_judge_correct() -> "object":
    def _judge(question: str, gold: str, generated: str) -> tuple[bool | None, str]:
        # Veredicto laxo: CORRECT si ambas tokens del gold aparecen en generated.
        gold_tokens = gold.lower().rstrip("!.").split()
        ok = all(t in generated.lower() for t in gold_tokens)
        verdict = "CORRECT" if ok else "WRONG"
        return ok, f"stub judge verdict: {verdict}"

    return _judge


def test_run_sample_scores_and_judges():
    sample = parse_record(_synthetic_record(), sample_id=0)
    store = InMemoryStore()
    populate_recall(store, sample)
    agent = _RecallReadingStub(store, query_term="Taylor Swift")
    initial = build_dmr_initial_state(sample)

    result = run_sample(
        sample,
        agent=agent,
        judge=_stub_judge_correct(),
        initial_state=initial,
    )
    assert isinstance(result, DMRResult)
    assert result.judge_correct is True
    assert result.rouge_l_recall > 0  # alguna palabra del gold debe estar.
    assert "Taylor Swift" in result.predicted


def test_run_benchmark_uses_per_sample_isolation():
    # Si un agente "lee" la recall de otro sample, falla — verificamos que el
    # runner construye un store por sample y no reutiliza.
    sample = parse_record(_synthetic_record(), sample_id=0)
    samples = [sample, sample, sample]

    seen_stores: list[InMemoryStore] = []

    def store_factory(s: DMRSample) -> InMemoryStore:
        s_ = InMemoryStore()
        seen_stores.append(s_)  # mantener referencia para evitar reuso de id().
        return s_

    def agent_factory(s: DMRSample, store) -> _RecallReadingStub:
        return _RecallReadingStub(store, query_term="Taylor Swift")

    summary = run_benchmark(
        samples,
        judge=_stub_judge_correct(),
        agent_factory=agent_factory,
        store_factory=store_factory,
    )

    assert summary.total == 3
    assert summary.correct == 3
    assert summary.accuracy == 1.0
    # Tres stores distintos: aislamiento garantizado.
    assert len({id(s) for s in seen_stores}) == 3


def test_run_baseline_benchmark_with_stub_falls_back_when_summary_lacks_fact():
    """El baseline no tiene Recall: si el summary no contiene el dato, falla.

    Reproducimos el modo de fallo que justifica MemGPT en el paper (Tabla 2).
    """
    record = _synthetic_record()
    # Borramos el dato clave del resumen para simular un baseline ciego.
    record["summary_speaker_1"] = [["I am a wrestler."]]
    sample = parse_record(record, sample_id=0)

    class _SummaryReadingStub:
        def __init__(self, sample: DMRSample) -> None:
            self._summary = build_baseline_summary(sample)

        def invoke(self, payload: dict, *, config: dict) -> dict:
            question = payload["messages"][0]
            answer = "Taylor Swift" if "Taylor Swift" in self._summary else "NO ANSWER"
            return {"messages": [question, AIMessage(content=answer)]}

    summary = run_baseline_benchmark(
        [sample],
        judge=_stub_judge_correct(),
        agent_factory=lambda s: _SummaryReadingStub(s),
    )
    assert summary.total == 1
    assert summary.correct == 0  # baseline ciego ⇒ falla.
    assert summary.results[0].predicted == "NO ANSWER"


def test_run_benchmark_summary_handles_judge_abstention():
    """Si el juez no se decide, el sample no cuenta como correcto ni como acc."""
    sample = parse_record(_synthetic_record(), sample_id=0)

    class _NeutralStub:
        def invoke(self, payload, *, config):
            return {"messages": [payload["messages"][0], AIMessage(content="...")]}

    def _abstaining_judge(q, g, gen):
        return None, "the judge abstains"

    summary = run_benchmark(
        [sample],
        judge=_abstaining_judge,
        agent_factory=lambda s, store: _NeutralStub(),
    )
    assert summary.total == 1
    assert summary.judged == 0
    assert summary.correct == 0
    assert summary.accuracy == 0.0


def test_default_store_factory_returns_inmemory():
    sample = parse_record(_synthetic_record(), sample_id=0)
    store = default_store_factory(sample)
    assert isinstance(store, InMemoryStore)


def test_dmr_session_dataclass_round_trip():
    s = Session(
        turns=(DialogTurn(speaker="Speaker 1", text="hello"),),
        days_ago=2.5,
    )
    assert s.turns[0].speaker == "Speaker 1"
    assert s.days_ago == 2.5
