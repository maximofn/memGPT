"""Tests del benchmark Document QA (Fase 10).

Cubren parsing del dataset, parser del formato ``ANSWER/DOCUMENT``,
ingesta a archival, baseline (top-K en prompt), métricas y el runner sin
llamar al LLM real. La validación contra LLM auténtico vive en el runner
CLI y se ejecuta on-demand.
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from memgpt.benchmarks.document_qa import (
    BASELINE_DOC_QA_ASSISTANT,
    BASELINE_QUERY_TEMPLATE,
    DOC_QA_ASSISTANT_PROMPT,
    DOC_QA_QUERY_TEMPLATE,
    Document,
    DocQAResult,
    DocumentSample,
    exact_match,
    format_baseline_documents,
    load_dataset,
    parse_judge_verdict,
    parse_record,
    parse_response,
    populate_archival,
    run_baseline_benchmark,
    run_baseline_sample,
    run_benchmark,
    run_sample,
)
from memgpt.memory_store import InMemoryStore


# ---------------------------------------------------------------------------
# Fixture sintético: un solo registro con la misma estructura del dataset real
# ---------------------------------------------------------------------------


def _synthetic_record() -> dict:
    return {
        "question": "Who wrote Fawlty Towers?",
        "answers": ["John Cleese", "Connie Booth"],
        "ctxs": [
            {
                "id": "wiki-1",
                "title": "Fawlty Towers",
                "text": (
                    "Fawlty Towers is a British sitcom produced by BBC2 in 1975 "
                    "and 1979. The series was written by John Cleese and Connie Booth, "
                    "who also starred in it."
                ),
                "hasanswer": True,
            },
            {
                "id": "wiki-2",
                "title": "Monty Python",
                "text": "Monty Python's Flying Circus is a British comedy series.",
                "hasanswer": False,
            },
            {
                "id": "wiki-3",
                "title": "British sitcoms",
                "text": "British sitcoms include shows like Blackadder and Yes Minister.",
                "hasanswer": False,
            },
        ],
    }


# ---------------------------------------------------------------------------
# Dataset parsing
# ---------------------------------------------------------------------------


def test_parse_record_extracts_question_and_answers():
    sample = parse_record(_synthetic_record(), sample_id=0)
    assert sample.question == "Who wrote Fawlty Towers?"
    assert sample.gold_answers == ("John Cleese", "Connie Booth")


def test_parse_record_loads_documents_with_has_answer_flag():
    sample = parse_record(_synthetic_record(), sample_id=0)
    assert len(sample.documents) == 3
    assert sample.documents[0].has_answer is True
    assert sample.documents[1].has_answer is False


def test_parse_record_accepts_documents_alias():
    record = _synthetic_record()
    record["documents"] = record.pop("ctxs")
    sample = parse_record(record, sample_id=0)
    assert len(sample.documents) == 3


def test_parse_record_accepts_string_answer():
    record = _synthetic_record()
    record["answers"] = "John Cleese"
    sample = parse_record(record, sample_id=0)
    assert sample.gold_answers == ("John Cleese",)


def test_parse_record_rejects_missing_question():
    record = _synthetic_record()
    record["question"] = ""
    try:
        parse_record(record, sample_id=0)
    except ValueError as exc:
        assert "question" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_parse_record_rejects_missing_answers():
    record = _synthetic_record()
    record["answers"] = []
    try:
        parse_record(record, sample_id=0)
    except ValueError as exc:
        assert "answers" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_load_dataset_with_limit(tmp_path: Path):
    target = tmp_path / "nq.jsonl"
    rec = _synthetic_record()
    with target.open("w") as fp:
        for _ in range(3):
            fp.write(json.dumps(rec) + "\n")
    samples = load_dataset(target, limit=2)
    assert len(samples) == 2
    assert all(isinstance(s, DocumentSample) for s in samples)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def test_parse_response_extracts_answer_and_document():
    text = "ANSWER: John Cleese, DOCUMENT: Fawlty Towers was written by..."
    parsed = parse_response(text)
    assert parsed.answer == "John Cleese"
    assert parsed.document is not None and "Fawlty Towers" in parsed.document
    assert parsed.has_both_fields is True
    assert parsed.insufficient is False


def test_parse_response_handles_multiline_document():
    text = (
        "ANSWER: John Cleese,\n"
        "DOCUMENT: Fawlty Towers is a British sitcom\n"
        "produced by BBC2."
    )
    parsed = parse_response(text)
    assert parsed.answer == "John Cleese"
    assert parsed.document is not None and "BBC2" in parsed.document


def test_parse_response_recognises_insufficient_information():
    parsed = parse_response("INSUFFICIENT INFORMATION")
    assert parsed.insufficient is True
    assert parsed.answer is None
    assert parsed.has_both_fields is False


def test_parse_response_returns_none_when_format_missing():
    parsed = parse_response("Just some random text without the format.")
    assert parsed.answer is None
    assert parsed.document is None
    assert parsed.insufficient is False


def test_parse_response_handles_empty_input():
    parsed = parse_response("")
    assert parsed.answer is None
    assert parsed.document is None


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------


def test_exact_match_finds_gold_substring_case_insensitive():
    parsed = parse_response("ANSWER: john cleese, DOCUMENT: ...")
    assert exact_match(parsed, ("John Cleese",)) is True


def test_exact_match_returns_false_when_insufficient():
    parsed = parse_response("INSUFFICIENT INFORMATION")
    assert exact_match(parsed, ("John Cleese",)) is False


def test_exact_match_returns_false_when_answer_missing():
    parsed = parse_response("DOCUMENT: only the document field.")
    assert exact_match(parsed, ("John Cleese",)) is False


def test_exact_match_strips_trailing_punctuation():
    parsed = parse_response("ANSWER: John Cleese., DOCUMENT: text")
    assert exact_match(parsed, ("John Cleese",)) is True


def test_parse_judge_verdict_recognises_correct():
    assert parse_judge_verdict("CORRECT") is True
    assert parse_judge_verdict("correct") is True
    assert parse_judge_verdict("The answer is CORRECT.") is True


def test_parse_judge_verdict_recognises_incorrect():
    assert parse_judge_verdict("INCORRECT") is False
    assert parse_judge_verdict("incorrect") is False


def test_parse_judge_verdict_distinguishes_incorrect_from_correct():
    # Sutil: "INCORRECT" contiene "CORRECT" como substring; el parser
    # debe respetar el word boundary y no devolver True por accidente.
    assert parse_judge_verdict("INCORRECT") is False


def test_parse_judge_verdict_returns_none_on_ambiguous():
    text = "It could be CORRECT or INCORRECT, hard to tell."
    assert parse_judge_verdict(text) is None


def test_parse_judge_verdict_returns_none_on_missing_verdict():
    assert parse_judge_verdict("no verdict here") is None
    assert parse_judge_verdict("") is None


# ---------------------------------------------------------------------------
# Archival ingestion
# ---------------------------------------------------------------------------


def test_populate_archival_inserts_every_doc():
    sample = parse_record(_synthetic_record(), sample_id=0)
    store = InMemoryStore()
    n = populate_archival(store, sample)
    assert n == 3
    assert len(store.archival) == 3


def test_populate_archival_skips_empty_docs():
    record = _synthetic_record()
    record["ctxs"][1]["text"] = ""  # vaciamos uno.
    sample = parse_record(record, sample_id=0)
    store = InMemoryStore()
    n = populate_archival(store, sample)
    assert n == 2
    assert len(store.archival) == 2


def test_populate_archival_serialises_title_and_text():
    sample = parse_record(_synthetic_record(), sample_id=0)
    store = InMemoryStore()
    populate_archival(store, sample)
    contents = [ep.content for ep in store.archival]
    assert any("[Fawlty Towers]" in c and "John Cleese" in c for c in contents)


def test_populate_archival_search_finds_relevant_doc():
    sample = parse_record(_synthetic_record(), sample_id=0)
    store = InMemoryStore()
    populate_archival(store, sample)
    hits = store.search_archival("John Cleese")
    assert hits, "search must find the relevant doc"
    assert any("Fawlty Towers" in h.content for h in hits)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def test_assistant_prompt_matches_appendix_6_1_4():
    # Anclas literales del apéndice 6.1.4.
    assert "MemGPT DOC-QA bot" in DOC_QA_ASSISTANT_PROMPT
    assert "archival memory" in DOC_QA_ASSISTANT_PROMPT
    assert "year is 2018" in DOC_QA_ASSISTANT_PROMPT


def test_query_template_demands_answer_document_format():
    assert "ANSWER:" in DOC_QA_QUERY_TEMPLATE
    assert "DOCUMENT:" in DOC_QA_QUERY_TEMPLATE
    assert "{question}" in DOC_QA_QUERY_TEMPLATE


def test_baseline_query_template_includes_insufficient_information():
    assert "INSUFFICIENT INFORMATION" in BASELINE_QUERY_TEMPLATE
    assert "{documents}" in BASELINE_QUERY_TEMPLATE
    assert "{question}" in BASELINE_QUERY_TEMPLATE


def test_baseline_assistant_mentions_2018():
    assert "2018" in BASELINE_DOC_QA_ASSISTANT


# ---------------------------------------------------------------------------
# Baseline document formatting
# ---------------------------------------------------------------------------


def test_format_baseline_documents_numbers_each_doc():
    sample = parse_record(_synthetic_record(), sample_id=0)
    block = format_baseline_documents(sample.documents)
    assert "Document [1]" in block
    assert "Document [2]" in block
    assert "Document [3]" in block
    assert "Fawlty Towers" in block


def test_format_baseline_documents_respects_top_k():
    sample = parse_record(_synthetic_record(), sample_id=0)
    block = format_baseline_documents(sample.documents, top_k=2)
    assert "Document [1]" in block
    assert "Document [2]" in block
    assert "Document [3]" not in block


def test_format_baseline_documents_handles_empty():
    block = format_baseline_documents(())
    assert "no documents" in block.lower()


# ---------------------------------------------------------------------------
# Runner con stubs (sin LLM real)
# ---------------------------------------------------------------------------


class _PerfectArchivalStub:
    """Stub que responde leyendo Archival y emitiendo el formato exigido.

    Imita un agente perfecto: busca la primera palabra significativa de la
    pregunta en archival y devuelve el primer match envuelto en ``ANSWER:
    .../DOCUMENT: ...``. Sirve para probar el cableado completo
    sample → store → agent → scoring sin gastar tokens.
    """

    def __init__(self, store: InMemoryStore, *, answer_text: str, query_term: str) -> None:
        self._store = store
        self._answer = answer_text
        self._term = query_term

    def invoke(self, payload: dict, *, config: dict) -> dict:
        question = next(
            (m for m in payload.get("messages", []) if isinstance(m, HumanMessage)),
            None,
        )
        assert question is not None
        hits = self._store.search_archival(self._term, limit=1)
        doc_text = hits[0].content if hits else "no match"
        # Emite tool_calls registrados para que _count_archival_calls cuente.
        ai_search = AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call-1",
                    "name": "archival_memory_search",
                    "args": {"query": self._term},
                }
            ],
        )
        ai_final = AIMessage(content=f"ANSWER: {self._answer}, DOCUMENT: {doc_text}")
        return {
            "messages": [question, ai_search, ai_final],
            "chained_heartbeats": 1,
        }


class _InsufficientStub:
    """Agente que siempre admite no saber — para validar el parsing del flag."""

    def invoke(self, payload: dict, *, config: dict) -> dict:
        question = next(
            (m for m in payload.get("messages", []) if isinstance(m, HumanMessage)),
            None,
        )
        return {
            "messages": [question, AIMessage(content="INSUFFICIENT INFORMATION")],
            "chained_heartbeats": 0,
        }


def _stub_judge_correct():
    """Veredicto laxo: CORRECT si alguna gold answer aparece en generated."""

    def _judge(question: str, gold_answers: tuple[str, ...], generated: str):
        ok = any(g.lower() in generated.lower() for g in gold_answers)
        verdict = "CORRECT" if ok else "INCORRECT"
        return ok, f"stub judge verdict: {verdict}"

    return _judge


def test_run_sample_scores_and_judges_perfect_agent():
    sample = parse_record(_synthetic_record(), sample_id=0)
    store = InMemoryStore()
    n = populate_archival(store, sample)
    agent = _PerfectArchivalStub(store, answer_text="John Cleese", query_term="Cleese")

    result = run_sample(
        sample,
        agent=agent,
        judge=_stub_judge_correct(),
        documents_ingested=n,
    )
    assert isinstance(result, DocQAResult)
    assert result.judge_correct is True
    assert result.exact_match is True
    assert result.predicted_answer == "John Cleese"
    assert result.predicted_document is not None
    assert result.archival_search_calls == 1
    assert result.documents_ingested == 3


def test_run_sample_marks_insufficient_information():
    sample = parse_record(_synthetic_record(), sample_id=0)
    store = InMemoryStore()
    populate_archival(store, sample)
    agent = _InsufficientStub()

    result = run_sample(sample, agent=agent, judge=_stub_judge_correct())
    assert result.insufficient is True
    assert result.exact_match is False
    assert result.judge_correct is False  # gold no aparece en "INSUFFICIENT INFORMATION".


def test_run_benchmark_uses_per_sample_isolation():
    sample = parse_record(_synthetic_record(), sample_id=0)
    samples = [sample, sample, sample]

    seen_stores: list[InMemoryStore] = []

    def store_factory(_s: DocumentSample) -> InMemoryStore:
        s_ = InMemoryStore()
        seen_stores.append(s_)
        return s_

    def agent_factory(_s: DocumentSample, store: InMemoryStore) -> _PerfectArchivalStub:
        return _PerfectArchivalStub(
            store, answer_text="John Cleese", query_term="Cleese"
        )

    summary = run_benchmark(
        samples,
        judge=_stub_judge_correct(),
        agent_factory=agent_factory,
        store_factory=store_factory,
    )

    assert summary.total == 3
    assert summary.correct == 3
    assert summary.accuracy == 1.0
    assert summary.exact_match_rate == 1.0
    assert len({id(s) for s in seen_stores}) == 3


def test_run_benchmark_with_shared_store_skips_per_sample_ingestion():
    """Modo "corpus global": el caller pre-carga el store y el runner no lo toca."""
    sample = parse_record(_synthetic_record(), sample_id=0)
    samples = [sample, sample]

    shared = InMemoryStore()
    # Cargamos manualmente UNA SOLA vez antes del runner.
    populate_archival(shared, sample)
    pre_count = len(shared.archival)

    def agent_factory(_s: DocumentSample, store: InMemoryStore) -> _PerfectArchivalStub:
        return _PerfectArchivalStub(
            store, answer_text="John Cleese", query_term="Cleese"
        )

    summary = run_benchmark(
        samples,
        judge=_stub_judge_correct(),
        agent_factory=agent_factory,
        shared_store=shared,
    )

    # Si el runner hubiera vuelto a poblar, archival tendría 2× los docs.
    assert len(shared.archival) == pre_count
    assert summary.total == 2
    # documents_ingested = 0 porque el caller lo gestiona aparte.
    assert all(r.documents_ingested == 0 for r in summary.results)


def test_run_baseline_sample_uses_query_template_with_documents():
    """El baseline DEBE inlining de docs en el prompt, sin usar archival."""
    sample = parse_record(_synthetic_record(), sample_id=0)

    captured: dict = {}

    class _CapturingStub:
        def invoke(self, payload: dict, *, config: dict):
            captured["payload"] = payload
            question = next(m for m in payload["messages"] if isinstance(m, HumanMessage))
            return {
                "messages": [
                    question,
                    AIMessage(
                        content="ANSWER: John Cleese, DOCUMENT: Fawlty Towers..."
                    ),
                ],
            }

    result = run_baseline_sample(
        sample,
        agent=_CapturingStub(),
        judge=_stub_judge_correct(),
    )
    user_text = captured["payload"]["messages"][0].content
    assert "Document [1]" in user_text
    assert "Fawlty Towers" in user_text
    assert "Who wrote Fawlty Towers?" in user_text
    assert result.judge_correct is True
    assert result.archival_search_calls == 0  # baseline no tiene archival.


def test_run_baseline_benchmark_returns_summary():
    sample = parse_record(_synthetic_record(), sample_id=0)

    class _AlwaysCorrect:
        def invoke(self, payload: dict, *, config: dict):
            question = next(m for m in payload["messages"] if isinstance(m, HumanMessage))
            return {
                "messages": [
                    question,
                    AIMessage(content="ANSWER: John Cleese, DOCUMENT: ..."),
                ],
            }

    def _factory(_s: DocumentSample) -> _AlwaysCorrect:
        return _AlwaysCorrect()

    summary = run_baseline_benchmark(
        [sample, sample],
        judge=_stub_judge_correct(),
        agent_factory=_factory,
        top_k=2,
    )
    assert summary.total == 2
    assert summary.correct == 2
    assert summary.accuracy == 1.0
    # top_k=2 ⇒ documents_ingested debe reflejar el truncamiento.
    assert all(r.documents_ingested == 2 for r in summary.results)
