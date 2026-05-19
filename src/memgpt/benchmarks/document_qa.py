"""Document QA benchmark — Fase 10, §3.2.3 + apéndices 6.1.4 / 6.1.5.

Replica el tercer benchmark del paper sobre **NaturalQuestions-Open**
(retriever-reader de Liu et al. 2023a, *"Lost in the middle"*). El paper
original usa los embeddings precalculados de los 20M artículos de
Wikipedia subidos a Hugging Face por los autores; cargarlos en cualquier
backend con extracción de entidades (p. ej. Graphiti) llevaría horas y
no es realista para la suite de tests. Por eso este benchmark usa
exclusivamente ``InMemoryStore`` — es vector store puro y la ingestión
es barata. Nuestra implementación admite **dos modos de uso**:

1. **Modo "lost-in-the-middle" (default y testable):** cada sample trae
   sus propios ``ctxs`` (típicamente 10-30 documentos por pregunta). El
   runner ingiere los ctxs del sample en un Archival aislado antes de
   preguntar. Reproduce la condición *"el documento gold puede estar
   fuera de los primeros K resultados"* sin necesitar montar pgvector +
   20M docs. Es el que validan los tests.
2. **Modo "corpus global":** un único Archival precargado se reutiliza
   entre samples (``populate_archival_from_corpus`` + ``corpus_store``).
   Lo dejamos como hook explícito para cuando montes la BD completa con
   los embeddings de HF — la lógica del agente (paginación + judge) es
   idéntica.

Pipeline por sample (modo lost-in-the-middle):
1. Construir un ``MemoryStore`` aislado para la archival.
2. Volcar los ``ctxs`` en Archival vía ``insert_archival`` (sin LLM).
3. Construir un agente con el system prompt del apéndice 6.1.4 (DOC-QA
   bot, "the year is 2018").
4. Lanzar la pregunta envuelta en el prompt-de-query del apéndice 6.1.4
   ("Search your archival memory… Format your response with 'ANSWER:
   [YOUR ANSWER], DOCUMENT: [ARCHIVAL MEMORY TEXT]'").
5. Puntuar con:
   - **LLM-judge** (apéndice 6.1.5): CORRECT/INCORRECT con un único
     token, contrato estricto sobre el formato ANSWER/DOCUMENT.
   - **Exact-match laxo** (sanity): cualquier ``answer`` aparece como
     substring case-insensitive del campo ANSWER. No reemplaza al juez,
     pero detecta regresiones del agente sin gastar tokens del juez.

Para el baseline (control sin archival): el system prompt incluye los
top-K ``ctxs`` y la pregunta usa el prompt "references" del apéndice
6.1.4 — el LLM tiene que extraer la respuesta del prompt sin tools,
exactamente lo que el paper compara contra (Figura 5).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph

from ..agent import build_agent
from ..memory_store import InMemoryStore, MemoryStore

# ---------------------------------------------------------------------------
# Prompts del apéndice 6.1.4
# ---------------------------------------------------------------------------

DOC_QA_ASSISTANT_PROMPT = (
    "You are MemGPT DOC-QA bot.  Your job is to answer questions about "
    "documents that are stored in your archival memory.  The answer to the "
    "users question will ALWAYS be in your archival memory, so remember to "
    "keep searching if you can't find the answer.\n"
    "Answer the questions as if though the year is 2018."
)

DOC_QA_QUERY_TEMPLATE = (
    "Search your archival memory to answer the provided question.  Provide "
    "both the answer and the archival memory result from which you "
    "determined your answer.  Format your response with the format "
    "'ANSWER: [YOUR ANSWER], DOCUMENT: [ARCHIVAL MEMORY TEXT]'. Your task "
    "is to answer the question:\n{question}"
)

BASELINE_QUERY_TEMPLATE = (
    "Answer the question provided according to the list of documents below "
    "(some of which might be irrelevant).  In your response, provide both "
    "the answer and the document text from which you determined the answer. "
    " Format your response with the format 'ANSWER: <YOUR ANSWER>, DOCUMENT: "
    "[DOCUMENT TEXT]'. If none of the documents provided have the answer to "
    "the question, reply with 'INSUFFICIENT INFORMATION'. Do NOT provide an "
    "answer if you cannot find it in the provided documents.\n"
    "Your response will only be considered correct if you provide both the "
    "answer and relevant document text, or say 'INSUFFICIENT INFORMATION'.  "
    "Answer the question as if though the current year is 2018.\n\n"
    "DOCUMENTS:\n{documents}\n\n"
    "QUESTION: {question}"
)

# ---------------------------------------------------------------------------
# LLM-judge — apéndice 6.1.5
# ---------------------------------------------------------------------------

JUDGE_PROMPT_TEMPLATE = (
    "Your task is to evaluate whether an LLM correct answered a question.  "
    "The LLM response should be the format \"ANSWER: [answer], DOCUMENT: "
    "[document_text]\" or say \"INSUFFICIENT INFORMATION\".\n"
    "The true answer is provided in the format \"TRUE ANSWER:[list of "
    "possible answers]\".  The questions is provided in the format "
    "\"QUESTION: [question]\".\n"
    "If the LLM response contains both the correct answer and corresponding "
    "document text, the response is correct.\n"
    "Even if the LLM's answer and the true answer are slightly different in "
    "wording, the response is still correct.  For example, if the answer is "
    "more specific than the true answer or uses a different phrasing that "
    "is still correct, the response is correct.\n"
    "If the LLM response if \"INSUFFICIENT INFORMATION\", or the \"DOCUMENT\" "
    "field is missing, the response is incorrect.\n"
    "Respond with a single token:  \"CORRECT\" or \"INCORRECT\".\n\n"
    "QUESTION: {question}\n"
    "TRUE ANSWER: {gold_answers}\n"
    "LLM RESPONSE: {generated_answer}"
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Document:
    """Pasaje de Wikipedia (o equivalente) recuperado para una pregunta.

    El campo ``has_answer`` viene directamente del dataset estilo "Lost in
    the middle" (clave ``hasanswer``). Solo lo usamos para diagnóstico y
    para el baseline: NO se le pasa al agente para no filtrar la tarea.
    """

    title: str
    text: str
    has_answer: bool = False

    def as_archival_text(self) -> str:
        if self.title:
            return f"[{self.title}] {self.text}"
        return self.text


@dataclass(frozen=True)
class DocumentSample:
    """Una pregunta + sus respuestas gold + el corpus de documentos asociado."""

    sample_id: int
    question: str
    gold_answers: tuple[str, ...]
    documents: tuple[Document, ...]


def _parse_document(raw: dict) -> Document:
    return Document(
        title=str(raw.get("title") or "").strip(),
        text=str(raw.get("text") or "").strip(),
        has_answer=bool(raw.get("hasanswer", raw.get("has_answer", False))),
    )


def parse_record(record: dict, *, sample_id: int) -> DocumentSample:
    """Convierte un registro JSON estilo DPR/lost-in-the-middle en ``DocumentSample``.

    Acepta dos campos para los documentos por compatibilidad: ``ctxs`` (DPR
    / Liu 2023a) o ``documents`` (esquema más explícito). El campo
    ``answers`` sigue el formato NaturalQuestions-Open: lista de respuestas
    gold aceptables.
    """
    question = str(record.get("question") or "").strip()
    if not question:
        raise ValueError(f"sample {sample_id} missing 'question'")

    answers = record.get("answers") or record.get("answer") or []
    if isinstance(answers, str):
        answers = [answers]
    gold_answers = tuple(a for a in (str(x).strip() for x in answers) if a)
    if not gold_answers:
        raise ValueError(f"sample {sample_id} has no gold 'answers'")

    raw_docs = record.get("ctxs") or record.get("documents") or []
    documents = tuple(_parse_document(d) for d in raw_docs if isinstance(d, dict))

    return DocumentSample(
        sample_id=sample_id,
        question=question,
        gold_answers=gold_answers,
        documents=documents,
    )


def load_dataset(
    path: Path,
    *,
    limit: int | None = None,
) -> list[DocumentSample]:
    """Carga un JSONL (o JSONL.gz) de NQ-Open / lost-in-the-middle a samples."""
    import gzip

    opener = gzip.open if path.suffix == ".gz" else open
    samples: list[DocumentSample] = []
    with opener(path, "rt", encoding="utf-8") as fp:
        for i, line in enumerate(fp):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            samples.append(parse_record(record, sample_id=i))
    return samples


def download_dataset(
    target: Path,
    *,
    repo_id: str = "MemGPT/qa_data",
    filename: str = "nq-open-30_total_documents_gold_at_14.jsonl.gz",
    repo_type: str = "dataset",
) -> Path:
    """Descarga el dataset desde HF Hub si no está ya en ``target``.

    Idempotente: si ``target`` existe se devuelve sin tocar la red. Por
    defecto apunta a ``MemGPT/qa_data`` (mirror oficial del equipo MemGPT
    de los splits de Liu et al. 2023a — 30 docs, gold en posición 14, el
    caso "lost in the middle" más interesante). Otras opciones del repo:
    ``nq-open-30_total_documents_gold_at_{0,4,9,19,24,29}.jsonl.gz``.

    Si ``target`` termina en ``.jsonl`` (no ``.gz``) descomprimimos en el
    sitio para que ``load_dataset`` lo abra directo.
    """
    if target.exists():
        return target
    import gzip
    import shutil

    from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]

    target.parent.mkdir(parents=True, exist_ok=True)
    cached = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
    )
    if filename.endswith(".gz") and target.suffix != ".gz":
        with gzip.open(cached, "rb") as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    else:
        shutil.copyfile(cached, target)
    return target


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------


_ANSWER_RE = re.compile(r"ANSWER\s*:\s*(.+?)(?:,\s*DOCUMENT\s*:|$)", re.IGNORECASE | re.DOTALL)
_DOCUMENT_RE = re.compile(r"DOCUMENT\s*:\s*(.+)$", re.IGNORECASE | re.DOTALL)
_INSUFFICIENT_RE = re.compile(r"INSUFFICIENT\s+INFORMATION", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedAnswer:
    answer: str | None
    document: str | None
    insufficient: bool

    @property
    def has_both_fields(self) -> bool:
        return bool(self.answer) and bool(self.document)


def parse_response(text: str) -> ParsedAnswer:
    """Parsea la respuesta del agente al formato ``ANSWER: X, DOCUMENT: Y``.

    Devuelve los campos extraídos (o ``None`` si faltan) y un flag para la
    abstención explícita ``INSUFFICIENT INFORMATION``. La regex es laxa con
    el separador (coma, salto de línea, dos puntos) porque los LLMs suelen
    escapar ligeramente del formato exigido.
    """
    if not text:
        return ParsedAnswer(answer=None, document=None, insufficient=False)
    insufficient = bool(_INSUFFICIENT_RE.search(text))
    answer_m = _ANSWER_RE.search(text)
    document_m = _DOCUMENT_RE.search(text)
    answer = answer_m.group(1).strip() if answer_m else None
    document = document_m.group(1).strip() if document_m else None
    return ParsedAnswer(answer=answer or None, document=document or None, insufficient=insufficient)


def exact_match(parsed: ParsedAnswer, gold_answers: tuple[str, ...]) -> bool:
    """¿Alguna respuesta gold aparece como substring del campo ANSWER?

    Métrica laxa para sanity-check (no reemplaza al juez). Comparación
    case-insensitive y sin puntuación final, idéntica a la de NQ-Open.
    """
    if parsed.insufficient or not parsed.answer:
        return False
    candidate = parsed.answer.lower().strip().rstrip(".!?,;:")
    for gold in gold_answers:
        if gold.lower().strip() in candidate:
            return True
    return False


_VERDICT_RE = re.compile(r"\b(CORRECT|INCORRECT)\b")


def parse_judge_verdict(text: str) -> bool | None:
    """Devuelve ``True`` si CORRECT, ``False`` si INCORRECT, ``None`` si ambiguo.

    El prompt del apéndice 6.1.5 pide *single token*: si el juez se pasa y
    suelta ambos términos, abstenemos en lugar de elegir el primero —
    cuenta como sample sin juicio. Match case-insensitive: aceptamos
    ``correct``/``Correct`` indistintamente.

    Atención: ``INCORRECT`` contiene ``CORRECT`` como substring, así que
    usamos ``\\b`` y comprobamos el conjunto único de matches.
    """
    matches = _VERDICT_RE.findall((text or "").upper())
    if not matches:
        return None
    unique = set(matches)
    if unique == {"CORRECT"}:
        return True
    if unique == {"INCORRECT"}:
        return False
    return None  # ambos presentes ⇒ ambiguo.


JudgeCallable = Callable[[str, tuple[str, ...], str], tuple[bool | None, str]]
"""``(question, gold_answers, generated) → (verdict|None, raw_text)``."""


def _normalise_litellm_model(model_id: str) -> str:
    if "/" in model_id or ":" not in model_id:
        return model_id
    provider, _, name = model_id.partition(":")
    return f"{provider}/{name}"


def default_judge(model_id: str | None = None) -> JudgeCallable:
    """Construye el LLM-judge del apéndice 6.1.5 (litellm, temperatura 0.0)."""
    from litellm import completion as _litellm_completion

    from ..config import get_settings

    settings = get_settings()
    model = _normalise_litellm_model(model_id or settings.primary_llm_model)

    def _judge(
        question: str, gold_answers: tuple[str, ...], generated: str
    ) -> tuple[bool | None, str]:
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            gold_answers=list(gold_answers),
            generated_answer=generated,
        )
        response = _litellm_completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        raw = response.choices[0].message.content or ""
        return parse_judge_verdict(raw), raw

    return _judge


# ---------------------------------------------------------------------------
# Ingesta a Archival
# ---------------------------------------------------------------------------


def populate_archival(
    store: MemoryStore,
    sample: DocumentSample,
    *,
    occurred_at: datetime | None = None,
) -> int:
    """Vuelca los ``documents`` del sample en Archival.

    Cada documento se inserta como una entrada de archival con
    ``[title] text``. Devuelve el número de docs ingeridos para que el
    runner pueda dejar traza en el resultado.

    Coste por documento depende del backend:
    - ``InMemoryStore``: barato (solo embedding local opcional).
    - ``GraphitiStore``: caro — cada ``insert_archival`` dispara
      extracción de entidades/relaciones por LLM dentro de Graphiti,
      por lo que un sample con 30 docs puede tardar varios minutos.
    """
    when = occurred_at or datetime.now(timezone.utc)
    count = 0
    for doc in sample.documents:
        if not doc.text:
            continue
        store.insert_archival(content=doc.as_archival_text(), occurred_at=when)
        count += 1
    return count


def populate_archival_from_corpus(
    store: MemoryStore,
    corpus: list[Document],
    *,
    occurred_at: datetime | None = None,
) -> int:
    """Carga un corpus global en un único Archival.

    Hook explícito para el modo "20M Wikipedia": pre-cargas la BD una sola
    vez fuera del bucle de samples y reutilizas la misma store entre
    preguntas. La implementación es idéntica a ``populate_archival`` pero
    con la firma orientada a un dataset compartido.
    """
    when = occurred_at or datetime.now(timezone.utc)
    count = 0
    for doc in corpus:
        if not doc.text:
            continue
        store.insert_archival(content=doc.as_archival_text(), occurred_at=when)
        count += 1
    return count


# ---------------------------------------------------------------------------
# Construcción del agente DOC-QA
# ---------------------------------------------------------------------------


def build_doc_qa_agent(
    store: MemoryStore,
    *,
    model_id: str | None = None,
) -> CompiledStateGraph:
    """Agente con el preprompt del apéndice 6.1.4 + memoria llena."""
    return build_agent(
        system_prompt=DOC_QA_ASSISTANT_PROMPT,
        memory_store=store,
        model_id=model_id,
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass
class DocQAResult:
    sample_id: int
    question: str
    gold_answers: tuple[str, ...]
    predicted_raw: str
    predicted_answer: str | None
    predicted_document: str | None
    insufficient: bool
    exact_match: bool
    judge_correct: bool | None
    judge_raw: str
    elapsed_seconds: float
    chained_heartbeats: int
    archival_search_calls: int
    documents_ingested: int


@dataclass
class DocQASummary:
    total: int
    judged: int
    correct: int
    accuracy: float
    exact_match_rate: float
    insufficient_rate: float
    mean_elapsed_seconds: float
    mean_archival_searches: float
    results: list[DocQAResult] = field(default_factory=list)


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
    msg = str(exc).lower()
    if "insufficient_quota" in msg or "exceeded your current quota" in msg:
        return False
    return "rate_limit" in msg or "rate limit" in msg or " 429" in msg or "429 " in msg


def _suggested_backoff(exc: BaseException, attempt: int) -> float:
    m = _RETRY_AFTER_RE.search(str(exc))
    if m:
        return float(m.group(1)) + 1.0
    return min(60.0, 2.0**attempt)


def _invoke_with_retry(
    agent: CompiledStateGraph,
    payload: dict,
    config: dict,
    *,
    max_retries: int = 6,
) -> Any:
    for attempt in range(max_retries):
        try:
            return agent.invoke(payload, config=config)
        except Exception as exc:
            if not _is_rate_limit(exc) or attempt == max_retries - 1:
                raise
            delay = _suggested_backoff(exc, attempt)
            print(f"[rate-limit] sleeping {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)
    raise RuntimeError("retry loop exited without returning")


def run_sample(
    sample: DocumentSample,
    *,
    agent: CompiledStateGraph,
    judge: JudgeCallable,
    documents_ingested: int = 0,
    query_template: str = DOC_QA_QUERY_TEMPLATE,
) -> DocQAResult:
    """Ejecuta la pregunta DOC-QA sobre ``agent`` y puntúa el resultado."""
    user_message = query_template.format(question=sample.question)
    payload: dict[str, Any] = {"messages": [HumanMessage(content=user_message)]}
    config = {"configurable": {"thread_id": f"doc-qa-s{sample.sample_id}"}}

    start = time.perf_counter()
    out = _invoke_with_retry(agent, payload, config)
    elapsed = time.perf_counter() - start

    messages = out.get("messages", []) if isinstance(out, dict) else []
    final = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
    predicted_raw = ""
    if final is not None:
        raw = final.content
        predicted_raw = raw if isinstance(raw, str) else str(raw or "")

    parsed = parse_response(predicted_raw)
    em = exact_match(parsed, sample.gold_answers)
    verdict, judge_raw = judge(sample.question, sample.gold_answers, predicted_raw)

    return DocQAResult(
        sample_id=sample.sample_id,
        question=sample.question,
        gold_answers=sample.gold_answers,
        predicted_raw=predicted_raw,
        predicted_answer=parsed.answer,
        predicted_document=parsed.document,
        insufficient=parsed.insufficient,
        exact_match=em,
        judge_correct=verdict,
        judge_raw=judge_raw,
        elapsed_seconds=elapsed,
        chained_heartbeats=int(out.get("chained_heartbeats", 0)) if isinstance(out, dict) else 0,
        archival_search_calls=_count_archival_calls(messages),
        documents_ingested=documents_ingested,
    )


AgentFactory = Callable[[DocumentSample, MemoryStore], CompiledStateGraph]
StoreFactory = Callable[[DocumentSample], MemoryStore]


def default_store_factory(_sample: DocumentSample) -> MemoryStore:
    """Factory por defecto: InMemoryStore sin embedder (substring search).

    Esto deja los tests deterministas. Para Document QA real construye el
    factory con ``make_store_factory(embedder=...)`` o pasa el flag
    ``--embedder local|openai`` al runner CLI.
    """
    return InMemoryStore()


def make_store_factory(
    embedder: Callable[[str], list[float]] | None,
) -> StoreFactory:
    """Factory configurable: cada sample obtiene un InMemoryStore que
    comparte el mismo ``embedder`` (carga del modelo una sola vez)."""

    def _factory(_sample: DocumentSample) -> MemoryStore:
        return InMemoryStore(embedder=embedder)

    return _factory


def make_default_agent_factory(model_id: str | None = None) -> AgentFactory:
    def _factory(_sample: DocumentSample, store: MemoryStore) -> CompiledStateGraph:
        return build_doc_qa_agent(store, model_id=model_id)

    return _factory


def run_benchmark(
    samples: list[DocumentSample],
    *,
    judge: JudgeCallable,
    agent_factory: AgentFactory | None = None,
    store_factory: StoreFactory = default_store_factory,
    on_result: Callable[[DocQAResult], None] | None = None,
    sleep_between_seconds: float = 0.0,
    shared_store: MemoryStore | None = None,
) -> DocQASummary:
    """Ejecuta el benchmark MemGPT sobre ``samples`` y devuelve el resumen.

    Por defecto cada sample tiene su propio Archival (modo "lost-in-the-
    middle"). Si pasas ``shared_store`` se reutiliza esa instancia entre
    samples (modo "corpus global"): NO se llama ni a ``populate_archival``
    ni a ``store_factory`` — asumimos que el caller ya cargó la corpus
    completa antes de invocar al runner. Útil cuando el corpus son los
    20M artículos de Wikipedia y montarlo cuesta horas.
    """
    factory = agent_factory or make_default_agent_factory()
    results: list[DocQAResult] = []
    for sample in samples:
        if shared_store is not None:
            store = shared_store
            ingested = 0  # ya estaba cargado fuera del runner.
            owned_store = False
        else:
            store = store_factory(sample)
            ingested = populate_archival(store, sample)
            owned_store = True
        try:
            agent = factory(sample, store)
            r = run_sample(
                sample,
                agent=agent,
                judge=judge,
                documents_ingested=ingested,
            )
            results.append(r)
            if on_result is not None:
                on_result(r)
            if sleep_between_seconds > 0:
                time.sleep(sleep_between_seconds)
        finally:
            if owned_store:
                store.close()

    return _summarize(results)


def _summarize(results: list[DocQAResult]) -> DocQASummary:
    total = len(results)
    judged = sum(1 for r in results if r.judge_correct is not None)
    correct = sum(1 for r in results if r.judge_correct is True)
    accuracy = correct / judged if judged else 0.0
    em_rate = sum(1 for r in results if r.exact_match) / total if total else 0.0
    insufficient_rate = sum(1 for r in results if r.insufficient) / total if total else 0.0
    elapsed_mean = sum(r.elapsed_seconds for r in results) / total if total else 0.0
    archival_mean = (
        sum(r.archival_search_calls for r in results) / total if total else 0.0
    )
    return DocQASummary(
        total=total,
        judged=judged,
        correct=correct,
        accuracy=accuracy,
        exact_match_rate=em_rate,
        insufficient_rate=insufficient_rate,
        mean_elapsed_seconds=elapsed_mean,
        mean_archival_searches=archival_mean,
        results=results,
    )


# ---------------------------------------------------------------------------
# Baseline (sin archival): top-K docs en el system prompt
# ---------------------------------------------------------------------------


BASELINE_DOC_QA_ASSISTANT = (
    "You are MemGPT DOC-QA bot.  Your job is to answer questions about "
    "documents.  Answer the questions as if though the year is 2018."
)


def format_baseline_documents(
    documents: tuple[Document, ...],
    *,
    top_k: int | None = None,
) -> str:
    """Serializa los documentos como bloque numerado para el baseline.

    Replica el modo "K documentos retrieved" del paper (Liu et al. 2023a):
    los pasajes llegan en el orden del retriever (no se reordena). Si
    ``top_k`` es ``None`` se usan todos; si es int, se truncan los
    primeros K (los más relevantes según el retriever).
    """
    docs = documents if top_k is None else documents[:top_k]
    if not docs:
        return "(no documents available)"
    parts: list[str] = []
    for i, d in enumerate(docs, start=1):
        header = f"Document [{i}]"
        if d.title:
            header += f" (Title: {d.title})"
        parts.append(f"{header}\n{d.text}")
    return "\n\n".join(parts)


BaselineAgentFactory = Callable[[DocumentSample], CompiledStateGraph]


def make_baseline_agent_factory(
    model_id: str | None = None,
) -> BaselineAgentFactory:
    """Construye agentes baseline (sin tools de memoria) por sample."""

    def _factory(_sample: DocumentSample) -> CompiledStateGraph:
        return build_agent(
            system_prompt=BASELINE_DOC_QA_ASSISTANT,
            memory_store=None,
            model_id=model_id,
        )

    return _factory


def run_baseline_sample(
    sample: DocumentSample,
    *,
    agent: CompiledStateGraph,
    judge: JudgeCallable,
    top_k: int | None = None,
) -> DocQAResult:
    """Versión baseline de ``run_sample``: docs incrustados en el prompt-de-query."""
    documents_block = format_baseline_documents(sample.documents, top_k=top_k)
    user_message = BASELINE_QUERY_TEMPLATE.format(
        documents=documents_block,
        question=sample.question,
    )
    payload: dict[str, Any] = {"messages": [HumanMessage(content=user_message)]}
    config = {"configurable": {"thread_id": f"doc-qa-baseline-s{sample.sample_id}"}}

    start = time.perf_counter()
    out = _invoke_with_retry(agent, payload, config)
    elapsed = time.perf_counter() - start

    messages = out.get("messages", []) if isinstance(out, dict) else []
    final = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
    predicted_raw = ""
    if final is not None:
        raw = final.content
        predicted_raw = raw if isinstance(raw, str) else str(raw or "")

    parsed = parse_response(predicted_raw)
    em = exact_match(parsed, sample.gold_answers)
    verdict, judge_raw = judge(sample.question, sample.gold_answers, predicted_raw)

    return DocQAResult(
        sample_id=sample.sample_id,
        question=sample.question,
        gold_answers=sample.gold_answers,
        predicted_raw=predicted_raw,
        predicted_answer=parsed.answer,
        predicted_document=parsed.document,
        insufficient=parsed.insufficient,
        exact_match=em,
        judge_correct=verdict,
        judge_raw=judge_raw,
        elapsed_seconds=elapsed,
        chained_heartbeats=int(out.get("chained_heartbeats", 0)) if isinstance(out, dict) else 0,
        archival_search_calls=0,  # baseline no tiene archival.
        documents_ingested=len(sample.documents) if top_k is None else min(top_k, len(sample.documents)),
    )


def run_baseline_benchmark(
    samples: list[DocumentSample],
    *,
    judge: JudgeCallable,
    agent_factory: BaselineAgentFactory | None = None,
    top_k: int | None = None,
    on_result: Callable[[DocQAResult], None] | None = None,
    sleep_between_seconds: float = 0.0,
) -> DocQASummary:
    """Baseline: top-K documentos en el system prompt, sin archival.

    Reproduce los puntos de la Figura 5 del paper (GPT-4 con K=10/20/30
    documentos): el LLM tiene que elegir el doc relevante leyendo el
    prompt entero. El paper muestra que la accuracy cae con K alto por el
    efecto "lost in the middle"; MemGPT no sufre eso porque pagina la
    archival bajo demanda.
    """
    factory = agent_factory or make_baseline_agent_factory()
    results: list[DocQAResult] = []
    for sample in samples:
        agent = factory(sample)
        r = run_baseline_sample(sample, agent=agent, judge=judge, top_k=top_k)
        results.append(r)
        if on_result is not None:
            on_result(r)
        if sleep_between_seconds > 0:
            time.sleep(sleep_between_seconds)
    return _summarize(results)
