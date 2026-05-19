"""Deep Memory Retrieval (DMR) benchmark — Fase 9, §3.1.1 + apéndices 6.1.1/6.1.2.

Evalúa la capacidad de MemGPT para recordar hechos de **conversaciones
anteriores** (sesiones 1-5) cuando se le formula una pregunta en una sesión
nueva (sesión 6). El dataset es ``MemGPT/MSC-Self-Instruct`` (Hugging Face),
una extensión auto-instruida del Multi-Session Chat de Xu et al. (2021) con
una pregunta + respuesta gold por par de personas.

Pipeline por par:
1. Construir un ``MemoryStore`` aislado (Recall + Archival).
2. Volcar las 5 sesiones (4 ``previous_dialogs`` + el ``dialog`` final) en
   Recall directamente vía ``persist_message`` — equivalente a haber
   conversado en otro contexto. Cada sesión se ancla en su propio
   ``occurred_at`` derivado de ``time_back`` para que las búsquedas con
   filtro temporal funcionen.
3. Construir un agente fresco con la persona de ``Speaker 1`` cargada en
   Core Memory y el system prompt del apéndice 6.1.1.
4. Lanzar la pregunta DMR (``self_instruct['B']``) como ``HumanMessage`` en
   un nuevo thread (sesión 6 simulada). El agente debe usar
   ``conversation_search`` para recuperar el dato y responder.
5. Puntuar:
   - **ROUGE-L recall** entre la respuesta y ``self_instruct['A']``.
   - **LLM-as-judge** con el prompt del apéndice 6.1.2.

Para el baseline (control): no hay tools de memoria; el system prompt
contiene el resumen lossy de las sesiones (``summary_speaker_*``) tal como
describe el apéndice 6.1.1.
"""

from __future__ import annotations

import json
import re
import string
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph

from ..agent import build_agent
from ..core_memory import default_core_memory
from ..memory_store import InMemoryStore, MemoryStore
from ..state import MemGPTState

# ---------------------------------------------------------------------------
# Prompts del apéndice 6.1.1
# ---------------------------------------------------------------------------

DMR_ASSISTANT_PROMPT = (
    "The following is information about myself. My task is to completely "
    "immerse myself in this role (I should never say that I am an AI, and "
    "should reply as if I am playing this role).\n"
    "If the user asks me a question, I should reply with a best guess using "
    "the information in core memory and conversation_search."
)

BASELINE_PREPROMPT_TEMPLATE = (
    "Your task is to answer a question from the user about your prior "
    "conversations.\n"
    "The following is a summary of all your prior conversations:\n"
    "{conversation_summary}\n"
    "Answer from the perspective of the persona provided (do not say that "
    "you are an AI assistant).\n"
    "If you do not have enough information to answer the question, reply 'NO "
    "ANSWER'. Either reply with the answer, or reply 'NO ANSWER', do not say "
    "anything else."
)

# ---------------------------------------------------------------------------
# LLM-judge — apéndice 6.1.2
# ---------------------------------------------------------------------------

JUDGE_PROMPT_TEMPLATE = (
    "Your task is to label an answer to a question as 'CORRECT' or 'WRONG'.\n"
    "You will be given the following data:\n"
    "(1) a question (posed by one user to another user), (2) a 'gold' (ground "
    "truth) answer, (3) a generated answer which you will score as CORRECT/WRONG.\n"
    "The point of the question is to ask about something one user should know "
    "about the other user based on their prior conversations.\n"
    "The gold answer will usually be a concise and short answer that includes "
    "the referenced topic, for example:\n"
    "Question: Do you remember what I got the last time I went to Hawaii?\n"
    "Gold answer: A shell necklace\n"
    "The generated answer might be much longer, but you should be generous with "
    "your grading - as long as it touches on the same topic as the gold answer, "
    "it should be counted as CORRECT.\n"
    "For example, the following answers would be considered CORRECT:\n"
    "Generated answer (CORRECT): Oh yeah, that was so fun! I got so much stuff "
    "there, including that shell necklace.\n"
    "Generated answer (CORRECT): I got a ton of stuff... that surfboard, the mug, "
    "the necklace, those coasters too..\n"
    "Generated answer (CORRECT): That cute necklace\n"
    "The following answers would be considered WRONG:\n"
    "Generated answer (WRONG): Oh yeah, that was so fun! I got so much stuff there, "
    "including that mug.\n"
    "Generated answer (WRONG): I got a ton of stuff... that surfboard, the mug, "
    "the necklace, those coasters too..\n"
    "Generated answer (WRONG): I'm sorry, I don't remember what you're talking "
    "about.\n"
    "Now it's time for the real question:\n"
    "Question: {question}\n"
    "Gold answer: {gold_answer}\n"
    "Generated answer: {generated_answer}\n"
    "First, provide a short (one sentence) explanation of your reasoning, then "
    "finish with CORRECT or WRONG. Do NOT include both CORRECT and WRONG in "
    "your response, or it will break the evaluation script."
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DialogTurn:
    speaker: str  # "Speaker 1" / "Speaker 2"
    text: str


@dataclass(frozen=True)
class Session:
    turns: tuple[DialogTurn, ...]
    days_ago: float  # cuántos días atrás respecto a la sesión 6 (0 para la 5).


@dataclass(frozen=True)
class DMRSample:
    """Un par de personas con sus 5 sesiones y la pregunta DMR.

    El agente actúa como ``agent_speaker`` (por defecto ``Speaker 1``) — la
    persona a la que la pregunta del DMR está dirigida.
    """

    sample_id: int
    agent_speaker: str
    other_speaker: str
    persona_agent: tuple[str, ...]
    persona_other: tuple[str, ...]
    summary_agent: tuple[str, ...]  # último resumen lossy disponible.
    summary_other: tuple[str, ...]
    sessions: tuple[Session, ...]  # sesiones 1..5 en orden cronológico.
    question: str
    gold_answer: str


_TIME_UNIT_DAYS = {
    "hour": 1.0 / 24.0,
    "hours": 1.0 / 24.0,
    "day": 1.0,
    "days": 1.0,
    "week": 7.0,
    "weeks": 7.0,
    "month": 30.0,
    "months": 30.0,
    "year": 365.0,
    "years": 365.0,
}


def _time_back_to_days(time_num: Any, time_unit: Any) -> float:
    """Convierte ``time_num`` + ``time_unit`` a días (float).

    El dataset trae unidades dispares ("days", "hours", "months", ...). Si la
    unidad no se reconoce, devolvemos 0.0 y dejamos que el caller acumule el
    delta — el orden cronológico se mantiene aunque la magnitud sea
    aproximada.
    """
    try:
        n = float(time_num)
    except (TypeError, ValueError):
        return 0.0
    unit = str(time_unit or "").lower().strip()
    return n * _TIME_UNIT_DAYS.get(unit, 1.0)


def _alternating_turns(
    raw_turns: Iterable[dict],
    *,
    first_speaker: str = "Speaker 1",
) -> tuple[DialogTurn, ...]:
    """Asigna speakers alternando si el dato no los trae explícitos.

    Las 4 sesiones previas (``previous_dialogs``) solo guardan ``{'text': ...}``
    — el dataset MSC asume estricta alternancia desde Speaker 1.
    """
    speakers = [first_speaker, "Speaker 2" if first_speaker == "Speaker 1" else "Speaker 1"]
    out: list[DialogTurn] = []
    for i, turn in enumerate(raw_turns):
        text = (turn.get("text") or "").strip()
        if not text:
            continue
        sid = turn.get("id") or speakers[i % 2]
        out.append(DialogTurn(speaker=str(sid), text=text))
    return tuple(out)


def parse_record(
    record: dict,
    *,
    sample_id: int,
    agent_speaker: str = "Speaker 1",
) -> DMRSample:
    """Convierte un registro JSON del dataset MSC-Self-Instruct en ``DMRSample``.

    Asumimos que ``self_instruct`` tiene la forma ``{'A': gold, 'B': question}``
    y que el speaker que responde (``A``) coincide con ``agent_speaker``. La
    lógica funciona para ambas asignaciones porque la pregunta siempre va de
    `B → A`.
    """
    other_speaker = "Speaker 2" if agent_speaker == "Speaker 1" else "Speaker 1"
    speaker_idx = 0 if agent_speaker == "Speaker 1" else 1

    personas: list[list[str]] = record["personas"]
    persona_agent = tuple(personas[speaker_idx])
    persona_other = tuple(personas[1 - speaker_idx])

    # El último update de persona disponible es el más rico. El dataset trae
    # `personas`, `personas_update1`, `personas_update2` — usamos el último no
    # vacío. Para resumen lossy usamos `summary_speaker_*[-1]`.
    summary_agent_list = record.get(f"summary_speaker_{speaker_idx + 1}") or []
    summary_other_list = record.get(f"summary_speaker_{2 - speaker_idx}") or []
    summary_agent = tuple(summary_agent_list[-1]) if summary_agent_list else ()
    summary_other = tuple(summary_other_list[-1]) if summary_other_list else ()

    sessions: list[Session] = []
    # `previous_dialogs` está en orden cronológico (sesión 1 → sesión 4) según
    # MSC. ``time_num/time_unit`` describe cuánto tiempo pasó **antes** de la
    # próxima sesión, así que para anclar cada sesión en el pasado acumulamos.
    prevs: list[dict] = list(record.get("previous_dialogs") or [])
    # Empezamos con la sesión 5 (current `dialog`) ⇒ days_ago=0.
    # Las sesiones previas se sitúan retrocediendo.
    cumulative = 0.0
    cumulative_per_session: list[float] = []
    # Calculamos cuántos días atrás respecto a la sesión 5 está cada sesión
    # previa: la sesión `prevs[-1]` está a `time_back(prevs[-1])` días, la
    # `prevs[-2]` a `time_back(prevs[-1]) + time_back(prevs[-2])`, etc.
    for prev in reversed(prevs):
        cumulative += _time_back_to_days(prev.get("time_num"), prev.get("time_unit"))
        cumulative_per_session.append(cumulative)
    cumulative_per_session.reverse()  # alinear con orden cronológico.

    for prev, days_ago in zip(prevs, cumulative_per_session):
        sessions.append(
            Session(
                turns=_alternating_turns(prev.get("dialog") or []),
                days_ago=days_ago,
            )
        )

    sessions.append(
        Session(
            turns=_alternating_turns(record.get("dialog") or []),
            days_ago=0.0,
        )
    )

    si = record.get("self_instruct") or {}
    # 'B' es la pregunta (la hace el otro), 'A' es la respuesta gold (la del
    # agente). El dataset es consistente en ese mapping.
    question = (si.get("B") or "").strip()
    gold_answer = (si.get("A") or "").strip()
    if not question or not gold_answer:
        raise ValueError(
            f"sample {sample_id} missing self_instruct A/B (got {list(si)!r})"
        )

    return DMRSample(
        sample_id=sample_id,
        agent_speaker=agent_speaker,
        other_speaker=other_speaker,
        persona_agent=persona_agent,
        persona_other=persona_other,
        summary_agent=summary_agent,
        summary_other=summary_other,
        sessions=tuple(sessions),
        question=question,
        gold_answer=gold_answer,
    )


def load_dataset(
    path: Path,
    *,
    limit: int | None = None,
    agent_speaker: str = "Speaker 1",
) -> list[DMRSample]:
    """Carga ``msc_self_instruct.jsonl`` en una lista de ``DMRSample``.

    Si ``limit`` está, corta la lista (orden del fichero, que es estable). El
    fichero se descarga manualmente vía ``huggingface_hub`` o ``curl``; ver
    ``download_dataset`` para un atajo.
    """
    samples: list[DMRSample] = []
    with path.open("r", encoding="utf-8") as fp:
        for i, line in enumerate(fp):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            samples.append(parse_record(record, sample_id=i, agent_speaker=agent_speaker))
    return samples


def download_dataset(target: Path) -> Path:
    """Descarga ``msc_self_instruct.jsonl`` si no está ya en ``target``.

    Usa ``huggingface_hub.hf_hub_download`` para evitar reimplementar caché y
    auth. La función es idempotente: si el fichero existe en ``target`` lo
    devuelve sin tocar la red.

    HF Hub devuelve un symlink al blob cacheado; copiamos el contenido real
    al destino para que ``target`` sea independiente del cache de HF (el
    cache puede caducar o moverse y no queremos romper el dataset).
    """
    if target.exists():
        return target
    import shutil

    from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]

    target.parent.mkdir(parents=True, exist_ok=True)
    cached = hf_hub_download(
        repo_id="MemGPT/MSC-Self-Instruct",
        filename="msc_self_instruct.jsonl",
        repo_type="dataset",
    )
    shutil.copyfile(cached, target)
    return target


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------


_PUNCT_RE = re.compile(rf"[{re.escape(string.punctuation)}]")


def _tokenize(text: str) -> list[str]:
    text = _PUNCT_RE.sub(" ", text.lower())
    return [t for t in text.split() if t]


def rouge_l_recall(reference: str, candidate: str) -> float:
    """ROUGE-L recall = |LCS(ref, cand)| / |ref|.

    Implementación pura para evitar la dependencia de ``rouge_score``. La
    métrica es la del paper (apéndice 3.1) — usa **recall** porque las
    respuestas de MemGPT suelen ser más verbosas que el gold.
    """
    ref_tokens = _tokenize(reference)
    cand_tokens = _tokenize(candidate)
    if not ref_tokens:
        return 0.0
    if not cand_tokens:
        return 0.0

    n, m = len(ref_tokens), len(cand_tokens)
    # DP roll-array O(min(n,m)) memoria.
    if m < n:
        ref_tokens, cand_tokens = cand_tokens, ref_tokens
        n, m = m, n
    prev = [0] * (n + 1)
    for j in range(1, m + 1):
        cur = [0] * (n + 1)
        cj = cand_tokens[j - 1]
        for i in range(1, n + 1):
            if ref_tokens[i - 1] == cj:
                cur[i] = prev[i - 1] + 1
            else:
                cur[i] = max(cur[i - 1], prev[i])
        prev = cur
    lcs = prev[n]
    # ``ref_tokens`` puede haber sido swappeado: la longitud del original sigue
    # siendo la del más corto, que es lo que toca como denominador del
    # recall **respecto a la referencia**. Corregimos:
    return lcs / len(_tokenize(reference))


_VERDICT_RE = re.compile(r"\b(CORRECT|WRONG)\b")


def parse_judge_verdict(text: str) -> bool | None:
    """Devuelve ``True`` si CORRECT, ``False`` si WRONG, ``None`` si ambiguo.

    Si aparecen ambos términos, es ambiguo (el prompt explícitamente lo
    prohíbe — ese fallo del juez se cuenta separado). Si solo aparece uno,
    nos quedamos con él independientemente de mayúsculas/minúsculas.
    """
    matches = _VERDICT_RE.findall((text or "").upper())
    if not matches:
        return None
    unique = set(matches)
    if unique == {"CORRECT"}:
        return True
    if unique == {"WRONG"}:
        return False
    return None  # ambos presentes: el juez ha violado el contrato.


JudgeCallable = Callable[[str, str, str], tuple[bool | None, str]]
"""``(question, gold, generated) → (verdict|None, raw_text)``."""


def _normalise_litellm_model(model_id: str) -> str:
    """Convierte ``provider:model`` (LangChain) a ``provider/model`` (litellm).

    El resto del código usa la convención de LangChain con dos puntos. Litellm
    requiere barra para reconocer el provider, así que sustituimos el primer
    ``:`` por ``/`` cuando aparece. Si el id ya viene con barra o sin
    provider, lo dejamos intacto.
    """
    if "/" in model_id or ":" not in model_id:
        return model_id
    provider, _, name = model_id.partition(":")
    return f"{provider}/{name}"


def default_judge(model_id: str | None = None) -> JudgeCallable:
    """Construye un juez basado en litellm con el prompt del apéndice 6.1.2.

    El modelo se resuelve a través de litellm: acepta cualquiera de los
    formatos soportados (``anthropic/claude-...``, ``gpt-4`` con
    ``OPENAI_API_KEY``, etc.). Temperatura 0.0 para hacer el veredicto lo más
    determinista posible.
    """
    from litellm import completion as _litellm_completion

    from ..config import get_settings

    settings = get_settings()
    model = _normalise_litellm_model(model_id or settings.primary_llm_model)

    def _judge(question: str, gold: str, generated: str) -> tuple[bool | None, str]:
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            gold_answer=gold,
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
# Ingesta de sesiones en Recall
# ---------------------------------------------------------------------------


def _now_minus_days(days: float, *, anchor: datetime) -> datetime:
    return anchor - timedelta(days=days)


def _stable_message_id(sample_id: int, session_idx: int, turn_idx: int) -> str:
    return f"dmr-s{sample_id}-sess{session_idx}-t{turn_idx}"


def populate_recall(
    store: MemoryStore,
    sample: DMRSample,
    *,
    anchor: datetime | None = None,
) -> None:
    """Vuelca las 5 sesiones en Recall directamente.

    El paper ingiere las conversaciones previas como mensajes de Recall sin
    pasar por el agente — es lo que reproducimos: no gastamos LLM en la
    ingesta y la búsqueda funciona idéntico cuando el agente luego usa
    ``conversation_search`` desde la sesión 6.

    El rol que persiste cada turno es:
    - ``assistant`` para los turnos del agente (``sample.agent_speaker``).
    - ``user`` para los turnos del otro (lo que el agente "oyó").
    """
    base = anchor or datetime.now(timezone.utc)
    for s_idx, session in enumerate(sample.sessions):
        # Anclamos cada turno en el centro de la sesión: a un nivel de
        # granularidad razonable basta con tener todos los turnos de una
        # sesión cerca en el tiempo y separados de las otras sesiones.
        session_when = _now_minus_days(session.days_ago, anchor=base)
        for t_idx, turn in enumerate(session.turns):
            role = "assistant" if turn.speaker == sample.agent_speaker else "user"
            store.persist_message(
                content=turn.text,
                role=role,
                occurred_at=session_when + timedelta(seconds=t_idx),
                message_id=_stable_message_id(sample.sample_id, s_idx, t_idx),
            )


# ---------------------------------------------------------------------------
# Construcción del agente DMR
# ---------------------------------------------------------------------------


def build_dmr_core_memory(
    sample: DMRSample,
    *,
    persona_block_limit: int = 2000,
) -> Any:
    """Carga las personas en Core Memory: ``persona`` (agente) + ``human`` (otro).

    Usa ``persona`` como label en lugar de ``assistant`` para alinearse con la
    nomenclatura de Letta/MemGPT (apéndice 6.1.1: "the persona of the agent"),
    pero la lógica del agente trata cualquier bloque por igual.
    """
    persona_text = "\n".join(sample.persona_agent)
    human_text = "\n".join(sample.persona_other)
    return default_core_memory(
        assistant=persona_text,
        human=human_text,
        block_limit=persona_block_limit,
    )


def build_dmr_initial_state(sample: DMRSample) -> MemGPTState:
    return MemGPTState(core_memory=build_dmr_core_memory(sample))


def build_dmr_agent(
    sample: DMRSample,
    store: MemoryStore,
    *,
    model_id: str | None = None,
) -> CompiledStateGraph:
    """Agente con persona + system prompt del apéndice 6.1.1 + memoria llena."""
    return build_agent(
        system_prompt=DMR_ASSISTANT_PROMPT,
        memory_store=store,
        model_id=model_id,
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass
class DMRResult:
    sample_id: int
    question: str
    gold_answer: str
    predicted: str
    rouge_l_recall: float
    judge_correct: bool | None
    judge_raw: str
    elapsed_seconds: float
    chained_heartbeats: int
    conversation_search_calls: int


@dataclass
class DMRSummary:
    total: int
    judged: int
    correct: int
    accuracy: float
    rouge_l_recall_mean: float
    mean_elapsed_seconds: float
    results: list[DMRResult] = field(default_factory=list)


def _count_search_calls(messages: list[Any]) -> int:
    return sum(
        1
        for m in messages
        if isinstance(m, AIMessage)
        for tc in (m.tool_calls or [])
        if tc.get("name") == "conversation_search"
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
    sample: DMRSample,
    *,
    agent: CompiledStateGraph,
    judge: JudgeCallable,
    initial_state: MemGPTState | None = None,
) -> DMRResult:
    """Ejecuta la sesión 6 (DMR question) sobre ``agent`` y puntúa el resultado."""
    payload: dict[str, Any] = {"messages": [HumanMessage(content=sample.question)]}
    if initial_state is not None:
        # Inyectamos el Core Memory inicial vía el primer step: LangGraph
        # acepta dicts derivados del state como input.
        payload = {**initial_state.model_dump(), **payload}

    config = {"configurable": {"thread_id": f"dmr-s{sample.sample_id}-session6"}}
    start = time.perf_counter()
    out = _invoke_with_retry(agent, payload, config)
    elapsed = time.perf_counter() - start

    messages = out.get("messages", []) if isinstance(out, dict) else []
    final = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
    predicted = ""
    if final is not None:
        raw = final.content
        predicted = raw if isinstance(raw, str) else str(raw or "")

    rouge = rouge_l_recall(sample.gold_answer, predicted)
    verdict, judge_raw = judge(sample.question, sample.gold_answer, predicted)

    return DMRResult(
        sample_id=sample.sample_id,
        question=sample.question,
        gold_answer=sample.gold_answer,
        predicted=predicted,
        rouge_l_recall=rouge,
        judge_correct=verdict,
        judge_raw=judge_raw,
        elapsed_seconds=elapsed,
        chained_heartbeats=int(out.get("chained_heartbeats", 0)) if isinstance(out, dict) else 0,
        conversation_search_calls=_count_search_calls(messages),
    )


AgentFactory = Callable[[DMRSample, MemoryStore], CompiledStateGraph]
StoreFactory = Callable[[DMRSample], MemoryStore]


def default_store_factory(_sample: DMRSample) -> MemoryStore:
    return InMemoryStore()


def make_default_agent_factory(model_id: str | None = None) -> AgentFactory:
    def _factory(sample: DMRSample, store: MemoryStore) -> CompiledStateGraph:
        return build_dmr_agent(sample, store, model_id=model_id)

    return _factory


def run_benchmark(
    samples: list[DMRSample],
    *,
    judge: JudgeCallable,
    agent_factory: AgentFactory | None = None,
    store_factory: StoreFactory = default_store_factory,
    on_result: Callable[[DMRResult], None] | None = None,
    sleep_between_seconds: float = 0.0,
) -> DMRSummary:
    """Ejecuta el benchmark MemGPT sobre ``samples`` y devuelve el resumen.

    Por sample se aísla un ``MemoryStore`` y un agente nuevos: las
    conversaciones de un par no deben contaminar las búsquedas de otro.
    """
    factory = agent_factory or make_default_agent_factory()
    results: list[DMRResult] = []
    for sample in samples:
        store = store_factory(sample)
        try:
            populate_recall(store, sample)
            agent = factory(sample, store)
            initial_state = build_dmr_initial_state(sample)
            r = run_sample(sample, agent=agent, judge=judge, initial_state=initial_state)
            results.append(r)
            if on_result is not None:
                on_result(r)
            if sleep_between_seconds > 0:
                time.sleep(sleep_between_seconds)
        finally:
            store.close()

    return _summarize(results)


def _summarize(results: list[DMRResult]) -> DMRSummary:
    total = len(results)
    judged = sum(1 for r in results if r.judge_correct is not None)
    correct = sum(1 for r in results if r.judge_correct is True)
    accuracy = correct / judged if judged else 0.0
    rouge_mean = sum(r.rouge_l_recall for r in results) / total if total else 0.0
    elapsed_mean = sum(r.elapsed_seconds for r in results) / total if total else 0.0
    return DMRSummary(
        total=total,
        judged=judged,
        correct=correct,
        accuracy=accuracy,
        rouge_l_recall_mean=rouge_mean,
        mean_elapsed_seconds=elapsed_mean,
        results=results,
    )


# ---------------------------------------------------------------------------
# Baseline (sin memoria)
# ---------------------------------------------------------------------------


def build_baseline_summary(sample: DMRSample) -> str:
    """Construye el ``CONVERSATION_SUMMARY`` del baseline.

    Formato: viñetas con los hechos del agente + del otro, tal como hace el
    paper para "mimic an extended recursive summarization procedure".
    """
    bits: list[str] = []
    if sample.summary_agent:
        bits.append(f"About {sample.agent_speaker}:")
        bits.extend(f"- {x}" for x in sample.summary_agent)
    if sample.summary_other:
        bits.append("")
        bits.append(f"About {sample.other_speaker}:")
        bits.extend(f"- {x}" for x in sample.summary_other)
    return "\n".join(bits) if bits else "(no prior conversation summary available)"


def build_baseline_system_prompt(sample: DMRSample) -> str:
    return BASELINE_PREPROMPT_TEMPLATE.format(
        conversation_summary=build_baseline_summary(sample)
    )


BaselineAgentFactory = Callable[[DMRSample], CompiledStateGraph]


def make_baseline_agent_factory(model_id: str | None = None) -> BaselineAgentFactory:
    def _factory(sample: DMRSample) -> CompiledStateGraph:
        return build_agent(
            system_prompt=build_baseline_system_prompt(sample),
            memory_store=None,  # sin tools de memoria.
            model_id=model_id,
        )

    return _factory


def run_baseline_benchmark(
    samples: list[DMRSample],
    *,
    judge: JudgeCallable,
    agent_factory: BaselineAgentFactory | None = None,
    on_result: Callable[[DMRResult], None] | None = None,
    sleep_between_seconds: float = 0.0,
) -> DMRSummary:
    factory = agent_factory or make_baseline_agent_factory()
    results: list[DMRResult] = []
    for sample in samples:
        agent = factory(sample)
        r = run_sample(sample, agent=agent, judge=judge, initial_state=None)
        results.append(r)
        if on_result is not None:
            on_result(r)
        if sleep_between_seconds > 0:
            time.sleep(sleep_between_seconds)
    return _summarize(results)
