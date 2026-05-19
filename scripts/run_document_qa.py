"""Runner del benchmark Document QA (Fase 10).

Uso típico:

    # Smoke con 5 preguntas.
    uv run scripts/run_document_qa.py --dataset datasets/nq_open.jsonl --limit 5

    # Benchmark completo (50 preguntas) con InMemoryStore.
    uv run scripts/run_document_qa.py --dataset datasets/nq_open.jsonl \\
        --output runs/doc_qa.json

    # Baseline (top-10 docs en el prompt, sin archival).
    uv run scripts/run_document_qa.py --dataset datasets/nq_open.jsonl \\
        --baseline --baseline-top-k 10 --output runs/doc_qa_baseline_k10.json

Flags relevantes:
- ``--dataset``: ruta al fichero JSONL (estilo DPR / lost-in-the-middle:
  cada record con ``question``, ``answers``, ``ctxs``).
- ``--limit``: cuántas preguntas evaluar (default: todas).
- ``--model``: id del LLM del agente (override de ``primary_llm_model``).
- ``--judge-model``: id del LLM juez (default: same as agent).
- ``--embedder``: backend de embeddings del Archival
  (``local`` [default] / ``openai`` / ``none``). El paper usa retrieval
  semántico real; ``none`` cae a substring y el benchmark deja de tener
  sentido — está solo para sanity checks.
- ``--embedder-model``: override del modelo de embeddings.
- ``--baseline``: ejecuta el control sin memoria (top-K docs en prompt).
- ``--baseline-top-k``: cuántos docs incrustar en el prompt baseline
  (default: todos los del sample). Para reproducir Figura 5: 10/20/30.
- ``--download``: descarga el dataset desde HF Hub si falta.
- ``--output``: ruta para volcar resultados en JSON (con dump incremental).

Nota: Document QA usa siempre ``InMemoryStore``. ``GraphitiStore`` está
descartado en este benchmark porque ``add_episode`` extrae entidades por
LLM en cada inserción, lo que con 30 docs/sample lo hace prohibitivo y
arquitecturalmente innecesario (Graphiti es memoria episódica, no vector
store de docs Wikipedia). Para evaluar Graphiti usa los benchmarks
DMR / MSC.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from memgpt.benchmarks.document_qa import (
    DocQAResult,
    DocQASummary,
    DocumentSample,
    build_doc_qa_agent,
    default_judge,
    download_dataset,
    load_dataset,
    make_baseline_agent_factory,
    make_store_factory,
    run_baseline_benchmark,
    run_benchmark,
)
from memgpt.embedders import make_embedder
from memgpt.memory_store import MemoryStore


def main() -> int:
    parser = argparse.ArgumentParser(description="MemGPT Document QA benchmark runner")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/nq_open.jsonl"),
        help="Path to the NQ-Open / lost-in-the-middle JSONL file.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the dataset from HF Hub if --dataset doesn't exist yet.",
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default="MemGPT/qa_data",
        help="HF Hub repo_id used by --download (default: MemGPT/qa_data).",
    )
    parser.add_argument(
        "--hf-filename",
        type=str,
        default="nq-open-30_total_documents_gold_at_14.jsonl.gz",
        help=(
            "Filename inside the HF repo to fetch. Default reproduces Figure 5 "
            "K=30 with gold at position 14 (lost-in-the-middle worst case). "
            "Other options: gold_at_{0,4,9,19,24,29}.jsonl.gz."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of questions to evaluate (default: all).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM id for the agent (e.g. anthropic:claude-sonnet-4-6).",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="LLM id for the judge (default: same as --model).",
    )
    parser.add_argument(
        "--embedder",
        type=str,
        default="local",
        choices=("local", "openai", "none"),
        help=(
            "Backend de embeddings para el Archival semántico. "
            "'local' (default) = sentence-transformers (sin red, requiere "
            "`uv sync --extra embeddings-local`). 'openai' = API de OpenAI "
            "(requiere OPENAI_API_KEY). 'none' = substring matching "
            "(útil solo para sanity checks; rompe el sentido del benchmark)."
        ),
    )
    parser.add_argument(
        "--embedder-model",
        type=str,
        default=None,
        help=(
            "Override del modelo de embeddings. Para 'local' un id de HF "
            "(p. ej. 'facebook/contriever-msmarco' para reproducir al "
            "paper). Para 'openai' un id de embedding model "
            "(p. ej. 'text-embedding-3-large')."
        ),
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run the no-archival baseline with top-K documents in the prompt.",
    )
    parser.add_argument(
        "--baseline-top-k",
        type=int,
        default=None,
        help=(
            "How many documents per sample to inline in the baseline prompt. "
            "Default: all of them. Use 10/20/30 to reproduce Figure 5."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to dump the JSON summary + per-sample results.",
    )
    parser.add_argument(
        "--sleep-between",
        type=float,
        default=0.0,
        help="Seconds to sleep after each sample to stay below provider TPM limits.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-sample progress output.",
    )
    args = parser.parse_args()

    load_dotenv()

    if args.download and not args.dataset.exists():
        print(f"[download] fetching {args.hf_repo}/{args.hf_filename} → {args.dataset}")
        download_dataset(
            args.dataset,
            repo_id=args.hf_repo,
            filename=args.hf_filename,
        )

    if not args.dataset.exists():
        parser.error(
            f"dataset not found at {args.dataset!s}. "
            "Re-run with --download or fetch it manually."
        )

    samples = load_dataset(args.dataset, limit=args.limit)
    if not samples:
        print("[abort] dataset is empty after filtering.")
        return 1

    judge = default_judge(model_id=args.judge_model or args.model)

    collected: list[DocQAResult] = []

    def progress(r: DocQAResult) -> None:
        collected.append(r)
        if args.output is not None:
            _write_partial(args.output, collected, mode_baseline=args.baseline)
        if args.quiet:
            return
        verdict = (
            "OK" if r.judge_correct is True
            else "MISS" if r.judge_correct is False
            else "ABSTAIN"
        )
        print(
            f"[s={r.sample_id:03d} {verdict:7s}] "
            f"{r.elapsed_seconds:5.1f}s "
            f"searches={r.archival_search_calls} "
            f"em={int(r.exact_match)}"
        )

    aborted_by: Exception | None = None
    summary: DocQASummary | None = None
    try:
        if args.baseline:
            summary = run_baseline_benchmark(
                samples,
                judge=judge,
                agent_factory=make_baseline_agent_factory(args.model),
                top_k=args.baseline_top_k,
                on_result=progress,
                sleep_between_seconds=args.sleep_between,
            )
        else:
            embedder = make_embedder(args.embedder, model_name=args.embedder_model)
            if embedder is None and args.embedder != "none":
                # No debería pasar con choices=, pero defensivo.
                parser.error(f"embedder inválido: {args.embedder}")
            if args.embedder == "none":
                print(
                    "[warn] --embedder=none usa substring matching: "
                    "el benchmark no reproduce el setup del paper."
                )

            def agent_factory(_sample: DocumentSample, store: MemoryStore):
                return build_doc_qa_agent(store, model_id=args.model)

            summary = run_benchmark(
                samples,
                judge=judge,
                agent_factory=agent_factory,
                store_factory=make_store_factory(embedder),
                on_result=progress,
                sleep_between_seconds=args.sleep_between,
            )
    except KeyboardInterrupt as exc:
        aborted_by = exc
        print("\n[abort] interrupted by user — dumping partial results.")
    except Exception as exc:
        aborted_by = exc
        print(f"\n[abort] {type(exc).__name__}: {exc}")
        print("[abort] dumping partial results before exit.")

    mode = "BASELINE (no archival)" if args.baseline else "MemGPT"
    print()
    print("=" * 60)
    print(f"Mode:                  {mode}")
    if args.baseline and args.baseline_top_k is not None:
        print(f"Baseline top-K:        {args.baseline_top_k}")

    if summary is None:
        summary = _summarize_partial(collected)
        print(
            f"Status:                ABORTED "
            f"({type(aborted_by).__name__ if aborted_by else 'unknown'})"
        )

    print(f"Total samples:         {summary.total}")
    print(f"Judged:                {summary.judged}")
    print(f"Correct (judge):       {summary.correct}")
    print(f"Accuracy:              {summary.accuracy:.3f}")
    print(f"Exact-match rate:      {summary.exact_match_rate:.3f}")
    print(f"Insufficient rate:     {summary.insufficient_rate:.3f}")
    print(f"Mean archival calls:   {summary.mean_archival_searches:.2f}")
    print(f"Mean time/sample:      {summary.mean_elapsed_seconds:.2f}s")
    print("=" * 60)

    if args.output is not None:
        _write_summary(args.output, summary, mode_baseline=args.baseline)
        print(f"Wrote {args.output}")

    if aborted_by is not None:
        return 2  # abort distinto del fail por accuracy.
    # Definición de hecho del paper: MemGPT ≥ baseline GPT-4 con K=10. La
    # CLI no conoce el baseline a priori, así que nos quedamos con un
    # umbral conservador (≥ 0.40) — el paper reporta ~0.45 para MemGPT.
    if summary.accuracy >= 0.40:
        return 0
    return 1


def _build_summary_dict(summary: DocQASummary, *, mode_baseline: bool) -> dict:
    return {
        "mode": "baseline" if mode_baseline else "memgpt",
        "total": summary.total,
        "judged": summary.judged,
        "correct": summary.correct,
        "accuracy": summary.accuracy,
        "exact_match_rate": summary.exact_match_rate,
        "insufficient_rate": summary.insufficient_rate,
        "mean_elapsed_seconds": summary.mean_elapsed_seconds,
        "mean_archival_searches": summary.mean_archival_searches,
        "results": [_result_dict(r) for r in summary.results],
    }


def _result_dict(r: DocQAResult) -> dict:
    d = asdict(r)
    # ``gold_answers`` es tuple en el dataclass; JSON quiere lista.
    d["gold_answers"] = list(r.gold_answers)
    return d


def _summarize_partial(results: list[DocQAResult]) -> DocQASummary:
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
        results=list(results),
    )


def _write_partial(path: Path, results: list[DocQAResult], *, mode_baseline: bool) -> None:
    summary = _summarize_partial(results)
    _write_summary(path, summary, mode_baseline=mode_baseline)


def _write_summary(path: Path, summary: DocQASummary, *, mode_baseline: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_build_summary_dict(summary, mode_baseline=mode_baseline), indent=2)
    )


__all__ = [
    "default_judge",
    "main",
]


if __name__ == "__main__":
    sys.exit(main())
