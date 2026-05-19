"""Runner del benchmark DMR (Fase 9).

Uso típico:

    # Smoke con 5 pares.
    uv run scripts/run_dmr.py --dataset datasets/msc_self_instruct.jsonl --limit 5

    # Benchmark completo (500 pares) con InMemoryStore.
    uv run scripts/run_dmr.py --dataset datasets/msc_self_instruct.jsonl \\
        --output runs/dmr.json

    # Baseline (sin memoria, summary lossy en system prompt).
    uv run scripts/run_dmr.py --dataset datasets/msc_self_instruct.jsonl \\
        --baseline --output runs/dmr_baseline.json

    # Contra Graphiti real (requiere docker compose up).
    uv run scripts/run_dmr.py --dataset datasets/msc_self_instruct.jsonl --graphiti

Flags relevantes:
- ``--dataset``: ruta al fichero ``msc_self_instruct.jsonl`` (descárgalo
  de ``MemGPT/MSC-Self-Instruct``).
- ``--limit``: cuántos pares evaluar (default: todos).
- ``--model``: id del LLM del agente (override de ``primary_llm_model``).
- ``--judge-model``: id del LLM juez (default: ``primary_llm_model``).
- ``--graphiti``: usa ``GraphitiStore`` real en lugar de ``InMemoryStore``.
- ``--baseline``: ejecuta el control sin memoria.
- ``--download``: descarga el dataset de HF Hub a ``--dataset`` si falta.
- ``--output``: ruta para volcar resultados en JSON (con dump incremental).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from memgpt.benchmarks.dmr import (
    DMRResult,
    DMRSample,
    DMRSummary,
    build_dmr_agent,
    build_dmr_initial_state,
    default_judge,
    default_store_factory,
    download_dataset,
    load_dataset,
    make_baseline_agent_factory,
    run_baseline_benchmark,
    run_benchmark,
)
from memgpt.memory_store import MemoryStore


def _build_graphiti_store_factory():
    """Lazy import: Graphiti requiere Neo4j en ejecución."""
    from graphiti_core import Graphiti  # type: ignore[import-not-found]

    from memgpt.config import get_settings
    from memgpt.memory_store import GraphitiStore

    settings = get_settings()

    def factory(sample: DMRSample) -> MemoryStore:
        client = Graphiti(
            settings.neo4j_uri,
            settings.neo4j_user,
            settings.neo4j_password,
        )
        return GraphitiStore(client, group_id=f"dmr-s{sample.sample_id}")

    return factory


def main() -> int:
    parser = argparse.ArgumentParser(description="MemGPT DMR benchmark runner")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/msc_self_instruct.jsonl"),
        help="Path to the MSC-Self-Instruct JSONL file.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the dataset from HF Hub if --dataset doesn't exist yet.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of pairs to evaluate (default: all 500).",
    )
    parser.add_argument(
        "--agent-speaker",
        type=str,
        default="Speaker 1",
        choices=["Speaker 1", "Speaker 2"],
        help="Which speaker the agent role-plays. The DMR question is asked by the other.",
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
        "--graphiti",
        action="store_true",
        help="Use the real GraphitiStore backend (requires Neo4j).",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run the no-memory baseline with the lossy summary in the system prompt.",
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

    if args.baseline and args.graphiti:
        parser.error("--baseline and --graphiti are mutually exclusive")

    load_dotenv()

    if args.download and not args.dataset.exists():
        print(f"[download] fetching MSC-Self-Instruct → {args.dataset}")
        download_dataset(args.dataset)

    if not args.dataset.exists():
        parser.error(
            f"dataset not found at {args.dataset!s}. "
            "Re-run with --download or fetch it manually."
        )

    samples = load_dataset(
        args.dataset,
        limit=args.limit,
        agent_speaker=args.agent_speaker,
    )
    if not samples:
        print("[abort] dataset is empty after filtering.")
        return 1

    judge = default_judge(model_id=args.judge_model or args.model)

    collected: list[DMRResult] = []

    def progress(r: DMRResult) -> None:
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
            f"searches={r.conversation_search_calls} "
            f"rouge_l={r.rouge_l_recall:.3f}"
        )

    aborted_by: Exception | None = None
    summary: DMRSummary | None = None
    try:
        if args.baseline:
            summary = run_baseline_benchmark(
                samples,
                judge=judge,
                agent_factory=make_baseline_agent_factory(args.model),
                on_result=progress,
                sleep_between_seconds=args.sleep_between,
            )
        else:
            if args.graphiti:
                store_factory = _build_graphiti_store_factory()
            else:
                store_factory = default_store_factory

            def agent_factory(sample: DMRSample, store: MemoryStore):
                return build_dmr_agent(sample, store, model_id=args.model)

            summary = run_benchmark(
                samples,
                judge=judge,
                agent_factory=agent_factory,
                store_factory=store_factory,
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

    mode = "BASELINE (no memory)" if args.baseline else "MemGPT"
    print()
    print("=" * 60)
    print(f"Mode:              {mode}")

    if summary is None:
        summary = _summarize_partial(collected)
        print(f"Status:            ABORTED ({type(aborted_by).__name__ if aborted_by else 'unknown'})")

    print(f"Total samples:     {summary.total}")
    print(f"Judged:            {summary.judged}")
    print(f"Correct:           {summary.correct}")
    print(f"Accuracy:          {summary.accuracy:.3f}")
    print(f"ROUGE-L recall μ:  {summary.rouge_l_recall_mean:.3f}")
    print(f"Mean time/sample:  {summary.mean_elapsed_seconds:.2f}s")
    print("=" * 60)

    if args.output is not None:
        _write_summary(args.output, summary, mode_baseline=args.baseline)
        print(f"Wrote {args.output}")

    if aborted_by is not None:
        return 2  # distinto del fail por accuracy.
    # Definición de hecho: accuracy ≥ 0.92 y ROUGE-L ≥ 0.80.
    if summary.accuracy >= 0.92 and summary.rouge_l_recall_mean >= 0.80:
        return 0
    return 1


def _build_summary_dict(summary: DMRSummary, *, mode_baseline: bool) -> dict:
    return {
        "mode": "baseline" if mode_baseline else "memgpt",
        "total": summary.total,
        "judged": summary.judged,
        "correct": summary.correct,
        "accuracy": summary.accuracy,
        "rouge_l_recall_mean": summary.rouge_l_recall_mean,
        "mean_elapsed_seconds": summary.mean_elapsed_seconds,
        "results": [asdict(r) for r in summary.results],
    }


def _summarize_partial(results: list[DMRResult]) -> DMRSummary:
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
        results=list(results),
    )


def _write_partial(path: Path, results: list[DMRResult], *, mode_baseline: bool) -> None:
    summary = _summarize_partial(results)
    _write_summary(path, summary, mode_baseline=mode_baseline)


def _write_summary(path: Path, summary: DMRSummary, *, mode_baseline: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_build_summary_dict(summary, mode_baseline=mode_baseline), indent=2)
    )


# Exposed for tests / re-use.
__all__ = [
    "build_dmr_initial_state",
    "default_judge",
    "main",
]


if __name__ == "__main__":
    sys.exit(main())
