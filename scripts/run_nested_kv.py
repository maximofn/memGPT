"""Runner del benchmark Nested KV (Fase 8).

Uso típico:

    uv run scripts/run_nested_kv.py --configs 30 --levels 0,1,2,3,4

Variables relevantes:
- ``--configs``: cuántas configuraciones (default 30, como el paper).
- ``--levels``: niveles de anidamiento a evaluar (default 0..4).
- ``--seed``: semilla para reproducibilidad.
- ``--model``: id del LLM a usar (override de ``primary_llm_model``).
- ``--graphiti``: usa ``GraphitiStore`` real en lugar de ``InMemoryStore``.
- ``--output``: ruta opcional para volcar los resultados en JSON.

El script imprime accuracy global y por nivel al terminar; si ``--output``
está activo, persiste los QueryResult crudos para análisis posterior.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from memgpt.agent import build_agent
from memgpt.benchmarks.nested_kv import (
    LEVELS,
    NESTED_KV_ASSISTANT,
    QueryResult,
    default_store_factory,
    generate_dataset,
    make_baseline_agent_builder,
    run_baseline_benchmark,
    run_benchmark,
)
from memgpt.memory_store import MemoryStore


def _parse_levels(spec: str) -> tuple[int, ...]:
    values = tuple(int(x.strip()) for x in spec.split(",") if x.strip())
    invalid = [v for v in values if v not in LEVELS]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"levels must be a subset of {list(LEVELS)}; got invalid {invalid}"
        )
    return values


def _build_graphiti_store_factory():
    """Lazy import: Graphiti requiere Neo4j en ejecución."""
    from graphiti_core import Graphiti  # type: ignore[import-not-found]

    from memgpt.config import get_settings
    from memgpt.memory_store import GraphitiStore

    settings = get_settings()

    def factory(config_id: int) -> MemoryStore:
        client = Graphiti(
            settings.neo4j_uri,
            settings.neo4j_user,
            settings.neo4j_password,
        )
        return GraphitiStore(client, group_id=f"nested-kv-cfg{config_id}")

    return factory


def main() -> int:
    parser = argparse.ArgumentParser(description="MemGPT Nested KV benchmark runner")
    parser.add_argument("--configs", type=int, default=30)
    parser.add_argument(
        "--levels",
        type=_parse_levels,
        default=LEVELS,
        help="Comma-separated nesting levels (default: 0,1,2,3,4)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM id (e.g. anthropic:claude-sonnet-4-6). Defaults to settings.primary_llm_model.",
    )
    parser.add_argument(
        "--graphiti",
        action="store_true",
        help="Use the real GraphitiStore backend (requires Neo4j). Default: InMemoryStore.",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help=(
            "Run the no-memory baseline: 140 pairs inlined as JSON in the "
            "system prompt, no archival tools. Reproduces the GPT-4 control "
            "of Figure 7. Mutually exclusive with --graphiti."
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--sleep-between",
        type=float,
        default=0.0,
        help=(
            "Seconds to sleep after each query. Use it to stay below provider "
            "TPM limits (e.g. 8.0 for OpenAI tier-1 GPT-4 at 40k TPM)."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-query progress output.",
    )
    args = parser.parse_args()

    if args.baseline and args.graphiti:
        parser.error("--baseline and --graphiti are mutually exclusive")

    load_dotenv()

    configs = generate_dataset(seed=args.seed, n_configs=args.configs)

    # Buffer compartido para poder volcar resultados parciales si el run aborta.
    collected: list[QueryResult] = []

    def progress(r: QueryResult) -> None:
        collected.append(r)
        if args.output is not None:
            _write_partial(args.output, collected, mode_baseline=args.baseline)
        if args.quiet:
            return
        flag = "OK" if r.correct else "MISS"
        print(
            f"[cfg={r.config_id:02d} lvl={r.nesting_level} {flag}] "
            f"{r.elapsed_seconds:5.1f}s "
            f"searches={r.archival_search_calls} "
            f"predicted={r.predicted}"
        )

    aborted_by: Exception | None = None
    summary = None
    try:
        if args.baseline:
            summary = run_baseline_benchmark(
                configs,
                levels=tuple(args.levels),
                agent_builder=make_baseline_agent_builder(args.model),
                on_result=progress,
                sleep_between_seconds=args.sleep_between,
            )
        else:
            if args.graphiti:
                store_factory = _build_graphiti_store_factory()
            else:
                store_factory = default_store_factory

            def agent_factory(store: MemoryStore):
                return build_agent(
                    system_prompt=NESTED_KV_ASSISTANT,
                    memory_store=store,
                    model_id=args.model,
                )

            summary = run_benchmark(
                configs,
                levels=tuple(args.levels),
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
    print(f"Mode:          {mode}")

    if summary is None:
        # Reconstruimos un resumen a partir de lo recolectado para poder
        # imprimir aunque el run haya petado a mitad.
        summary = _summarize_partial(collected)
        print(f"Status:        ABORTED ({type(aborted_by).__name__})")

    print(f"Total queries: {summary.total}")
    print(f"Correct:       {summary.correct}")
    print(f"Accuracy:      {summary.accuracy:.3f}")
    print(f"Mean time/q:   {summary.mean_elapsed_seconds:.2f}s")
    print("Accuracy by level:")
    for lvl in sorted(summary.accuracy_by_level):
        print(f"  level {lvl}: {summary.accuracy_by_level[lvl]:.3f}")
    print("=" * 60)

    if args.output is not None:
        # Sobrescribe el dump incremental con el snapshot final consistente.
        _write_summary(args.output, summary, mode_baseline=args.baseline)
        print(f"Wrote {args.output}")

    if aborted_by is not None:
        return 2  # exit distinto para que CI lo distinga del mero fallo de accuracy.
    return 0 if summary.accuracy >= 0.95 else 1


def _build_summary_dict(summary, *, mode_baseline: bool) -> dict:
    return {
        "mode": "baseline" if mode_baseline else "memgpt",
        "total": summary.total,
        "correct": summary.correct,
        "accuracy": summary.accuracy,
        "accuracy_by_level": summary.accuracy_by_level,
        "mean_elapsed_seconds": summary.mean_elapsed_seconds,
        "results": [asdict(r) for r in summary.results],
    }


def _summarize_partial(results: list[QueryResult]):
    """Replica BenchmarkSummary a partir de una lista cruda de resultados."""
    from memgpt.benchmarks.nested_kv import BenchmarkSummary, LEVELS

    total = len(results)
    correct = sum(1 for r in results if r.correct)
    accuracy = correct / total if total else 0.0
    by_level: dict[int, float] = {}
    for level in LEVELS:
        bucket = [r for r in results if r.nesting_level == level]
        if bucket:
            by_level[level] = sum(1 for r in bucket if r.correct) / len(bucket)
    mean_elapsed = sum(r.elapsed_seconds for r in results) / total if total else 0.0
    return BenchmarkSummary(
        total=total,
        correct=correct,
        accuracy=accuracy,
        accuracy_by_level=by_level,
        mean_elapsed_seconds=mean_elapsed,
        results=list(results),
    )


def _write_partial(path: Path, results: list[QueryResult], *, mode_baseline: bool) -> None:
    summary = _summarize_partial(results)
    _write_summary(path, summary, mode_baseline=mode_baseline)


def _write_summary(path: Path, summary, *, mode_baseline: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_build_summary_dict(summary, mode_baseline=mode_baseline), indent=2))


if __name__ == "__main__":
    sys.exit(main())
