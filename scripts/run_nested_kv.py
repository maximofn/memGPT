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
    CHAIN_LENGTH,
    NESTED_KV_ASSISTANT,
    PAIRS_PER_CONFIG,
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
    # Solo exigimos enteros no negativos: el tope real lo marca --chain-length
    # (una cadena de N nodos admite niveles 0..N-1) y se valida en main() una
    # vez conocida esa longitud.
    invalid = [v for v in values if v < 0]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"levels must be non-negative integers; got invalid {invalid}"
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
        "--chain-length",
        type=int,
        default=CHAIN_LENGTH,
        help=(
            "Longitud de la cadena guía (nº de nodos). Una cadena de N nodos "
            f"admite niveles de anidamiento 0..N-1. Default: {CHAIN_LENGTH} "
            "(niveles 0..4, fiel a la Figura 7 del paper). Súbelo para probar "
            "anidamientos más profundos."
        ),
    )
    parser.add_argument(
        "--levels",
        type=_parse_levels,
        default=None,
        help=(
            "Comma-separated nesting levels a ejecutar. Default: todos los que "
            "permite --chain-length (0..chain_length-1)."
        ),
    )
    parser.add_argument(
        "--pairs-per-config",
        type=int,
        default=PAIRS_PER_CONFIG,
        help=(
            "Nº total de pares KV por config (cadena + distractores). Default: "
            f"{PAIRS_PER_CONFIG} (paper). En baseline todos van inline en el "
            "system prompt, así que subirlo agranda el pajar y degrada a "
            "modelos que de otro modo resuelven la cadena inline. Debe ser "
            ">= --chain-length."
        ),
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

    if args.chain_length < 2:
        parser.error("--chain-length must be >= 2")

    if args.pairs_per_config < args.chain_length:
        parser.error(
            f"--pairs-per-config ({args.pairs_per_config}) must be >= "
            f"--chain-length ({args.chain_length})"
        )

    # Niveles a ejecutar: si no se pidieron explícitamente, todos los que
    # permite la cadena (0..chain_length-1). Si se pidieron, validamos que no
    # excedan la profundidad disponible.
    if args.levels is None:
        levels = tuple(range(args.chain_length))
    else:
        too_deep = [lv for lv in args.levels if lv >= args.chain_length]
        if too_deep:
            parser.error(
                f"--levels {too_deep} exceed the chain depth; with "
                f"--chain-length {args.chain_length} valid levels are "
                f"0..{args.chain_length - 1}"
            )
        levels = args.levels

    load_dotenv()

    configs = generate_dataset(
        seed=args.seed,
        n_configs=args.configs,
        n_pairs=args.pairs_per_config,
        chain_length=args.chain_length,
    )

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
                levels=levels,
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
                levels=levels,
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
    from memgpt.benchmarks.nested_kv import BenchmarkSummary

    total = len(results)
    correct = sum(1 for r in results if r.correct)
    accuracy = correct / total if total else 0.0
    by_level: dict[int, float] = {}
    # Derivamos los niveles de los propios resultados (no de la constante
    # LEVELS) para soportar --chain-length > 5.
    for level in sorted({r.nesting_level for r in results}):
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
