"""Gráfica comparativa de métricas escalares: baseline vs MemGPT.

Para los benchmarks que no tienen eje de "nivel" (DMR — Fase 9; Document QA —
Fase 10) la comparación natural es un gráfico de barras agrupadas con las
métricas que el runner emite: accuracy (LLM-judge), ROUGE-L recall,
exact-match, insufficient-rate, etc.

El script autodetecta qué métricas existen en cada JSON (mirando las claves de
``results``) y solo grafica las comunes a ambos runs. Las métricas se
recalculan desde ``results`` para que los dumps parciales también valgan.

Uso típico:

    # DMR
    uv run --extra viz scripts/plot_metrics.py \
        --baseline runs/dmr_baseline_4o_mini.json \
        --memgpt   runs/dmr_memgpt_4o_mini.json \
        --output   runs/dmr-4o-mini.png \
        --title    "DMR — gpt-4o-mini: baseline vs MemGPT"

    # Document QA
    uv run --extra viz scripts/plot_metrics.py \
        --baseline runs/doc_qa_smoke_baseline-gpt4o-mini.json \
        --memgpt   runs/doc_qa_smoke_memgpt-gpt4o-mini.json \
        --output   runs/doc-qa-4o-mini.png \
        --title    "Document QA — gpt-4o-mini: baseline vs MemGPT"

Requiere matplotlib (extra ``viz``): ``uv sync --extra viz`` o el flag
``uv run --extra viz`` como arriba.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Registro de métricas conocidas. Cada una se calcula desde un campo de los
# ``results``. ``lower_better`` solo afecta a la anotación (no invierte ejes).
#   - bool_over_judged: media de ``field is True`` sobre los items con veredicto
#     (``field is not None``). Es la accuracy del juez LLM.
#   - mean: media aritmética del campo (ratios/recall ya en [0, 1]).
METRICS: dict[str, dict] = {
    "accuracy": {
        "field": "judge_correct",
        "agg": "bool_over_judged",
        "label": "Accuracy\n(LLM-judge)",
    },
    "rouge_l": {
        "field": "rouge_l_recall",
        "agg": "mean",
        "label": "ROUGE-L\nrecall",
    },
    "exact_match": {
        "field": "exact_match",
        "agg": "mean",
        "label": "Exact\nmatch",
    },
    "insufficient": {
        "field": "insufficient",
        "agg": "mean",
        "label": "Insufficient\nrate",
        "lower_better": True,
    },
}


def _load(path: Path) -> tuple[list[dict], str]:
    data = json.loads(path.read_text())
    results = data.get("results", [])
    if not results:
        raise ValueError(f"{path} no tiene 'results' que graficar")
    return results, data.get("mode", path.stem)


def _compute(results: list[dict], spec: dict) -> float | None:
    field = spec["field"]
    if field not in results[0]:
        return None
    if spec["agg"] == "bool_over_judged":
        judged = [r[field] for r in results if r[field] is not None]
        if not judged:
            return None
        return sum(1 for v in judged if v is True) / len(judged)
    # mean
    vals = [r[field] for r in results if r.get(field) is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _plot(
    baseline: Path,
    memgpt: Path,
    output: Path,
    *,
    title: str,
    baseline_label: str,
    memgpt_label: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base_results, _ = _load(baseline)
    mem_results, _ = _load(memgpt)

    # Solo métricas presentes (y calculables) en AMBOS runs.
    keys: list[str] = []
    base_vals: list[float] = []
    mem_vals: list[float] = []
    labels: list[str] = []
    lower_better: list[bool] = []
    for key, spec in METRICS.items():
        bv = _compute(base_results, spec)
        mv = _compute(mem_results, spec)
        if bv is None or mv is None:
            continue
        keys.append(key)
        base_vals.append(bv)
        mem_vals.append(mv)
        labels.append(spec["label"])
        lower_better.append(spec.get("lower_better", False))

    if not keys:
        raise ValueError("No hay métricas comunes a ambos runs que graficar")

    x = range(len(keys))
    width = 0.38

    fig, ax = plt.subplots(figsize=(1.8 * len(keys) + 3, 5))
    bars_b = ax.bar(
        [i - width / 2 for i in x], base_vals, width,
        label=f"{baseline_label} (n={len(base_results)})", color="#d62728",
    )
    bars_m = ax.bar(
        [i + width / 2 for i in x], mem_vals, width,
        label=f"{memgpt_label} (n={len(mem_results)})", color="#2ca02c",
    )
    for bars in (bars_b, bars_m):
        for rect in bars:
            ax.annotate(
                f"{rect.get_height():.0%}",
                (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                textcoords="offset points", xytext=(0, 4),
                ha="center", fontsize=9,
            )

    # Marca las métricas donde "menos es mejor" para no malinterpretar.
    tick_labels = [
        f"{lab}\n(↓ mejor)" if lb else lab
        for lab, lb in zip(labels, lower_better)
    ]
    ax.set_xticks(list(x))
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel("Valor (proporción)")
    ax.set_ylim(0, 1.12)
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right")
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    print(f"Gráfica escrita en {output}")
    for key, bv, mv in zip(keys, base_vals, mem_vals):
        print(f"  {key:14s} baseline={bv:.3f}  memgpt={mv:.3f}  Δ={mv - bv:+.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gráfica de barras comparativa (DMR / Document QA): baseline vs MemGPT"
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--memgpt", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("runs/metrics-comparison.png")
    )
    parser.add_argument("--title", type=str, default="Baseline vs MemGPT")
    parser.add_argument("--baseline-label", type=str, default="Baseline (sin memoria)")
    parser.add_argument("--memgpt-label", type=str, default="MemGPT")
    args = parser.parse_args()

    for p in (args.baseline, args.memgpt):
        if not p.exists():
            parser.error(f"no existe el fichero {p}")

    try:
        import matplotlib  # noqa: F401
    except ModuleNotFoundError:
        parser.error(
            "matplotlib no está instalado. Instala el extra: "
            "`uv sync --extra viz` o ejecuta con `uv run --extra viz ...`."
        )

    _plot(
        args.baseline,
        args.memgpt,
        args.output,
        title=args.title,
        baseline_label=args.baseline_label,
        memgpt_label=args.memgpt_label,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
