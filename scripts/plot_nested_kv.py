"""Gráfica comparativa del benchmark Nested KV (Fase 8).

Reproduce el estilo de la Figura 7 del paper MemGPT (arXiv:2310.08560):
accuracy en función del nivel de anidamiento, una línea por run. Sirve para
visualizar el contraste baseline (todo inline, sin memoria) vs MemGPT
(recuperación por clave desde Archival).

Uso típico:

    uv run --extra viz scripts/plot_nested_kv.py \
        --baseline runs/nested-kv-baseline-gpt4o-mini.json \
        --memgpt   runs/nested-kv-memgpt-gpt4o-mini.json \
        --output   runs/nested-kv-gpt4o-mini.png

Requiere matplotlib (extra ``viz``): ``uv sync --extra viz`` o el flag
``uv run --extra viz`` como arriba. Las métricas se recalculan desde el campo
``results`` de cada JSON (no del resumen ``accuracy_by_level``) para que los
dumps parciales/abortados también se grafiquen correctamente.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def _accuracy_by_level(path: Path) -> tuple[dict[int, float], dict[int, int], float, str]:
    """Devuelve (accuracy_por_nivel, n_por_nivel, accuracy_global, mode).

    Recalcula desde ``results`` para ser robusto ante runs abortados, donde el
    resumen guardado podría no reflejar el último estado.
    """
    data = json.loads(path.read_text())
    results = data.get("results", [])
    if not results:
        raise ValueError(f"{path} no tiene 'results' que graficar")

    hits: dict[int, int] = defaultdict(int)
    totals: dict[int, int] = defaultdict(int)
    for r in results:
        lv = r["nesting_level"]
        totals[lv] += 1
        if r["correct"]:
            hits[lv] += 1

    by_level = {lv: hits[lv] / totals[lv] for lv in totals}
    overall = sum(hits.values()) / sum(totals.values())
    mode = data.get("mode", path.stem)
    return by_level, dict(totals), overall, mode


def _plot(
    baseline: Path,
    memgpt: Path,
    output: Path,
    *,
    title: str,
    baseline_label: str | None,
    memgpt_label: str | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")  # backend sin display: escribimos a fichero.
    import matplotlib.pyplot as plt

    series = []
    for path, default_label, color, marker in (
        (baseline, baseline_label or "Baseline (sin memoria)", "#d62728", "s"),
        (memgpt, memgpt_label or "MemGPT", "#2ca02c", "o"),
    ):
        by_level, totals, overall, mode = _accuracy_by_level(path)
        label = f"{default_label} (global {overall:.0%}, n={sum(totals.values())})"
        series.append((by_level, totals, label, color, marker))

    # Eje X = unión de niveles presentes en ambos runs.
    levels = sorted({lv for by_level, *_ in series for lv in by_level})

    fig, ax = plt.subplots(figsize=(8, 5))
    for by_level, totals, label, color, marker in series:
        xs = [lv for lv in levels if lv in by_level]
        ys = [by_level[lv] for lv in xs]
        ax.plot(xs, ys, marker=marker, color=color, linewidth=2, markersize=8, label=label)
        # Anota n por punto si difiere del resto (útil con runs parciales).
        for x, y in zip(xs, ys):
            ax.annotate(
                f"{y:.0%}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
                color=color,
            )

    ax.set_xlabel("Nivel de anidamiento (saltos de indirección)")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_xticks(levels)
    ax.set_ylim(-0.05, 1.08)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="lower left")
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    print(f"Gráfica escrita en {output}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gráfica comparativa Nested KV: baseline vs MemGPT"
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="JSON del run baseline (--baseline en run_nested_kv.py).",
    )
    parser.add_argument(
        "--memgpt",
        type=Path,
        required=True,
        help="JSON del run MemGPT.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/nested-kv-comparison.png"),
        help="Ruta de salida del PNG (default: runs/nested-kv-comparison.png).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Nested KV: baseline vs MemGPT",
        help="Título de la gráfica.",
    )
    parser.add_argument("--baseline-label", type=str, default=None)
    parser.add_argument("--memgpt-label", type=str, default=None)
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
