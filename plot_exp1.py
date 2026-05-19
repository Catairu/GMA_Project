#!/usr/bin/env python3
"""Plot Esperimento 1: scaling N client.

Usage:
    python plot_exp1.py --results results_exp1/amazon/results.json
    python plot_exp1.py --results results_exp1/amazon/results.json --metric fraud_f1
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


PARTITION_STYLES = {
    "random":    {"color": "#2196F3", "label": "Random"},
    "dirichlet": {"color": "#FF5722", "label": "Dirichlet (α=0.5)"},
    "kmeans":    {"color": "#4CAF50", "label": "K-Means++"},
}

METRIC_LABELS = {
    "fraud_f1": "F1 classe frode (classe 1)",
    "macro_f1": "F1-macro",
    "balanced_accuracy": "Balanced accuracy",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True, help="Path al results.json di run_exp1.py")
    p.add_argument("--metric", default="fraud_f1", choices=list(METRIC_LABELS))
    p.add_argument("--out", default=None, help="Salva figura (es. exp1.pdf). Se omesso, mostra interattivo.")
    return p.parse_args()


def _extract(entries: list[dict], metric: str) -> list[float]:
    vals = []
    for e in entries:
        m = e.get("metrics") or {}
        v = m.get(metric)
        if v is not None:
            vals.append(v)
    return vals


def main() -> None:
    cfg = parse_args()
    results_path = Path(cfg.results)
    if not results_path.exists():
        raise FileNotFoundError(f"File non trovato: {results_path}")

    with open(results_path) as f:
        results = json.load(f)

    metric = cfg.metric
    ylabel = METRIC_LABELS.get(metric, metric)

    n_values = sorted(int(k) for k in results["federated"])
    partitions = list(PARTITION_STYLES)

    # -----------------------------------------------------------------------
    # Centralized ceiling
    # -----------------------------------------------------------------------
    cent_vals = _extract(results["centralized"], metric)
    cent_mean = float(np.mean(cent_vals)) if cent_vals else None
    cent_std = float(np.std(cent_vals)) if cent_vals else None

    # -----------------------------------------------------------------------
    # Federated: una curva per partizione
    # -----------------------------------------------------------------------
    fed_data: dict[str, tuple[list[float], list[float]]] = {}
    for part in partitions:
        means, stds = [], []
        for n in n_values:
            entries = results["federated"].get(str(n), {}).get(part, [])
            vals = _extract(entries, metric)
            means.append(float(np.mean(vals)) if vals else float("nan"))
            stds.append(float(np.std(vals)) if vals else float("nan"))
        fed_data[part] = (means, stds)

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.array(n_values, dtype=float)

    # Federated curves
    for part, (means, stds) in fed_data.items():
        style = PARTITION_STYLES[part]
        means_arr = np.array(means)
        stds_arr = np.array(stds)
        ax.plot(x, means_arr, "o-", color=style["color"], label=f"FedSAGE – {style['label']}", linewidth=2, markersize=6)
        ax.fill_between(x, means_arr - stds_arr, means_arr + stds_arr, color=style["color"], alpha=0.15)

    # Centralized ceiling (tratteggiata arancione/nera)
    if cent_mean is not None:
        ax.axhline(cent_mean, linestyle="--", color="#FF9800", linewidth=2, label="Centralized (ceiling)")
        if cent_std is not None and cent_std > 0:
            ax.axhspan(cent_mean - cent_std, cent_mean + cent_std, color="#FF9800", alpha=0.10)

    # Assi
    ax.set_xscale("log")
    ax.set_xticks(n_values)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel("Numero di client N (scala log)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    dataset_label = results_path.parent.name.capitalize()
    ax.set_title(f"FedGraphSAGE – scaling in N  ({dataset_label})", fontsize=13)
    ax.legend(fontsize=10, loc="lower left")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    if cfg.out:
        out_path = Path(cfg.out)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Figura salvata: {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
