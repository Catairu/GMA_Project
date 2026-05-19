#!/usr/bin/env python3
"""Experiment 1: FedGraphSAGE scaling with varying number of clients.

Configurations:
  - N ∈ {2, 5, 10, 20, 50}
  - Partitions: random, dirichlet, kmeans
  - ≥3 seeds per configuration
  - Baselines: centralized (ceiling) and isolated local (floor)

Results saved to --output-dir/<dataset>/results.json with automatic resume.
"""

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

from src.experiments.centralized import run_centralized
from src.experiments.federated_graphsage import run_experiment


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exp1: scaling N client")
    p.add_argument("--dataset", default="amazon", choices=["amazon", "yelp", "cora"])
    p.add_argument("--n-values", type=int, nargs="+", default=[2, 5, 10, 20, 50])
    p.add_argument("--partitions", nargs="+", default=["random", "dirichlet", "kmeans"])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    p.add_argument("--global-rounds", type=int, default=200)
    p.add_argument("--local-epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--dirichlet-alpha", type=float, default=0.5)
    p.add_argument("--output-dir", default="results_exp1")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_exp_args(cfg: argparse.Namespace, *, seed: int, n_clients: int, partition: str) -> Namespace:
    """Build a Namespace compatible with run_experiment / run_isolated_local."""
    return Namespace(
        dataset=cfg.dataset,
        n_clients=n_clients,
        global_rounds=cfg.global_rounds,
        local_epochs=cfg.local_epochs,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        eval_every=cfg.eval_every,
        partition_method=partition,
        dirichlet_alpha=cfg.dirichlet_alpha,
        class_weighting="local",
        selection_metric="macro_f1",
        seed=seed,
        dont_aggregate=False,
    )


def _done_seeds(entries: list[dict]) -> set[int]:
    return {e["seed"] for e in entries}


def save(results: dict, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    tmp.replace(path)


def load_or_init(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"centralized": [], "federated": {}}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = parse_args()
    out = Path(cfg.output_dir) / cfg.dataset
    out.mkdir(parents=True, exist_ok=True)
    results_file = out / "results.json"
    results = load_or_init(results_file)

    total = (
        len(cfg.seeds)                                             # centralized
        + len(cfg.n_values) * len(cfg.partitions) * len(cfg.seeds)  # federated
    )
    done = 0

    def progress(label: str) -> None:
        nonlocal done
        done += 1
        print(f"[{done}/{total}] {label}", flush=True)

    # -----------------------------------------------------------------------
    # Centralized baseline (ceiling)
    # -----------------------------------------------------------------------
    for seed in cfg.seeds:
        if seed in _done_seeds(results["centralized"]):
            done += 1
            continue
        progress(f"Centralized | seed={seed}")
        args = make_exp_args(cfg, seed=seed, n_clients=1, partition="random")
        metrics = run_centralized(args)
        results["centralized"].append({"seed": seed, "metrics": metrics})
        save(results, results_file)

    # -----------------------------------------------------------------------
    # Federated + Isolated local
    # -----------------------------------------------------------------------
    for n in cfg.n_values:
        sn = str(n)
        results["federated"].setdefault(sn, {})

        for partition in cfg.partitions:
            results["federated"][sn].setdefault(partition, [])
            done_fed = _done_seeds(results["federated"][sn][partition])

            for seed in cfg.seeds:
                args = make_exp_args(cfg, seed=seed, n_clients=n, partition=partition)

                if seed not in done_fed:
                    progress(f"Federated   | N={n:2d} | {partition:10s} | seed={seed}")
                    try:
                        result = run_experiment(args, verbose=False)
                        metrics = result["best_test_metrics"] or result["final_test_metrics"]
                    except Exception as e:
                        print(f"  ERROR: {e}", file=sys.stderr)
                        metrics = None
                    results["federated"][sn][partition].append({"seed": seed, "metrics": metrics})
                    save(results, results_file)
                else:
                    done += 1

    print(f"\nDone! Risultati in {results_file}")


if __name__ == "__main__":
    main()
