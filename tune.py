import argparse
import contextlib
import copy
import json
import os
from typing import Any

import optuna
import torch

from src.experiments import add_experiment_args, run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter tuning for federated GraphSAGE."
    )

    add_experiment_args(parser)

    # Default più sobri per tuning federato.
    # Puoi comunque sovrascriverli da terminale.
    parser.set_defaults(
        global_rounds=100,
        eval_every=10,
        local_epochs=1,
        class_weighting="none",
    )

    parser.add_argument("--n-trials", type=int, default=30)

    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Maximum tuning time in seconds.",
    )

    parser.add_argument(
        "--study-name",
        type=str,
        default="fedgraphsage",
    )

    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optional Optuna storage URI, e.g. sqlite:///optuna.db.",
    )

    parser.add_argument(
        "--sampler-seed",
        type=int,
        default=None,
        help="Seed for Optuna's sampler. Defaults to --seed.",
    )

    parser.add_argument(
        "--pruner-warmup-rounds",
        type=int,
        default=20,
        help="Number of global rounds before Optuna can prune a trial.",
    )

    parser.add_argument(
        "--results-json",
        type=str,
        default=None,
        help="Optional path where tuning results are written as JSON.",
    )

    parser.add_argument(
        "--verbose-trials",
        action="store_true",
        help="Print the full training log for every trial.",
    )

    parser.add_argument(
        "--skip-final-test",
        action="store_true",
        help="Do not rerun the best configuration on the test split after tuning.",
    )

    parser.add_argument(
        "--final-seeds",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Optional list of seeds for final evaluation, e.g. "
            "--final-seeds 42 43 44 45 46. "
            "If omitted, the final run uses --seed only."
        ),
    )

    return parser


def suggest_hyperparameters(
    trial: optuna.Trial,
    base_args: argparse.Namespace,
) -> argparse.Namespace:
    """
    Suggerisce solo iperparametri del modello/ottimizzazione.

    Nota importante:
    - local_epochs NON viene tunato qui.
    - class_weighting NON viene tunato qui.

    Questi due parametri incidono molto sull'interpretazione dell'esperimento
    federato, quindi è meglio fissarli da CLI.
    """
    args = copy.deepcopy(base_args)

    args.lr = trial.suggest_float(
        "lr",
        1e-5,
        5e-3,
        log=True,
    )

    args.weight_decay = trial.suggest_categorical(
        "weight_decay",
        [0.0, 1e-6, 1e-5, 1e-4, 5e-4, 1e-3],
    )

    args.hidden_dim = trial.suggest_categorical(
        "hidden_dim",
        [32, 64, 128],
    )

    args.dropout = trial.suggest_float(
        "dropout",
        0.0,
        0.5,
        step=0.1,
    )

    if base_args.partition_method == "dirichlet":
        args.dirichlet_alpha = trial.suggest_float(
            "dirichlet_alpha",
            0.1,
            10.0,
            log=True,
        )

    return args


def apply_best_params(
    args: argparse.Namespace,
    params: dict[str, object],
) -> argparse.Namespace:
    best_args = copy.deepcopy(args)

    for name, value in params.items():
        setattr(best_args, name, value)

    return best_args


def objective_factory(base_args: argparse.Namespace):
    metric = base_args.selection_metric

    def objective(trial: optuna.Trial) -> float:
        trial_args = suggest_hyperparameters(trial, base_args)

        def report_progress(
            round_idx: int,
            val_metrics: dict[str, float],
        ) -> None:
            value = val_metrics[metric]
            trial.report(value, step=round_idx)

            if trial.should_prune():
                raise optuna.TrialPruned()

        if base_args.verbose_trials:
            result = run_experiment(
                trial_args,
                verbose=True,
                evaluate_test=False,
                progress_callback=report_progress,
            )
        else:
            stream = open(os.devnull, "w", encoding="utf-8")
            with stream, contextlib.redirect_stdout(stream):
                result = run_experiment(
                    trial_args,
                    verbose=False,
                    evaluate_test=False,
                    progress_callback=report_progress,
                )

        best_val_metrics = result["best_val_metrics"]
        best_round = int(result["best_round"])

        for name, value in best_val_metrics.items():
            trial.set_user_attr(f"best_val_{name}", value)

        trial.set_user_attr("best_round", best_round)

        return float(best_val_metrics[metric])

    return objective


def print_best_trial(study: optuna.Study) -> None:
    best = study.best_trial

    print("\nBest trial")
    print(f"  number: {best.number}")
    print(f"  value: {best.value:.6f}")
    print(f"  params: {json.dumps(best.params, sort_keys=True)}")
    print(f"  best_round: {best.user_attrs.get('best_round')}")

    best_val = {
        key.removeprefix("best_val_"): value
        for key, value in best.user_attrs.items()
        if key.startswith("best_val_")
    }

    print(f"  best_val: {json.dumps(best_val, sort_keys=True)}")


def _trial_to_dict(trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
    return {
        "number": trial.number,
        "state": str(trial.state),
        "value": trial.value,
        "params": trial.params,
        "user_attrs": trial.user_attrs,
    }


def write_results_json(
    path: str,
    study: optuna.Study,
    final_result: dict[str, object] | list[dict[str, object]] | None,
) -> None:
    best = study.best_trial

    payload = {
        "best_trial": best.number,
        "best_value": best.value,
        "best_params": best.params,
        "best_user_attrs": best.user_attrs,
        "all_trials": [_trial_to_dict(trial) for trial in study.trials],
        "final_result": final_result,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def print_cuda_info() -> None:
    print(f"CUDA disponibile: {torch.cuda.is_available()}")
    print(f"Numero GPU: {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        print(f"GPU corrente: {torch.cuda.current_device()}")
        print(f"Nome GPU: {torch.cuda.get_device_name(0)}")


def run_final_evaluation(
    args: argparse.Namespace,
    best_params: dict[str, object],
) -> dict[str, object] | list[dict[str, object]]:
    """
    Esegue il final run con gli iperparametri migliori.

    Se args.final_seeds è None, esegue un solo run con args.seed.
    Se args.final_seeds contiene più seed, esegue un run per ogni seed.
    """
    seeds = args.final_seeds if args.final_seeds else [args.seed]

    final_results: list[dict[str, object]] = []

    for seed in seeds:
        print(f"\nFinal run with best hyperparameters | seed={seed}")

        final_args = apply_best_params(args, best_params)
        final_args.seed = seed

        result = run_experiment(
            final_args,
            verbose=True,
            evaluate_test=True,
        )

        print(
            "\nTest metrics at validation-selected round "
            f"{result['best_round']}: "
            f"{json.dumps(result['best_test_metrics'], sort_keys=True)}"
        )

        print(
            "Final-round test metrics: "
            f"{json.dumps(result['final_test_metrics'], sort_keys=True)}"
        )

        final_results.append(
            {
                "seed": seed,
                "best_round": result["best_round"],
                "best_val_metrics": result["best_val_metrics"],
                "best_test_metrics": result["best_test_metrics"],
                "final_val_metrics": result["final_val_metrics"],
                "final_test_metrics": result["final_test_metrics"],
            }
        )

    if len(final_results) == 1:
        return final_results[0]

    return final_results


def main() -> None:
    args = build_parser().parse_args()

    if args.n_trials <= 0:
        raise ValueError("--n-trials must be > 0.")

    if args.pruner_warmup_rounds < 0:
        raise ValueError("--pruner-warmup-rounds must be >= 0.")

    print_cuda_info()

    print("\nTuning configuration")
    print(f"  dataset: {args.dataset}")
    print(f"  n_clients: {args.n_clients}")
    print(f"  partition_method: {args.partition_method}")
    print(f"  split_mode: {getattr(args, 'split_mode', 'not_available')}")
    print(f"  train_ratio: {args.train_ratio}")
    print(f"  val_ratio: {args.val_ratio}")
    print(f"  global_rounds: {args.global_rounds}")
    print(f"  eval_every: {args.eval_every}")
    print(f"  local_epochs: {args.local_epochs}")
    print(f"  class_weighting: {args.class_weighting}")
    print(f"  selection_metric: {args.selection_metric}")
    print(f"  seed: {args.seed}")

    direction = "minimize" if args.selection_metric == "loss" else "maximize"

    sampler_seed = args.sampler_seed if args.sampler_seed is not None else args.seed

    sampler = optuna.samplers.TPESampler(
        seed=sampler_seed,
    )

    pruner = optuna.pruners.MedianPruner(
        n_warmup_steps=args.pruner_warmup_rounds,
        interval_steps=max(args.eval_every, 1),
    )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=args.storage is not None,
    )

    study.optimize(
        objective_factory(args),
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
    )

    print_best_trial(study)

    final_result = None

    if not args.skip_final_test:
        final_result = run_final_evaluation(
            args=args,
            best_params=study.best_trial.params,
        )

    if args.results_json is not None:
        write_results_json(
            path=args.results_json,
            study=study,
            final_result=final_result,
        )

        print(f"\nSaved tuning summary to {args.results_json}")


if __name__ == "__main__":
    main()