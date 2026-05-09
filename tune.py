import argparse
import contextlib
import copy
import json
import os

import optuna
import torch

from src.experiments import add_experiment_args, run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter tuning for federated GraphSAGE."
    )
    add_experiment_args(parser)
    parser.set_defaults(global_rounds=100, eval_every=10)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--study-name", type=str, default="fedgraphsage")
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
        help="Optional path where the best trial summary is written as JSON.",
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
    return parser


def suggest_hyperparameters(
    trial: optuna.Trial,
    base_args: argparse.Namespace,
) -> argparse.Namespace:
    args = copy.deepcopy(base_args)
    args.lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    args.weight_decay = trial.suggest_categorical(
        "weight_decay",
        [0.0, 1e-6, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3],
    )
    args.hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])
    args.dropout = trial.suggest_float("dropout", 0.0, 0.6, step=0.1)
    args.local_epochs = trial.suggest_categorical("local_epochs", [1, 5, 10, 20])
    args.class_weighting = trial.suggest_categorical(
        "class_weighting",
        ["local", "none"],
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

        def report_progress(round_idx: int, val_metrics: dict[str, float]) -> None:
            trial.report(val_metrics[metric], step=round_idx)
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


def write_results_json(
    path: str,
    study: optuna.Study,
    final_result: dict[str, object] | None,
) -> None:
    best = study.best_trial
    payload = {
        "best_trial": best.number,
        "best_value": best.value,
        "best_params": best.params,
        "best_user_attrs": best.user_attrs,
        "final_result": final_result,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def main() -> None:
    args = build_parser().parse_args()
    if args.n_trials <= 0:
        raise ValueError("--n-trials must be > 0.")
    if args.pruner_warmup_rounds < 0:
        raise ValueError("--pruner-warmup-rounds must be >= 0.")
    
    print(f"CUDA disponibile: {torch.cuda.is_available()}")
    print(f"Numero GPU: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"GPU corrente: {torch.cuda.current_device()}")
        print(f"Nome GPU: {torch.cuda.get_device_name(0)}")

    direction = "minimize" if args.selection_metric == "loss" else "maximize"
    sampler_seed = args.sampler_seed if args.sampler_seed is not None else args.seed
    sampler = optuna.samplers.TPESampler(seed=sampler_seed)
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
        print("\nFinal run with best hyperparameters")
        best_args = apply_best_params(args, study.best_trial.params)
        final_result = run_experiment(best_args, verbose=True, evaluate_test=True)
        print(
            "\nTest metrics at validation-selected round "
            f"{final_result['best_round']}: "
            f"{json.dumps(final_result['best_test_metrics'], sort_keys=True)}"
        )
        print(
            "Final-round test metrics: "
            f"{json.dumps(final_result['final_test_metrics'], sort_keys=True)}"
        )

    if args.results_json is not None:
        write_results_json(args.results_json, study, final_result)
        print(f"\nSaved tuning summary to {args.results_json}")


if __name__ == "__main__":
    main()
'''
first optuna hyperparameter tuning for federated GraphSAGE.

Best trial
  number: 25
  value: 0.923857
  params: {"class_weighting": "none", "dropout": 0.0, "hidden_dim": 128, "local_epochs": 20, "lr": 0.0026759617806772903, "weight_decay": 1e-05}
  best_round: 30
  best_val: {"accuracy": 0.9823083403538332, "balanced_accuracy": 0.9011410474777222, "loss": 0.09893313214212056, "macro_f1": 0.9238565564155579, "macro_precision": 0.9498133659362793, "macro_recall": 0.9011410474777222}
  
Test metrics at validation-selected round 30: {"accuracy": 0.9725343320848939, "balanced_accuracy": 0.8664166927337646, "loss": 0.12187391743830467, "macro_f1": 0.8895187377929688, "macro_precision": 0.9164140224456787, "macro_recall": 0.8664166927337646}
Final-round test metrics: {"accuracy": 0.9667082813150228, "balanced_accuracy": 0.8632805347442627, "loss": 0.17145857933359906, "macro_f1": 0.8712949752807617, "macro_precision": 0.8797491788864136, "macro_recall": 0.8632805347442627}

'''