from src.experiments import add_experiment_args, build_arg_parser, run_experiment

__all__ = ["add_experiment_args", "build_arg_parser", "run_experiment"]


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    results = run_experiment(args)
    print("Experiment completed. Results:")
    print(f"best_round: {results['best_round']}")
    print(f"best_val_metrics: {results['best_val_metrics']}")
    print(f"best_test_metrics: {results['best_test_metrics']}")
    print(f"final_val_metrics: {results['final_val_metrics']}")
    print(f"final_test_metrics: {results['final_test_metrics']}")

if __name__ == "__main__":
    main()
