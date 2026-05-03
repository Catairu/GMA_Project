from src.experiments import add_experiment_args, build_arg_parser, run_experiment

__all__ = ["add_experiment_args", "build_arg_parser", "run_experiment"]


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
