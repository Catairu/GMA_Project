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

'''
experiments are dumped here

baseline 1 client (global graph only, no partitioning, no class weighting):
    python main.py --dataset amazon --n-clients 1 --lr 0.003 --weight-decay 1e-05 --hidden-dim 128 --dropout 0.1 --local-epochs 20 --global-rounds 400 --partition-method kmeans --class-weighting none
    best_round: 320
    best_val_metrics: {'loss': 9.09607982635498, 'accuracy': 0.957286432160804, 'balanced_accuracy': 0.6890243887901306, 'macro_precision': 0.9780739545822144, 'macro_recall': 0.6890243887901306, 'macro_f1': 0.7631275057792664}
    best_test_metrics: {'loss': 7.801283836364746, 'accuracy': 0.9531380753138076, 'balanced_accuracy': 0.6634116172790527, 'macro_precision': 0.9671403765678406, 'macro_recall': 0.6634116172790527, 'macro_f1': 0.7331738471984863}
baseline n isolated clients (no communication, each client trains on its own subgraph only):
    python main.py --dataset amazon --n-clients 10 --lr 0.0026 --weight-decay 1e-05 --hidden-dim 128 --dropout 0.1 --local-epochs 40 --global-rounds 1 --partition-method kmeans --class-weighting none --dont-aggregate
    best_round: 1
    best_val_metrics: {'loss': 2.3845528578220945, 'accuracy': 0.9519797809604044, 'balanced_accuracy': 0.7538033723831177, 'macro_precision': 0.8196357488632202, 'macro_recall': 0.7538033723831177, 'macro_f1': 0.7822123765945435}
    best_test_metrics: {'loss': 3.355046176176814, 'accuracy': 0.943404078235539, 'balanced_accuracy': 0.7427372336387634, 'macro_precision': 0.7943984270095825, 'macro_recall': 0.7427372336387634, 'macro_f1': 0.7655206322669983}

'''