Federated GraphSAGE for fraud graph datasets.

## Project structure

```text
.
├── main.py                         # Training CLI entry point
├── tune.py                         # Optuna tuning CLI entry point
└── src/
    ├── datasets/                   # Dataset loading and preprocessing
    ├── experiments/                # Experiment orchestration and CLI arguments
    ├── federated/                  # Federated client/server logic
    ├── models/                     # Neural network architectures
    ├── training/                   # Splits, metrics, and training utilities
    └── utilities/                  # Graph partitioning methods
```

## Run training

```bash
python3 main.py --dataset amazon --partition-method metis
```

## Run tuning

```bash
python3 tune.py --dataset amazon --partition-method random --n-trials 30
```
