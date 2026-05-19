# Federated GraphSAGE for Fraud Detection

Graph Mining project — Master's degree.

Federated learning with GraphSAGE on graph-based fraud detection datasets (Amazon, Yelp).
Experiments compare different graph partitioning strategies (Random, METIS, K-Means, Dirichlet)
across varying numbers of federated clients, using FedAvg aggregation.

---

## Project Structure

```
.
├── main.py              # Single experiment entry point (CLI)
├── tune.py              # Optuna hyperparameter tuning
├── run_exp1.py          # Exp 1: scaling N clients (multi-seed + baselines)
├── plot_exp1.py         # Plot results from run_exp1.py
├── test.sh              # Batch tuning runs across datasets and clients
├── analysis.ipynb       # Analysis and result exploration notebook
│
├── src/
│   ├── datasets/        # Dataset loading (Amazon, Yelp, Cora via DGL)
│   ├── experiments/
│   │   ├── federated_graphsage.py  # FedAvg + GraphSAGE training loop
│   │   └── centralized.py          # Centralized baseline
│   ├── federated/       # FedAvg server and client
│   ├── models/          # GraphSAGE classifier (DGL)
│   ├── training/        # Metrics, class weights, train/val/test splits
│   └── utilities/       # Graph partitioning algorithms
│
├── data/                # Auto-downloaded dataset files (gitignored)
└── results_*/           # Experiment outputs — JSON, CSV, plots (gitignored)
```

---

## Setup

Requires [uv](https://docs.astral.sh/uv/). Install it if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install the project dependencies:

```bash
uv sync
```

This creates `.venv/` and installs all pinned dependencies (DGL 1.1.2, PyTorch 2.1.2, etc.).

> **GPU note:** the default PyTorch wheel is CPU. For CUDA support see the [PyTorch install guide](https://pytorch.org/get-started/locally/) and replace the `torch` wheel in `pyproject.toml` accordingly.

---

## How to Run

All scripts are run with `uv run <script>` (no manual activation needed).

### Single experiment

```bash
uv run main.py \
  --dataset amazon \
  --n-clients 5 \
  --partition-method random \
  --global-rounds 200 \
  --local-epochs 3 \
  --seed 42
```

Key arguments:

| Argument              | Description                                              |
|-----------------------|----------------------------------------------------------|
| `--dataset`           | `amazon`, `yelp`, or `cora`                              |
| `--n-clients`         | Number of federated clients                              |
| `--partition-method`  | `random`, `metis`, `kmeans`, `dirichlet`                 |
| `--global-rounds`     | FL communication rounds (default: 100)                   |
| `--local-epochs`      | Local SGD epochs per round (default: 1)                  |
| `--hidden-dim`        | GraphSAGE hidden size (default: 128)                     |
| `--lr`                | Learning rate (default: 5e-4)                            |
| `--dropout`           | Dropout rate (default: 0.2)                              |
| `--class-weighting`   | `none`, `local`, or `global` (default: none)             |
| `--selection-metric`  | Metric for best-round selection (default: `macro_f1`)    |
| `--seed`              | Random seed (default: 42)                                |

---

### Hyperparameter tuning (Optuna)

Tunes `lr`, `weight_decay`, `hidden_dim`, `dropout` (and `dirichlet_alpha` if applicable)
using TPE sampler + Median pruner.

```bash
uv run tune.py \
  --dataset amazon \
  --n-clients 10 \
  --partition-method random \
  --n-trials 30 \
  --global-rounds 100 \
  --local-epochs 3 \
  --results-json results_random/amazon_random_10clients.json
```

Additional tuning arguments:

| Argument                | Description                                                 |
|-------------------------|-------------------------------------------------------------|
| `--n-trials`            | Number of Optuna trials (default: 30)                       |
| `--timeout`             | Max tuning time in seconds (optional)                       |
| `--pruner-warmup-rounds`| Rounds before pruning kicks in (default: 20)                |
| `--skip-final-test`     | Skip final re-run with best params on the test set          |
| `--final-seeds`         | Seeds for final evaluation, e.g. `--final-seeds 42 43 44`  |
| `--results-json`        | Path to save tuning results as JSON                         |
| `--storage`             | Optuna storage URI, e.g. `sqlite:///optuna.db`              |

---

### Batch tuning — `test.sh`

Runs `tune.py` for multiple datasets (`amazon`, `yelp`) and client counts (`5`, `10`, `20`),
then aggregates results into a `summary.csv` and a `summary.md` table.

```bash
# Random partitioning (default)
bash test.sh

# Different partition method
PARTITION_METHOD=kmeans bash test.sh

# Override parameters
PARTITION_METHOD=metis N_TRIALS=50 GLOBAL_ROUNDS=200 bash test.sh
```

Environment variables:

| Variable           | Default                      | Description                       |
|--------------------|------------------------------|-----------------------------------|
| `PARTITION_METHOD` | `random`                     | Partitioning strategy             |
| `RESULT_DIR`       | `results_<partition>`        | Output directory                  |
| `N_TRIALS`         | `20`                         | Optuna trials per run             |
| `GLOBAL_ROUNDS`    | `150`                        | FL communication rounds           |
| `LOCAL_EPOCHS`     | `3`                          | Local epochs per round            |
| `SEED`             | `42`                         | Random seed                       |

Results land in `results_<partition>/`:
- `<dataset>_<partition>_<N>clients.json` — full Optuna trial data
- `summary.csv` — aggregated test metrics
- `summary.md` — formatted Markdown table

---

### Experiment 1 — scaling N clients (`run_exp1.py`)

Sweeps over N ∈ `{2, 5, 10, 20, 50}`, multiple partition methods, and multiple seeds.
For each configuration runs:
- **Federated** (FedAvg + GraphSAGE)
- **Isolated local** (each client trains alone, floor baseline)
- **Centralized** (all data pooled, ceiling baseline)

```bash
uv run run_exp1.py \
  --dataset amazon \
  --n-values 2 5 10 20 \
  --partitions random dirichlet kmeans \
  --seeds 42 123 456 \
  --global-rounds 200 \
  --output-dir results_exp1
```

Results are saved to `results_exp1/<dataset>/results.json` with **automatic resume**:
already-completed (dataset, N, partition, seed) combinations are skipped on re-run.

---

### Plot Experiment 1 (`plot_exp1.py`)

```bash
uv run plot_exp1.py --results results_exp1/amazon/results.json
uv run plot_exp1.py --results results_exp1/amazon/results.json --metric fraud_f1
```

Available metrics: `fraud_f1`, `macro_f1`, `balanced_accuracy`.
Plots are saved next to the results JSON.

---

## Datasets

Downloaded automatically by DGL on first run and cached in `data/`.

| Dataset | Nodes  | Edges   | Task                     |
|---------|--------|---------|--------------------------|
| Amazon  | ~11k   | ~4.4M   | Fraud review detection   |
| Yelp    | ~45k   | ~3.8M   | Fraud review detection   |
| Cora    | ~2.7k  | ~5.4k   | Node classification      |

---

## Partitioning Strategies

| Method      | Description                                                    |
|-------------|----------------------------------------------------------------|
| `random`    | Uniform random node assignment                                 |
| `metis`     | Graph structure-aware partitioning (METIS algorithm)           |
| `kmeans`    | Feature-space K-Means++ clustering                             |
| `dirichlet` | Label-skewed partitioning via Dirichlet distribution (α=0.5)   |
