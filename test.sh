#!/usr/bin/env bash
set -euo pipefail

# =========================
# Configurazione esperimento
# =========================

DATASETS=("amazon" "yelp")
CLIENTS=(1 5 10 20)

PARTITION_METHOD="kmeans"

RESULT_DIR="${RESULT_DIR:-results_kmeans}"
N_TRIALS="${N_TRIALS:-20}"
GLOBAL_ROUNDS="${GLOBAL_ROUNDS:-150}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-3}"
TRAIN_RATIO="${TRAIN_RATIO:-0.8}"
VAL_RATIO="${VAL_RATIO:-0.1}"
CLASS_WEIGHTING="${CLASS_WEIGHTING:-local}"
SEED="${SEED:-42}"

mkdir -p "$RESULT_DIR"

SUMMARY_CSV="$RESULT_DIR/summary.csv"
SUMMARY_MD="$RESULT_DIR/summary.md"

echo "dataset,n_clients,test_macro_recall,test_balanced_accuracy,json_path" > "$SUMMARY_CSV"

# =========================
# Run tuning
# =========================

for DATASET in "${DATASETS[@]}"; do
  for N_CLIENTS in "${CLIENTS[@]}"; do

    RUN_NAME="${DATASET}_${PARTITION_METHOD}_${N_CLIENTS}clients"
    JSON_PATH="$RESULT_DIR/${RUN_NAME}.json"

    echo ""
    echo "============================================================"
    echo "Running: dataset=${DATASET} | partition=${PARTITION_METHOD} | clients=${N_CLIENTS}"
    echo "Output JSON: ${JSON_PATH}"
    echo "============================================================"

    python tune.py \
      --dataset "$DATASET" \
      --partition-method "$PARTITION_METHOD" \
      --n-clients "$N_CLIENTS" \
      --train-ratio "$TRAIN_RATIO" \
      --val-ratio "$VAL_RATIO" \
      --global-rounds "$GLOBAL_ROUNDS" \
      --local-epochs "$LOCAL_EPOCHS" \
      --class-weighting "$CLASS_WEIGHTING" \
      --n-trials "$N_TRIALS" \
      --seed "$SEED" \
      --results-json "$JSON_PATH"

    python - "$DATASET" "$N_CLIENTS" "$JSON_PATH" "$SUMMARY_CSV" <<'PY'
import csv
import json
import sys
from pathlib import Path

dataset = sys.argv[1]
n_clients = int(sys.argv[2])
json_path = Path(sys.argv[3])
summary_csv = Path(sys.argv[4])

with json_path.open("r", encoding="utf-8") as f:
    payload = json.load(f)

final_result = payload.get("final_result")

if final_result is None:
    raise RuntimeError(
        f"No final_result found in {json_path}. "
        "Do not use --skip-final-test if you want test metrics."
    )

# Se final_result è una lista perché hai usato --final-seeds,
# qui facciamo la media dei best_test_metrics sui seed.
if isinstance(final_result, list):
    macro_recalls = []
    balanced_accs = []

    for item in final_result:
        test_metrics = item["best_test_metrics"]
        macro_recalls.append(float(test_metrics["macro_recall"]))
        balanced_accs.append(float(test_metrics["balanced_accuracy"]))

    test_macro_recall = sum(macro_recalls) / len(macro_recalls)
    test_balanced_accuracy = sum(balanced_accs) / len(balanced_accs)

else:
    test_metrics = final_result["best_test_metrics"]
    test_macro_recall = float(test_metrics["macro_recall"])
    test_balanced_accuracy = float(test_metrics["balanced_accuracy"])

with summary_csv.open("a", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            dataset,
            n_clients,
            test_macro_recall,
            test_balanced_accuracy,
            str(json_path),
        ]
    )
PY

  done
done

# =========================
# Creazione tabella finale
# =========================

python - "$SUMMARY_CSV" "$SUMMARY_MD" <<'PY'
import csv
import sys
from pathlib import Path

summary_csv = Path(sys.argv[1])
summary_md = Path(sys.argv[2])

rows = []

with summary_csv.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

def get_row(dataset: str, n_clients: int):
    for row in rows:
        if row["dataset"] == dataset and int(row["n_clients"]) == n_clients:
            return row
    return None

clients = [1, 5, 10, 20]

lines = []
lines.append("# KMeans partitioning - Test results")
lines.append("")
lines.append(
    "| Amazon - Clients | Amazon Test Macro Recall | Amazon Test Balanced Acc |  | Yelp - Clients | Yelp Test Macro Recall | Yelp Test Balanced Acc |"
)
lines.append(
    "|---:|---:|---:|---|---:|---:|---:|"
)

for n in clients:
    amazon = get_row("amazon", n)
    yelp = get_row("yelp", n)

    if amazon is None:
        amazon_macro = "NA"
        amazon_bal = "NA"
    else:
        amazon_macro = f"{float(amazon['test_macro_recall']):.4f}"
        amazon_bal = f"{float(amazon['test_balanced_accuracy']):.4f}"

    if yelp is None:
        yelp_macro = "NA"
        yelp_bal = "NA"
    else:
        yelp_macro = f"{float(yelp['test_macro_recall']):.4f}"
        yelp_bal = f"{float(yelp['test_balanced_accuracy']):.4f}"

    lines.append(
        f"| {n} | {amazon_macro} | {amazon_bal} |  | {n} | {yelp_macro} | {yelp_bal} |"
    )

summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("")
print("============================================================")
print("FINAL TABLE")
print("============================================================")
print(summary_md.read_text(encoding="utf-8"))
print(f"Saved CSV: {summary_csv}")
print(f"Saved Markdown table: {summary_md}")
PY