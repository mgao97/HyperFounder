#!/usr/bin/env bash
# Leave-One-Dataset-Out (LODO) cross-validation: for each of the 8 datasets,
# pre-train on the OTHER 7, then linear-probe on the held-out one.
# This directly measures cross-domain transferability of the structural encoding.
# Usage: bash scripts/run_lodo.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-python3}"
cd "$ROOT"

DATASETS=(news20 ca_cora cc_cora cc_citeseer dblp4k_paper dblp4k_term dblp4k_conf imdb_aw)
OUT_ROOT="outputs/lodo"
mkdir -p "$OUT_ROOT"

for i in "${!DATASETS[@]}"; do
  target="${DATASETS[$i]}"
  # pool = all datasets except the held-out target
  pool=("${DATASETS[@]:0:$i}" "${DATASETS[@]:$((i+1))}")
  fold_dir="$OUT_ROOT/fold_$i"
  echo "=== LODO fold $i : hold out '$target' (pre-train on ${#pool[@]} datasets) ==="
  $PY main.py \
    --pretrain_datasets "${pool[@]}" \
    --epochs "${EPOCHS:-50}" \
    --max_nodes "${MAX_NODES:-3000}" \
    --output_dir "$fold_dir" \
    --device "${DEVICE:-cpu}"
  $PY evaluate.py \
    --checkpoint "$fold_dir/hypergfse_encoder.pt" \
    --eval_datasets "$target" \
    --folds "${FOLDS:-10}" \
    --output "$fold_dir/eval_${target}.csv"
done

echo "=== Aggregating LODO results ==="
$PY - <<'PYEOF'
import csv, glob, os
rows=[]
for f in sorted(glob.glob("outputs/lodo/fold_*/eval_*.csv")):
    with open(f) as fh:
        r=list(csv.DictReader(fh))[0]
    rows.append(r)
out="outputs/lodo/aggregate.csv"
with open(out,"w",newline="") as fh:
    w=csv.DictWriter(fh, fieldnames=["dataset","domain","nodes","classes","raw_acc","raw_std","pse_acc","pse_std","delta"])
    w.writeheader(); w.writerows(rows)
print(f"LODO aggregate -> {out}  ({len(rows)} folds)")
PYEOF
