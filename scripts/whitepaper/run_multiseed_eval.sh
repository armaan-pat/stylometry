#!/usr/bin/env bash
# Multi-seed held-out Claude+Gemini eval for the whole version lineage.
# Each (ckpt, seed) reruns the exact heldoutCG ablation with a different probe draw
# (sender/enroll/query/other sampling), so we can report mean±std over probe draws
# on top of the within-draw 1000x bootstrap. Eval-only; no training.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export WANDB_MODE=disabled
CFG=configs/experiments/_v12_heldout_eval.yaml
OUT=results/whitepaper/multiseed
LOG=results/whitepaper/logs
SEEDS="${SEEDS:-0 1 2 3 4}"
mkdir -p "$OUT" "$LOG"

declare -A CKPT=(
  [v11]=runs/v11/lora/checkpoint_best.pt
  [v12]=runs/v12/lora/checkpoint_best.pt
  [v13]=runs/v13/lora/checkpoint_best.pt
  [v14]=runs/v14/lora/checkpoint_best.pt
  [v14b]=runs/v14b/lora/checkpoint_best.pt
)
ORDER="v11 v12 v13 v14 v14b"

for ver in $ORDER; do
  for s in $SEEDS; do
    tag="heldoutCG_${ver}_s${s}"
    out="$OUT/${tag}.json"
    if [ -f "$out" ]; then echo "[skip] $out exists"; continue; fi
    echo "=== $ver seed $s -> $out ==="
    python scripts/ablate_adaptive_scorers.py \
      --config "$CFG" --checkpoint "${CKPT[$ver]}" \
      --split synthetic --rank-by tpr1 --bootstrap 1000 --seed "$s" \
      --out-dir "$OUT" --tag "$tag" \
      > "$LOG/${tag}.log" 2>&1 \
      && echo "[ok] $tag" || echo "[FAIL] $tag (see $LOG/${tag}.log)"
  done
done
echo "ALL_DONE_MULTISEED"
