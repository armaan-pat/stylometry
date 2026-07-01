#!/usr/bin/env bash
# Master GPU pipeline — serializes all remaining GPU work so nothing contends on
# the single A40. Order:
#   0) wait for the multiseed eval (wp_multiseed tmux) to finish
#   1) novel-vendor GPU eval (needs wp_gen dataset done): v12 + v14b on Qwen/DeepSeek
#   2) train v14b reproduction seed (seed=1) + heldout multiseed eval
#   3) train 2x2 ablation (enron44, no-syn) + heldout multiseed eval
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
LOG=results/whitepaper/logs
mkdir -p "$LOG"

echo "[pipeline] $(date) waiting for wp_multiseed to finish..."
while tmux has-session -t wp_multiseed 2>/dev/null; do sleep 20; done
echo "[pipeline] $(date) multiseed done."

# ---- 1) novel-vendor GPU eval (wait for generation dataset) ----
echo "[pipeline] waiting for novel-vendor dataset..."
for _ in $(seq 1 180); do
  [ -d data/synthetic/enron_synthetic_novelvendor ] && break; sleep 20
done
if [ -d data/synthetic/enron_synthetic_novelvendor ]; then
  export WANDB_MODE=disabled
  OUT=results/whitepaper/novelvendor; mkdir -p "$OUT"
  declare -A CK=( [v12]=runs/v12/lora/checkpoint_best.pt [v14b]=runs/v14b/lora/checkpoint_best.pt )
  for ver in v12 v14b; do for s in 0 1 2; do
    out="$OUT/novelvendor_${ver}_s${s}.json"; [ -f "$out" ] && { echo "[skip] $out"; continue; }
    python scripts/ablate_adaptive_scorers.py \
      --config configs/experiments/_wp_novelvendor_eval.yaml --checkpoint "${CK[$ver]}" \
      --split synthetic --rank-by tpr1 --bootstrap 1000 --seed "$s" \
      --out-dir "$OUT" --tag "novelvendor_${ver}_s${s}" \
      > "$LOG/novelvendor_${ver}_s${s}.log" 2>&1 \
      && echo "[ok] novelvendor ${ver} s${s}" || echo "[FAIL] novelvendor ${ver} s${s}"
  done; done
else echo "[WARN] novel-vendor dataset never appeared; skipping novelvendor eval"; fi

# ---- 2 & 3) training chain ----
echo "[pipeline] starting training chain"
bash scripts/whitepaper/run_training.sh

echo "ALL_DONE_PIPELINE $(date)"
