#!/usr/bin/env bash
# Novel-vendor generalization spot-check (API-bound; safe alongside GPU jobs).
# Generates forgeries from vendors OUTSIDE both the training set (GPT-4o-mini,
# Llama-3.1-70B) AND the held-out eval set (Claude-3.5-haiku, Gemini-2.5-flash):
#   Qwen-2.5-72B + DeepSeek-V3. Then scores the production model (v14b) and the
# prior best (v12) on this never-touched pool — directly testing the
# "a fully novel future vendor should be spot-checked" caveat.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export WANDB_MODE=disabled
LOG=results/whitepaper/logs
OUT=results/whitepaper/novelvendor
NVSET=data/synthetic/enron_synthetic_novelvendor
mkdir -p "$LOG" "$OUT"

# ---- 1) generate the novel-vendor pool (44 Enron senders, 2 vendors) ----
if [ ! -d "$NVSET" ]; then
  echo "=== generate novel-vendor forgeries (Qwen + DeepSeek) ==="
  python scripts/generate_synthetic_emails.py \
    --config configs/experiments/_v12_heldout_eval.yaml \
    --generators openrouter:qwen/qwen-2.5-72b-instruct,openrouter:deepseek/deepseek-chat \
    --from-split train --n-per-sender 8 --n-examples 5 \
    --cross-register-fraction 0.4 --cross-length-fraction 0.2 \
    --output "$NVSET" --seed 42 \
    > "$LOG/gen_novelvendor.log" 2>&1 \
    && echo "[ok] generated novel-vendor set" || { echo "[FAIL] generation (see $LOG/gen_novelvendor.log)"; exit 1; }
else echo "[skip] $NVSET exists"; fi

# ---- 2) score v12 (prior best) and v14b (production) on the novel pool ----
declare -A CKPT=( [v12]=runs/v12/lora/checkpoint_best.pt [v14b]=runs/v14b/lora/checkpoint_best.pt )
for ver in v12 v14b; do
  for s in 0 1 2; do
    out="$OUT/novelvendor_${ver}_s${s}.json"
    [ -f "$out" ] && { echo "[skip] $out"; continue; }
    python scripts/ablate_adaptive_scorers.py \
      --config configs/experiments/_wp_novelvendor_eval.yaml --checkpoint "${CKPT[$ver]}" \
      --split synthetic --rank-by tpr1 --bootstrap 1000 --seed "$s" \
      --out-dir "$OUT" --tag "novelvendor_${ver}_s${s}" \
      > "$LOG/novelvendor_${ver}_s${s}.log" 2>&1 \
      && echo "[ok] eval ${ver} s${s}" || echo "[FAIL] eval ${ver} s${s}"
  done
done
echo "ALL_DONE_NOVELVENDOR"
