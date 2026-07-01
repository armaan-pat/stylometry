#!/usr/bin/env bash
# Whitepaper training runs, chained so one A40 runs them back-to-back overnight.
#   1) v14b reproduction seed (seed=1)  -> headline-robustness
#   2) 2x2 ablation (44 authors, no-syn) -> completes identity x synthetic grid
# Each run is followed by the held-out Claude+Gemini eval across probe seeds 0-4,
# so the new checkpoints sit on the same multiseed footing as the lineage table.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export WANDB_MODE=offline
HELDOUT_CFG=configs/experiments/_v12_heldout_eval.yaml
MS=results/whitepaper/multiseed
LOG=results/whitepaper/logs
mkdir -p "$MS" "$LOG"

heldout_eval () {  # $1=checkpoint  $2=tag
  local ckpt="$1" tag="$2" s
  [ -f "$ckpt" ] || ckpt="$(dirname "$ckpt")/checkpoint_last.pt"
  for s in 0 1 2 3 4; do
    local out="$MS/heldoutCG_${tag}_s${s}.json"
    [ -f "$out" ] && { echo "[skip] $out"; continue; }
    WANDB_MODE=disabled python scripts/ablate_adaptive_scorers.py \
      --config "$HELDOUT_CFG" --checkpoint "$ckpt" \
      --split synthetic --rank-by tpr1 --bootstrap 1000 --seed "$s" \
      --out-dir "$MS" --tag "heldoutCG_${tag}_s${s}" \
      > "$LOG/heldoutCG_${tag}_s${s}.log" 2>&1 \
      && echo "[ok] eval ${tag} s${s}" || echo "[FAIL] eval ${tag} s${s}"
  done
}

# ---------- 1) v14b reproduction seed (seed=1) ----------
echo "=== TRAIN v14b seed=1 ==="
if [ ! -f runs/v14b_seed1/lora/checkpoint_best.pt ]; then
  python scripts/train.py --config configs/experiments/v14b_manyauthor_syn_lora.yaml \
    --output-dir runs/v14b_seed1/lora --seed 1 \
    > "$LOG/train_v14b_seed1.log" 2>&1 \
    && echo "[ok] train v14b_seed1" || echo "[FAIL] train v14b_seed1"
else echo "[skip] v14b_seed1 exists"; fi
heldout_eval runs/v14b_seed1/lora/checkpoint_best.pt v14b_seed1

# ---------- 2) 2x2 ablation: (44 authors, no synthetics) ----------
echo "=== TRAIN 2x2 ablation (enron44, no-syn) ==="
if [ ! -f runs/wp_ablate_enron44_nosyn/lora/checkpoint_best.pt ]; then
  python scripts/train.py --config configs/experiments/wp_ablate_enron44_nosyn_lora.yaml \
    --output-dir runs/wp_ablate_enron44_nosyn/lora --seed 0 \
    > "$LOG/train_enron44_nosyn.log" 2>&1 \
    && echo "[ok] train enron44_nosyn" || echo "[FAIL] train enron44_nosyn"
else echo "[skip] enron44_nosyn exists"; fi
heldout_eval runs/wp_ablate_enron44_nosyn/lora/checkpoint_best.pt enron44_nosyn

echo "ALL_DONE_TRAINING"
