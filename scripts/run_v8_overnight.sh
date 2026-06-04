#!/usr/bin/env bash
# =============================================================================
# Overnight V7 syn-v1-vs-syn-v2 comparison — RunPod, single GPU, sequential.
#
# Produces, under ONE W&B group (WANDB_RUN_GROUP), these runs:
#
#   job_type=train   v7-synv1         V7 trained on ORIGINAL synthetic (control)
#   job_type=train   v7-synv2         V7 trained on NEW cross-register synthetic
#   job_type=probe   v7-synv1-probe   genuine-vs-synthetic classification head
#   job_type=probe   v7-synv2-probe       (AuthenticityProbe, frozen + finetune)
#   job_type=ablate  v7-synv1-ablate  data-dependent vs fixed scorer table
#   job_type=ablate  v7-synv2-ablate      (adaptive.py SCORERS vs linear_z3)
#
# The per-sender-fraud metrics (auc/genuine_vs_synthetic, pAUC, TPR@FPR) are
# logged INLINE by the CentroidProbe every epoch of each train run — so the
# train runs themselves are the "per-sender fraud" comparison.
#
# Sectioning: group + job_type + display name are set via WANDB_* env vars.
#   - train runs take their name from the config (wandb.name); env name ignored.
#   - probe/ablate runs read WANDB_NAME from the env (they don't pass name=).
#   - tags (syn-v1 / syn-v2) come from each run's --config wandb.tags.
#
# Resilience: every stage runs through run_step(), which tees to a log file and
# CONTINUES on failure so a single bad stage doesn't abort the night. Downstream
# stages are gated on their checkpoint actually existing.
#
# Launch it so an SSH drop can't kill it:
#   tmux new -s v8 'bash scripts/run_v8_overnight.sh 2>&1 | tee runs/_v8_overnight/console.log'
#   # or:  nohup bash scripts/run_v8_overnight.sh > runs/_v8_overnight/console.log 2>&1 &
# =============================================================================
set -uo pipefail   # NOT -e: we want to push through failures so all runs complete.

cd "$(dirname "${BASH_SOURCE[0]}")/.."   # project root
ROOT="$(pwd)"

# -----------------------------------------------------------------------------
# Config (override by exporting before you call the script)
# -----------------------------------------------------------------------------
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-v8-syn-compare}"   # the campaign
export WANDB_PROJECT="${WANDB_PROJECT:-email-fraud-detection}"
GEN_MODEL="${GEN_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
# Mistral-7B is ~14GB in fp16 — fits the A40's 48GB with room to spare, and
# fp16 generation is far faster than bitsandbytes 4-bit (which only helps on
# <16GB cards). Set GEN_4BIT=1 only on a small GPU. GEN_BATCH controls batch size.
GEN_4BIT="${GEN_4BIT:-0}"
GEN_BATCH="${GEN_BATCH:-16}"
GEN_FLAGS="--batch-size $GEN_BATCH"
[ "$GEN_4BIT" = "1" ] && GEN_FLAGS="$GEN_FLAGS --load-in-4bit"
GEN_CONFIG="configs/experiments/v7_luar_lora_syn_mahal_eval.yaml"  # arch-only, loads cleanly
SYN_V1="data/synthetic/enron_synthetic_v1"     # control: cross-register-fraction 0.0
SYN_V2="data/synthetic/enron_synthetic_v2"     # treatment: cross-register-fraction 0.4
DIAGNOSE_CKPT="${DIAGNOSE_CKPT:-}"             # optional prior ckpt for the v2 go/no-go gate

LOGDIR="runs/_v8_overnight"
mkdir -p "$LOGDIR"

# -----------------------------------------------------------------------------
run_step() {   # run_step <name> <cmd...>
  local name="$1"; shift
  local log="$LOGDIR/${name}.log"
  echo ""
  echo "=================================================================="
  echo ">> [$(date '+%F %T')] START  $name"
  echo "   cmd: $*"
  echo "   log: $log"
  echo "=================================================================="
  # tee → live terminal output AND a persistent per-stage log. pipefail (set at
  # the top) makes the pipeline's exit status reflect the command, not tee.
  if "$@" 2>&1 | tee "$log"; then
    echo "<< [$(date '+%F %T')] OK     $name"
    return 0
  else
    local rc=${PIPESTATUS[0]}
    echo "!! [$(date '+%F %T')] FAILED $name (exit $rc) — continuing. See $log"
    return $rc
  fi
}

# =============================================================================
# 0. Environment sanity (does not abort on failure, just reports)
# =============================================================================
run_step 00_setup bash setup_runpod.sh || true
python -c "import wandb,os; assert os.environ.get('WANDB_API_KEY') or os.path.exists(os.path.expanduser('~/.netrc')), 'No WANDB_API_KEY / netrc — wandb runs will fail'; print('wandb auth OK')" \
  || echo "WARN: wandb not authenticated — export WANDB_API_KEY first."
test -d data/processed/enron \
  || echo "WARN: data/processed/enron missing — run scripts/prepare_data.py or sync the volume first."

# =============================================================================
# 1. Generate both synthetic datasets (skip if already present)
# =============================================================================
if [ -d "$SYN_V1" ]; then
  echo "skip gen v1 — $SYN_V1 exists"
else
  run_step 01_gen_synv1 env WANDB_JOB_TYPE=generate WANDB_NAME=gen-synv1 \
    python scripts/generate_synthetic_emails.py \
    --config "$GEN_CONFIG" --model "$GEN_MODEL" \
    --n-per-sender 10 --cross-register-fraction 0.0 \
    --output "$SYN_V1" $GEN_FLAGS --wandb
fi

if [ -d "$SYN_V2" ]; then
  echo "skip gen v2 — $SYN_V2 exists"
else
  run_step 02_gen_synv2 env WANDB_JOB_TYPE=generate WANDB_NAME=gen-synv2 \
    python scripts/generate_synthetic_emails.py \
    --config "$GEN_CONFIG" --model "$GEN_MODEL" \
    --n-per-sender 15 --cross-register-fraction 0.4 \
    --output "$SYN_V2" $GEN_FLAGS --wandb
fi

# =============================================================================
# 1b. OPTIONAL go/no-go on the v2 cross-register positives.
#     Needs a PRIOR checkpoint (we have none before training). Provide one via
#     DIAGNOSE_CKPT=runs/<old_run>/checkpoint_best.pt to enable.
# =============================================================================
if [ -n "$DIAGNOSE_CKPT" ] && [ -f "$DIAGNOSE_CKPT" ]; then
  run_step 03_diagnose_v2 python scripts/diagnose_synthetic_quality.py \
    --config "$GEN_CONFIG" --checkpoint "$DIAGNOSE_CKPT" \
    --synthetic "$SYN_V2" --data-dir data/processed/enron \
    --out-json results/synthetic_quality_v2.json
else
  echo "skip diagnose — no DIAGNOSE_CKPT provided (decision is made empirically by the two train runs)."
fi

# =============================================================================
# 2. Train V7 on each dataset (per-sender fraud metrics logged inline each epoch)
# =============================================================================
run_step 10_train_synv1 env WANDB_JOB_TYPE=train \
  python scripts/train.py --config configs/experiments/v7_synv1.yaml \
  --output-dir runs/v7_synv1

run_step 11_train_synv2 env WANDB_JOB_TYPE=train \
  python scripts/train.py --config configs/experiments/v7_synv2.yaml \
  --output-dir runs/v7_synv2

# =============================================================================
# 3. Post-hoc evals on each checkpoint (gated on the checkpoint existing)
# =============================================================================
post_evals() {   # post_evals <arm: synv1|synv2>
  local arm="$1"
  local cfg="configs/experiments/v7_${arm}.yaml"
  local ckpt="runs/v7_${arm}/checkpoint_best.pt"
  [ -f "$ckpt" ] || ckpt="runs/v7_${arm}/checkpoint_last.pt"
  if [ ! -f "$ckpt" ]; then
    echo "!! no checkpoint for $arm (looked in runs/v7_${arm}) — skipping its probe+ablation."
    return
  fi

  # 3a. genuine-vs-synthetic classification head (frozen LR + finetuned head)
  run_step "20_probe_${arm}" env WANDB_JOB_TYPE=probe WANDB_NAME="v7-${arm}-probe" \
    python scripts/probe_authenticity.py \
    --checkpoint "$ckpt" --config "$cfg" \
    --split train --mode both \
    --out "results/v7/probe_${arm}.json" --wandb

  # 3b. data-dependent (adaptive.py) vs fixed (linear_z3) scorer ablation
  run_step "21_ablate_${arm}" env WANDB_JOB_TYPE=ablate WANDB_NAME="v7-${arm}-ablate" \
    python scripts/ablate_adaptive_scorers.py \
    --config "$cfg" --checkpoint "$ckpt" \
    --split synthetic --rank-by tpr1 --bootstrap 1000 \
    --k-sweep 4,8,16,25 \
    --out-dir "results/v7" --tag "scorer_ablation_${arm}" --wandb
}

post_evals synv1
post_evals synv2

# =============================================================================
# 4. Done
# =============================================================================
echo ""
echo "=================================================================="
echo "ALL STAGES ATTEMPTED at $(date '+%F %T'). Per-stage logs: $LOGDIR/"
echo "W&B group: $WANDB_RUN_GROUP  (project: $WANDB_PROJECT)"
echo "Local artifacts: results/v7/probe_synv{1,2}.json, results/v7/scorer_ablation_synv{1,2}.json"
echo "=================================================================="
