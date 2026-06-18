#!/usr/bin/env bash
# =============================================================================
# V13 — expanded multi-generator adversary (roadmap P1) + the final-rigor P3
# (length-matched enrollment) measurement. Single GPU, sequential, tmux-friendly.
# Mirrors scripts/run_v11_synv1.sh structure (run_step, push-through-failures,
# skip-existing).
#
# Pipeline (front-loads guaranteed wins; the v13 retrain is the long pole, last):
#   05  P3 (no train): length-matched-enrollment baseline/treatment/regression on
#       the EXISTING v12 checkpoint, full bootstrap=1000. (Validation already
#       showed it HURTS — these are the definitive runs for the writeup.)
#   10  GENERATE v13 train set: GPT-4o-mini + Llama-3.1-70B + DeepSeek-Chat,
#       n_per_sender 24, hard-negatives only. Claude+Gemini stay held out.
#   20  TRAIN v13 (150 epochs, ~75-95 min on A40).
#   30  BUILD held-out OOD pairs from the Claude+Gemini set (same as v12).
#   40  EVAL_OOD v13 on held-out gen/len/register slices + PAN/blog domains.
#   50  ABLATE v13 headline: TPR@1%/FPR_other on held-out pool, bootstrap=1000.
#   90  DIGEST: v12-vs-v13 comparison + P3 before/after.
#
# NOTE: P2 (v12 domain re-test) already ran successfully ->
#       results/v12/ood_domains_v12.json (pan 0.779 / blog 0.839, ~unchanged).
#
# Launch (survives SSH drops):
#   tmux new -s v13 'bash scripts/run_v13_overnight.sh 2>&1 | tee results/v13/logs/console.log'
#
# Re-running skips steps whose output exists (SKIP_EXISTING=0 to force).
# =============================================================================
set -uo pipefail   # NOT -e: push through failures so all steps are attempted.

cd "$(dirname "${BASH_SOURCE[0]}")/.."   # project root

export WANDB_MODE="${WANDB_MODE:-offline}"   # hermetic: no network dependency overnight
SKIP_EXISTING="${SKIP_EXISTING:-1}"

RESDIR="results/v13"
LOGDIR="$RESDIR/logs"
V12RES="results/v12"
CKPT_V12="runs/v12/lora/checkpoint_best.pt"
CKPT_V13="runs/v13/lora/checkpoint_best.pt"
V13_TRAIN="data/synthetic/enron_synthetic_v13_train"
HELDOUT="data/synthetic/enron_synthetic_v12_heldout"   # Claude+Gemini, reuse
V13_CFG="configs/experiments/v13_multigen_lora.yaml"
HELDOUT_CFG="configs/experiments/_v12_heldout_eval.yaml"
# Reuse v12's exact held-out pairs for a bit-identical apples-to-apples eval;
# rebuild only if that file is absent.
OODPAIRS="$V12RES/ood_v12heldout_pairs.jsonl"
mkdir -p "$LOGDIR" "$RESDIR"

run_step() {   # run_step <name> <cmd...>
  local name="$1"; shift
  local log="$LOGDIR/${name}.log"
  echo ""
  echo "=================================================================="
  echo ">> [$(date '+%F %T')] START  $name"
  echo "   cmd: $*"
  echo "   log: $log"
  echo "=================================================================="
  if "$@" 2>&1 | tee "$log"; then
    echo "<< [$(date '+%F %T')] OK     $name"
  else
    local rc=${PIPESTATUS[0]}
    echo "!! [$(date '+%F %T')] FAILED $name (exit $rc) — continuing. See $log"
  fi
}

# =============================================================================
# 0. Sanity
# =============================================================================
test -d data/processed/enron_shortmail || echo "WARN: processed corpus missing."
test -d "$HELDOUT" || echo "WARN: held-out (Claude+Gemini) set missing — v13 eval will fail."
test -f "$CKPT_V12" || echo "WARN: v12 checkpoint missing — P3 + v12 baseline skipped."
python -c "import os; assert os.environ.get('OPENROUTER_API_KEY'), 'no OPENROUTER_API_KEY'; print('OpenRouter key OK')" \
  || echo "WARN: OPENROUTER_API_KEY not set — v13 generation will fail."

# =============================================================================
# 05. P3 — length-matched enrollment, definitive bootstrap=1000 runs (no train).
#     split=other isolates the cross-length authorship signal; --query-bucket
#     short isolates short queries (the centroid analogue of lenmix).
# =============================================================================
if [ -f "$CKPT_V12" ]; then
  P3_COMMON=(--config "$HELDOUT_CFG" --checkpoint "$CKPT_V12"
             --split other --rank-by auc --bootstrap 1000
             --n-profile-senders 60 --n-enroll 8 --n-query 20
             --out-dir "$RESDIR")
  run_step "05a_p3_baseline_shortq" \
    python scripts/ablate_adaptive_scorers.py "${P3_COMMON[@]}" \
    --query-bucket short --tag p3_lenmatch_baseline_shortq
  run_step "05b_p3_lenmatched_shortq" \
    python scripts/ablate_adaptive_scorers.py "${P3_COMMON[@]}" \
    --query-bucket short --length-matched-enrollment --tag p3_lenmatch_on_shortq
  run_step "05c_p3_lenmatched_longq" \
    python scripts/ablate_adaptive_scorers.py "${P3_COMMON[@]}" \
    --query-bucket long --length-matched-enrollment --tag p3_lenmatch_on_longq
  # Regression guard: no flag, no bucket -> must reproduce existing v12 numbers.
  run_step "05d_p3_regression_check" \
    python scripts/ablate_adaptive_scorers.py \
    --config "$HELDOUT_CFG" --checkpoint "$CKPT_V12" \
    --split synthetic --rank-by tpr1 --bootstrap 1000 \
    --out-dir "$RESDIR" --tag p3_regression_check
fi

# =============================================================================
# 10. Generate v13 expanded adversary (GPT + Llama + DeepSeek), hard-neg only.
# =============================================================================
if [ "$SKIP_EXISTING" = "1" ] && [ -d "$V13_TRAIN" ]; then
  echo "skip generate — $V13_TRAIN exists (SKIP_EXISTING=0 to regenerate)."
else
  run_step "10_generate_v13_train" \
    python scripts/generate_synthetic_emails.py \
    --config "$V13_CFG" \
    --generators openrouter:openai/gpt-4o-mini \
                 openrouter:meta-llama/llama-3.1-70b-instruct \
                 openrouter:deepseek/deepseek-chat \
    --from-split train --n-per-sender 24 --llm-positives exclude \
    --request-workers 4 --output "$V13_TRAIN"
fi
# Verify composition (row count + per-vendor balance).
if [ -d "$V13_TRAIN" ]; then
  python - "$V13_TRAIN" <<'PY'
import sys
from collections import Counter
from datasets import load_from_disk
d = load_from_disk(sys.argv[1])
print(f"v13_train rows: {len(d)}")
print("by generator:", dict(Counter(d['generator'])))
PY
fi

# =============================================================================
# 20. Train v13 (150 epochs). Monitor: pauc/min_other_synthetic_5pct (anti-Goodhart).
# =============================================================================
if [ "$SKIP_EXISTING" = "1" ] && [ -f "runs/v13/lora/checkpoint_last.pt" ]; then
  echo "skip train — runs/v13/lora/checkpoint_last.pt exists."
elif [ ! -d "$V13_TRAIN" ]; then
  echo "!! v13 train set missing — skipping training + v13 eval."
else
  run_step "20_train_v13" env WANDB_JOB_TYPE=train WANDB_NAME=v13-multigen-lora \
    python scripts/train.py --config "$V13_CFG" --output-dir runs/v13/lora
fi

# =============================================================================
# 30-50. Evaluate v13 on the SAME held-out set v12 used (apples-to-apples).
# =============================================================================
CKPT="$CKPT_V13"; [ -f "$CKPT" ] || CKPT="runs/v13/lora/checkpoint_last.pt"
if [ -f "$CKPT" ]; then
  echo ">> evaluating v13 with $CKPT"

  if [ -f "$OODPAIRS" ]; then
    echo "reuse existing held-out pairs (identical to v12 eval): $OODPAIRS"
  else
    OODPAIRS="$RESDIR/ood_heldoutCG_pairs.jsonl"
    run_step "30_build_ood_heldout" \
      python scripts/build_ood_eval.py \
      --config "$HELDOUT_CFG" --synthetic-path "$HELDOUT" \
      --output "$OODPAIRS"
  fi

  run_step "40_eval_ood_v13" \
    python scripts/eval_ood.py \
    --checkpoint "$CKPT" --config "$HELDOUT_CFG" \
    --pairs "$OODPAIRS" \
    --extra-pairs domain:pan20_xtopic=data/ood/pan20_xtopic_pairs.jsonl \
    --extra-pairs domain:blog=data/ood/blog_pairs.jsonl \
    --json "$RESDIR/ood_v13lora_heldoutCG.json"

  run_step "50_ablate_v13_heldout" \
    python scripts/ablate_adaptive_scorers.py \
    --config "$HELDOUT_CFG" --checkpoint "$CKPT" \
    --split synthetic --rank-by tpr1 --bootstrap 1000 --crop-syn \
    --out-dir "$RESDIR" --tag heldoutCG_v13lora
else
  echo "!! no v13 checkpoint — skipping v13 eval."
fi

# =============================================================================
# 90. Digest — v12 vs v13 held-out + P3 before/after.
# =============================================================================
echo ""
echo "=================================================================="
echo "V13 DIGEST  ($(date '+%F %T'))"
echo "=================================================================="
python - <<'PY'
import json, pathlib
R = pathlib.Path("results")

def load(p):
    p = R / p
    return json.loads(p.read_text()) if p.exists() else None

print("\n--- P1: held-out (Claude+Gemini) per-generator AUC, v12 vs v13 ---")
v12 = load("v12/ood_v12lora_final_heldoutCG.json") or load("v12/ood_domains_v12.json")
v13 = load("v13/ood_v13lora_heldoutCG.json")
def gens(d):
    if not d: return {}
    return {k.split("/")[-1]: round(v.get("AUC", float("nan")), 3)
            for k, v in d.items() if k.startswith("gen:")}
print("  v12 gen AUC:", gens(v12))
print("  v13 gen AUC:", gens(v13))
if v13:
    print("  v13 domain:", {k: round(v.get('AUC',float('nan')),3) for k,v in v13.items() if k.startswith('domain')})
    print("  v13 len/register:", {k: round(v.get('AUC',float('nan')),3) for k,v in v13.items() if k.startswith(('len','register'))})

print("\n--- P1: held-out pool headline (baseline_linear_z3), v13 ablation ---")
ab = load("v13/heldoutCG_v13lora.json")
if ab:
    r = next((x for x in ab["rows"] if x["scorer"] == "baseline_linear_z3"), None)
    if r:
        print(f"  AUC={r['auc']:.3f}  TPR@1%={r['tpr1']:.3f}  TPR@5%={r['tpr5']:.3f}  "
              f"FPR_other@5={r.get('fpr_other_at_5')}  TPR1crop={r.get('tpr1_crop')}")

print("\n--- P3: length-matched enrollment, short queries (baseline_linear_z3) ---")
b = load("v13/p3_lenmatch_baseline_shortq.json")
t = load("v13/p3_lenmatch_on_shortq.json")
def row(d): return next((x for x in d["rows"] if x["scorer"]=="baseline_linear_z3"), None) if d else None
rb, rt = row(b), row(t)
if rb and rt:
    print(f"  {'metric':6} {'full-enroll':>12} {'len-matched':>12} {'delta':>8}")
    for m in ["auc","pauc5","tpr1","tpr5"]:
        print(f"  {m:6} {rb[m]:>12.3f} {rt[m]:>12.3f} {rt[m]-rb[m]:>+8.3f}")
    print("  (validation showed length-matching HURTS — confirm here with bootstrap=1000)")
PY

echo ""
echo "ALL STAGES ATTEMPTED at $(date '+%F %T'). Logs: $LOGDIR/  Results: $RESDIR/"
echo "Next: fill docs/v13_results.md from this digest."
echo "=================================================================="
