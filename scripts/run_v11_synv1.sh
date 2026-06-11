#!/usr/bin/env bash
# =============================================================================
# V11 — V9 recipe on NO-LLM-positive data (syn-v1), three arms. Single GPU,
# sequential, screen/tmux-friendly. Mirrors scripts/run_lineage_v6_v9.sh.
#
#   arm       config                              isolates
#   --------  ----------------------------------  ---------------------------------
#   frozen    v11_synv1_frozen.yaml               V9 recipe, syn-v1, LUAR FROZEN
#                                                 (projection-only) — backbone-
#                                                 adaptation ablation
#   lora      v11_synv1_lora.yaml                 V9 recipe, syn-v1, LUAR+LoRA —
#                                                 the clean "v8->v9 objective
#                                                 minus cross-register data" cell
#   detector  v11_llm_detector.yaml               BCE classification head
#                                                 (human vs LLM), LoRA — the best
#                                                 LLM-detector, not authorship
#
# Evaluations per arm (checkpoint_best.pt — every config monitors
# pauc/genuine_vs_synthetic_5pct — falling back to checkpoint_last.pt):
#   probe          genuine-vs-synthetic classifier probe, own corpus
#   ablate-own     scorer ablation on the arm's OWN corpus (enron_shortmail +
#                  syn-v1) — the no-cross-register evaluation
#   ablate-common  scorer ablation on the COMMON production-like corpus
#                  (enron_shortmail + syn-v2) — apples-to-apples with the
#                  docs/v9_lineage_memo.md v6..v9 table
#
# Launch (survives SSH drops):
#   screen -S v11 bash -c 'bash scripts/run_v11_synv1.sh 2>&1 | tee runs/_v11/console.log'
#
# Re-running after a crash skips arms whose checkpoint already exists
# (SKIP_EXISTING=0 to force retrain). Rough wall-clock on one A40: frozen
# ~50 min, lora ~100 min, detector ~100 min, evals ~10 min/arm -> ~5 h total.
# =============================================================================
set -uo pipefail   # NOT -e: push through failures so all arms complete.

cd "$(dirname "${BASH_SOURCE[0]}")/.."   # project root

export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-v11-synv1}"
export WANDB_PROJECT="${WANDB_PROJECT:-email-fraud-detection}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

LOGDIR="runs/_v11"
RESDIR="results/v11"
EVALCFGDIR="$LOGDIR/eval_cfgs"
mkdir -p "$LOGDIR" "$RESDIR" "$EVALCFGDIR"

ARMS=(frozen lora detector frozen_supcon lora_supcon)
declare -A CFG=(
  [frozen]=configs/experiments/v11_synv1_frozen.yaml
  [lora]=configs/experiments/v11_synv1_lora.yaml
  [detector]=configs/experiments/v11_llm_detector.yaml
  [frozen_supcon]=configs/experiments/v11_synv1_frozen_supcon.yaml
  [lora_supcon]=configs/experiments/v11_synv1_lora_supcon.yaml
)

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
# 0. Sanity: datasets present
# =============================================================================
test -d data/processed/enron_shortmail \
  || echo "WARN: data/processed/enron_shortmail missing — prepare it first (see v9 config header)."
test -d data/synthetic/enron_synthetic_v1 \
  || echo "WARN: data/synthetic/enron_synthetic_v1 missing — V11 training will fail."
test -d data/synthetic/enron_synthetic_v2 \
  || echo "WARN: data/synthetic/enron_synthetic_v2 missing — common-corpus eval will fall back to own corpus."

python -c "import wandb,os; assert os.environ.get('WANDB_API_KEY') or os.path.exists(os.path.expanduser('~/.netrc')) or os.environ.get('WANDB_MODE')=='disabled', 'no auth'; print('wandb auth OK')" \
  || echo "WARN: wandb not authenticated — export WANDB_API_KEY or use WANDB_MODE=disabled."

# =============================================================================
# 1. Train all three arms sequentially
# =============================================================================
for arm in "${ARMS[@]}"; do
  out="runs/v11/$arm"
  if [ "$SKIP_EXISTING" = "1" ] && [ -f "$out/checkpoint_last.pt" ]; then
    echo "skip train $arm — $out/checkpoint_last.pt exists (SKIP_EXISTING=0 to retrain)"
    continue
  fi
  run_step "10_train_${arm}" env WANDB_JOB_TYPE=train \
    python scripts/train.py --config "${CFG[$arm]}" --output-dir "$out"
done

# =============================================================================
# 2. Per-arm evaluations
# =============================================================================
# Derive a "common eval" config from each arm's config: same encoder arch (so
# the checkpoint loads), but data pointed at the shared production-like corpus
# (enron_shortmail + syn-v2), exactly as run_lineage_v6_v9.sh does.
make_common_eval_cfg() {   # make_common_eval_cfg <arm-config> <out-path>
  python - "$1" "$2" <<'PYEOF'
import sys, yaml
src, out = sys.argv[1], sys.argv[2]
with open(src) as fh:
    cfg = yaml.safe_load(fh)
data = cfg.setdefault("data", {})
data["processed_dir"] = "data/processed/enron_shortmail"
data.setdefault("preprocessing", {}).update(
    {"strip_signatures": False, "min_body_chars": 20, "min_body_words": 5}
)
data.setdefault("augmentation", {})["synthetic_path"] = "data/synthetic/enron_synthetic_v2"
with open(out, "w") as fh:
    yaml.safe_dump(cfg, fh, sort_keys=False)
print(f"wrote {out}")
PYEOF
}

post_evals() {   # post_evals <arm>
  local arm="$1"
  local cfg="${CFG[$arm]}"
  local ckpt="runs/v11/$arm/checkpoint_best.pt"
  [ -f "$ckpt" ] || ckpt="runs/v11/$arm/checkpoint_last.pt"
  if [ ! -f "$ckpt" ]; then
    echo "!! no checkpoint for $arm (looked in runs/v11/$arm) — skipping its evals."
    return
  fi
  echo ">> evaluating $arm with $ckpt"

  # 2a. genuine-vs-synthetic classifier probe (own corpus, syn-v1)
  run_step "20_probe_${arm}" env WANDB_JOB_TYPE=probe WANDB_NAME="v11-${arm}-probe" \
    python scripts/probe_authenticity.py \
    --checkpoint "$ckpt" --config "$cfg" \
    --split train --mode both \
    --out "$RESDIR/probe_${arm}.json" --wandb

  # 2b. scorer ablation, own corpus (enron_shortmail + syn-v1 — no cross-register)
  run_step "21_ablate_own_${arm}" env WANDB_JOB_TYPE=ablate WANDB_NAME="v11-${arm}-ablate-own" \
    python scripts/ablate_adaptive_scorers.py \
    --config "$cfg" --checkpoint "$ckpt" \
    --split synthetic --rank-by tpr1 --bootstrap 1000 --k-sweep 4,8,16,25 \
    --out-dir "$RESDIR" --tag "ablate_own_${arm}" --wandb

  # 2c. scorer ablation, COMMON corpus (enron_shortmail + syn-v2 — lineage parity)
  local evalcfg="$EVALCFGDIR/${arm}_common.yaml"
  make_common_eval_cfg "$cfg" "$evalcfg"
  run_step "22_ablate_common_${arm}" env WANDB_JOB_TYPE=ablate WANDB_NAME="v11-${arm}-ablate-common" \
    python scripts/ablate_adaptive_scorers.py \
    --config "$evalcfg" --checkpoint "$ckpt" \
    --split synthetic --rank-by tpr1 --bootstrap 1000 --k-sweep 4,8,16,25 \
    --out-dir "$RESDIR" --tag "ablate_common_${arm}" --wandb

  # 2d. NEW (post-merge): sliced OOD evaluation — pairwise verification metrics
  #     per failure axis (length: short/medium/long/lenmix; register: cross/same)
  #     on UNSEEN test senders. Directly measures whether crop aug + register-
  #     stratified sampling bought length/register invariance. Generator slices
  #     are skipped (single-Mistral set has no `generator` column).
  if [ -f "$OODPAIRS" ]; then
    run_step "23_ood_${arm}" env WANDB_JOB_TYPE=ood WANDB_NAME="v11-${arm}-ood" \
      python scripts/eval_ood.py \
      --checkpoint "$ckpt" --config "$cfg" \
      --pairs "$OODPAIRS" \
      --json "$RESDIR/ood_${arm}.json" --wandb
  else
    echo "!! OOD pairs $OODPAIRS missing — skipping OOD eval for $arm."
  fi
}

# Build the sliced OOD eval set ONCE (shared across arms): unseen TEST senders
# from enron_shortmail, length + register slices; impostor/generator slices from
# syn-v2 (skipped automatically — no `generator` column on the single-Mistral set).
OODPAIRS="$RESDIR/ood_pairs.jsonl"
if [ ! -f "$OODPAIRS" ]; then
  run_step "05_build_ood" python scripts/build_ood_eval.py \
    --config configs/experiments/v11_synv1_lora.yaml \
    --split test --synthetic-path data/synthetic/enron_synthetic_v2 \
    --n-per-slice 300 --output "$OODPAIRS"
fi

for arm in "${ARMS[@]}"; do
  post_evals "$arm"
done

# =============================================================================
# 3. Digest — the common-corpus table for docs/v11_synv1_memo.md
# =============================================================================
echo ""
echo "=================================================================="
echo "V11 DIGEST (common eval corpus: enron_shortmail + syn-v2)"
echo "=================================================================="
python - <<'PYEOF'
import json, pathlib

resdir = pathlib.Path("results/v11")
print(f"{'arm':9} {'scorer':22} {'auc':>6} {'tpr5':>6} {'tpr1':>6} [tpr1 95% CI]")
for arm in ["frozen", "lora", "detector", "frozen_supcon", "lora_supcon"]:
    p = resdir / f"ablate_common_{arm}.json"
    if not p.exists():
        print(f"{arm:9} — missing ({p})")
        continue
    d = json.loads(p.read_text())
    rows = {r["scorer"]: r for r in d["rows"]}
    for scorer in ["baseline_linear_z3", "baseline_cosine", "mahalanobis", "z_persender_sigmoid"]:
        r = rows.get(scorer)
        if r is None:
            continue
        print(
            f"{arm:9} {scorer:22} {r['auc']:6.3f} {r['tpr5']:6.3f} {r['tpr1']:6.3f}"
            f" [{r['tpr1_lo']:.3f}, {r['tpr1_hi']:.3f}]"
        )
    rec = d.get("recommendation", {})
    print(f"     → {rec.get('text', '')[:110]}")
PYEOF

echo ""
echo "=================================================================="
echo "ALL STAGES ATTEMPTED at $(date '+%F %T')."
echo "Per-stage logs:   $LOGDIR/"
echo "Result JSON/CSV:  $RESDIR/  (ablate_own_* = syn-v1 corpus, ablate_common_* = syn-v2 corpus)"
echo "W&B group:        $WANDB_RUN_GROUP  (project: $WANDB_PROJECT)"
echo "Now fill the TBD tables in docs/v11_synv1_memo.md from the digest above and"
echo "compare against the v7/v8/v9 rows in docs/v9_lineage_memo.md §5."
echo "=================================================================="
