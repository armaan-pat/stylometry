#!/usr/bin/env bash
# =============================================================================
# V6 → V9 lineage benchmark — single GPU, sequential, screen/tmux-friendly.
#
# Trains the four generations of the recipe back-to-back, then evaluates every
# checkpoint two ways so the memo table (docs/v9_lineage_memo.md) can be filled
# in directly:
#
#   arm  config                                    isolates
#   ---  ----------------------------------------  --------------------------------
#   v6   v6_bench.yaml                             baseline recipe (τ=0.07, n_syn=2,
#                                                  q/v LoRA, 100 ep, original synth)
#   v7   v7_luar_lora_syn_mahal.yaml               V7.3 training recipe (τ=0.05,
#                                                  n_syn=4, +key LoRA, 150 ep)
#   v8   v7_synv2.yaml                             syn-v2 data (cross-register
#                                                  positives), same V7.3 recipe
#   v9   v9_episodic_shortmail.yaml                episodic variable-K loss +
#                                                  short-email data/crop aug
#
# Evaluations per arm (checkpoint_best.pt — trustworthy now that every config
# monitors pauc/genuine_vs_synthetic_5pct — falling back to checkpoint_last.pt):
#   probe   genuine-vs-synthetic classifier probe, own corpus
#   ablate-own     scorer ablation on the arm's OWN training corpus/synthetics
#                  (comparable to the historical V7/V8 numbers)
#   ablate-common  scorer ablation on the COMMON production-like corpus:
#                  enron_shortmail (short emails present, signatures kept) +
#                  syn-v2 impostors — same eval set for all arms; THESE are the
#                  apples-to-apples numbers for the lineage memo.
#
# Launch (survives SSH drops):
#   screen -S lineage bash -c 'bash scripts/run_lineage_v6_v9.sh 2>&1 | tee runs/_lineage/console.log'
#   # or: tmux new -s lineage 'bash scripts/run_lineage_v6_v9.sh 2>&1 | tee runs/_lineage/console.log'
#
# Re-running after a crash skips arms whose checkpoint already exists
# (SKIP_EXISTING=0 to force retrain). Rough wall-clock on one A40: v6 ~50 min,
# v7/v8 ~75 min each, v9 ~100 min (batch 128), evals ~10 min/arm → ~6 h total.
# =============================================================================
set -uo pipefail   # NOT -e: push through failures so all arms complete.

cd "$(dirname "${BASH_SOURCE[0]}")/.."   # project root

export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-v9-lineage}"
export WANDB_PROJECT="${WANDB_PROJECT:-email-fraud-detection}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

LOGDIR="runs/_lineage"
RESDIR="results/lineage"
EVALCFGDIR="$LOGDIR/eval_cfgs"
mkdir -p "$LOGDIR" "$RESDIR" "$EVALCFGDIR"

ARMS=(v6 v7 v8 v9)
declare -A CFG=(
  [v6]=configs/experiments/v6_bench.yaml
  [v7]=configs/experiments/v7_luar_lora_syn_mahal.yaml
  [v8]=configs/experiments/v7_synv2.yaml
  [v9]=configs/experiments/v9_episodic_shortmail.yaml
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
# 0. Sanity: datasets present (build the common eval corpus if missing)
# =============================================================================
test -d data/processed/enron \
  || echo "WARN: data/processed/enron missing — v6/v7/v8 training will fail."
test -d data/synthetic/enron_synthetic \
  || echo "WARN: data/synthetic/enron_synthetic missing — v6/v7 training will fail."
test -d data/synthetic/enron_synthetic_v2 \
  || echo "WARN: data/synthetic/enron_synthetic_v2 missing — v8/v9 training will fail."

if [ ! -d data/processed/enron_shortmail ]; then
  run_step 00_prepare_shortmail python scripts/prepare_data.py \
    --output-dir data/processed/enron_shortmail \
    --min-body-chars 20 --min-body-words 5 --no-strip-signatures
fi

python -c "import wandb,os; assert os.environ.get('WANDB_API_KEY') or os.path.exists(os.path.expanduser('~/.netrc')), 'no auth'; print('wandb auth OK')" \
  || echo "WARN: wandb not authenticated — export WANDB_API_KEY or use WANDB_MODE=disabled."

# =============================================================================
# 1. Train all four arms sequentially
# =============================================================================
for arm in "${ARMS[@]}"; do
  out="runs/lineage/$arm"
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
# the checkpoint loads), but data pointed at the shared production-like corpus.
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
  local ckpt="runs/lineage/$arm/checkpoint_best.pt"
  [ -f "$ckpt" ] || ckpt="runs/lineage/$arm/checkpoint_last.pt"
  if [ ! -f "$ckpt" ]; then
    echo "!! no checkpoint for $arm (looked in runs/lineage/$arm) — skipping its evals."
    return
  fi
  echo ">> evaluating $arm with $ckpt"

  # 2a. genuine-vs-synthetic classifier probe (own corpus)
  run_step "20_probe_${arm}" env WANDB_JOB_TYPE=probe WANDB_NAME="lineage-${arm}-probe" \
    python scripts/probe_authenticity.py \
    --checkpoint "$ckpt" --config "$cfg" \
    --split train --mode both \
    --out "$RESDIR/probe_${arm}.json" --wandb

  # 2b. scorer ablation, own corpus (parity with historical V7/V8 numbers)
  run_step "21_ablate_own_${arm}" env WANDB_JOB_TYPE=ablate WANDB_NAME="lineage-${arm}-ablate-own" \
    python scripts/ablate_adaptive_scorers.py \
    --config "$cfg" --checkpoint "$ckpt" \
    --split synthetic --rank-by tpr1 --bootstrap 1000 --k-sweep 4,8,16,25 \
    --out-dir "$RESDIR" --tag "ablate_own_${arm}" --wandb

  # 2c. scorer ablation, COMMON corpus (the lineage-memo numbers)
  local evalcfg="$EVALCFGDIR/${arm}_common.yaml"
  make_common_eval_cfg "$cfg" "$evalcfg"
  run_step "22_ablate_common_${arm}" env WANDB_JOB_TYPE=ablate WANDB_NAME="lineage-${arm}-ablate-common" \
    python scripts/ablate_adaptive_scorers.py \
    --config "$evalcfg" --checkpoint "$ckpt" \
    --split synthetic --rank-by tpr1 --bootstrap 1000 --k-sweep 4,8,16,25 \
    --out-dir "$RESDIR" --tag "ablate_common_${arm}" --wandb
}

for arm in "${ARMS[@]}"; do
  post_evals "$arm"
done

# =============================================================================
# 3. Digest — the common-corpus table for docs/v9_lineage_memo.md
# =============================================================================
echo ""
echo "=================================================================="
echo "LINEAGE DIGEST (common eval corpus: enron_shortmail + syn-v2)"
echo "=================================================================="
python - <<'PYEOF'
import json, pathlib

resdir = pathlib.Path("results/lineage")
print(f"{'arm':4} {'scorer':22} {'auc':>6} {'tpr5':>6} {'tpr1':>6} [tpr1 95% CI]")
for arm in ["v6", "v7", "v8", "v9"]:
    p = resdir / f"ablate_common_{arm}.json"
    if not p.exists():
        print(f"{arm:4} — missing ({p})")
        continue
    d = json.loads(p.read_text())
    rows = {r["scorer"]: r for r in d["rows"]}
    for scorer in ["baseline_linear_z3", "mahalanobis", "z_persender_sigmoid"]:
        r = rows.get(scorer)
        if r is None:
            continue
        print(
            f"{arm:4} {scorer:22} {r['auc']:6.3f} {r['tpr5']:6.3f} {r['tpr1']:6.3f}"
            f" [{r['tpr1_lo']:.3f}, {r['tpr1_hi']:.3f}]"
        )
    rec = d.get("recommendation", {})
    print(f"     → {rec.get('text', '')[:110]}")
PYEOF

echo ""
echo "=================================================================="
echo "ALL STAGES ATTEMPTED at $(date '+%F %T')."
echo "Per-stage logs:   $LOGDIR/"
echo "Result JSON/CSV:  $RESDIR/  (ablate_own_* = per-arm corpus, ablate_common_* = shared corpus)"
echo "W&B group:        $WANDB_RUN_GROUP  (project: $WANDB_PROJECT)"
echo "Now fill in the TBD tables in docs/v9_lineage_memo.md from the digest above."
echo "=================================================================="
