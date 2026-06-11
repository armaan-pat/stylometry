#!/usr/bin/env bash
# =============================================================================
# v6-repro vs v11-lora-k4 — clean A/B in ONE repo, ONE eval harness.
#
# Trains two arms sequentially on a single GPU and prints a side-by-side digest
# of the now-comparable CentroidProbe metrics (incl. the new genuine_vs_other
# operating points added to centroid_probe.py).
#
#   arm        config                                 what it is
#   ---------  -------------------------------------  --------------------------
#   v6_repro   v6_luar_lora_syn_repro.yaml            faithful rerun of the
#                                                     performant v6 run snapshot
#                                                     (supcon, episode_k=4, full
#                                                     enron, 125 senders)
#   lora_k4    v11_synv1_lora_k4.yaml                 v11 lora + episode_k=4
#                                                     (single-knob authorship
#                                                     recovery vs v11_synv1_lora)
#
# Both are scored by the CURRENT harness, so v6's wrong-sender tail is reported
# on the same axes as v11 (tpr_at_fpr/other_1pct, op/other/*). For the EXISTING
# v11 arms (runs/v11/{lora,frozen,...}) re-score or re-train to populate the new
# other-tail keys; their old summaries predate the metric and won't have them.
#
# Launch (survives SSH drops):
#   screen -S v6k4 bash -c 'bash scripts/run_v6_vs_k4.sh 2>&1 | tee runs/_v6_vs_k4/console.log'
#
# Re-running skips arms whose checkpoint already exists (SKIP_EXISTING=0 forces
# retrain). Rough wall-clock on one A40: v6_repro ~20 min (100 ep), lora_k4
# ~110 min (150 ep, episode_k=4 is ~slightly slower than k=1).
# =============================================================================
set -uo pipefail   # NOT -e: push through a failed arm so the other still runs.

cd "$(dirname "${BASH_SOURCE[0]}")/.."   # project root

export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-v6-vs-k4}"
export WANDB_PROJECT="${WANDB_PROJECT:-email-fraud-detection}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

LOGDIR="runs/_v6_vs_k4"
mkdir -p "$LOGDIR"

ARMS=(v6_repro lora_k4)
declare -A CFG=(
  [v6_repro]=configs/experiments/v6_luar_lora_syn_repro.yaml
  [lora_k4]=configs/experiments/v11_synv1_lora_k4.yaml
)
declare -A OUT=(
  [v6_repro]=runs/v6_repro
  [lora_k4]=runs/v11/lora_k4
)

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

# -----------------------------------------------------------------------------
# 0. Sanity: datasets present (v6 uses full enron + enron_synthetic; k4 uses
#    enron_shortmail + enron_synthetic_v1).
# -----------------------------------------------------------------------------
for d in data/processed/enron data/processed/enron_shortmail \
         data/synthetic/enron_synthetic data/synthetic/enron_synthetic_v1; do
  test -d "$d" || echo "WARN: $d missing — the arm that needs it will fail."
done
python -c "import wandb,os; assert os.environ.get('WANDB_API_KEY') or os.path.exists(os.path.expanduser('~/.netrc')) or os.environ.get('WANDB_MODE')=='disabled', 'no auth'; print('wandb auth OK')" \
  || echo "WARN: wandb not authenticated — export WANDB_API_KEY or run with WANDB_MODE=disabled."

# -----------------------------------------------------------------------------
# 1. Train both arms sequentially.
# -----------------------------------------------------------------------------
for arm in "${ARMS[@]}"; do
  out="${OUT[$arm]}"
  if [ "$SKIP_EXISTING" = "1" ] && [ -f "$out/checkpoint_last.pt" ]; then
    echo "skip train $arm — $out/checkpoint_last.pt exists (SKIP_EXISTING=0 to retrain)"
    continue
  fi
  run_step "train_${arm}" env WANDB_JOB_TYPE=train WANDB_NAME="${arm}" \
    python scripts/train.py --config "${CFG[$arm]}" --output-dir "$out"
done

# -----------------------------------------------------------------------------
# 2. Digest — final-epoch CentroidProbe metrics, read straight from each run's
#    wandb-summary.json (works even with WANDB_MODE=offline/disabled).
# -----------------------------------------------------------------------------
echo ""
echo "=================================================================="
echo "DIGEST — final-epoch CentroidProbe (single-email queries, same harness)"
echo "=================================================================="
RUN_DIRS="${OUT[v6_repro]} ${OUT[lora_k4]} runs/v11/lora" \
python - <<'PYEOF'
import json, os, glob

# Include the existing v11 lora as a reference row when present (its old summary
# may lack the new other-tail keys — shown as "—").
labels = {"runs/v6_repro": "v6_repro", "runs/v11/lora_k4": "lora_k4", "runs/v11/lora": "v11_lora(old)"}
run_dirs = os.environ["RUN_DIRS"].split()

def load_summary(run_dir):
    hits = glob.glob(f"{run_dir}/wandb/*/files/wandb-summary.json") \
        + glob.glob(f"{run_dir}/wandb/latest-run/files/wandb-summary.json")
    if not hits:
        return None
    hits.sort(key=os.path.getmtime)
    with open(hits[-1]) as fh:
        return json.load(fh)

keys = [
    ("auc/genuine_vs_other",            "auc_other"),
    ("auc/genuine_vs_synthetic",        "auc_syn"),
    ("auc/genuine_vs_all",              "auc_all"),
    ("tpr_at_fpr/other_1pct",           "tpr1_other"),
    ("tpr_at_fpr/synthetic_1pct",       "tpr1_syn"),
    ("tpr_at_fpr/all_1pct",             "tpr1_all"),
    ("score/synthetic_harder_than_other","syn>other"),
]
hdr = f"{'arm':16}" + "".join(f"{short:>12}" for _, short in keys) + f"{'epoch':>7}"
print(hdr)
print("-" * len(hdr))
for run_dir in run_dirs:
    s = load_summary(run_dir)
    name = labels.get(run_dir, os.path.basename(run_dir))
    if s is None:
        print(f"{name:16}  (no wandb-summary.json yet — not trained?)")
        continue
    row = f"{name:16}"
    for k, _ in keys:
        v = s.get(k)
        row += f"{v:12.3f}" if isinstance(v, (int, float)) else f"{'—':>12}"
    row += f"{int(s.get('epoch', -1)):7d}"
    print(row)

print()
print("Read: tpr1_all is set by the WORSE tail. For the v11 family that is")
print("tpr1_other (syn>other < 0). If lora_k4's tpr1_other rises toward v6_repro's,")
print("episode_k=4 explains v6's authorship edge; if a gap remains, crop aug is next")
print("(set crop_prob: 0.0 in v11_synv1_lora_k4.yaml).")
PYEOF

echo ""
echo "Per-stage logs: $LOGDIR/   |   W&B group: $WANDB_RUN_GROUP"
echo "Checkpoints:    ${OUT[v6_repro]}/  and  ${OUT[lora_k4]}/"
echo "=================================================================="
