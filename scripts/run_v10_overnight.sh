#!/usr/bin/env bash
# =============================================================================
# V10 overnight — train the AUTHORSHIP specialist, prove the v9/v10 pair.
#
# Context (docs/v9_lineage_results_analysis.md): the v9 episodic recipe makes
# two specialists in one run — synthetic separation peaks ~ep10, human-impostor
# discrimination ~ep150 — and the old synthetic-only monitor freezes the wrong
# one for authorship. Tonight:
#
#   1. train v10  = v9 recipe, monitor pauc/genuine_vs_other_5pct (NEW metric)
#                   → the authorship specialist
#   2. evals      v10 probe + scorer ablation (best AND last ckpt, --crop-syn
#                   short-impostor pool, FPR_other columns — both NEW)
#   3. re-ablate  v9 best+last with the new columns (v9 = LLM-detector role)
#   4. fusion     v10-best × v9-best AND-gate + soft-min (the deliverable)
#   5. soup       v9 best↔last weight blend (single-model baseline vs the pair)
#   6. v6 repair  optional (INCLUDE_V6=1): rerun the crashed v6 arm with the
#                   anti-Goodhart min(pauc_syn, pauc_other) monitor
#
# Failures are LOUD: every failed stage is collected and echoed in a banner at
# the end (and the script exits nonzero). Training stages retry once — the
# 2026-06-09 v6 crash died silently at the first hard-negative mining pass.
#
# Launch:
#   nohup bash scripts/run_v10_overnight.sh > runs/_v10_overnight/console.log 2>&1 &
# =============================================================================
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-v10-two-model}"
export WANDB_PROJECT="${WANDB_PROJECT:-email-fraud-detection}"
INCLUDE_V6="${INCLUDE_V6:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

LOGDIR="runs/_v10_overnight"
RESDIR="results/lineage_v2"
mkdir -p "$LOGDIR" "$RESDIR"

V10_CFG=configs/experiments/v10_episodic_authorship.yaml
V9_DIR=runs/lineage/v9                  # existing 2026-06-10 checkpoints
V10_DIR=runs/lineage_v2/v10
V6_DIR=runs/lineage_v2/v6

FAILURES=()

run_step() {   # run_step <name> <cmd...>
  local name="$1"; shift
  local log="$LOGDIR/${name}.log"
  echo ""
  echo "=================================================================="
  echo ">> [$(date '+%F %T')] START  $name"
  echo "   log: $log"
  echo "=================================================================="
  if "$@" 2>&1 | tee "$log"; then
    echo "<< [$(date '+%F %T')] OK     $name"
    return 0
  else
    local rc=${PIPESTATUS[0]}
    echo ""
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "!! [$(date '+%F %T')] FAILED $name (exit $rc) — see $log"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    FAILURES+=("$name (exit $rc, log: $log)")
    return $rc
  fi
}

train_with_retry() {   # train_with_retry <name> <config> <outdir>
  local name="$1" cfg="$2" out="$3"
  if [ "$SKIP_EXISTING" = "1" ] && [ -f "$out/checkpoint_last.pt" ]; then
    echo "** SKIP $name — $out/checkpoint_last.pt exists (SKIP_EXISTING=0 to force)"
    return 0
  fi
  if ! run_step "$name" env WANDB_JOB_TYPE=train \
      python scripts/train.py --config "$cfg" --output-dir "$out"; then
    echo "** retrying $name once after 60s..."
    sleep 60
    run_step "${name}_retry" env WANDB_JOB_TYPE=train \
      python scripts/train.py --config "$cfg" --output-dir "$out"
  fi
}

pick_ckpt() {  # pick_ckpt <dir> <best|last>
  local dir="$1" which="$2"
  if [ "$which" = best ] && [ -f "$dir/checkpoint_best.pt" ]; then
    echo "$dir/checkpoint_best.pt"
  elif [ -f "$dir/checkpoint_last.pt" ]; then
    echo "$dir/checkpoint_last.pt"
  else
    echo ""
  fi
}

ablate() {   # ablate <tag> <config> <ckpt>
  local tag="$1" cfg="$2" ckpt="$3"
  [ -n "$ckpt" ] || { echo "!! ablate $tag: no checkpoint"; FAILURES+=("ablate_$tag (no ckpt)"); return 1; }
  run_step "ablate_${tag}" env WANDB_JOB_TYPE=ablate WANDB_NAME="v10run-ablate-${tag}" \
    python scripts/ablate_adaptive_scorers.py \
    --config "$cfg" --checkpoint "$ckpt" \
    --split synthetic --rank-by tpr1 --bootstrap 1000 --k-sweep 4,8,16,25 \
    --crop-syn --out-dir "$RESDIR" --tag "ablate_${tag}" --wandb
}

# =============================================================================
# 0. Preflight
# =============================================================================
test -d data/processed/enron_shortmail || { echo "FATAL: enron_shortmail corpus missing"; exit 1; }
test -f "$V9_DIR/checkpoint_best.pt" || { echo "FATAL: v9 checkpoints missing ($V9_DIR)"; exit 1; }
python - <<'PYEOF' || { echo "FATAL: pauc/genuine_vs_other_5pct metric missing — centroid_probe.py not updated?"; exit 1; }
import sys; sys.path.insert(0, "src")
import numpy as np
from email_fraud.scoring.centroid_probe import _metrics_for_score_set
m = _metrics_for_score_set(np.array([1.0, 0.9]), np.array([0.1, 0.2]), np.array([0.1, 0.3]))
assert "pauc/genuine_vs_other_5pct" in m and "pauc/min_other_synthetic_5pct" in m
PYEOF

# =============================================================================
# 1. Train v10 (the authorship specialist) — the critical path
# =============================================================================
train_with_retry 10_train_v10 "$V10_CFG" "$V10_DIR"

# =============================================================================
# 2. v10 evals: probe + ablation on best AND last
# =============================================================================
V10_BEST=$(pick_ckpt "$V10_DIR" best)
V10_LAST=$(pick_ckpt "$V10_DIR" last)
if [ -n "$V10_BEST" ]; then
  run_step 20_probe_v10 env WANDB_JOB_TYPE=probe WANDB_NAME="v10run-probe" \
    python scripts/probe_authenticity.py \
    --checkpoint "$V10_BEST" --config "$V10_CFG" \
    --split train --mode both --out "$RESDIR/probe_v10.json" --wandb
  ablate v10_best "$V10_CFG" "$V10_BEST"
  [ "$V10_LAST" != "$V10_BEST" ] && ablate v10_last "$V10_CFG" "$V10_LAST"
else
  echo "!! v10 produced no checkpoint — falling back to v9 checkpoint_last as the authorship model."
  FAILURES+=("v10 training (no checkpoint)")
fi

# =============================================================================
# 3. v9 re-ablation with the new FPR_other / crop columns (detector role)
# =============================================================================
ablate v9_best configs/experiments/v9_episodic_shortmail.yaml "$V9_DIR/checkpoint_best.pt"
ablate v9_last configs/experiments/v9_episodic_shortmail.yaml "$V9_DIR/checkpoint_last.pt"

# =============================================================================
# 4. The deliverable: v10 (authorship) × v9 (detector) fusion
# =============================================================================
AUTH_CKPT="${V10_BEST:-$V9_DIR/checkpoint_last.pt}"
run_step 40_fusion python scripts/eval_two_model_fusion.py \
  --config "$V10_CFG" \
  --authorship-ckpt "$AUTH_CKPT" \
  --detector-ckpt "$V9_DIR/checkpoint_best.pt" \
  --out "$RESDIR/fusion_v10xv9.json"

# =============================================================================
# 5. Soup baseline: can ONE blended model match the two-model pair?
# =============================================================================
run_step 50_soup_v9 python scripts/eval_checkpoint_soup.py \
  --config configs/experiments/v9_episodic_shortmail.yaml \
  --ckpt-a "$V9_DIR/checkpoint_best.pt" \
  --ckpt-b "$V9_DIR/checkpoint_last.pt" \
  --alphas 0,0.25,0.5,0.75,1 \
  --out "$RESDIR/soup_v9.json" \
  --save-best-to runs/lineage_v2/v9_soup/checkpoint_soup.pt

# =============================================================================
# 6. Optional: repair the crashed v6 baseline under the anti-Goodhart monitor
# =============================================================================
if [ "$INCLUDE_V6" = "1" ]; then
  V6_CFG="$LOGDIR/v6_minmonitor.yaml"
  python - "$V6_CFG" <<'PYEOF'
import sys, yaml
with open("configs/experiments/v6_bench.yaml") as fh:
    cfg = yaml.safe_load(fh)
cfg["training"]["monitor"] = "pauc/min_other_synthetic_5pct"
cfg.setdefault("wandb", {})["name"] = "v6-bench-minmonitor"
with open(sys.argv[1], "w") as fh:
    yaml.safe_dump(cfg, fh, sort_keys=False)
print(f"wrote {sys.argv[1]}")
PYEOF
  train_with_retry 60_train_v6 "$V6_CFG" "$V6_DIR"
  V6_BEST=$(pick_ckpt "$V6_DIR" best)
  # Common-corpus eval cfg for v6 (its native corpus differs)
  V6_EVAL="$LOGDIR/v6_common.yaml"
  python - "$V6_CFG" "$V6_EVAL" <<'PYEOF'
import sys, yaml
with open(sys.argv[1]) as fh:
    cfg = yaml.safe_load(fh)
data = cfg.setdefault("data", {})
data["processed_dir"] = "data/processed/enron_shortmail"
data.setdefault("preprocessing", {}).update(
    {"strip_signatures": False, "min_body_chars": 20, "min_body_words": 5})
data.setdefault("augmentation", {})["synthetic_path"] = "data/synthetic/enron_synthetic_v2"
with open(sys.argv[2], "w") as fh:
    yaml.safe_dump(cfg, fh, sort_keys=False)
print(f"wrote {sys.argv[2]}")
PYEOF
  [ -n "$V6_BEST" ] && ablate v6_repair "$V6_EVAL" "$V6_BEST"
fi

# =============================================================================
# 7. Digest
# =============================================================================
DIGEST="$RESDIR/DIGEST.txt"
python - <<'PYEOF' | tee "$DIGEST"
import json, pathlib
res = pathlib.Path("results/lineage_v2")

print("=" * 78)
print("V10 OVERNIGHT DIGEST — two-model pair (common corpus, linear_z3, K=8)")
print("=" * 78)

def row(tag, label):
    p = res / f"ablate_{tag}.json"
    if not p.exists():
        print(f"{label:24} — MISSING ({p.name})")
        return
    d = json.loads(p.read_text())
    r = {x["scorer"]: x for x in d["rows"]}["baseline_linear_z3"]
    crop = f"{r.get('tpr1_crop', float('nan')):.3f}" if "tpr1_crop" in r else "  n/a"
    print(f"{label:24} auc={r['auc']:.3f} tpr1={r['tpr1']:.3f} tpr5={r['tpr5']:.3f} "
          f"fpr_oth@5={r.get('fpr_other_at_5', float('nan')):.3f} "
          f"auc_g_oth={r.get('auc_g_other', float('nan')):.3f} tpr1_crop={crop}")

row("v9_best",  "v9 best  (LLM detector)")
row("v9_last",  "v9 last")
row("v10_best", "v10 best (authorship)")
row("v10_last", "v10 last")
row("v6_repair","v6 repaired (min mon.)")

p = res / "soup_v9.json"
if p.exists():
    d = json.loads(p.read_text())
    b = max(d["rows"], key=lambda r: r["min_auc"])
    print(f"\nv9 soup best α={b['alpha']:.2f}: auc_syn={b['auc_g_syn']:.3f} "
          f"auc_oth={b['auc_g_oth']:.3f} tpr5={b['tpr5']:.3f} fpr_oth@5={b['fpr_other_at_5']:.3f}")

p = res / "fusion_v10xv9.json"
if p.exists():
    d = json.loads(p.read_text())
    print(f"\nFUSION v10(ep {d['authorship_epoch']}) × v9(ep {d['detector_epoch']}):")
    for label, s in [("authorship alone", d["authorship_alone"]),
                     ("detector alone", d["detector_alone"]),
                     ("fused soft-min", d["soft_min"])]:
        print(f"  {label:18} auc_syn={s['auc_g_syn']:.3f} auc_oth={s['auc_g_oth']:.3f} "
              f"tpr5={s['tpr5_syn']:.3f} fpr_oth@syn5={s['fpr_other_at_syn5']:.3f}")
    for tag, g in d["and_gate"].items():
        print(f"  AND-gate {tag:6} TPR={g['tpr']:.3f} FPR_other={g['fpr_other']:.3f} "
              f"FPR_syn={g['fpr_syn']:.3f}")
PYEOF

echo ""
echo "=================================================================="
if [ ${#FAILURES[@]} -gt 0 ]; then
  echo "!! COMPLETED WITH ${#FAILURES[@]} FAILURE(S):"
  for f in "${FAILURES[@]}"; do echo "!!   - $f"; done
  echo "=================================================================="
  exit 1
else
  echo "ALL STAGES OK at $(date '+%F %T'). Digest: $DIGEST"
  echo "=================================================================="
fi
