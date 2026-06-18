#!/usr/bin/env bash
# Evaluate a v14 checkpoint on every probe, vs v12. Run after training completes.
#   bash scripts/run_v14_eval.sh [CKPT]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export WANDB_MODE="${WANDB_MODE:-offline}"

CKPT="${1:-runs/v14/lora/checkpoint_best.pt}"
[ -f "$CKPT" ] || CKPT="runs/v14/lora/checkpoint_last.pt"
RES="results/v14"
HELDOUT_CFG="configs/experiments/_v12_heldout_eval.yaml"
mkdir -p "$RES/logs"
echo ">> evaluating v14: $CKPT"

run() { echo "== $1 =="; shift; "$@" 2>&1 | grep -ivE "wandb:|FutureWarning|UserWarning|warnings.warn" | tail -25; }

# 1. Large blog held-out probe (150 authors) — uses v14's own config (encoder arch).
run "blog probe" python scripts/eval_ood.py \
  --checkpoint "$CKPT" \
  --pairs data/ood/blog_heldout_pairs.jsonl \
  --json "$RES/blog_probe_v14lora.json"

# 2. Enron held-out gen/len/register (same pairs v12 used) + PAN + blog domain, one shot.
run "ood gen/len/register + domains" python scripts/eval_ood.py \
  --checkpoint "$CKPT" \
  --pairs results/v12/ood_v12heldout_pairs.jsonl \
  --extra-pairs domain:pan20_xtopic=data/ood/pan20_xtopic_pairs.jsonl \
  --extra-pairs domain:blog=data/ood/blog_pairs.jsonl \
  --json "$RES/ood_v14lora_heldoutCG.json"

# 3. Held-out Claude+Gemini pool: TPR@1%/FPR_other (config supplies the held-out synthetic
#    pool; v14 encoder weights load via --checkpoint since arch matches).
run "heldout-gen ablation" python scripts/ablate_adaptive_scorers.py \
  --config "$HELDOUT_CFG" --checkpoint "$CKPT" \
  --split synthetic --rank-by tpr1 --bootstrap 1000 \
  --out-dir "$RES" --tag heldoutCG_v14lora

# 4. Digest: v12 vs v14
python - <<'PY'
import json, pathlib
R = pathlib.Path("results")
def load(p):
    p = R / p
    return json.loads(p.read_text()) if p.exists() else None
def auc(d, s): return round(d[s]["AUC"],3) if d and s in d else None

print("\n==================== v12 vs v14 DIGEST ====================")
# blog probe
bv12, bv14 = load("v14/blog_probe_v12lora.json"), load("v14/blog_probe_v14lora.json")
print("\nBlog held-out probe (150 unseen authors):")
for s in ["blog:random","blog:sametopic"]:
    print(f"  {s:16} v12 AUC {auc(bv12,s)}  ->  v14 AUC {auc(bv14,s)}")

# ood slices
ov14 = load("v14/ood_v14lora_heldoutCG.json")
ov12 = load("v12/ood_v12lora_final_heldoutCG.json") or load("v12/ood_domains_v12.json")
print("\nOOD slices (v12 -> v14 AUC):")
for s in ["gen:openrouter:anthropic/claude-3.5-haiku","gen:openrouter:google/gemini-2.5-flash",
          "domain:pan20_xtopic","domain:blog","lenmix:short_long","len:short","register:cross"]:
    print(f"  {s:42} {auc(ov12,s)} -> {auc(ov14,s)}")

# heldout ablation
av14 = load("v14/heldoutCG_v14lora.json")
av12 = load("v12/heldoutCG_v12lora_final.json")
def row(d):
    return next((r for r in d["rows"] if r["scorer"]=="baseline_linear_z3"), None) if d else None
r12, r14 = row(av12), row(av14)
print("\nHeld-out Claude+Gemini pool (baseline_linear_z3):")
if r14:
    for m in ["auc","tpr1","tpr5"]:
        v12v = r12[m] if r12 else None
        print(f"  {m:6} v12 {v12v} -> v14 {round(r14[m],3)}")
    print(f"  fpr_other@5 v14: {r14.get('fpr_other_at_5')}")
print("\nWIN = v14 lifts blog/PAN/Enron authorship without losing held-out-generator catch.")
PY
echo "Done. Results in $RES/. Fill docs/v14_results.md from the digest."
