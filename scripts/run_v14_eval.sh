#!/usr/bin/env bash
# Evaluate a v14 checkpoint on every probe, vs v12. Run after training completes.
#   bash scripts/run_v14_eval.sh [CKPT]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export WANDB_MODE="${WANDB_MODE:-offline}"

CKPT="${1:-runs/v14/lora/checkpoint_best.pt}"
TAG="${2:-v14lora}"      # output suffix, e.g. v14lora or v14blora
[ -f "$CKPT" ] || CKPT="$(dirname "$CKPT")/checkpoint_last.pt"
RES="results/v14"
HELDOUT_CFG="configs/experiments/_v12_heldout_eval.yaml"
mkdir -p "$RES/logs"
echo ">> evaluating $TAG: $CKPT"

run() { echo "== $1 =="; shift; "$@" 2>&1 | grep -ivE "wandb:|FutureWarning|UserWarning|warnings.warn" | tail -25; }

# 1. Large blog held-out probe (150 authors) — uses the ckpt's own config (encoder arch).
run "blog probe" python scripts/eval_ood.py \
  --checkpoint "$CKPT" \
  --pairs data/ood/blog_heldout_pairs.jsonl \
  --json "$RES/blog_probe_${TAG}.json"

# 2. Enron held-out gen/len/register (same pairs v12 used) + PAN + blog domain, one shot.
run "ood gen/len/register + domains" python scripts/eval_ood.py \
  --checkpoint "$CKPT" \
  --pairs results/v12/ood_v12heldout_pairs.jsonl \
  --extra-pairs domain:pan20_xtopic=data/ood/pan20_xtopic_pairs.jsonl \
  --extra-pairs domain:blog=data/ood/blog_pairs.jsonl \
  --json "$RES/ood_${TAG}_heldoutCG.json"

# 3. Held-out Claude+Gemini pool: TPR@1%/FPR_other (config supplies the held-out synthetic
#    pool; encoder weights load via --checkpoint since arch matches).
run "heldout-gen ablation" python scripts/ablate_adaptive_scorers.py \
  --config "$HELDOUT_CFG" --checkpoint "$CKPT" \
  --split synthetic --rank-by tpr1 --bootstrap 1000 \
  --out-dir "$RES" --tag heldoutCG_${TAG}

# 4. Digest: v12 (baseline) vs v14 (identity-only) vs v14b (synthesis), if present.
python - <<'PY'
import json, pathlib
R = pathlib.Path("results")
def load(p):
    p = R / p
    return json.loads(p.read_text()) if p.exists() else None
def auc(d, s): return round(d[s]["AUC"],3) if d and s in d else None
def col(x): return f"{x}" if x is not None else "  - "

print("\n================ v12 vs v14 vs v14b DIGEST ================")
tags = [("v12","v12lora"),("v14","v14lora"),("v14b","v14blora")]

# blog probe (v12 baseline is blog_probe_v12lora.json)
print("\nBlog held-out probe (150 unseen authors) — AUC:")
bp = {n: load(f"v14/blog_probe_{t}.json") for n,t in tags}
for s in ["blog:random","blog:sametopic"]:
    print(f"  {s:16} " + "  ".join(f"{n}={col(auc(bp[n],s))}" for n,_ in tags))

# ood slices
ov = {"v12": load("v12/ood_v12lora_final_heldoutCG.json") or load("v12/ood_domains_v12.json"),
      "v14": load("v14/ood_v14lora_heldoutCG.json"),
      "v14b": load("v14/ood_v14blora_heldoutCG.json")}
print("\nOOD slices — AUC:")
for s in ["gen:openrouter:anthropic/claude-3.5-haiku","gen:openrouter:google/gemini-2.5-flash",
          "domain:pan20_xtopic","domain:blog","lenmix:short_long","len:short","register:cross"]:
    print(f"  {s:42} " + "  ".join(f"{n}={col(auc(ov[n],s))}" for n in ["v12","v14","v14b"]))

# heldout ablation
av = {"v12": load("v12/heldoutCG_v12lora_final.json"),
      "v14": load("v14/heldoutCG_v14lora.json"),
      "v14b": load("v14/heldoutCG_v14blora.json")}
def row(d): return next((r for r in d["rows"] if r["scorer"]=="baseline_linear_z3"), None) if d else None
print("\nHeld-out Claude+Gemini pool (baseline_linear_z3):")
for m in ["auc","tpr1","tpr5"]:
    vals=[]
    for n in ["v12","v14","v14b"]:
        r=row(av[n]); vals.append(f"{n}={col(round(r[m],3) if r else None)}")
    print(f"  {m:6} " + "  ".join(vals))
rb = row(av["v14b"])
if rb: print(f"  fpr_other@5 v14b: {rb.get('fpr_other_at_5')}")
print("\nWIN (v14b) = keep v14's PAN ~0.88 AND recover v12's gen: ~0.83 / pool TPR@1% ~0.13.")
PY
echo "Done. Results in $RES/. Fill docs/v14_results.md from the digest."
