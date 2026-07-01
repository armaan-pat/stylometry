#!/usr/bin/env bash
# Generate the novel-vendor held-out pool. The multi-generator pool path 400s on
# OpenRouter, so generate each vendor SEPARATELY (both work fine alone) and merge.
# Vendors: Qwen-2.5-72B + DeepSeek-V3 — outside training {GPT,Llama} AND eval {Claude,Gemini}.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export WANDB_MODE=disabled
LOG=results/whitepaper/logs; mkdir -p "$LOG"
CFG=configs/experiments/_v12_heldout_eval.yaml
QDIR=data/synthetic/_novelvendor_qwen
DDIR=data/synthetic/_novelvendor_deepseek
MERGED=data/synthetic/enron_synthetic_novelvendor

gen () {  # $1=slug  $2=outdir
  [ -d "$2" ] && { echo "[skip] $2 exists"; return 0; }
  python scripts/generate_synthetic_emails.py --config "$CFG" \
    --generators "openrouter:$1" --from-split train \
    --n-per-sender 4 --n-examples 5 \
    --cross-register-fraction 0.4 --cross-length-fraction 0.2 \
    --output "$2" --seed 42 \
    >> "$LOG/gen_novelvendor.log" 2>&1 \
    && echo "[ok] generated $1 -> $2" || { echo "[FAIL] $1"; return 1; }
}

gen "qwen/qwen-2.5-72b-instruct" "$QDIR"
gen "deepseek/deepseek-chat"     "$DDIR"

python - "$QDIR" "$DDIR" "$MERGED" <<'PY'
import sys
from datasets import load_from_disk, concatenate_datasets
q, d, out = sys.argv[1], sys.argv[2], sys.argv[3]
ds = concatenate_datasets([load_from_disk(q), load_from_disk(d)])
ds.save_to_disk(out)
import collections
print("merged rows:", len(ds))
print("by generator:", dict(collections.Counter(ds["generator"])))
print("senders:", len(set(ds["sender_id"])))
PY
echo "ALL_DONE_GEN_NOVELVENDOR"
