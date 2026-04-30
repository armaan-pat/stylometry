#!/usr/bin/env bash
# download_and_test.sh — Download the Enron corpus and smoke-test preprocessing.
#
# Usage:
#   bash scripts/download_and_test.sh               # full download + dry-run
#   bash scripts/download_and_test.sh --skip-download  # already have the data
#
# What this does:
#   1. Creates data/raw/enron/
#   2. Downloads enron_mail_20150507.tar.gz from CMU (~450 MB) if not present
#   3. Extracts the tarball if maildir/ doesn't exist yet
#   4. Installs Python dependencies (pip install -e .)
#   5. Runs prepare_data.py --dry-run (5 senders only) to verify preprocessing
#   6. Prints a sample of cleaned emails

set -euo pipefail

# ── Resolve project root (the directory containing this script's parent) ──────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# DATA_ROOT can be overridden to keep data outside the repo (e.g. on RunPod:
#   export DATA_ROOT=/workspace && bash scripts/download_and_test.sh)
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT}"

DATA_DIR="$DATA_ROOT/data/raw/enron"
TARBALL="$DATA_DIR/enron_mail_20150507.tar.gz"
MAILDIR="$DATA_DIR/maildir"
PROCESSED_DIR="$DATA_ROOT/data/processed/enron_dryrun"
ENRON_URL="https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz"

SKIP_DOWNLOAD=false
for arg in "$@"; do
  [[ "$arg" == "--skip-download" ]] && SKIP_DOWNLOAD=true
done

echo "========================================================"
echo "  Enron download + preprocessing smoke-test"
echo "  Project root : $PROJECT_ROOT"
echo "  Data root    : $DATA_ROOT"
echo "  Data dir     : $DATA_DIR"
echo "========================================================"

# ── 1. Create data directory ─────────────────────────────────────────────────
mkdir -p "$DATA_DIR"

# ── 2. Download ───────────────────────────────────────────────────────────────
if [[ "$SKIP_DOWNLOAD" == true ]]; then
  echo "[skip] --skip-download set; using existing data."
elif [[ -d "$MAILDIR" ]]; then
  echo "[skip] maildir/ already exists — skipping download."
elif [[ -f "$TARBALL" ]]; then
  echo "[skip] Tarball already downloaded — skipping curl."
else
  echo "[1/4] Downloading Enron corpus (~450 MB) …"
  curl -L --progress-bar -o "$TARBALL" "$ENRON_URL"
  echo "      Done: $TARBALL"
fi

# ── 3. Extract ────────────────────────────────────────────────────────────────
if [[ -d "$MAILDIR" ]]; then
  echo "[skip] maildir/ already extracted."
elif [[ -f "$TARBALL" ]]; then
  echo "[2/4] Extracting tarball …"
  tar --no-same-owner -xzf "$TARBALL" -C "$DATA_DIR/"
  echo "      Done: $MAILDIR"
else
  echo "ERROR: No tarball found at $TARBALL and maildir/ does not exist."
  echo "       Run without --skip-download, or place the tarball manually."
  exit 1
fi

# ── 4. Install dependencies ───────────────────────────────────────────────────
echo "[3/4] Installing Python dependencies …"
pip install -e "$PROJECT_ROOT[dev]" --quiet

# ── 5. Dry-run preprocessing ──────────────────────────────────────────────────
echo ""
echo "[4/4] Running preprocessing dry-run (5 senders) …"
python "$SCRIPT_DIR/prepare_data.py" \
  --data-dir   "$DATA_DIR" \
  --output-dir "$PROCESSED_DIR" \
  --min-emails 10 \
  --dry-run \
  --fix-encoding \
  --strip-quoted \
  --strip-signatures \
  --no-entity-masking \
  --min-body-chars 50 \
  --max-body-chars 4000

# ── 6. Show a sample ──────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  Sample from the processed dataset"
echo "========================================================"
PROCESSED_DIR="$PROCESSED_DIR" python - <<'PYEOF'
import sys, os, pathlib
sys.path.insert(0, "src")

from datasets import load_from_disk

processed = pathlib.Path(os.environ["PROCESSED_DIR"])
ds = load_from_disk(str(processed))
train = ds["train"]
print(f"\nDatasetDict loaded successfully.")
print(f"  train      : {len(train):>6} emails from {len(set(train['sender_id']))} senders")
print(f"  validation : {len(ds['validation']):>6} emails")
print(f"  test       : {len(ds['test']):>6} emails")
print()
for i in range(min(3, len(train))):
    sid  = train[i]["sender_id"]
    body = train[i]["text"]
    preview = body[:200].replace("\n", " ")
    print(f"  [{sid}] {preview!r}")
    print()
PYEOF

echo ""
echo "Smoke-test complete.  To run the full preprocessing:"
echo ""
echo "  python scripts/prepare_data.py \\"
echo "    --data-dir data/raw/enron \\"
echo "    --output-dir data/processed/enron \\"
echo "    --min-emails 100 \\"
echo "    --workers \$(nproc)"
