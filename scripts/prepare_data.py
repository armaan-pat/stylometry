"""Prepare the Enron dataset from a raw maildir download.

Parses, preprocesses, filters, splits, saves the Enron corpus as a
HuggingFace Arrow DatasetDict, and generates JSONL verification pairs
for evaluation.

Usage::

    # Full run (defaults)
    python scripts/prepare_data.py

    # Explicit paths / options
    python scripts/prepare_data.py \\
        --data-dir   data/raw/enron \\
        --output-dir data/processed/enron \\
        --min-emails 100 \\
        --workers    8

    # Quick smoke-test — first 5 senders only, no download needed
    python scripts/prepare_data.py --dry-run

    # Skip pair generation
    python scripts/prepare_data.py --no-pairs

Expected raw layout::

    data/raw/enron/
    └── maildir/
        ├── allen-p/
        │   ├── sent/
        │   ├── sent_items/
        │   └── ...
        └── ...

Each file is a raw RFC-2822 email.  Only files inside sent-folder
subdirectories are used (clean single-author signal).

Output layout::

    data/processed/enron/
    ├── dataset_dict.json
    ├── train/
    ├── validation/
    ├── test/
    ├── train_pairs.jsonl
    ├── validation_pairs.jsonl
    └── test_pairs.jsonl

Each Arrow record: {"text": str, "sender_id": str}.
Each JSONL line:   {"pair": ["<text1>", "<text2>"], "same": true|false}

Split strategy: sender-disjoint.  No sender appears in more than one split.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Iterator

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from email_fraud.config import PreprocessingConfig
from email_fraud.data.preprocessing import clean_email_raw

logger = logging.getLogger(__name__)

SENT_FOLDERS = {"sent", "sent_items", "sent_mail", "_sent_mail"}
BATCH_SIZE   = 500


# ---------------------------------------------------------------------------
# Cross-platform directory iteration
#
# The Enron maildir was created on Unix; files named "1." "2." etc. are valid
# on NTFS but Python's open() normalises away trailing dots via the Windows
# API.  The Windows branch uses FindFirstFileW / FindNextFileW to retrieve the
# 8.3 short name and open via that.  On macOS / Linux the plain pathlib branch
# is used instead.
# ---------------------------------------------------------------------------

try:
    import ctypes
    import ctypes.wintypes

    class _FILETIME(ctypes.Structure):
        _fields_ = [("dwLow", ctypes.wintypes.DWORD), ("dwHigh", ctypes.wintypes.DWORD)]

    class _WIN32_FIND_DATAW(ctypes.Structure):
        _fields_ = [
            ("dwFileAttributes",   ctypes.wintypes.DWORD),
            ("ftCreationTime",     _FILETIME),
            ("ftLastAccessTime",   _FILETIME),
            ("ftLastWriteTime",    _FILETIME),
            ("nFileSizeHigh",      ctypes.wintypes.DWORD),
            ("nFileSizeLow",       ctypes.wintypes.DWORD),
            ("dwReserved0",        ctypes.wintypes.DWORD),
            ("dwReserved1",        ctypes.wintypes.DWORD),
            ("cFileName",          ctypes.c_wchar * 260),
            ("cAlternateFileName", ctypes.c_wchar * 14),
        ]

    _k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    _k32.FindFirstFileW.restype  = ctypes.c_void_p
    _k32.FindFirstFileW.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(_WIN32_FIND_DATAW)]
    _k32.FindNextFileW.restype   = ctypes.wintypes.BOOL
    _k32.FindNextFileW.argtypes  = [ctypes.c_void_p, ctypes.POINTER(_WIN32_FIND_DATAW)]
    _k32.FindClose.restype       = ctypes.wintypes.BOOL
    _k32.FindClose.argtypes      = [ctypes.c_void_p]
    _INVALID_HANDLE              = ctypes.c_void_p(-1).value

    def _iter_dir(folder: str) -> Iterator[tuple[str, str, bool]]:
        """Yield (long_name, short_name, is_dir) for every entry in folder."""
        data = _WIN32_FIND_DATAW()
        h = _k32.FindFirstFileW(folder + "\\*", ctypes.byref(data))
        if h is None or h == _INVALID_HANDLE:
            return
        try:
            while True:
                long  = data.cFileName
                short = data.cAlternateFileName or long
                is_dir = bool(data.dwFileAttributes & 0x10)
                if long not in (".", ".."):
                    yield long, short, is_dir
                if not _k32.FindNextFileW(h, ctypes.byref(data)):
                    break
        finally:
            _k32.FindClose(h)

except AttributeError:
    # macOS / Linux: plain pathlib iteration (trailing-dot filenames are fine)
    def _iter_dir(folder: str) -> Iterator[tuple[str, str, bool]]:  # type: ignore[misc]
        for entry in pathlib.Path(folder).iterdir():
            yield entry.name, entry.name, entry.is_dir()


# ---------------------------------------------------------------------------
# Maildir scanning
# ---------------------------------------------------------------------------

def _collect_enron_paths(
    root: pathlib.Path,
) -> list[tuple[str, str, str]]:
    """Return [(file_path, sender_id, folder_name), ...] for all sent-folder files.

    Uses the cross-platform _iter_dir so trailing-dot filenames on Windows
    are opened via their 8.3 short names.
    """
    if not root.exists():
        raise FileNotFoundError(
            f"Maildir not found: {root}\n"
            "Download Enron with:\n"
            "  curl -L -O https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz\n"
            "  tar --no-same-owner -xzf enron_mail_20150507.tar.gz -C data/raw/enron/"
        )

    items: list[tuple[str, str, str]] = []
    for person_long, person_short, is_dir in _iter_dir(str(root)):
        if not is_dir:
            continue
        person_path = root / person_short
        for folder_long, folder_short, f_is_dir in _iter_dir(str(person_path)):
            if not f_is_dir:
                continue
            if folder_long.lower() not in SENT_FOLDERS:
                continue
            folder_path = person_path / folder_short
            for _, fname_short, is_file_dir in _iter_dir(str(folder_path)):
                if not is_file_dir:
                    items.append((
                        str(folder_path / fname_short),
                        person_long,
                        folder_long,
                    ))
    return items


def _filter_by_min_count(
    paths: list[tuple[str, str, str]],
    min_count: int,
) -> tuple[list[tuple[str, str, str]], int, int]:
    """Drop senders with fewer than min_count emails."""
    by_author: dict[str, list] = defaultdict(list)
    for item in paths:
        by_author[item[1]].append(item)
    kept = {a: p for a, p in by_author.items() if len(p) >= min_count}
    n_dropped = len(by_author) - len(kept)
    flat = [item for p in kept.values() for item in p]
    return flat, len(kept), n_dropped


# ---------------------------------------------------------------------------
# Parallel batch processing
# ---------------------------------------------------------------------------

def _clean_batch(
    batch: list[tuple[str, str, str]],
    config: PreprocessingConfig,
) -> list[dict]:
    """Clean one batch of (file_path, sender_id, folder) tuples.

    Designed to be called from a ProcessPoolExecutor worker.
    Returns list of {"text": str, "sender_id": str} dicts.
    """
    results = []
    for path_str, sender_id, _folder in batch:
        try:
            raw = pathlib.Path(path_str).read_text(encoding="latin-1")
        except OSError:
            continue
        body = clean_email_raw(raw, config)
        if body is None:
            continue
        results.append({"text": body, "sender_id": sender_id})
    return results


def build_records(
    all_paths: list[tuple[str, str, str]],
    config: PreprocessingConfig,
    workers: int = 1,
) -> list[dict]:
    """Parse and clean all email files in parallel; return {"text", "sender_id"} records."""
    batches = [
        all_paths[i : i + BATCH_SIZE]
        for i in range(0, len(all_paths), BATCH_SIZE)
    ]

    try:
        from tqdm import tqdm
        wrap = lambda it, **kw: tqdm(it, **kw)  # noqa: E731
    except ImportError:
        wrap = lambda it, **kw: it  # noqa: E731

    worker_fn = partial(_clean_batch, config=config)
    all_records: list[dict] = []

    if workers == 1:
        for result in wrap(map(worker_fn, batches), total=len(batches), desc="cleaning", unit="batch"):
            all_records.extend(result)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for result in wrap(
                pool.map(worker_fn, batches, chunksize=4),
                total=len(batches), desc="cleaning", unit="batch",
            ):
                all_records.extend(result)

    logger.info("Built %d records from %d files", len(all_records), len(all_paths))
    return all_records


# ---------------------------------------------------------------------------
# Sender-disjoint splitting
# ---------------------------------------------------------------------------

def split_by_sender(
    records: list[dict],
    val_frac: float,
    test_frac: float,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Partition records into sender-disjoint train / validation / test sets.

    Senders are shuffled with the given seed before assignment so the
    distribution is not deterministically biased by name alphabetical order.

    Returns:
        (train_records, val_records, test_records)
    """
    sender_map: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        sender_map[rec["sender_id"]].append(rec)

    senders = list(sender_map.keys())
    rng = random.Random(seed)
    rng.shuffle(senders)

    n = len(senders)
    n_test  = max(1, round(n * test_frac))
    n_val   = max(1, round(n * val_frac))
    n_train = n - n_val - n_test

    if n_train < 1:
        raise ValueError(
            f"Not enough senders ({n}) for val_frac={val_frac}, test_frac={test_frac}. "
            "Lower the fractions or reduce --min-emails."
        )

    test_senders  = senders[:n_test]
    val_senders   = senders[n_test : n_test + n_val]
    train_senders = senders[n_test + n_val:]

    def collect(ss: list[str]) -> list[dict]:
        return [rec for s in ss for rec in sender_map[s]]

    train = collect(train_senders)
    val   = collect(val_senders)
    test  = collect(test_senders)

    logger.info(
        "Split: %d train senders (%d emails) | %d val senders (%d emails) | "
        "%d test senders (%d emails)",
        len(train_senders), len(train),
        len(val_senders),   len(val),
        len(test_senders),  len(test),
    )
    return train, val, test


# ---------------------------------------------------------------------------
# JSONL pair generation
# ---------------------------------------------------------------------------

def generate_pairs(
    records: list[dict],
    output_path: pathlib.Path,
    n_pairs: int,
    seed: int,
    split_name: str = "",
) -> None:
    """Write balanced same/different-author JSONL verification pairs to *output_path*.

    Each line: {"pair": ["<text1>", "<text2>"], "same": true|false}
    """
    by_sender: dict[str, list[str]] = defaultdict(list)
    for rec in records:
        by_sender[rec["sender_id"]].append(rec["text"])

    same_senders = [s for s, texts in by_sender.items() if len(texts) >= 2]
    all_senders  = list(by_sender.keys())

    if len(same_senders) < 1 or len(all_senders) < 2:
        logger.warning(
            "Skipping pair generation for '%s': not enough senders "
            "(same-eligible=%d, total=%d).",
            split_name, len(same_senders), len(all_senders),
        )
        return

    rng    = random.Random(seed)
    n_each = n_pairs // 2
    pairs: list[dict] = []

    while len(pairs) < n_each:
        sender = rng.choice(same_senders)
        t1, t2 = rng.sample(by_sender[sender], 2)
        pairs.append({"pair": [t1, t2], "same": True})

    while len(pairs) < n_each * 2:
        s1, s2 = rng.sample(all_senders, 2)
        t1 = rng.choice(by_sender[s1])
        t2 = rng.choice(by_sender[s2])
        pairs.append({"pair": [t1, t2], "same": False})

    rng.shuffle(pairs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        for record in pairs:
            fh.write(json.dumps(record) + "\n")

    n_same = sum(1 for p in pairs if p["same"])
    logger.info(
        "  %-10s pairs: %d total (%d same, %d different) → %s",
        split_name, len(pairs), n_same, len(pairs) - n_same, output_path,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare Enron dataset for training.")
    p.add_argument("--data-dir",    default="data/raw/enron",
                   help="Root of the raw Enron download (must contain maildir/).")
    p.add_argument("--output-dir",  default="data/processed/enron",
                   help="Where to save the Arrow DatasetDict.")
    p.add_argument("--min-emails",  type=int, default=25,
                   help="Min sent emails per sender (default 25; use 100 for cleaner data).")
    p.add_argument("--val-split",   type=float, default=0.1)
    p.add_argument("--test-split",  type=float, default=0.1)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--workers",     type=int, default=None,
                   help="Parallel workers (default: os.cpu_count()).")
    # Preprocessing flags — each maps to a PreprocessingConfig field
    p.add_argument("--strip-quoted",      action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--strip-signatures",  action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--strip-boilerplate", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--entity-masking",    action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--fix-encoding",      action=argparse.BooleanOptionalAction, default=True,
                   help="Run ftfy to fix garbled encoding artifacts.")
    p.add_argument("--min-body-chars",    type=int, default=50,
                   help="Drop emails with fewer than this many characters after cleaning.")
    p.add_argument("--min-body-words",    type=int, default=20,
                   help="Drop emails with fewer than this many whitespace-split tokens after cleaning.")
    p.add_argument("--min-alnum-ratio",   type=float, default=0.60,
                   help="Drop emails where less than this fraction of characters are alphanumeric or space.")
    p.add_argument("--max-body-chars",    type=int, default=4000,
                   help="Truncate email bodies to this many characters.")
    p.add_argument("--dry-run", action="store_true",
                   help="Use only the first 5 senders — quick smoke-test without downloading data.")
    # Pair generation
    p.add_argument("--no-pairs", action="store_true",
                   help="Skip JSONL verification pair generation.")
    p.add_argument("--pairs-per-split", type=int, default=2000,
                   help="Total pairs per split (half same, half different; default: 2000).")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    from datasets import Dataset, DatasetDict

    config = PreprocessingConfig(
        strip_quoted=args.strip_quoted,
        strip_signatures=args.strip_signatures,
        strip_boilerplate=args.strip_boilerplate,
        entity_masking=args.entity_masking,
        fix_encoding=args.fix_encoding,
        min_body_chars=args.min_body_chars,
        min_body_words=args.min_body_words,
        min_alnum_ratio=args.min_alnum_ratio,
        max_body_chars=args.max_body_chars,
    )

    maildir = pathlib.Path(args.data_dir) / "maildir"
    logger.info("Scanning %s …", maildir)
    all_paths = _collect_enron_paths(maildir)

    all_paths, n_authors, n_dropped = _filter_by_min_count(all_paths, args.min_emails)
    logger.info(
        "%d authors kept  (%d dropped, <%d emails)  |  %d files to process",
        n_authors, n_dropped, args.min_emails, len(all_paths),
    )

    if args.dry_run:
        # Keep only the first 5 senders
        seen: dict[str, list] = defaultdict(list)
        for item in all_paths:
            seen[item[1]].append(item)
        all_paths = [item for s in list(seen.keys())[:5] for item in seen[s]]
        logger.info("--dry-run: limited to %d senders (%d files)", min(5, len(seen)), len(all_paths))

    workers = args.workers or os.cpu_count() or 1
    logger.info("Processing with %d worker(s)…", workers)
    records = build_records(all_paths, config, workers=workers)

    if not records:
        raise RuntimeError(
            "No usable records after preprocessing. "
            "Check --data-dir points to a valid Enron maildir and --min-emails is not too high."
        )

    train_recs, val_recs, test_recs = split_by_sender(
        records,
        val_frac=args.val_split,
        test_frac=args.test_split,
        seed=args.seed,
    )

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dict = DatasetDict({
        "train":      Dataset.from_list(train_recs),
        "validation": Dataset.from_list(val_recs),
        "test":       Dataset.from_list(test_recs),
    })
    dataset_dict.save_to_disk(str(output_dir))

    # Sanity-check: splits must be sender-disjoint
    train_s = {r["sender_id"] for r in train_recs}
    val_s   = {r["sender_id"] for r in val_recs}
    test_s  = {r["sender_id"] for r in test_recs}
    assert not (train_s & val_s),  "BUG: train/val senders overlap!"
    assert not (train_s & test_s), "BUG: train/test senders overlap!"
    assert not (val_s & test_s),   "BUG: val/test senders overlap!"

    logger.info("Saved DatasetDict → %s", output_dir)
    logger.info("  train:      %d emails  (%d senders)", len(train_recs), len(train_s))
    logger.info("  validation: %d emails  (%d senders)", len(val_recs),   len(val_s))
    logger.info("  test:       %d emails  (%d senders)", len(test_recs),  len(test_s))
    logger.info("Sender-disjoint split verified ✓")

    if not args.no_pairs:
        logger.info("Generating JSONL verification pairs (%d per split)…", args.pairs_per_split)
        for split_name, split_recs in [
            ("train",      train_recs),
            ("validation", val_recs),
            ("test",       test_recs),
        ]:
            generate_pairs(
                records=split_recs,
                output_path=output_dir / f"{split_name}_pairs.jsonl",
                n_pairs=args.pairs_per_split,
                seed=args.seed,
                split_name=split_name,
            )
        logger.info("Pair generation complete.")


if __name__ == "__main__":
    main()
