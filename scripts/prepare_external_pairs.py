"""Convert an external authorship corpus into domain-OOD verification pairs.

Domain generalization is the one OOD axis the Enron-only pipeline cannot
manufacture: you need *real* text from another domain. This script turns a
downloaded corpus into the unified ``{pair, same, slice}`` JSONL that
scripts/eval_ood.py ingests via ``--extra-pairs``, so a new domain becomes one
more row in the per-slice OOD report.

Two input formats:

  --format pan
      PAN authorship-verification shared-task files. Reads a pairs file
      (``{"id": ..., "pair": [t1, t2], ...}``) and a truth file
      (``{"id": ..., "same": true/false, ...}``), joins on ``id``, and emits the
      labelled pairs directly. PAN already provides balanced pairs, so we pass
      them through (optionally capped with --max-pairs). PAN21 is cross-topic,
      which is exactly the hard generalization case.

  --format authors
      A generic corpus: one JSON object per line with a text field and an author
      field (``--text-field`` / ``--author-field``). We sample balanced
      same/different-author pairs ourselves (reusing build_ood_eval's samplers).
      Use this for Blog Authorship, Reddit (PAN/LUAR style), Amazon reviews, etc.
      once you've reshaped them to ``{"text": ..., "author": ...}`` lines.

Every output line is tagged with ``--slice`` (default ``domain:<name-from-output>``)
so eval_ood groups it as its own OOD slice.

Usage:
    # PAN (cross-topic) → one domain slice
    python scripts/prepare_external_pairs.py --format pan \\
        --pairs-file data/raw/pan/pan21/pairs.jsonl \\
        --truth-file data/raw/pan/pan21/truth.jsonl \\
        --slice domain:pan21 --max-pairs 2000 \\
        --output data/ood/pan21_pairs.jsonl

    # Generic {text, author} corpus → balanced pairs
    python scripts/prepare_external_pairs.py --format authors \\
        --input data/raw/blog/blogs.jsonl --text-field text --author-field id \\
        --slice domain:blog --n-per-class 1000 \\
        --output data/ood/blog_pairs.jsonl

    # Then fold into the OOD report:
    python scripts/eval_ood.py --run runs/<...> \\
        --pairs data/ood/enron_ood_pairs.jsonl \\
        --extra-pairs domain:pan21=data/ood/pan21_pairs.jsonl \\
        --extra-pairs domain:blog=data/ood/blog_pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

# Reuse the balanced-pair samplers so domain pairs are built the same way as the
# in-distribution OOD slices.
from build_ood_eval import _sample_diff, _sample_same  # noqa: E402


def _read_jsonl(path: Path):
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _truncate(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[:max_chars]


def _clean_text(text: str, args, cfg) -> str | None:
    """Truncate, and (if --preprocess) run project preprocess(); None = drop."""
    text = _truncate(str(text), args.max_chars)
    if args.preprocess:
        from email_fraud.data.preprocessing import preprocess

        text = preprocess(text, cfg.data.preprocessing)
    return text  # may be None when preprocess rejects it


# --------------------------------------------------------------------------- #
# PAN
# --------------------------------------------------------------------------- #

def _build_pan(args, cfg) -> list[dict]:
    pairs_path = Path(args.pairs_file)
    truth_path = Path(args.truth_file)
    if not pairs_path.exists() or not truth_path.exists():
        raise FileNotFoundError(
            f"PAN needs both --pairs-file ({pairs_path}) and --truth-file ({truth_path})."
        )
    truth: dict[str, bool] = {}
    for rec in _read_jsonl(truth_path):
        truth[str(rec["id"])] = bool(rec["same"])

    raw: list[tuple] = []  # (t1, t2, same)
    missing = 0
    for rec in _read_jsonl(pairs_path):
        rid = str(rec.get("id"))
        if rid not in truth:
            missing += 1
            continue
        if "pair" in rec:
            t1, t2 = rec["pair"]
        else:
            t1 = rec.get("text1") or rec.get("text_a")
            t2 = rec.get("text2") or rec.get("text_b")
        raw.append((str(t1), str(t2), truth[rid]))
    if missing:
        print(f"[pan] {missing} pairs had no matching truth id; skipped.")

    # Truncate (and optionally preprocess) each side; drop a pair if either side
    # is rejected by preprocess().
    cleaned: list[tuple[str, str, bool]] = []
    for t1, t2, same in raw:
        c1 = _clean_text(t1, args, cfg)
        c2 = _clean_text(t2, args, cfg)
        if c1 is None or c2 is None:
            continue
        cleaned.append((c1, c2, same))

    rng = random.Random(args.seed)
    rng.shuffle(cleaned)
    if args.max_pairs:
        # Keep class balance when capping.
        pos = [r for r in cleaned if r[2]][: args.max_pairs // 2]
        neg = [r for r in cleaned if not r[2]][: args.max_pairs // 2]
        cleaned = pos + neg
        rng.shuffle(cleaned)

    return [{"pair": [t1, t2], "same": same, "slice": args.slice}
            for t1, t2, same in cleaned]


# --------------------------------------------------------------------------- #
# Generic {text, author}
# --------------------------------------------------------------------------- #

def _build_authors(args, cfg) -> list[dict]:
    inp = Path(args.input)
    if not inp.exists():
        raise FileNotFoundError(f"--input not found: {inp}")

    by_author: dict[str, list[str]] = defaultdict(list)
    n_read = 0
    for rec in _read_jsonl(inp):
        text = rec.get(args.text_field)
        author = rec.get(args.author_field)
        if text is None or author is None:
            continue
        n_read += 1
        text = _clean_text(text, args, cfg)
        if text is None or len(text.split()) < args.min_words:
            continue
        by_author[str(author)].append(text)

    n_authors = len(by_author)
    n_multi = sum(1 for v in by_author.values() if len(v) >= 2)
    print(f"[authors] read {n_read} docs → {n_authors} authors "
          f"({n_multi} with ≥2 docs for same-author pairs)")

    rng = random.Random(args.seed)
    same = _sample_same(by_author, args.n_per_class, rng)
    diff = _sample_diff(by_author, args.n_per_class, rng)
    records = (
        [{"pair": [a, b], "same": True, "slice": args.slice} for a, b in same]
        + [{"pair": [a, b], "same": False, "slice": args.slice} for a, b in diff]
    )
    rng.shuffle(records)
    return records


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert an external corpus into domain-OOD verification pairs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--format", required=True, choices=["pan", "authors"])
    p.add_argument("--output", required=True, help="Output JSONL path.")
    p.add_argument("--slice", default=None,
                   help="Slice tag for every pair (default: domain:<output stem>).")
    p.add_argument("--config", default="configs/base.yaml",
                   help="Config supplying preprocessing settings when --preprocess is set.")
    p.add_argument("--preprocess", action="store_true",
                   help="Run the project's preprocess() on each text (drops too-short/junk). "
                        "Off by default — external docs aren't emails.")
    p.add_argument("--max-chars", type=int, default=4000,
                   help="Truncate each text to this many characters (default: 4000, "
                        "matching training max_body_chars).")
    p.add_argument("--seed", type=int, default=123)
    # PAN
    p.add_argument("--pairs-file", help="[pan] PAN pairs JSONL.")
    p.add_argument("--truth-file", help="[pan] PAN truth JSONL.")
    p.add_argument("--max-pairs", type=int, default=None,
                   help="[pan] Cap total pairs (kept class-balanced).")
    # authors
    p.add_argument("--input", help="[authors] {text, author} JSONL.")
    p.add_argument("--text-field", default="text", help="[authors] text field name.")
    p.add_argument("--author-field", default="author", help="[authors] author field name.")
    p.add_argument("--min-words", type=int, default=5,
                   help="[authors] drop docs with fewer than this many words.")
    p.add_argument("--n-per-class", type=int, default=1000,
                   help="[authors] same pairs (and equally many different) to sample.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output)
    if args.slice is None:
        args.slice = f"domain:{out.stem}"
    if not args.slice.startswith("domain:"):
        print(f"[warn] slice '{args.slice}' doesn't start with 'domain:'; "
              "eval_ood will still group it, but the convention is domain:<name>.")

    from email_fraud.config import load_config
    cfg = load_config(args.config) if args.preprocess else None

    if args.format == "pan":
        records = _build_pan(args, cfg)
    else:
        records = _build_authors(args, cfg)

    n_same = sum(1 for r in records if r["same"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    print(f"Wrote {len(records)} pairs ({n_same} same, {len(records) - n_same} different) "
          f"tagged '{args.slice}' → {out}")
    print(f"Use it: python scripts/eval_ood.py --run <run> --pairs <ood.jsonl> "
          f"--extra-pairs {args.slice}={out}")


if __name__ == "__main__":
    main()
