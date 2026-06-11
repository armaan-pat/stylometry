"""Analyze centroid-probe score distributions and recommend thresholds.

Why this exists
---------------
The training-time threshold metrics (threshold_0.95/*) use fixed cutoffs on a
score that is structurally bounded below 0.95 for realistic data. They are
near-zero by construction and tell us nothing about the model. This script
reads the dumped probe data and produces the metrics that actually matter:

  - Score histograms per pool (genuine / other / synthetic).
  - ROC + AUC for genuine vs other, vs synthetic, vs pooled impostors.
  - Score threshold for a target FPR, with precision and recall there.
  - Score threshold for a target precision, with FPR and recall.

Two input formats supported:

  1. scores_final.json — canonical-fn score arrays from one training run.
     Single score function; arguments to --score-fn / --compare are ignored.

  2. probe_raw.json — raw (cos_sim, spread) per query, so this script can
     compute scores under *any* score function in the registry. Pair with
     --score-fn linear_z3|linear_z2|cosine|sigmoid_z|neg_z, or use
     --compare-all to print a side-by-side AUC table.

Usage
-----
    # Operating points for the canonical score function:
    python scripts/analyze_thresholds.py runs/v6_luar_lora_syn/<TS>

    # Try a different score function on the same dumped run:
    python scripts/analyze_thresholds.py runs/.../ --score-fn cosine

    # Compare every score function in the registry on the same dump:
    python scripts/analyze_thresholds.py runs/.../ --compare-all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

# The script imports from the source tree so adding a new score function in
# src/email_fraud/scoring/score_functions.py makes it instantly available here.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
from email_fraud.scoring.score_functions import (  # noqa: E402
    SCORE_FNS,
    ALL_SCORE_FNS,
    CALIBRATED_SCORE_FNS,
    resolve as resolve_score_fn,
    resolve_calibrated,
)


def _load_scores(path: Path) -> dict[str, np.ndarray]:
    """Load canonical-fn scores from scores_final.json."""
    if path.is_dir():
        path = path / "scores_final.json"
    with path.open() as fh:
        raw = json.load(fh)
    out: dict[str, np.ndarray] = {}
    for k in ("genuine", "other", "synthetic"):
        out[k] = np.asarray(raw.get(k, []), dtype=np.float64)
    return out


def _load_raw(path: Path) -> dict | None:
    """Load probe_raw.json if present — needed for --score-fn / --compare-all."""
    if path.is_dir():
        path = path / "probe_raw.json"
    if not path.exists():
        return None
    with path.open() as fh:
        return json.load(fh)


def _scores_from_raw(raw: dict, score_fn_name: str) -> dict[str, np.ndarray]:
    # Raw rows are (cos_sim, spread) in older dumps and (cos_sim, spread,
    # z_scale) in current ones; calibrated fns need the third element and
    # plain fns ignore it.
    calibrated = score_fn_name in CALIBRATED_SCORE_FNS
    fn = resolve_calibrated(score_fn_name) if calibrated else resolve_score_fn(score_fn_name)
    out: dict[str, np.ndarray] = {}
    for pool in ("genuine", "other", "synthetic"):
        rows = raw.get(pool, [])
        vals = []
        for row in rows:
            c, s = float(row[0]), float(row[1])
            if calibrated:
                zs = float(row[2]) if len(row) > 2 else float("nan")
                vals.append(fn(c, s, zs))
            else:
                vals.append(fn(c, s))
        out[pool] = np.array(vals, dtype=np.float64)
    return out


def _load(path: Path) -> dict[str, np.ndarray]:
    # Kept for backward compatibility with the previous single-file flow.
    return _load_scores(path)


def _histogram(values: np.ndarray, bins: int = 30, width: int = 50) -> str:
    if len(values) == 0:
        return "  (empty)"
    counts, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    peak = counts.max() if counts.max() > 0 else 1
    lines = []
    for i, c in enumerate(counts):
        bar = "#" * int(width * c / peak)
        lines.append(f"  [{edges[i]:.2f}, {edges[i+1]:.2f})  {c:>4}  {bar}")
    return "\n".join(lines)


def _stats(name: str, arr: np.ndarray) -> str:
    if len(arr) == 0:
        return f"  {name:<10s}  n=0"
    return (
        f"  {name:<10s}  n={len(arr):>4d}  "
        f"mean={arr.mean():.3f}  std={arr.std():.3f}  "
        f"min={arr.min():.3f}  p25={np.percentile(arr, 25):.3f}  "
        f"med={np.median(arr):.3f}  p75={np.percentile(arr, 75):.3f}  "
        f"max={arr.max():.3f}"
    )


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    labels = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    scores = np.concatenate([pos, neg])
    return float(roc_auc_score(labels, scores))


def _threshold_for_fpr(pos: np.ndarray, neg: np.ndarray, target_fpr: float) -> dict:
    if len(neg) == 0 or len(pos) == 0:
        return {}
    tau = float(np.quantile(neg, 1.0 - target_fpr))
    recall = float((pos > tau).mean())
    tp = float((pos > tau).sum())
    fp = float((neg > tau).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return {
        "target_fpr": target_fpr,
        "threshold": tau,
        "recall": recall,
        "precision": precision,
        "actual_fpr": fp / len(neg),
    }


def _threshold_for_precision(
    pos: np.ndarray, neg: np.ndarray, target_precision: float
) -> dict:
    """Find the lowest τ where precision >= target. Returns recall at that τ."""
    if len(pos) == 0 or len(neg) == 0:
        return {}
    # Sweep over candidate thresholds = unique scores, find max recall meeting
    # the precision target. Standard PR-curve traversal.
    fpr, tpr, thresholds = roc_curve(
        np.concatenate([np.ones_like(pos), np.zeros_like(neg)]),
        np.concatenate([pos, neg]),
    )
    n_pos = len(pos)
    n_neg = len(neg)
    best = None
    for fp_rate, tp_rate, tau in zip(fpr, tpr, thresholds):
        tp = tp_rate * n_pos
        fp = fp_rate * n_neg
        if tp + fp == 0:
            continue
        precision = tp / (tp + fp)
        if precision >= target_precision and (best is None or tp_rate > best["recall"]):
            best = {
                "target_precision": target_precision,
                "threshold": float(tau),
                "recall": float(tp_rate),
                "precision": float(precision),
                "actual_fpr": float(fp_rate),
            }
    return best or {}


def _print_op(label: str, op: dict) -> None:
    if not op:
        print(f"  {label}: (no operating point found)")
        return
    extras = ""
    if "target_fpr" in op:
        extras = f"target_fpr={op['target_fpr']:.0%} actual={op['actual_fpr']:.1%}"
    elif "target_precision" in op:
        extras = f"target_prec={op['target_precision']:.0%}"
    print(
        f"  {label:<28s} τ={op['threshold']:.4f}  "
        f"recall={op['recall']:.3f}  precision={op['precision']:.3f}  ({extras})"
    )


def _report(scores: dict[str, np.ndarray], label: str = "") -> None:
    """Print the full diagnostic block for one score set."""
    g = scores.get("genuine", np.array([]))
    o = scores.get("other", np.array([]))
    s = scores.get("synthetic", np.array([]))

    if label:
        print(f"\n############## {label} ##############")

    print("\n== Score distributions ==")
    print(_stats("genuine", g))
    print(_stats("other", o))
    print(_stats("synthetic", s))

    print("\n== Histograms (counts, 30 bins over [0, 1]) ==")
    print("genuine:")
    print(_histogram(g))
    print("other:")
    print(_histogram(o))
    print("synthetic:")
    print(_histogram(s))

    print("\n== AUC ==")
    print(f"  genuine vs other      {_auc(g, o):.4f}")
    print(f"  genuine vs synthetic  {_auc(g, s):.4f}")
    impostors = np.concatenate([o, s]) if (len(o) or len(s)) else np.array([])
    print(f"  genuine vs all        {_auc(g, impostors):.4f}")

    print("\n== Operating points (anchored on FPR over pooled impostors) ==")
    for fpr in (0.01, 0.05, 0.10):
        _print_op(f"all impostors @ {fpr:.0%} FPR", _threshold_for_fpr(g, impostors, fpr))

    if len(s):
        print("\n== Operating points (anchored on synthetic-only FPR) ==")
        for fpr in (0.01, 0.05, 0.10):
            _print_op(f"synthetic @ {fpr:.0%} FPR", _threshold_for_fpr(g, s, fpr))

    print("\n== Operating points (anchored on precision over pooled impostors) ==")
    for prec in (0.80, 0.90, 0.95, 0.99):
        _print_op(f"precision >= {prec:.0%}", _threshold_for_precision(g, impostors, prec))


def _compare_all_table(raw: dict) -> None:
    """One-row-per-score-fn AUC summary so you can see at a glance which one
    discriminates best. AUC is invariant to monotone re-scalings, so when fns
    are monotone in z they should match — useful as a sanity check."""
    fmt = "  {fn:<20s}  {auc_o:>10s}  {auc_s:>10s}  {auc_a:>10s}"
    print("\n== Side-by-side AUC across score functions ==")
    print(fmt.format(fn="score_fn", auc_o="vs_other", auc_s="vs_synth", auc_a="vs_all"))
    print("  " + "─" * 50)
    # Calibrated fns are only comparable when the dump carries z_scale (3-elem rows).
    rows = raw.get("genuine", [])
    has_z_scale = bool(rows) and len(rows[0]) > 2
    fn_names = list(ALL_SCORE_FNS) + (list(CALIBRATED_SCORE_FNS) if has_z_scale else [])
    for fn_name in fn_names:
        scores = _scores_from_raw(raw, fn_name)
        g = scores["genuine"]
        o = scores["other"]
        s = scores["synthetic"]
        impostors = np.concatenate([o, s]) if (len(o) or len(s)) else np.array([])

        def fmtn(x: float) -> str:
            return f"{x:.4f}" if not np.isnan(x) else "  n/a"
        print(
            fmt.format(
                fn=fn_name,
                auc_o=fmtn(_auc(g, o)),
                auc_s=fmtn(_auc(g, s)),
                auc_a=fmtn(_auc(g, impostors)),
            )
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path, help="scores_final.json or a run directory")
    p.add_argument(
        "--score-fn",
        default=None,
        choices=sorted(SCORE_FNS),
        help=(
            "Compute scores from probe_raw.json under this score function. "
            "Requires probe_raw.json to be present alongside the dump."
        ),
    )
    p.add_argument(
        "--compare-all",
        action="store_true",
        help="Print AUC for every registered score function (requires probe_raw.json).",
    )
    args = p.parse_args()

    if not args.path.exists():
        print(f"Path not found: {args.path}", file=sys.stderr)
        sys.exit(1)

    raw = _load_raw(args.path)

    if args.compare_all:
        if raw is None:
            print(
                "--compare-all needs probe_raw.json (run a fresh training "
                "epoch with the updated trainer).",
                file=sys.stderr,
            )
            sys.exit(2)
        _compare_all_table(raw)
        print()
        return

    if args.score_fn:
        if raw is None:
            print(
                "--score-fn needs probe_raw.json (run a fresh training "
                "epoch with the updated trainer).",
                file=sys.stderr,
            )
            sys.exit(2)
        scores = _scores_from_raw(raw, args.score_fn)
        _report(scores, label=f"score_fn = {args.score_fn}")
        print()
        return

    # Default path: read canonical scores_final.json.
    scores = _load_scores(args.path)
    g = scores.get("genuine", np.array([]))
    o = scores.get("other", np.array([]))
    s = scores.get("synthetic", np.array([]))

    print("\n== Score distributions ==")
    print(_stats("genuine", g))
    print(_stats("other", o))
    print(_stats("synthetic", s))

    print("\n== Histograms (counts, 30 bins over [0, 1]) ==")
    print("genuine:")
    print(_histogram(g))
    print("other:")
    print(_histogram(o))
    print("synthetic:")
    print(_histogram(s))

    print("\n== AUC ==")
    print(f"  genuine vs other      {_auc(g, o):.4f}")
    print(f"  genuine vs synthetic  {_auc(g, s):.4f}")
    impostors = np.concatenate([o, s]) if (len(o) or len(s)) else np.array([])
    print(f"  genuine vs all        {_auc(g, impostors):.4f}")

    print("\n== Operating points (anchored on FPR over pooled impostors) ==")
    for fpr in (0.01, 0.05, 0.10):
        _print_op(f"all impostors @ {fpr:.0%} FPR", _threshold_for_fpr(g, impostors, fpr))

    if len(s):
        print("\n== Operating points (anchored on synthetic-only FPR) ==")
        for fpr in (0.01, 0.05, 0.10):
            _print_op(f"synthetic @ {fpr:.0%} FPR", _threshold_for_fpr(g, s, fpr))

    print("\n== Operating points (anchored on precision over pooled impostors) ==")
    for prec in (0.80, 0.90, 0.95, 0.99):
        _print_op(f"precision >= {prec:.0%}", _threshold_for_precision(g, impostors, prec))

    print()


if __name__ == "__main__":
    main()
