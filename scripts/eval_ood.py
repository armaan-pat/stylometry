"""Evaluate a checkpoint on a sliced OOD pair-set and report metrics per slice.

Pairs come from scripts/build_ood_eval.py (each line carries a ``slice`` field).
This script encodes every pair once, then computes the full verification metric
suite (AUC, pAUC, TPR@FPR, EER, c@1, F0.5u) for EACH slice plus an overall row,
so you can see exactly which OOD axis the model is weak on — short emails, an
unseen impersonator LLM, register shifts, or a new domain.

Usage:
    # Evaluate the best checkpoint in a run dir on an OOD pair-set
    python scripts/eval_ood.py --run runs/v9_episodic_shortmail/2026-... \\
        --pairs data/ood/enron_ood_pairs.jsonl

    # Fold in an external-domain pair-set (tagged slice "domain:pan")
    python scripts/eval_ood.py --run runs/... --pairs data/ood/enron_ood_pairs.jsonl \\
        --extra-pairs domain:pan=data/ood/pan_test_pairs.jsonl

    # Specific checkpoint + JSON dump + W&B
    python scripts/eval_ood.py --checkpoint runs/.../checkpoint_best.pt \\
        --pairs data/ood/enron_ood_pairs.jsonl --json results/ood/v9.json --wandb
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")

import email_fraud.encoders  # noqa: F401  — trigger @register
import email_fraud.heads     # noqa: F401
import email_fraud.losses    # noqa: F401
from email_fraud.config import load_config
from email_fraud.scoring.metrics import compute_verification_metrics
from email_fraud.scoring.pairwise import load_encoder, score_pairs
from email_fraud.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def _abs(p: str | Path) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (_PROJECT_ROOT / pp).resolve()


def _resolve_checkpoint(args) -> tuple[Path, Path]:
    """Return (checkpoint_path, config_path) from --checkpoint or --run."""
    if args.checkpoint:
        ck = _abs(args.checkpoint)
        if not ck.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ck}")
        cfg = _abs(args.config) if args.config else ck.parent / "config.yaml"
    elif args.run:
        run_dir = _abs(args.run)
        for name in ("checkpoint_best.pt", "checkpoint_last.pt"):
            ck = run_dir / name
            if ck.exists():
                break
        else:
            epochs = sorted(run_dir.glob("checkpoint_epoch_*.pt"))
            if not epochs:
                raise FileNotFoundError(f"No checkpoint found in {run_dir}")
            ck = epochs[-1]
        cfg = _abs(args.config) if args.config else run_dir / "config.yaml"
    else:
        raise ValueError("Provide --run DIR or --checkpoint PATH.")
    if not cfg.exists():
        raise FileNotFoundError(f"No config.yaml at {cfg}; pass --config.")
    return ck, cfg


def _load_sliced_pairs(path: Path, default_slice: str | None = None):
    """Load {pair, same, slice} lines → (texts_a, texts_b, labels, slices)."""
    a: list[str] = []
    b: list[str] = []
    labels: list[int] = []
    slices: list[str] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "pair" in rec:
                t1, t2 = rec["pair"]
            else:
                t1 = rec.get("text1") or rec.get("text_a")
                t2 = rec.get("text2") or rec.get("text_b")
            a.append(str(t1))
            b.append(str(t2))
            labels.append(int(bool(rec.get("same", rec.get("label", 0)))))
            slices.append(rec.get("slice", default_slice or "overall"))
    return a, b, labels, slices


# Headline metrics surfaced as summary scalars + per-axis aggregates. These are
# the numbers you compare across runs in the W&B table; the rest stay per-slice.
_HEADLINE = ("AUC", "pAUC@5%", "TPR@FPR=1%", "EER")


def _axis_of(slice_name: str) -> str:
    """Axis a slice belongs to: the part before the first ':' (len, gen, ...)."""
    return slice_name.split(":", 1)[0] if ":" in slice_name else slice_name


def _axis_aggregates(rows: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """Mean of each headline metric across the slices of each axis.

    One robust number per axis (mean pAUC@5% over all len:* slices, all gen:*
    slices, etc.) — the most useful cross-run comparison signal, and it makes a
    new impersonator model or domain show up as a single movement instead of
    being buried in many per-slice keys.
    """
    by_axis: dict[str, list[dict[str, float]]] = defaultdict(list)
    for sl, r in rows.items():
        if sl == "overall":
            continue
        by_axis[_axis_of(sl)].append(r)
    agg: dict[str, dict[str, float]] = {}
    for axis, slice_rows in by_axis.items():
        agg[axis] = {
            "n_slices": float(len(slice_rows)),
            **{f"{m}_mean": float(np.mean([sr[m] for sr in slice_rows]))
               for m in _HEADLINE},
        }
    return agg


def _metrics_table(rows: dict[str, dict[str, float]]) -> str:
    """Render a slice × metric table sorted by slice name, overall last."""
    metric_keys = ["AUC", "pAUC@5%", "TPR@FPR=1%", "TPR@FPR=5%", "EER", "c@1"]
    header = f"{'slice':<30} {'n':>6} " + " ".join(f"{m:>11}" for m in metric_keys)
    lines = [header, "─" * len(header)]
    ordered = sorted(k for k in rows if k != "overall") + (
        ["overall"] if "overall" in rows else []
    )
    for sl in ordered:
        r = rows[sl]
        cells = " ".join(
            f"{r[m]:>11.4f}" if m in r and r[m] == r[m] else f"{'—':>11}"
            for m in metric_keys
        )
        lines.append(f"{sl:<30} {int(r['n']):>6} {cells}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-slice OOD evaluation of a checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", metavar="DIR", help="Run dir; uses checkpoint_best.pt.")
    g.add_argument("--checkpoint", metavar="PATH", help="Specific .pt checkpoint.")
    p.add_argument("--config", default=None, help="Override config.yaml path.")
    p.add_argument("--pairs", required=True, help="Sliced OOD pair JSONL (from build_ood_eval.py).")
    p.add_argument("--extra-pairs", action="append", default=[], metavar="SLICE=PATH",
                   help="Additional pair JSONL folded in under a fixed slice name, e.g. "
                        "domain:pan=data/ood/pan_pairs.jsonl. Repeatable.")
    p.add_argument("--min-slice-pairs", type=int, default=20,
                   help="Skip metrics for slices with fewer pairs than this (default: 20).")
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--json", default=None, help="Write per-slice metrics to this JSON path.")
    p.add_argument("--wandb", action="store_true",
                   help="Log per-slice + per-axis OOD metrics to W&B (own run by default).")
    p.add_argument("--wandb-run-id", default=None,
                   help="Attach OOD metrics to an EXISTING W&B run id (e.g. the training run "
                        "for this checkpoint) instead of starting a new run.")
    p.add_argument("--wandb-name", default=None, help="Name for the new W&B run (if not attaching).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    import torch

    ck, cfg_path = _resolve_checkpoint(args)
    cfg = load_config(str(cfg_path))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Checkpoint: %s", ck)
    logger.info("Config    : %s", cfg_path)
    logger.info("Device    : %s", device)

    encoder, ckpt = load_encoder(ck, cfg, device)
    logger.info("Loaded checkpoint (epoch %s)", ckpt.get("epoch", "?"))

    # Gather all pairs (main file + extras) into one flat list, scored together.
    a, b, labels, slices = _load_sliced_pairs(_abs(args.pairs))
    for spec in args.extra_pairs:
        if "=" not in spec:
            raise ValueError(f"--extra-pairs must be SLICE=PATH, got '{spec}'.")
        slice_name, path = spec.split("=", 1)
        ea, eb, el, _ = _load_sliced_pairs(_abs(path), default_slice=slice_name)
        a += ea; b += eb; labels += el; slices += [slice_name] * len(ea)

    logger.info("Scoring %d pairs across %d slices …", len(a), len(set(slices)))
    scores = score_pairs(encoder, list(zip(a, b)), device, args.batch_size)
    labels_arr = np.array(labels, dtype=np.int64)

    # Group indices by slice; add an 'overall' group over everything.
    groups: dict[str, list[int]] = defaultdict(list)
    for i, sl in enumerate(slices):
        groups[sl].append(i)
    groups["overall"] = list(range(len(a)))

    rows: dict[str, dict[str, float]] = {}
    for sl, idxs in groups.items():
        y = labels_arr[idxs]
        s = scores[idxs]
        if len(idxs) < args.min_slice_pairs:
            logger.warning("slice %s: only %d pairs (< %d); skipping.",
                           sl, len(idxs), args.min_slice_pairs)
            continue
        if y.min() == y.max():
            logger.warning("slice %s: single-class (%d pairs, all same=%d); "
                           "AUC undefined, skipping.", sl, len(idxs), int(y[0]))
            continue
        m = compute_verification_metrics(y, s)
        m["n"] = float(len(idxs))
        m["n_same"] = float(int(y.sum()))
        rows[sl] = m

    print("\n" + _metrics_table(rows) + "\n")

    # Highlight the weakest slices by pAUC@5% (operational metric).
    ranked = sorted(
        ((sl, r["pAUC@5%"]) for sl, r in rows.items() if sl != "overall"),
        key=lambda kv: kv[1],
    )
    if ranked:
        print("Weakest slices (pAUC@5%, lower = worse):")
        for sl, v in ranked[:5]:
            print(f"  {sl:<30} {v:.4f}")
        print()

    if args.json:
        out = _abs(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2))
        print(f"Wrote per-slice metrics → {out}")

    axis_agg = _axis_aggregates(rows)
    if axis_agg:
        print("Per-axis mean pAUC@5% (one number per OOD axis):")
        for axis in sorted(axis_agg):
            print(f"  {axis:<12} {axis_agg[axis]['pAUC@5%_mean']:.4f} "
                  f"(over {int(axis_agg[axis]['n_slices'])} slices)")
        print()

    if args.wandb:
        _log_wandb(cfg, args, ck, ckpt, rows, axis_agg, ranked)


def _log_wandb(cfg, args, ck, ckpt, rows, axis_agg, ranked) -> None:
    """Log OOD metrics to W&B as summary scalars + per-axis aggregates + a table.

    Opens its OWN run by default (tagged ``ood-eval``), so these metrics are NOT
    subject to the trainer's `_WANDB_KEEP` allowlist — the full per-slice detail
    is reported. Pass --wandb-run-id to instead ATTACH the OOD metrics to an
    existing run (e.g. the training run that produced this checkpoint) so they
    land in that run's summary alongside the in-distribution numbers.
    """
    import wandb

    resume = "allow" if args.wandb_run_id else None
    run = wandb.init(
        project=cfg.wandb.project, entity=cfg.wandb.entity,
        id=args.wandb_run_id, resume=resume,
        name=args.wandb_name, tags=[*cfg.wandb.tags, "ood-eval"],
        config={
            "checkpoint": str(ck), "epoch": ckpt.get("epoch", None),
            "pairs": str(args.pairs),
            "extra_pairs": [s.split("=", 1)[0] for s in args.extra_pairs],
            "n_slices": len([s for s in rows if s != "overall"]),
        },
    )

    summary: dict[str, float] = {}
    # Per-slice headline scalars (comparable across runs).
    for sl, r in rows.items():
        for k, v in r.items():
            summary[f"ood/slice/{sl}/{k}"] = v
    # Per-axis aggregate means (the cross-run comparison signal).
    for axis, agg in axis_agg.items():
        for k, v in agg.items():
            summary[f"ood/axis/{axis}/{k}"] = v
    # Headline overall scalars promoted to short keys for quick filtering.
    if "overall" in rows:
        for m in _HEADLINE:
            summary[f"ood/overall/{m}"] = rows["overall"][m]
    # Weakest slice (worst operational pAUC@5%) — what to fix next.
    if ranked:
        worst_sl, worst_v = ranked[0]
        summary["ood/weakest_slice"] = worst_sl
        summary["ood/weakest_pAUC@5%"] = worst_v

    wandb.summary.update(summary)

    tbl = wandb.Table(
        columns=["slice", "axis", "n", "n_same", "AUC", "pAUC@5%", "TPR@FPR=1%", "EER", "c@1"],
        data=[[sl, _axis_of(sl), int(r["n"]), int(r.get("n_same", 0)),
               r["AUC"], r["pAUC@5%"], r["TPR@FPR=1%"], r["EER"], r["c@1"]]
              for sl, r in rows.items()],
    )
    wandb.log({"ood/by_slice": tbl})
    print(f"[wandb] logged {len(summary)} OOD scalars + per-slice table to run "
          f"{run.name} (tags: ood-eval)"
          + (f" — attached to existing run {args.wandb_run_id}" if args.wandb_run_id else ""))
    run.finish()


if __name__ == "__main__":
    main()
