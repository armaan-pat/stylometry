"""Evaluate a trained checkpoint on the Enron test set (or any JSONL pairs).

Interactive checkpoint discovery + one-command evaluation.  Automatically
reads the config.yaml saved alongside each checkpoint so you never have to
specify the config manually.

Usage::

    # List all available checkpoints
    python scripts/eval_checkpoint.py --list

    # Evaluate the best checkpoint in a run directory (auto-reads saved config)
    python scripts/eval_checkpoint.py --run runs/v2_luar_lora/2026-05-02_21-59-01

    # Evaluate any specific checkpoint file
    python scripts/eval_checkpoint.py --checkpoint runs/v2_luar_loa/2026-05-02_21-59-01/checkpoint_best.pt

    # Evaluate on a specific split (train / validation / test)
    python scripts/eval_checkpoint.py --run runs/v2_luar_loa/... --split validation

    # Log results to W&B
    python scripts/eval_checkpoint.py --run runs/v2_luar_loa/... --wandb
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")

import email_fraud.encoders  # noqa: F401  — trigger @register
import email_fraud.heads     # noqa: F401
import email_fraud.losses    # noqa: F401
from email_fraud.config import load_config
from email_fraud.scoring.metrics import compute_verification_metrics
from email_fraud.utils.logging import setup_logging

logger = logging.getLogger(__name__)

_RUNS_DIR       = _PROJECT_ROOT / "runs"
_DEFAULT_DATA   = _PROJECT_ROOT / "data" / "processed" / "enron"


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def _find_checkpoints() -> list[dict]:
    """Return info dicts for every .pt file found under runs/."""
    results = []
    if not _RUNS_DIR.exists():
        return results
    for pt in sorted(_RUNS_DIR.rglob("*.pt")):
        run_dir = pt.parent
        config_path = run_dir / "config.yaml"
        results.append(
            dict(
                checkpoint=pt,
                run_dir=run_dir,
                experiment=run_dir.parent.name,
                timestamp=run_dir.name,
                config=config_path if config_path.exists() else None,
                is_best=pt.name == "checkpoint_best.pt",
                is_last=pt.name == "checkpoint_last.pt",
            )
        )
    return results


def list_checkpoints() -> None:
    """Print a formatted table of every available checkpoint."""
    checkpoints = _find_checkpoints()
    if not checkpoints:
        print("No checkpoints found under runs/")
        return

    col = dict(idx=4, exp=38, ts=22, file=28, cfg=7)
    header = (
        f"{'#':<{col['idx']}} {'Experiment':<{col['exp']}} "
        f"{'Timestamp':<{col['ts']}} {'File':<{col['file']}} {'Config?'}"
    )
    print(f"\n{header}")
    print("─" * (sum(col.values()) + 10))
    for i, ck in enumerate(checkpoints):
        cfg_flag = "yes" if ck["config"] else "MISSING"
        tag = " ← best" if ck["is_best"] else (" ← last" if ck["is_last"] else "")
        print(
            f"{i:<{col['idx']}} {ck['experiment']:<{col['exp']}} "
            f"{ck['timestamp']:<{col['ts']}} {ck['checkpoint'].name + tag:<{col['file']}} {cfg_flag}"
        )
    print(f"\n{len(checkpoints)} checkpoint(s) found.\n")
    print("Evaluate one with:")
    print("  python scripts/eval_checkpoint.py --checkpoint <path/to/checkpoint.pt>")
    print("  python scripts/eval_checkpoint.py --run <path/to/run_dir>")


# ---------------------------------------------------------------------------
# Checkpoint + config resolution
# ---------------------------------------------------------------------------

def _abs(p: str | Path) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (_PROJECT_ROOT / pp).resolve()


def _resolve(args: argparse.Namespace) -> tuple[Path, Path]:
    """Return (checkpoint_path, config_path) derived from CLI args."""
    if args.checkpoint:
        ck = _abs(args.checkpoint)
        if not ck.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ck}")
        cfg = _abs(args.config) if args.config else ck.parent / "config.yaml"
        if not cfg.exists():
            raise FileNotFoundError(
                f"No config.yaml found at {cfg}.\n"
                "Pass --config explicitly or ensure the checkpoint directory "
                "contains a config.yaml (training saves one automatically)."
            )
        return ck, cfg

    if args.run:
        run_dir = _abs(args.run)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        # Prefer best → last → highest-numbered epoch
        for name in ("checkpoint_best.pt", "checkpoint_last.pt"):
            candidate = run_dir / name
            if candidate.exists():
                ck = candidate
                break
        else:
            epoch_pts = sorted(run_dir.glob("checkpoint_epoch_*.pt"))
            if not epoch_pts:
                raise FileNotFoundError(f"No .pt checkpoint found in {run_dir}")
            ck = epoch_pts[-1]
        cfg = _abs(args.config) if args.config else run_dir / "config.yaml"
        if not cfg.exists():
            raise FileNotFoundError(f"No config.yaml found at {cfg}")
        return ck, cfg

    raise ValueError("Provide --checkpoint PATH, --run DIR, or --list.")


# ---------------------------------------------------------------------------
# Pair loading
# ---------------------------------------------------------------------------

def _load_pairs(data_path: Path, split: str) -> list[tuple[str, str, int]]:
    """Load verification pairs from a .jsonl file or a directory.

    Directory priority:
      1. <data_path>/<split>_pairs.jsonl   (e.g. test_pairs.jsonl)
      2. Any *.jsonl files under data_path
    """
    if data_path.is_file():
        jsonl_files = [data_path]
    elif data_path.is_dir():
        preferred = data_path / f"{split}_pairs.jsonl"
        if preferred.exists():
            jsonl_files = [preferred]
            logger.info("Using %s", preferred.name)
        else:
            jsonl_files = sorted(data_path.rglob("*.jsonl"))
            if not jsonl_files:
                raise FileNotFoundError(f"No JSONL files found under {data_path}")
    else:
        raise FileNotFoundError(f"Data path not found: {data_path}")

    pairs: list[tuple[str, str, int]] = []
    for path in jsonl_files:
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if "pair" in rec:
                    text1, text2 = rec["pair"]
                else:
                    text1 = rec.get("text1") or rec.get("text_a")
                    text2 = rec.get("text2") or rec.get("text_b")
                    if text1 is None or text2 is None:
                        raise ValueError(f"Cannot parse pair from: {rec}")
                label = int(bool(rec.get("same", rec.get("label", 0))))
                pairs.append((str(text1), str(text2), label))

    logger.info("Loaded %d pairs", len(pairs))
    return pairs


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_pairs(
    encoder,
    pairs: list[tuple[str, str, int]],
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Encode all texts and return per-pair cosine similarities mapped to [0, 1]."""
    flat_texts = [text for pair in pairs for text in pair[:2]]
    all_embs: list = []

    with torch.no_grad():
        for start in tqdm(
            range(0, len(flat_texts), batch_size),
            desc="Encoding",
            unit="batch",
            total=(len(flat_texts) + batch_size - 1) // batch_size,
        ):
            batch = flat_texts[start : start + batch_size]
            token_dict = encoder.tokenize(batch)
            token_dict = {k: v.to(device) for k, v in token_dict.items()}
            embs = encoder.encode(**token_dict)
            all_embs.append(embs.cpu())

    all_embs = torch.cat(all_embs, dim=0)
    scores: list[float] = []
    for idx in range(0, all_embs.size(0), 2):
        sim = F.cosine_similarity(
            all_embs[idx].unsqueeze(0),
            all_embs[idx + 1].unsqueeze(0),
        ).item()
        scores.append((sim + 1.0) / 2.0)  # map [-1, 1] → [0, 1]

    return np.array(scores, dtype=np.float64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a checkpoint on the Enron test set.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--list", action="store_true",
        help="List all available checkpoints and exit.",
    )
    mode.add_argument(
        "--checkpoint", metavar="PATH",
        help="Path to a specific .pt checkpoint file.",
    )
    mode.add_argument(
        "--run", metavar="DIR",
        help="Path to a run directory; uses checkpoint_best.pt automatically.",
    )
    p.add_argument(
        "--config", default=None,
        help="Override config path. Defaults to config.yaml in the checkpoint's directory.",
    )
    p.add_argument(
        "--data-dir", default=None, metavar="PATH",
        help=(
            "Evaluation data: a directory or a single .jsonl file. "
            f"Defaults to {_DEFAULT_DATA}."
        ),
    )
    p.add_argument(
        "--split", choices=["train", "validation", "test"], default="test",
        help="Which pairs file to use when --data-dir is a processed enron directory (default: test).",
    )
    p.add_argument("--device", default=None, help="Override torch device (cpu / cuda).")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--wandb", action="store_true", help="Log metrics to W&B.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()

    if args.list or (not args.checkpoint and not args.run):
        list_checkpoints()
        return

    checkpoint_path, config_path = _resolve(args)

    data_path = _abs(args.data_dir) if args.data_dir else _DEFAULT_DATA

    logger.info("Checkpoint : %s", checkpoint_path)
    logger.info("Config     : %s", config_path)
    logger.info("Data       : %s", data_path)

    cfg = load_config(str(config_path))

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device     : %s", device)

    from email_fraud.registry import resolve

    EncoderClass = resolve("encoder", cfg.encoder.name)
    encoder = EncoderClass(cfg.encoder)
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    encoder.load_state_dict(ckpt["model_state_dict"])
    encoder.eval()
    encoder.to(device)
    logger.info("Loaded checkpoint (epoch %s)", ckpt.get("epoch", "?"))

    pairs = _load_pairs(data_path, args.split)
    labels = np.array([lbl for _, _, lbl in pairs], dtype=np.int64)
    scores = _score_pairs(encoder, pairs, device, args.batch_size)
    metrics = compute_verification_metrics(labels, scores)

    # Display
    sep = "─" * 44
    print(f"\n{sep}")
    print(f"  Checkpoint : {checkpoint_path.name}")
    print(f"  Experiment : {checkpoint_path.parent.parent.name}")
    print(f"  Run        : {checkpoint_path.parent.name}")
    print(f"  Epoch      : {ckpt.get('epoch', '?')}")
    print(f"  Split      : {args.split}  ({len(pairs)} pairs)")
    print(sep)
    for key, value in metrics.items():
        print(f"  {key:<10} {value:.4f}")
    print(f"{sep}\n")

    if args.wandb:
        import wandb

        run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            tags=cfg.wandb.tags,
            notes=cfg.wandb.notes,
            config={
                "checkpoint": str(checkpoint_path),
                "config": str(config_path),
                "data": str(data_path),
                "split": args.split,
                "device": device,
            },
        )
        wandb.log({f"test/{k}": v for k, v in metrics.items()})
        wandb.summary.update({f"test/{k}": v for k, v in metrics.items()})
        run.finish()


if __name__ == "__main__":
    main()
