"""Probe a checkpoint's ability to distinguish genuine vs synthetic email.

This is a *diagnostic*, not the fraud pipeline. It answers one question:

    Does the encoder's embedding space separate human-written emails from
    LLM-generated ("synthetic") ones — independent of sender identity?

It pools every real email (label 0) against every synthetic email (label 1),
splits *senders* into a train/test bucket (sender-disjoint, keyed on
source_sender_id), and fits a classifier on top of the encoder:

  - frozen   : sklearn LogisticRegression on frozen embeddings ("what the
               representation already encodes")
  - finetune : a linear head trained with BCE, optionally with the encoder
               unfrozen ("what the model can be trained to encode")

Usage::

    # Frozen + finetune probe of the best checkpoint in a run dir
    python scripts/probe_authenticity.py --run runs/v6_luar_lora_syn/2026-05-02_...

    # Specific checkpoint, frozen probe only
    python scripts/probe_authenticity.py --checkpoint runs/.../checkpoint_best.pt --mode frozen

    # Point at a synthetic dataset explicitly (defaults to the config's
    # data.augmentation.synthetic_path, then data/synthetic/enron_synthetic)
    python scripts/probe_authenticity.py --run runs/... --synthetic-path data/synthetic/enron_synthetic

    # Fine-tune with the encoder unfrozen, log to W&B
    python scripts/probe_authenticity.py --run runs/... --mode finetune --unfreeze-encoder --wandb
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")

import email_fraud.encoders  # noqa: F401  — trigger @register
import email_fraud.heads     # noqa: F401
import email_fraud.losses    # noqa: F401
from email_fraud.config import load_config
from email_fraud.scoring.authenticity_probe import AuthenticityProbe
from email_fraud.utils.logging import setup_logging

logger = logging.getLogger(__name__)

_RUNS_DIR = _PROJECT_ROOT / "runs"
_DEFAULT_SYNTHETIC = _PROJECT_ROOT / "data" / "synthetic" / "enron_synthetic"


def _abs(p: str | Path) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (_PROJECT_ROOT / pp).resolve()


def _resolve_checkpoint(args: argparse.Namespace) -> tuple[Path, Path]:
    """Return (checkpoint_path, config_path) from --checkpoint or --run."""
    if args.checkpoint:
        ck = _abs(args.checkpoint)
        if not ck.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ck}")
        cfg = _abs(args.config) if args.config else ck.parent / "config.yaml"
    elif args.run:
        run_dir = _abs(args.run)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
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
    else:
        raise ValueError("Provide --checkpoint PATH or --run DIR.")

    if not cfg.exists():
        raise FileNotFoundError(
            f"No config.yaml found at {cfg}. Pass --config explicitly."
        )
    return ck, cfg


def _resolve_synthetic_path(args: argparse.Namespace, cfg) -> Path:
    """Synthetic dataset: --synthetic-path → config's augmentation path → default."""
    if args.synthetic_path:
        return _abs(args.synthetic_path)
    cfg_path = cfg.data.augmentation.synthetic_path
    if cfg_path:
        return _abs(cfg_path)
    return _DEFAULT_SYNTHETIC


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Probe genuine-vs-synthetic discrimination for a checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--checkpoint", metavar="PATH", help="Path to a .pt checkpoint.")
    mode.add_argument("--run", metavar="DIR", help="Run dir; uses checkpoint_best.pt.")
    p.add_argument("--config", default=None, help="Override config path.")
    p.add_argument(
        "--synthetic-path", default=None,
        help="Path to the synthetic Arrow dataset (text, source_sender_id).",
    )
    p.add_argument(
        "--split", choices=["train", "validation", "test"], default="train",
        help="Genuine-email split to draw from. Synthetic data is generated from "
             "train senders, so 'train' is the default (still sender-disjoint internally).",
    )
    p.add_argument(
        "--mode", choices=["frozen", "finetune", "both"], default="both",
        help="Which probe(s) to run (default: both).",
    )
    p.add_argument("--test-frac", type=float, default=0.3, help="Fraction of senders held out for test.")
    p.add_argument("--max-per-class", type=int, default=None, help="Cap examples per class per bucket.")
    p.add_argument("--no-balance", action="store_true", help="Do not downsample the majority (genuine) class.")
    p.add_argument("--C", type=float, default=1.0, help="Inverse L2 strength for the frozen LogisticRegression.")
    p.add_argument("--epochs", type=int, default=5, help="Fine-tune epochs.")
    p.add_argument("--lr", type=float, default=1e-3, help="Fine-tune head LR.")
    p.add_argument("--encoder-lr", type=float, default=2e-5, help="Fine-tune encoder LR (when --unfreeze-encoder).")
    p.add_argument("--unfreeze-encoder", action="store_true", help="Train encoder params in finetune mode.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default=None, help="Override torch device (cpu / cuda).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="Write metrics JSON to this path.")
    p.add_argument("--wandb", action="store_true", help="Log metrics to W&B.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()

    checkpoint_path, config_path = _resolve_checkpoint(args)
    cfg = load_config(str(config_path))
    synthetic_path = _resolve_synthetic_path(args, cfg)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Checkpoint : %s", checkpoint_path)
    logger.info("Config     : %s", config_path)
    logger.info("Synthetic  : %s", synthetic_path)
    logger.info("Device     : %s", device)

    if not synthetic_path.exists():
        raise FileNotFoundError(
            f"Synthetic dataset not found at {synthetic_path}. Generate it with "
            "scripts/generate_synthetic_emails.py or pass --synthetic-path."
        )

    # --- Load encoder + checkpoint ---
    from email_fraud.registry import resolve

    EncoderClass = resolve("encoder", cfg.encoder.name)
    encoder = EncoderClass(cfg.encoder)
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    encoder.load_state_dict(ckpt["model_state_dict"])
    encoder.to(device)
    logger.info("Loaded checkpoint (epoch %s)", ckpt.get("epoch", "?"))

    # --- Load genuine + synthetic text ---
    from datasets import load_from_disk

    from email_fraud.data.enron import EnronDataset

    genuine = EnronDataset(cfg.data, split=args.split)
    genuine_texts = list(genuine._texts)
    genuine_senders = list(genuine._sender_ids_list)

    syn_ds = load_from_disk(str(synthetic_path))
    if "source_sender_id" not in syn_ds.column_names:
        raise ValueError(
            f"Synthetic dataset at {synthetic_path} lacks 'source_sender_id' "
            f"(has {syn_ds.column_names}). Regenerate with the current generate script."
        )
    synthetic_texts = list(syn_ds["text"])
    synthetic_sources = list(syn_ds["source_sender_id"])

    probe = AuthenticityProbe(
        genuine_texts=genuine_texts,
        genuine_senders=genuine_senders,
        synthetic_texts=synthetic_texts,
        synthetic_source_senders=synthetic_sources,
        test_frac=args.test_frac,
        max_per_class=args.max_per_class,
        balance=not args.no_balance,
        seed=args.seed,
    )
    logger.info("Probe counts: %s", probe.counts)

    metrics: dict[str, float] = {}
    if args.mode in ("frozen", "both"):
        logger.info("Running frozen linear probe ...")
        metrics.update(probe.evaluate_frozen(encoder, device, args.batch_size, C=args.C))
    if args.mode in ("finetune", "both"):
        logger.info("Running fine-tune probe (unfreeze_encoder=%s) ...", args.unfreeze_encoder)
        metrics.update(probe.evaluate_finetune(
            encoder, device,
            epochs=args.epochs, lr=args.lr, encoder_lr=args.encoder_lr,
            batch_size=args.batch_size, unfreeze_encoder=args.unfreeze_encoder,
        ))

    # --- Display ---
    sep = "─" * 52
    print(f"\n{sep}")
    print(f"  Checkpoint : {checkpoint_path.name}")
    print(f"  Experiment : {checkpoint_path.parent.parent.name}")
    print(f"  Split      : {args.split}  (genuine vs synthetic, sender-disjoint)")
    print(f"  Counts     : {probe.counts}")
    print(sep)
    for key in sorted(metrics):
        print(f"  {key:<34} {metrics[key]:.4f}")
    print(f"{sep}\n")

    if args.out:
        out_path = _abs(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as fh:
            json.dump(metrics, fh, indent=2)
        logger.info("Wrote metrics to %s", out_path)

    if args.wandb:
        import wandb

        run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            tags=[*cfg.wandb.tags, "authenticity-probe"],
            notes=cfg.wandb.notes,
            config={
                "checkpoint": str(checkpoint_path),
                "split": args.split,
                "mode": args.mode,
                "test_frac": args.test_frac,
                "unfreeze_encoder": args.unfreeze_encoder,
            },
        )
        wandb.log(metrics)
        wandb.summary.update(metrics)
        run.finish()


if __name__ == "__main__":
    main()
