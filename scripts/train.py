"""Training entry point.

Usage::

    # Standard run — output goes to runs/<config-stem>/<timestamp>/
    python scripts/train.py --config configs/experiments/roberta_lora_supcon.yaml

    # Explicit output dir
    python scripts/train.py --config configs/experiments/roberta_lora_supcon.yaml \\
        --output-dir runs/my_experiment

    # Resume from a checkpoint
    python scripts/train.py --config configs/experiments/roberta_lora_supcon.yaml \\
        --resume runs/roberta_lora_supcon/2026-04-29_15-30-00/checkpoint_last.pt

    # Force CPU (useful for debugging)
    python scripts/train.py --config configs/experiments/roberta_frozen.yaml \\
        --device cpu
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

# Anchor everything relative to the project root (parent of scripts/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add src/ to path so email_fraud is importable without installation
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")

# Import subpackages to trigger @register side-effects
import email_fraud.data.enron    # noqa: F401
import email_fraud.encoders      # noqa: F401
import email_fraud.heads         # noqa: F401
import email_fraud.losses        # noqa: F401
from email_fraud.config import load_config
from email_fraud.registry import list_components, resolve
from email_fraud.utils.logging import log_config, setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a contrastive encoder for email sender fingerprinting."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to experiment YAML (e.g. configs/experiments/roberta_lora_supcon.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Root directory for checkpoints and logs. "
            "Defaults to runs/<config-stem>/<YYYY-MM-DD_HH-MM-SS>/ "
            "under the project root."
        ),
    )
    parser.add_argument(
        "--resume",
        default=None,
        metavar="CHECKPOINT",
        help="Path to a .pt checkpoint file to resume training from.",
    )
    parser.add_argument("--device", default=None, help="Override torch device (cpu/cuda).")
    parser.add_argument("--runpod", action="store_true", help="Launch on RunPod.")
    return parser.parse_args()


def _maybe_register_labels(cfg, train_dataset) -> list[str] | None:
    """Per-index register labels for register-stratified sampling, or None.

    Returns None unless cfg.data.augmentation.register_stratified is set. Labels
    are computed from the *uncropped* texts (._texts is exposed by every dataset
    wrapper, including SyntheticAugmentedDataset) so they align with sender_ids,
    which is what the sampler indexes into. Synthetic ``__syn`` rows are labelled
    too; that is harmless — stratification only ever affects which of a sender's
    own emails fill its K slots.
    """
    if not cfg.data.augmentation.register_stratified:
        return None
    from email_fraud.data.register import detect_register

    texts = train_dataset._texts
    labels = [detect_register(t) for t in texts]
    counts = Counter(labels)
    logger.info(
        "Register-stratified sampling ON: %d emails — formal=%d casual=%d terse=%d",
        len(labels), counts.get("formal", 0), counts.get("casual", 0), counts.get("terse", 0),
    )
    return labels


def _resolve_output_dir(args: argparse.Namespace, config_path: Path) -> Path:
    """Return the fully-resolved output directory for this run."""
    if args.output_dir is not None:
        return Path(args.output_dir).resolve()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return (_PROJECT_ROOT / "runs" / config_path.stem / timestamp).resolve()


def main() -> None:
    args = parse_args()
    setup_logging()

    config_path = Path(args.config).resolve()
    cfg = load_config(str(config_path))
    log_config(cfg)

    # Default run name to the config file stem (e.g. "roberta_lora_triplet")
    # so W&B shows a meaningful name instead of an auto-generated one.
    if cfg.wandb.name is None:
        cfg.wandb.name = config_path.stem

    output_dir = _resolve_output_dir(args, config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy the experiment config into the output dir for reproducibility
    shutil.copy2(config_path, output_dir / "config.yaml")
    logger.info("Output directory: %s", output_dir)

    logger.info("Registered components: %s", list_components())
    EncoderClass = resolve("encoder", cfg.encoder.name)
    LossClass = resolve("loss", cfg.loss.name)
    HeadClass = resolve("head", cfg.head.name)
    logger.info(
        "Resolved  encoder=%s  loss=%s  head=%s",
        EncoderClass.__name__,
        LossClass.__name__,
        HeadClass.__name__,
    )

    pod_id: str | None = None
    if args.runpod and cfg.runpod is not None:
        from email_fraud.utils.runpod import launch_pod, wait_for_running

        pod_id = launch_pod(
            gpu_type=cfg.runpod.gpu_type,
            disk_gb=cfg.runpod.disk_gb,
            container_image=cfg.runpod.container_image,
        )
        wait_for_running(pod_id)

    try:
        _run_training(cfg, EncoderClass, LossClass, HeadClass, args, output_dir)
    finally:
        if pod_id is not None:
            from email_fraud.utils.runpod import terminate_pod

            terminate_pod(pod_id)


def _run_training(cfg, EncoderClass, LossClass, HeadClass, args, output_dir: Path) -> None:
    import torch
    from torch.utils.data import DataLoader

    from email_fraud.data.samplers import PKSampler
    from email_fraud.training.trainer import Trainer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    encoder = EncoderClass(cfg.encoder)
    loss_fn = LossClass(
        **{
            k: v
            for k, v in {
                "temperature": cfg.loss.temperature,
                "margin": cfg.loss.margin,
                "mining": cfg.loss.mining,
                "support_k_min": cfg.loss.support_k_min,
                "support_k_max": cfg.loss.support_k_max,
                "supcon_weight": cfg.loss.supcon_weight,
            }.items()
            if k in LossClass.__init__.__code__.co_varnames
        }
    )
    head = HeadClass(
        **{
            k: v
            for k, v in {
                "confidence_tiers": cfg.confidence_tiers,
                "score_fn": cfg.head.score_fn,
                "mahalanobis_min_k": cfg.head.mahalanobis_min_k,
                "ridge": cfg.head.ridge,
            }.items()
            if k in HeadClass.__init__.__code__.co_varnames
        }
    )

    DatasetClass = resolve("dataset", cfg.data.dataset)
    train_dataset = DatasetClass(cfg.data, split="train")
    val_dataset = DatasetClass(cfg.data, split="validation")

    p = cfg.training.batch_size // cfg.data.emails_per_sender_k
    if p < 2:
        raise ValueError(
            f"batch_size ({cfg.training.batch_size}) // emails_per_sender_k "
            f"({cfg.data.emails_per_sender_k}) = {p} — need at least P=2 senders per batch."
        )

    if cfg.data.augmentation.synthetic_path:
        from email_fraud.data.samplers import SyntheticBalancedSampler
        from email_fraud.data.synthetic import SyntheticAugmentedDataset

        n_syn = cfg.data.augmentation.n_syn_per_batch
        logger.info(
            "Augmenting train set with synthetic dataset at %s (n_syn=%d)",
            cfg.data.augmentation.synthetic_path,
            n_syn,
        )
        # llm_negatives_only=True (the default) drops any synthetic row stored
        # under a real sender_id — LLM text is only ever a hard negative, never a
        # positive — regardless of how the on-disk dataset was generated.
        train_dataset = SyntheticAugmentedDataset(
            train_dataset, cfg.data.augmentation.synthetic_path
        )
        register_labels = _maybe_register_labels(cfg, train_dataset)
        train_sampler = SyntheticBalancedSampler(
            sender_ids=train_dataset.sender_ids,
            p=p,
            k=cfg.data.emails_per_sender_k,
            n_syn=n_syn,
            seed=0,
            register_labels=register_labels,
        )
    else:
        register_labels = _maybe_register_labels(cfg, train_dataset)
        train_sampler = PKSampler(
            sender_ids=train_dataset.sender_ids,
            p=p,
            k=cfg.data.emails_per_sender_k,
            seed=0,
            register_labels=register_labels,
        )

    # Crop augmentation wraps LAST so synthetic emails are cropped too, while
    # the probe / hard-negative mining keep reading uncropped ._texts.
    if cfg.data.augmentation.crop_prob > 0:
        from email_fraud.data.augment import CropAugmentedDataset

        logger.info(
            "Crop augmentation enabled: prob=%.2f, span=%d-%d words",
            cfg.data.augmentation.crop_prob,
            cfg.data.augmentation.crop_min_words,
            cfg.data.augmentation.crop_max_words,
        )
        train_dataset = CropAugmentedDataset(
            train_dataset,
            crop_prob=cfg.data.augmentation.crop_prob,
            min_words=cfg.data.augmentation.crop_min_words,
            max_words=cfg.data.augmentation.crop_max_words,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=train_dataset.episode_collate,
        num_workers=4,
        pin_memory=(device != "cpu"),
    )

    # LUAR episode pooling requires P×K structure in every batch so that
    # consecutive episode_k rows always belong to the same sender.
    # Use PKSampler for val too; a plain DataLoader would interleave senders
    # and break the label-reduction stride (labels[::episode_k]).
    if cfg.encoder.pooling == "luar_episode":
        # Val/test splits are sender-disjoint and smaller, so fewer senders may
        # qualify with >= k emails.  Cap val_p to however many are eligible.
        k = cfg.data.emails_per_sender_k
        val_counts = Counter(val_dataset.sender_ids)
        val_eligible = sum(1 for c in val_counts.values() if c >= k)
        val_p = max(2, min(p, val_eligible))
        val_sampler = PKSampler(
            sender_ids=val_dataset.sender_ids,
            p=val_p,
            k=k,
            seed=1,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_sampler,
            collate_fn=val_dataset.episode_collate,
            num_workers=2,
            pin_memory=(device != "cpu"),
        )
    else:
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            collate_fn=val_dataset.episode_collate,
            num_workers=2,
        )

    centroid_probe = _build_centroid_probe(cfg, train_dataset, val_dataset, output_dir)

    trainer = Trainer(
        model=encoder,
        loss_fn=loss_fn,
        head=head,
        config=cfg.training,
        wandb_config=cfg.wandb,
        output_dir=output_dir,
        resume_from=args.resume,
        device=device,
        eval_config_path=Path(args.config).resolve(),
        eval_data_dir=cfg.data.processed_dir,
        centroid_probe=centroid_probe,
        preprocessing=cfg.data.preprocessing,
        train_dataset=train_dataset,
    )
    trainer.train(train_loader, val_loader)


def _build_centroid_probe(cfg, train_dataset, val_dataset, output_dir):
    """Construct a CentroidProbe for inference-style validation, if possible.

    Profiles a subset of training senders, draws impostor queries from the
    validation set (sender-disjoint from train), and pulls synthetic hard
    negatives from cfg.data.augmentation.synthetic_path if it is set.
    """
    from email_fraud.scoring.centroid_probe import CentroidProbe

    # Filter synthetic rows out of the profile pool — only real emails should
    # form the centroids.  Synthetic-augmented datasets append __syn senders
    # at the end; the rest are real.
    raw_texts = list(train_dataset._texts)
    raw_senders = list(train_dataset._sender_ids_list)
    train_texts = [t for t, s in zip(raw_texts, raw_senders) if not s.endswith("__syn")]
    train_senders = [s for s in raw_senders if not s.endswith("__syn")]
    other_texts = list(val_dataset._texts)
    other_senders = list(val_dataset._sender_ids_list)

    syn_texts: list[str] = []
    syn_sources: list[str] = []
    if cfg.data.augmentation.synthetic_path:
        from datasets import load_from_disk
        try:
            syn_ds = load_from_disk(cfg.data.augmentation.synthetic_path)
            syn_texts = list(syn_ds["text"])
            syn_sources = list(syn_ds["source_sender_id"])
        except Exception as e:
            logger.warning("Could not load synthetic dataset for probe: %s", e)

    return CentroidProbe(
        train_texts=train_texts,
        train_senders=train_senders,
        other_texts=other_texts,
        other_senders=other_senders,
        synthetic_texts=syn_texts or None,
        synthetic_source_senders=syn_sources or None,
        n_profile_senders=cfg.probe.n_profile_senders,
        n_enroll_per_sender=cfg.probe.n_enroll_per_sender,
        n_query_per_sender=cfg.probe.n_query_per_sender,
        n_other_queries=cfg.probe.n_other_queries,
        n_synthetic_queries=cfg.probe.n_synthetic_queries,
        confidence_tiers=cfg.confidence_tiers,
        score_fns=cfg.head.eval_score_fns or None,
        seed=cfg.probe.seed,
    )


if __name__ == "__main__":
    main()
