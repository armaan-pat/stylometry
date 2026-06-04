"""Pydantic v2 config schema and YAML loader.

configs/base.yaml holds defaults; experiment files only override what changes.
_deep_merge handles nested keys so you don't have to repeat sibling fields.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class LoRAConfig(BaseModel):
    """LoRA adapter settings. alpha/r scales the adapter learning rate."""
    model_config = ConfigDict(extra="forbid")

    r: int = 8
    alpha: int = 16
    target_modules: list[str] = Field(default_factory=lambda: ["query", "value"])
    dropout: float = 0.1


class EncoderConfig(BaseModel):
    """Text encoder config. pooling: "mean", "cls", or "luar_episode"."""
    model_config = ConfigDict(extra="forbid")

    name: str = "hf"
    model_name_or_path: str = "roberta-base"
    pooling: str = "mean"
    lora: LoRAConfig | None = None
    freeze_backbone: bool = True
    max_length: int = 512
    projection_dim: int | None = None  # trainable linear projection on top of backbone
    episode_k: int | None = None  # emails per LUAR episode; required when pooling='luar_episode'
    trust_remote_code: bool = False  # set True for models with custom HF code (e.g. LUAR-MUD)


class LossConfig(BaseModel):
    """Contrastive loss config. name: "supcon", "triplet", or "contrastive"."""
    model_config = ConfigDict(extra="forbid")

    name: str = "supcon"
    temperature: float = 0.1
    margin: float = 0.3
    mining: str = "batch_hard"


class HeadConfig(BaseModel):
    """Scoring head config."""
    model_config = ConfigDict(extra="forbid")

    name: str = "prototypical"
    distance: str = "cosine"
    shrinkage: str = "ledoit_wolf"
    # Production score function the deployed head uses (see scoring/score_functions.py
    # for the (cos, spread) registry, plus "mahalanobis"/"adaptive_k" handled in the head).
    score_fn: str = "linear_z3"
    # Extra score functions the CentroidProbe evaluates every epoch alongside
    # score_fn, for apples-to-apples comparison in W&B. The first metric set
    # (score_fn itself) keeps the un-prefixed metric names. See scoring ablation.
    eval_score_fns: list[str] = Field(default_factory=list)
    # k below which mahalanobis/adaptive_k modes fall back to cosine (Σ too noisy).
    mahalanobis_min_k: int = 5
    ridge: float = 1e-4


class PreprocessingConfig(BaseModel):
    """Email cleaning pipeline flags."""
    model_config = ConfigDict(extra="forbid")

    strip_quoted: bool = True
    strip_signatures: bool = True
    strip_boilerplate: bool = True
    entity_masking: bool = False
    fix_encoding: bool = True          # run ftfy to fix garbled unicode/encoding artifacts
    min_body_chars: int = 50           # drop emails shorter than this after cleaning
    min_body_words: int = 20           # drop emails with fewer than this many whitespace-split tokens
    min_alnum_ratio: float = 0.60      # drop emails where <60% of chars are alphanumeric or space
    max_body_chars: int = 4000         # truncate bodies longer than this


class AugmentationConfig(BaseModel):
    """LLM-generated synthetic hard-negative settings. synthetic_path=None disables augmentation."""
    model_config = ConfigDict(extra="forbid")

    synthetic_path: str | None = None
    n_syn_per_batch: int = 2


class DataConfig(BaseModel):
    """Dataset paths and sampling parameters."""
    model_config = ConfigDict(extra="forbid")

    dataset: str = "enron"
    data_dir: str = "data/raw/enron"
    processed_dir: str = "data/processed/enron"
    train_senders: int = 100
    min_emails_per_sender: int = 25
    emails_per_sender_k: int = 16
    val_split: float = 0.1
    test_split: float = 0.1
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)


class HardNegativeMiningConfig(BaseModel):
    """Offline hard negative mining with curriculum scheduling."""
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    warmup_epochs: int = 20       # epochs before any hard batches appear
    ramp_epochs: int = 30         # epochs to ramp hard batch fraction from 0 to max
    interval: int = 5             # re-mine every N epochs after warmup
    n_pairs: int = 20             # top-K most confusable sender pairs to track
    max_hard_fraction: float = 0.5  # maximum fraction of batches that are hard batches


class TrainingConfig(BaseModel):
    """Training loop hyperparameters and checkpointing settings."""
    model_config = ConfigDict(extra="forbid")

    epochs: int = 10
    batch_size: int = 64
    lr: float = 2e-5
    scheduler: str = "cosine"
    warmup_steps: int = 100
    grad_clip: float = 1.0
    mixed_precision: bool = True
    output_dir: str = "runs"          # root directory for all experiment outputs
    checkpoint_every_n: int = 1       # save a checkpoint every N epochs
    keep_last_n: int = 3              # keep only the N most recent epoch checkpoints (0 = keep all)
    save_best: bool = True            # maintain a checkpoint_best.pt tracking the monitored metric
    # Metric that drives checkpoint_best.pt selection AND early stopping. Any
    # key present in the per-epoch log payload works: "val/loss" (default,
    # back-compat), or an operational metric from the CentroidProbe such as
    # "auc/genuine_vs_synthetic", "pauc/genuine_vs_synthetic_5pct", or
    # "tpr_at_fpr/synthetic_1pct". Use monitor_mode to say which direction is better.
    monitor: str = "val/loss"
    monitor_mode: str = "min"         # "min" (lower is better, e.g. loss) or "max" (e.g. AUC)
    early_stopping_patience: int = 0  # epochs without monitor improvement before stopping; 0 disables
    early_stopping_min_delta: float = 0.0
    hard_negative_mining: HardNegativeMiningConfig = Field(
        default_factory=HardNegativeMiningConfig
    )


class WandbConfig(BaseModel):
    """W&B experiment tracking settings."""
    model_config = ConfigDict(extra="forbid")

    project: str = "email-fraud-detection"
    entity: str | None = None
    name: str | None = None
    tags: list[str] = Field(default_factory=list)
    notes: str = ""


class RunpodConfig(BaseModel):
    """RunPod GPU pod settings. Only used with --runpod in train.py."""
    model_config = ConfigDict(extra="forbid")

    gpu_type: str = "NVIDIA A100-SXM4-80GB"
    disk_gb: int = 50
    container_image: str = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"


class ExperimentConfig(BaseModel):
    """Root config. confidence_tiers maps k-ranges (e.g. "1-4") to tier labels."""

    model_config = ConfigDict(extra="forbid")

    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    head: HeadConfig = Field(default_factory=HeadConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    runpod: RunpodConfig | None = None
    confidence_tiers: dict[str, str] = Field(
        default_factory=lambda: {
            "1-4": "low",
            "5-9": "medium",
            "10-24": "high",
            "25+": "very_high",
        }
    )


_BASE_YAML = Path(__file__).parent.parent.parent / "configs" / "base.yaml"
_PROJECT_ROOT = _BASE_YAML.parent.parent


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base; nested dicts are merged, not replaced."""
    merged = base.copy()
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def load_config(path: str) -> ExperimentConfig:
    """Load base.yaml, deep-merge the experiment file on top, validate with Pydantic."""
    base_path = _BASE_YAML
    base_data: dict[str, Any] = {}
    if base_path.exists():
        with open(base_path) as fh:
            base_data = yaml.safe_load(fh) or {}

    with open(path) as fh:
        experiment_data: dict[str, Any] = yaml.safe_load(fh) or {}

    merged = _deep_merge(base_data, experiment_data)
    cfg = ExperimentConfig.model_validate(merged)

    # Resolve relative paths against project root so scripts work from any CWD
    # (e.g. RunPod where cwd may be /workspace, not the repo root).
    def _abs(p: str) -> str:
        pp = Path(p)
        return str(_PROJECT_ROOT / pp if not pp.is_absolute() else pp)

    cfg.data.data_dir = _abs(cfg.data.data_dir)
    cfg.data.processed_dir = _abs(cfg.data.processed_dir)
    return cfg
