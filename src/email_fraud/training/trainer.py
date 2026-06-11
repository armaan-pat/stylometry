"""Contrastive training loop with checkpointing and W&B logging.

Checkpoint layout under output_dir/:
  checkpoint_epoch_NNN.pt, checkpoint_last.pt, checkpoint_best.pt, config.yaml

Pass resume_from=<path> to __init__ to restart from epoch+1 with full state.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from email_fraud.config import PreprocessingConfig, TrainingConfig, WandbConfig
from email_fraud.encoders.base import BaseEncoder
from email_fraud.heads.base import BaseHead
from email_fraud.losses.base import BaseLoss

logger = logging.getLogger(__name__)

# Keys that are forwarded to W&B. Everything else (raw score means, loose
# threshold bands, per-score-fn duplicates, sampler internals, probe counts)
# is computed locally but not logged — it's noisy and not SLA-relevant.
_WANDB_KEEP: frozenset[str] = frozenset({
    # Training health
    "epoch", "train/loss", "train/lr", "val/loss",
    # Monitor tracking
    "monitor/value", "monitor/best",
    # Embedding-space discrimination
    "embedding/pair_auroc", "embedding/knn_accuracy",
    # Centroid headline AUROCs
    "auc/genuine_vs_other", "auc/genuine_vs_synthetic", "auc/genuine_vs_all",
    # pAUC in the low-FPR region (SLA = operate below 5% FPR)
    "pauc/genuine_vs_synthetic_5pct", "pauc/genuine_vs_other_5pct",
    "pauc/min_other_synthetic_5pct",  # composite anti-Goodhart monitor
    # TPR at tight FPR anchors
    "tpr_at_fpr/synthetic_1pct", "tpr_at_fpr/all_1pct",
    # FPR-anchored operating point at 1% (tightest SLA target)
    "op/all/fpr_0.01/recall", "op/all/fpr_0.01/precision", "op/all/fpr_0.01/threshold",
    "op/synthetic/fpr_0.01/recall", "op/synthetic/fpr_0.01/precision",
    "op/synthetic/fpr_0.01/threshold",
    # Score geometry (gaps only; raw means are less informative)
    "score/gap_other", "score/gap_synthetic", "score/synthetic_harder_than_other",
    # High-confidence threshold band (0.95 is the deployment operating region)
    "threshold_0.95/recall", "threshold_0.95/fpr_synthetic", "threshold_0.95/precision",
    # Selective-classifier coverage at tight accuracy
    "coverage/at_acc_0.95",
    # Inline PAN test-set verification
    "test/auc", "test/eer",
    # FPR comparison reference — 5% and 10% operating points
    # (kept alongside the 1% SLA anchor for trend comparison, not operational targets)
    "pauc/genuine_vs_synthetic_10pct",
    "pauc/genuine_vs_all_5pct", "pauc/genuine_vs_all_10pct",
    "tpr_at_fpr/synthetic_5pct", "tpr_at_fpr/all_5pct",
    "op/all/fpr_0.05/recall", "op/all/fpr_0.05/precision", "op/all/fpr_0.05/threshold",
    "op/all/fpr_0.10/recall", "op/all/fpr_0.10/precision", "op/all/fpr_0.10/threshold",
    "op/synthetic/fpr_0.05/recall", "op/synthetic/fpr_0.05/precision", "op/synthetic/fpr_0.05/threshold",
    "op/synthetic/fpr_0.10/recall", "op/synthetic/fpr_0.10/precision", "op/synthetic/fpr_0.10/threshold",
})


def _filter_wandb_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return only the SLA-relevant subset of a W&B log payload."""
    return {k: v for k, v in payload.items() if k in _WANDB_KEEP}


def _fmt(metrics: dict[str, float], key: str, fmt: str = "{:.3f}", missing: str = "  —  ") -> str:
    v = metrics.get(key)
    return fmt.format(v) if v is not None else missing


def _format_epoch_summary(
    epoch: int,
    total_epochs: int,
    train_loss: float,
    val_metrics: dict[str, float],
    centroid_metrics: dict[str, float],
    pan_metrics: dict[str, float],
) -> str:
    width = len(str(total_epochs))
    header = f"Epoch {epoch:>{width}}/{total_epochs}"

    lines = [
        f"{header}  loss train={train_loss:.4f} val={_fmt(val_metrics, 'val/loss', '{:.4f}')}"
        f"  │  embed  pair_auc={_fmt(val_metrics, 'embedding/pair_auroc')}"
        f"  knn_f1={_fmt(val_metrics, 'embedding/knn_macro_f1')}"
        f"  knn_acc={_fmt(val_metrics, 'embedding/knn_accuracy')}"
    ]

    if centroid_metrics:
        lines.append(
            " " * len(header) + "  "
            f"centroid auc    vs_other={_fmt(centroid_metrics, 'auc/genuine_vs_other')}"
            f"  vs_syn={_fmt(centroid_metrics, 'auc/genuine_vs_synthetic')}"
            f"  vs_all={_fmt(centroid_metrics, 'auc/genuine_vs_all')}"
        )
        lines.append(
            " " * len(header) + "  "
            f"          gaps   other={_fmt(centroid_metrics, 'score/gap_other', '{:+.3f}')}"
            f"  syn={_fmt(centroid_metrics, 'score/gap_synthetic', '{:+.3f}')}"
            f"  harder={_fmt(centroid_metrics, 'score/synthetic_harder_than_other', '{:+.3f}')}"
        )
        lines.append(
            " " * len(header) + "  "
            f"@0.95          prec={_fmt(centroid_metrics, 'threshold_0.95/precision')}"
            f"  rec={_fmt(centroid_metrics, 'threshold_0.95/recall')}"
            f"  fpr_syn={_fmt(centroid_metrics, 'threshold_0.95/fpr_synthetic')}"
            f"  cov@acc={_fmt(centroid_metrics, 'coverage/at_acc_0.95')}"
        )

    if pan_metrics:
        lines.append(
            " " * len(header) + "  "
            f"test (PAN)     auc={_fmt(pan_metrics, 'auc')}"
            f"  eer={_fmt(pan_metrics, 'eer')}"
            f"  f1={_fmt(pan_metrics, 'f1')}"
        )

    return "\n".join(lines)


class Trainer:
    """Contrastive training loop with checkpointing, resume, and W&B logging."""

    def __init__(
        self,
        model: BaseEncoder,
        loss_fn: BaseLoss,
        head: BaseHead,
        config: TrainingConfig,
        wandb_config: WandbConfig,
        output_dir: Path | str,
        resume_from: Path | str | None = None,
        device: str | None = None,
        eval_config_path: str | Path | None = None,
        eval_data_dir: str | None = None,
        centroid_probe: Any = None,
        preprocessing: PreprocessingConfig | None = None,
        train_dataset: Any = None,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.head = head
        self.config = config
        self.wandb_config = wandb_config
        self.output_dir = Path(output_dir)
        self.eval_config_path = Path(eval_config_path) if eval_config_path is not None else None
        self.eval_data_dir = Path(eval_data_dir) if eval_data_dir is not None else None
        self.centroid_probe = centroid_probe
        self.preprocessing = preprocessing
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Extract real-sender texts/ids for hard negative mining (strip synthetic).
        self._train_texts: list[str] | None = None
        self._train_sender_ids: list[str] | None = None
        if train_dataset is not None and hasattr(train_dataset, "_texts"):
            raw_senders = list(train_dataset._sender_ids_list)
            self._train_texts = [
                t for t, s in zip(train_dataset._texts, raw_senders)
                if not s.endswith("__syn")
            ]
            self._train_sender_ids = [s for s in raw_senders if not s.endswith("__syn")]
        self._hard_pairs: list[tuple[str, str]] = []

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model.to(self.device)

        # Cache episode_k so the per-batch loop avoids repeated attribute lookups.
        self._episode_k: int | None = getattr(model, "episode_k", None)
        # Episodic losses need raw sender-id strings to spot __syn hard negatives.
        self._loss_wants_sender_ids: bool = bool(
            getattr(loss_fn, "requires_sender_ids", False)
        )

        trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
        if not trainable_params:
            raise ValueError(
                "No trainable parameters found in the encoder. "
                "Set freeze_backbone=False, add a LoRA config, or set projection_dim."
            )
        self.optimizer = torch.optim.AdamW(trainable_params, lr=config.lr)

        self.scaler: torch.amp.GradScaler | None = (
            torch.amp.GradScaler()
            if config.mixed_precision and self.device != "cpu"
            else None
        )

        self._start_epoch: int = 1
        # Best-checkpoint selection + early stopping track config.monitor in the
        # direction given by config.monitor_mode ("min"/"max"). Default is
        # val/loss/min for back-compat, but operational runs should monitor a
        # low-FPR CentroidProbe metric (the loss is a known non-monotonic proxy).
        self._monitor: str = config.monitor
        self._monitor_mode: str = config.monitor_mode
        if self._monitor_mode not in {"min", "max"}:
            raise ValueError(
                f"monitor_mode must be 'min' or 'max', got {self._monitor_mode!r}"
            )
        self._best_monitor: float = (
            float("inf") if self._monitor_mode == "min" else float("-inf")
        )
        self._epochs_since_improvement: int = 0

        if resume_from is not None:
            self._load_checkpoint(Path(resume_from))

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> None:
        """Run the full training loop from _start_epoch to config.epochs."""
        import wandb

        run = wandb.init(
            project=self.wandb_config.project,
            entity=self.wandb_config.entity,
            name=self.wandb_config.name,
            tags=self.wandb_config.tags,
            notes=self.wandb_config.notes,
            dir=str(self.output_dir),
            config={
                "epochs": self.config.epochs,
                "lr": self.config.lr,
                "batch_size": self.config.batch_size,
                "scheduler": self.config.scheduler,
                "warmup_steps": self.config.warmup_steps,
                "mixed_precision": self.config.mixed_precision,
                "output_dir": str(self.output_dir),
            },
            resume="allow",
        )

        scheduler = self._build_scheduler(len(train_loader))

        try:
            for epoch in range(self._start_epoch, self.config.epochs + 1):
                self._maybe_update_hard_negatives(epoch, train_loader)
                train_loss = self._train_epoch(train_loader, scheduler)
                val_metrics = self._validate(val_loader)
                val_loss = val_metrics.get("val/loss", float("inf"))
                current_lr = self.optimizer.param_groups[0]["lr"]

                centroid_metrics: dict[str, float] = {}
                if self.centroid_probe is not None:
                    try:
                        centroid_metrics = self.centroid_probe.evaluate(
                            self.model, self.device
                        )
                    except Exception as e:
                        logger.warning("CentroidProbe.evaluate failed: %s", e)

                # PAN verification metrics every 5 epochs, logged into the same run.
                pan_metrics: dict[str, float] = {}
                if epoch % 5 == 0 and self.eval_data_dir is not None:
                    try:
                        pan_metrics = self._inline_pan_eval()
                    except Exception as e:
                        logger.warning("Inline PAN eval failed: %s", e)

                # SyntheticBalancedSampler exposes pop_epoch_stats; plain PKSampler doesn't.
                sampler_stats: dict[str, float] = {}
                pop_fn = getattr(
                    getattr(train_loader, "batch_sampler", None),
                    "pop_epoch_stats",
                    None,
                )
                if callable(pop_fn):
                    sampler_stats = pop_fn()

                log_payload = {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/lr": current_lr,
                    **val_metrics,
                    **centroid_metrics,
                    **sampler_stats,
                    **{f"test/{k}": v for k, v in pan_metrics.items()},
                }

                # Pull the monitored metric from the full payload. It may live
                # in val_metrics (val/loss), centroid_metrics (auc/*, pauc/*,
                # tpr_at_fpr/*), or the test/* PAN block. If it's absent this
                # epoch (e.g. PAN metric on a non-multiple-of-5 epoch, or the
                # probe raised), we can't judge improvement — leave best/patience
                # untouched rather than guessing.
                monitor_val = log_payload.get(self._monitor)
                log_payload["monitor/value"] = (
                    float(monitor_val) if monitor_val is not None else float("nan")
                )
                log_payload["monitor/best"] = self._best_monitor
                wandb.log(_filter_wandb_payload(log_payload))
                logger.info(
                    "%s",
                    _format_epoch_summary(
                        epoch,
                        self.config.epochs,
                        train_loss,
                        val_metrics,
                        centroid_metrics,
                        pan_metrics,
                    ),
                )

                if epoch % self.config.checkpoint_every_n == 0:
                    self._save_epoch_checkpoint(epoch, val_loss)
                self._save_last_checkpoint(epoch, val_loss)

                improved = self._is_improvement(monitor_val)
                if improved:
                    self._epochs_since_improvement = 0
                    self._best_monitor = float(monitor_val)
                    if self.config.save_best:
                        self._save_best_checkpoint(epoch, val_loss)
                        logger.info(
                            "New best %s=%.4f (mode=%s) at epoch %d → checkpoint_best.pt",
                            self._monitor, self._best_monitor, self._monitor_mode, epoch,
                        )
                elif monitor_val is not None:
                    # Only count a real (present-but-not-better) epoch against patience.
                    self._epochs_since_improvement += 1
                if self.config.keep_last_n > 0:
                    self._prune_old_checkpoints(epoch)

                if (
                    self.config.early_stopping_patience > 0
                    and self._epochs_since_improvement
                    >= self.config.early_stopping_patience
                ):
                    logger.info(
                        "Early stopping at epoch %d: no %s improvement "
                        "for %d epochs (best=%.4f).",
                        epoch,
                        self._monitor,
                        self._epochs_since_improvement,
                        self._best_monitor,
                    )
                    wandb.log({"early_stopped_at_epoch": epoch})
                    break

        finally:
            wandb.finish()

    def _is_improvement(self, monitor_val: float | None) -> bool:
        """True iff monitor_val beats the running best by at least min_delta.

        Returns False when the metric is absent this epoch (can't judge) so
        neither the best checkpoint nor the early-stopping counter moves on a
        blind epoch.
        """
        if monitor_val is None:
            return False
        delta = self.config.early_stopping_min_delta
        if self._monitor_mode == "max":
            return monitor_val > self._best_monitor + delta
        return monitor_val < self._best_monitor - delta

    def _save_epoch_checkpoint(self, epoch: int, val_loss: float) -> None:
        path = self.output_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        torch.save(self._build_payload(epoch, val_loss), path)
        logger.debug("Saved epoch checkpoint: %s", path)

    def _save_last_checkpoint(self, epoch: int, val_loss: float) -> None:
        path = self.output_dir / "checkpoint_last.pt"
        torch.save(self._build_payload(epoch, val_loss), path)

    def _save_best_checkpoint(self, epoch: int, val_loss: float) -> None:
        # The caller logs the monitored metric; logging val_loss here as
        # "new best" was misleading whenever monitor != val/loss.
        path = self.output_dir / "checkpoint_best.pt"
        torch.save(self._build_payload(epoch, val_loss), path)

    def _build_payload(self, epoch: int, val_loss: float) -> dict:
        return {
            "epoch": epoch,
            "val_loss": val_loss,
            "monitor": self._monitor,
            "monitor_mode": self._monitor_mode,
            "best_monitor": self._best_monitor,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self._scheduler_state,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
        }

    def _load_checkpoint(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        logger.info("Resuming from checkpoint: %s", path)
        payload = torch.load(path, map_location=self.device)
        self.model.load_state_dict(payload["model_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        if payload.get("scaler_state_dict") and self.scaler is not None:
            self.scaler.load_state_dict(payload["scaler_state_dict"])
        # Restore the running best. New checkpoints store "best_monitor"; older
        # ones only have "best_val_loss" — fall back to it only when the current
        # monitor is still val/loss, otherwise start fresh in the right direction.
        if "best_monitor" in payload:
            self._best_monitor = payload["best_monitor"]
        elif self._monitor == "val/loss" and "best_val_loss" in payload:
            self._best_monitor = payload["best_val_loss"]
        # else: keep the +/-inf init from __init__ for the new monitor.
        self._start_epoch = payload["epoch"] + 1
        # Scheduler state is loaded later in train() once steps_per_epoch is known.
        self._resume_scheduler_state = payload.get("scheduler_state_dict")
        logger.info("Resuming from epoch %d (best %s so far: %.4f)",
                    payload["epoch"], self._monitor, self._best_monitor)

    def _prune_old_checkpoints(self, current_epoch: int) -> None:
        n = self.config.keep_last_n
        epoch_ckpts = sorted(
            self.output_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )
        for old in epoch_ckpts[:-n]:
            old.unlink()
            logger.debug("Pruned old checkpoint: %s", old)

    # Updated after every scheduler step so _build_payload can read it without
    # passing the scheduler object through the call chain.
    _scheduler_state: dict | None = None
    _resume_scheduler_state: dict | None = None

    def _compute_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        sender_ids: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stride labels/sender_ids to episode granularity and apply the loss.

        LUAR episode pooling shrinks P*K rows → P*(K/episode_k); labels (and
        sender_ids, when the loss wants them) are strided by the same factor
        to stay aligned with the embedding rows. Returns (loss, batch_labels).
        """
        batch_labels = labels[::self._episode_k] if self._episode_k else labels
        if self._loss_wants_sender_ids:
            batch_sender_ids = (
                sender_ids[:: self._episode_k] if self._episode_k else sender_ids
            )
            loss = self.loss_fn(embeddings, batch_labels, sender_ids=batch_sender_ids)
        else:
            loss = self.loss_fn(embeddings, batch_labels)
        return loss, batch_labels

    def _train_epoch(self, loader: DataLoader, scheduler: Any) -> float:
        """Single training epoch; returns mean loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(loader, desc="train", leave=False):
            texts: list[str] = batch.texts
            labels: torch.Tensor = batch.labels.to(self.device)

            token_dict = self.model.tokenize(texts)
            token_dict = {k: v.to(self.device) for k, v in token_dict.items()}

            self.optimizer.zero_grad()

            if self.scaler is not None:
                with torch.amp.autocast(device_type=self.device):
                    embeddings = self.model.encode(**token_dict)
                    loss, _ = self._compute_loss(embeddings, labels, batch.sender_ids)
                self.scaler.scale(loss).backward()
                # Unscale before clip so the norm is in true fp32 units.
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                embeddings = self.model.encode(**token_dict)
                loss, _ = self._compute_loss(embeddings, labels, batch.sender_ids)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()

            scheduler.step()
            self._scheduler_state = scheduler.state_dict()
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _validate(self, loader: DataLoader) -> dict[str, float]:
        """Compute val loss and embedding-space classification metrics."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_embs: list[torch.Tensor] = []
        all_labels: list[int] = []

        with torch.no_grad():
            for batch in tqdm(loader, desc="val", leave=False):
                texts: list[str] = batch.texts
                labels: torch.Tensor = batch.labels.to(self.device)

                token_dict = self.model.tokenize(texts)
                token_dict = {k: v.to(self.device) for k, v in token_dict.items()}

                embeddings = self.model.encode(**token_dict)
                loss, batch_labels = self._compute_loss(
                    embeddings, labels, batch.sender_ids
                )
                total_loss += loss.item()
                n_batches += 1

                all_embs.append(embeddings.detach().cpu())
                all_labels.extend(batch_labels.cpu().tolist())

        metrics: dict[str, float] = {"val/loss": total_loss / max(n_batches, 1)}
        if all_embs:
            embs = torch.cat(all_embs, dim=0)
            labels_t = torch.tensor(all_labels)
            metrics.update(self._compute_embedding_metrics(embs, labels_t))
        return metrics

    def _compute_embedding_metrics(
        self, embs: torch.Tensor, labels: torch.Tensor
    ) -> dict[str, float]:
        """1-NN accuracy, macro F1, and pairwise authorship AUROC."""
        import numpy as np
        import torch.nn.functional as F
        from sklearn.metrics import f1_score, roc_auc_score

        N = embs.size(0)
        if N < 2:
            return {}

        embs_norm = F.normalize(embs, dim=1)
        sim = embs_norm @ embs_norm.T

        sim_loo = sim.clone()
        sim_loo.fill_diagonal_(-2.0)
        nn_labels = labels[sim_loo.argmax(dim=1)]

        knn_acc = (nn_labels == labels).float().mean().item()
        macro_f1 = float(
            f1_score(labels.numpy(), nn_labels.numpy(), average="macro", zero_division=0)
        )

        labels_np = labels.numpy()
        sim_np = sim.numpy()
        triu_i, triu_j = np.triu_indices(N, k=1)
        pair_sims = sim_np[triu_i, triu_j]
        pair_labels = (labels_np[triu_i] == labels_np[triu_j]).astype(int)
        auroc = (
            float(roc_auc_score(pair_labels, pair_sims))
            if 0 < pair_labels.sum() < len(pair_labels)
            else 0.5
        )

        return {
            "embedding/knn_accuracy": knn_acc,
            "embedding/knn_macro_f1": macro_f1,
            "embedding/pair_auroc": auroc,
        }

    def _inline_pan_eval(self) -> dict[str, float]:
        """Score test_pairs.jsonl and return PAN metrics (AUC/EER/F1) inline."""
        import json
        import numpy as np
        import torch.nn.functional as F
        from email_fraud.scoring.metrics import compute_verification_metrics

        pairs_path = self.eval_data_dir / "test_pairs.jsonl"
        if not pairs_path.exists():
            return {}

        pairs: list[tuple[str, str, int]] = []
        with pairs_path.open() as fh:
            for line in fh:
                rec = json.loads(line)
                if "pair" in rec:
                    text1, text2 = rec["pair"]
                else:
                    text1 = rec.get("text1") or rec.get("text_a")
                    text2 = rec.get("text2") or rec.get("text_b")
                label = int(bool(rec.get("same", rec.get("label", 0))))
                pairs.append((str(text1), str(text2), label))

        from email_fraud.data.preprocessing import preprocess
        flat_texts_raw = [t for p in pairs for t in p[:2]]
        flat_texts = (
            [preprocess(t, self.preprocessing) or t for t in flat_texts_raw]
            if self.preprocessing else flat_texts_raw
        )

        was_training = self.model.training
        self.model.eval()

        # LUAR episode encoders return one embedding per episode of episode_k texts.
        # Force episode_k=1 so each text gets its own embedding for pair scoring.
        saved_episode_k: int | None = None
        if hasattr(self.model, "config") and hasattr(self.model.config, "episode_k"):
            saved_episode_k = self.model.config.episode_k
            self.model.config.episode_k = 1

        all_embs: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(flat_texts), 64):
                batch = flat_texts[start : start + 64]
                tok = self.model.tokenize(batch)
                tok = {k: v.to(self.device) for k, v in tok.items()}
                all_embs.append(self.model.encode(**tok).detach().cpu())

        if saved_episode_k is not None:
            self.model.config.episode_k = saved_episode_k
        if was_training:
            self.model.train()
        embs = torch.cat(all_embs, dim=0)

        scores = []
        for i in range(0, embs.size(0), 2):
            sim = F.cosine_similarity(embs[i].unsqueeze(0), embs[i + 1].unsqueeze(0)).item()
            scores.append((sim + 1.0) / 2.0)
        labels = np.array([lbl for _, _, lbl in pairs], dtype=np.int64)
        return compute_verification_metrics(labels, np.array(scores, dtype=np.float64))

    def _maybe_update_hard_negatives(self, epoch: int, train_loader: Any) -> None:
        """Re-mine hard pairs and update the sampler's batch composition."""
        hnm = self.config.hard_negative_mining
        if not hnm.enabled or not self._train_texts:
            return
        if epoch < hnm.warmup_epochs:
            return

        alpha = min(1.0, (epoch - hnm.warmup_epochs) / max(hnm.ramp_epochs, 1))
        hard_fraction = alpha * hnm.max_hard_fraction

        epochs_since_warmup = epoch - hnm.warmup_epochs
        if epochs_since_warmup % hnm.interval == 0:
            from email_fraud.data.hard_negatives import mine_hard_pairs

            logger.info(
                "Hard negative mining at epoch %d (alpha=%.2f, fraction=%.2f)...",
                epoch, alpha, hard_fraction,
            )
            self._hard_pairs = mine_hard_pairs(
                self.model,
                self._train_texts,
                self._train_sender_ids,
                self.device,
                n_pairs=hnm.n_pairs,
            )

        if not self._hard_pairs:
            return

        sampler = getattr(train_loader, "batch_sampler", None)
        if sampler is not None and hasattr(sampler, "set_hard_pairs"):
            sampler.set_hard_pairs(self._hard_pairs, hard_fraction=hard_fraction)

    def _build_scheduler(self, steps_per_epoch: int) -> Any:
        """Build the LR scheduler. All schedulers step per batch, not per epoch."""
        total_steps = steps_per_epoch * self.config.epochs
        warmup = self.config.warmup_steps

        if self.config.scheduler == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

            warmup_sched = LinearLR(
                self.optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup
            )
            cosine_sched = CosineAnnealingLR(
                self.optimizer, T_max=max(total_steps - warmup, 1)
            )
            scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_sched, cosine_sched],
                milestones=[warmup],
            )
        elif self.config.scheduler == "linear":
            from torch.optim.lr_scheduler import LinearLR

            scheduler = LinearLR(
                self.optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps
            )
        elif self.config.scheduler == "constant":
            from torch.optim.lr_scheduler import ConstantLR

            scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=total_steps)
        else:
            raise ValueError(
                f"Unknown scheduler '{self.config.scheduler}'. "
                "Choose from: 'cosine', 'linear', 'constant'."
            )

        if self._resume_scheduler_state is not None:
            scheduler.load_state_dict(self._resume_scheduler_state)
            self._resume_scheduler_state = None

        self._scheduler_state = scheduler.state_dict()
        return scheduler
