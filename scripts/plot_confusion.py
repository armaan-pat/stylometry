"""Confusion-matrix visualisations at operationally-meaningful thresholds.

For each (checkpoint, scoring function) combination, this script produces a
grid of confusion matrices at three different thresholds:
  - τ such that FPR on synthetics = 1%   (very conservative — for "high-trust" channel)
  - τ such that FPR on synthetics = 5%   (typical deployment)
  - τ such that FPR on synthetics = 10%  (relaxed)

Two output formats per (checkpoint, scoring fn):
  1. **Binary 2×2** — genuine vs ANY impostor (the standard view)
  2. **Extended 3×2** — breaks out impostor pool into other-sender vs synthetic,
     so you can see how the threshold trades off the two kinds of false positive.

Usage
-----
    # v6 with the v6 default scoring
    python scripts/plot_confusion.py \\
        --checkpoint runs/v6_luar_lora_syn/2026-05-26_19-09-22/checkpoint_best.pt \\
        --config configs/experiments/v6_luar_lora_syn.yaml \\
        --tag v6 --scorers linear_z3 mahal_per_sender

    # v7 once training finishes
    python scripts/plot_confusion.py \\
        --checkpoint runs/v7_luar_lora_syn_mahal/<TS>/checkpoint_best.pt \\
        --config configs/experiments/v7_luar_lora_syn_mahal.yaml \\
        --tag v7 --scorers linear_z3 mahal_per_sender

Reads
-----
Probe definition (which senders, which queries) mirrors V7.0 — see
scripts/eval_v7_scoring.py.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")

import email_fraud.data.enron  # noqa
import email_fraud.encoders    # noqa
import email_fraud.heads       # noqa
import email_fraud.losses      # noqa
from email_fraud.config import load_config
from email_fraud.data.enron import EnronDataset
from email_fraud.registry import resolve as resolve_component
from email_fraud.utils.logging import setup_logging

from scripts.eval_v7_scoring import (
    SenderProfile, _encode_texts, _mahal_distance,
    score_cosine, score_linear_z3, score_sigmoid_z, score_mahal_per_sender,
    score_mahal_tied, score_linear_z3_median,
    _compute_tied_precision, _build_probe,
)

logger = logging.getLogger(__name__)


SCORERS_REGISTRY = {
    "cosine": score_cosine,
    "linear_z3": score_linear_z3,
    "linear_z3_median": score_linear_z3_median,
    "sigmoid_z": score_sigmoid_z,
    "mahal_per_sender": score_mahal_per_sender,
    "mahal_tied": score_mahal_tied,
}


# =============================================================================
# Confusion matrix helpers
# =============================================================================


def _threshold_at_fpr(impostor_scores: np.ndarray, fpr: float) -> float:
    """Return the score threshold τ such that exactly `fpr` fraction of the
    impostor pool has score > τ."""
    if len(impostor_scores) == 0:
        return float("nan")
    return float(np.quantile(impostor_scores, 1.0 - fpr))


def _confusion_2x2(gen: np.ndarray, imp: np.ndarray, tau: float) -> dict:
    """Standard binary confusion matrix.
    Positive class = "genuine" — score > τ is predicted genuine.
    """
    tp = int((gen > tau).sum())
    fn = int((gen <= tau).sum())
    fp = int((imp > tau).sum())
    tn = int((imp <= tau).sum())
    total = tp + fn + fp + tn
    return {
        "matrix": np.array([[tp, fn], [fp, tn]]),
        "tpr": tp / max(tp + fn, 1),
        "fpr": fp / max(fp + tn, 1),
        "precision": tp / max(tp + fp, 1),
        "accuracy": (tp + tn) / max(total, 1),
        "tau": tau,
    }


def _confusion_3x2(gen: np.ndarray, oth: np.ndarray, syn: np.ndarray, tau: float) -> dict:
    """3-row × 2-col confusion: split the impostor pool by source so we can
    see how the threshold treats other-sender vs synthetic differently."""
    tp = int((gen > tau).sum())
    fn = int((gen <= tau).sum())
    fp_o = int((oth > tau).sum())
    tn_o = int((oth <= tau).sum())
    fp_s = int((syn > tau).sum())
    tn_s = int((syn <= tau).sum())
    return {
        "matrix": np.array([[tp, fn], [fp_o, tn_o], [fp_s, tn_s]]),
        "tpr": tp / max(tp + fn, 1),
        "fpr_other": fp_o / max(fp_o + tn_o, 1),
        "fpr_synthetic": fp_s / max(fp_s + tn_s, 1),
        "tau": tau,
    }


# =============================================================================
# Plotting
# =============================================================================


def _draw_confusion_2x2(ax, cm: dict, title: str) -> None:
    """One 2x2 confusion matrix in a single subplot."""
    m = cm["matrix"]
    # Normalise per row so the colour reflects rate, not raw count.
    row_sums = m.sum(axis=1, keepdims=True)
    m_norm = m / np.maximum(row_sums, 1)

    im = ax.imshow(m_norm, cmap="Blues", vmin=0, vmax=1, aspect="equal")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred:\nGenuine", "Pred:\nFraud"], fontsize=9)
    ax.set_yticklabels(["Actual:\nGenuine", "Actual:\nFraud"], fontsize=9)

    for i in range(2):
        for j in range(2):
            count = m[i, j]
            rate = m_norm[i, j]
            color = "white" if rate > 0.5 else "#222"
            ax.text(j, i, f"{count}\n({rate:.1%})",
                    ha="center", va="center", fontsize=12,
                    color=color, fontweight="bold")

    subtitle = (
        f"τ={cm['tau']:.3f}  |  "
        f"TPR={cm['tpr']:.1%}  FPR={cm['fpr']:.1%}\n"
        f"Precision={cm['precision']:.1%}  Accuracy={cm['accuracy']:.1%}"
    )
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)
    return im


def _draw_confusion_3x2(ax, cm: dict, title: str) -> None:
    """3x2 confusion: actual-row split into Genuine / Other-sender / Synthetic."""
    m = cm["matrix"]
    row_sums = m.sum(axis=1, keepdims=True)
    m_norm = m / np.maximum(row_sums, 1)

    im = ax.imshow(m_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(["Pred:\nGenuine\n(pass)", "Pred:\nFraud\n(block)"], fontsize=9)
    ax.set_yticklabels(
        ["Actual:\nGenuine",
         "Actual:\nOther-sender\nimpostor",
         "Actual:\nLLM imitation\n(synthetic)"],
        fontsize=9,
    )

    for i in range(3):
        for j in range(2):
            count = m[i, j]
            rate = m_norm[i, j]
            color = "white" if rate > 0.5 else "#222"
            ax.text(j, i, f"{count}\n({rate:.1%})",
                    ha="center", va="center", fontsize=11,
                    color=color, fontweight="bold")

    subtitle = (
        f"τ={cm['tau']:.3f}  |  Real kept={cm['tpr']:.1%}\n"
        f"FPR other={cm['fpr_other']:.1%}  "
        f"FPR synthetic={cm['fpr_synthetic']:.1%}"
    )
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)
    return im


def _make_figure_2x2(scorer_name: str, gen, oth, syn, taus_by_label, out: Path,
                     header_caption: str) -> None:
    """Three side-by-side 2x2 confusion matrices for one (model, scorer)."""
    imp = np.concatenate([oth, syn]) if len(oth) and len(syn) else (oth if len(oth) else syn)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (label, tau) in zip(axes, taus_by_label.items()):
        cm = _confusion_2x2(gen, imp, tau)
        _draw_confusion_2x2(ax, cm, label)
    fig.suptitle(
        f"{header_caption}  —  scorer: {scorer_name}\n"
        f"Confusion matrices at three operating points\n"
        f"({len(gen)} genuine + {len(imp)} impostor queries)",
        fontsize=12, y=1.05,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("saved %s", out)


def _make_figure_3x2(scorer_name: str, gen, oth, syn, taus_by_label, out: Path,
                     header_caption: str) -> None:
    """Three side-by-side 3x2 confusion matrices (impostor pool broken out)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    for ax, (label, tau) in zip(axes, taus_by_label.items()):
        cm = _confusion_3x2(gen, oth, syn, tau)
        _draw_confusion_3x2(ax, cm, label)
    fig.suptitle(
        f"{header_caption}  —  scorer: {scorer_name}\n"
        f"Confusion matrices, impostor pool broken out\n"
        f"({len(gen)} genuine + {len(oth)} other + {len(syn)} synthetic queries)",
        fontsize=12, y=1.03,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("saved %s", out)


# =============================================================================
# Pipeline
# =============================================================================


def _encode_pools(encoder, train_ds, val_ds, syn_path, args, device):
    """Build the probe and encode every pool once.

    Returns the probe dict + dict-of-pool-embeddings + sender_id assignments.
    """
    probe = _build_probe(
        train_ds, val_ds, syn_path,
        n_profile_senders=args.n_profile_senders,
        n_enroll=args.n_enroll,
        n_query=args.n_query,
        n_other=args.n_other,
        n_synth=args.n_synth,
        seed=args.seed,
    )
    enroll = _encode_texts(encoder, probe["enroll_texts"], device)
    gen = _encode_texts(encoder, probe["gen_texts"], device)
    oth = _encode_texts(encoder, probe["other_texts"], device)
    syn = (_encode_texts(encoder, probe["syn_texts"], device)
           if probe["syn_texts"] else np.empty((0, gen.shape[1])))

    sid_to_idx = defaultdict(list)
    for i, sid in enumerate(probe["enroll_sids"]):
        sid_to_idx[sid].append(i)
    profiles = {sid: SenderProfile(sid, enroll[idxs]) for sid, idxs in sid_to_idx.items()}
    rng = random.Random(args.seed)
    chosen = probe["chosen_senders"]
    oth_sids = [rng.choice(chosen) for _ in range(len(oth))]
    syn_sids = list(probe["syn_sids"])
    gen_sids = list(probe["gen_sids"])

    return {
        "profiles": profiles,
        "tied_prec": _compute_tied_precision(profiles),
        "gen_emb": gen, "oth_emb": oth, "syn_emb": syn,
        "gen_sids": gen_sids, "oth_sids": oth_sids, "syn_sids": syn_sids,
    }


def _score_pools(scorer_name, payload):
    """Apply one scorer across all three pools and return per-pool arrays."""
    fn = SCORERS_REGISTRY[scorer_name]
    ctx = {"tied_prec": payload["tied_prec"]}
    profiles = payload["profiles"]

    def _arr(emb_pool, sids):
        return np.array([fn(emb_pool[i], profiles[s], ctx) for i, s in enumerate(sids)])

    gen = _arr(payload["gen_emb"], payload["gen_sids"])
    oth = _arr(payload["oth_emb"], payload["oth_sids"])
    syn = _arr(payload["syn_emb"], payload["syn_sids"]) if len(payload["syn_emb"]) else np.array([])
    return gen, oth, syn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tag", required=True, help="Filename prefix (e.g. v6, v7).")
    p.add_argument("--out-dir", default="results/v7/confusion")
    p.add_argument("--scorers", nargs="+",
                   default=["linear_z3", "mahal_per_sender"],
                   help="Score functions to plot (see SCORERS_REGISTRY).")
    p.add_argument("--n-profile-senders", type=int, default=30)
    p.add_argument("--n-enroll", type=int, default=8)
    p.add_argument("--n-query", type=int, default=4)
    p.add_argument("--n-other", type=int, default=200)
    p.add_argument("--n-synth", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()

    cfg_path = _PROJECT_ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    cfg = load_config(str(cfg_path))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    EncoderClass = resolve_component("encoder", cfg.encoder.name)
    encoder = EncoderClass(cfg.encoder)
    ckpt_path = _PROJECT_ROOT / args.checkpoint if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    payload = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    encoder.load_state_dict(payload["model_state_dict"])
    encoder.to(device).eval()
    epoch = payload.get("epoch", "?")
    logger.info("Loaded epoch %s from %s", epoch, ckpt_path)

    train_ds = EnronDataset(cfg.data, split="train")
    val_ds = EnronDataset(cfg.data, split="validation")

    enc_payload = _encode_pools(
        encoder, train_ds, val_ds, cfg.data.augmentation.synthetic_path,
        args, device,
    )

    out_dir = _PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    header_caption = f"{args.tag.upper()}  (epoch {epoch})"

    for scorer_name in args.scorers:
        logger.info("scoring with %s", scorer_name)
        gen, oth, syn = _score_pools(scorer_name, enc_payload)

        # Compute three operating thresholds anchored on synthetic FPR.
        taus = {
            "Conservative\n(FPR_syn = 1%)": _threshold_at_fpr(syn, 0.01),
            "Operational\n(FPR_syn = 5%)": _threshold_at_fpr(syn, 0.05),
            "Relaxed\n(FPR_syn = 10%)":     _threshold_at_fpr(syn, 0.10),
        }

        # 1) Standard 2x2 confusion (genuine vs ANY impostor)
        out2 = out_dir / f"{args.tag}_{scorer_name}_confusion_2x2.png"
        _make_figure_2x2(scorer_name, gen, oth, syn, taus, out2, header_caption)

        # 2) Extended 3x2 confusion (impostor split)
        out3 = out_dir / f"{args.tag}_{scorer_name}_confusion_3x2.png"
        _make_figure_3x2(scorer_name, gen, oth, syn, taus, out3, header_caption)

    logger.info("Done. Wrote PNGs to %s", out_dir)


if __name__ == "__main__":
    main()
