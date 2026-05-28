"""V6 vs V7.3 side-by-side comparison plots.

Generates two comparison figures:

  1. results/v7/confusion/v6_vs_v7_comparison.png  —  4 confusion matrices in
     a 2x2 grid: rows = {V6, V7}, columns = {operational, conservative}.
     One scoring function (configurable). Annotated so the bottom-line numbers
     pop out without reading the table.

  2. results/v7/confusion/v6_vs_v7_k_curves.png  —  TPR@5%FPR_syn vs K for
     each (model, scorer) combination. Visualises the "Mahalanobis advantage
     grows with K" finding and the V6 → V7 gap at every K.

Inputs are the dumped k-sweep JSONs:
  results/v7/v6_k_sweep.json     (created here on the fly from V6 checkpoint
                                  if it doesn't exist — small enough)
  results/v7/v7_3_k_sweep.json   (already exists)
"""

from __future__ import annotations

import argparse
import json
import logging
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
    SenderProfile, _encode_texts, _build_probe, _compute_tied_precision,
    score_linear_z3, score_mahal_per_sender, score_cosine,
)

logger = logging.getLogger(__name__)


SCORERS = {
    "linear_z3": score_linear_z3,
    "mahal_per_sender": score_mahal_per_sender,
    "cosine": score_cosine,
}


# ============================================================================
# Re-encode a checkpoint and return per-pool, per-scorer score arrays at one K
# ============================================================================


def _score_checkpoint(cfg_path, ckpt_path, K, args, device):
    cfg = load_config(str(cfg_path))
    EncoderClass = resolve_component("encoder", cfg.encoder.name)
    encoder = EncoderClass(cfg.encoder)
    payload = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    encoder.load_state_dict(payload["model_state_dict"])
    encoder.to(device).eval()

    train_ds = EnronDataset(cfg.data, split="train")
    val_ds = EnronDataset(cfg.data, split="validation")

    probe = _build_probe(
        train_ds, val_ds, cfg.data.augmentation.synthetic_path,
        n_profile_senders=args.n_profile_senders,
        n_enroll=K,
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
    tied_prec = _compute_tied_precision(profiles)
    ctx = {"tied_prec": tied_prec}

    import random
    rng = random.Random(args.seed)
    chosen = probe["chosen_senders"]
    oth_sids = [rng.choice(chosen) for _ in range(len(oth))]
    syn_sids = list(probe["syn_sids"])
    gen_sids = list(probe["gen_sids"])

    out = {}
    for name, fn in SCORERS.items():
        out[name] = {
            "gen": np.array([fn(gen[i], profiles[s], ctx) for i, s in enumerate(gen_sids)]),
            "oth": np.array([fn(oth[i], profiles[s], ctx) for i, s in enumerate(oth_sids)]),
            "syn": np.array([fn(syn[i], profiles[s], ctx) for i, s in enumerate(syn_sids)])
                   if len(syn) else np.array([]),
        }
    return out


# ============================================================================
# Plot 1: 2x2 confusion grid
# ============================================================================


def _conf_3x2(gen, oth, syn, tau):
    return np.array([
        [int((gen > tau).sum()), int((gen <= tau).sum())],
        [int((oth > tau).sum()), int((oth <= tau).sum())],
        [int((syn > tau).sum()), int((syn <= tau).sum())],
    ])


def _draw_panel(ax, gen, oth, syn, tau, title):
    cm = _conf_3x2(gen, oth, syn, tau)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = cm / np.maximum(row_sums, 1)
    ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(["Pred:\nGenuine\n(pass)", "Pred:\nFraud\n(block)"], fontsize=9)
    ax.set_yticklabels(
        ["Actual:\nGenuine",
         "Actual:\nOther-sender",
         "Actual:\nLLM imitation"],
        fontsize=9,
    )
    for i in range(3):
        for j in range(2):
            count = cm[i, j]
            rate = cm_norm[i, j]
            color = "white" if rate > 0.5 else "#222"
            ax.text(j, i, f"{count}\n({rate:.1%})",
                    ha="center", va="center", fontsize=10,
                    color=color, fontweight="bold")
    tpr = cm[0, 0] / max(cm[0].sum(), 1)
    fpr_oth = cm[1, 0] / max(cm[1].sum(), 1)
    fpr_syn = cm[2, 0] / max(cm[2].sum(), 1)
    ax.set_title(
        f"{title}\nReal kept: {tpr:.1%}  |  "
        f"FPR_other: {fpr_oth:.1%}  FPR_syn: {fpr_syn:.1%}",
        fontsize=10,
    )


def make_comparison_grid(v6, v7, scorer, K, out_path):
    """Two rows (V6, V7) × two cols (Conservative 1%, Operational 5%)."""
    s6 = v6[scorer]
    s7 = v7[scorer]

    tau6_cons = float(np.quantile(s6["syn"], 0.99))
    tau6_op = float(np.quantile(s6["syn"], 0.95))
    tau7_cons = float(np.quantile(s7["syn"], 0.99))
    tau7_op = float(np.quantile(s7["syn"], 0.95))

    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    _draw_panel(axes[0, 0], s6["gen"], s6["oth"], s6["syn"], tau6_cons,
                f"V6  —  Conservative (FPR_syn = 1%)\nτ = {tau6_cons:.3f}")
    _draw_panel(axes[0, 1], s6["gen"], s6["oth"], s6["syn"], tau6_op,
                f"V6  —  Operational (FPR_syn = 5%)\nτ = {tau6_op:.3f}")
    _draw_panel(axes[1, 0], s7["gen"], s7["oth"], s7["syn"], tau7_cons,
                f"V7  —  Conservative (FPR_syn = 1%)\nτ = {tau7_cons:.3f}")
    _draw_panel(axes[1, 1], s7["gen"], s7["oth"], s7["syn"], tau7_op,
                f"V7  —  Operational (FPR_syn = 5%)\nτ = {tau7_op:.3f}")
    fig.suptitle(
        f"V6 vs V7 — confusion matrices  ({scorer}, K={K})\n"
        f"({len(s6['gen'])} genuine, {len(s6['oth'])} other, {len(s6['syn'])} synthetic queries)",
        fontsize=13, y=0.99,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    logger.info("saved %s", out_path)


# ============================================================================
# Plot 2: K-curves
# ============================================================================


def _tpr_at_fpr(gen, neg, fpr):
    if len(gen) == 0 or len(neg) == 0:
        return float("nan")
    tau = float(np.quantile(neg, 1.0 - fpr))
    return float((gen > tau).mean())


def make_k_curve_plot(v6_k, v7_k, out_path):
    """For each K, plot TPR@5%FPR_syn and TPR@1%FPR_syn."""
    Ks = sorted(set(v6_k.keys()) & set(v7_k.keys()))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    targets = {"5% synthetic FPR": 0.05, "1% synthetic FPR": 0.01}

    for ax, (label, fpr) in zip(axes, targets.items()):
        for color, model_name, k_data in [
            ("#9d8189", "V6", v6_k),
            ("#2a9d8f", "V7", v7_k),
        ]:
            for ls, scorer in [("--", "linear_z3"), ("-", "mahal_per_sender")]:
                ys = []
                for K in Ks:
                    s = k_data[K][scorer]
                    ys.append(_tpr_at_fpr(s["gen"], s["syn"], fpr))
                ax.plot(Ks, ys, ls + "o", color=color,
                        label=f"{model_name}  {scorer}",
                        linewidth=2, markersize=8)
        ax.set_xlabel("Enrollment K (emails per sender)", fontsize=11)
        ax.set_ylabel(f"Real email kept at {label}", fontsize=11)
        ax.set_title(f"TPR at {label}\n(higher is better)", fontsize=12)
        ax.set_xticks(Ks)
        ax.grid(alpha=0.3)
        ax.set_ylim(0.4, 1.0)
        ax.legend(loc="lower right", fontsize=9)

    fig.suptitle("V6 vs V7 — operating-point curves across enrollment K",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    logger.info("saved %s", out_path)


# ============================================================================
# Main
# ============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--v6-config", default="configs/experiments/v6_luar_lora_syn.yaml")
    p.add_argument("--v6-checkpoint",
                   default="runs/v6_luar_lora_syn/2026-05-26_19-09-22/checkpoint_best.pt")
    p.add_argument("--v7-config", default="configs/experiments/v7_luar_lora_syn_mahal_eval.yaml")
    p.add_argument("--v7-checkpoint",
                   default="runs/v7_luar_lora_syn_mahal/2026-05-28_08-54-56/checkpoint_epoch_150.pt")
    p.add_argument("--n-profile-senders", type=int, default=30)
    p.add_argument("--n-query", type=int, default=4)
    p.add_argument("--n-other", type=int, default=200)
    p.add_argument("--n-synth", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ks", nargs="+", type=int, default=[4, 8, 16, 25, 40])
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", default="results/v7/confusion")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = _PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    v6_k: dict[int, dict] = {}
    v7_k: dict[int, dict] = {}
    for K in args.ks:
        logger.info("V6 K=%d", K)
        v6_k[K] = _score_checkpoint(
            _PROJECT_ROOT / args.v6_config,
            _PROJECT_ROOT / args.v6_checkpoint,
            K, args, device,
        )
        logger.info("V7 K=%d", K)
        v7_k[K] = _score_checkpoint(
            _PROJECT_ROOT / args.v7_config,
            _PROJECT_ROOT / args.v7_checkpoint,
            K, args, device,
        )

    # Confusion grids at the most informative K values.
    for K_show in [8, 16]:
        for scorer in ["linear_z3", "mahal_per_sender"]:
            out = out_dir / f"v6_vs_v7_K{K_show}_{scorer}_grid.png"
            make_comparison_grid(v6_k[K_show], v7_k[K_show], scorer, K_show, out)

    # K-curves
    make_k_curve_plot(v6_k, v7_k, out_dir / "v6_vs_v7_k_curves.png")

    logger.info("Done. Plots in %s", out_dir)


if __name__ == "__main__":
    main()
