"""Plot V7 vs V6 — AUROC and TPR curves across K, scoring functions, and time.

Generates two figures:
  1. results/v7/v7_k_sweep_comparison.png — line plot of AUC[g/syn] across K
     for each (model, scoring_fn) combination. Shows where Mahalanobis wins.
  2. results/v7/v7_scoring_bar.png — grouped bar chart of AUC + TPR at K=8
     for cosine vs linear_z3 vs mahal_per_sender, comparing v6 and v7.

How to read
-----------
A line going up with K means the scorer benefits from more enrollment data.
Mahalanobis lines should climb steeply between K=4 and K=16. Cosine lines
should be roughly flat past K=8 (the centroid is well-estimated already).
The gap between the v6 and v7 lines for the same scorer measures
training-side improvement; the gap between two scorers on the same model
measures scoring-side improvement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def _plot_k_sweep(v6: dict, v7: dict, out: Path) -> None:
    rows6 = {r["K"]: r["scores"] for r in v6["rows"]}
    rows7 = {r["K"]: r["scores"] for r in v7["rows"]}
    ks = sorted(set(rows6) & set(rows7))
    scorers = ["cosine", "linear_z3", "mahal_per_sender"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    palette = {"cosine": "#1f77b4", "linear_z3": "#ff7f0e", "mahal_per_sender": "#2ca02c"}

    for ax, metric, ylabel in [
        (axes[0], "auc_g_syn", "AUROC (genuine vs synthetic)"),
        (axes[1], "tpr@5pct_syn", "TPR @ 5% FPR (genuine vs synthetic)"),
    ]:
        for sc in scorers:
            y6 = [rows6[k][sc].get(metric, np.nan) for k in ks]
            y7 = [rows7[k][sc].get(metric, np.nan) for k in ks]
            ax.plot(ks, y6, marker="o", linestyle="--", color=palette[sc],
                    label=f"v6 {sc}", alpha=0.7)
            ax.plot(ks, y7, marker="s", linestyle="-", color=palette[sc],
                    label=f"v7 {sc}", alpha=1.0)
        ax.set_xlabel("Enrollment K (emails per sender)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(ks)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2, loc="lower right")
    axes[0].set_title("AUROC: genuine vs synthetic")
    axes[1].set_title("TPR @ 5% FPR: genuine vs synthetic")

    fig.suptitle("V7 vs V6 — Mahalanobis-aware retraining + scoring", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"saved {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--v6-k", default="results/v7/v6_k_sweep.json")
    p.add_argument("--v7-k", default="results/v7/v7_k_sweep.json")
    p.add_argument("--out", default="results/v7/v7_k_sweep_comparison.png")
    args = p.parse_args()

    v6 = _load(Path(args.v6_k))
    v7 = _load(Path(args.v7_k))
    _plot_k_sweep(v6, v7, Path(args.out))


if __name__ == "__main__":
    main()
