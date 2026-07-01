#!/usr/bin/env python3
"""Whitepaper figures from the new multi-seed / novel-vendor / 2x2 results.

Outputs to docs/figures/whitepaper/:
  wp_fig1_multiseed_progression.png  -- v11->v14b held-out AUC, mean +/- std (error bars)
  wp_fig2_2x2_grid.png               -- identity x synthetic factorial (mahalanobis AUC)
  wp_fig3_novelvendor.png            -- v12 vs v14b on Qwen+DeepSeek (AUC + ranking)
Run: python scripts/whitepaper/make_figures.py
"""
from __future__ import annotations
import glob, json, os, statistics
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MS = os.path.join(ROOT, "results/whitepaper/multiseed")
NV = os.path.join(ROOT, "results/whitepaper/novelvendor")
OUT = os.path.join(ROOT, "docs/figures/whitepaper")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.axisbelow": True, "figure.dpi": 130})


def rows(path, scorer):
    try:
        d = json.load(open(path))
    except Exception:
        return None
    return next((r for r in d.get("rows", []) if r["scorer"] == scorer), None)


def msd(stem, scorer, metric, base=MS, pat="heldoutCG_{}_s[0-9]*.json"):
    vals = []
    for f in sorted(glob.glob(os.path.join(base, pat.format(stem)))):
        r = rows(f, scorer)
        if r and r.get(metric) is not None:
            vals.append(r[metric])
    if not vals:
        return None
    return (statistics.mean(vals), statistics.pstdev(vals) if len(vals) > 1 else 0.0, len(vals))


# ---------- Fig 1: multiseed progression with error bars ----------
def fig1():
    vers = ["v11", "v12", "v13", "v14", "v14b"]
    labels = ["v11\nsingle-gen", "v12\nmulti-gen", "v13\n+DeepSeek", "v14\nidentity\n(no syn)", "v14b\nsynthesis"]
    auc = [msd(v, "mahalanobis", "auc") for v in vers]
    means = [a[0] if a else np.nan for a in auc]
    stds = [a[1] if a else 0 for a in auc]
    x = np.arange(len(vers))
    fig, ax = plt.subplots(figsize=(9, 5.2))
    colors = ["#9aa0a6", "#4285f4", "#4285f4", "#fbbc04", "#34a853"]
    ax.bar(x, means, 0.6, yerr=stds, capsize=6, color=colors,
           error_kw=dict(ecolor="#202124", lw=1.5))
    for xi, m, s in zip(x, means, stds):
        if not np.isnan(m):
            ax.text(xi, m + s + 0.012, f"{m:.3f}\n±{s:.3f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(0.5, ls="--", c="grey", lw=1); ax.text(len(vers)-0.5, 0.515, "chance", color="grey", ha="right", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Held-out Claude+Gemini AUC (mahalanobis)")
    ax.set_ylim(0.45, 1.04)
    ax.set_title("Cross-generator detection across the version lineage\n(mean ± std over 5 probe draws; held-out generators)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "wp_fig1_multiseed_progression.png")); plt.close(fig)


# ---------- Fig 2: 2x2 identity x synthetic grid ----------
def fig2():
    grid_stems = {("44", "nosyn"): "enron44_nosyn", ("44", "syn"): "v12",
                  ("844", "nosyn"): "v14", ("844", "syn"): "v14b"}
    M = np.full((2, 2), np.nan)
    txt = [["", ""], ["", ""]]
    rowi = {"844": 0, "44": 1}; coli = {"nosyn": 0, "syn": 1}
    for (a, s), stem in grid_stems.items():
        m = msd(stem, "mahalanobis", "auc")
        if m:
            M[rowi[a], coli[s]] = m[0]
            txt[rowi[a]][coli[s]] = f"{m[0]:.3f}\n±{m[1]:.3f}\n({stem})"
        else:
            txt[rowi[a]][coli[s]] = "(pending)"
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0.5, vmax=1.0)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["no synthetics", "+ synthetics"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["844 authors\n(Enron+Blog)", "44 authors\n(Enron)"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, txt[i][j], ha="center", va="center", fontsize=10, fontweight="bold")
    ax.set_title("Identity × synthetic 2×2\nHeld-out Claude+Gemini AUC (mahalanobis)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="AUC")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "wp_fig2_2x2_grid.png")); plt.close(fig)


# ---------- Fig 3: novel-vendor generalization ----------
def fig3():
    vers = ["v12", "v14b"]
    auc = [msd(v, "mahalanobis", "auc", base=NV, pat="novelvendor_{}_s[0-9]*.json") for v in vers]
    gor = [msd(v, "mahalanobis", "auc_g_other", base=NV, pat="novelvendor_{}_s[0-9]*.json") for v in vers]
    if not any(auc):
        return
    x = np.arange(len(vers)); w = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    a_m = [a[0] if a else np.nan for a in auc]; a_s = [a[1] if a else 0 for a in auc]
    g_m = [g[0] if g else np.nan for g in gor]; g_s = [g[1] if g else 0 for g in gor]
    b1 = ax.bar(x - w/2, a_m, w, yerr=a_s, capsize=5, label="AUC (forgery vs genuine)", color="#34a853")
    b2 = ax.bar(x + w/2, g_m, w, yerr=g_s, capsize=5, label="AUC (genuine vs wrong-human)", color="#4285f4")
    for bars, ms in ((b1, a_m), (b2, g_m)):
        for b, m in zip(bars, ms):
            if not np.isnan(m):
                ax.text(b.get_x()+b.get_width()/2, m+0.012, f"{m:.3f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(0.5, ls="--", c="grey", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(["v12", "v14b"])
    ax.set_ylabel("AUC"); ax.set_ylim(0.45, 1.04)
    ax.set_title("Generalization to NOVEL vendors (Qwen-2.5-72B + DeepSeek-V3)\nnever in training OR held-out eval")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "wp_fig3_novelvendor.png")); plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2(); fig3()
    print("[written]", OUT)
    for f in sorted(glob.glob(os.path.join(OUT, "*.png"))):
        print("  ", os.path.relpath(f, ROOT))
