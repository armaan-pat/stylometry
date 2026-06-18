#!/usr/bin/env python
"""Generate presentation figures for the v11->v14b story.

All numbers are read from the committed result JSONs (no hardcoding of metrics
beyond the slice/scorer selection). Outputs PNGs to docs/figures/.

Run: python scripts/make_v14b_figures.py
"""
from __future__ import annotations
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---- consistent palette ----
C = {
    "v11": "#9aa0a6",  # grey  - single-generator baseline
    "v12": "#4285f4",  # blue  - multi-generator
    "v13": "#a142f4",  # purple- +vendor/volume (plateau)
    "v14": "#f4a142",  # orange- identity-only (regression)
    "v14b": "#34a853", # green - the synthesis (win)
}
VERS = ["v11", "v12", "v13", "v14", "v14b"]
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.axisbelow": True, "figure.dpi": 130})


def L(p):
    p = R / p
    return json.loads(p.read_text()) if p.exists() else None

def slice_auc(d, s):
    return d[s]["AUC"] if d and s in d and "AUC" in d[s] else None

def prow(d, scorer):
    if not d or "rows" not in d:
        return None
    return next((r for r in d["rows"] if r["scorer"] == scorer), None)

# ---- load data ----
ood = {
    "v11": L("v12/ood_v11lora_heldoutCG.json"),
    "v12": L("v12/ood_v12lora_final_heldoutCG.json"),
    "v13": L("v13/ood_v13lora_heldoutCG.json"),
    "v14": L("v14/ood_v14lora_heldoutCG.json"),
    "v14b": L("v14/ood_v14blora_heldoutCG.json"),
}
abl = {
    "v11": L("v12/heldoutCG_v11lora.json"),
    "v12": L("v12/heldoutCG_v12lora_final.json"),
    "v13": L("v13/heldoutCG_v13lora.json"),
    "v14": L("v14/heldoutCG_v14lora.json"),
    "v14b": L("v14/heldoutCG_v14blora.json"),
}
SCORER = "mahalanobis"  # deployment scorer; guardrail-satisfying; fair across versions
CLAUDE = "gen:openrouter:anthropic/claude-3.5-haiku"
GEMINI = "gen:openrouter:google/gemini-2.5-flash"


def barlabels(ax, bars, fmt="{:.3f}", dy=0.012):
    for b in bars:
        h = b.get_height()
        if h is None or np.isnan(h):
            continue
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h),
                ha="center", va="bottom", fontsize=9)


# =====================================================================
# FIG 1 — Headline: cross-generator AUC on HELD-OUT generators
# =====================================================================
fig, ax = plt.subplots(figsize=(9, 5.2))
x = np.arange(len(VERS))
w = 0.38
cla = [slice_auc(ood[v], CLAUDE) for v in VERS]
gem = [slice_auc(ood[v], GEMINI) for v in VERS]
b1 = ax.bar(x - w/2, cla, w, label="Claude-3.5-haiku (held out)", color="#4285f4")
b2 = ax.bar(x + w/2, gem, w, label="Gemini-2.5-flash (held out)", color="#ea4335")
barlabels(ax, b1); barlabels(ax, b2)
ax.axhline(0.5, ls="--", c="grey", lw=1)
ax.text(len(VERS)-0.5, 0.51, "chance", color="grey", fontsize=9, ha="right")
ax.set_xticks(x); ax.set_xticklabels(
    ["v11\nsingle-gen", "v12\nmulti-gen", "v13\n+vendor", "v14\nidentity-only", "v14b\nSYNTHESIS"])
ax.set_ylabel("AUC (forgery vs genuine)")
ax.set_ylim(0.45, 1.02)
ax.set_title("Cross-generator generalization: detecting forgeries from UNSEEN models\n"
             "(Claude + Gemini never appear in training)", fontsize=12)
ax.legend(loc="lower left")
fig.tight_layout(); fig.savefig(OUT / "fig1_cross_generator_auc.png"); plt.close(fig)

# =====================================================================
# FIG 2 — Held-out pool: AUC + TPR@5% progression (deployment scorer)
# =====================================================================
fig, ax = plt.subplots(figsize=(9, 5.2))
pool_auc = [prow(abl[v], SCORER)["auc"] for v in VERS]
pool_tpr5 = [prow(abl[v], SCORER)["tpr5"] for v in VERS]
b1 = ax.bar(x - w/2, pool_auc, w, label="Pool AUC", color="#34a853")
b2 = ax.bar(x + w/2, pool_tpr5, w, label="TPR @ 5% FPR  (forgeries caught)", color="#fbbc04")
barlabels(ax, b1); barlabels(ax, b2)
ax.set_xticks(x); ax.set_xticklabels(VERS)
ax.set_ylabel("score")
ax.set_ylim(0, 1.05)
ax.set_title(f"Held-out Claude+Gemini pool — detection performance ({SCORER} scorer)\n"
             "TPR@5% = share of real forgeries flagged at a 5% false-alarm budget", fontsize=12)
ax.legend(loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "fig2_pool_progression.png"); plt.close(fig)

# =====================================================================
# FIG 3 — The "split of effects": why v14b needed BOTH levers
#   grouped bars over 3 capability axes for v12 / v14 / v14b
# =====================================================================
fig, ax = plt.subplots(figsize=(9.5, 5.4))
groups = ["Imitation-catching\n(held-out gen pool AUC)",
          "Content-invariance\n(PAN cross-topic AUC)",
          "Email in-domain\n(register:cross AUC)"]
def gpan(v): return slice_auc(ood[v], "domain:pan20_xtopic")
def greg(v): return slice_auc(ood[v], "register:cross")
data = {
    "v12 multi-gen":      [prow(abl["v12"], SCORER)["auc"], gpan("v12"), greg("v12")],
    "v14 identity-only":  [prow(abl["v14"], SCORER)["auc"], gpan("v14"), greg("v14")],
    "v14b SYNTHESIS":     [prow(abl["v14b"], SCORER)["auc"], gpan("v14b"), greg("v14b")],
}
# v12 has no PAN measurement -> show as None (gap)
data["v12 multi-gen"][1] = gpan("v12")  # None
xg = np.arange(len(groups)); ww = 0.26
cols = {"v12 multi-gen": "#4285f4", "v14 identity-only": "#f4a142", "v14b SYNTHESIS": "#34a853"}
for i, (lab, vals) in enumerate(data.items()):
    vals = [np.nan if v is None else v for v in vals]
    bars = ax.bar(xg + (i-1)*ww, vals, ww, label=lab, color=cols[lab])
    barlabels(ax, bars, dy=0.01)
ax.set_xticks(xg); ax.set_xticklabels(groups)
ax.set_ylabel("AUC"); ax.set_ylim(0, 1.05)
ax.set_title("Why the synthesis was needed: each prior model wins only ONE axis\n"
             "v14 buys content-invariance but loses imitation-catching; v14b keeps both",
             fontsize=12)
ax.legend(loc="lower left", ncol=1, fontsize=9)
ax.annotate("v12 PAN\nnot measured", xy=(1 - ww, 0.02), xytext=(1 - ww, 0.30),
            ha="center", fontsize=8, color="#4285f4",
            arrowprops=dict(arrowstyle="->", color="#4285f4"))
fig.tight_layout(); fig.savefig(OUT / "fig3_split_of_effects.png"); plt.close(fig)

# =====================================================================
# FIG 4 — Confusion matrices at the operating point (5% genuine FPR)
#   3-class: Genuine / LLM-forgery / Wrong-human, v12 vs v14b
#   counts from probe; rates from eval (TPR@5%, fpr_other@5, 5% genuine FPR)
# =====================================================================
def confusion(d):
    """Return 3x2 count matrix [Accept, Flag] for rows Genuine/Forgery/Wrong-human."""
    probe = d["probe"]; r = prow(d, SCORER)
    ng, nf, no = probe["n_genuine"], probe["n_synthetic"], probe["n_other"]
    fpr_g = 0.05                       # operating point: 5% genuine flagged by construction
    tpr_f = r["tpr5"]                  # forgeries flagged
    acc_o = r["fpr_other_at_5"]        # wrong-humans wrongly accepted
    M = np.array([
        [ng*(1-fpr_g), ng*fpr_g],     # Genuine: Accept(correct), Flag(false alarm)
        [nf*(1-tpr_f), nf*tpr_f],     # LLM-forgery: Accept(MISS), Flag(caught)
        [no*acc_o,     no*(1-acc_o)], # Wrong-human: Accept(MISS), Flag(caught)
    ])
    return np.rint(M).astype(int)

def plot_cm(ax, M, title):
    rows = ["Genuine\nsender", "LLM\nforgery", "Wrong\nhuman"]
    cols = ["Accepted\n(allowed)", "Flagged\n(blocked)"]
    # color: green where the decision is correct, red where wrong
    correct = np.array([[1, 0], [0, 1], [0, 1]])  # 1 = good cell
    norm = M / M.sum(axis=1, keepdims=True)
    rgba = np.zeros((3, 2, 3))
    for i in range(3):
        for j in range(2):
            shade = 0.25 + 0.6*norm[i, j]
            if correct[i, j]:
                rgba[i, j] = [1-shade*0.85, 1-shade*0.15, 1-shade*0.55]  # green-ish
            else:
                rgba[i, j] = [1-shade*0.05, 1-shade*0.7, 1-shade*0.7]    # red-ish
    ax.imshow(rgba, aspect="auto")
    for i in range(3):
        for j in range(2):
            ax.text(j, i, f"{M[i,j]}\n({norm[i,j]*100:.0f}%)",
                    ha="center", va="center", fontsize=12, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(cols)
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(rows)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("model decision"); ax.grid(False)

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
plot_cm(axes[0], confusion(abl["v12"]),
        f"v12 (multi-gen)  —  {SCORER}\n5% false-alarm operating point")
plot_cm(axes[1], confusion(abl["v14b"]),
        f"v14b (SYNTHESIS)  —  {SCORER}\n5% false-alarm operating point")
fig.suptitle("Confusion at deployment threshold — held-out Claude+Gemini eval\n"
             "green = correct decision, red = error", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT / "fig4_confusion_v12_vs_v14b.png"); plt.close(fig)

# =====================================================================
# FIG 5 — Guardrail: forgeries caught vs wrong-humans wrongly accepted
#   shows v14b deploy choice (mahalanobis) vs the Goodharted linear scorer
# =====================================================================
fig, ax = plt.subplots(figsize=(9, 5.2))
mah = prow(abl["v14b"], "mahalanobis")
lin = prow(abl["v14b"], "baseline_linear_z3")
labels = ["mahalanobis\n(DEPLOY)", "baseline_linear_z3\n(Goodharted)"]
tpr5 = [mah["tpr5"], lin["tpr5"]]
fpo = [mah["fpr_other_at_5"], lin["fpr_other_at_5"]]
xx = np.arange(2)
b1 = ax.bar(xx - w/2, tpr5, w, label="Forgeries caught (TPR@5%) — higher better", color="#34a853")
b2 = ax.bar(xx + w/2, fpo, w, label="Wrong-humans wrongly accepted (FPR_other) — lower better", color="#ea4335")
barlabels(ax, b1); barlabels(ax, b2)
ax.axhline(0.10, ls="--", c="#ea4335", lw=1)
ax.text(1.4, 0.11, "0.10 guardrail", color="#ea4335", fontsize=9, ha="right")
ax.set_xticks(xx); ax.set_xticklabels(labels)
ax.set_ylim(0, 1.0); ax.set_ylabel("rate")
ax.set_title("v14b scorer choice: mahalanobis satisfies the wrong-human guardrail\n"
             "linear scorer's high TPR is Goodharted — it leaks 30% of impostor humans",
             fontsize=12)
ax.legend(loc="center", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "fig5_guardrail_scorer_choice.png"); plt.close(fig)

print("Wrote figures to", OUT)
for p in sorted(OUT.glob("*.png")):
    print("  ", p.relative_to(ROOT))
