"""One-off: build the figures for docs/v11_results_analysis_memo.md.

Parses each arm's W&B output.log (epoch-indexed centroid-probe trajectories)
and the final-epoch wandb-summary.json, then renders three figures into
results/v11/figures/. Pure read-only on the run dirs.
"""
import json
import re
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path("/workspace/stylometry")
RUNS = ROOT / "runs/v11"
OUT = ROOT / "results/v11/figures"
OUT.mkdir(parents=True, exist_ok=True)

ARMS = ["frozen", "frozen_supcon", "lora", "lora_supcon", "detector"]

EPOCH_RE = re.compile(r"Epoch (\d+)/\d+")
AUC_RE = re.compile(r"centroid auc\s+vs_other=([\d.]+)\s+vs_syn=([\d.]+)\s+vs_all=([\d.]+)")
GAP_RE = re.compile(r"gaps\s+other=([+\-\d.]+)\s+syn=([+\-\d.]+)\s+harder=([+\-\d.]+)")
BEST_RE = re.compile(r"New best pauc/genuine_vs_synthetic_5pct=([\d.]+).*?at epoch (\d+)")


def logpath(arm):
    runs = sorted((RUNS / arm / "wandb").glob("run-*"))
    if not runs:
        return None
    p = runs[-1] / "files/output.log"
    return p if p.exists() else None


def parse(arm):
    p = logpath(arm)
    if p is None:
        return None
    epoch = 0
    rows = []          # (epoch, vs_other, vs_syn, vs_all)
    gaps = []          # (epoch, harder)
    best = []          # (epoch, monitor)
    for line in p.read_text(errors="ignore").splitlines():
        m = EPOCH_RE.search(line)
        if m:
            epoch = int(m.group(1))
        m = AUC_RE.search(line)
        if m:
            rows.append((epoch, float(m.group(1)), float(m.group(2)), float(m.group(3))))
        m = GAP_RE.search(line)
        if m:
            gaps.append((epoch, float(m.group(3))))
        m = BEST_RE.search(line)
        if m:
            best.append((int(m.group(2)), float(m.group(1))))
    return {"rows": np.array(rows), "gaps": np.array(gaps), "best": np.array(best)}


def summary(arm):
    p = logpath(arm)
    if p is None:
        return {}
    sp = p.parent / "wandb-summary.json"
    return json.loads(sp.read_text()) if sp.exists() else {}


data = {a: parse(a) for a in ARMS}
summ = {a: summary(a) for a in ARMS}

# ---------------------------------------------------------------- Figure 1
# Centroid-AUC trajectories: vs_syn peaks early & decays while vs_other climbs.
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
for ax, arm, title in [
    (axes[0], "detector", "detector arm (BCE head + 0.3 SupCon)"),
    (axes[1], "lora", "lora arm (episodic, full V9 recipe)"),
]:
    d = data[arm]
    if d is None or not len(d["rows"]):
        continue
    e, o, s, a = d["rows"].T
    ax.plot(e, s, color="tab:red", lw=2, label="genuine vs synthetic (LLM)")
    ax.plot(e, o, color="tab:blue", lw=2, label="genuine vs other (wrong human)")
    ax.plot(e, a, color="tab:gray", lw=1.3, ls="--", label="genuine vs all (pooled)")
    if len(d["best"]):
        be = d["best"][-1, 0]
        ax.axvline(be, color="k", ls=":", lw=1)
        ax.text(be + 2, 0.55, f"checkpoint_best\nep {int(be)}", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
    ax.set_ylim(0.55, 1.0)
axes[0].set_ylabel("centroid-probe AUC")
axes[0].legend(loc="lower right", fontsize=8)
fig.suptitle(
    "Fig 1 — The two axes separate during training: synthetic separability "
    "peaks early then decays;\nwrong-human (authorship) separability climbs "
    "slowly. The monitor's checkpoint_best lands on the synthetic peak.",
    fontsize=10,
)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUT / "fig1_training_dynamics.png", dpi=130)
plt.close(fig)

# ---------------------------------------------------------------- Figure 2
# synthetic_harder_than_other over epochs vs the v6 baseline sign.
fig, ax = plt.subplots(figsize=(9, 4.8))
colors = {"detector": "tab:red", "lora": "tab:green", "frozen": "tab:purple"}
for arm, c in colors.items():
    d = data[arm]
    if d is None or not len(d["gaps"]):
        continue
    e, h = d["gaps"].T
    ax.plot(e, h, color=c, lw=1.8, label=f"v11 {arm}")
ax.axhline(0.0, color="k", lw=1)
ax.axhline(0.0832, color="tab:orange", ls="--", lw=1.5,
           label="v6 final (+0.083): synthetic IS the harder tail")
ax.text(75, 0.095, "synthetic harder than other  (v6 regime)", fontsize=8, color="tab:orange")
ax.text(75, -0.36, "other (wrong-human) is the harder tail  (v11 regime)", fontsize=8, color="dimgray")
ax.set_xlabel("epoch")
ax.set_ylabel("score/synthetic_harder_than_other\n= mean_synthetic − mean_other")
ax.set_title(
    "Fig 2 — Sign flip: v11 trains synthetics into the easy tail, so wrong-human\n"
    "impostors (not LLM text) become the binding negative. v6 was the opposite.",
    fontsize=10,
)
ax.legend(loc="center right", fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "fig2_harder_than_other.png", dpi=130)
plt.close(fig)

# ---------------------------------------------------------------- Figure 3
# Final-epoch (ep150) own-corpus separability: arms + v6. pAUC@5% panel + TPR@1%.
def g(arm, key):
    return summ.get(arm, {}).get(key, np.nan)

V6 = {  # v6 own-corpus epoch-100 summary (different corpus — see caveat in memo)
    "pauc_syn": 0.6358, "pauc_other": np.nan, "pauc_all": 0.6842,
    "tpr_syn1": 0.55, "tpr_all1": 0.6667,
}
labels = ["v6 (ep100)", "frozen", "frozen_supcon", "lora", "lora_supcon", "detector"]
arms_only = ["frozen", "frozen_supcon", "lora", "lora_supcon", "detector"]

pauc_syn = [V6["pauc_syn"]] + [g(a, "pauc/genuine_vs_synthetic_5pct") for a in arms_only]
pauc_other = [V6["pauc_other"]] + [g(a, "pauc/genuine_vs_other_5pct") for a in arms_only]
pauc_all = [V6["pauc_all"]] + [g(a, "pauc/genuine_vs_all_5pct") for a in arms_only]
tpr_syn1 = [V6["tpr_syn1"]] + [g(a, "tpr_at_fpr/synthetic_1pct") for a in arms_only]
tpr_all1 = [V6["tpr_all1"]] + [g(a, "tpr_at_fpr/all_1pct") for a in arms_only]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(len(labels))
w = 0.26
ax = axes[0]
ax.bar(x - w, pauc_syn, w, label="vs synthetic", color="tab:red")
ax.bar(x, pauc_other, w, label="vs other (wrong human)", color="tab:blue")
ax.bar(x + w, pauc_all, w, label="vs all (pooled)", color="tab:gray")
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
ax.set_ylabel("pAUC @ 5% FPR"); ax.set_ylim(0, 1)
ax.set_title("pAUC@5% by negative type", fontsize=10); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

ax = axes[1]
ax.bar(x - w/2, tpr_syn1, w, label="TPR@1%FPR synthetic", color="tab:red")
ax.bar(x + w/2, tpr_all1, w, label="TPR@1%FPR all (pooled)", color="tab:gray")
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
ax.set_ylabel("TPR @ 1% FPR"); ax.set_ylim(0, 1)
ax.set_title("Detection at a 1% false-alarm budget", fontsize=10); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

fig.suptitle(
    "Fig 3 — Final-epoch (ep150) own-corpus separability. Synthetic separation is near-saturated "
    "everywhere;\nthe pooled metric is gated by the wrong-human tail, which only the LoRA arms hold up. "
    "(v6 = different corpus; rough anchor only.)",
    fontsize=10,
)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(OUT / "fig3_final_comparison.png", dpi=130)
plt.close(fig)

# ---------------------------------------------------------------- console dump
print("figures ->", OUT)
for a in ARMS:
    d = data[a]
    if d is None:
        print(f"{a:14} NO LOG")
        continue
    nb = d["best"][-1] if len(d["best"]) else (np.nan, np.nan)
    print(f"{a:14} epochs_parsed={len(d['rows']):3d}  best_monitor ep{int(nb[0]) if nb[0]==nb[0] else -1}={nb[1]:.4f}"
          f"  syn5%={g(a,'pauc/genuine_vs_synthetic_5pct'):.3f}  other5%={g(a,'pauc/genuine_vs_other_5pct'):.3f}"
          f"  all5%={g(a,'pauc/genuine_vs_all_5pct'):.3f}  harder={g(a,'score/synthetic_harder_than_other'):+.3f}")
