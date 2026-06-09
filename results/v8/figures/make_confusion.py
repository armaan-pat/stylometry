import json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base = "/workspace/stylometry/results/v8"
runs = {
    "syn-v1 (control)": json.load(open(f"{base}/probe_synv1.json")),
    "syn-v2 (treatment)": json.load(open(f"{base}/probe_synv2.json")),
}
probes = ["frozen", "finetune"]

fig, axes = plt.subplots(2, 2, figsize=(10, 9))
for r, (rname, d) in enumerate(runs.items()):
    for c, p in enumerate(probes):
        tp = d[f"probe_{p}/tp"]; fp = d[f"probe_{p}/fp"]
        tn = d[f"probe_{p}/tn"]; fn = d[f"probe_{p}/fn"]
        # rows = actual (genuine, synthetic), cols = predicted (genuine, synthetic)
        # positive class = synthetic
        cm = np.array([[tn, fp],
                       [fn, tp]])
        ax = axes[r][c]
        im = ax.imshow(cm, cmap="Blues")
        acc = d[f"probe_{p}/accuracy"]; auc = d[f"probe_{p}/roc_auc"]
        prec = d[f"probe_{p}/precision"]; rec = d[f"probe_{p}/recall"]
        ax.set_title(f"{rname} — {p}\nacc={acc:.3f}  AUC={auc:.3f}  P={prec:.3f}  R={rec:.3f}",
                     fontsize=10)
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["pred genuine","pred synthetic"])
        ax.set_yticklabels(["actual genuine","actual synthetic"])
        labels = [["TN","FP"],["FN","TP"]]
        for i in range(2):
            for j in range(2):
                v = cm[i,j]
                ax.text(j, i, f"{labels[i][j]}\n{int(v)}", ha="center", va="center",
                        color="white" if v > cm.max()/2 else "black", fontsize=12,
                        fontweight="bold")
fig.suptitle("v8 overnight — authenticity probe confusion (positive class = synthetic)",
             fontsize=13, y=0.995)
fig.tight_layout()
out = "/workspace/stylometry/results/v8/figures/v8_confusion.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
