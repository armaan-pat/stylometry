"""Compare runs for a given experiment using per-epoch output.log data.

Produces:
  - <experiment>_curves.png        — per-epoch line graphs for all logged metrics
  - <experiment>_comparison.png    — bar charts of final-epoch values
  - <experiment>_confusion.png     — confusion matrices at τ=0.50

Usage:
    python scripts/plot_runs.py --experiment v6_luar_lora_syn
    python scripts/plot_runs.py --experiment v6_luar_lora_syn --out-dir results/
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_summaries(experiment: str) -> dict[str, dict]:
    runs_dir = _PROJECT_ROOT / "runs" / experiment
    summaries: dict[str, dict] = {}
    for run_dir in sorted(runs_dir.iterdir()):
        files = list(run_dir.glob("wandb/run-*/files/wandb-summary.json"))
        if not files:
            continue
        with files[0].open() as fh:
            summaries[run_dir.name] = json.load(fh)
    if not summaries:
        raise ValueError(f"No wandb summaries found under {runs_dir}")
    return summaries


def load_histories(experiment: str) -> dict[str, dict[str, list]]:
    """Parse output.log files into per-epoch metric dicts."""
    runs_dir = _PROJECT_ROOT / "runs" / experiment
    histories: dict[str, dict[str, list]] = {}
    for run_dir in sorted(runs_dir.iterdir()):
        logs = list(run_dir.glob("wandb/run-*/files/output.log"))
        if not logs:
            continue
        histories[run_dir.name] = _parse_log(logs[0])
    return histories


# Regex patterns matching _format_epoch_summary output lines.
_RE_EPOCH = re.compile(
    r"Epoch\s+(\d+)/\d+\s+loss train=([0-9.]+)\s+val=([0-9.]+)"
    r".*pair_auc=([0-9.]+).*knn_f1=([0-9.]+).*knn_acc=([0-9.]+)"
)
_RE_CENTROID = re.compile(
    r"centroid auc\s+vs_other=([0-9.]+)\s+vs_syn=([0-9.]+)\s+vs_all=([0-9.]+)"
)
_RE_GAPS = re.compile(
    r"gaps\s+other=([+-][0-9.]+)\s+syn=([+-][0-9.]+)\s+harder=([+-][0-9.]+)"
)
_RE_THRESHOLD = re.compile(
    r"@0\.95\s+prec=([0-9.]+)\s+rec=([0-9.]+)\s+fpr_syn=([0-9.]+)\s+cov@acc=([0-9.]+)"
)
_RE_PAN = re.compile(
    r"test \(PAN\)\s+auc=([0-9.]+)\s+eer=([0-9.]+)"
)


def _parse_log(path: Path) -> dict[str, list]:
    data: dict[str, list] = {
        "epoch": [], "train_loss": [], "val_loss": [],
        "pair_auc": [], "knn_f1": [], "knn_acc": [],
        "vs_other": [], "vs_syn": [], "vs_all": [],
        "gap_other": [], "gap_syn": [], "harder": [],
        "prec_95": [], "rec_95": [], "fpr_syn_95": [], "cov_95": [],
        "test_auc": [], "test_auc_epoch": [],
        "test_eer": [],
    }
    current_epoch: int | None = None

    with path.open(errors="replace") as fh:
        for line in fh:
            m = _RE_EPOCH.search(line)
            if m:
                current_epoch = int(m.group(1))
                data["epoch"].append(current_epoch)
                data["train_loss"].append(float(m.group(2)))
                data["val_loss"].append(float(m.group(3)))
                data["pair_auc"].append(float(m.group(4)))
                data["knn_f1"].append(float(m.group(5)))
                data["knn_acc"].append(float(m.group(6)))
                continue

            m = _RE_CENTROID.search(line)
            if m:
                data["vs_other"].append(float(m.group(1)))
                data["vs_syn"].append(float(m.group(2)))
                data["vs_all"].append(float(m.group(3)))
                continue

            m = _RE_GAPS.search(line)
            if m:
                data["gap_other"].append(float(m.group(1)))
                data["gap_syn"].append(float(m.group(2)))
                data["harder"].append(float(m.group(3)))
                continue

            m = _RE_THRESHOLD.search(line)
            if m:
                data["prec_95"].append(float(m.group(1)))
                data["rec_95"].append(float(m.group(2)))
                data["fpr_syn_95"].append(float(m.group(3)))
                data["cov_95"].append(float(m.group(4)))
                continue

            m = _RE_PAN.search(line)
            if m and current_epoch is not None:
                data["test_auc"].append(float(m.group(1)))
                data["test_auc_epoch"].append(current_epoch)
                data["test_eer"].append(float(m.group(2)))

    return data


# ---------------------------------------------------------------------------
# Line graphs
# ---------------------------------------------------------------------------

def plot_curves(histories: dict[str, dict[str, list]], out: Path) -> None:
    if not histories:
        print("No output.log files found — skipping curve plots.")
        return

    panels = [
        ("Loss",                  [("train_loss", "Train Loss"), ("val_loss", "Val Loss")],          "epoch"),
        ("Embedding Quality",     [("pair_auc", "Pair AUROC"), ("knn_f1", "KNN Macro-F1"), ("knn_acc", "KNN Acc")], "epoch"),
        ("Centroid AUROC",        [("vs_syn", "vs Synthetic"), ("vs_other", "vs Other"), ("vs_all", "vs All")],      "epoch"),
        ("Score Gaps",            [("gap_syn", "Genuine−Syn"), ("gap_other", "Genuine−Other"), ("harder", "Syn Harder Than Other")], "epoch"),
        ("@0.95 Operating Point", [("prec_95", "Precision"), ("rec_95", "Recall"), ("fpr_syn_95", "FPR Synthetic"), ("cov_95", "Coverage@Acc")], "epoch"),
        ("Test AUC (every 5ep)", [("test_auc", "AUC"), ("test_eer", "EER ↓")],                      "test_auc_epoch"),
    ]

    colors = plt.cm.tab10(np.linspace(0, 0.5, len(histories)))
    linestyles = ["-", "--", ":", "-."]

    fig, axes = plt.subplots(len(panels), 1, figsize=(13, 4 * len(panels)))
    fig.suptitle("Training Curves", fontsize=14, fontweight="bold")

    for ax, (title, series, x_key) in zip(axes, panels):
        for run_idx, (run, hist) in enumerate(histories.items()):
            x = hist.get(x_key, [])
            if not x:
                continue
            for s_idx, (key, label) in enumerate(series):
                y = hist.get(key, [])
                if not y or len(y) != len(x):
                    continue
                ax.plot(x, y,
                        color=colors[run_idx],
                        linestyle=linestyles[s_idx % len(linestyles)],
                        linewidth=1.5,
                        label=f"{run} — {label}")

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="best", ncol=2)

    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved to {out}")


# ---------------------------------------------------------------------------
# Bar charts (final-epoch summary)
# ---------------------------------------------------------------------------

def _get(summary: dict, key: str) -> float | None:
    val = summary.get(key)
    return float(val) if val is not None else None


def plot_bars(summaries: dict[str, dict], out: Path) -> None:
    run_labels = list(summaries.keys())
    colors = plt.cm.tab10(np.linspace(0, 0.4, len(run_labels)))

    groups = [
        {
            "title": "Centroid Probe — AUROC",
            "metrics": [
                ("auc/genuine_vs_synthetic", "vs Synthetic"),
                ("auc/genuine_vs_other",     "vs Other"),
                ("auc/genuine_vs_all",        "vs All"),
            ],
        },
        {
            "title": "Centroid Probe — pAUC (low-FPR)",
            "metrics": [
                ("pauc/genuine_vs_synthetic_5pct",  "Synthetic @5%"),
                ("pauc/genuine_vs_synthetic_10pct", "Synthetic @10%"),
                ("pauc/genuine_vs_all_5pct",         "All @5%"),
                ("pauc/genuine_vs_all_10pct",        "All @10%"),
            ],
        },
        {
            "title": "Centroid Probe — TPR @ Fixed FPR",
            "metrics": [
                ("tpr_at_fpr/synthetic_1pct", "Synthetic @FPR=1%"),
                ("tpr_at_fpr/synthetic_5pct", "Synthetic @FPR=5%"),
                ("tpr_at_fpr/all_1pct",        "All @FPR=1%"),
                ("tpr_at_fpr/all_5pct",        "All @FPR=5%"),
            ],
        },
        {
            "title": "Test Set — PAN Metrics",
            "metrics": [
                ("test/AUC",        "AUC"),
                ("test/pAUC@5%",    "pAUC@5%"),
                ("test/pAUC@10%",   "pAUC@10%"),
                ("test/TPR@FPR=1%", "TPR@FPR=1%"),
                ("test/TPR@FPR=5%", "TPR@FPR=5%"),
                ("test/EER",        "EER ↓"),
                ("test/c@1",        "c@1"),
                ("test/F0.5u",      "F0.5u"),
            ],
        },
        {
            "title": "Score Distribution",
            "metrics": [
                ("score/mean_genuine",   "Mean Genuine"),
                ("score/mean_synthetic", "Mean Synthetic"),
                ("score/mean_other",     "Mean Other"),
                ("score/gap_synthetic",  "Gap (Genuine−Syn)"),
                ("score/gap_other",      "Gap (Genuine−Other)"),
            ],
        },
        {
            "title": "Threshold τ=0.50 Operating Point",
            "metrics": [
                ("threshold_0.50/recall",        "Recall"),
                ("threshold_0.50/precision",      "Precision"),
                ("threshold_0.50/fpr_synthetic",  "FPR Synthetic"),
                ("threshold_0.50/fpr_other",      "FPR Other"),
                ("threshold_0.50/accuracy",       "Accuracy"),
            ],
        },
    ]

    fig, axes = plt.subplots(len(groups), 1, figsize=(14, 4 * len(groups)))
    fig.suptitle("Final-Epoch Comparison", fontsize=14, fontweight="bold")

    for ax, group in zip(axes, groups):
        keys   = [m[0] for m in group["metrics"]]
        labels = [m[1] for m in group["metrics"]]
        x = np.arange(len(keys))
        width = 0.8 / len(run_labels)

        for j, (run, summary) in enumerate(summaries.items()):
            for i, (k, v) in enumerate(zip(keys, [_get(summary, k) for k in keys])):
                offset = x[i] + j * width - (len(run_labels) - 1) * width / 2
                if v is not None:
                    bar = ax.bar(offset, v, width * 0.9, color=colors[j],
                                 label=run if i == 0 else None)
                    ax.text(bar[0].get_x() + bar[0].get_width() / 2,
                            bar[0].get_height() + 0.005,
                            f"{v:.2f}", ha="center", va="bottom", fontsize=7)
                else:
                    ax.bar(offset, 0.02, width * 0.9, color="lightgrey", hatch="//",
                           label=run if i == 0 else None)

        ax.set_title(group["title"], fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved to {out}")


# ---------------------------------------------------------------------------
# Confusion matrices
# ---------------------------------------------------------------------------

def _confusion_counts(summary: dict, task: str) -> np.ndarray | None:
    n_genuine = _get(summary, "probe/n_genuine_queries")
    recall    = _get(summary, "threshold_0.50/recall")
    if task == "synthetic":
        n_neg = _get(summary, "probe/n_synthetic_queries")
        fpr   = _get(summary, "threshold_0.50/fpr_synthetic")
    elif task == "other":
        n_neg = _get(summary, "probe/n_other_queries")
        fpr   = _get(summary, "threshold_0.50/fpr_other")
    else:
        n_syn = _get(summary, "probe/n_synthetic_queries")
        n_oth = _get(summary, "probe/n_other_queries")
        fpr   = _get(summary, "threshold_0.50/fpr_overall")
        n_neg = (n_syn or 0) + (n_oth or 0) if (n_syn is not None or n_oth is not None) else None
    if any(v is None for v in [n_genuine, recall, n_neg, fpr]):
        return None
    tp = round(recall * n_genuine)
    fn = round(n_genuine) - tp
    fp = round(fpr * n_neg)
    tn = round(n_neg) - fp
    return np.array([[tp, fn], [fp, tn]], dtype=int)


def plot_confusion(summaries: dict[str, dict], out: Path) -> None:
    tasks       = ["synthetic", "other", "all"]
    task_titles = ["Genuine vs Synthetic", "Genuine vs Other", "Genuine vs All"]
    row_labels  = ["Actual Genuine", "Actual Impostor"]
    col_labels  = ["Pred Genuine", "Pred Impostor"]

    # Latest run only for the focused confusion matrix plot
    latest_run  = list(summaries.keys())[-1]
    latest_only = {latest_run: summaries[latest_run]}

    _plot_confusion_grid(latest_only, tasks, task_titles, row_labels, col_labels,
                         out.with_stem(out.stem + "_latest"), title=f"Confusion Matrices — {latest_run} (τ=0.50)")
    _plot_confusion_grid(summaries, tasks, task_titles, row_labels, col_labels,
                         out, title="Confusion Matrices — All Runs (τ=0.50)")


def _plot_confusion_grid(
    summaries: dict[str, dict],
    tasks: list[str],
    task_titles: list[str],
    row_labels: list[str],
    col_labels: list[str],
    out: Path,
    title: str,
) -> None:
    n_runs  = len(summaries)
    fig, axes = plt.subplots(n_runs, 3, figsize=(12, 3.5 * n_runs), squeeze=False)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for row, (run, summary) in enumerate(summaries.items()):
        for col, (task, title) in enumerate(zip(tasks, task_titles)):
            ax  = axes[row][col]
            cm  = _confusion_counts(summary, task)
            if cm is None:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                ax.axis("off")
                ax.set_title(f"{title}\n{run}", fontsize=9)
                continue
            total = cm.sum()
            ax.imshow(cm / max(total, 1), cmap="Blues", vmin=0, vmax=1)
            for i in range(2):
                for j in range(2):
                    pct   = 100 * cm[i, j] / max(total, 1)
                    color = "white" if (cm[i, j] / max(total, 1)) > 0.5 else "black"
                    ax.text(j, i, f"{cm[i, j]}\n({pct:.1f}%)",
                            ha="center", va="center", fontsize=11,
                            fontweight="bold", color=color)
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(col_labels, fontsize=8)
            ax.set_yticklabels(row_labels, fontsize=8)
            ax.set_title(f"{title}\n{run}", fontsize=9, fontweight="bold")

    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved to {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="v6_luar_lora_syn")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else _PROJECT_ROOT / "results"
    exp     = args.experiment

    summaries = load_summaries(exp)
    histories = load_histories(exp)
    print(f"Found {len(summaries)} runs: {list(summaries.keys())}")

    plot_curves(histories, out_dir / f"{exp}_curves.png")
    plot_bars(summaries,   out_dir / f"{exp}_comparison.png")
    plot_confusion(summaries, out_dir / f"{exp}_confusion.png")


if __name__ == "__main__":
    main()
