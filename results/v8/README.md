# v8 overnight run — syn-v1 vs syn-v2 synthetic-data A/B

This directory holds the artifacts from the v8 overnight run
(`runs/_v8_overnight/`), which executed the full **gen → train → probe →
ablate** pipeline for two arms that differ in **exactly one variable**: the
synthetic-data generation recipe.

| Arm | `--cross-register-fraction` | Composition | Role |
|-----|-----------------------------|-------------|------|
| **syn-v1** | 0.0 | 100% hard negatives (438 rows) | control (pre-CHANGELOG behaviour) |
| **syn-v2** | 0.4 | 60% hard negatives (394) + 40% cross-register positives (264), expanded topic pools + quality filter | treatment |

Everything else (LUAR-MUD + LoRA r16, SupCon τ=0.05, prototypical `adaptive_k`
head, 150 epochs) is identical across the two arms, so any difference is
attributable to the dataset change alone.

> Note: the underlying configs live at `configs/experiments/v7_synv1.yaml` and
> `v7_synv2.yaml`, and the run also wrote copies into `results/v7/`. The files
> here are the canonical v8 copies.

## Files

| File | What it is |
|------|------------|
| `probe_synv1.json` / `probe_synv2.json` | Authenticity-probe metrics (frozen + finetune) on the sender-disjoint test split. Contains TP/FP/TN/FN, accuracy, ROC/PR-AUC, precision, recall. Positive class = **synthetic**. |
| `scorer_ablation_synv1.{json,csv}` / `scorer_ablation_synv2.{json,csv}` | Adaptive-scorer ablation: 9 scorers ranked on **TPR@1% FPR** (`tpr1`), with AUC, pAUC5, TPR@5%, 1−EER, and bootstrapped Δ-vs-baseline 95% CIs (1000 resamples, K_enroll=8, 30 senders). Includes the K-sweep over K∈{4,8,16,25}. |
| `figures/v8_confusion.png` | Confusion-matrix figure (see below). |
| `figures/make_confusion.py` | Self-contained script that regenerates the figure from the `probe_*.json` files in this directory. Run: `python figures/make_confusion.py`. |

## Figures

### `figures/v8_confusion.png`

A 2×2 grid of confusion matrices: rows = arm (syn-v1 control / syn-v2
treatment), columns = probe type (**frozen** linear probe / **finetune** probe).
Positive class = **synthetic** (the thing we want to catch), so:

- **TN** (top-left) = genuine correctly kept, **FP** (top-right) = genuine wrongly flagged as synthetic
- **FN** (bottom-left) = synthetic missed, **TP** (bottom-right) = synthetic caught

Each panel title shows accuracy, ROC-AUC, precision, and recall. How to read it:

- **All error is false-positive** (FN ≈ 0 everywhere): the detector never misses
  a synthetic, it only occasionally over-flags a genuine email.
- **Frozen beats finetune in both arms** (1→17 FP for v1, 5→21 FP for v2):
  fine-tuning the probe head overfits the boundary and inflates false positives.
  The frozen LUAR+LoRA embedding is already near-perfectly separable — **ship the
  frozen probe.**
- syn-v2's frozen probe shows a few more FP (5 vs 1) and one FN — the expected,
  small cost of a harder/more realistic dataset and a 50%-larger test set
  (390 vs 260). The payoff shows up in the ablation, not the probe.

## Headline result

The linear probe is near-saturated (AUC ≈ 1.0) for both arms, so the real signal
is in the strict-operating-point metric from the ablation:

| `baseline_linear_z3`, K=8 | syn-v1 | syn-v2 | Δ |
|---|---|---|---|
| **TPR@1% FPR (`tpr1`)** | 0.442 | **0.575** | **+0.133** |
| AUC | 0.941 | 0.947 | +0.006 |
| 1−EER | 0.675 | 0.750 | +0.075 |

Cross-register positives (syn-v2) lift detection at a 1% false-positive budget
from 44% → 58% while AUC stays flat — i.e. the gain lives entirely in the hard
tail, exactly where teaching the model that legitimate register-shift ≠ forgery
should help. `baseline_linear_z3` remains the best scorer in both arms (no
alternative beats it with a CI excluding 0). **syn-v2 is the better dataset.**
