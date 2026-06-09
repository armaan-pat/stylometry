# Results digest

Scanned `results` — 18 JSON files, 152 scorer rows parsed.

## 1. Best operating points found (genuine-vs-synthetic)

- **Best AUC[g/syn]: 0.966** — `mahal_per_sender` @ K=16 (AUC[g/syn]=0.966, TPR@5%=0.875, TPR@1%=0.792) — _v7_3_k_sweep_
- **Best TPR@5%FPR_syn: 0.917** — `linear_z3` @ K=25 (AUC[g/syn]=0.956, TPR@5%=0.917, TPR@1%=0.875) — _v7_3_k_sweep_
- **Best TPR@1%FPR_syn: 0.875** — `linear_z3` @ K=25 (AUC[g/syn]=0.956, TPR@5%=0.917, TPR@1%=0.875) — _v7_3_k_sweep_

## 2. Per-experiment leaderboard (best scorer in each file)

| experiment | K | best scorer | AUC[g/syn] | TPR@5% | TPR@1% | EER | source |
|---|---|---|---|---|---|---|---|
| ksweep_ci_synv2_K16 | 16 | `mahalanobis` | 0.908 | 0.667 | 0.633 |   -   | results/v8/ksweep_ci_synv2_K16.json |
| ksweep_ci_synv2_K25 | 25 | `ewma_centroid_z3` | 0.897 | 0.700 | 0.600 |   -   | results/v8/ksweep_ci_synv2_K25.json |
| ksweep_ci_synv2_K4 | 4 | `z_global_cal` | 0.858 | 0.575 | 0.500 |   -   | results/v8/ksweep_ci_synv2_K4.json |
| ksweep_ci_synv2_K8 | 8 | `z_global_cal` | 0.859 | 0.650 | 0.458 |   -   | results/v8/ksweep_ci_synv2_K8.json |
| scorer_ablation_synv1 | 8 | `mahalanobis` | 0.949 | 0.692 | 0.483 |   -   | results/v7/scorer_ablation_synv1.json |
| scorer_ablation_synv2 | 8 | `mahalanobis` | 0.957 | 0.858 | 0.575 |   -   | results/v7/scorer_ablation_synv2.json |
| v7_0_scoring_sweep | - | `mahal_per_sender` | 0.875 | 0.733 | 0.533 | 0.177 | results/v7/v7_0_scoring_sweep.json |
| v7_1_hybrid | - | `hybrid_alpha_0.3` | 0.836 | 0.483 | 0.375 | 0.266 | results/v7/v7_1_hybrid.json |
| v7_2_k_sweep | 40 | `linear_z3` | 0.923 | 0.775 | 0.775 | 0.134 | results/v7/v7_2_k_sweep.json |
| v7_3_k_sweep | 25 | `linear_z3` | 0.956 | 0.917 | 0.875 | 0.070 | results/v7/v7_3_k_sweep.json |
| v7_3_scoring_sweep | - | `cosine_snorm` | 0.935 | 0.700 | 0.517 | 0.157 | results/v7/v7_3_scoring_sweep.json |
| v7_3_scoring_sweep_ep150 | - | `linear_z3_median` | 0.893 | 0.767 | 0.675 | 0.182 | results/v7/v7_3_scoring_sweep_ep150.json |

## 3. Enrollment-K scaling (best scorer per K)

| experiment | K | best scorer | AUC[g/syn] | TPR@5% | TPR@1% |
|---|---|---|---|---|---|
| ksweep_ci_synv2_K16 | 16 | `mahalanobis` | 0.908 | 0.667 | 0.633 |
| ksweep_ci_synv2_K25 | 25 | `ewma_centroid_z3` | 0.897 | 0.700 | 0.600 |
| ksweep_ci_synv2_K4 | 4 | `z_global_cal` | 0.858 | 0.575 | 0.500 |
| ksweep_ci_synv2_K8 | 8 | `z_global_cal` | 0.859 | 0.650 | 0.458 |
| scorer_ablation_synv1 | 8 | `mahalanobis` | 0.949 | 0.692 | 0.483 |
| scorer_ablation_synv2 | 8 | `mahalanobis` | 0.957 | 0.858 | 0.575 |
| v7_2_k_sweep | 4 | `cosine` | 0.896 | 0.700 | 0.533 |
| v7_2_k_sweep | 8 | `mahal_per_sender` | 0.875 | 0.733 | 0.533 |
| v7_2_k_sweep | 16 | `mahal_per_sender` | 0.942 | 0.825 | 0.708 |
| v7_2_k_sweep | 25 | `mahal_per_sender` | 0.938 | 0.850 | 0.758 |
| v7_2_k_sweep | 40 | `linear_z3` | 0.923 | 0.775 | 0.775 |
| v7_3_k_sweep | 4 | `cosine` | 0.941 | 0.750 | 0.650 |
| v7_3_k_sweep | 8 | `mahal_per_sender` | 0.904 | 0.783 | 0.658 |
| v7_3_k_sweep | 16 | `mahal_per_sender` | 0.966 | 0.875 | 0.792 |
| v7_3_k_sweep | 25 | `linear_z3` | 0.956 | 0.917 | 0.875 |
| v7_3_k_sweep | 40 | `mahal_per_sender` | 0.957 | 0.858 | 0.825 |

## 4. Adaptive-scorer ablation verdicts (bootstrapped)

- **results/v7/scorer_ablation_synv1.json**
  - mean Ledoit-Wolf shrinkage α = 0.648
  - winner: **baseline_linear_z3** (rank by tpr1)
  - _keep 'baseline_linear_z3' — top scorer 'mahalanobis' leads on tpr1 point estimate (Δ=+0.042) but its Δ 95% CI [-0.067,+0.167] includes 0 (P(win)=0.74) — not distinguishable from noise._
- **results/v7/scorer_ablation_synv2.json**
  - mean Ledoit-Wolf shrinkage α = 0.652
  - winner: **baseline_linear_z3** (rank by tpr1)
  - _keep 'baseline_linear_z3' — it ranks #1 on tpr1 and no other scorer significantly beats it._
- **results/v8/ksweep_ci_synv2_K16.json**
  - mean Ledoit-Wolf shrinkage α = 0.519
  - winner: **baseline_linear_z3** (rank by tpr1)
  - _keep 'baseline_linear_z3' — top scorer 'mahalanobis' leads on tpr1 point estimate (Δ=+0.058) but its Δ 95% CI [-0.225,+0.184] includes 0 (P(win)=0.72) — not distinguishable from noise._
- **results/v8/ksweep_ci_synv2_K25.json**
  - mean Ledoit-Wolf shrinkage α = 0.451
  - winner: **baseline_linear_z3** (rank by tpr1)
  - _keep 'baseline_linear_z3' — top scorer 'ewma_centroid_z3' leads on tpr1 point estimate (Δ=+0.017) but its Δ 95% CI [-0.042,+0.100] includes 0 (P(win)=0.54) — not distinguishable from noise._
- **results/v8/ksweep_ci_synv2_K4.json**
  - mean Ledoit-Wolf shrinkage α = 0.412
  - winner: **baseline_linear_z3** (rank by tpr1)
  - _keep 'baseline_linear_z3' — it ranks #1 on tpr1 and no other scorer significantly beats it._
- **results/v8/ksweep_ci_synv2_K8.json**
  - mean Ledoit-Wolf shrinkage α = 0.541
  - winner: **baseline_linear_z3** (rank by tpr1)
  - _keep 'baseline_linear_z3' — it ranks #1 on tpr1 and no other scorer significantly beats it._
- **results/v8/scorer_ablation_synv1.json**
  - mean Ledoit-Wolf shrinkage α = 0.648
  - winner: **baseline_linear_z3** (rank by tpr1)
  - _keep 'baseline_linear_z3' — top scorer 'mahalanobis' leads on tpr1 point estimate (Δ=+0.042) but its Δ 95% CI [-0.067,+0.167] includes 0 (P(win)=0.74) — not distinguishable from noise._
- **results/v8/scorer_ablation_synv2.json**
  - mean Ledoit-Wolf shrinkage α = 0.652
  - winner: **baseline_linear_z3** (rank by tpr1)
  - _keep 'baseline_linear_z3' — it ranks #1 on tpr1 and no other scorer significantly beats it._

## 5. Authenticity probes (genuine-vs-synthetic classifier; +=synthetic)

| source | mode | ROC-AUC | acc | precision | recall | FP | FN | n |
|---|---|---|---|---|---|---|---|---|
| results/v7/probe_synv1.json | frozen | 1.000 | 0.996 | 0.992 | 1.000 | 1 | 0 | 260 |
| results/v7/probe_synv1.json | finetune | 0.999 | 0.935 | 0.884 | 1.000 | 17 | 0 | 260 |
| results/v7/probe_synv2.json | frozen | 1.000 | 0.985 | 0.975 | 0.995 | 5 | 1 | 390 |
| results/v7/probe_synv2.json | finetune | 0.999 | 0.946 | 0.903 | 1.000 | 21 | 0 | 390 |
| results/v8/probe_synv1.json | frozen | 1.000 | 0.996 | 0.992 | 1.000 | 1 | 0 | 260 |
| results/v8/probe_synv1.json | finetune | 0.999 | 0.935 | 0.884 | 1.000 | 17 | 0 | 260 |
| results/v8/probe_synv2.json | frozen | 1.000 | 0.985 | 0.975 | 0.995 | 5 | 1 | 390 |
| results/v8/probe_synv2.json | finetune | 0.999 | 0.946 | 0.903 | 1.000 | 21 | 0 | 390 |
