# Experiment Status & Path Forward

*Consolidated landscape as of 2026-06-09. Single source of truth for "what's
been tested, what worked, what hasn't, and what to run next." Regenerate the
numeric tables any time with `python scripts/summarize_results.py`.*

**Objective:** flag emails that claim to be from a profiled sender but were not
written by them — BEC and LLM-generated impersonation ("spam/AI-generated
emails for a given user profile"). The system builds a per-sender style
centroid from enrolled emails and scores how far a new email sits from it.

---

## 1. TL;DR — the current best

- **Ship encoder:** V7.3 recipe (LUAR-MUD + LoRA r16 on q/v/k, SupCon τ=0.05,
  n_syn=4, 150 epochs) trained on the **syn-v2** synthetic dataset
  (cross-register positives). Checkpoint: `runs/v7_synv2/checkpoint_epoch_150.pt`
  (or `checkpoint_last.pt`). **Do not use `checkpoint_best.pt`** — the
  `auc/genuine_vs_synthetic` monitor peaks at ~epoch 7 and names the wrong file.
- **Ship scorer:** `linear_z3` for K < 16, `mahal_per_sender` (Ledoit-Wolf) for
  K ≥ 16. Bootstrapped ablation says nothing beats `linear_z3` with a CI
  excluding 0 at K=8, so `linear_z3` is the safe default; Mahalanobis only pulls
  clearly ahead once K ≥ 16.
- **Headline numbers (V7.3, real-email retention at synthetic-fraud FPR):**

  | Enrollment K | TPR@5% FPR_syn | TPR@1% FPR_syn | AUC[g/syn] | best scorer |
  |---|---|---|---|---|
  | 8  | 0.767 | 0.633 | 0.909 | linear_z3 |
  | 16 | 0.875 | 0.792 | 0.966 | mahal_per_sender |
  | 25 | **0.917** | **0.875** | 0.956 | linear_z3 |

- **Best synthetic-dataset arm:** **syn-v2 beats syn-v1.** At the strict
  operating point (K=8, `linear_z3`) cross-register positives lift TPR@1% FPR
  from **0.442 → 0.575 (+13.3 pp)** while AUC stays flat — the gain lives
  entirely in the hard low-FPR tail, exactly where it should.

---

## 2. Experiment lineage (what was tested)

| Series | What it tested | Verdict |
|---|---|---|
| **V2** | Backbone bake-off: LUAR vs RoBERTa vs MPNet, LoRA vs frozen | **LUAR-MUD wins by a mile** (val AUROC 0.956 vs ~0.53). RoBERTa/MPNet encode topic, not style. LoRA edges frozen for LUAR (+2 pp). |
| **V3–V5** | LoRA/synthetic iterations on LUAR & RoBERTa | Converged on LUAR+LoRA+synthetic as the line to push. |
| **V6** | LUAR+LoRA+synthetic baseline (n_syn=2, τ=0.07, q/v LoRA, 100 ep) | Baseline: AUC[g/syn] 0.875, TPR@5% 0.675 @ K=8. The "before" picture. |
| **V7.0** | **Scoring-only** sweep on V6 weights: cosine, linear_z3, Mahalanobis (per-sender Ledoit-Wolf), tied-Σ, S-norm | `mahal_per_sender` wins low-FPR tail (+5.8 pp TPR@5%). Tied-Σ and S-norm hurt. No retraining. |
| **V7.1** | Hybrid cos+mahal, honest (held-out) S-norm cohort | α-sweep monotone → cosine & Mahalanobis aren't complementary. Honest S-norm makes things *worse* (cohort ≠ LLM-impostor distribution). |
| **V7.2** | Enrollment-K sweep (K=4→40) per scorer | Three regimes: K≤4 cosine wins (Σ rank-deficient); K=8 Mahalanobis ties on AUC, +6pp tail; **K≥16 Mahalanobis clearly ahead** (+9.2 pp TPR@5% at K=16). → adaptive-by-K policy. |
| **V7.3** | **Retrain** with Mahal-aware recipe: n_syn 2→4, τ 0.07→0.05, +key LoRA, 150 ep, centroid-align aux loss | Crossed the 0.90 ship threshold. Beats V6 on every operating-point metric. This is the shipped encoder recipe. |
| **V8 (overnight)** | **syn-v1 vs syn-v2 A/B**: identical recipe, only the synthetic-data generation differs (cross-register-fraction 0.0 vs 0.4). Full gen→train→probe→ablate. | **syn-v2 wins** (+13.3 pp TPR@1% at K=8). Cross-register positives teach the model that legitimate register-shift ≠ forgery. |

Run the consolidated leaderboard yourself:
```bash
python scripts/summarize_results.py                       # print digest
python scripts/summarize_results.py --md results/SUMMARY.md   # also write file
```

---

## 3. What worked / what didn't

**Worked**
- LUAR-MUD backbone (authorship-pretrained) — the single biggest lever.
- Per-sender Ledoit-Wolf Mahalanobis scoring at K ≥ 16 (no retraining needed).
- The V7.3 training recipe (more synthetic pressure + sharper τ + more LoRA + longer training).
- **Cross-register positive synthetic data (syn-v2)** — fixes the #1 false-negative
  mode (same author switching formal↔casual register), lifts the hard tail.

**Didn't work / dead ends**
- RoBERTa / MPNet backbones — wrong inductive bias (topic over style); LoRA didn't fix it.
- Tied-Σ Mahalanobis — pools away the per-sender idiosyncrasy that carries the signal.
- S-norm (both contaminated and honest cohort) — the cohort distribution doesn't
  match LLM impostors, so it hurts AUC[g/syn].
- Medoid instead of centroid — throws away the denoising benefit of averaging.
- Cos+Mahal hybrid — no complementary information; Mahalanobis alone is the better summary.
- `checkpoint_best.pt` under the `auc/genuine_vs_synthetic` monitor — misleading; use last-epoch.

**Open risks / caveats (from the diagnosis + memos)**
- Probe is small (120 genuine / 200 synth) → AUROC σ ≈ 0.02. Sub-2pp gains may be noise; the ablation now reports bootstrap CIs — trust those.
- Synthetic corpus is narrow: one LLM (Mistral-7B), one template, ~10–15 per sender. A detector that only beats this adversary may be brittle vs Claude/GPT/Gemini.
- 30% false-positive mode from the 2026-06-03 diagnosis: short, stylistically-flat corporate prose. ~3.3% of emails are <10 words → near-zero signal.
- V7 trades a little `FPR_other` for `FPR_synthetic` gains — not a regression for the BEC threat model, but flag it if a downstream check expects low absolute FPR on the easy case.

---

## 4. How to run (and what it costs)

All runs log to **W&B** (`WANDB_MODE=online`, authenticated via `~/.netrc`,
project `email-fraud-detection`, entity `brown-university-deep-learning`). The
overnight script groups them under `WANDB_RUN_GROUP` (default `v8-syn-compare`).

### The full overnight pipeline
```bash
tmux new -s v8 'bash scripts/run_v8_overnight.sh 2>&1 | tee runs/_v8_overnight/console.log'
```
Stages: setup → (gen v1, gen v2) → train v1 → train v2 → (probe + ablate) ×2.
It **skips** generation if `data/synthetic/enron_synthetic_v{1,2}` already exist
(they do), and pushes through failures so one bad stage doesn't abort the night.

### A single training run
```bash
python scripts/train.py --config configs/experiments/v7_synv2.yaml --output-dir runs/v7_synv2
```

### Post-hoc evals on a checkpoint
```bash
python scripts/probe_authenticity.py --checkpoint <ckpt> --config <cfg> \
    --split train --mode both --out results/v8/probe_X.json --wandb
python scripts/ablate_adaptive_scorers.py --config <cfg> --checkpoint <ckpt> \
    --split synthetic --rank-by tpr1 --bootstrap 1000 --k-sweep 4,8,16,25 \
    --out-dir results/v8 --tag scorer_ablation_X --wandb
```

### Time estimates (single A40, measured from the overnight logs)

| Stage | Wall-clock |
|---|---|
| Synthetic generation (per arm, Mistral-7B fp16) | ~13–15 min |
| Train (150 epochs, 1 arm) | **~75 min** |
| Authenticity probe (1 arm) | ~1–2 min |
| Scorer ablation w/ 1000-boot + K-sweep (1 arm) | ~2 min |
| **Full overnight, datasets already present (2 arms)** | **~2.5–2.75 h** |
| **Full overnight incl. regenerating both datasets** | **~3.25 h** |

Per-epoch is ~9–10 s wall (11 train batches + the inline CentroidProbe + the
every-5-epoch PAN eval + hard-negative mining after warmup).

### Smoke test before a real launch
A 3-epoch smoke config exercises the whole train→probe→ablate path in ~2–3 min:
```bash
WANDB_MODE=disabled python scripts/train.py --config configs/experiments/_smoke.yaml --output-dir runs/_smoke
WANDB_MODE=disabled python scripts/probe_authenticity.py --checkpoint runs/_smoke/checkpoint_last.pt \
    --config configs/experiments/_smoke.yaml --split train --mode both --out /tmp/smoke_probe.json
WANDB_MODE=disabled python scripts/ablate_adaptive_scorers.py --config configs/experiments/_smoke.yaml \
    --checkpoint runs/_smoke/checkpoint_last.pt --split synthetic --rank-by tpr1 \
    --bootstrap 50 --k-sweep 4,8 --out-dir /tmp --tag smoke
```
(`_smoke.yaml` and `runs/_smoke/` are throwaway — safe to delete.)

---

## 5. Recommended next experiments (prioritized)

Ordered by expected impact / cost. The full 17-item menu is in
`experiments/v7/CHANGELOG_V7.md` — these are the high-value picks for the
stated objective (robust spam/AI-impersonation detection).

1. ~~**Lock in syn-v2 + bootstrap CIs on the headline operating points.**~~
   **DONE (2026-06-09).** Bootstrapped (1000×) CI ablation run at every K on the
   syn-v2 epoch-150 checkpoint; results in `results/v8/ksweep_ci_synv2_K{4,8,16,25}.json`,
   logged to W&B group `v8-ksweep-ci` (entity `klconvergence`). Verdict below.

   | K | baseline `linear_z3` tpr1 [95% CI] | tpr5 | auc | top by tpr1 (P(win)) |
   |---|---|---|---|---|
   | 4  | 0.500 [0.283, 0.608] | 0.575 | 0.851 | linear_z3 |
   | 8  | 0.458 [0.350, 0.625] | 0.650 | 0.858 | linear_z3 |
   | 16 | 0.575 [0.442, 0.683] | 0.700 | 0.901 | mahalanobis (0.72) |
   | 25 | 0.583 [0.450, 0.700] | 0.683 | 0.907 | ewma_centroid_z3 (0.54) |

   **Takeaway:** CIs are wide (±~0.13 on tpr1) — exactly the small-probe noise
   floor the memos warned about. No alternative scorer beats `linear_z3` with a
   Δ-CI excluding 0 at any K, so **keep `linear_z3` as the shipped default**;
   Mahalanobis leads on the point estimate at K≥16 but is not yet distinguishable
   from noise on this probe. To make the K≥16 Mahalanobis win real, the probe
   needs to be larger / the test corpus more diverse (→ item 2).
2. **Diversify the adversary (biggest robustness lever).** Regenerate synthetics
   from ≥2 more LLMs (Claude, GPT-4, Llama-3) and/or multiple temperatures/prompts.
   The current detector is tuned against one Mistral template; this is the most
   likely real-world brittleness. ~50 syn/sender would also stop the encoder
   memorising specific synth strings (n_syn=4 recycles every synth ~every 3 batches).
3. **Short-email abstain gate.** ~3.3% of emails are <10 words → near-zero
   stylometric signal and a top FP/FN driver. Abstain (don't decide) below a
   word/char floor in the scoring pipeline rather than misclassify.
4. **Per-sender / isotonic threshold calibration.** Score distributions vary
   per sender; a global cosine threshold near 0.5 drives the 30% FP mode. Calibrate
   per-sender on held-out enrollment, or shift the global boundary toward ~0.62.
5. **More training senders (100 → 150).** Enron has 150+ usable senders; more
   diverse negatives → better generalization. Cheap.
6. **Per-sender LDA projection before Mahalanobis** (scoring-side, no retraining):
   project onto the sender's most-discriminative directions vs a cohort. Expected
   +1–2 pp at low FPR.
7. **Sub-centroid prototypes (KMeans k=2 per sender at enrollment)** for senders
   with genuinely bimodal style — try as a scoring change before any retrain.

---

## 6. Repo map for results

| Path | What |
|---|---|
| `results/v7/v7_0_scoring_sweep.*` | V7.0 scoring-fn sweep on V6 weights |
| `results/v7/v7_1_hybrid.*` | V7.1 hybrid + honest S-norm |
| `results/v7/v7_2_k_sweep.json` | V7.2 K-sweep on V6 weights |
| `results/v7/v7_3_k_sweep.json`, `v7_3_scoring_sweep_ep150.*` | V7.3 retrained-encoder sweeps (**the shipped numbers**) |
| `results/v7/confusion/*.png` | Confusion matrices, V6 vs V7, per K/operating-point |
| `results/v8/{probe,scorer_ablation}_synv{1,2}.*` | **V8 syn A/B** (canonical copies; `results/v7/` holds identical duplicates) |
| `results/v8/figures/v8_confusion.png` | V8 2×2 confusion grid (arm × probe type) |
| `runs/_v8_overnight/*.log` | Per-stage logs from the overnight run |

**Note:** `results/v7/probe_synv*.json` and `results/v7/scorer_ablation_synv*.*`
are byte-identical duplicates of the `results/v8/` files (the overnight script
wrote both). Treat `results/v8/` as canonical for the syn A/B.
