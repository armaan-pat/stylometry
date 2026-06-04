# Scoring Ablation — Data-Driven Scorers

Prototypes the "make every parameter a function of the data" direction from
`scoring_explained.md` and measures it head-to-head against the production
score on a real checkpoint.

- **Prototypes:** `src/email_fraud/scoring/adaptive.py`
- **Harness:** `scripts/ablation_scoring.py`
- **Run that produced the numbers below:**
  ```
  python scripts/ablation_scoring.py --run runs/minilm_m5/2026-04-29_19-12-10 \
      --split train --vary-enroll --by-tier --n-profile-senders 60
  ```

---

## What's being compared

All scorers consume the **same embeddings** (encoded once, cached under
`runs/_ablation_cache/`), so any difference is the *scoring rule alone*.

| Scorer | Idea | Data-driven? |
|--------|------|--------------|
| `baseline_linear_z3` | `max(0, 1−z/3)` — production default | no (fixed /3) |
| `baseline_cosine` | `(cos+1)/2` — spread-free | no |
| `ewma_centroid_z3` | z3 on a fixed-α EWMA centroid | partial (recency) |
| `z_global_cal` | divisor = pooled p90 genuine z | **global, from data** |
| `z_persender_cal` | divisor = per-sender **shrunk** p90 genuine z | **per-sender, from data** |
| `z_persender_sigmoid` | smooth calibrated version of the above | **per-sender, from data** |
| `mahalanobis` | per-sender Ledoit-Wolf distance, k-gated → cosine | **per-sender covariance** |
| `mahal_blend` | smooth cosine↔Mahalanobis blend in *precision space* | **per-sender + k-adaptive** |
| `tier_switch` | cosine if k<5 else Mahalanobis | k-conditional |

The two genuinely-adaptive ideas:
1. **Per-sender z calibration** — replace the hard-coded `/3` with each sender's
   own genuine z-distribution (leave-one-out p90), **shrunk toward a global
   prior** with pseudo-count `n0=8` so sparse senders don't over-fit.
2. **k-adaptive cosine↔Mahalanobis blend** — `w·Σ⁻¹ + (1−w)·I` with
   `w = clip((k−5)/10, 0, 1)`. Cold-start senders fall back to cosine ranking;
   data-rich senders get the full covariance metric. No cliff at k=5.

---

## Results (MiniLM, epoch-2 frozen — 240 genuine / 400 impostor)

> ⚠️ **Caveat first:** this checkpoint is a barely-trained, frozen-backbone
> MiniLM (only 2 epochs saved). Absolute AUCs sit at 0.52–0.57 — close to
> chance. Treat these as a **methodology demonstration and relative ranking**,
> not a final number. The gaps should widen on a properly trained encoder.

### Ranking quality (AUC + paired significance vs baseline)

| scorer | AUC | pAUC@5% | TPR@5% | EER | ΔAUC vs baseline |
|--------|-----|---------|--------|-----|------------------|
| **mahal_blend** | **0.570** | **0.094** | **0.133** | **0.450** | **+0.035  [+0.010,+0.063]  SIG** |
| mahalanobis / tier_switch | 0.556 | 0.029 | 0.050 | 0.455 | +0.022  [−0.023,+0.069] |
| baseline_cosine | 0.547 | 0.068 | 0.104 | 0.458 | +0.013 |
| z_persender_sigmoid | 0.539 | 0.033 | 0.063 | 0.457 | +0.004 |
| z_persender_cal | 0.539 | 0.033 | 0.063 | 0.457 | +0.004 |
| `baseline_linear_z3` * | 0.535 | 0.044 | 0.067 | 0.467 | — |
| z_global_cal | 0.535 | 0.044 | 0.067 | 0.467 | −0.000 |
| ewma_centroid_z3 | 0.520 | 0.030 | 0.046 | 0.495 | −0.016 |

**`mahal_blend` is the only scorer that beats the baseline with a paired
bootstrap CI excluding zero** — and it also wins pAUC@5% and TPR@5% (the
deployment-relevant low-FPR region) by ~2×.

### Per-tier AUC (grouped by the claimed sender's enrollment k)

| scorer | low(1–4) | med(5–9) | high(10–24) | vhigh(25+) |
|--------|----------|----------|-------------|------------|
| baseline_linear_z3 | 0.538 | 0.576 | 0.555 | 0.495 |
| z_persender_sigmoid | **0.561** | 0.572 | 0.556 | 0.494 |
| mahalanobis | 0.547 | **0.588** | **0.589** | **0.546** |
| mahal_blend | 0.547 | 0.586 | 0.588 | **0.546** |

The k-story is exactly as designed:
- **Mahalanobis dominates wherever there's enough data** (med/high/vhigh:
  +0.01 to +0.05 AUC), and the blend matches it there.
- **It correctly does *not* help at low-k** (rank-deficient covariance) —
  there the **per-sender z calibration** is the best option (0.561 vs 0.538).
- This is the argument for a **tier-/k-conditional** production scorer rather
  than one global rule.

### Calibration health — what AUC can't see

AUC is a pure ranking metric, so per-sender calibration barely moves it *by
construction*. Its real payoff is **score reachability** (does a genuine email
ever earn a high score?) and **cross-sender comparability** (does one global
threshold mean the same thing for everyone?).

Genuine pool, fraction reaching a high score:

| scorer | mean | p95 | %>0.8 | %>0.95 |
|--------|------|-----|-------|--------|
| baseline_linear_z3 | 0.585 | 0.810 | **5.8%** | 0.0% |
| z_persender_sigmoid | 0.709 | 0.883 | **42.1%** | 0.0% |
| baseline_cosine | 0.858 | 0.927 | 85.0% | 0.0% |

Cross-sender consistency (std of per-sender mean genuine score, **lower =
score means the same thing per sender**):

| baseline_linear_z3 | z_persender_cal | z_persender_sigmoid | baseline_cosine |
|---|---|---|---|
| 0.110 | 0.107 | **0.0996** | 0.028 |

**Takeaways on calibration:**
- `linear_z3` is badly bunched — only **5.8%** of *genuine* emails clear 0.8,
  confirming the structural ceiling described in `scoring_explained.md`.
- The **sigmoid per-sender** calibration opens the range (42% >0.8) *and*
  improves cross-sender consistency, **without hurting AUC** — so a fixed
  operating threshold transfers better across senders.
- The *linear* per-sender variant (`z_persender_cal`) actually compresses the
  mean — the **sigmoid form is the one to keep**.
- `cosine` looks best on both calibration axes but is spread-blind (ignores
  per-sender style variance), which is why it trails on low-FPR metrics.

---

## How to read / extend

- **Add a scorer:** drop a `(bank, query, sid) -> float` fn into
  `adaptive.py::SCORERS`; it appears in the table automatically.
- **Honest significance:** the paired bootstrap resamples the *same* queries for
  baseline and candidate, so the ΔAUC CI is the number to trust, not the raw
  AUC gap.
- **Tiers:** `--vary-enroll` randomizes enrollment k per sender so low→vhigh
  tiers populate; without it every profile lands in one tier.
- **Caching:** embeddings cache by (checkpoint, split); re-runs with different
  scorers/seeds are instant.

## Recommended next steps (in priority order)

1. **Ship `mahal_blend` as the default scorer** — only significant AUC win,
   best low-FPR behavior, graceful k-degradation. Wire it into
   `PrototypicalHead` (it already has the Mahalanobis machinery) behind a
   `score_fn="mahal_blend"` option.
2. **Adopt `z_persender_sigmoid` for the *reported* score** (the [0,1] number a
   human sees / thresholds on) even if ranking uses Mahalanobis — it fixes
   reachability and cross-sender threshold transfer.
3. **Re-run this ablation on a strong checkpoint** (a LUAR/RoBERTa LoRA run,
   not epoch-2 MiniLM) to confirm the gaps widen, then again **with synthetic
   (LLM) impostors** — the hard negative the system actually cares about.
4. **Per-sender / per-tier thresholds** — the next data-driven lever, measurable
   with the same harness (extend it to sweep τ per tier).
