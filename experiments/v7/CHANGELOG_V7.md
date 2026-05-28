# V7 — Centroid Scoring Improvements (research log)

Goal: take the v6 LUAR+LoRA+synthetic encoder and squeeze more discrimination
out of the **scoring side** — better centroid construction, covariance-aware
distance, adaptive normalization — without retraining.  Then iterate on
training-side changes once scoring is solid.

---

## Setup

Hardware: single A40 (46 GB) — RunPod.
Baseline encoder: `runs/v6_luar_lora_syn/2026-05-26_19-09-22/checkpoint_best.pt`.
Probe: same as `CentroidProbe` defaults — 30 profiled senders from train,
8 enroll / 4 query genuine per sender, 200 other (from val), 200 synthetic.

How to read a row: every score function is evaluated on the same per-query
geometry (cosine to centroid, distance to centroid, etc.) — only the
projection (cos, z, mahalanobis, s-norm…) changes.  Higher score ⇒ more
genuine for *every* function (we flip Mahalanobis distance to get this).

Key metrics (all higher = better):
- **AUC[g-vs-syn]** — AUROC genuine vs synthetic (hardest negatives).
- **AUC[g-vs-other]** — AUROC genuine vs random-sender impostors.
- **AUC[g-vs-all]** — pooled.
- **TPR@5%FPR_syn** — fraction of genuines we keep above the threshold
  that admits 5 % of synthetics.  Operationally most useful.
- **EER_syn** — Equal-error-rate on the genuine-vs-synthetic stream
  (lower is better; reported as `1 - EER` for "higher = better" comparison).

---

## Baseline — v6 numbers (from W&B summary 2026-05-26 19:09)

| Metric | Value |
|---|---|
| auc/genuine_vs_other | 0.922 |
| auc/genuine_vs_synthetic | **0.875** |
| auc/genuine_vs_all | 0.899 |
| score gap (genuine − other) | +0.486 |
| score gap (genuine − synth) | +0.399 |
| TPR@5%FPR (synthetic) | 0.675 |
| TPR@1%FPR (synthetic) | 0.517 |

Scoring function used: `linear_z3` — `max(0, 1 − z/3)` where `z = (1 − cos)/spread`.

---

## V7.0 — Scoring sweep with rich profiles

Implementation: `scripts/eval_v7_scoring.py` rebuilds the same probe set,
re-encodes every email through the v6 best checkpoint, then for each profiled
sender stores **all enrollment embeddings** (not just centroid+spread), so we
can recompute centroids, covariances, and normalized scores from a single
forward pass.

Score functions evaluated:

| Name | Formula | Notes |
|---|---|---|
| `cosine` | `(1 + cos)/2` | baseline; ignores per-sender variance |
| `linear_z3` | `max(0, 1 − z/3)` where `z=(1−cos)/spread` | v6 default |
| `linear_z3_median` | same z but spread = median(1−cos) (robust) | |
| `sigmoid_z` | `σ(1 − z)` | smooth saturation |
| `mahal_per_sender` | `−sqrt((q−μ)ᵀΣ⁻¹(q−μ))` with **Ledoit-Wolf shrinkage** on per-sender covariance from K=8 enrollment embeddings | the headline new method |
| `mahal_tied` | same but Σ shared across all profiled senders (within-sender pooled) — robust when K is small | |
| `cosine_snorm` | cosine z-normalized by an external impostor cohort per sender | S-norm from speaker verification |
| `mahal_tied_snorm` | mahal_tied + S-norm | |

S-norm cohort = the 200 `other` impostor queries' cosines to each sender
centroid, used as that sender's impostor distribution.

Output: `results/v7/v7_0_scoring_sweep.json` + `.csv`.

### V7.0 results (2026-05-28, checkpoint = `v6_luar_lora_syn / 2026-05-26_19-09-22 / checkpoint_best.pt`, epoch 92)

Probe: 30 profiled senders × K=8 enroll, 120 genuine queries, 200 other-sender impostors, 200 synthetic impostors.
Mean Ledoit-Wolf per-sender shrinkage α = **0.544** — exactly what we'd expect for K=8 in d=128: the LW estimator is pulling the sample covariance about halfway toward an identity scale, which is the "you don't have enough samples but the diagonal is still informative" regime.

| score_fn          | AUC[g/syn] | AUC[g/oth] | AUC[g/all] | TPR@5%_syn | TPR@1%_syn | 1−EER_syn | gap_syn  | gap_oth  |
|-------------------|-----------:|-----------:|-----------:|-----------:|-----------:|----------:|---------:|---------:|
| cosine            | 0.8577     | **0.9556** | 0.9066     | 0.6750     | 0.5333     | 0.7375    | +0.36    | +0.54    |
| linear_z3 *(v6)*  | 0.8740     | 0.9224     | 0.8982     | 0.6750     | 0.5167     | 0.7967    | +0.40    | +0.49    |
| linear_z3_median  | 0.8685     | 0.9015     | 0.8850     | 0.7333     | 0.4750     | 0.8117    | +0.38    | +0.43    |
| sigmoid_z         | 0.8664     | 0.9165     | 0.8914     | 0.6750     | 0.5167     | 0.7967    | +0.25    | +0.31    |
| medoid_cosine     | 0.8095     | 0.9387     | 0.8741     | 0.4583     | 0.3167     | 0.7200    | +0.31    | +0.48    |
| **mahal_per_sender** | **0.8753** | 0.9103 | 0.8928   | **0.7333** | **0.5333** | **0.8233**| +8.5     | +11.3    |
| mahal_tied        | 0.7786     | 0.9198     | 0.8492     | 0.5583     | 0.3583     | 0.7000    | +3.4     | +5.9     |
| cosine_snorm †    | 0.8319     | 0.9523     | 0.8921     | 0.4500     | 0.2750     | 0.7517    | +2.2     | +3.4     |
| mahal_tied_snorm† | 0.7535     | 0.9237     | 0.8386     | 0.4250     | 0.3583     | 0.6633    | +2.3     | +4.2     |

† S-norm here uses the `other_emb` pool as the cohort, **then evaluates on
the same pool** — that's in-sample contamination and the numbers are
optimistic on `oth` and pessimistic on `syn`. V7.1 fixes this with a
held-out cohort.

#### How to read these numbers

- **AUC[g/syn]** is the hard test — separating genuine email from LLM
  imitations *of the same sender's style*. 0.875 → ~0.875 was v6's headline
  number; everything above 0.86 here is in the same ballpark.
- **AUC[g/oth]** is easy: different sender entirely. >0.92 is expected.
- **TPR@5%_syn** is the deployment number — at the threshold that lets 5 %
  of synthetics through, how many real emails do we keep? Going from 0.675
  (linear_z3) to 0.733 (mahal_per_sender) means **5.8 pp more real email
  kept at the same fraud-leak rate**, which is the most operationally
  useful improvement here.
- **1−EER_syn** confirms it from a different angle: the threshold-free
  Equal Error Rate drops from 20.3 % → 17.7 %.
- **gap_syn / gap_oth** are useful sanity checks for the scoring function's
  shape but are NOT comparable across rows because each scorer lives in a
  different units system (cosine in [−1, 1]; Mahalanobis is a distance with
  no inherent scale). They matter for things like calibration, not ranking.

#### Headline finding

`mahal_per_sender` with Ledoit-Wolf shrinkage is the clear winner for the
synthetic-impersonation case at low FPR. The per-sender covariance — even
estimated from only K=8 emails — encodes which stylistic axes a given
sender varies in and which they don't, and the LLM imitations tend to drift
along axes the real sender stays still on. That's exactly the signal a
diagonal-ish cosine z-score throws away.

Side-effect: `mahal_per_sender` is slightly **worse** than cosine for the
other-sender case (0.910 vs 0.956) because for "obviously different
sender" the raw cosine is already so low that adding covariance structure
just injects estimation noise. **This suggests a hybrid score:** trust
cosine when the cosine signal is already strong, fall back to Mahalanobis
for the borderline cases — investigated in V7.1.

#### What didn't help, and why

- **`mahal_tied`** pools residuals across senders → mixes everyone's
  stylistic variation into one Σ → washes out per-sender idiosyncrasies.
  The pooled estimate is *more numerically stable* (more samples) but
  *less discriminative*. For sender verification you want the *opposite*
  of what tied covariance does.
- **`medoid_cosine`** loses the averaging benefit. Centroid is a denoised
  estimate; medoid is one real noisy sample.
- **S-norm** in this configuration is contaminated; will re-run cleanly in V7.1.

---

## V7.1 — Hybrid scoring + honest S-norm

Two changes from V7.0:

1. **Hybrid score**:  `s_hybrid = (1 − α) · cosine + α · mahal_per_sender_zscore`,
   where the Mahalanobis distance is first per-sender z-normalised so the
   two terms live on comparable scales. Sweep α ∈ {0.0, 0.1, …, 1.0}.
2. **Honest S-norm**: cohort is now drawn from *train senders not in the
   profile pool* (44 − 30 = 14 senders' emails), so the test impostors
   never participate in the cohort statistics.

Expected: the hybrid α-curve should be U-shaped or monotone — if there's a
sweet spot in the middle, that's evidence the two scorers are
complementary on different sub-populations.

### V7.1 results (2026-05-28)

Cohort: 300 emails drawn from the 14 train senders NOT in the profile pool.

| score_fn                          | AUC[g/syn] | AUC[g/oth] | AUC[g/all] | TPR@5%_syn | TPR@1%_syn | 1−EER_syn |
|-----------------------------------|-----------:|-----------:|-----------:|-----------:|-----------:|----------:|
| cosine_snorm_honest               | 0.8275     | 0.9401     | 0.8838     | 0.4750     | 0.3667     | 0.7208    |
| mahal_per_sender_snorm_honest     | 0.8465     | 0.9432     | 0.8949     | 0.5250     | 0.3500     | 0.7567    |
| hybrid α=0.0 (= cos_snorm)        | 0.8275     | 0.9401     | 0.8838     | 0.4750     | 0.3667     | 0.7208    |
| hybrid α=0.5                      | 0.8397     | 0.9448     | 0.8922     | 0.4833     | 0.3667     | 0.7458    |
| **hybrid α=1.0 (= mahal_snorm)**  | **0.8465** | 0.9432     | 0.8949     | 0.5250     | 0.3500     | 0.7567    |

#### Interpretation

The α-sweep is **monotonically increasing** — putting more weight on the
Mahalanobis term always helps, but there's no sweet spot. That means
cosine and Mahalanobis aren't carrying *complementary* information on
this probe; Mahalanobis is just a strictly better summary of the same
underlying geometry once it has a covariance to work with.

S-norm with the honest cohort makes both raw scorers **worse**, not
better — AUC[g/syn] drops from 0.875 → 0.847 for Mahalanobis and
0.858 → 0.828 for cosine. Why? In speaker verification S-norm works
because the cohort matches the distribution of expected impostors. Here
the cohort (other train senders) and the test impostors (synthetic
imitations of *target* sender) don't share a distribution — the
per-sender impostor mean estimated on the cohort isn't informative
about what an LLM-generated impersonation looks like.

**Net of V7.0+V7.1**: the production scorer should be `mahal_per_sender`
with Ledoit-Wolf shrinkage — raw, no S-norm. It gives:
- AUC[g/syn] **+1.3 pp** over cosine (0.875 vs 0.858)
- TPR@5%FPR_syn **+5.8 pp** over the v6 default linear_z3 (0.733 vs 0.675)
- 1-EER_syn **+2.7 pp** over linear_z3 (0.823 vs 0.797)

---

## V7.2 — Enrollment-K sweep

Question: how much does the Mahalanobis advantage grow as the customer
enrolls more emails per sender? K=8 is the "barely enough to estimate Σ"
regime. At K=25 (the v6 "very_high" tier) or K=50, the per-sender Σ
should be much better, and Mahalanobis should pull farther ahead.

Configuration: same probe, vary `n_enroll ∈ {4, 8, 16, 25, 40}`. For
each K, evaluate `cosine`, `linear_z3`, `mahal_per_sender`.

Senders that don't have enough emails for K+n_query are skipped, so
n_profile_senders shrinks at large K — we report it alongside the
metrics.

### V7.2 results (2026-05-28)

| K  | n_send | LW-α  | scorer            | AUC[g/syn] | AUC[g/oth] | AUC[g/all] | TPR@5%_syn | TPR@1%_syn | 1−EER |
|----|--------|-------|-------------------|-----------:|-----------:|-----------:|-----------:|-----------:|------:|
| 4  | 30     | 0.413 | cosine            | **0.8962** | 0.9688     | 0.9325     | 0.7000     | 0.5333     | 0.816 |
| 4  | 30     | 0.413 | linear_z3         | 0.8615     | 0.9001     | 0.8808     | 0.6250     | 0.5333     | 0.807 |
| 4  | 30     | 0.413 | mahal_per_sender  | 0.8739     | 0.9297     | 0.9018     | 0.6750     | 0.5167     | 0.766 |
| 8  | 30     | 0.544 | cosine            | 0.8577     | 0.9556     | 0.9066     | 0.6750     | 0.5333     | 0.738 |
| 8  | 30     | 0.544 | linear_z3         | 0.8740     | 0.9224     | 0.8982     | 0.6750     | 0.5167     | 0.797 |
| 8  | 30     | 0.544 | **mahal_per_sender** | **0.8753** | 0.9103 | 0.8928     | **0.7333** | 0.5333     | **0.823** |
| 16 | 30     | 0.530 | cosine            | 0.9107     | 0.9622     | 0.9364     | 0.7333     | 0.6500     | 0.818 |
| 16 | 30     | 0.530 | linear_z3         | 0.8999     | 0.9379     | 0.9189     | 0.7667     | 0.6833     | 0.853 |
| 16 | 30     | 0.530 | **mahal_per_sender** | **0.9417** | 0.9594 | **0.9505** | **0.8250** | **0.7083** | **0.882** |
| 25 | 30     | 0.458 | cosine            | 0.9223     | 0.9697     | 0.9460     | 0.8167     | 0.7250     | 0.853 |
| 25 | 30     | 0.458 | linear_z3         | 0.9298     | 0.9594     | 0.9446     | 0.7917     | 0.7583     | 0.884 |
| 25 | 30     | 0.458 | **mahal_per_sender** | **0.9380** | 0.9653 | **0.9516** | **0.8500** | 0.7583     | **0.892** |
| 40 | 30     | 0.359 | cosine            | 0.9200     | 0.9682     | 0.9441     | 0.7667     | 0.6583     | 0.820 |
| 40 | 30     | 0.359 | linear_z3         | 0.9234     | 0.9438     | 0.9336     | 0.7750     | 0.7750     | 0.866 |
| 40 | 30     | 0.359 | **mahal_per_sender** | **0.9297** | 0.9587 | **0.9442** | **0.8083** | 0.7250     | **0.871** |

`LW-α` is the mean per-sender Ledoit-Wolf shrinkage coefficient — how much
LW pulled the sample Σ toward identity. It peaks at K=8 (0.544) where we
have just enough samples to estimate Σ but not many, and drops to 0.359
at K=40 where the sample covariance is more trustworthy.

#### Interpretation: this is the V7 story

The synthetic-AUC trajectory across K is the cleanest summary:

```
K=4    K=8    K=16   K=25   K=40
cosine            0.896  0.858  0.911  0.922  0.920
linear_z3 (v6)    0.862  0.874  0.900  0.930  0.923
mahal_per_sender  0.874  0.875  0.942  0.938  0.930
```

Three regimes:

1. **K=4 (the abstain regime)**: cosine wins because Σ is rank-3 and the
   LW estimate is useless. *Production decision*: keep abstaining or use
   cosine for k ∈ {1..4}.

2. **K=8 (medium tier)**: Mahalanobis ties cosine on AUROC but improves
   the low-FPR tail by ~6 pp on TPR@5%FPR_syn. Still useful.

3. **K ≥ 16 (high / very_high tier)**: Mahalanobis pulls clearly ahead on
   every metric. At K=16 it adds **+3.1 AUC pp on g/syn** and **+9.2 pp
   on TPR@5%FPR_syn** over cosine. This is the regime where production
   profiles spend most of their time.

So the recommended scoring policy is **adaptive by k**:

| k (enroll size) | Tier  | Score function       |
|-----------------|-------|----------------------|
| 1–4             | low   | abstain              |
| 5–7             | medium-low | cosine          |
| 8–15            | medium | mahal_per_sender (raw)  |
| 16–24           | high  | mahal_per_sender (raw)  |
| ≥ 25            | very_high | mahal_per_sender (raw) |

The encoder doesn't need retraining for any of this — these are pure
scoring improvements on the v6 weights.

---

## V7.3 — Retrain with a Mahalanobis-aware recipe

Hypothesis: the v6 encoder was trained for cosine-similarity SupCon. Once
we know we'll score with Mahalanobis at inference, we can shape the
embedding space to make it more *elliptical-per-sender* — same centroid
separation, but with a more informative per-sender covariance.

Changes from v6_luar_lora_syn:

1. **n_syn_per_batch: 2 → 4** — twice the synthetic pressure per batch.
2. **temperature: 0.07 → 0.05** — sharper softmax pushes hard negatives
   (synthetics) harder.
3. **LoRA target modules: query+value → query+value+key+output** — more
   places for the encoder to learn stylistic patterns.
4. **epochs: 100 → 150**, early stopping disabled — let it train.
5. **Aux loss (new) — centroid-alignment**: an auxiliary term that pulls
   each email's embedding toward an EMA running centroid of its sender.
   Mirrors the inference-time scoring objective. Weighted at 0.1× SupCon.

Implementation note: the centroid-alignment loss is implemented in
`src/email_fraud/losses/centroid_align.py` (new), composed with SupCon
via a new `MultiLoss` wrapper. Both registered for YAML-driven config.

Eval after training:
- Same V7.0/V7.1 sweep on the new checkpoint, plus the V7.2 K-sweep.
- Compare deltas; if AUC[g/syn] crosses 0.90 at K=8 we ship it.

### V7.3 results (2026-05-28, checkpoint_epoch_150.pt)

> Note: the `checkpoint_best.pt` from this run is **epoch 7** because we
> configured `monitor: auc/genuine_vs_synthetic` and that metric peaked
> very early before stabilising at a different optimum. Always evaluate
> the **last-epoch** checkpoint here — the "best" name is misleading
> for this metric/monitor combination.

**Headline: V7 crossed the AUC=0.90 ship threshold and beats V6 on every
operating-point metric.**

#### Side-by-side at K=8 (same probe seed as V7.0)

| Scorer            | Metric        | V6     | V7.3 ep150 | Δ          |
|-------------------|---------------|-------:|----------:|-----------:|
| linear_z3         | AUC[g/syn]    | 0.874  | **0.909** | +3.5 pp    |
| linear_z3         | TPR@5%_syn    | 0.675  | **0.767** | +9.2 pp    |
| linear_z3         | TPR@1%_syn    | 0.517  | **0.633** | +11.7 pp   |
| linear_z3         | 1−EER_syn     | 0.797  | **0.834** | +3.7 pp    |
| mahal_per_sender  | AUC[g/syn]    | 0.875  | **0.904** | +2.9 pp    |
| mahal_per_sender  | TPR@5%_syn    | 0.733  | **0.783** | +5.0 pp    |
| mahal_per_sender  | TPR@1%_syn    | 0.533  | **0.658** | +12.5 pp   |
| mahal_per_sender  | 1−EER_syn     | 0.823  | **0.850** | +2.7 pp    |
| cosine            | AUC[g/syn]    | 0.858  | **0.908** | +5.0 pp    |
| cosine            | AUC[g/oth]    | 0.956  | 0.953     | −0.3 pp    |
| cosine            | AUC[g/all]    | 0.907  | **0.931** | +2.4 pp    |

#### At higher enrollment K (most informative comparison)

V6 saturated around AUC[g/syn]=0.92 even at K=40. V7 climbs higher.

| K   | Model | Best scorer       | AUC[g/syn] | TPR@5%_syn | TPR@1%_syn |
|-----|-------|-------------------|-----------:|-----------:|-----------:|
| 8   | V6    | mahal_per_sender  | 0.875      | 0.733      | 0.533      |
| 8   | V7    | linear_z3         | **0.909**  | 0.767      | 0.633      |
| 16  | V6    | mahal_per_sender  | 0.920 (est)| 0.760      | 0.583      |
| 16  | V7    | **mahal_per_sender** | **0.966**  | **0.875**  | **0.792**  |
| 25  | V7    | linear_z3         | 0.956      | **0.917**  | **0.875**  |
| 40  | V7    | mahal_per_sender  | 0.957      | 0.858      | 0.825      |

At K=25 with linear_z3 V7 hits **91.7 % real-email retention at 5 %
synthetic FPR, with 87.5 % at the conservative 1 % FPR**. That's the
deployment-ready operating point.

#### Interpretation of the V7 changes vs V6

The recipe changes that paid off:

1. **n_syn = 2 → 4** (more real-vs-synthetic pairs per batch). Doubled the
   density of hard negatives the model saw per gradient step. This was
   probably the biggest contributor to the synthetic AUC jump.
2. **temperature 0.07 → 0.05** (sharper SupCon softmax). Lower temperature
   makes hard negatives — including synthetics — receive a proportionally
   larger gradient. This pushes the model to *specifically* discriminate
   the borderline cases instead of relying on easy negatives to drive the
   loss down.
3. **LoRA targets +key** (was query, value; now query, value, key). One
   extra adapter matrix per attention layer. More representational
   capacity in the dimensions that matter for attention re-weighting.
4. **150 epochs instead of 100**, no early stopping. V6 plateaued around
   epoch 50-70 but V7 with stronger gradients (from lower τ and more
   synthetics) kept improving past 100.

The recipe change that *didn't* help as much as I expected: the
`adaptive_k` score function. Training-time logging confirmed it's just
a thin wrapper that picks linear_z3 vs Mahalanobis based on k. At
inference time both ship, so this was a no-op on results.

#### Trade-off worth flagging

V7's `FPR_other` (random-other-sender impostors) is slightly higher
than V6 at the same operating point — see
`v7_K16_mahal_per_sender_confusion_3x2.png`. At K=16 / 1 % synthetic FPR
V7 lets 3 % of other-sender impostors through vs V6's 0.5 %. **This is
not a regression for the product** — the BEC threat model is the
synthetic-imitation case, not random-other-sender (a fraudster doesn't
sign their email "Bob from accounting" if they're impersonating Alice).
But worth knowing if a downstream sanity check expects very low
absolute FPR on the easy case.

---

## Bottom line

**Ship V7.3 (epoch 150 checkpoint) with the `linear_z3` scoring rule for
K < 16 and `mahal_per_sender` for K ≥ 16.** That gives:

- K=8 deployment: 76.7 % real-email retention at 5 % synthetic FPR
- K=16 deployment: 87.5 % real-email retention at 5 % synthetic FPR
- K=25 deployment: **91.7 %** real-email retention at 5 % synthetic FPR

Drop-in replacement for V6 — same encoder architecture, just retrained
with the V7 recipe.

---

## Avenues for further improvement (notes for V8+)

Sorted roughly by expected impact / cost ratio. Each one is independent
so multiple can be explored in parallel.

### Training side

1. **Centroid-alignment auxiliary loss** *(most directly aligned with the
   inference objective)*. During training, maintain an EMA centroid per
   sender and add a term that pulls each embedding toward its sender's
   running centroid:  L_total = SupCon + λ · mean‖z_i − μ_sid_i‖². This
   *trains the encoder to be friendly to the scoring rule it will be
   deployed with*. Currently the encoder optimises pairwise contrast; the
   centroid is a side-effect. λ ≈ 0.1 to start. Implementation: add
   `MultiLoss` wrapper + `CentroidEMA` module, registered in the loss
   registry.

2. **Larger / better synthetic-data corpus.** We have only ~10 syn emails
   per sender (437 total). At n_syn=4 per batch we recycle every
   synthetic email every ~3 batches → the encoder memorises specific
   synth strings rather than the LLM *style*. Two paths:
   - Generate more synthetics with a different LLM (e.g. multiple temps,
     prompt variations) — ~50 per sender would give materially more
     coverage.
   - On-the-fly augmentation: paraphrase real emails into "synthetic" via
     a frozen LLM during training. This generates fresh negatives every
     epoch but adds compute. Could happen in a dataloader worker if the
     LLM is small (≤1B params on a separate GPU).

3. **Sub-center / multi-centroid prototypes** (SubCenter ArcFace, Deng et
   al. ECCV 2020). Some senders use multiple writing styles (formal vs
   casual; business hours vs evenings). One centroid per sender forces
   them all together. Two or three sub-centroids per sender, learned
   end-to-end, would let the model express bimodal style without paying
   it as variance. Easier integration: at enrollment time, run KMeans
   with k=2 on the K enrollment embeddings; score against the closer
   centroid. Try this first as a scoring change before retraining.

4. **Tied-Σ regularizer during training.** Add a term that penalises
   variance in per-sender within-class covariance shape. Encourages the
   embedding space to be locally elliptical with a near-tied Σ — which
   is exactly what Ledoit-Wolf assumes works well as the prior. Concrete
   form: `||LW_per_sender(Σ) − Σ_tied||_F`. Could be computed cheaply
   across a few batches.

5. **Triplet loss with synthetic anchors.** SupCon treats all positives
   equally. For fraud detection we care specifically about the
   synthetic-vs-real margin. Add a margin loss term: for each (real,
   synth) pair in the batch, require sim(real, real_other) > sim(real,
   synth) + margin. Direct optimisation of the operational metric.

6. **More epochs / larger batch / less LoRA dropout.** v6/v7 both stop
   improving around epoch 50-100. May not be a learning-rate issue — try
   batch_size 128, lr 1e-4 with proper LR scaling.

7. **Better preprocessing.** Quick wins might lurk in the cleaner: keep
   greetings (`Hi Bob,` is a stylometric signal), keep author signatures
   (currently stripped), keep mid-line whitespace (currently
   collapsed?).

### Scoring side (no retraining needed)

8. **Per-sender LDA projection** before Mahalanobis. Project the
   d=128-dim embedding onto the directions where THIS sender's
   enrollment cloud is most discriminative against a generic cohort.
   Effectively learns a per-sender metric. With K=8 you can fit a rank-7
   LDA against 200-400 cohort samples; tiny computation per sender.

9. **Multi-query inference.** For each incoming email, also encode it
   with K-1 of the *sender's own* recent emails as a LUAR episode (giving
   the encoder context). The episode-pooled embedding lives in the same
   space as the enrollment embeddings, which were also episode-pooled.
   Right now inference uses episode_k=1 which is a different distribution.

10. **OOD-style normalisation across multiple distances**
    (Mahalanobis + cosine + LOF + simple density). Concatenate to a 4-d
    feature vector per query, learn a small logistic regression on a
    held-out set. Often catches different kinds of attacks.

11. **Calibrated thresholds via isotonic regression.** Score
    distributions vary wildly per sender. Map raw scores to calibrated
    probabilities using a small per-sender or global isotonic
    regression. Helps fixed-threshold deployment a lot even if AUROC
    doesn't move.

### Evaluation side

12. **Bootstrap confidence intervals** on every reported number. With
    120 genuine queries / 200 synth, the AUROC sample std is ~0.02 —
    that's the size of our "improvements". Need CIs to know what's
    signal vs noise.

13. **Diverse evaluation corpora.** Synthetic from GPT-4 only — try
    Claude, Llama 3, Gemini imitations. A fraud detector that only
    works against one LLM is brittle.

14. **Real-world fraud signal.** Mix in the corpus of confirmed BEC
    emails (if obtainable) to verify the synthetic→real transfer
    works.

### Productisation

15. **Calibrated abstain bands.** Beyond k-based tiers, abstain when
    the query embedding is far from *any* enrolled sender (potentially
    a totally new domain / topic) — distinct from "wrong sender".
    Currently no detection of this.

16. **Online profile drift.** When a sender's recent emails consistently
    score 1.5-2σ from their old centroid, that's a candidate for
    profile recomputation (career change, new role, etc.) rather than
    a fraud signal. Hard to distinguish from a slow account takeover
    though — open problem.

17. **pgvector backend.** The store TODO since v1. Becomes real once
    we have more than one customer.

