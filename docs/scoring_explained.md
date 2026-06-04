# Scoring, Explained — and Where to Take It Next

A deep walk through how the email-fraud system turns a raw email into an
anomaly score, every parameter that shapes that score, and a menu of concrete
improvements (dynamic/adaptive parameters chief among them).

The whole stack lives in `src/email_fraud/`:

| Stage | File | Job |
|-------|------|-----|
| Preprocess | `data/preprocessing.py` | strip quotes/sigs, mask entities |
| Encode | `encoders/hf_encoder.py` | text → embedding vector `(d,)` |
| Profile | `heads/prototypical.py`, `profiles/store.py` | per-sender centroid + spread (+ optional covariance) |
| Score | `heads/prototypical.py` + `scoring/score_functions.py` | geometry → number |
| Decide | `scoring/centroid_probe.py`, `scripts/analyze_thresholds.py` | number → threshold → verdict |
| Measure | `scoring/metrics.py` | AUC / EER / c@1 / pAUC / TPR@FPR |

---

## 1. The core idea

Every sender has a **fingerprint**: the average position of their emails in a
learned stylometric embedding space (the *centroid*), plus how tightly their
emails cluster around it (the *spread*). A new email claiming to be from that
sender is scored by **how far it lands from the centroid, measured in units of
the sender's own spread**. Close = genuine; far = possible impersonation.

That "distance in units of spread" is a **z-score**, and it is the single
quantity everything else is built on.

---

## 2. Building the fingerprint (`fit` / `upsert`)

There are **two** profile builders, and they differ in one important way.

### 2a. `PrototypicalHead.fit()` — equal-weight running mean
Used at enrollment / evaluation. The centroid is the *true* mean of every email
seen, via an exact online update:

```
centroid_new = (centroid_old · k_old + Σ batch_embeddings) / k_new
```

- **spread** = mean cosine distance of emails to the centroid, `mean(1 − cos_sim)`.
  Range `[0, 2]`; in practice small (e.g. 0.05–0.3).
- **k** = number of emails incorporated → drives the confidence tier.
- If `store_embeddings=True` (default), the raw embeddings are kept so a
  per-sender covariance can be fit later for Mahalanobis scoring.

> ⚠️ Note: on incremental updates, `spread` is recomputed **only from the new
> batch** against the updated centroid — an approximation, not the true pooled
> spread. This is a known shortcut (see §6, item 6).

### 2b. `SenderProfileStore.upsert()` — EWMA (recency-biased)
Used for live traffic where you want recent style to matter more:

```
centroid_new = (1 − α)·centroid_old + α·embedding      # then re-normalized
spread_new   = (1 − α)·spread_old   + α·(1 − cos_dist)
```

with `α = 0.1` (a fixed hyperparameter). A single weird email moves the
centroid only 10%. **Equal-weight (`fit`) vs recency-weighted (`upsert`) is a
real design fork** — see improvement §6, item 1.

---

## 3. Turning geometry into a score

At query time the head computes two numbers — `cos_sim` (query to centroid) and
`spread` — then runs them through a **score function**. The z-score is:

```
z = (1 − cos_sim) / max(spread, 1e-9)
```

`z < 0` would mean closer-than-average (more genuine than a typical enrollment
email); `z = 3` means three "style-standard-deviations" out.

`scoring/score_functions.py` is the single source of truth for the mappings:

| Name | Formula | Range | Notes |
|------|---------|-------|-------|
| `linear_z3` *(default)* | `max(0, 1 − z/3)` | [0,1] | score > 0.95 is **structurally unreachable** (needs z < 0.15) |
| `linear_z2` | `max(0, 1 − z/2)` | [0,1] | sharper; > 0.95 needs z < 0.10 |
| `cosine` | `(cos_sim + 1)/2` | [0,1] | ignores spread entirely; uses full range |
| `sigmoid_z` | `σ(1 − z)` | (0,1) | smooth, differentiable; z=0→0.73, z=3→0.12 |
| `neg_z` | `−z` | unbounded | pure ranking signal for ROC/AUC |

**Key consequence:** because all five are monotone in `z` (except `cosine`,
which uses a different geometry), **they produce nearly identical AUC** — AUC is
invariant to monotone rescaling. Picking a score function changes *where fixed
thresholds land*, not *ranking quality*. This is why the project moved to
**FPR-anchored thresholds** instead of fixed 0.5/0.8/0.95 cutoffs (§5).

### The Mahalanobis path (the "smarter" distance)
`score_fn = "mahalanobis"` or `"adaptive_k"` swaps the isotropic z-score for a
full per-sender covariance distance:

```
d_M(q) = sqrt( (q − μ)ᵀ Σ⁻¹ (q − μ) ),   score = −d_M
```

- `Σ` is estimated with **Ledoit-Wolf shrinkage** (`sklearn.covariance.LedoitWolf`),
  then ridge-regularized (`Σ + ridge·scale·I`, `ridge=1e-4`) before inversion.
- Computed **lazily** (only on first query after an upsert) and cached as a
  precision matrix; `_prec_dirty` flags staleness.
- **Falls back to cosine when `k < mahalanobis_min_k` (default 5)** — a rank-(k−1)
  covariance is too noisy below that. This is already a *dynamic, k-conditional*
  parameter, and the template for many improvements below.

Why it helps: cosine/z-score assume style varies equally in all embedding
directions. Mahalanobis learns that a sender may be highly consistent in some
stylistic dimensions and loose in others. Reported **+3 AUC pp on genuine-vs-
synthetic at K=16–25** (per `experiments/v7/CHANGELOG_V7.md`).

---

## 4. Confidence tiers & abstention

Independent of the score, every profile gets a **tier** from its email count `k`:

```
1–4 → low (abstain=True)   5–9 → medium   10–24 → high   25+ → very_high
```

`low` ⇒ `abstain=True`: the system refuses to flag fraud when it has seen too
few emails to trust the centroid. Unknown sender ⇒ `score=0, abstain=True`.
This abstention is what the **c@1** and **F0.5u** metrics reward/penalize.

---

## 5. From score to verdict — thresholds

There is no single magic cutoff. `scripts/analyze_thresholds.py` and the
in-training `centroid_probe.py` derive operating points three ways:

1. **Fixed bands** (0.5 / 0.8 / 0.95) — mostly diagnostic; the 0.95 band is
   ~always zero because of `linear_z3`'s structural ceiling.
2. **FPR-anchored** — pick τ as the `(1 − FPR)` quantile of the *impostor*
   score distribution, so you commit to a false-alarm budget (1%, 5%, 10%) and
   read off the recall/precision you get. **This is the deployment-relevant
   view.** Anchored separately on pooled impostors and on synthetic-only
   (the hardest negatives).
3. **Precision-anchored** — lowest τ achieving target precision (80–99%),
   reporting recall there.
4. **Coverage-at-accuracy** — selective-classifier view: confidence
   `c = |score − 0.5|`, how much traffic can you auto-decide at accuracy ≥ T.

The three negative pools that matter: **genuine vs other-sender** (easy),
**genuine vs synthetic** (hard — same person, LLM-written), **genuine vs all**.

---

## 6. The parameters that shape every score

A quick inventory — these are the knobs, and most are **currently static**:

| Parameter | Where | Default | Currently |
|-----------|-------|---------|-----------|
| `score_fn` | head | `linear_z3` | static, global |
| z divisor (the `/3`) | score fn | 3 | hard-coded |
| `ewma_alpha` (α) | store | 0.1 | static, global |
| `mahalanobis_min_k` | head | 5 | static (but k-gated ✓) |
| `ridge` | head | 1e-4 | static, global |
| `confidence_tiers` | head/store | 1-4/5-9/… | static, global |
| spread floor `_EPS` | score fn | 1e-9 | static |
| decision threshold τ | analyze | per-FPR | global, not per-sender |

The throughline of every improvement below: **these are global constants that
should arguably be functions of the data in front of them.**

---

## 7. Avenues for improvement

### A. Dynamic / adaptive parameters (the headline)

**1. Per-sender adaptive `ewma_alpha`.** A sender with 200 emails and stable
style should barely move (α→small); a new sender (k=3) should adapt fast
(α→large). Classic form: `α_k = max(α_min, 1/(k+1))` interpolating from a true
mean toward recency bias. Removes the "is it `fit` or `upsert`?" fork entirely.

**2. Adaptive z-divisor / per-sender calibration.** The `/3` in `linear_z3` is a
universal guess. Replace it with a per-sender scale fit from that sender's own
genuine z-score distribution (e.g. divisor = their 99th-percentile z). Turns the
score into a **calibrated tail probability** instead of a linear ramp — `score =
P(z ≥ z_query | genuine)`. Directly fixes the "0.95 is unreachable" pathology.

**3. Adaptive `mahalanobis_min_k` and shrinkage.** Already k-gated; go further —
make `ridge`/shrinkage a function of `k` and `d` (more regularization when
`k ≪ d`). Ledoit-Wolf already adapts α internally, but the ridge floor doesn't.
A smooth **cosine→Mahalanobis blend** weighted by `min(1, (k−min_k)/window)`
avoids the hard cliff at k=5.

**4. Per-sender decision thresholds.** Today τ is global (one FPR-anchored cut
for everyone). High-variance senders deserve a looser τ. Fit τ per sender (or
per tier) from a held-out genuine/impostor split, or per-cohort by writing
volume. Biggest single lever on real-world precision.

**5. Tier-conditional score functions.** Use `cosine` (full-range, robust) for
low-k senders and Mahalanobis for high-k — the tier already exists, just branch
on it.

### B. Correctness / robustness fixes

**6. True pooled spread on incremental `fit`.** Right now `spread` is recomputed
only from the incoming batch, drifting from the real pooled value. Maintain
running sum-of-cosine-distances (or Welford-style) for an exact online spread.

**7. Spread floor instead of `_EPS=1e-9`.** A sender with near-zero spread makes
`z` explode and every query look fraudulent. Floor spread at a small, data-
derived minimum (e.g. global median spread × 0.1).

**8. Two-sided scoring.** `z < 0` (suspiciously *more* on-style than any real
email) can itself signal a templated/AI-polished forgery, yet `linear_z3` clamps
it to a perfect score. Consider penalizing extreme negative z too.

**9. Length / content normalization.** Very short emails ("ok, thanks") have
unstable embeddings. Weight enrollment by length, or abstain below a token floor.

### C. Modeling upgrades

**10. Wire Mahalanobis into `SenderProfileStore`** (currently `NotImplementedError`)
so live/production profiles get the same covariance scoring the head has.

**11. Score fusion.** Combine `cos_sim`, Mahalanobis, and a kNN-to-enrollment
distance via a small logistic blender fit on the probe set — typically beats any
single geometry.

**12. Calibration layer.** Fit isotonic / Platt scaling on probe scores so the
output is a true `P(genuine)`, making thresholds portable across senders and
model versions (right now score distributions shift every retrain).

**13. Cross-encoder re-rank** (stub exists) for borderline scores near τ — cheap
because it only fires in the decision band.

### D. Evaluation / ops

**14. Per-sender / per-tier metric breakdowns** — global AUC hides that the
system may be excellent on high-k senders and near-random on low-k ones.
**15. Threshold drift monitoring** — track impostor-quantile τ over time; a
sudden shift means the embedding space moved and τ must be re-anchored.
**16. Population-level fast path** — for unknown senders (score=0 today), back
off to a global/cohort style model instead of a hard abstain.

---

## 8. Where I'd start

If the goal is the **biggest score-quality gain for the least risk**, in order:

1. **Per-sender calibration of the z-divisor (#2)** — fixes the unreachable-0.95
   bug *and* makes scores comparable across senders. Pure post-processing, no
   retraining.
2. **Adaptive `ewma_alpha` (#1)** — unifies the two profile builders and
   improves both cold-start and stability.
3. **Smooth cosine→Mahalanobis blend (#3)** — removes the k=5 cliff.
4. **Per-sender/tier thresholds (#4)** — the real-world precision lever.

Each is independently shippable and measurable on the existing centroid-probe
harness (`auc/genuine_vs_synthetic`, `pauc/...`, `tpr_at_fpr/...`).
