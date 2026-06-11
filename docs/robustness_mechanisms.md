# Robustness Mechanisms — Low-K Enrollment and Short Emails

*2026-06-09. Design memo: how to make the detector robust to (a) senders with
few enrolled emails and (b) short queries, as modeling changes rather than
abstain gates. Companion to the roadmap in `docs/EXPERIMENT_STATUS.md` §5.*

## The framing: both are uncertainty problems, not coverage problems

The two failure modes the diagnosis surfaced share a root cause:

- **Low K** — the *profile* is uncertain. A centroid from 4 emails is a noisy
  estimate of the sender's true style; spread and covariance are worse.
- **Short emails** — the *query* is uncertain. A 9-word email is a noisy
  measurement of the author's style; its embedding could have come from
  almost anyone.

Today the head treats both as if they were exact: a k=4 centroid is scored
against with the same confidence as a k=40 one, and a 9-word query's z-score
is trusted as much as a 300-word one. The borderline scores then land near
the global threshold and flip a coin — that's the 30% FP mode.

Abstaining removes those queries from the decision stream (a coverage
tradeoff). The robust alternative is to make every hard-coded constant in the
scoring chain a *function of the evidence*: amount of enrollment data,
length of the query, and what the population of senders looks like. Formally,
the head becomes a small hierarchical Bayesian model and the score a
posterior quantity that automatically widens when evidence is thin — extreme
verdicts then require extreme evidence.

The per-sender z calibration shipped on 2026-06-09 (`z_persender_sigmoid`,
shrunk toward a pooled prior with pseudo-count n0=8) is the first instance of
this pattern. The mechanisms below extend it along both axes.

---

## A. Low-K robustness

### A1. Episodic, K-matched training (the big retrain lever)

**Problem.** SupCon optimizes pairwise contrast; the centroid the deployed
system actually uses is a side-effect the encoder was never asked to make
good. Nothing in training rewards "the mean of 4 of my embeddings is a stable
description of me."

**Mechanism.** Train the way we infer (prototypical-network episodes,
Snell et al. 2017 — the head's own citation):

1. Each batch, for each sender, sample a *support set* of size K′ with
   **K′ ~ uniform{2..16}** (vary it — this is the robustness part) and a
   *query set* of the remaining emails.
2. Build prototypes = support means; classify each query against all
   prototypes in-batch with the deployed distance; cross-entropy on the
   sender identity (synthetic hard negatives stay in the query pool as
   "not-this-sender" targets).
3. Keep SupCon as an auxiliary term: `L = L_proto + β·L_supcon` (β≈0.5 to
   start), so pairwise structure isn't lost.

Because K′ is sampled small *during training*, the encoder is explicitly
optimized to produce embeddings whose **small-sample means are already
discriminative** — low-K robustness is baked into the representation, not
patched at scoring time. This subsumes/extends the centroid-align aux loss
from the V7.3 recipe (CHANGELOG_V7 item 1).

Cost: a sampler + loss change, one retrain (~75 min). Measure with a
K-stratified probe (see §C).

### A2. Hierarchical profile: shrink the centroid, not just the scale

**Problem.** At k=4 the centroid itself is the noisiest object in the chain.

**Mechanism.** Treat each sender's true style vector μ_s as a draw from a
population: μ_s ~ N(μ₀, Σ₀), estimated once from all training senders (or
per-cluster of senders — "terse trader" vs "verbose counsel" — via KMeans on
training centroids). Then the profile used for scoring is the posterior mean:

    μ̂_s = (k·x̄_s + n₀·μ₀) / (k + n₀)        (precision-weighted; n₀ ≈ 4–8)

and the predictive variance used in the z denominator inflates by the
centroid's own uncertainty:

    spread_eff² = spread_pop²·(1 + 1/k)       (→ Student-t-flavored score)

At k=2 the profile is mostly population prior plus a nudge — so an impostor
isn't compared against two random emails, but against "a typical sender,
adjusted toward what we've seen." As k grows the prior washes out
automatically. **Scoring-side only, no retrain**; drops straight into
`PrototypicalHead.fit()` next to the z_scale shrinkage, which is the same
formula applied to the scale instead of the mean.

### A3. Population-prior Mahalanobis (fix "only wins at K≥16")

**Problem.** Per-sender Ledoit-Wolf Σ is rank-deficient below k≈16, which is
why `mahal_per_sender` only pulls ahead late and the ship rule needs a
K-switch. Tied-Σ failed because it *replaced* the per-sender structure.

**Mechanism.** Use the population covariance as a *prior*, not a substitute:

    Σ̂_s = (k·Σ_s + ν·Σ_pop) / (k + ν)         (ν ≈ 10–20)

where Σ_pop is the average within-sender covariance over all training
senders, computed once offline. At k=4 the metric is mostly Σ_pop (a far
better default than the identity cosine implicitly assumes — it knows which
embedding directions are generically noisy vs identity-bearing); at k=30 it
converges to today's per-sender LW. This removes the k-cliff entirely and
should move the Mahalanobis win from K≥16 down to K≈4–8.
`mahal_blend` in `scoring/adaptive.py` already blends toward the *identity*;
this is the same code path with `np.eye(d)` replaced by a fitted Σ_pop.
**Scoring-side only.**

### A4. Episode-pooled enrollment and multi-query inference

**Problem.** Training pools episodes of `episode_k=4` emails; enrollment and
inference encode single emails (`episode_k=1`) — a distribution mismatch that
is worst exactly when k is small and each embedding matters.

**Mechanism.** (CHANGELOG_V7 item 9.) At enrollment, encode the K emails as
episodes (the representation the LUAR backbone was actually trained to emit);
at query time, encode the query *together with* a few of the sender's recent
emails as an episode and compare in the same space. The encoder itself then
aggregates evidence — typically stronger few-shot behavior than averaging
single-email embeddings post-hoc. Needs an inference-pipeline change but no
retrain to pilot.

---

## B. Short-email robustness

### B1. Close the train/test length mismatch (data + augmentation)

**Problem.** `base.yaml` sets `min_body_words: 50` — *the encoder never sees
an email under 50 words during training*, yet production must score them
(3.3% of traffic is under 10 words). The model isn't bad at short emails
because short emails are hopeless; it's bad partly because they're
out-of-distribution.

**Mechanism.**
- Lower the training floor (keep a sanity floor of ~5 words) so short
  genuine emails exist in training.
- **Crop augmentation:** with some probability, replace a training email by
  a random contiguous span (first n sentences, or a random window) of
  itself, labeled as the same sender. SupCon/episodic loss then explicitly
  demands that a 15-word fragment of an Alice email embeds near full Alice
  emails — i.e. *length-invariant style features*. This is the single
  cheapest training change targeted at the 30% FP mode, and it also
  manufactures unlimited short positives without new data.
- Keep greetings and sign-offs during preprocessing (CHANGELOG_V7 item 7):
  "Hey—" vs "Dear Keith," *is* the signal for short mail, and stripping
  signatures deletes most of what a 10-word email has.

### B2. Heteroscedastic scoring: length-aware observation noise

**Problem.** A z-score treats the query embedding as an exact measurement.
For short emails it isn't — their embeddings are high-variance, so they
produce extreme z values in both directions and land in both FP and FN piles.

**Mechanism.** Give the query an observation-noise term that grows as length
shrinks, and widen the z denominator accordingly:

    z_robust = (1 − cos(q, μ̂_s)) / sqrt(spread_s² + σ²(len_q))

Fit σ(len) empirically, once, globally: take held-out genuine emails, bucket
by length, and record the variance of their deviation from their own sender's
centroid per bucket — a 10-line isotonic fit. The behavior this buys:

- A 9-word email that's far from the centroid is *discounted* — distance is
  divided by a large denominator, so the score drifts toward the indecision
  region instead of a confident "fraud."
- A 9-word email can no longer earn a confident "genuine" either.
- A 300-word email keeps exactly today's behavior (σ→0).

This is the *soft, principled* version of the abstain gate: instead of a
hard word-count cliff, confidence degrades continuously with signal content,
and downstream thresholds/tiers see honestly calibrated scores. Composes
multiplicatively with the per-sender z_scale calibration (scale handles
"who is this sender", σ(len) handles "how much did this email tell us").
**Scoring-side only, no retrain.**

### B3. Probabilistic embeddings (the deeper retrain version of B2)

Make the encoder output a Gaussian per email — μ(x) and a learned variance
σ(x) (Probabilistic Face Embeddings / HIB style) — trained with a mutual
likelihood score so the model *learns* to widen σ for low-signal inputs
(short, boilerplate, forwarded fragments) rather than us hand-fitting a
length curve. Scoring uses the closed-form likelihood that two Gaussians
share a source. Strictly more general than B2 (it also catches *long but
stylistically empty* corporate boilerplate — the other half of the FP mode,
which a pure length gate misses). Higher effort: new head on the encoder +
loss term + a retrain; do B2 first, then promote to B3 if the σ(len) curve
shows large residual variance unexplained by length.

### B4. Confidence-weighted profile updates

Symmetric consequence of B2/B3: when *enrolling*, weight each email's
contribution to the centroid/spread by its information content
(1/σ² weights). A sender enrolled mostly via one-liners currently gets a
mushy centroid that mis-scores their occasional long email; precision
weighting fixes the profile side of the same problem. Scoring-side only.

---

## C. Measurement prerequisite (or none of this is provable)

Every mechanism above moves tail metrics by single-digit pp. The 2026-06-09
CI run showed the old probe could not resolve anything under ~13 pp on TPR@1%.
Prerequisites, partially shipped today:

1. **Bigger probe** — done (60×6 / 600 / 600, `probe:` config). Halves the
   noise floor; still recommend the multi-LLM synthetic corpus for the rest.
2. **Stratified reporting** — required next: report AUC / TPR@FPR broken out
   **by enrollment K ∈ {2,4,8,16,25} × query-length bucket
   {<10, 10–25, 25–75, 75+} words**. Aggregate metrics hide exactly the cells
   these mechanisms target; a change that lifts the (K=4, <10w) cell by 15 pp
   can vanish in the pooled number.
3. **Per-cell bootstrap CIs** — the ablation harness already does this; wire
   the strata in.

## Suggested order of attack

| # | Mechanism | Side | Retrain? | Targets |
|---|-----------|------|----------|---------|
| 1 | B1 crop augmentation + lower word floor + keep greetings | data | yes (one run) | short-email FP/FN at the source |
| 2 | A1 episodic variable-K training | training | same run as 1 | low-K robustness in the representation |

> **Status (2026-06-09):** 1+2 are implemented —
> `configs/experiments/v9_episodic_shortmail.yaml` is the shared retrain
> (episodic loss in `losses/episodic.py`, crop augmentation in
> `data/augment.py`, enforced word/alnum floors in `data/preprocessing.py`,
> re-prepared corpus at `data/processed/enron_shortmail`). One adaptation vs
> the sketch in §A1: K′ is sampled from {2..6}, not {2..16} — synthetic
> senders have only 8–9 emails, capping `emails_per_sender_k` at 8, so the
> support/query split happens inside each sender's 8 in-batch embeddings
> (with `episode_k: 1`, so they are single-email embeddings). 3–6 remain open.
| 3 | B2 σ(len) heteroscedastic z | scoring | no | short-email tails, immediately |
| 4 | A2 centroid shrinkage + A3 Σ_pop-prior Mahalanobis | scoring | no | low-K scoring floor, kills the K-switch |
| 5 | A4 episode-pooled enrollment/inference | pipeline | no | few-shot headroom |
| 6 | B3 probabilistic embeddings | model | yes | boilerplate FPs beyond length |

1+2 share a retrain; 3+4 are pure scoring changes measurable on existing
checkpoints with the expanded probe. That gives two independent workstreams
that don't block each other.
