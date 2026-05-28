# Metrics Guide — How to Read the W&B Dashboard

A complete, self-contained reference for every metric this project reports
to Weights & Biases during training. Read top-to-bottom the first time,
then use the table of contents below as a lookup index.

> This document is paired with the model architecture summary in
> `summary.md` and the V7 research log in `experiments/v7/CHANGELOG_V7.md`.

---

## Table of contents

1. [The data the model sees vs the data the metrics see](#1-the-data-the-model-sees-vs-the-data-the-metrics-see)
2. [Anatomy of "a score" — how a single number gets made](#2-anatomy-of-a-score--how-a-single-number-gets-made)
3. [Two confusion matrices: pair-AUROC vs centroid-AUROC](#3-two-confusion-matrices-pair-auroc-vs-centroid-auroc)
4. [Loss metrics: `train/loss`, `val/loss`](#4-loss-metrics-trainloss-valloss)
5. [Embedding metrics: `embedding/*`](#5-embedding-metrics-embedding)
6. [Centroid AUROCs: `auc/genuine_vs_{other,synthetic,all}`](#6-centroid-aurocs-aucgenuine_vsotherssyntheticall)
7. [Score statistics: `score/mean_*`, `score/gap_*`](#7-score-statistics-scoremean_-scoregap_)
8. [Threshold-band metrics: `threshold_0.50/*`, `threshold_0.80/*`, `threshold_0.95/*`](#8-threshold-band-metrics-threshold_050-threshold_080-threshold_095)
9. [FPR-anchored operating points: `op/all/fpr_*`, `op/synthetic/fpr_*`](#9-fpr-anchored-operating-points-opallfpr_-opsyntheticfpr_)
10. [Partial AUC: `pauc/*`](#10-partial-auc-pauc)
11. [TPR at fixed FPR: `tpr_at_fpr/*`](#11-tpr-at-fixed-fpr-tpr_at_fpr)
12. [Coverage at accuracy: `coverage/at_acc_*`](#12-coverage-at-accuracy-coverageat_acc_)
13. [Probe diagnostics: `probe/n_*`](#13-probe-diagnostics-proben_)
14. [Sampler diagnostics: `train/sampler/*`](#14-sampler-diagnostics-trainsampler)
15. [PAN verification metrics: `test/*`](#15-pan-verification-metrics-test)
16. [Sanity-check FAQ](#16-sanity-check-faq)

---

## 1. The data the model sees vs the data the metrics see

A common confusion: **the metrics on W&B are NOT computed on the data the
model trained on, and they're NOT computed the same way the model
optimised its loss.** They emulate the deployment scenario.

### What the model trained on

Each training batch is a `P × K` grid:

- `P = 16` senders chosen at random per batch.
- `K = 8` emails per sender per batch.
- Of those P senders, `n_syn=2` (or 4 in v7) come as *pairs*: one slot for
  the real sender, one for an LLM imitation of that sender. The remaining
  P − 2·n_syn slots are unpaired real senders.
- Each batch is therefore 128 emails (64 in the actual encoded shape after
  LUAR's episode pooling of 4 emails per episode).

The model never sees a *single* email and a *single* claimed sender; it
sees a batch and runs a Supervised-Contrastive loss that pushes
same-sender embeddings together and different-sender embeddings apart.
The training loss is computed on the batch labels (`labels[i] == labels[j]`
defines positives), not on a notion of "is this email fraud."

### What the metrics see — the **CentroidProbe**

The metrics simulate the inference-time decision problem: **given a
sender's "profile" and an incoming email, how confidently can we say
this email is from the claimed sender?** That's done by a stand-alone
`CentroidProbe` object that's constructed once before training and
re-evaluated each epoch with the *current* encoder weights.

```
                       ┌──── PROFILED SENDERS (30 of them) ────┐
                       │ Each is enrolled with 8 of their emails│
                       │ (the "enrollment pool"). The mean of   │
                       │ these 8 embeddings forms the centroid. │
                       └────────────────────────────────────────┘
                                          │
                ┌─────────────────────────┼───────────────────────────┐
                │                         │                           │
        ┌───────▼──────┐    ┌─────────────▼──────────┐    ┌──────────▼────────┐
        │  GENUINE     │    │  OTHER-SENDER          │    │  SYNTHETIC        │
        │  120 queries │    │  IMPOSTORS             │    │  IMPERSONATORS    │
        │              │    │  200 queries (drawn    │    │  200 queries      │
        │  Real emails │    │  from validation       │    │  (LLM-generated   │
        │  from the    │    │  senders — disjoint    │    │  to imitate the   │
        │  same 30     │    │  from training)        │    │  profiled senders'│
        │  senders,    │    │                        │    │  styles)          │
        │  HELD OUT    │    │  These come from       │    │                   │
        │  from the    │    │  totally different     │    │  These come from  │
        │  enrollment  │    │  people. SHOULD score  │    │  the SAME people  │
        │  pool        │    │  LOW.                  │    │  but were written │
        │              │    │                        │    │  by an LLM.       │
        │  SHOULD      │    │                        │    │  SHOULD score     │
        │  score HIGH. │    │                        │    │  LOW — but this   │
        │              │    │                        │    │  is the hard case.│
        └──────────────┘    └────────────────────────┘    └───────────────────┘
                │                         │                           │
                └──────────► same scoring rule (cosine z-score) ◄─────┘
                                          │
                       ┌──────────────────┴──────────────────┐
                       │  Three groups of scores per epoch:   │
                       │    genuine[120]   other[200]   syn[200] │
                       └─────────────────────────────────────┘
```

Where each pool comes from:

| Pool         | Source                                          | What it tests                          |
|--------------|-------------------------------------------------|----------------------------------------|
| `genuine`    | held-out real emails of the 30 profiled senders | "does the model recognise its own users?" |
| `other`      | val-split senders (sender-disjoint from train)  | easy negatives — different person      |
| `synthetic`  | LLM imitations of the 30 profiled senders       | **hard** negatives — same identity claim, fake author |

The probe is constructed once at training startup (the *texts* are fixed)
but each epoch the texts are re-encoded with the latest weights, so the
scores evolve as the encoder improves.

**Key point**: the metrics live in the "inference world." `val/loss` is
the only thing on the dashboard that lives in the "training world."

---

## 2. Anatomy of "a score" — how a single number gets made

For a single (query, claimed_sender) pair, the pipeline computes:

```
1. encode(query_text) → q ∈ ℝ^128  (L2-normalised)
2. Look up the sender's stored profile:
       centroid c ∈ ℝ^128  (mean of 8 enrollment embeddings)
       spread s              (mean cosine distance of those 8 to c)
       k                     (how many emails contributed)
3. cos_sim = c · q                       ∈ [-1, +1]
4. z       = (1 - cos_sim) / spread      ∈ [0, ∞)   "how many spreads away"
5. score   = max(0, 1 - z / 3)           ∈ [0, 1]   the V6 default "linear_z3"
```

Higher `score` = more like the claimed sender. The choice of step 5 (the
"score function") matters and has alternatives — `cosine`, `sigmoid_z`,
`mahalanobis` (V7+) — see `experiments/v7/CHANGELOG_V7.md`. Crucially:

- The metrics reported to W&B as `auc/*`, `pauc/*`, `tpr_at_fpr/*`,
  `coverage/*` are **ranking metrics** — they depend only on the
  ordering of scores within each pool, not on the absolute score values
  or which score function was used. Changing `linear_z3` → `cosine`
  does not change them as long as both are monotone in cos_sim.
- The metrics reported as `threshold_X/*` and `score/mean_*` and
  `score/gap_*` DO depend on absolute score values. Those numbers will
  shift if the score function changes shape.

This is the single most common source of "why did my AUC stay the same
but my threshold metrics changed?" confusion.

---

## 3. Two confusion matrices: pair-AUROC vs centroid-AUROC

The project actually reports **two different AUROC numbers** that solve
different problems. Don't confuse them.

### 3a. Pair-AUROC (`embedding/pair_auroc`, `test/AUC`)

Treats authorship verification as a *pairwise* task:

> Given two emails X and Y, did the same person write them?

Score = cosine similarity between encodings of X and Y. Label = 1 if same
author, 0 if different. The AUROC is over all O(N²) pairs in a batch.

Used to evaluate the **embedding space** itself — does the encoder cluster
same-author emails together? This is the same task LUAR was pre-trained
on and the same task the PAN authorship benchmark scores.

### 3b. Centroid-AUROC (`auc/genuine_vs_*`)

Treats fraud detection as an *anomaly* task:

> Given an email and a claimed sender, is the email consistent with that
> sender's stored profile?

Score = the `linear_z3` (or other) score above, in [0, 1]. Label = 1 if
the email really is from the claimed sender, 0 otherwise. AUROC is over
all (query, claimed_sender) pairs.

This is the metric that matters for the actual product. Same encoder can
have great pair-AUROC and only middling centroid-AUROC if the embedding
norms / centroid geometry aren't well-behaved, or vice-versa.

You'll see them both move during training but they often *don't* move
together. v6's `embedding/pair_auroc` reached 1.0 long before
`auc/genuine_vs_synthetic` stopped improving.

---

## 4. Loss metrics: `train/loss`, `val/loss`

The **only** numbers on the dashboard that live in the training world.
Computed exactly the same way during validation as training, but on the
held-out val split.

- `train/loss` — mean SupCon loss over the epoch's batches.
- `val/loss` — same on val. Lower = better in principle, but for SupCon
  the loss is dominated by which hard negatives the sampler happened to
  pick that epoch. **Don't treat val/loss as your decision metric.**
  It's a sanity number; if it explodes you have an unstable run.

For SupCon specifically: the loss can plateau at non-zero values even
when the embeddings are perfectly separable, because the loss involves
log-sum-exp over all in-batch negatives. Don't expect it to hit zero.

The number we actually monitor for early-stopping / best-checkpoint in
V7 is `auc/genuine_vs_synthetic` (configured via `monitor:` in the YAML).

---

## 5. Embedding metrics: `embedding/*`

Computed on the held-out validation batches, before any scoring.

### `embedding/knn_accuracy`

For each val embedding, find the nearest other val embedding (cosine
similarity, leave-one-out). Does its sender_id match? Average over all
val embeddings.

- **What it tests**: are same-sender emails the nearest neighbour of each
  other in embedding space?
- **Caveat**: the val split typically has only 6 senders; even random
  guessing gets 1/6 ≈ 0.17, but with a few enrollment emails per sender
  even a mediocre encoder hits >0.9 quickly. So **this saturates early
  and is mostly a "didn't break" check.**

### `embedding/knn_macro_f1`

Same as knn_accuracy but reported as macro-F1 (so that a single sender
with mostly-confused emails brings the score down). Useful if some
senders are systematically harder than others.

### `embedding/pair_auroc`

The pair-AUROC defined in §3a, computed on all O(B²) pairs from the val
batch. Bounded above by ~1.0; below ~0.5 means the encoder is
anti-clustering same authors (broken).

**How to interpret**: this is the cleanest "is the encoder learning?"
signal in the first few epochs. Once it crosses ~0.95 it stops being a
useful discriminator between checkpoints — `auc/genuine_vs_synthetic`
takes over.

---

## 6. Centroid AUROCs: `auc/genuine_vs_{other,synthetic,all}`

These are the headline numbers for the product.

### What AUROC means here

For a given group of negatives (`other`, `synthetic`, or both):

1. Combine genuine scores (labels = 1) with negative scores (labels = 0).
2. Sort by score descending.
3. AUROC = probability that a randomly-picked genuine email scores higher
   than a randomly-picked negative.

Equivalent geometric interpretation: the area under the curve plotting
TPR vs FPR as you sweep the threshold from +∞ down to −∞.

Range: 0.5 (random) to 1.0 (perfect separation). AUROC is
**threshold-free** — you don't pick a cutoff, the metric considers all
of them. That makes it a clean ranking score but also disconnects it
from "does this work at the threshold we'll actually deploy?"

### The three variants

| Metric                       | Negatives                | Test it really runs                          |
|------------------------------|--------------------------|----------------------------------------------|
| `auc/genuine_vs_other`       | random-other-sender (200)| can we tell two different people apart?      |
| `auc/genuine_vs_synthetic`   | LLM imitations (200)     | can we tell a real person from their AI clone? |
| `auc/genuine_vs_all`         | concatenation (400)      | overall sender-verification quality          |

Typically:
```
auc/genuine_vs_other     >    auc/genuine_vs_all    >    auc/genuine_vs_synthetic
   (easiest)                                                  (hardest)
```

If `vs_synthetic` is much lower than `vs_other`, the model relies on
*topic* / *vocabulary* signal rather than *style* — an LLM imitation
reuses the same vocab/topics and slips through. This is the diagnostic
you actually care about for fraud.

---

## 7. Score statistics: `score/mean_*`, `score/gap_*`

These describe the **raw score distribution** rather than its ranking.

- `score/mean_genuine` — average score of the 120 genuine queries
- `score/mean_other` — average score of the 200 other impostors
- `score/mean_synthetic` — average score of the 200 synthetics

- `score/gap_other` = `mean_genuine` − `mean_other`
- `score/gap_synthetic` = `mean_genuine` − `mean_synthetic`
- `score/synthetic_harder_than_other` = `gap_other` − `gap_synthetic`

### Why gaps matter even when AUROC moves first

Two distributions can have the same AUROC but very different gaps:

```
A: genuines  0.55  0.65  0.75       AUROC vs negatives 1.0
   negatives 0.10  0.20  0.30       gap = 0.45

B: genuines  0.91  0.92  0.93       AUROC vs negatives 1.0
   negatives 0.90  0.90  0.90       gap = 0.025
```

Both have AUROC 1.0 — perfectly separable. But B is *fragile* — any
encoding noise that shifts a genuine down by 0.01 starts producing false
negatives. A has a 0.45 cushion. The `gap` is your **margin** — a
proxy for how robust the score is to small perturbations.

### `synthetic_harder_than_other`

Positive ⇒ synthetics are harder than random-other (the model finds
LLM-imitations more genuine-looking than random other senders). This is
the expected direction. Big positive value means the encoder learned a
useful style signal that LLMs partially defeat but not totally.

Negative ⇒ synthetics are *easier* than random-other. **Bad sign**:
either the LLM is leaving obvious artifacts the model can pick up
(distinctive boilerplate phrases), in which case the model is "cheating"
and won't generalise to better LLMs, or the synthetic dataset is
miscalibrated. Investigate.

---

## 8. Threshold-band metrics: `threshold_0.50/*`, `threshold_0.80/*`, `threshold_0.95/*`

These are the operating-point metrics — what happens **if you draw a
hard line in the sand at score τ ∈ {0.50, 0.80, 0.95}**.

Setup: an email is *predicted genuine* if its score > τ, *predicted
fraud* otherwise. Confusion matrix:

```
                    PREDICTED: genuine (score > τ)    PREDICTED: fraud (score ≤ τ)
ACTUAL: genuine            TP                                  FN
ACTUAL: fraud              FP                                  TN
```

Where ACTUAL: fraud = the impostor pools (other ∪ synthetic).

### Per-threshold sub-metrics

For each τ ∈ {0.50, 0.80, 0.95} the dashboard reports:

#### `threshold_τ/recall` = TP / (TP + FN)

Fraction of *real genuine emails* the model lets through. Same as TPR.

> "If I deploy with cutoff τ, how many of my legitimate users' emails do
> I keep?"

#### `threshold_τ/fpr_other` = FP_other / (FP_other + TN_other)

Fraction of *random-other-sender impostors* that scored above τ — the
model wrongly believed they were the claimed sender.

#### `threshold_τ/fpr_synthetic` = FP_syn / (FP_syn + TN_syn)

Same but for synthetics. **This is your "how often do fraudsters slip
through?"** at this cutoff.

#### `threshold_τ/fpr_overall` = FP / (FP + TN) pooled

Pooled FPR over other+synthetic.

#### `threshold_τ/precision` = TP / (TP + FP)

Of the emails the model said "genuine," what fraction were actually
genuine? Pooled impostors used in the denominator.

> "When I trust the model's verdict, how often am I right?"

#### `threshold_τ/accuracy` = (TP + TN) / (TP + TN + FP + FN)

Plain accuracy. Less informative than the previous two because the
class balance in the probe set is artificial — accuracy is dominated by
whichever pool is larger.

#### `threshold_τ/report_rate`

Fraction of the entire query stream (genuine + impostors) the model
"reports on" by saying "score > τ". Useful to see how *picky* the model
is at this threshold. Report rate near 1.0 = model approves everything;
near 0.0 = model approves almost nothing.

### Why three thresholds?

The choice of 0.50 / 0.80 / 0.95 is somewhat arbitrary — they let you
inspect three quite different operating regimes:

| τ    | Regime              | Use-case                                            |
|------|---------------------|-----------------------------------------------------|
| 0.50 | very permissive     | "let everything through unless clearly off"         |
| 0.80 | balanced            | "ask for confirmation on borderline cases"          |
| 0.95 | very conservative   | "only let through if the encoder is highly confident" |

### CRITICAL: Why `linear_z3` scores never exceed ~0.95

The default score function is `score = max(0, 1 − z/3)` where
`z = (1 − cos_sim) / spread`. For `score > 0.95` you need `z < 0.15`,
which means the query is 6× *closer* to the centroid than the average
enrollment email. Statistically impossible for any real-world email
because the enrollment emails themselves don't get that close.

> **`threshold_0.95/*` metrics are usually all zero or near-zero by
> construction, NOT because the model is broken.**

This is why the V7 sweep evaluates several score functions side-by-side
and why the actually-useful operating-point metrics moved into
`op/all/fpr_*` and `op/synthetic/fpr_*` (next section).

---

## 9. FPR-anchored operating points: `op/all/fpr_*`, `op/synthetic/fpr_*`

Instead of fixing a score and reading off the metrics, **fix the FPR
and find the threshold that achieves it.** This is the deployment-relevant
view.

For each target FPR ∈ {0.01, 0.05, 0.10}:

1. Find τ such that exactly that fraction of impostors fall above τ.
   Computed as the `(1 − fpr)`-quantile of the impostor score
   distribution.
2. Report at that τ:
   - `op/{pool}/fpr_X/threshold` — the τ value itself
   - `op/{pool}/fpr_X/recall` — TPR = fraction of genuines admitted
   - `op/{pool}/fpr_X/precision` — TP/(TP+FP)
   - `op/{pool}/fpr_X/fpr_synthetic`, `fpr_other` — break-out of who the
     impostors were

Two anchor pools:

- `op/all/fpr_X/*` — τ chosen on pooled impostor distribution.
- `op/synthetic/fpr_X/*` — τ chosen on synthetic-only distribution.
  More relevant for fraud detection because that's the hard adversary.

### How to read these

`op/synthetic/fpr_0.05/recall = 0.675` means:

> "If we calibrate the threshold so 5% of synthetic-fraudsters pass
> through, we keep 67.5% of legitimate emails."

That's the bottom-line operational quality of the system. The same number
appears as `tpr_at_fpr/synthetic_5pct` (see §11) — they should match.

---

## 10. Partial AUC: `pauc/*`

Full AUROC weights all FPRs equally — including FPR=0.9, which nobody
will ever deploy at. Partial AUC restricts the integral to FPR < 0.05
or < 0.10 and *renormalises* so the value still lives in [0, 1].

```
                  1 - normaliser
pAUC@5% = ────────────────────────────
              0.05 (the FPR cap)
```

(Implemented in `src/email_fraud/scoring/metrics.py` via the McClish
correction.)

You'll see:
- `pauc/genuine_vs_synthetic_5pct` — pAUC against synthetics, FPR ≤ 5%
- `pauc/genuine_vs_synthetic_10pct` — same, FPR ≤ 10%
- `pauc/genuine_vs_all_5pct`, `pauc/genuine_vs_all_10pct` — pooled
  impostors

These are more sensitive to improvements at the low-FPR end. If your
encoder mostly improved the *easy* impostor cases (FPR 0.3-0.9), full
AUROC moves but pAUC@5% doesn't. If it improved the *hard* tail (the
synthetics that almost looked real), pAUC@5% moves and full AUROC may
not.

For fraud detection where false alarms have real cost, **pAUC at small
FPR is the most relevant single ranking metric.**

---

## 11. TPR at fixed FPR: `tpr_at_fpr/*`

A complementary view to pAUC: instead of a single integral, report two
operating points.

- `tpr_at_fpr/synthetic_1pct` — TPR when synthetic FPR = 1%
- `tpr_at_fpr/synthetic_5pct` — TPR when synthetic FPR = 5%
- `tpr_at_fpr/all_1pct`, `tpr_at_fpr/all_5pct` — same on pooled
  impostors

**This is the metric I personally read first.** If
`tpr_at_fpr/synthetic_5pct = 0.7`, I know: at a threshold that lets 5%
of synthetics through, we admit 70% of real email. Customers can map
that straight onto their SLA.

Identical to `op/{pool}/fpr_X/recall` but reported separately for
historical/dashboard-organisation reasons.

---

## 12. Coverage at accuracy: `coverage/at_acc_*`

A selective-classifier view: instead of asking "at threshold τ, what's
my accuracy?" we ask "what's the *biggest* fraction of decisions we can
keep while maintaining a target accuracy?"

### How it's computed

1. Define confidence = |score − 0.5| (distance from the indecision
   point, assuming the score is in [0,1]). High confidence = far from
   0.5; low confidence = near 0.5.
2. Compute prediction = (score > 0.5) → "genuine", else "fraud".
3. Sort queries by confidence descending — most confident first.
4. For each prefix of length k, compute its accuracy. The accuracy
   curve usually starts high (top-k of confident predictions are mostly
   right) and trends downward as you include more low-confidence
   predictions.
5. `coverage/at_acc_T` = the largest k/N where running accuracy ≥ T.

So `coverage/at_acc_0.95 = 0.6` means: "we can confidently decide on
60% of queries with 95% accuracy; for the other 40% we'd want to
abstain."

The three thresholds {0.50, 0.80, 0.95} measure increasingly conservative
selective classifiers. `coverage/at_acc_0.50` is trivially close to 1.0
unless the model is broken (50% accuracy = random).

Beware: like the fixed-threshold metrics, this depends on the absolute
score value and the score function. It's mostly meaningful for `[0, 1]`
scorers — for `mahalanobis` (raw distance) it doesn't make sense.

---

## 13. Probe diagnostics: `probe/n_*`

Pure diagnostics — how many queries each pool actually had.

- `probe/n_genuine_queries` — should be `n_profile_senders × n_query_per_sender` = 30 × 4 = 120
- `probe/n_other_queries` — bounded by val-set size; 200 by default
- `probe/n_synthetic_queries` — bounded by available synthetics; 200 if enough exist

If any of these are 0 → the corresponding AUC will be NaN or absent from
the dashboard. Check the probe construction logs.

---

## 14. Sampler diagnostics: `train/sampler/*`

Reported by `SyntheticBalancedSampler` to show how each batch was
composed:

- `train/sampler/n_batches` — batches in this epoch
- `train/sampler/pool_paired_real` — count of real senders that had a
  synthetic counterpart available to pair them with
- `train/sampler/pool_real_only` — count of real senders WITHOUT a
  synthetic counterpart (used as fillers)
- `train/sampler/n_filler_paired_solo` — number of slots filled by
  unpaired real senders even though they had synthetics (because the
  per-batch n_syn quota was already filled)
- `train/sampler/n_filler_real_only` — number of slots filled by
  real-only senders
- `train/sampler/filler_fallback_fraction` — fraction of filler slots
  that came from `pool_real_only` rather than `pool_paired_real`
- `train/sampler/pair_only_senders` — senders that ONLY appear in pairs
  (no real-only path) this epoch

These are mostly internal but useful if you're debugging "why aren't my
synthetics influencing training" — if `n_filler_paired_solo` is huge,
you're not getting many synthetic-real contrast pairs per batch.

---

## 15. PAN verification metrics: `test/*`

Logged every 5 epochs from the inline PAN evaluation against
`data/processed/enron/test_pairs.jsonl` — a sender-disjoint test set
formatted as same/different-author pairs (PAN task).

Different from the centroid probe in a crucial way: **PAN pairs are
about discriminating two random emails, NOT about scoring against a
profiled centroid.**

- `test/AUC` — pairwise AUROC (§3a) on the test pairs
- `test/EER` — equal error rate (FAR = FRR); lower = better
- `test/c@1` — PAN 2011 metric that gives partial credit for abstention
- `test/F0.5u` — PAN 2019 metric; F-score with β=0.5 (precision-leaning),
  with unanswered questions counted
- `test/pAUC@5%`, `test/pAUC@10%` — pairwise pAUCs on test
- `test/TPR@FPR=1%`, `test/TPR@FPR=5%` — pairwise operating points

**Why both pair- and centroid-based metrics?** Pair-AUROC is the
historically-standard authorship verification metric and lets us
compare against published PAN baselines. Centroid-AUROC is the actual
product metric. The two should rise together but their absolute
numbers aren't directly comparable.

---

## 16. Sanity-check FAQ

### Why does `threshold_0.50/recall` drop while `threshold_0.80/recall` rises during training?

This is exactly the question you asked. Here's what's happening:

At early epochs, the model's score distribution is broad and unfocused —
many genuines score in the 0.55-0.75 range (above 0.50 but below 0.80).
- `threshold_0.50/recall` is high (most genuines pass 0.50)
- `threshold_0.80/recall` is low (few cross 0.80)

As training improves, the distribution **sharpens**: genuines drift
toward 1.0, impostors drift toward 0. But two things happen at once:

1. Some genuines that *were* at 0.6 might temporarily dip into 0.45 as
   the encoder rearranges them — they're still above the impostors but
   below 0.5. `recall@0.50` falls.
2. Meanwhile, a bunch of genuines that *were* at 0.7 now scale up to
   0.85+. `recall@0.80` rises.

The two thresholds are sampling different parts of the same distribution
shifting. Plotting `score/mean_genuine` alongside makes this visible —
if it's *rising*, your high-threshold recall will rise; if it's
*stable* but the *spread is shrinking* (variance decreasing), the
low-threshold recall can fall even as the model gets better.

This is mostly noise / artifact of the `linear_z3` score function
having the [0,1] bound. The AUC numbers (which are scale-free) are the
honest summary of "is the model getting better?"

### Why do my AUC numbers move up but my `threshold_0.95/*` numbers stay zero?

See §8 — `linear_z3` can't reach 0.95 except for queries 6× closer to
the centroid than the average enrollment email. Statistically impossible
on real data. The threshold band 0.95 is broken by design when paired
with `linear_z3`. Use `op/synthetic/fpr_0.01/*` for the
"very-conservative threshold" view instead.

### `val/loss` is climbing but `auc/genuine_vs_synthetic` is also climbing. Bad?

Not necessarily. SupCon loss depends on which in-batch negatives the
sampler picked; once the easy negatives are crushed, the loss starts to
focus on harder examples and can plateau or rise even as the embedding
quality keeps improving. **Trust the centroid AUROC, not val/loss, as
your improvement signal.**

### `score/synthetic_harder_than_other` is negative. Should I worry?

Yes. It means the model thinks LLM imitations are *less* like the
sender than random other people are — which usually points to a flaw in
the *synthetic* dataset, not in the model. Common causes:

- The LLM was prompted with a generic style and the outputs all look
  similarly bland — they cluster together in embedding space far from
  every real sender.
- The synthetic emails contain LLM-isms ("Certainly!", "I hope this
  finds you well!") that the model has learned as a "tell."

In both cases the model is "cheating" — it would NOT generalise to
better LLMs or human impersonators. Worth investigating.

### How do I pick "the threshold to deploy at"?

Decide your tolerance for false positives on the synthetic set. For
fraud detection that's usually ~1-5%. Read off
`op/synthetic/fpr_0.05/threshold` from the dashboard — that's the
score value you'd use as your deployment cutoff. Read off
`op/synthetic/fpr_0.05/recall` to see what fraction of real email
you'd keep.

For a binary keep/flag decision you may also want a *second* threshold
where you escalate to manual review — a "yellow" band between
`op/synthetic/fpr_0.10/threshold` (definitely fine) and
`op/synthetic/fpr_0.01/threshold` (highly suspicious). Anything in
between → human-in-the-loop.

### Which single number should I look at to know if v7 beat v6?

Two:
1. **`auc/genuine_vs_synthetic`** — the rank-quality on the hardest case.
2. **`tpr_at_fpr/synthetic_5pct`** — the deployment quality at the
   threshold you'd actually use.

If both went up, v7 is better. If only AUC went up, v7 reordered the
score distribution but didn't improve the operating point — a fine
training-time signal but no immediate product win.
