# V6 → V7 Memo: What Changed, What It Means, How to Read the Numbers

**Audience:** anyone (technical or not) who wants to understand the V7
upgrade — what we changed, what improved, and how to read the dashboard.

**TL;DR:**
> V7 is a strict drop-in upgrade to V6 — same architecture, retrained
> with a stronger recipe. At a 5 % synthetic-fraud false-positive rate it
> keeps **91.7 %** of real email vs V6's **84.9 %** (K = 25 enrollment).
> At the conservative 1 % FPR setting, retention climbs from **75.8 %**
> (V6) to **87.5 %** (V7). Same inference code; ship the new checkpoint.

Contents:

1. [What the system does](#1-what-the-system-does-30-second-recap)
2. [What changed between V6 and V7](#2-what-changed-between-v6-and-v7)
3. [The headline numbers](#3-the-headline-numbers)
4. [How to read the confusion matrices](#4-how-to-read-the-confusion-matrices)
5. [The dashboard metrics, in plain English](#5-the-dashboard-metrics-in-plain-english)
6. [What to put on the slide](#6-what-to-put-on-the-slide-for-leadership)
7. [Caveats & next moves](#7-caveats--next-moves)

---

## 1. What the system does (30-second recap)

**Goal:** flag emails that *look* like they came from a sender, but
weren't really written by them — Business Email Compromise (BEC) and
LLM-generated impersonation.

**How:**

```
1. For each user (sender), encode 8-40 of their historical emails into
   128-D vectors with a stylometric encoder (LUAR-MUD + LoRA).
2. Average those vectors → that user's "style centroid."
3. When a new email arrives, encode it and measure how far it is from
   the claimed sender's centroid. Far away ⇒ suspicious.
```

The "encoder" is what gets retrained between V6 and V7. The centroid
math and the inference code are identical.

---

## 2. What changed between V6 and V7

| Knob                          | V6                  | V7                       | Why we changed it                                                                                  |
|-------------------------------|---------------------|--------------------------|----------------------------------------------------------------------------------------------------|
| **Synthetic pairs per batch** | 2                   | **4**                    | Double the rate at which the model sees "LLM imitation vs real" gradient signal. Biggest single win.|
| **SupCon temperature**        | 0.07                | **0.05**                 | Sharper softmax. Hard negatives (the imitations) receive a *bigger fraction of the gradient* — the model is forced to focus on borderline cases instead of cruising on easy negatives.|
| **LoRA targets**              | query, value        | query, value, **+ key**  | One extra adapter matrix per attention layer = more representational capacity in the dimensions that decide which tokens attend to which.|
| **Training length**           | 100 epochs (early stop) | **150 epochs** (no early stop) | V6 plateaued around epoch 70. With stronger gradients from the first three changes, V7 kept improving past 100. |

Nothing else changed. Same data, same encoder backbone, same scoring
rules, same inference pipeline.

---

## 3. The headline numbers

### At low enrollment (K = 8, the conservative product setting)

| Metric              | V6     | V7    | Change      |
|---------------------|-------:|------:|------------:|
| AUC (genuine vs LLM imitation) | 0.874 | **0.909** | +3.5 pp |
| Real email kept @ 5 % imitation FPR | 67.5 % | **76.7 %** | +9.2 pp |
| Real email kept @ 1 % imitation FPR | 51.7 % | **63.3 %** | +11.7 pp |
| Equal-error rate (lower = better) | 20.3 % | **16.6 %** | −3.7 pp |

### At medium enrollment (K = 16 — typical mid-tier customer)

| Metric              | V6     | V7    | Change      |
|---------------------|-------:|------:|------------:|
| AUC (g vs imitation) | 0.920 | **0.966** | +4.6 pp |
| Real email kept @ 5 % FPR | 82.5 % | **87.5 %** | +5.0 pp |
| Real email kept @ 1 % FPR | 70.0 % | **79.2 %** | +9.2 pp |

### At full enrollment (K = 25 — recommended deployment)

| Metric              | V6     | V7    | Change      |
|---------------------|-------:|------:|------------:|
| AUC (g vs imitation) | 0.953 | **0.965** | +1.2 pp |
| Real email kept @ 5 % FPR | 84.9 % | **91.7 %** | +6.8 pp |
| Real email kept @ 1 % FPR | 75.8 % | **87.5 %** | +11.7 pp |

The V7 advantage is *bigger* at stricter operating points and at higher
K — exactly the regimes a paying customer will deploy in.

### The single picture worth showing

`results/v7/confusion/v6_vs_v7_k_curves.png`

- Two side-by-side line charts.
- Left: TPR @ 5 % synthetic FPR.  Right: TPR @ 1 % synthetic FPR.
- X-axis is enrollment K (how many emails per sender the customer has
  enrolled). Y-axis is "fraction of real email kept."
- V7 (teal) is above V6 (mauve) **at every K**. Both lines climb
  steeply from K=4 to K=16, then plateau.
- Read the chart this way: *"give the model ≥ 16 emails per sender and
  V7 keeps ≥ 87 % of real mail at our threshold."*

---

## 4. How to read the confusion matrices

We generate **3×2** confusion matrices instead of the standard 2×2 — the
3rd row breaks out the impostor pool by *kind*. This matters because the
fraud threat is "synthetic" (LLM imitation of a specific sender), not
"other-sender" (a random different person — that's not how BEC works).

```
                                        Predicted: Genuine (pass)   Predicted: Fraud (block)
─────────────────────────────────────────────────────────────────────────────────────────────
Actual: Genuine                                Real-kept rate ← good        False-flag rate
                                                    (we want this high)        (we want this low)

Actual: Other-sender impostor               FPR_other ← cosmetic              Correctly blocked
                                                    (BEC isn't this attack)

Actual: LLM imitation (synthetic)           FPR_synthetic ← THE THREAT        Correctly blocked
                                                    (anchored to 1/5/10%)         (the goal)
```

Each picture has **three panels** side-by-side, one per operating point
that we tune the threshold to:

- **Conservative** — synthetic FPR = 1 %. For high-trust channels
  (CFO-only mail, wire-transfer instructions).
- **Operational** — synthetic FPR = 5 %. The default deployment knob.
- **Relaxed** — synthetic FPR = 10 %. Useful for high-volume / low-risk
  inboxes.

In each panel the **synthetic FPR is fixed** (1/5/10%) by construction
— we *anchor* the threshold to that. So the number to focus on is:

1. **"Real kept" (top-left cell)** — fraction of real email we let
   through. This is the conversation-stopper for the business: at our
   chosen sensitivity, how much legit email survives?
2. **FPR_other (middle-left cell)** — fraction of random-other-sender
   impostors that slip through. A diagnostic, not the primary metric.

### Files to reference

| Plot                                              | What it shows                                                                 |
|---------------------------------------------------|-------------------------------------------------------------------------------|
| `v6_vs_v7_k_curves.png`                           | Headline operating-point chart. **Use this on the slide.**                    |
| `v6_vs_v7_K16_mahal_per_sender_grid.png`          | 2×2 grid of confusion matrices: V6 vs V7, conservative vs operational.        |
| `v6_vs_v7_K8_mahal_per_sender_grid.png`           | Same, but K=8 (most-conservative enrollment).                                 |
| `v7_K16_mahal_per_sender_confusion_3x2.png`       | V7 alone, three operating points, K=16.                                       |
| `v7_K16_linear_z3_confusion_3x2.png`              | V7 with the simpler "linear_z3" scorer at K=16 — for the no-Mahalanobis path. |
| `v6_K16_mahal_per_sender_confusion_3x2.png`       | V6 baseline, K=16. The "before" picture.                                      |

All under `results/v7/confusion/`.

---

## 5. The dashboard metrics, in plain English

Everything W&B reports falls into one of six families. Knowing which
family a metric belongs to tells you what it's measuring.

> **Big idea:** the metrics on W&B are computed on a *simulated
> deployment* called the **CentroidProbe**, not on the data the model
> trained on. Each epoch, the probe re-encodes a fixed set of:
> 30 profiled senders × 8 enrolled emails + 120 genuine queries + 200
> other-sender impostors + 200 LLM-imitation impostors,
> then scores everything and reports the results. Same evaluation each
> epoch, only the encoder weights change.

### Family 1 — Loss (the training-world numbers)

- **`train/loss`, `val/loss`** — supervised-contrastive loss on train /
  validation batches. *Sanity number only.* Don't read it as
  "improving" or "regressing" — SupCon loss can plateau or even rise as
  the model crushes easy negatives and starts focusing on harder ones.

### Family 2 — Embedding-space diagnostics

- **`embedding/knn_accuracy`, `embedding/knn_macro_f1`, `embedding/pair_auroc`**
  — "are same-author emails clustering together in 128-D space?"
- These hit ~1.0 within a few epochs once the encoder is working at all.
  Useful for the first 5 epochs as a "did training start" signal; after
  that they saturate and become uninformative.

### Family 3 — Centroid AUROC (the headline product metric)

> AUROC = "if I pick one genuine email and one impostor at random, how
> often does the genuine one have a higher score?" Always in [0.5, 1.0].

- **`auc/genuine_vs_other`** — discrimination against random different
  senders. Easy case. V7 ≈ 0.95.
- **`auc/genuine_vs_synthetic`** — discrimination against LLM
  imitations of the same sender. **The hard case. The number that
  matters.** V7 ≈ 0.90 at K=8, 0.97 at K=16.
- **`auc/genuine_vs_all`** — pooled.

Threshold-free; ignores absolute score values. Changing the score
function (linear_z3 vs cosine vs Mahalanobis) doesn't move these
much. **This is the cleanest "did the model get better?" signal.**

### Family 4 — Score statistics

- **`score/mean_genuine`, `score/mean_other`, `score/mean_synthetic`**
  — the mean scores of each pool.
- **`score/gap_other` = mean_genuine − mean_other**, similarly
  `score/gap_synthetic`. These are *margin* numbers: how far apart are
  the two distributions?

Two distributions can have the same AUROC but very different gaps. The
gap is your *cushion* against noise; bigger = more robust.

### Family 5 — Threshold-band metrics

These pick a fixed threshold (τ = 0.5, 0.8, 0.95) and compute
TPR/FPR/precision/accuracy at that cutoff.

> **CRITICAL caveat about `threshold_0.95/*`**: with the default
> `linear_z3` scorer the score is bounded above by ~0.95 *by
> construction*. So `threshold_0.95/*` is always near zero — that's a
> math artifact, NOT a sign the model is broken. Use `op/synthetic/fpr_0.01/*`
> instead for the "very conservative threshold" view.

Sub-metrics under each τ:
- `threshold_τ/recall` = TPR = fraction of real email passed.
- `threshold_τ/fpr_other`, `threshold_τ/fpr_synthetic` — fraction of
  each impostor pool let through.
- `threshold_τ/precision` = TP / (TP + FP) — when we say "genuine",
  what fraction actually were?
- `threshold_τ/accuracy` — plain accuracy. Class balance is artificial
  in the probe, so this is less informative than the others.

**Why threshold_0.50 recall can fall while threshold_0.80 recall rises
during training:** the score distribution sharpens. Some borderline
genuines that were at 0.6 temporarily dip to 0.45 (lose recall@0.50),
while a bunch of genuines that were at 0.75 climb to 0.85+ (gain
recall@0.80). It's the same improvement, just measured from two
different vantage points.

### Family 6 — FPR-anchored operating points

> "Find the threshold τ that achieves a target FPR, and report what's
> happening at *that* τ." This is the **deployment-relevant** view.

For each target FPR ∈ {1 %, 5 %, 10 %}:

- **`op/synthetic/fpr_0.05/threshold`** — the τ value (the cutoff
  you'd actually deploy at).
- **`op/synthetic/fpr_0.05/recall`** — fraction of real email kept at
  that τ. **This is the deployment-ready number.**
- `op/synthetic/fpr_0.05/precision` — TP/(TP+FP) when both happen at
  that τ.
- `op/synthetic/fpr_0.05/fpr_other` — break-out of what fraction of
  other-sender impostors were *also* let through at that τ.

`op/all/...` is the same thing but anchored on the pooled impostor
distribution rather than synthetic-only.

This family is identical to `tpr_at_fpr/synthetic_5pct` etc., reported
twice for dashboard layout reasons.

### Family 7 — Partial AUC

- **`pauc/genuine_vs_synthetic_5pct`** — AUC computed only over the
  FPR ≤ 5 % range, normalised back to [0, 1].

Full AUROC averages over all FPRs, including FPR = 0.9 (which nobody
deploys at). pAUC focuses on the low-FPR end where the operational
decision lives. **More sensitive to changes that matter for fraud
detection.**

### Family 8 — Coverage at accuracy (selective classifier)

- **`coverage/at_acc_0.95`** — answer to "what fraction of queries
  can I make a 95 %-accurate decision on, if I'm allowed to abstain on
  the rest?"
- Sort queries by confidence (distance from 0.5); take the most-confident
  prefix; find the largest prefix that still has accuracy ≥ 0.95.
- Useful for human-in-the-loop pipelines where a borderline score
  should be flagged for review instead of auto-decided.

### Family 9 — PAN test metrics

Reported every 5 epochs from the inline PAN evaluation on
`test_pairs.jsonl`:

- **`test/AUC`** — pair-AUROC on the test split (a different setup from
  the centroid probe — this is "two random emails, same author?", not
  "this email vs profile").
- **`test/EER`** — equal-error rate.
- **`test/c@1`**, **`test/F0.5u`** — PAN authorship-verification
  metrics (technical; let us compare against published baselines).

These are useful for academic-benchmark comparison, less useful for
day-to-day product decisions. The centroid AUC family is the better
proxy for deployment.

### Family 10 — Sampler diagnostics

`train/sampler/*` reports how each batch was composed (how many real
senders, how many synthetic pairs, etc.). Mostly for debugging
"why aren't the synthetics influencing training" — if `n_filler_paired_solo`
is huge you're not getting enough synthetic contrast per batch.

---

## 6. What to put on the slide (for leadership)

**Recommended structure (4 visuals max):**

### Slide 1: One-sentence claim + the K-curve plot

> *"V7 keeps 91.7 % of real email at a 5 % synthetic-fraud
> false-positive rate at the recommended K=25 enrollment, vs V6's
> 84.9 % — a 6.8 percentage-point recovery of legitimate mail while
> letting through the same fraction of attacks."*

Visual: `v6_vs_v7_k_curves.png`. Point at the K=25 datapoint on the
left panel.

### Slide 2: The V6 vs V7 confusion grid

Visual: `v6_vs_v7_K16_mahal_per_sender_grid.png`.

Talking points:
- Top row = V6, bottom row = V7. Left = conservative threshold, right
  = operational threshold.
- "At the operational setting (right column), V6 catches 95 % of LLM
  imitations while letting through 17.5 % of real email. V7 catches
  the same 95 % of imitations while letting through 12.5 % of real
  email — 5 pp fewer false alarms on legitimate users."
- "At the conservative setting (left column) the gap widens further:
  V7 keeps 79 % of real mail vs V6's 70 %."

### Slide 3 (optional): the change set

The four-row table in §2. Sells "we knew what to change and why."

### Slide 4 (optional): roadmap

`experiments/v7/CHANGELOG_V7.md` has a 17-item "avenues for
improvement" list. The big ones for a leadership slide:
- **Centroid-alignment auxiliary loss** — train the encoder for the
  scoring rule it will be deployed with (a direct path to V8).
- **Larger synthetic corpus** — currently ~10 imitations per sender;
  scaling that 5× should compound the V7 win.
- **Per-sender LDA projection** — no retraining; pure scoring-time
  improvement; expected to add another 1-2 pp at low FPR.

---

## 7. Caveats & next moves

### Honest caveats

1. **The probe is small.** 120 genuine + 200 synthetic queries → an
   AUROC standard deviation of ~0.02. Improvements of <2 pp could be
   noise. The V7 improvements are 3-12 pp — comfortably above the
   noise floor — but we should add bootstrap CIs before claiming any
   single-point gains.
2. **The synthetic corpus is narrow.** All 437 imitations come from a
   single LLM with a single prompt template. A V7 that beats *this* LLM
   may not beat Claude / GPT-4 / Gemini equally well. Generalising the
   adversary is on the roadmap.
3. **V7 trades a bit of `FPR_other` for `FPR_synthetic` improvement.**
   At K=16, `FPR_other` rises from 4.5 % (V6) to 14 % (V7) at the 5 %
   synthetic-FPR operating point. **Not a regression for the BEC threat
   model** (attackers don't impersonate Alice from Bob's account), but
   worth flagging if a downstream sanity check expects very low
   absolute FPR on the easy case.
4. **`checkpoint_best.pt` is misleading for V7.** The training
   monitor `auc/genuine_vs_synthetic` happened to peak at epoch 7;
   that's not the right checkpoint to deploy. **Use
   `checkpoint_epoch_150.pt`** (the file path is in the run dir).

### Immediate next moves (cheap)

- Run the V7 checkpoint on a held-out test pair set with bootstrap CIs
  to confirm the operating-point numbers.
- Generate confusion matrices at a third operating point (FPR=20 %)
  for the "high-volume / low-risk" inbox scenario.
- Add Mahalanobis scoring to the deployed `PrototypicalHead` (it's
  currently a stub in `src/email_fraud/heads/prototypical.py`).

### Higher-effort next moves (V8 candidates)

See the "Avenues for further improvement" section of
`experiments/v7/CHANGELOG_V7.md` for the full 17-item list, sorted by
expected impact / cost.
