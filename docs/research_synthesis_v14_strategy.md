# Drawing-board synthesis → v14 strategy: how to reach production-grade low-FPR detection

*2026-06-18. Literature review + strategic pivot after the v13 plateau. Companion to
`docs/findings_and_roadmap.md` and `docs/v13_results.md`. Research gathered via the
deep-research harness (15 primary arXiv sources, 72 claims); the harness's automated
verification phase was knocked out by a session rate-limit, so the two load-bearing,
surprising claims were re-verified by hand against the source PDFs/HTML (noted inline).*

---

## 0. The problem, restated honestly

We have spent v11→v13 improving a **hybrid** model that does two things at once:

- **(A) Authorship verification** — "is this *this specific person's* idiolect?"
  Generator-**invariant** by construction: a forgery fails to match the person no
  matter which LLM wrote it.
- **(B) Generator-artifact detection** — "does this text smell LLM-written?"
  Learned as a trained head on synthetic hard-negatives.

v13 proved **(B) plateaus**: a classifier trained to reject LLM text learns
generator-specific artifacts, and adding generators has diminishing returns
(Mistral-only TPR@1% 0.91→0.045 off-generator; 2 gens → pool AUC 0.87; 3rd gen → no
gain, Gemini stuck ~0.75). The literature says this is **not a tuning problem — it's
the nature of trained MGT classifiers.** We were climbing the wrong hill.

---

## 1. What the literature actually says (cited)

### 1.1 Trained generator-artifact detectors do NOT generalize — stop adding generators
- Fine-tuned classifier detectors are **severely train-distribution-biased**: a RoBERTa
  detector at 95%+ on its training generator "rarely exceeds 60%" on the same domains
  from a *different* model (RAID, [arXiv:2405.07940]).
- Adapting detectors to new/unseen LLMs (class-incremental) stays far below full
  training (~0.59–0.66 vs ~0.81–0.89 F1) — "a hard, unsolved moving target"
  ([arXiv:2412.17242]). Holding out GPT-4 / Llama-3 causes the **largest** drops
  ([arXiv:2507.00838]).
- **Implication:** our v13 result is the expected one. Do **not** invest further in a
  trained generator-classifier head or a 4th training vendor.

### 1.2 But generator-*agnostic* zero-shot detectors sidestep the moving target
- **Binoculars** ([arXiv:2401.12070]): zero-shot, **no per-generator training** — scores
  perplexity normalized by the cross-perplexity of two closely-related base LLMs.
  Reported **>90% detection at 0.01% FPR** across modern LLMs incl. generators never
  seen in development. (Caveat: numbers are on their benchmark/domains; frontier-2026
  models + short email text need our own measurement.)
- **Fast-DetectGPT** ([arXiv:2310.05130]): zero-shot "conditional probability
  curvature," ~75% relative over DetectGPT in **both** white- and black-box settings.
- **Why this matters:** these don't *train on generators at all*, so they have no
  moving-target failure mode the way our trained head does. They are the right way to
  get an "LLM-ness" signal — as a **frozen** input feature, not a learned classifier.

### 1.3 Authorship verification is robust to LLM mimicry — the generator-invariant backbone
- **Verified by hand ([arXiv:2505.14195]):** LLM mimicry of a target author *remains
  detectable* — verification accuracy **0.59–0.89** on mimicked text; "person-specific
  stylistic traits cannot be reliably forged to completely evade verification."
- A second study ([arXiv:2603.29454]) reports prompted GPT-4o impersonations **failed
  to bypass** AV systems and were rejected *more* reliably than genuine different-author
  negatives — on **Enron** specifically, LUAR's rejection rate on LLM impersonations
  improved ~69% relative. Proposed mechanism: LLM text is systematically more lexically
  diverse / higher-entropy / less redundant than human writing, so the verifier is
  *also* implicitly catching LLM-ness.
- **Implication:** the authorship signal is the durable backbone; mimicry attacks (the
  scariest threat) do not defeat it.

### 1.4 …but OUR backbone (LUAR) confounds content with style — this is our real bug
- **Verified by hand ([arXiv:2410.12757], StyleDistance):** "the LUAR model considers
  both style **and content**, hence confounding" them. On content-independence
  evaluations LUAR underperforms purpose-built content-independent embeddings
  (StyleDistance 0.46 vs LUAR 0.38 on their style-eval; LUAR scores near-floor on the
  strict STEL-or-Content probe).
- **This directly explains our open failures:** PAN cross-topic collapses to 0.78 and
  length/register shifts hurt **because the embedding leans on topic/content**, not pure
  idiolect. Fixing content-disentanglement is the single highest-leverage *model* change.
- **How to fix it (three cited, composable levers):**
  1. **Content-controlled positives.** StyleDistance trains on **near-exact synthetic
     paraphrases with controlled style variation** (40 features) so positives differ in
     *style only*; the **synthetic-only** variant still transfers to natural text
     ([arXiv:2410.12757]). We can generate these with our existing OpenRouter pipeline.
  2. **Conversation/content-controlled negatives.** Sampling different-author pairs from
     the *same conversation* isolates style from content and lifts AV AUC 0.58→0.69
     ([arXiv:2204.04907]).
  3. **All-layers pooling.** Using all transformer layers (not just the last) improves
     **out-of-domain** authorship attribution in 15/16 settings, up to ~50% MRR for LUAR
     transferring Fanfiction→Reddit ([arXiv:2503.00958]) — a cheap architectural tweak.

### 1.5 Low-FPR is achievable but needs explicit calibration
- **Conformal prediction** gives a *statistical guarantee* that FPR ≤ chosen α. MCP
  ([arXiv:2505.05084]) holds FPR within bounds down to α=0.5% and reports large TPR
  gains under a strict 0.5% FPR constraint (e.g. +157% TP@0.5% on MAGE). This is the
  right tool for our "1% FPR budget" north-star — and it composes with per-author
  thresholds (the v9@ep10 lesson: anchor per-sender, always report FPR_other).

---

## 2. The strategic call: FUSE (iii), with a reframe — not (i), not pure (ii)

**Verdict.** Evidence favors a **two-signal fusion**, but the composition is the
opposite of what we've been building:

| Signal | Old (v11–v13) | New (v14) | Why |
|---|---|---|---|
| LLM-ness | **trained** generator-classifier head (moving target, plateaus) | **frozen zero-shot** detector (Binoculars / Fast-DetectGPT) | generator-agnostic by construction; no retrain per vendor (§1.1–1.2) |
| Authorship | LUAR+LoRA centroid (content-confounded) | **content-disentangled** style embedding | fixes domain/length OOD; robust to mimicry (§1.3–1.4) |
| Decision | per-sender z threshold | **conformal-calibrated fusion** of both, per-author | guaranteed ≤1% FPR (§1.5) |

- **Why not (i) keep improving MGT detection:** §1.1 — trained detectors are a moving
  target; we already hit the plateau.
- **Why not (ii) pure authorship only:** authorship is the backbone, but a free,
  orthogonal, generator-agnostic LLM-ness signal strictly helps at low FPR and is cheap
  to add (frozen, no training). Dropping it leaves catch-rate on the table.
- **Failure modes to watch:** (a) zero-shot detectors degrade on very short text and
  newest frontier models — must measure on *our* short emails, not trust paper numbers;
  (b) content-disentangled embeddings can *lose* discriminative power if over-regularized
  — verify AV AUC doesn't drop; (c) fusion can overfit the calibration set — use
  conformal with a held-out calibration split.

---

## 3. v14 plan — validate cheaply BEFORE any retrain

The cycle's lesson is to not burn GPU on faith. Two **eval-only, zero-training** probes
test the two new pillars first; retrain only what they justify.

### Stage A — *(cheap, eval-only)* Does a frozen zero-shot detector beat our trained head off-generator?
Run **Fast-DetectGPT** (and/or Binoculars) on our held-out **Claude+Gemini** set and the
`gen:*` slices, score TPR@1%/AUC, compare to v12/v13's trained-head numbers.
- **Win condition:** zero-shot TPR@1% on Gemini/Claude materially beats v13's ~0.13.
- **If it wins:** the LLM-ness signal is solved for free → fold in as a frozen feature.
- Implement as a new `scripts/eval_zeroshot_detector.py` reusing the held-out pairs.

### Stage B — *(cheap, eval-only)* Does a content-independent embedding fix our domain/length OOD?
Swap a pretrained **StyleDistance** (or Wegmann content-independent style) embedding into
`eval_ood.py` as an alternative encoder; measure PAN cross-topic, blog, and `lenmix`
vs LUAR's 0.78 / 0.84 / 0.93.
- **Win condition:** cross-topic AUC rises meaningfully above 0.78 with no AV-base loss.
- **If it wins:** justifies retraining/finetuning on a content-disentangled backbone.

### Stage C — *(retrain, only if A/B justify)* v14 model
- Backbone per Stage B result (StyleDistance-style finetune, or LUAR + **all-layers
  pooling** + **content-controlled positives** generated via OpenRouter).
- **Drop the trained LLM-detector head**; replace with the frozen Stage-A signal.
- Keep the anti-Goodhart monitor + crop/short-mail recipe that already works.

### Stage D — *(fusion + calibration)* the production operating point
Fuse authorship-distance + frozen LLM-ness via a small calibrated combiner; apply
**conformal/per-author thresholds** for a guaranteed ≤1% FPR. Report TPR@1% **and**
FPR_other per held-out generator.

### Stage E — *(eval rigor, prerequisite for trusting C/D)*
The held-out probe is **41–44 senders** → ±0.12 TPR@1% CIs that swamp real deltas. Before
claiming any v14 win: enlarge the unseen-sender probe (more identities; consider a second
corpus), and add explicit **held-out-generator**, **cross-domain**, and **mimicry-attack**
(LLM-prompted-with-victim-emails) eval slices — the mimicry slice is the threat we most
need to measure and currently don't.

---

## 4. Highest-leverage changes, ranked

1. **Content-disentangled style backbone** (Stage B→C) — fixes the *root cause* of
   domain/length OOD; the biggest single lever (§1.4).
2. **Frozen zero-shot LLM-ness signal** (Stage A) — replaces the plateaued trained head
   with a generator-agnostic one, possibly for free (§1.2).
3. **Conformal + per-author calibration** (Stage D) — converts AUC into a trustworthy
   1%-FPR operating point (§1.5).
4. **Bigger, threat-complete eval** (Stage E) — without it we can't *measure* 1–3.
5. **Stop**: 4th training vendor / more trained-head capacity — proven dead end (§1.1).

## 5. Sources
[2405.07940] RAID · [2412.17242] cross-gen/incremental MGT · [2401.12070] Binoculars ·
[2310.05130] Fast-DetectGPT · [2505.05084] MCP conformal · [2505.14195] mimicry detectable
(hand-verified) · [2603.29454] impersonation vs AV (Enron) · [2410.12757] StyleDistance /
LUAR content-confound (hand-verified) · [2204.04907] content-controlled AV ·
[2503.00958] all-layers OOD attribution · [2507.00838] stylometric MGT brittleness.

*Caveat: the deep-research harness's adversarial verification did not run (session
rate-limit); claims here are from primary sources, with the two pivot-critical ones
re-verified by hand. Remaining numbers should be treated as literature-reported, to be
reproduced on our data in Stage A/B before we depend on them.*
