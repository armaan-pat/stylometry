# v14 cheap-validation results: both drop-in pivots fail — the path narrows to authorship

*2026-06-18. Eval-only Stage A/B from `docs/research_synthesis_v14_strategy.md`, run
BEFORE any retrain to avoid burning GPU on faith. Both validations are clean negatives
that **redirect** the strategy (they don't refute the research insight — they kill the
cheap drop-in *form* of it). Results in `results/v14/`.*

## TL;DR

| Pivot tested (drop-in, no training) | Result | Verdict |
|---|---|---|
| **Stage A:** frozen zero-shot MGT detector (Fast-DetectGPT *and* Binoculars) on held-out Claude+Gemini | pooled AUC **0.49–0.54**, len-matched **0.52–0.54**, Claude **< chance**, TPR@1% ~0.01–0.03 | **Dead for our threat.** Generic LLM-text detection cannot catch few-shot *imitation* emails. |
| **Stage B:** off-the-shelf content-independent embedding (StyleDistance) on the same OOD pairs | PAN cross-topic **0.748** vs LUAR **0.779**; blog **0.793** vs **0.839**; worse on every slice | **No free lunch.** A foreign style embedding does not beat our finetuned LUAR, even OOD. |

**Net:** the two-signal v14 plan's *cheap* form doesn't work. The evidence-based path to
production-grade narrows to **one thing: make the finetuned authorship backbone more
content/length-invariant via targeted retraining** — because (a) authorship is the only
signal that catches imitations at all, and (b) neither drop-in alternative helps.

---

## 1. Stage A — zero-shot MGT detection is ≈ chance on imitation emails

Two independent zero-shot detectors, same held-out Claude+Gemini set v12/v13 used, same
metrics (`results/v14/zeroshot_fastdetectgpt_heldoutCG.json`,
`zeroshot_binoculars_heldoutCG.json`). Authenticity = human-ness (higher = genuine);
genuine pool is sender-matched real Enron emails.

| Detector | Claude AUC | Gemini AUC | Pooled AUC | Pooled (len-matched) | TPR@1% |
|---|---|---|---|---|---|
| Fast-DetectGPT (GPT-Neo-1.3B) | **0.399** | 0.693 | 0.538 | 0.536 | 0.012 |
| Binoculars (Qwen2.5-1.5B base/instruct) | **0.398** | 0.582 | 0.485 | 0.523 | 0.016 |
| *v12/v13 authorship signal (for ref)* | *0.83* | *0.75* | *0.86* | — | *0.13–0.16* |

**Both detectors put Claude *below chance*** — Claude's imitations look *more* human (lower
perplexity / higher Binoculars) than real, messy Enron email. This is not a weak-model
artifact: two different methods agree after length-control. **Why:** our synthetics are
*few-shot imitation* emails (generated with the victim's real emails as style context to
impersonate them). That is exactly the adversarial case the MGT literature says defeats
zero-shot detection — the text is crafted to be fluent, low-perplexity, and human-like.

**Consequence for the architecture:** the "frozen zero-shot LLM-ness signal" pillar of the
v14 plan is **dead for this threat model**. You cannot catch a competent imitation as
"generic LLM text," because it isn't generic. This also reframes what the v12/v13 model's
0.86 actually is: not true "LLM-ness" detection, but **generator-artifact** sensitivity
layered on an authorship model — which is precisely why it plateaus and doesn't generalize
to new generators (v13). The durable signal is **person-specific authorship**.

## 2. Stage B — off-the-shelf content-independent embedding doesn't beat finetuned LUAR

StyleDistance over the exact OOD pair files (`results/v14/styledistance_ood.json`), vs
LUAR's committed v12 numbers:

| Slice | StyleDistance AUC | LUAR v12 AUC | Δ |
|---|---|---|---|
| domain:pan20_xtopic (cross-topic) | 0.748 | 0.779 | **−0.031** |
| domain:blog | 0.793 | 0.839 | −0.046 |
| len:short / lenmix:short_long | 0.641 / 0.580 | 0.931 / 0.931 | −0.29 / −0.35 |
| gen:claude / gen:gemini | 0.560 / 0.583 | 0.829 / 0.746 | −0.27 / −0.16 |

StyleDistance is **worse everywhere**, including the cross-topic slice where its
content-independence was supposed to help. **Caveat:** StyleDistance is off-the-shelf
(trained on synthetic paraphrases, never on email); LUAR-MUD is Reddit-pretrained +
LoRA-finetuned on Enron, so it has a large domain/finetuning advantage. The clean lesson
is **not** "content-independence is wrong" — it's that the fix **cannot be a drop-in
foreign model**. Content-disentanglement has to be trained *into our own* encoder.

## 3. What this means — refined v14 direction

The cheap validations did their job: they **eliminated two expensive dead-ends** before we
spent GPU on them. The path to production-grade is now specific:

**Keep** the finetuned-LUAR authorship backbone — it is already the best signal we have and
is the only one that catches imitations (gen: Claude 0.83 / Gemini 0.75), and it is
generator-*invariant* (unlike the plateaued synthetic-artifact augmentation).

**Invest** in the authorship signal's **content/length invariance**, in training (not as a
swap). Concrete, cited levers (from `research_synthesis_v14_strategy.md`), now the primary
plan:
1. **Content-controlled positives** — generate near-exact paraphrases of a sender's emails
   that vary *style-held, content-changed* and *content-held, style-changed*, so the
   contrastive objective is forced onto idiolect not topic (StyleDistance method, applied
   to our encoder; our OpenRouter pipeline can produce these).
2. **All-layers pooling** — use all transformer layers, not just the last; cited ~50% MRR
   OOD gain for LUAR Fanfiction→Reddit (arXiv:2503.00958). Cheap architectural change.
3. **Length curriculum / cross-length positives in training** — the genuine fix for the
   pairwise cross-length weakness (P3 enrollment matching was already refuted).

**De-prioritize:** more synthetic generators (v13 plateau) and any frozen zero-shot
detector (Stage A dead). The synthetic-hard-negative augmentation is a saturated secondary
lever.

**Prerequisite (Stage E):** the held-out probe is 41–44 senders (±0.12 TPR@1% CIs). Before
trusting any v14 retrain delta, enlarge the unseen-sender probe and add an explicit
**imitation-attack** eval slice (LLM prompted with a victim's real emails) — Stage A shows
that is the threat we most need to measure and currently fold into the generic synthetic pool.

## 4. Artifacts
- `scripts/eval_zeroshot_detector.py` (Fast-DetectGPT + Binoculars; eval-only)
- `scripts/eval_style_embedding.py` (any sentence-transformers encoder over OOD pairs)
- `results/v14/zeroshot_fastdetectgpt_heldoutCG.json`, `zeroshot_binoculars_heldoutCG.json`,
  `styledistance_ood.json`
