# v14 results: identity expansion SOLVES content-invariance, but synthetics are load-bearing for imitation-catching

*2026-06-18. Stage C/E from `docs/research_synthesis_v14_strategy.md` +
`docs/v14_validation_results.md`. v14 = the v12 authorship recipe trained on 844 authors
(Enron 44 + Blog Authorship Corpus 800), no synthetics, plain PKSampler. Model:
`runs/v14/lora/checkpoint_best.pt` (epoch 88, monitor `pauc/genuine_vs_other_5pct`).
Results in `results/v14/`. A clean split-of-effects result: one breakthrough, two
explained regressions, and an obvious synthesis (v14b).*

## TL;DR

| Axis | v12 | v14 | Read |
|---|---|---|---|
| **PAN cross-topic** (clean cross-domain OOD, NOT in training) | 0.779 AUC / 0.129 TPR@1% | **0.879 / 0.284** | **Breakthrough.** The stuck content-invariance problem moved (+0.10 AUC, TPR@1% 2.2×). |
| Blog held-out probe (150 unseen authors) | 0.786 | **0.916** | General authorship much stronger (partly in-domain now — see caveat). |
| Blog held-out, same-industry negatives | 0.788 | **0.914** | Content-control: not relying on topic. |
| **gen:Claude / gen:Gemini** (catch LLM imitation) | 0.829 / 0.746 | **0.541 / 0.573** | **Regressed to ~chance** — synthetics removed → no imitation signal. |
| Held-out Claude+Gemini pool (TPR@1%) | 0.129 | **0.042** | Same cause: lost the synthetic hard-negative signal. |
| Enron len/register slices | 0.92–0.95 | 0.57–0.71 | Regressed — 44 Enron senders diluted among 800 blog authors. |

**One sentence:** *identity diversity buys content-invariant authorship (the problem that was
stuck for the whole project), but it does not by itself catch LLM imitations — that came from
the synthetic hard-negatives we dropped. v14b must combine both.*

## 1. The breakthrough — content-invariance via identity expansion

The project's clearest-unsolved problem was content/topic confounding (the research traced
it to LUAR mixing style+content; symptom = PAN cross-topic stuck at ~0.78 for every prior
model). v14 expanded training from **44 → 844 authors**, forcing the contrastive objective
across far more authors × topics. Result on **PAN cross-topic fanfiction** (a clean test:
cross-topic *and* cross-domain, and **not in v14 training** — verified):

| | AUC | pAUC@5% | TPR@1% |
|---|---|---|---|
| v12 (44 authors) | 0.779 | 0.191 | 0.129 |
| **v14 (844 authors)** | **0.879** | **0.384** | **0.284** |

This is the first real movement on cross-topic authorship in the project. The
identity-expansion thesis (more identities decorrelate style from topic) is **validated**.
Blog held-out authors also jumped 0.786→0.916, though that is partly in-domain now (v14
trained on the blog corpus; test authors are disjoint but same domain) — **PAN is the clean
win**, blog is corroborating.

## 2. The regressions — both explained, both fixable

**(a) Imitation-catching collapsed (gen: 0.83→0.54, 0.75→0.57; pool TPR@1% 0.13→0.04).**
v14 dropped synthetic augmentation (to let PKSampler cover all 844 authors instead of the
SyntheticBalancedSampler bottlenecking the epoch on 44 synthetic pairs). Consequence: the
model never learned to push LLM imitations off-centroid. This **confirms** the v14-validation
finding — imitations are caught by the *synthetic-trained* signal, not by generic authorship
or zero-shot detection. The synthetics are load-bearing.

**(b) Enron-specific slices dropped (len/register 0.92–0.95 → 0.57–0.71).** The 44 Enron
senders are now 5% of training; the model spread capacity across 800 blog authors and lost
Enron in-domain sharpness. The deployment target is email, so this matters — v14 is
"blog-dominant."

## 3. Synthesis → v14b (the model that should reach performance-grade)

v14 and v12 each win on one axis; the production model needs both:

- **Identity diversity** (v14) → content-invariant authorship backbone.
- **Synthetic hard-negatives** (v12) → imitation-catching.
- **Domain balance** → keep email (Enron) as the primary domain, blog as identity/topic
  diversity, not a takeover.

**v14b plan:**
1. **Fix the sampler** so synthetic hard-negatives AND 844 authors coexist: either make
   `SyntheticBalancedSampler.__len__` scale to the real-author pool (not the 44 synthetic
   pairs), or oversample Enron+synthetics while interleaving blog authors each epoch. This is
   the key code change.
2. **Domain-balanced sampling** — upweight Enron senders (and/or generate synthetics for a
   subset of blog authors so the imitation signal isn't Enron-only).
3. Keep the anti-Goodhart `min(other,synthetic)` monitor (synthetics are back) and crop aug.
4. Eval on the full suite (blog probe, PAN, Enron gen/len/register, held-out Claude+Gemini)
   — target: **keep v14's PAN ~0.88 AND recover v12's gen: ~0.83 / pool TPR@1% ~0.13**, ideally
   higher on both thanks to the better backbone.

**Stretch (from the research, if v14b's content-invariance needs more):** content-controlled
positives (style-held/content-changed paraphrases) and all-layers pooling.

## 3b. v14b RESULT — the synthesis works; the cross-generator gap is closed

v14b = identity diversity (844 authors) **+** synthetic hard-negatives (GPT+Llama),
enabled by the fixed `SyntheticBalancedSampler` (epoch scales to 105 batches → blog authors
covered AND synthetic pairs guaranteed; Enron upweighted via pairs, fixing v14's dilution).
Model: `runs/v14b/lora/checkpoint_best.pt`. Eval is apples-to-apples with v12 (same
held-out Claude+Gemini pool, same probe).

**v14b beats v12 on EVERY axis simultaneously:**

| Axis | v12 | v14 | **v14b** |
|---|---|---|---|
| Held-out **Gemini** AUC (the stuck one) | 0.746 | 0.573 | **0.953** |
| Held-out **Claude** AUC | 0.829 | 0.541 | **0.972** |
| Held-out pool AUC (mahalanobis) | 0.854 | 0.59 | **0.975** |
| Held-out pool TPR@5% | 0.394 | 0.06 | **0.879** |
| PAN cross-topic (content-invariance) | 0.779 | 0.879 | **0.857** |
| Blog held-out (150 authors) | 0.786 | 0.916 | **0.909** |
| register:cross / lenmix / len:short | 0.95/0.92/0.92 | 0.71/0.57/0.65 | **0.969/0.93/0.925** |

The **cross-generator generalization gap — the project's central problem since v11 — is
essentially closed**: Gemini-2.5-Flash, stuck at ~0.75 through v12/v13, is now **0.953** on
the same held-out eval. Content-invariance (PAN 0.857) and identity scale (blog 0.909) from
v14 are kept; the Enron in-domain slices are fully recovered. Two drivers: (a) the sampler
fix gave ~5× more synthetic-pair exposure/epoch (105 vs 11 batches), and (b) the richer
844-author backbone makes the style centroid sharper, so LLM imitations fall further from it
— validating the core thesis that **better authorship modeling is what catches imitations**.

**Wrong-human guardrail — read carefully (the v9@ep10 lesson):** at the synthetic-anchored
threshold, `baseline_linear_z3` shows `fpr_other@5 = 0.30` — but this is a **scorer/threshold
artifact, not a real leak**: `auc_g_other = 0.96` (genuine vs wrong-human separates
excellently in *ranking*), and **mahalanobis holds the guardrail at `fpr_other@5 = 0.083`
(≤0.10) with TPR@5% 0.879, AUC 0.975**. Because v14b separates synthetics so well, anchoring
the threshold purely on the (now-easy) synthetic pool places it poorly for the other-sender
axis under the linear scorer. **Deployment recommendation: mahalanobis scorer** (satisfies
the guardrail; still catches ~88% of forgeries at 5% FPR). The `baseline_linear_z3`
TPR@1%=0.591 operating point is Goodharted (leaks 30% wrong-human) — do not ship it.

**Conclusion: v14b is the performance-grade model.** Content-invariant authorship backbone
(many identities) + synthetic-hard-negative imitation-catching + a trustworthy 150-author
eval + a guardrail-satisfying operating point (mahalanobis). Remaining honest caveats: blog
is partly in-domain (PAN is the clean OOD); the held-out generators (Claude+Gemini) transfer
from GPT+Llama training but a fully novel future vendor should be spot-checked; the operating
point should be set on `min(other,synthetic)`, not the synthetic pool alone.

## 4. Artifacts
- **Best model (performance-grade): `runs/v14b/lora/checkpoint_best.pt`** — config
  `configs/experiments/v14b_manyauthor_syn_lora.yaml`. Deploy with the **mahalanobis** scorer.
- v14 (identity-only): `runs/v14/lora/checkpoint_best.pt` (ep88), `v14_manyauthor_lora.yaml`.
- Data: `data/processed/blog_authors`, `data/processed/enron_blog` (844 authors).
- Eval: `results/v14/{blog_probe,ood,heldoutCG}_v14blora*` and `_v14lora*`; v12 baselines in
  `results/v14/blog_probe_v12lora.json` + `results/v12/`.
- Scripts: `scripts/prepare_blog_authors.py`, `build_blog_probe.py`, `run_v14_eval.sh`.
- Sampler fix that enabled v14b: `src/email_fraud/data/samplers.py` (`SyntheticBalancedSampler`).
