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

## 4. Artifacts
- Model: `runs/v14/lora/checkpoint_best.pt` (ep88). Config: `configs/experiments/v14_manyauthor_lora.yaml`.
- Data: `data/processed/blog_authors`, `data/processed/enron_blog` (844 authors).
- Eval: `results/v14/blog_probe_v14lora.json`, `ood_v14lora_heldoutCG.json`,
  `heldoutCG_v14lora.json`; baseline `blog_probe_v12lora.json`.
- Scripts: `scripts/prepare_blog_authors.py`, `build_blog_probe.py`, `run_v14_eval.sh`.
