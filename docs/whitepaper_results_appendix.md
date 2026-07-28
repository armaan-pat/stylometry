# Whitepaper results appendix — multi-seed rigor, factorial ablation, novel-vendor

*Prepared 2026-06-29. Consolidates the experiments run to make the v14b story
whitepaper-defensible: multi-seed stability, the completed identity×synthetic 2×2,
a training-seed reproduction, and a novel-vendor generalization spot-check. All numbers
regenerate from committed JSONs via `python scripts/whitepaper/aggregate.py`
(→ `results/whitepaper/AGGREGATE.md`); figures via `python scripts/whitepaper/make_figures.py`
(→ `docs/figures/whitepaper/`). Eval protocol unchanged from prior cycles: held-out
Claude+Gemini pool, 1000× paired bootstrap; here also repeated over **5 probe-draw seeds**
(sender/enroll/query/other sampling) and reported as mean±std.*

---

## A. Main result — multi-seed held-out Claude+Gemini (the headline, now with error bars)

Deployment scorer = **mahalanobis**; mean±std over 5 probe seeds, n=5 each.

| version | AUC | TPR@5% | TPR@1% | FPR_other@5 | AUC g/other |
|---|---|---|---|---|---|
| v11 (single-gen) | 0.690±0.020 | 0.121±0.017 | 0.043±0.012 | 0.000±0.000 | 0.967±0.003 |
| v12 (multi-gen) | 0.871±0.016 | 0.453±0.064 | 0.155±0.036 | 0.001±0.001 | 0.965±0.002 |
| v13 (+DeepSeek) | 0.872±0.021 | 0.499±0.069 | 0.255±0.047 | 0.001±0.001 | 0.970±0.004 |
| v14 (identity, no-syn) | 0.610±0.016 | 0.137±0.015 | 0.060±0.013 | 0.010±0.003 | 0.667±0.015 |
| **v14b (synthesis)** | **0.980±0.003** | **0.909±0.017** | **0.533±0.055** | **0.079±0.014** | **0.967±0.003** |

**Reading:** every version-to-version gap is far larger than its seed-std, so the
progression is real, not probe-draw luck. The committed seed-0 figure (AUC 0.975, TPR@5%
0.879) is a slightly conservative draw; the 5-seed mean is 0.980 / 0.909.
Figure: `docs/figures/whitepaper/wp_fig1_multiseed_progression.png`.

### A.1 The guardrail is robust, not a single-seed artifact
At the synthetic-anchored 5% threshold, the two scorers across all 5 seeds:

| scorer | TPR@5% | FPR_other@5 (wrong-human leak) |
|---|---|---|
| baseline_linear_z3 | 0.943±0.013 | **0.253±0.032** ← breaches 0.10 guardrail |
| **mahalanobis (deploy)** | 0.909±0.017 | **0.079±0.014** ← holds ≤0.10 |

`AUC_g_other` is 0.94–0.97 for *both* scorers — wrong-humans separate fine in ranking; the
linear scorer just sits at a bad threshold. Confirms the deployment recommendation: ship
**mahalanobis**, anchor on `min(other, synthetic)`. (The exact v9@ep10 lesson, now shown
stable over seeds.)

---

## B. Completed identity × synthetic 2×2 (the ablation, with the missing corner)

Held-out Claude+Gemini AUC (mahalanobis), multiseed mean±std. The (44,−syn) corner was
trained this cycle (`runs/wp_ablate_enron44_nosyn`, v14 recipe on Enron-44, step budget
~2600 — between v12 and v14 — to keep the identity comparison free of an undertraining
confound).

|              | no synthetics | + synthetics |
|---|---|---|
| **844 authors** (Enron+Blog) | 0.610±0.016 (v14) | **0.980±0.003 (v14b)** |
| **44 authors** (Enron) | 0.640±0.012 (new) | 0.871±0.016 (v12) |

**This is a super-additive interaction, not two independent levers:**
- **Identity alone does ~nothing for forgery-catching:** 0.640 → 0.610 (44→844, no-syn) — flat.
- **Synthetics are the primary lever:** +0.231 at 44 authors (0.640→0.871).
- **Identity *amplifies* synthetics:** the synthetic gain grows to +0.370 at 844 authors
  (0.610→0.980); equivalently, identity expansion only pays off *with* synthetics present
  (0.871→0.980).

**Precise claim for the paper:** *identity diversity's value for catching LLM imitations is
unlocked by synthetic hard-negatives.* Identity's standalone benefit is **content-invariance**,
which shows up on the orthogonal PAN cross-topic axis (v12 0.779 → v14/v14b 0.857–0.879),
not on the cross-generator axis. The production model needs both because they target two
different failure modes. Figure: `wp_fig2_2x2_grid.png`.

---

## C. Training-seed reproduction (the headline isn't a lucky init)

v14b retrained from scratch with a different seed (`--seed 1`: new weight init + data order;
the flag added to `scripts/train.py`, default 0 = byte-identical to all prior runs). Eval is
the same 5-probe-seed protocol.

| | AUC | TPR@5% | TPR@1% | FPR_other@5 |
|---|---|---|---|---|
| v14b (train seed 0) | 0.980±0.003 | 0.909±0.017 | 0.513±0.057 | 0.079±0.014 |
| v14b (train seed 1) | 0.976±0.005 | 0.867±0.037 | 0.493±0.052 | 0.038±0.017 |

**Reading:** the **ranking (AUC) reproduces within 0.004** — inside the probe-seed noise —
and the guardrail holds for both. Honest nuance worth stating: the **operating-point tail**
(TPR@5%/@1%) carries more training-seed variance (~±0.04) than AUC, so report it as a band,
not a point.

---

## D. Novel-vendor generalization spot-check (the "unseen future vendor" caveat)

Forgeries from **Qwen-2.5-72B + DeepSeek-V3** — vendors outside *both* the training set
(GPT-4o-mini, Llama-3.1-70B) *and* the held-out eval set (Claude, Gemini). 343 forgeries,
44 senders. Mahalanobis, mean over 3 probe seeds.

| version | AUC (forgery vs genuine) | AUC g/other (rank) | TPR@5% |
|---|---|---|---|
| v12 | 0.922±0.005 | 0.964±0.000 | 0.630±0.030 |
| **v14b** | **0.996±0.001** | **0.966±0.003** | 0.986±0.008 |

**Reading:** v14b's separation generalizes to vendors it has *never* touched (AUC 0.996,
genuine-vs-impostor ranking 0.966) — it did **not** overfit to Claude/Gemini. This closes the
"novel future vendor" caveat for the ranking metric.

**Caveat (report this explicitly):** the raw `FPR_other@5` on this pool reads 0.58±**0.21** —
a **threshold-placement artifact, not a real leak**. Because v14b separates these synthetics
near-perfectly (AUC 0.996), the 5%-synthetic-anchored threshold lands in a degenerate flat
region; the large ±0.21 variance is the tell. `AUC_g_other`=0.97 shows wrong-humans still
separate cleanly. This is a *stronger* demonstration of §A.1's rule than Claude/Gemini was:
**the better the model gets on synthetics, the more essential it is to anchor the operating
point on the wrong-human axis, never the synthetic pool alone.** Figure: `wp_fig3_novelvendor.png`.

---

## E. Honest limitations (carry into the paper)
- **Sender ceiling.** The cross-generator probe is 44 Enron senders (post-filter max); its
  TPR@1% CI stays wide (~±0.06 even at the multiseed mean). AUC/TPR@5% are tight; lead with
  those. Enlarging beyond 44 needs a second enrollment corpus.
- **Blog partly in-domain.** The clean OOD content-invariance number is PAN cross-topic
  (0.857); blog (0.909) corroborates but isn't independent.
- **Length invariance unsolved.** `lenmix:short↔long` remains near-chance for every version
  (`results/v13/p3_*`) — report as an open problem, do not paper over.
- **Operating-point tail is train-seed sensitive** (§C): report TPR@5% as a band.

---

## F. Reproduce
```bash
# Eval-only (minutes each):
bash scripts/whitepaper/run_multiseed_eval.sh          # 5 versions × 5 probe seeds
bash scripts/whitepaper/gen_novelvendor.sh             # Qwen+DeepSeek pool (OpenRouter)
# Training (A40): v14b seed-1 (~90m) + (44,-syn) ablation (~2.3h), each auto-evaluated:
bash scripts/whitepaper/run_training.sh
# Consolidate:
python scripts/whitepaper/aggregate.py                 # results/whitepaper/AGGREGATE.md
python scripts/whitepaper/make_figures.py              # docs/figures/whitepaper/*.png
```
Artifacts: `results/whitepaper/{multiseed,novelvendor}/*.json`,
`runs/{v14b_seed1,wp_ablate_enron44_nosyn}/lora/checkpoint_best.pt`,
configs `configs/experiments/{wp_ablate_enron44_nosyn_lora,_wp_novelvendor_eval}.yaml`.
New training runs were `WANDB_MODE=offline`; sync with
`wandb sync runs/<run>/lora/wandb/offline-run-*` if dashboard panels are wanted.
