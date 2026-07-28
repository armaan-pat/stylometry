# Whitepaper outline — evidence map

*Working backbone for the paper. Each section lists the claim, the figure/table, and
the on-disk evidence. "NEW" = experiments set up this cycle for the whitepaper (multi-seed
rigor, novel-vendor spot-check, completed 2x2). Regenerate the results tables with
`python scripts/whitepaper/aggregate.py` (writes `results/whitepaper/AGGREGATE.md`).*

---

## Title (working)
**Catching Cross-Generator Email Impersonation with Content-Invariant Authorship Embeddings**

## Abstract
One paragraph: problem (BEC + LLM impersonation), the cross-generator generalization gap as
the central difficulty, the two-lever fix (identity diversity + synthetic hard-negatives),
and the headline (held-out Claude+Gemini AUC 0.975, 88% forgeries caught @5% FPR, with the
wrong-human guardrail held at 8.3%).

## 1. Introduction & threat model
- Sender fingerprinting via per-sender style centroid; flag deviations.
- Two impostor classes: different-human (easy) and LLM-forgery (hard).
- The honesty rule: **always evaluate on held-out generators** (Claude+Gemini never trained on).
- Contribution list (see §8).

## 2. System
- LUAR-MUD authorship encoder + LoRA; episodic SupCon; prototypical head.
- Enrollment → centroid + per-sender spread (k); Mahalanobis / linear-z scoring.
- Evidence: `README.md`, `docs/architecture.md`, `docs/metrics.md`.

## 3. Backbone choice (why authorship, not topic)
- **Claim:** topic encoders (RoBERTa/MPNet) ≈ chance (0.53); LUAR ≈ 0.95.
- Table: V2 bake-off. Evidence: `results/lineage/`, `docs/EXPERIMENT_STATUS.md` §2.

## 4. Scoring & operating point
- **Claim:** Mahalanobis (Ledoit-Wolf, per-sender) wins low-FPR tail at K≥16; linear-z3 safe
  at low K; bootstrap CIs.
- Table: scorer ablation + K-sweep. Evidence: `results/v7/`, `results/lineage/ablate_*`.
- **Deployment guardrail (load-bearing):** anchor threshold on `min(other, synthetic)`;
  mahalanobis holds `FPR_other` ≤ 0.10 while linear leaks 30%. Fig 5.

## 5. The central problem — cross-generator generalization
- **Claim:** a detector trained on one generator (Mistral) is a shortcut — collapses on
  held-out generators (TPR@1% 0.91 → 0.045). This reframes the metric to **TPR@low-FPR on
  HELD-OUT generators**, not full-range AUC on the training generator.
- Evidence: `results/v12/heldoutgen_v11lora.json`; memory `multigen-generalization-gap`.
- Lit context: trained MGT detectors don't generalize (RAID etc.); zero-shot detectors and
  off-the-shelf style embeddings **both failed our validation** (§7). `docs/research_synthesis_v14_strategy.md`.

## 6. Method — two levers, and the version lineage
The paper's spine. Each version is a controlled change.
| Version | Change | Result |
|---|---|---|
| v11 | single-generator (Mistral) | held-out collapse (the problem) |
| v12 | multi-generator train (GPT+Llama), Claude+Gemini held out | first real fix |
| v13 | +DeepSeek (3rd vendor) | **plateau** — generators not the bottleneck |
| v14 | identity expansion 44→844 authors, **no synthetics** | content-invariance SOLVED; imitation collapsed |
| v14b | **844 authors + synthetics** (sampler fix) | synthesis — beats all on every axis |

- **Main result table:** multi-seed held-out Claude+Gemini, mean±std (NEW — `results/whitepaper/AGGREGATE.md`).
- **Figures:** fig1 (cross-gen AUC), fig2 (pool progression), fig4 (confusion v12 vs v14b).
  `docs/figures/`, regen `scripts/make_v14b_figures.py`.

## 7. Ablations
- **7.1 Identity × synthetic 2×2 (NEW — completed).** Four corners:
  (44,+syn)=v12, (844,−syn)=v14, (844,+syn)=v14b, **(44,−syn)=new** (0.640±0.012).
  Result is a **super-additive interaction**, not two independent levers: identity alone
  ≈ does nothing for forgery-catching (0.640→0.610), synthetics are primary (+0.23 @44),
  and identity *amplifies* synthetics (+0.37 @844 → 0.980). Identity's standalone payoff is
  content-invariance (PAN), a separate axis. See `docs/whitepaper_results_appendix.md` §B,
  fig `docs/figures/whitepaper/wp_fig2_2x2_grid.png`.
- **7.2 Synthetic-data design** (syn-v1 vs syn-v2 cross-register positives). `results/v8/`.
- **7.3 Failed drop-in pivots** (must report — strengthens the retrain decision):
  zero-shot detectors (Binoculars, Fast-DetectGPT) ≈ chance on imitation; off-the-shelf
  StyleDistance < finetuned LUAR. `results/v14/zeroshot_*`, `styledistance_ood.json`.

## 8. Robustness & honesty
- **8.1 Multi-seed stability (NEW).** Headline deltas survive probe-draw variance
  (seed-std ≪ version gaps). `results/whitepaper/AGGREGATE.md`.
- **8.2 Novel-vendor spot-check (NEW).** v14b on Qwen+DeepSeek (outside train AND eval) —
  tests the "novel future vendor" caveat. `results/whitepaper/novelvendor/`.
- **8.3 Content-invariance (clean OOD).** PAN cross-topic 0.857 (the independent win);
  blog corroborates but is partly in-domain.
- **8.4 Known limitation:** length-invariance (`lenmix`) remains near-chance.
  `results/v13/p3_*`. Report honestly; do not paper over.

## 9. Deployment / SLA
- TPR@5% as the SLA number; guardrail term; per-sender calibration as next step.
  `docs/stakeholder_memo_v14b.md` §6.

## 10. Conclusion & future work
- Better authorship modeling is what catches imitations (the thesis).
- Next: per-sender/conformal calibration; length invariance; novel-vendor monitoring.

---

## Evidence status checklist
- [x] Backbone bake-off — `results/lineage/`
- [x] Scoring ablation + guardrail — `results/v7/`, `results/lineage/`, figs
- [x] Cross-generator lineage v11→v14b — `results/v12,v13,v14/heldoutCG_*`
- [x] Failed pivots (zero-shot, StyleDistance) — `results/v14/`
- [x] **Multi-seed held-out table (NEW)** — `results/whitepaper/multiseed/` (DONE; mahal AUC v14b 0.980±0.003)
- [x] **Novel-vendor spot-check (NEW)** — `results/whitepaper/novelvendor/` (DONE; v14b AUC 0.996, g/other 0.966)
- [x] **2×2 completion: (44,−syn) cell (NEW)** — `runs/wp_ablate_enron44_nosyn/` (DONE; 0.640±0.012 — super-additive interaction)
- [x] **v14b reproduction seed (NEW)** — `runs/v14b_seed1/` (DONE; AUC 0.976±0.005, reproduces)
- [x] PAN content-invariance, length limitation — `results/v14/`, `results/v13/`
