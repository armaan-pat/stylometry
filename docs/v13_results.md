# V13 — roadmap P1/P2/P3 cycle: diminishing returns on vendor count, two clean negatives

*2026-06-18. Executes `docs/findings_and_roadmap.md` P1+P2+P3. Results in
`results/v13/` and `results/v12/ood_domains_v12.json`. Companion: `docs/v12_multigen_results.md`.
Memory: `multigen-generalization-gap`.*

## TL;DR

This cycle ran three roadmap items. **Two are clean negative results that redirect
effort, and the headline retrain shows diminishing returns** — useful, not a win:

1. **P1 (v13 retrain, +DeepSeek 3rd vendor):** adding a 3rd training generator
   produced **no statistically significant change** on the held-out frontier
   generators. Gemini-2.5-Flash is still the hardest (AUC 0.746→**0.755**); every
   v13 vs v12 pool delta sits well inside v13's (wide) bootstrap CI. The guardrail
   held (FPR_other ≈ 0) and there's no len/register regression — but the gap did
   **not** close further.
   **Critical caveat:** v13 did *not* actually add volume (see §1) — it split ~equal
   total data across 3 vendors, so per-vendor depth *dropped* 522→340 rows. So this
   isolates "more vendor diversity at equal total," which has plateaued; the
   "more volume" lever remains **untested** and is now the clear next step.
2. **P2 (v12 domain re-test):** multi-generator training left off-domain authorship
   **unchanged** (PAN 0.779, blog 0.839 — same as v11). Authorship robustness is a
   **separate lever** from generator robustness, as hypothesized.
3. **P3 (length-matched enrollment):** **refuted with evidence.** Matching a query
   to its sender's same-length enrollment subset *hurts* short-query verification
   (TPR@1% 0.508→0.425, bootstrap=1000). The centroid probe already handles short
   queries (~0.92); the `lenmix` weakness lives in *pairwise* verification, which has
   no enrollment to match. The real lever is length-*conditioned thresholds* or
   cross-length training negatives, not enrollment matching.

---

## 1. P1 — v13 expanded-adversary retrain

**Design.** v12 recipe, byte-identical except the training adversary
(`configs/experiments/v13_multigen_lora.yaml`): GPT-4o-mini + Llama-3.1-70B +
**DeepSeek-Chat** (new 3rd vendor), `--n-per-sender 24`, hard-negatives only.
Claude-3.5-Haiku + Gemini-2.5-Flash stay **held out** (reused
`enron_synthetic_v12_heldout`, so the held-out eval pairs are bit-identical to v12).
Anti-Goodhart `pauc/min_other_synthetic_5pct` monitor; `checkpoint_best` = epoch 129
(monitor 0.688). Model: `runs/v13/lora/checkpoint_best.pt`.

**Volume caveat (read before trusting the comparison).** The Enron train split has
only **44 source senders**. Round-robin over 3 vendors at the same total job budget
gives:

| set | rows | senders | vendors | rows/vendor |
|---|---|---|---|---|
| v12_train | 1045 | 44 | 2 (GPT, Llama) | ~522 |
| v13_train | 1022 | 44 | 3 (GPT, Llama, **DeepSeek**) | ~340 |

So v13 has **~equal total data, fewer rows per vendor**. This is a clean test of
*adding a vendor*, but it does **not** exercise the "more volume per generator" lever
the roadmap also called for — that needs more source senders or a higher
`--n-per-sender`. The data bottleneck is enrollment source senders (44), not vendors.

**Result — held-out (Claude+Gemini) pool, `baseline_linear_z3`, bootstrap=1000**
(`results/v13/heldoutCG_v13lora.json`):

| metric | v12 | v13 | v13 95% CI |
|---|---|---|---|
| pool AUC | 0.866 | 0.859 | [0.827, 0.889] |
| pool pAUC@5% | 0.233 | 0.251 | [0.152, 0.399] |
| pool TPR@1% | 0.129 | 0.155 | [0.038, 0.273] |
| pool TPR@5% | 0.394 | 0.496 | [0.295, 0.610] |
| Claude-3.5-Haiku AUC | 0.829 | 0.830 | — |
| **Gemini-2.5-Flash AUC** | **0.746** | **0.755** | — |
| FPR_other @ 5% (guardrail ≤0.10) | ~0 | **0.007** | — |
| len:short / lenmix / register:cross | 0.93 / 0.93 / 0.96 | 0.931 / 0.931 / 0.973 | — |

**Read.** Every v12 point estimate falls comfortably inside v13's CI → **no
significant change**. The held-out probe is small (41–44 senders, 304 synthetics), so
CIs are wide (±0.12 on TPR@1%) — but the direction is flat-to-marginal, not a step
change like v11→v12. The dramatic "diversity transfers" gain (2 vendors) **plateaus
at the 3rd vendor**. Guardrail intact (FPR_other ≈ 0), no authorship regression.

*Aside:* within v13 on the held-out synthetic split, `mahalanobis` beats
`baseline_linear_z3` on TPR@1% (0.280 vs 0.155, ΔCI [+0.045,+0.243], P(win)=0.99).
This contradicts the prior "no scorer beats baseline with CI excluding 0" finding *on
this held-out probe* — worth a look, but it's a scorer choice, orthogonal to the
generator-robustness question.

**Conclusion.** Vendor *count* has diminishing returns; the next P1 experiment should
test **data volume/diversity of source senders**, not a 4th vendor. Options: pull more
Enron senders into the train split (the 100-sender cap is moot — only 44 exist
post-filter), or augment with a second corpus.

## 2. P2 — domain axis on v12 (`results/v12/ood_domains_v12.json`)

| slice | v11-lora | v12-lora |
|---|---|---|
| domain:pan20_xtopic AUC | 0.79 | **0.779** (pAUC@5% 0.191, TPR@1% 0.129) |
| domain:blog AUC | 0.84 | **0.839** (pAUC@5% 0.426, TPR@1% 0.343) |

Multi-generator training **neither helped nor hurt** off-domain authorship. Cross-topic
PAN still collapses at the operating point. Authorship robustness is a **separate
lever** from generator robustness and earns its own roadmap line (the v12 embedding
reshaping was generator-specific). LUAR-MUD's Reddit pretraining makes PAN/blog the
honest OOD tests.

## 3. P3 — length-matched enrollment: refuted

Implemented opt-in in `src/email_fraud/scoring/adaptive.py` (`len_bucket`,
`MIN_BUCKET_K=3`, per-bucket centroids, `centroid_for`) + flags
(`--length-matched-enrollment`, `--query-bucket`) in `scripts/ablate_adaptive_scorers.py`.
Default path is byte-identical (verified by the regression-check run + a 7-assertion
smoke test).

**Short-query genuine-vs-other, v12 checkpoint, bootstrap=1000**
(`results/v13/p3_lenmatch_*`):

| metric | full enrollment | length-matched | Δ |
|---|---|---|---|
| AUC | 0.916 | 0.912 | −0.004 |
| pAUC@5% | 0.599 | 0.518 | −0.081 |
| TPR@1% | 0.508 | 0.425 | −0.083 |
| TPR@5% | 0.667 | 0.717 | +0.051 |

Robust across K=8 and K=16 (hurts more at K=16). **Why:** restricting the centroid to
one length bucket discards length-*invariant* style signal — the full mixed-length
centroid is a better author model. And the centroid probe already verifies short
queries at AUC ~0.92, so there's no problem to fix here. The `lenmix:short↔long`
0.50–0.64 weakness is a property of **pairwise** single-vs-single verification (no
enrollment set exists to match against), which this scoring-side change cannot touch.

**Recommendation.** Drop length-matched enrollment. For the genuine pairwise
cross-length weakness, the levers are: (a) **length-conditioned thresholds**
(operating-point calibration per query-length bucket — doesn't change ranking, so it
needs a threshold-band eval, not an AUC eval), or (b) **cross-length hard negatives**
in training (`generate_synthetic_emails.py --cross-length-fraction`, currently unused).
The P3 code stays as the substrate for (a).

## 4. Reproduce / artifacts

| Thing | Path |
|---|---|
| v13 model (ep129) | `runs/v13/lora/checkpoint_best.pt` |
| v13 config | `configs/experiments/v13_multigen_lora.yaml` |
| v13 train set (GPT+Llama+DeepSeek) | `data/synthetic/enron_synthetic_v13_train` (1022 rows) |
| v13 held-out eval | `results/v13/ood_v13lora_heldoutCG.json`, `results/v13/heldoutCG_v13lora.json` |
| P2 domain eval | `results/v12/ood_domains_v12.json` |
| P3 runs | `results/v13/p3_lenmatch_{baseline,on}_shortq.json`, `p3_lenmatch_on_longq.json`, `p3_regression_check.json` |
| Orchestration | `scripts/run_v13_overnight.sh` (logs in `results/v13/logs/`) |

## 5. Next (revised roadmap priorities)

- **P1':** test the *volume/source-sender* lever (the untested half) — expand the
  Enron train split beyond 44 senders or add a second enrollment corpus, holding
  vendors fixed. Adding a 4th vendor is **not** the priority (count has plateaued).
- **P3':** if cross-length is still a target, implement **length-conditioned
  thresholds** (threshold-band eval) and/or `--cross-length-fraction` training
  negatives — *not* enrollment matching.
- **Eval rigor (deferred P4):** the held-out probe is 41–44 senders → ±0.12 TPR@1%
  CIs swamp the v12↔v13 deltas. Enlarging the unseen-sender probe is now the
  precondition for any further P1 claim to be measurable.
