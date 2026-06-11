# V6 → V9 Lineage Memo — what changed, why, and what it bought

*Drafted 2026-06-09, ahead of the benchmark run. Status: **FILLED 2026-06-11**
from the 2026-06-10 run. Historical numbers below come from the V7/V8 research
logs; fresh numbers from `results/lineage/`. **Full analysis — confusion
matrices, checkpoint-selection failure, paired deltas — in
`docs/v9_lineage_results_analysis.md`.** The run was produced by:*

```bash
screen -S lineage bash -c 'bash scripts/run_lineage_v6_v9.sh 2>&1 | tee runs/_lineage/console.log'
```

*(~6 h on one A40: four sequential trains + probe and two scorer ablations
per arm. The script prints a digest table at the end; re-runs skip arms whose
checkpoint already exists.)*

---

## 1. The four arms

Each arm changes (approximately) one thing relative to the previous one, so
the lineage doubles as an ablation ladder:

| Arm | Config | Recipe | What it isolates |
|---|---|---|---|
| **v6** | `v6_bench.yaml` | LUAR-MUD + LoRA r16 (q/v), SupCon τ=0.07, n_syn=2, 100 ep, original synthetic | the baseline |
| **v7** | `v7_luar_lora_syn_mahal.yaml` | + τ→0.05, n_syn→4, +key LoRA, 150 ep | the **V7.3 training recipe** (same data) |
| **v8** | `v7_synv2.yaml` | + syn-v2 synthetic data (40% cross-register positives) | the **training data** (same recipe) |
| **v9** | `v9_episodic_shortmail.yaml` | + episodic variable-K′ loss, episode_k 4→1, batch 128, short-mail corpus (floors lowered, signatures kept), 30% crop augmentation | the **training objective + data distribution** |

All four arms use the trustworthy checkpoint monitor
(`pauc/genuine_vs_synthetic_5pct`), the expanded CentroidProbe, and the same
per-epoch `eval_score_fns` — so the W&B curves (group `v9-lineage`) are
directly comparable, unlike the original V6/V7 runs.

**Two evaluation corpora per arm.** `ablate_own_*` evaluates each checkpoint
against its *own* training corpus and synthetic set (parity with the
historical numbers). `ablate_common_*` evaluates every checkpoint on the same
production-like corpus — `enron_shortmail` (short emails present, signatures
kept) + syn-v2 impostors. **The common-corpus numbers are the
apples-to-apples lineage comparison**; the own-corpus numbers flatter v6–v8
because their eval set, like their training set, contains no short emails.

---

## 2. v6 → v7: train harder against the adversary you score against

**Changes.** τ 0.07→0.05, n_syn 2→4, LoRA targets +key, 100→150 epochs.

**Mechanism.** All four knobs push the same direction — more gradient signal
on the hardest negatives. Sharper SupCon temperature upweights near-anchor
negatives (which, by construction, are mostly the LLM impostors); doubling
n_syn doubles how often a real/synthetic contrast pair appears per batch;
+key LoRA gives attention one more place to encode stylistic, not topical,
features; longer training matters because the synthetic-separation signal
emerges late (V6 plateaued around epoch 90 on the easy metrics while the
hard-tail metrics were still moving).

**Observed historically** (own-corpus probe, K=8, `linear_z3` — V7.3 log):

| Metric | V6 | V7.3 | Δ |
|---|---|---|---|
| AUC genuine-vs-synthetic | 0.875 | 0.909 | +3.4 pp |
| TPR @ 5% FPR_syn | 0.675 | 0.767 | +9.2 pp |
| TPR @ 1% FPR_syn | 0.517 | 0.633 | +11.6 pp |

The gain concentrates in the low-FPR tail rather than the AUC — consistent
with the mechanism: the recipe doesn't make the average case better, it makes
the *hardest impostors* separable. Known cost: a little `FPR_other` is traded
for `FPR_synthetic` (acceptable for the BEC threat model, flagged in
`docs/EXPERIMENT_STATUS.md` §3).

**Fresh numbers (2026-06-10, common corpus, K=8, `linear_z3`):** confirmed,
and then some — AUC 0.873→0.947, TPR@5% 0.557→0.822, TPR@1% 0.420→0.670;
paired Δtpr1 +0.250 [+0.148, +0.322], P(win)=1.00. Caveats: the v6 arm
**crashed at epoch 20/100** (silently skipped on relaunch), so this delta is
v.s. an undertrained baseline; and v7's checkpoint (best at ep76) also trades
FPR_other (0.172 at the 5%-FPR_syn threshold vs v6's 0.023).

---

## 3. v7 → v8: teach it what is NOT fraud (cross-register positives)

**Changes.** Identical recipe; only the synthetic dataset changes
(`enron_synthetic` → `enron_synthetic_v2`, generated with
cross-register-fraction 0.4).

**Mechanism.** The 2026-06-03 diagnosis showed the dominant false-negative
mode was *the same author switching register* (formal memo ↔ casual note) —
the model confused legitimate style-shift with author-change. syn-v2 adds
LLM-written same-author/opposite-register emails stored under the *real*
sender id, i.e. as positives. The loss is thereby forced to find features
that survive a register flip — closer to authorial identity, further from
surface register. The hard-negative half of the synthetic set is unchanged,
so impostor pressure stays constant.

**Observed historically** (V8 A/B, K=8, `linear_z3`): TPR@1% FPR
0.442 → 0.575 (**+13.3 pp**) with AUC flat — again the gain lives entirely in
the hard tail, which is exactly what "fixing one confusion mode" should look
like (the easy 90% of decisions were never the problem).

*Caveat:* the V8 A/B compared syn-v1 vs syn-v2 (both regenerated); the v7 arm
here trains on the *original* synthetic set, so the v7→v8 delta in this
benchmark bundles "regenerated corpus + quality filter" with "cross-register
positives". Directionally the same story; don't over-read a pp-level split.

**Fresh numbers (2026-06-10):** the historical gain did **not** reproduce —
v7→v8 is a significant *regression* on the common corpus: TPR@1% 0.670→0.466,
paired Δ −0.205 [−0.333, −0.110], P(win)=0.00; AUC 0.947→0.922. Reconciliation
(see analysis memo §5): different checkpoints (monitor picked v7@76 vs v8@37;
both arms' vs_syn decays after epoch ~7–10), different comparison (v7 here
trains on *original* syn-v1), and — mechanistically — cross-register positives
are LLM text attracted toward sender centroids, which erodes the
"reject-LLM-flavored-text" shortcut that a same-generator (Mistral-only)
impostor pool maximally rewards. v8's FPR_other is consistently *better* than
v7's (0.117 vs 0.172 at the deployed threshold), so the transfer claim isn't
dead — it's untestable until a multi-generator eval exists.

---

## 4. v8 → v9: train the objective you deploy, on the traffic you'll see

**Changes** (one shared retrain, items 1+2 of
`docs/robustness_mechanisms.md`):

1. **Episodic, variable-K′ loss** replaces plain SupCon (SupCon stays as a
   0.5-weighted aux term). Per batch, each sender's embeddings split into a
   support set of K′ ~ uniform{2..6} and queries; prototypes = support means;
   queries are classified against all in-batch prototypes with the deployed
   cosine distance. Synthetic `__syn` emails never form prototypes — they are
   repulsion-only queries against their mimicked sender.
2. **episode_k 4→1**: training embeddings are single-email, the same
   representation enrollment/inference use (closes the §A4 train/infer
   episode mismatch).
3. **Short-mail corpus**: word floor lowered to 5 (it was effectively ~50
   chars before — and the configured word floor was never enforced),
   signatures/sign-offs kept. 13.2% of train emails are now <10 words.
4. **Crop augmentation**: 30% of training emails replaced by a random
   contiguous 5–60-word span of themselves, same label.

**Mechanism.** SupCon optimizes pairwise contrast; the deployed system scores
against a *centroid of K embeddings*, an object SupCon never sees. The
episodic loss makes "the mean of a few of my embeddings is a stable, telling
description of me" the literal training objective, with K′ sampled small so
low-K enrollment is the trained regime, not an extrapolation. The data-side
changes attack the 30% false-positive mode at its root: the encoder was bad
at short emails partly because they were out-of-distribution, and crop
augmentation manufactures unlimited short positives that force
length-invariant style features (a 15-word fragment of Alice must embed near
full Alice emails).

**Expected signature** (this is the part to check carefully, not just the
headline):

- Pooled AUC / TPR@FPR on the *own*/legacy-style probe may move only
  modestly — v9's targets are the (low-K, short-query) cells that pooled
  metrics average away.
- The **common-corpus** ablation (short emails included) is where v9 should
  separate from v8, and where v6–v8 should *drop* relative to their own-corpus
  numbers (they've never seen a <50-char email).
- The K-sweep (`K=4,8` columns of the ablation) is the low-K check: v9's
  K=4/8 numbers should close the gap to its K=16/25 numbers more than v8's do.
- If v9's pooled numbers are flat AND its low-K/short-mail cells are flat,
  the episodic loss isn't earning its complexity — fall back to v8's encoder
  and take the scoring-side mechanisms (§A2/A3/B2 of the robustness memo)
  instead.

**Fresh numbers (2026-06-10), against the expected signature:**

- Pooled numbers *did* move: AUC 0.922→0.953, TPR@5% 0.708→0.837 (paired Δ
  +0.129 [+0.049, +0.197], significant), EER 0.147→0.099. TPR@1% +0.106 is
  not significant ([−0.133, +0.269]).
- The low-K check **failed at the benchmarked checkpoint**: v9 K=4 TPR@1%
  0.458 < v7's 0.553 (though v9+cosine reaches 0.809 at K=25, the best cell
  in the run).
- The short-mail cliff is gone on the genuine side (<10w accept 0.78 vs v8's
  0.50 at the 5% threshold) — but the synthetic pool contains **zero emails
  under 26 words**, so the fraud side of the short-mail trade is unmeasured.
- **The decisive caveat:** the monitor picked **epoch 10 of 150**. That
  checkpoint accepts 59.5% of wrong-sender real-human emails at the deployed
  threshold (v6: 2.3%) — it's an LLM-detector, not an authorship verifier.
  The epoch-150 weights are the mirror image: best human-impostor
  discrimination of any arm (AUC[g/other] 0.933, FPR_other 0.002) with a
  collapsed synthetic tail (TPR@1% 0.223). The episodic loss is earning its
  complexity — on the *unmonitored* objective. Don't fall back to v8;
  fix checkpoint selection and/or decouple the two tasks (analysis memo §7).

---

## 5. Results (fill in after the run)

Common corpus (`enron_shortmail` + syn-v2), `baseline_linear_z3`, K=8 enroll
— from the digest the script prints, or
`results/lineage/ablate_common_<arm>.json`:

| Arm | AUC[g/syn] | TPR@5%FPR | TPR@1%FPR [95% CI] | best scorer by tpr1 (P(win)) |
|---|---|---|---|---|
| v6 | 0.873 | 0.557 | 0.420 [0.341, 0.530] | mahalanobis 0.432 (0.57) — keep baseline |
| v7 | 0.947 | 0.822 | **0.670 [0.598, 0.739]** | baseline_linear_z3 (ranks #1) |
| v8 | 0.922 | 0.708 | 0.466 [0.311, 0.572] | mahalanobis 0.492 (0.79) — keep baseline |
| v9 | **0.953** | **0.837** | 0.572 [0.318, 0.686] | baseline_cosine 0.606 (0.75) — keep baseline |

(Best-checkpoint epochs: v6@14 — *run crashed at ep20/100*, v7@76, v8@37,
v9@**10**/150. At the 5%-FPR_syn threshold, FPR_other is 0.023 / 0.172 /
0.117 / **0.595** respectively — see the analysis memo's confusion matrices
before reading this table as a ranking.)

K-sweep at the same operating point (tpr1 by enrollment K, from the
`k_sweep` block of each JSON):

| Arm | K=4 | K=8 | K=16 | K=25 |
|---|---|---|---|---|
| v6 | 0.265 | 0.420 | 0.569 | 0.593 |
| v7 | **0.553** | **0.670** | 0.801 | 0.793 |
| v8 | 0.341 | 0.466 | 0.667 | 0.679 |
| v9 | 0.458 | 0.572 | 0.703 | 0.724 |
| v9 (cosine) | 0.511 | 0.606 | **0.764** | **0.809** |

Own-corpus (parity with historical numbers, K=8, `linear_z3`): v6 AUC 0.861 /
tpr1 0.454, v7 0.906 / 0.647, v8 0.904 / 0.531, v9 same as common (its own
corpus *is* the common corpus). Note v6–v8 did **not** drop own→common; v7
even rises (0.647→0.670).

## 6. How to read the deltas honestly

- **The noise floor is real.** The 2026-06-09 CI run put ±~0.13 on TPR@1%
  with the old probe; the expanded probe roughly halves that, but sub-2 pp
  differences are still noise. Trust the bootstrap Δ-CIs in the JSON
  (`tpr1_dlo/dhi`, `*_pwin`), not point-estimate orderings.
- **Same eval set or no comparison.** Only `ablate_common_*` rows are
  mutually comparable; own-corpus rows each measure a different test.
- **Checkpoints**: all arms monitor `pauc/genuine_vs_synthetic_5pct`, so
  `checkpoint_best.pt` is trustworthy (the script prefers it, falls back to
  `checkpoint_last.pt`). This differs from the historical V7.3/V8 numbers,
  which used the epoch-150/last checkpoint.
- **Watch `FPR_other`** alongside the synthetic tail — the V7 recipe already
  traded a little of it; v9 should not compound that silently.
- The single biggest known threat to all four arms is unchanged: one
  generator (Mistral-7B) produced every synthetic impostor. A future
  multi-LLM adversary suite (item 2 in `docs/EXPERIMENT_STATUS.md` §5) may
  reorder these arms.
