# V6→V9 Lineage Benchmark — Results Analysis

*2026-06-11. Analysis of the 2026-06-10 lineage run (`scripts/run_lineage_v6_v9.sh`,
W&B group `v9-lineage`). Companion to the design memo `docs/v9_lineage_memo.md`;
raw numbers in `results/lineage/`, figures in `results/lineage/figures/`,
post-hoc analysis code in `scripts/lineage_confusion_report.py` (regenerates
confusion matrices, paired cross-arm bootstraps, and all figures from the
checkpoints).*

---

## 0. TL;DR

1. **v7 wins the benchmark as scored; v9 wins the headline metrics.** On the
   common corpus at K=8 with `linear_z3`, v7 has the best hard-tail number
   (TPR@1%FPR 0.670 vs v9's 0.572) while v9 has the best AUC (0.953), best
   TPR@5% (0.837) and best EER (0.099). The paired bootstrap says v6→v7 was a
   real +25 pp jump, v7→v8 a real −20 pp regression, and v8→v9 a real +13 pp
   recovery at TPR@5% (TPR@1% deltas involving v9 are not significant).
2. **But the v9 number is an artifact of checkpoint selection, and the
   benchmark's headline split hides a catastrophic failure.** The
   `pauc/genuine_vs_synthetic_5pct` monitor picked v9's **epoch 10** of 150.
   That checkpoint is a superb *LLM-text detector* and a poor *authorship
   verifier*: at its 5%-FPR_syn operating point it accepts **59.5% of
   wrong-sender real-human emails** (v6: 2.3%). The confusion matrices in §3
   make this visible; the genuine-vs-synthetic-only ranking metric cannot.
3. **v9's recipe is doing exactly what it was designed to do — at the other
   end of training.** By epoch 150, v9 has the best human-impostor
   discrimination of any model we've ever evaluated (AUC genuine-vs-other
   0.933, FPR_other ≈ 0 at deploy thresholds) and a flat accept-rate across
   query length (the short-mail cliff is gone). Its synthetic tail collapses
   instead (TPR@1% 0.223). **No single v9 checkpoint has both.** The two
   objectives — "reject LLM mimics" and "identify the human author" — are
   pulling the single embedding space in different directions, and our
   single-metric monitor slides the choice to one extreme.
4. **The evaluation itself has two holes** that cap what this benchmark can
   prove: the synthetic impostor pool contains **zero emails under 26 words**
   (so short-query forgery is unmeasured while short-query genuine traffic is
   rewarded), and one generator (Mistral-7B) produced every impostor (so
   "reject LLM-ness" is a winning shortcut).
5. Housekeeping: **v6's training crashed at epoch 20/100** (during the first
   hard-negative mining pass) and was silently skipped on relaunch, so the v6
   baseline is a 19-epoch model. The probe also capped at **44 senders (need
   60)**, which is why the TPR@1% CIs are ±0.1–0.25 wide.

**Recommended next steps** (detail in §7): fix the monitor and report the
(FPR_syn, FPR_other) pair everywhere; decouple the two tasks into an
authorship score + a dedicated LLM-detector head and fuse; add short and
multi-generator synthetic impostors; then re-run this lineage.

---

## 1. What was run

Four arms, each changing ~one thing, all evaluated on their own corpus
(`ablate_own_*`) and on a common production-like corpus (`ablate_common_*`:
`enron_shortmail` + syn-v2 impostors, 44 profiled senders × 8 enrolled
emails, 264 genuine / 600 other-sender / 600 synthetic queries, seed 0).
Checkpoints: `checkpoint_best.pt` under the `pauc/genuine_vs_synthetic_5pct`
monitor.

| Arm | Recipe delta | Best ckpt epoch | Notes |
|---|---|---|---|
| v6 | baseline (τ=0.07, n_syn=2, q/v LoRA, 100 ep, syn-v1) | 14 | **run crashed at ep 20/100** — undertrained baseline |
| v7 | V7.3 recipe (τ=0.05, n_syn=4, +key LoRA, 150 ep) | 76 | |
| v8 | + syn-v2 data (40% cross-register positives) | 37 | |
| v9 | + episodic variable-K′ loss, episode_k=1, short-mail corpus, 30% crop aug | **10** | best ckpt is 7% into training |

All "common corpus" numbers below are mutually comparable — same probe
texts, same enrollment, only the encoder differs. v9's own-corpus numbers are
identical to its common-corpus numbers because the common corpus *is* its
training distribution.

---

## 2. Headline results (common corpus, `baseline_linear_z3`, K=8)

| Arm | AUC[g/syn] | pAUC@5% | TPR@5%FPR | TPR@1%FPR [95% CI] | EER | **FPR_other @5%FPR_syn thr** |
|---|---|---|---|---|---|---|
| v6 | 0.873 | 0.479 | 0.557 | 0.420 [0.341, 0.530] | 0.209 | **0.023** |
| v7 | **0.947** | **0.725** | 0.822 | **0.670 [0.598, 0.739]** | 0.118 | 0.172 |
| v8 | 0.922 | 0.555 | 0.708 | 0.466 [0.311, 0.572] | 0.147 | 0.117 |
| v9 | **0.953** | 0.658 | **0.837** | 0.572 [0.318, 0.686] | **0.099** | **0.595** |

The last column is not part of the standard ablation output — it comes from
the post-hoc confusion analysis (§3) and is the most important number in the
table: the fraction of *real emails from the wrong sender* accepted when the
threshold is set to let through 5% of synthetic impostors.

**Paired cross-arm deltas.** Because every arm scored the identical probe
texts, we can bootstrap arm-vs-arm deltas with shared resamples (2,000
replicates; `results/lineage/confusion_report.json → paired_deltas`). This is
much tighter than comparing the per-arm CIs above:

| Transition | ΔTPR@1% [95% CI] | ΔTPR@5% [95% CI] | ΔAUC [95% CI] | Verdict |
|---|---|---|---|---|
| v6→v7 | **+0.250 [+0.148, +0.322]** | **+0.265 [+0.193, +0.330]** | **+0.074 [+0.058, +0.091]** | real, large gain |
| v7→v8 | **−0.205 [−0.333, −0.110]** | **−0.114 [−0.174, −0.049]** | **−0.025 [−0.037, −0.013]** | real regression |
| v8→v9 | +0.106 [−0.133, +0.269] | **+0.129 [+0.049, +0.197]** | **+0.031 [+0.014, +0.048]** | real gain except at the 1% tail |
| v7→v9 | −0.098 [−0.345, +0.015] | +0.015 [−0.049, +0.083] | +0.006 [−0.010, +0.022] | v9 ≈ v7 on AUC/TPR@5; v7 likely better at the 1% tail |

Scorer ablation: in every arm the recommendation engine kept
`baseline_linear_z3` — no adaptive scorer beat it with a CI excluding zero.
Notably, on v9 plain `cosine` leads the point estimates (TPR@1% 0.606,
P(win)=0.75; K=25 TPR@1% 0.809, the best single cell in the whole run) —
consistent with the episodic loss being trained through cosine distances.

**Own-corpus parity check** (`ablate_own_*`): v6 0.454 / v7 0.647 / v8 0.531
TPR@1%. The expected "v6–v8 drop on the common corpus because of short
emails" did **not** materialize — v7 actually scores *higher* on the common
corpus (0.670 vs 0.647). Short genuine queries turn out not to be the
problem for v7 at these operating points (see §4), and the syn-v2 impostors
are evidently not harder for it than syn-v1.

---

## 3. Confusion matrices

Setup: each query is scored against its claimed sender's centroid (K=8
enrolled emails); accept if score ≥ threshold. The threshold is set per arm
on the synthetic pool (the deployment rule: "hold the synthetic-impostor
accept rate at X%"). Three query pools: **genuine** (264; claimed sender =
true author — should be accepted), **other-sender** (600; real human email,
wrong claimed sender — should be rejected), **synthetic** (600; LLM
impersonating the claimed sender — should be rejected). Confusion "positive"
= accept-as-genuine, so for the two impostor rows every accept is a missed
fraud.

Figure: `results/lineage/figures/confusion_grid.png` (rates + counts),
`score_distributions.png` (the underlying score histograms with thresholds),
`roc_log.png` (log-FPR ROC).

### Operating point A — strict: threshold at 1% FPR_synthetic

| | v6 acc/rej | v7 acc/rej | v8 acc/rej | v9 acc/rej |
|---|---|---|---|---|
| genuine (264) | 107 / 157 (41%) | **173 / 91 (66%)** | 111 / 153 (42%) | 147 / 117 (56%) |
| other-sender (600) | 4 / 596 (0.7%) | 32 / 568 (5.3%) | 4 / 596 (0.7%) | **152 / 448 (25.3%)** |
| synthetic (600) | 6 / 594 (1%) | 6 / 594 (1%) | 6 / 594 (1%) | 6 / 594 (1%) |

### Operating point B — deployed: threshold at 5% FPR_synthetic

| | v6 acc/rej | v7 acc/rej | v8 acc/rej | v9 acc/rej |
|---|---|---|---|---|
| genuine (264) | 145 / 119 (55%) | 217 / 47 (82%) | 185 / 79 (70%) | **221 / 43 (84%)** |
| other-sender (600) | 14 / 586 (2.3%) | 103 / 497 (17.2%) | 70 / 530 (11.7%) | **357 / 243 (59.5%)** |
| synthetic (600) | 30 / 570 (5%) | 30 / 570 (5%) | 30 / 570 (5%) | 30 / 570 (5%) |

### How to read these

- **The synthetic row is identical by construction** (the threshold is
  defined on it). All the information is in how much genuine traffic survives
  (row 1) and what the same threshold costs on the *unmonitored* impostor
  type (row 2).
- **v6 and v8 are conservative everywhere**: thresholds sit high
  (0.55–0.63 in score space), so they reject ~half the genuine pool but
  almost never accept any impostor of either kind. v6's profile is what an
  undertrained encoder looks like; v8's is similar but better.
- **v7 is the best-balanced model**: 82% genuine acceptance at 5% synthetic
  leakage and 17% other-sender leakage. Its score distributions
  (`score_distributions.png`) show three cleanly ordered populations —
  genuine highest, other in the middle, synthetic lowest.
- **v9 (epoch-10 checkpoint) has effectively collapsed the genuine and
  other-sender distributions onto each other.** It buys its excellent
  genuine-vs-synthetic numbers (AUC 0.953, EER 0.099) by carving out a
  direction that separates *LLM text from human text* — not *this human from
  other humans*. At the deployed threshold, 6 out of 10 wrong-sender real
  emails sail through. For the BEC threat model, where an attacker can also
  paste/forward genuine human text or write manually, this checkpoint is not
  shippable.
- The 1%-FPR point shows the same ordering, compressed: v9 still leaks 25%
  of other-sender mail where v6/v8 leak 0.7% and v7 5.3%.

### Why v9's matrix looks like this: the checkpoint-selection failure

`results/lineage/figures/training_dynamics.png` plots the in-training
centroid AUC of every arm, split into *vs-synthetic* and *vs-other-sender*,
with the chosen `checkpoint_best` marked:

| Arm | best ep | AUC vs_other / vs_syn at best | at last epoch | peak vs_syn |
|---|---|---|---|---|
| v6 | 14/19 | 0.886 / 0.861 | 0.891 / 0.825 | 0.894 @ ep6 |
| v7 | 76/150 | 0.880 / 0.906 | 0.885 / 0.883 | 0.936 @ ep9 |
| v8 | 37/150 | 0.880 / 0.875 | 0.909 / 0.848 | 0.907 @ ep7 |
| v9 | **10**/150 | **0.703 / 0.944** | **0.937 / 0.899** | 0.944 @ ep10 |

Two facts jump out:

1. **Every arm learns synthetic separation almost immediately** (vs_syn peaks
   at epoch 6–10 in all four) **and then partially unlearns it** as training
   reallocates the representation toward sender-vs-sender contrast. The
   genuine-vs-synthetic monitor therefore systematically prefers early
   checkpoints, and the lineage partially ranks *where the monitor happened
   to fire*, not the recipes.
2. **v9's two curves cross.** Its episodic loss starts with weak
   human-impostor discrimination (vs_other 0.70 at ep10) and grinds upward for
   140 epochs to the best vs_other of any arm (0.937), while vs_syn drifts
   down only ~4.5 pp. The monitor froze it at the worst point on that
   frontier.

To confirm this isn't just an in-training-metric story, we re-evaluated the
**epoch-150 (`checkpoint_last`) weights on the identical common probe**
(`results/lineage/last_checkpoint_eval.json`):

| Model | AUC[g/syn] | TPR@1% | TPR@5% | FPR_other @5%thr | AUC[g/other] |
|---|---|---|---|---|---|
| v7 ep76 (benchmarked) | 0.947 | 0.670 | 0.822 | 0.172 | — |
| v7 ep150 | 0.919 | 0.602 | 0.742 | 0.060 | 0.909 |
| v8 ep150 | 0.874 | 0.390 | 0.561 | 0.013 | 0.897 |
| v9 ep10 (benchmarked) | 0.953 | 0.572 | 0.837 | **0.595** | ~0.70 (in-training) |
| v9 ep150 | 0.888 | **0.223** | 0.473 | **0.002** | **0.933** |

v9-ep150 is the mirror image of v9-ep10: near-perfect rejection of human
impostors, weakest synthetic tail of the lineage. **The v9 recipe produced
the best authorship model and the best LLM-detector we have — in the same
run, 140 epochs apart — and no checkpoint that is both.** That tension, not
any single headline number, is the real finding of this benchmark.

---

## 4. Length stratification — did the short-mail mechanisms work?

Genuine accept rate at the 5%-FPR_syn threshold, by query length
(figure: `length_strata.png`):

| Arm | <10w (n=36) | 10–25w (n=93) | 26–60w (n=65) | >60w (n=70) | cliff (<10w vs >60w) |
|---|---|---|---|---|---|
| v6 | 0.33 | 0.52 | 0.58 | 0.67 | −34 pp |
| v7 | 0.78 | 0.80 | 0.92 | 0.79 | −1 pp |
| v8 | 0.50 | 0.69 | 0.82 | 0.71 | −21 pp |
| v9 | **0.78** | **0.86** | 0.85 | 0.83 | **−5 pp** |

- **The short-mail cliff is real in v6/v8 and gone in v9** — the crop
  augmentation + short-mail corpus did what they were designed to do on the
  genuine side, with the flattest profile of any arm.
- **The surprise is v7**: it never saw a sub-50-word email in training, yet
  its profile is also flat. Combined with v9's other-sender leakage being
  flat across length too (0.66/0.67/0.63/0.45 — even *long* wrong-sender
  emails get in), this suggests v9-ep10's short-mail "wins" come
  substantially from generic leniency, not length-invariant style features.
  The v9-ep150 checkpoint is where the genuine length-invariance claim
  should be re-tested.
- **The eval cannot currently falsify short-query fraud detection**: the
  synthetic pool has *zero* emails under 26 words (54 at 26–60w, 546 over
  60w). A model could accept every short email sight-unseen and lose nothing
  on FPR_syn. Until the generator produces short impostors (or we crop the
  synthetic pool the way we crop training positives), the short-mail cells
  of this benchmark only measure the genuine side of the trade.

---

## 5. The v7→v8 regression, and what it says about syn-v2

The historical V8 A/B reported +13.3 pp TPR@1% for syn-v2; this benchmark
shows a paired **−20.5 pp** for the same data change. Reconciliation:

1. **Different checkpoints.** The historical numbers used epoch-150/last
   checkpoints; this run used the monitor's picks (v7@76, v8@37). The
   trajectories show vs_syn decaying after epoch ~7–10 in both arms, at
   different rates — the monitor sampled the two arms at non-comparable
   points on their decay curves. (At ep150 the gap persists though: v7-last
   0.602 vs v8-last 0.390 TPR@1%.)
2. **Different comparison.** The historical A/B regenerated *both* corpora
   (syn-v1′ vs syn-v2) and evaluated each arm against its own synthetic set;
   here v7 trains on the *original* syn-v1 and both arms are scored against
   syn-v2 impostors.
3. **A mechanism that makes the regression expected, not anomalous:**
   cross-register positives are *LLM-written text labeled as the genuine
   sender*. Training attracts Mistral-flavored text toward sender centroids
   — by design, to suppress the register shortcut, but the same gradient
   also erodes the "reject anything LLM-flavored" feature that this
   benchmark's impostor pool (same generator!) maximally rewards. v7, which
   only ever saw LLM text as repulsion targets, keeps the shortcut intact
   and tops the leaderboard.

So the honest reading is **not** "syn-v2 hurts." It is: *on a
single-generator eval, removing a generator-artifact shortcut looks like a
regression.* Evidence that v8/v9 traded that shortcut for something more
transferable: their FPR_other at deploy thresholds is consistently below
v7's (0.117/0.013/0.002 for v8-best/v8-last/v9-last vs 0.172/0.060 for v7),
and the linear-probe separability of synthetic text keeps *rising* with
arm (v6 0.965 → v7 0.969 → v8 0.990 → v9 1.000 frozen-probe accuracy) —
the embedding space still contains the LLM-ness signal, the *centroid
geometry* just stops using it. Which is exactly why a dedicated probe head
should own that job (§7.2). A multi-generator eval is the only way to settle
this.

---

## 6. Per-arm scorecard

**v6 — undertrained baseline (19/100 epochs; crashed at the first
hard-negative mining pass, silently skipped on relaunch).** Conservative
everywhere; worst genuine retention (55% at the deployed point); huge
short-mail cliff. Its role as "the before picture" is compromised — rerun
before quoting v6 deltas externally. The crash itself needs a postmortem
(v7/v9 survived 27 mining events; v6's log ends mid-mining at ep20).

**v7 — the best single checkpoint in the benchmark, with an asterisk.**
Best hard tail (TPR@1% 0.670), best-balanced confusion matrix, flat length
profile. Weaknesses: 17% other-sender leakage at the deployed threshold —
2.5× v8's; and an unknown fraction of its synthetic-tail advantage is the
single-generator shortcut (§5). It is the right ship candidate *for the
benchmark as defined*; it is not obviously the most robust model produced.

**v8 — quietly reasonable, badly served by its checkpoint.** Significantly
worse than v7 on the synthetic tail, significantly better than v6, low
FPR_other (0.117 best / 0.013 last). The cross-register mechanism appears to
work (lowest probe-confusion on register flips per the V8 logs) but its
benefit is invisible — arguably *inverted* — on a same-generator eval.

**v9 — two excellent models wearing one checkpoint name.**
- *Strengths:* best AUC/EER/TPR@5% as benchmarked; short-mail cliff
  eliminated; ep150 weights are the best human-impostor verifier we have
  (AUC[g/other] 0.933, FPR_other 0.2%); `cosine` at K=25 gives the single
  best tail cell (0.809), and its K-sweep (0.51→0.81 from K=4→25) shows the
  episodic objective converting enrollment data into accuracy more steeply
  than any other arm.
- *Weaknesses:* the monitored checkpoint (ep10) is unshippable
  (FPR_other 59.5%); ep150's synthetic tail (TPR@1% 0.223) is the worst in
  the lineage; the expected low-K advantage did not materialize at the
  benchmarked checkpoint (K=4 TPR@1% 0.458 vs v7's 0.553); and its perfect
  frozen-probe separability (1.000) is a red flag for single-generator
  overfitting somewhere in the space.

---

## 7. Where to go next

Ordered by leverage-per-effort; 1–3 are prerequisites for trusting any
future lineage comparison.

### 7.1 Fix the measurement first (cheap, blocks everything else)

- **Monitor a composite, not one tail.** Checkpoint selection on
  `pauc/genuine_vs_synthetic_5pct` provably Goodharts (§3). Switch to the
  genuine-vs-**all** split (the ablation already supports `--split all`) or
  an explicit min/geometric mean of pauc_syn and pauc_other, and log both
  components every epoch. Then re-emit the lineage digest at *matched*
  checkpoints (best-composite and last) — the v7-vs-v9 ordering may flip.
- **Print FPR_other next to every TPR@FPR_syn number** (the confusion-report
  script now does this; fold it into `ablate_adaptive_scorers.py` so it's
  not post-hoc).
- **Probe power**: only 44/60 senders were eligible (K=8+6 emails needed).
  Lower `n_query` to 4 or pull senders from train+val to restore 60+, and
  run 3–5 probe seeds — the v9 TPR@1% CI spans 37 pp, which is wider than
  every effect we're chasing.
- **Re-run v6** (and add a loud failure banner to the lineage script — the
  crash was only discoverable by counting epochs in the log).

### 7.2 Decouple the two tasks (the main modeling idea this run motivates)

The benchmark demonstrates that one cosine geometry struggles to serve two
adversaries. But both component solutions already exist in this run:

- *Authorship*: v9-ep150 (AUC[g/other] 0.933, FPR_other 0.002).
- *LLM detection*: a frozen linear probe on the same embeddings hits
  0.99–1.00 accuracy on syn-v2 (`probe_v9.json`) — i.e. the encoder
  *retains* the LLM-ness direction even when centroid scoring stops using it.

So: score every query twice — per-sender style distance (episodic encoder,
v9-last) **and** a global synthetic-text logit (linear/shallow probe, ~free
to train) — and fuse (logistic stacking on a held-out split, or two
independent thresholds where either alarm rejects). This converts the
checkpoint-selection dilemma into an architecture where each head trains to
convergence on its own objective. It also gives the fraud system separable
dials: "how paranoid about LLMs" vs "how strict about identity." Validate the
probe head against non-Mistral generators before trusting it (see 7.3).

Cheaper variants worth one afternoon each: (a) **weight-space interpolation /
model soup** between v9-ep10 and v9-ep150 (LoRA deltas interpolate well;
sweep α and trace the (TPR_syn, FPR_other) Pareto curve); (b) checkpoint
EMA during training; (c) two-phase schedule — train episodic to convergence,
then a short synthetic-pressure finetune with a small LR.

### 7.3 Fix the adversary distribution (data)

- **Short synthetic impostors.** Crop the synthetic pool exactly the way
  training positives are cropped (5–60-word spans) or generate short BEC-style
  one-liners ("Are you at your desk? Need a wire approved today.") — the
  current eval has zero impostors under 26 words, so the short-mail FPR is
  literally unmeasurable, and short BEC lures are the canonical real attack.
- **Multi-generator suite** (already roadmap item 2 in
  `docs/EXPERIMENT_STATUS.md` §5): add Claude/GPT/Gemini/Llama impostors,
  *evaluate* against held-out generators. This is the only way to
  distinguish "v7 is better" from "v7 memorized Mistral artifacts," and the
  only honest test of syn-v2's transfer claim (§5).
- **Human-impostor hard negatives**: the other-sender pool is currently
  random val emails with a random claimed sender. Mine *stylistically
  near-claimed-sender* human impostors (nearest-centroid others) for a hard
  FPR_other tail metric — that's the insider-fraud / compromised-thread case.

### 7.4 Scoring-side items, re-prioritized by this run

- The adaptive scorers (z_persender, mahal_blend, tier_switch) again failed
  to beat `linear_z3` anywhere — with the current noise floor they cannot
  win. Park them until 7.1 lands.
- On v9, plain `cosine` ≥ `linear_z3` at every K (and clearly at K≥16):
  scorer choice interacts with the training objective. Make the scorer a
  per-encoder decision, not a global constant.
- B2 (length-aware observation noise σ(len)) from
  `docs/robustness_mechanisms.md` is still worth piloting, but on **v9-last**
  — and after 7.3, since today no short impostors exist to set σ against.
- A2/A3 (population-prior centroid/covariance shrinkage) target the low-K
  cliff that v9's episodic loss did *not* fix at the benchmarked checkpoint
  (K=4 TPR@1% 0.458 < v7's 0.553) — still live, still retrain-free.

### 7.5 Longer-horizon

- Probabilistic embeddings (B3) subsume the length story and the
  "long-but-boilerplate" FP mode; promote only if B2's residuals demand it.
- Episode-pooled enrollment/inference (A4) — v9 already trains at
  `episode_k=1`; the symmetric experiment (enroll as episodes) is untested.
- If the two-head fusion (7.2) works, revisit the loss: an explicit
  multi-task objective (episodic authorship + synthetic-contrast head with a
  gradient-isolated branch) would train the decoupling end-to-end instead of
  relying on checkpoint geometry.

---

## Appendix A — artifact index

| Artifact | Path |
|---|---|
| Ablation JSON/CSV per arm/corpus | `results/lineage/ablate_{own,common}_v{6..9}.{json,csv}` |
| Probe (genuine-vs-syn classifier) | `results/lineage/probe_v{6..9}.json` |
| Confusion matrices + strata + paired deltas | `results/lineage/confusion_report.json` |
| Raw scores per pool (best ckpts) | `results/lineage/scores_v{6..9}.npz` |
| Raw scores (last ckpts, v7–v9) | `results/lineage/scores_v{7..9}_last.npz`, `last_checkpoint_eval.json` |
| In-training trajectories | `results/lineage/train_trajectories.json` |
| Figures (confusion grid, score dists, log-ROC, length strata, training dynamics) | `results/lineage/figures/*.png` |
| Regeneration script | `scripts/lineage_confusion_report.py` |
| Per-stage logs | `runs/_lineage/*.log` |

## Appendix B — K-sweep, TPR@1% on the common corpus (`linear_z3`; v9 also `cosine`)

| Arm | K=4 | K=8 | K=16 | K=25 |
|---|---|---|---|---|
| v6 | 0.265 | 0.420 | 0.569 | 0.593 |
| v7 | **0.553** | **0.670** | 0.801 | 0.793 |
| v8 | 0.341 | 0.466 | 0.667 | 0.679 |
| v9 | 0.458 | 0.572 | 0.703 | 0.724 |
| v9 (cosine) | 0.511 | 0.606 | **0.764** | **0.809** |
