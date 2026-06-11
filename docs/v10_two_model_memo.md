# V10 + V9: The Two-Model Design — one recipe, two specialists

*2026-06-11. Design + results memo for the v10 run
(`scripts/run_v10_overnight.sh`, W&B group `v10-two-model`). Builds directly
on the lineage analysis in `docs/v9_lineage_results_analysis.md`; results land
in `results/lineage_v2/` (digest: `results/lineage_v2/DIGEST.txt`).*

---

## 1. The roles

| Model | Checkpoint selection | Question it answers | Owns which adversary |
|---|---|---|---|
| **v9** — *LLM-detector specialist* | `pauc/genuine_vs_synthetic_5pct` (existing `runs/lineage/v9/checkpoint_best.pt`, ep 10) | "Is this email LLM-written mimicry of the claimed sender?" | synthetic / AI-generated impersonation |
| **v10** — *authorship specialist* | `pauc/genuine_vs_other_5pct` (**new metric**, `runs/lineage_v2/v10/checkpoint_best.pt`) | "Was this email written by the claimed human?" | real-human impostors (wrong sender, compromised thread, manual BEC) |

v10's config (`configs/experiments/v10_episodic_authorship.yaml`) is
byte-identical to v9's **except the monitor line**. That is deliberate: the
lineage run proved the v9 recipe already produces both specialists in a single
training run — what was missing was a selection rule that knows there are two.

## 2. Why two models — the evidence, stated plainly

**Fact 1: the two skills mature at opposite ends of training.** The
per-epoch curves from the 2026-06-10 run
(`results/lineage/figures/training_dynamics.png`) show, for the v9 recipe:

- *genuine-vs-synthetic* AUC peaks at **epoch 10** (0.944) and then decays to
  0.899 by epoch 150;
- *genuine-vs-other* AUC starts at **0.70 at epoch 10** and climbs for 140
  epochs to **0.937** — the best human-impostor discrimination this project
  has measured.

Why this shape: "LLM-ness" is an easy, global feature — one generator
(Mistral-7B) wrote every synthetic impostor, and a frozen linear probe
separates its text from human text with ~100% accuracy (`probe_v9.json`). The
encoder finds that direction in a handful of epochs. Per-sender authorship is
the hard, fine-grained task the episodic loss actually optimizes, and it keeps
reshaping the embedding space long after — slowly trading away some of the
early synthetic margin as capacity reallocates toward sender-vs-sender
contrast.

**Fact 2: no single checkpoint holds both peaks.** Measured on the identical
common probe (`results/lineage/last_checkpoint_eval.json`):

| Checkpoint | AUC genuine-vs-syn | TPR@1%FPR_syn | AUC genuine-vs-other | FPR_other @5%FPR_syn thr |
|---|---|---|---|---|
| v9 ep10 (detector) | **0.953** | **0.572** | ~0.70 | **0.595** ← unshippable alone |
| v9 ep150 (authorship-like) | 0.888 | 0.223 | **0.933** | 0.002 |

The ep-10 model waves through 6 of 10 wrong-sender *human* emails at its
deployed threshold; the ep-150 model misses 78% of synthetic impostors at the
1% tail. A weight-space blend doesn't escape the trade either — the α-sweep
(`results/lineage_v2/soup_v9.json`) just walks along the same frontier
(α=0.5: AUC_syn 0.957 but AUC_other only 0.854). **The trade-off is real,
not a checkpoint-selection accident** — so the right response is to stop
forcing one embedding geometry to answer two different questions.

**Fact 3: the two errors are independent enough to fuse profitably.** The
detector's weakness (human impostors) is exactly the authorship model's
strength, and vice versa. A preliminary fusion (v9-ep10 detector × v9-ep150
as a stand-in authorship model, before v10 finished training) already gave:

| System | TPR (genuine kept) | FPR_other | FPR_syn |
|---|---|---|---|
| detector alone @5% FPR_syn | 0.837 | 0.595 | 0.050 |
| authorship-stand-in alone | 0.473 @1% tail | 0.002 | (weak tail) |
| **AND-gate fusion, 5% targets** | **0.716** | **0.045** | **0.022** |

No single checkpoint in the entire v6–v9 lineage reaches any operating point
with *both* FPRs ≤ 5% and TPR > 0.6. The fused pair does it immediately —
and v10 should improve the genuine-retention side further, since it is
selected at the authorship peak rather than approximated by `checkpoint_last`.

## 3. How the fusion works (deployment semantics)

Each model enrolls senders in its own embedding space and scores every query
against the claimed sender's centroid (`linear_z3`, K=8 — same as today).
Two rules, evaluated by `scripts/eval_two_model_fusion.py`:

- **AND-gate (the deployable rule).** Accept iff
  `score_v10 ≥ τ_auth` **and** `score_v9 ≥ τ_det`, where τ_det is anchored on
  v9's synthetic pool (hold FPR_syn at target) and τ_auth on v10's
  other-sender pool (hold FPR_other at target). Each specialist gates the
  adversary it owns, so the two dials are independent: tightening the
  LLM-paranoia knob doesn't change how strictly identity is checked, and
  vice versa. An alarm can also be *attributed*: "rejected by the detector"
  vs "rejected by identity" — useful for analyst triage.
- **Soft-min (the threshold-free check).** Rank-normalize each model's score
  against its own impostor distribution and take the min. Used to verify the
  fusion dominates on AUC, not just at one operating point.

Cost: 2× encoder inference per query (~LUAR-MUD forward, small) and a second
enrolled centroid per sender. No new training objective, no new data.

## 4. Results (2026-06-11 overnight run)

> **Status: RUNNING — filled from `results/lineage_v2/DIGEST.txt` when the
> overnight run completes.** Tables below list the exact source files.

**4a. v10 vs the field** (common corpus, `linear_z3`, K=8; from
`results/lineage_v2/ablate_v10_{best,last}.json`, `ablate_v9_{best,last}.json`):

| Model | AUC[g/syn] | TPR@1% | TPR@5% | AUC[g/other] | FPR_other@5% | TPR@1% (cropped short impostors) |
|---|---|---|---|---|---|---|
| v9 best (detector) | 0.953 | 0.572 | 0.837 | 0.723¹ | 0.595 | 0.242¹ |
| v9 last | TBD | TBD | TBD | TBD | TBD | TBD |
| **v10 best (authorship)** | TBD | TBD | TBD | TBD | TBD | TBD |
| v10 last | TBD | TBD | TBD | TBD | TBD | TBD |

¹ from the smoke-test ablation; final numbers may differ by sampling noise.

Acceptance criteria for "v10 is the best authorship model": AUC[g/other] ≥
0.933 (the v9-ep150 value it must at least match) and FPR_other@5% ≤ 0.01 —
with the *synthetic* columns explicitly allowed to be weak (that's v9's job).

**4b. The fused pair** (from `results/lineage_v2/fusion_v10xv9.json`):

| System | TPR | FPR_other | FPR_syn |
|---|---|---|---|
| AND-gate @1% targets | TBD | TBD | TBD |
| AND-gate @5% targets | TBD | TBD | TBD |
| soft-min AUCs | TBD (auc_syn / auc_oth) | | |

Compare against: best single-model joint operating point in the v1 lineage
was v7 (TPR 0.822 with FPR_other 0.172 / FPR_syn 0.050 — FPR_other 3.4× over
budget).

**4c. Single-model alternatives, for honesty** — the soup
(`soup_v9.json`) and the repaired v6 baseline under the anti-Goodhart
`min(pauc_syn, pauc_other)` monitor (`ablate_v6_repair.json`): TBD.

## 5. Why not just one balanced model?

Three options were on the table after the lineage analysis; the two-model
design wins on all three axes we care about:

1. **A composite monitor** (min of both paucs — now implemented and used for
   the v6 repair arm) picks the *least-bad compromise checkpoint*. It
   prevents the ep-10 disaster, but the trajectory data says the compromise
   point is strictly inside the Pareto frontier the two endpoints define —
   you give up tail performance against both adversaries to get one model.
2. **Weight-space soup** interpolates along the same frontier (the α-sweep
   confirms it) — same compromise, just tunable.
3. **The two-model pair** keeps both peaks. Its costs are operational
   (2× inference, two profiles per sender), not statistical. And it
   future-proofs: the detector half can be swapped for a multi-generator
   ensemble (the §7.3 roadmap) without touching the identity model, and the
   single-generator-shortcut risk (v9's perfect probe separability on
   Mistral text) is *quarantined* in the component built to be replaced —
   it no longer contaminates the authorship score.

The deeper lesson for the loss-design roadmap: if a multi-task objective is
ever built (analysis memo §7.5), it should train **two heads** on a shared
trunk, not one score — the lineage data is direct evidence that a single
scalar geometry cannot represent both decisions at their respective peaks.

## 6. Caveats

- v10's authorship monitor is measured on the in-training CentroidProbe
  (44 senders); its peak may be noisy — that's why the overnight run
  evaluates `checkpoint_last` alongside `checkpoint_best`.
- The detector half is still single-generator (Mistral-7B). Its numbers
  against GPT/Claude/Gemini impostors are unknown; the AND-gate's FPR_syn
  guarantee only extends to the generators it's tested on. Multi-LLM
  impostor suite remains the top data priority.
- The cropped-short-impostor column (new `--crop-syn` ablation flag) is the
  first measurement of short-query forgery; expect it to be ugly for every
  model (the smoke run showed v9-best at 0.24 TPR@1% vs cropped impostors) —
  it defines the next battlefield, not a regression.
- Fusion thresholds are set on the same probe they're evaluated on; before
  shipping, re-anchor on a held-out calibration split.
