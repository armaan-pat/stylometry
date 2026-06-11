# V11 — the V9 recipe on no-LLM-positive data, + an explicit LLM-detector

*Drafted 2026-06-11, ahead of the run. Status: **RUNNING** —
`scripts/run_v11_synv1.sh`, W&B group `v11-synv1`, results land in
`results/v11/` (digest printed at the end of the run). Fill the TBD tables from
`results/v11/ablate_common_*.json` and compare against the v7/v8/v9 rows in
`docs/v9_lineage_memo.md` §5.*

```bash
screen -dmS v11 bash -c 'PYTHONPATH=$PWD/src bash scripts/run_v11_synv1.sh 2>&1 | tee runs/_v11/console.log'
```

Five arms, run sequentially on one A40, each trained then evaluated three ways
(own-corpus ablation on syn-v1, common-corpus ablation on syn-v2, and the new
sliced OOD harness). Rebased onto `origin/main@c38fd2b` ("eval, metrics, etc").

---

## 1. Why this run exists

`docs/v9_lineage_memo.md` §3 flags a confound that the lineage never resolved:
the **v7→v8** step changed *two* things at once — it regenerated the synthetic
corpus (quality filter, expanded topics) **and** introduced **cross-register
LLM positives** (syn-v2's 264 emails written by the LLM but stored under the
*real* sender id, as positives). The **v8→v9** step then changed the *objective*
(episodic variable-K′ loss) and the *data distribution* (short-mail corpus +
crop augmentation). So the headline "v9 is best" bundles an objective change, a
distribution change, and a training-data-composition change — you cannot read
off how much each contributed.

**V11 isolates the objective + distribution change from the cross-register-data
change.** It runs the full V9 recipe — episodic variable-K′ loss, `episode_k=1`,
batch 128, `enron_shortmail` corpus, 30% crop augmentation — but on
`enron_synthetic_v1`: **438 rows, 100% `__syn` hard negatives, zero
cross-register positives** (vs syn-v2's 264). Read against v9 (which is the same
recipe on syn-v2), the V11-lora ↔ v9 gap is *exactly* the cross-register-positive
contribution; the V11-lora ↔ v7 gap is the episodic-objective + short-mail
contribution, now cleanly separated from it.

Constraint honored throughout: **no LLM-generated positive** ever enters
training. The only LLM text is the `__syn` hard-negative pool (repulsion-only in
the episodic loss; the positive class for the detector). Crop augmentation makes
short positives, but they are crops of *real* emails, not generated text.

## 1b. Changes incorporated from the 2026-06-11 merge (`c38fd2b`)

This run was rebased onto the eval/metrics update before relaunch. What that
commit changed and how V11 uses it:

- **LLM text is a hard negative, never a positive — now enforced in code.**
  `SyntheticAugmentedDataset(llm_negatives_only=True)` drops any synthetic row
  stored under a *real* sender_id at load time. V11 passes this **explicitly**
  in `train.py` (not just by default) because it is a hard invariant for these
  runs. **Verified at runtime**: every one of the 5 arms loads 438 synthetic
  rows, all 438 `__syn` hard negatives, **0 LLM positives** — and syn-v1 has no
  positives on disk to begin with, so the guarantee holds at three independent
  layers (data, loader, sampler).
- **Register invariance from REAL text** replaces v8's cross-register LLM
  positives. `data.augmentation.register_stratified: true` (now on for all 5
  arms, matching the post-merge `v9_episodic_shortmail.yaml`) makes the sampler
  fill each sender's K episode slots to span as many of *their own* registers
  (formal/casual/terse, via the shared `data/register.py:detect_register`) as
  possible — so every episode carries genuine same-author cross-register
  positives drawn from real email, never from a forgery. Length invariance comes
  from the existing crop augmentation (`crop_prob: 0.3`). Together these recover
  the invariance v8 bought with LLM positives, **without any LLM positive**.
- **Sliced OOD evaluation** (`scripts/build_ood_eval.py` + `scripts/eval_ood.py`,
  `scoring/pairwise.py`). The runner now builds one tagged `{pair, same, slice}`
  set from **unseen TEST senders** and scores every arm's checkpoint per failure
  axis: `len:short|medium|long`, `lenmix:short_long`, `register:cross|same`
  (impersonation/`gen:*` slices are auto-skipped — the single-Mistral syn sets
  carry no `generator` column). This is where the crop + register-stratified
  changes are *measured*: `register:cross` and `len:short` are the slices those
  mechanisms target. Per-slice metrics → `results/v11/ood_<arm>.json` and W&B
  (`ood/slice/*`, `ood/weakest_slice`).
- W&B payloads are now filtered to an SLA-relevant allowlist (`_WANDB_KEEP` in
  `trainer.py`); `eval_ood --wandb` is exempt and logs full per-slice detail.

The `llm_detector` loss, the trainer optimizer/checkpoint plumbing for its head,
and the `embedding_dim` kwarg all merged cleanly alongside these (no conflict;
full suite **71 passing**, incl. the 8 `test_llm_detector_loss.py` tests).

## 2. The five arms

| Arm | Config | Backbone | Objective | What it isolates |
|---|---|---|---|---|
| **frozen** | `v11_synv1_frozen.yaml` | LUAR **frozen**, projection-only (lr 1e-3) | episodic (+0.5 SupCon aux) | how much of V11 needs backbone adaptation vs. a better projection of a fixed LUAR space |
| **lora** | `v11_synv1_lora.yaml` | LUAR + LoRA r16 q/k/v (lr 2e-4) | episodic (+0.5 SupCon aux) | the clean **v8→v9 objective+distribution** cell, cross-register data removed |
| **detector** | `v11_llm_detector.yaml` | LUAR + LoRA (lr 2e-4) | **`llm_detector`** — BCE classification head (+0.3 SupCon aux) | the **best LLM-detector**, objective-built rather than emergent |
| **frozen_supcon** | `v11_synv1_frozen_supcon.yaml` | LUAR **frozen** (lr 1e-3) | **supcon** (non-episodic) | episodic-loss contribution under a frozen backbone (vs `frozen`) |
| **lora_supcon** | `v11_synv1_lora_supcon.yaml` | LUAR + LoRA (lr 2e-4) | **supcon** (non-episodic) | episodic-loss contribution under LoRA (vs `lora`) |

All five: `enron_shortmail` + `enron_synthetic_v1`, `episode_k=1`, batch 128,
crop 0.3, lowered floors, signatures kept, monitor
`pauc/genuine_vs_synthetic_5pct`. The two `*_supcon` arms differ from their
episodic namesakes by **only the loss** (`episodic→supcon`, `episode_k` held at
1) — so the episodic↔supcon delta isolates the episodic variable-K′ loss's
contribution, holding the short-mail distribution, crop augmentation, batch
size and single-email pooling all fixed (`docs/v9_lineage_memo.md` §4's open
"is the episodic loss earning its complexity?" question, answered directly).

## 3. How the detector is made a detector (the novel piece)

The metric-learning losses (supcon, episodic) optimize **per-sender clustering**
and acquire the LLM-vs-human axis only *incidentally*. The v10 analysis showed
that axis is **easy and global** — a frozen linear probe separates Mistral text
from human ~100% — and that it **peaks early (v9 genuine-vs-syn AUC at ep10) then
decays** as capacity reallocates to authorship. So "select the early checkpoint"
(v9-ep10) gives a detector, but a fragile, unmonitored one (it waved through 60%
of wrong-sender *human* mail — see memory `lineage-v9-checkpoint-goodhart`).

V11's detector optimizes that axis **directly and monotonically**. The new loss
`llm_detector` (`src/email_fraud/losses/llm_detector.py`) attaches a linear
classification head — `nn.Linear(128, 1)` on the pooled embedding — and trains it
with **`BCEWithLogitsLoss`**, label `1` iff `sender_id.endswith("__syn")` (LLM
impostor), `0` for human. A per-batch `pos_weight = n_neg/n_pos` corrects the
synthetic-minority imbalance (n_syn=4 of P=16).

Three reasons a classification head + BCE beats metric learning *for detection*:
1. **It maximizes margin on the one decision we care about** instead of diffusing
   capacity across P-choose-2 sender contrasts.
2. **Every email is signal for the boundary** — BCE contrasts *all* synthetics
   against *all* humans each batch, vs. the episodic loss's handful of
   repulsion-only `__syn` queries.
3. **It is sender-agnostic** — the boundary uses no enrolled centroid, so it
   generalizes to senders never seen at enrollment.

A 0.3 SupCon aux is retained so the embedding still clusters per-sender enough
for centroid enrollment — that keeps the `genuine_vs_synthetic` CentroidProbe and
the checkpoint monitor meaningful and comparable to the rest of the lineage. The
head's weights persist in the checkpoint under `loss_state_dict` (Trainer change),
so the trained classifier is available for direct-logit scoring if the embedding
ever collapses too far for centroid scoring.

**Plumbing** (so the head actually trains): `BaseLoss` is an `nn.Module`; the
Trainer now folds `loss_fn.parameters()` into the AdamW group and moves the loss
to device (`training/trainer.py`), and `train.py` passes `embedding_dim =
encoder.projection_dim` to the loss (filtered, so only `llm_detector` receives
it). Verified end-to-end by `tests/test_llm_detector_loss.py` (8 tests) and the
`_smoke_v11_detector.yaml` smoke run (checkpoint round-trips with
`classifier.{weight,bias}`).

## 4. Results (fill after the run)

Common corpus (`enron_shortmail` + syn-v2), K=8 enroll — from
`results/v11/ablate_common_<arm>.json`. **Report `FPR_other` alongside the
synthetic tail** (the recorded Goodhart trap) and note the best-checkpoint epoch.

| Arm | best scorer | AUC[g/syn] | TPR@5% | TPR@1% [95% CI] | AUC[g/other] | FPR_other@5% | ep |
|---|---|---|---|---|---|---|---|
| frozen | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| lora | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| detector | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

K-sweep (tpr1 by enrollment K), own corpus (syn-v1) from
`results/v11/ablate_own_<arm>.json`:

| Arm | K=4 | K=8 | K=16 | K=25 |
|---|---|---|---|---|
| frozen | TBD | TBD | TBD | TBD |
| lora | TBD | TBD | TBD | TBD |
| detector | TBD | TBD | TBD | TBD |

Frozen-linear-probe separability (genuine vs synthetic) from
`results/v11/probe_<arm>.json`: TBD — expect ~ceiling for the detector arm.

**Sliced OOD verification** (unseen test senders, pairwise AUC / pAUC@5% per
slice) from `results/v11/ood_<arm>.json` — this is the direct test of whether
crop + register-stratified bought invariance:

| Arm | len:short | len:long | lenmix:short_long | register:cross | register:same | weakest slice |
|---|---|---|---|---|---|---|
| frozen | TBD | TBD | TBD | TBD | TBD | TBD |
| lora | TBD | TBD | TBD | TBD | TBD | TBD |
| detector | TBD | TBD | TBD | TBD | TBD | TBD |
| frozen_supcon | TBD | TBD | TBD | TBD | TBD | TBD |
| lora_supcon | TBD | TBD | TBD | TBD | TBD | TBD |

The two telling cells: **`register:cross`** (different-register same-author pairs
— should *not* drop far below `register:same` if register-stratified worked) and
**`len:short`** / **`lenmix:short_long`** (should *not* collapse if crop aug
worked). The detector arm may score these lower — it optimizes a global LLM/human
boundary, not pairwise authorship — which is expected, not a regression.

## 5. How to read it (the questions this run answers)

1. **Did the episodic objective + short-mail distribution earn the v9 gain on
   their own, without cross-register positives?** Compare **V11-lora** to
   `docs/v9_lineage_memo.md` v7 (0.947 AUC / 0.670 tpr1, common corpus). If
   V11-lora ≳ v7, the objective/distribution change stands alone; the v9↔V11-lora
   gap then attributes the *remainder* to cross-register data.
2. **Does V11-lora's `FPR_other` avoid v9's collapse?** v9's monitored ep-10
   checkpoint hit FPR_other 0.595. Hard-neg-only training (no LLM positives
   pulling LLM text toward sender centroids) may *strengthen* the
   reject-LLM-flavored-text shortcut — watch whether that re-inflates the early
   genuine-vs-syn peak and the Goodhart risk with it.
3. **How much does freezing cost?** frozen vs lora on the same data/objective.
4. **Is the objective-built detector better than the emergent v9-ep10 one?**
   Compare the detector arm's genuine-vs-syn AUC/TPR@1% to v9-ep10's 0.953/0.572
   — and confirm its genuine-vs-other is correspondingly weak (by design; it is
   the detector half of the two-model design in `docs/v10_two_model_memo.md`,
   not an authorship model).

## 6. Caveats

- Single generator (Mistral-7B) still wrote every `__syn` impostor — the
  detector's numbers against GPT/Claude/Gemini are unknown. This is the standing
  top data risk across the whole lineage.
- The detector's monitor is centroid-based; if the 0.3 SupCon aux is too weak to
  keep the embedding centroid-usable, select it by `checkpoint_last` and score
  via the classifier logit instead (head is persisted in `loss_state_dict`).
- Trust the bootstrap Δ-CIs in the ablation JSON, not point-estimate orderings —
  sub-2pp TPR@1% gaps are noise even with the expanded probe.
