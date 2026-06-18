# V12 — Multi-generator adversary: closing the cross-generator generalization gap

*2026-06-17. Builds on `docs/v11_results_analysis_memo.md`. Data/results under
`data/synthetic/enron_synthetic_*` and `results/v12/`. Memory:
`multigen-generalization-gap`, `domain-ood-eval-data`.*

## 1. Why v12 exists

v11-lora's headline LLM-forgery detection (TPR@1%FPR_syn 0.875) was measured
against a **single generator (Mistral-7B)**. The v11 memos repeatedly flagged this
as the #1 risk. We measured it directly (2026-06-17): on the **same probe/senders/
scorer**, swapping the synthetic generator from Mistral to held-out frontier
vendors collapses detection.

**The gap (v11-lora, centroid probe, baseline_linear_z3):**

| Adversary | AUC g/syn | TPR@1%FPR_syn |
|---|---|---|
| Mistral-7B (trained on) | ~0.96 | **0.91** |
| Held-out (GPT-4o-mini / Claude / Llama / Gemini, pooled) | 0.75 | **0.045** |

Per-generator OOD (pairwise AUC): Gemini-2.5-Flash 0.614, Claude-3.5-Haiku 0.639,
GPT-4o-mini 0.709, Llama-3.1-70B 0.735. The "reject LLM text" skill was a
Mistral-specific shortcut.

## 2. v12 design

Identical to `v11_synv1_lora.yaml` except:
- **Adversary diversified:** trained on a multi-generator hard-negative set
  (`enron_synthetic_v12_train`, 1045 rows: GPT-4o-mini + Llama-3.1-70B, via
  OpenRouter, hard-negatives only).
- **Held out for eval:** Claude-3.5-Haiku + Gemini-2.5-Flash
  (`enron_synthetic_v12_heldout`, 327 rows) — never seen in training, the honest
  generalization test.
- **Anti-Goodhart monitor:** `pauc/min_other_synthetic_5pct` (the worse of the two
  impostor tails) so checkpoint selection can't slide to the easy synthetic axis.

Config: `configs/experiments/v12_multigen_lora.yaml`; eval config
`configs/experiments/_v12_heldout_eval.yaml`.

## 3. Result (v12 FINAL — checkpoint_best @ epoch 72)

*The 150-epoch run completed and confirms ep72: the anti-Goodhart monitor peaked
at epoch 72 and nothing beat it through 150 (monitor/best 0.656, flat). On the
held-out adversaries, `checkpoint_last` (ep150) is equal-or-marginally-worse than
ep72 (Claude 0.811 vs 0.829, Gemini 0.727 vs 0.746) — training past ~72 plateaus
and slightly overfits to the GPT+Llama training generators. ep72 is the model.*

**v11 → v12 on the HELD-OUT (Claude+Gemini) adversaries the model never trained on:**

| Metric | v11-lora | v12-lora (ep~80) |
|---|---|---|
| Claude-3.5-Haiku AUC | 0.639 | **0.829** |
| Gemini-2.5-Flash AUC | 0.614 | **0.746** |
| `gen` axis mean pAUC@5% | 0.041 | **0.142** (3.4×) |
| held-out pool TPR@1% (centroid probe) | 0.027 | **0.129** (≈5×) |
| held-out pool AUC | 0.696 | **0.866** |

**No authorship regression:** `len:*` and `register:*` slices (seen senders) are
essentially unchanged v11→v12 (e.g. register:cross AUC 0.962→0.955, len:medium
0.963→0.970). The FPR_other guardrail is not violated.

**Conclusion:** training on two generators (GPT+Llama) **transfers to two unseen
frontier generators** (Claude+Gemini). The multi-generator approach works; the gap
is substantially (not yet fully — Gemini remains hardest) closed.

## 4. Caveats
- The 150-epoch run hit a pod disk-quota crash twice (~ep80); resolved by deleting
  scratch data + throwaway smoke runs + (with user authorization) the v6–v9
  `lineage`/`lineage_v2`/`v7_*` checkpoints (their `results/` analyses are kept).
  The final model is `runs/v12/lora/checkpoint_best.pt` (epoch 72).
- `FPR_other≈0` at the synthetic-anchored threshold on the held-out pool is an
  artifact (when synthetics look genuine the threshold is pushed very high); read
  the standard-probe other-sender leak instead.
- The `len/register` OOD slices here use SEEN train senders (inflated vs the
  unseen-test-sender base eval) — only the `gen:*` slices are the v12 finding.

## 5. Next
- Finish the 150-epoch run; refresh §3.
- Gemini still hardest (AUC 0.746) — consider adding a Gemini-family or a 5th
  generator to training, or more volume.
- Fold v12 into the lineage table; re-run the `domain:*` (PAN/blog) axis on v12.
