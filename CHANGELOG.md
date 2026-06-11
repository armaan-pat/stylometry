# Changelog

All notable changes to the email fraud / stylometry detection project are
documented here.  Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

---

## [Unreleased]

### Changed (2026-06-11) — LLM synthetics are hard negatives ONLY; LLM-free register/length invariance; multi-LLM generation; sliced OOD eval

Three connected changes: (1) enforce that LLM-generated text is never a positive,
(2) recover register/length invariance from real data instead of LLM positives,
(3) generate hard negatives from multiple LLMs, and (4) add an OOD evaluation
harness that reports metrics per failure axis.

**Enforce: LLM text is a hard negative, never a positive** — the project
invariant, enforced at two layers so it holds regardless of what is already on
disk:

- `scripts/generate_synthetic_emails.py`: `--llm-positives` now defaults to
  `exclude` (was `include`). New datasets emit only `hard_neg` rows
  (`sid__syn`); the `cross_register` / `cross_length` LLM-positive paths are
  deprecated and only reachable via the explicit ablation.
- `data/synthetic.py`: `SyntheticAugmentedDataset(llm_negatives_only=True)`
  (default) **drops any synthetic row stored under a real `sender_id` at load
  time**, with a loud warning. `enron_synthetic_v2` (used by v9/v10) contains old
  LLM positives — they are now dropped automatically, so no regeneration is
  required to be safe. Storing an LLM impersonation under a real sender_id
  teaches the encoder a convincing forgery belongs in the author's cluster,
  opposing the fraud signal — overrides the V8 A/B result that endorsed it.

**B — LLM-free register/length positives** (replace the deprecated LLM positives
with real-data signal):

- New `data/register.py` — shared `detect_register` (formal/casual/terse) +
  `partition_by_register`, the single source of truth for both the generator and
  the sampler. `generate_synthetic_emails.py` now imports it instead of its own
  copy.
- **Register-stratified episode sampling** (`data/samplers.py`,
  `_register_stratified_pick`): when `data.augmentation.register_stratified` is
  set, each sender's K in-batch emails are picked to cover as many registers as
  the sender wrote in, so every episode contains genuine cross-register
  same-author positives. Both `PKSampler` and `SyntheticBalancedSampler` take an
  optional `register_labels` list; `train.py` computes labels via
  `_maybe_register_labels` and passes them through.
- **Length invariance** comes from the existing crop augmentation (now on,
  `crop_prob: 0.3`). `register_stratified: true` + `crop_prob: 0.3` enabled in
  `v9_episodic_shortmail.yaml` and `v10_episodic_authorship.yaml`.

**A — Multi-LLM hard-negative generation** (`scripts/llm_backends.py`, new):

- Pluggable `Generator` backends: `hf` (local transformers, moved here so
  torch/transformers load lazily — API-only runs need no GPU stack), `groq`,
  `openrouter`, `together`, `deepseek`, `openai` (one OpenAI-compatible client),
  `gemini`, `ollama`. API backends read `<BACKEND>_API_KEY`, run concurrently
  (`--request-workers`), and retry on 429/5xx with backoff.
- `generate_synthetic_emails.py` gains `--generators backend:model ...` and
  round-robins jobs across a MIX of generators in one run; each row is tagged in
  the existing `generator` column (now `<backend>:<model>`). This keeps the
  detector from overfitting to one model's fingerprint and enables
  held-out-generator OOD eval. `--model` stays as a single-HF back-compat
  shorthand. Generator composition is printed and logged to W&B.

**OOD evaluation harness** (new):

- `scoring/pairwise.py` — reusable `load_encoder` + `score_pairs` (dedups texts,
  cosine→[0,1]); replaces the duplicated `_score_pairs` going forward.
- `scripts/build_ood_eval.py` — builds one tagged `{pair, same, slice}` JSONL
  from on-disk assets. Slices: `len:short|medium|long`, `lenmix:short_long`,
  `register:cross|same`, and `gen:<backend>:<model>` (impersonation, one per
  generator). Length buckets: short ≤20 words, long ≥80.
- `scripts/eval_ood.py` — scores all pairs once, reports
  `compute_verification_metrics` per slice + overall, ranks weakest by pAUC@5%,
  skips single-class slices. Domain OOD folds in via
  `--extra-pairs domain:<name>=pairs.jsonl`.

**OOD eval — unseen senders + domain corpora** (follow-ups closing the two gaps
the harness left open):

- `generate_synthetic_emails.py` gains `--from-split {train,validation,test}`
  (default train). `--from-split test` builds impostors for UNSEEN test senders;
  the run prints an eval-only warning and the output carries a `source_split`
  column. `build_ood_eval.py` now looks up the real side of impersonation pairs
  across ALL splits (splits are sender-disjoint), so `gen:*` slices work for
  either train-sender (SEEN) or test-sender (UNSEEN) synthetics; provenance is
  read from `source_split` and printed. This resolves the earlier
  "`gen:*` slices only cover seen senders" caveat.
- `scripts/prepare_external_pairs.py` (new) — converts an external corpus into
  the unified `{pair, same, slice}` JSONL eval_ood ingests. `--format pan`
  joins PAN pairs+truth files on `id`; `--format authors` samples balanced
  same/different pairs from a generic `{text, author}` JSONL (Blog Authorship,
  Reddit, Amazon, …). Optional `--preprocess` runs the project cleaner;
  `--max-chars` truncates. This is the end-to-end path for domain-OOD slices
  (real text from another domain — the one OOD axis Enron data can't synthesise).

Tests: `tests/test_external_pairs.py` (4) + 2 added to `tests/test_ood_eval.py`.

**OOD metrics on W&B** (`scripts/eval_ood.py --wandb`):

- Logs every per-slice metric as a summary scalar (`ood/slice/<slice>/<metric>`),
  plus **per-axis aggregate means** (`ood/axis/<len|gen|register|domain|...>/<metric>_mean`)
  — one robust number per OOD axis so a new impersonator model or domain shows up
  as a single movement instead of being buried across many per-slice keys.
  Promotes overall headline scalars (`ood/overall/AUC` etc.) and the weakest
  slice (`ood/weakest_slice`, `ood/weakest_pAUC@5%`), and logs an `ood/by_slice`
  table (slice, axis, n, AUC, pAUC@5%, TPR@FPR=1%, EER, c@1).
- `--wandb-run-id` attaches the OOD metrics to an EXISTING run (e.g. the training
  run for the checkpoint) so they land in that run's summary next to the
  in-distribution numbers; otherwise a new `ood-eval`-tagged run is opened. Note:
  eval_ood logs to its own/attached run and is NOT subject to the trainer's
  `_WANDB_KEEP` allowlist, so full per-slice detail is reported.
- Generation runs now log provenance: `from_split` / `eval_only` in stats +
  config and `split-<split>` / `eval-only` tags, so test-sender (eval-only)
  impostor datasets are filterable from training augmentation sets.

Tests: `tests/test_register_sampling.py` (11), `tests/test_ood_eval.py` (9),
`tests/test_external_pairs.py` (4) — full suite 63 passing. Memory:
`never-llm-positives`,
`register-stratified-sampling`, `ood-eval-harness`, `editable-install-divergence`
(the editable install resolves to a different checkout than this clone — use
`PYTHONPATH=$PWD/src` to test clone code).

---

### Added (2026-06-09) — episodic variable-K training + short-email fixes (V9 recipe)

Implements items 1+2 of the attack order in `docs/robustness_mechanisms.md`
(A1 + B1) — they share one retrain, launched via the new
`configs/experiments/v9_episodic_shortmail.yaml`.

**A1 — Episodic, variable-K prototype loss** (`losses/episodic.py`, registered
as `episodic`)

- Trains the way we infer: per batch, each sender's embeddings are split into
  a support set of size K′ ~ uniform{`support_k_min`..`support_k_max`} and a
  query set; prototypes are (renormalized) support means; queries are
  classified against all in-batch prototypes with the deployed cosine
  distance via cross-entropy. Because K′ is sampled small, the encoder is
  optimized so small-sample means are already discriminative.
- Synthetic `__syn` hard negatives never form prototypes (deployment never
  enrolls them); they stay in the query pool and are repelled from their
  mimicked sender's prototype via `-log(1 - p(mimicked))`. Cross-register
  positives (real sender_id) participate as ordinary episode members.
- SupCon kept as an aux term: `L = L_proto + supcon_weight·L_supcon`
  (default 0.5). New `LossConfig` fields: `support_k_min`, `support_k_max`,
  `supcon_weight`. `BaseLoss.requires_sender_ids` + Trainer plumbing pass the
  raw sender-id strings (episode-strided) into losses that need them.
- The V9 config also sets `encoder.episode_k: 1` so training embeddings are
  single-email — the same representation enrollment/inference use (closes the
  §A4 episode mismatch as a side effect) — and `batch_size: 128` (P=16
  senders × k=8; k must stay 8 because synthetic senders have only 8–9 emails).

**B1 — train/test length mismatch** (`data/preprocessing.py`,
`data/augment.py`, `scripts/prepare_data.py`)

- `min_body_words` and `min_alnum_ratio` were declared in
  `PreprocessingConfig` but **never enforced** — `_is_usable` only checked
  chars. They are now enforced, and the floors are lowered to sanity levels
  (chars 50→20, words →5) in `base.yaml` / config defaults / `prepare_data.py`
  CLI (new `--min-body-words`, `--min-alnum-ratio` flags).
- **Crop augmentation**: with `data.augmentation.crop_prob` (default 0 = off),
  a training email is replaced by a random contiguous 5–60-word span of
  itself (same label; half the crops anchored at the greeting). Crops slice
  the original string so line breaks survive. Implemented as
  `CropAugmentedDataset`, wrapped last in `train.py` so synthetics are
  cropped too while the probe / hard-negative mining still read uncropped
  `._texts`.
- Regenerated `data/processed/enron_shortmail` with
  `--min-body-chars 20 --min-body-words 5 --no-strip-signatures` (keep
  sign-offs — most of a 10-word email's signal): same 44 train senders,
  34,476 train emails (+4.4k vs the old floor), 13.2% under 10 words; all 44
  synthetic-real sampler pairs still eligible at k=8.

Tests: `tests/test_episodic_loss.py`, `tests/test_crop_augment.py`. Smoke
config: `configs/experiments/_smoke_episodic.yaml`.

**Lineage benchmark** — `scripts/run_lineage_v6_v9.sh` sequentially trains
v6 (`v6_bench.yaml`, the V6 recipe with the modern monitor) → v7 (V7.3
recipe) → v8 (V7.3 + syn-v2) → v9, then runs the genuine-vs-synthetic probe
plus scorer ablations on each checkpoint **twice**: against each arm's own
corpus (historical parity) and against a common production-like corpus
(enron_shortmail + syn-v2) for apples-to-apples comparison. Companion memo
with mechanisms, historical numbers, and fill-in result tables:
`docs/v9_lineage_memo.md`.

---

### Changed (2026-06-09) — scoring calibration, checkpoint monitor, probe size

**Per-sender score calibration shipped into the production head**
(`heads/prototypical.py`, `scoring/score_functions.py`)

- New `CALIBRATED_SCORE_FNS` family: `z_persender_sigmoid(cos, spread, z_scale)`
  where `z_scale` is the sender's leave-one-out genuine-z p90, shrunk toward a
  pooled global prior with pseudo-count `n0=8` (James-Stein style — sparse
  senders trust the population, dense senders trust themselves). Ported from
  the `scoring/adaptive.py` ablation winner (`docs/scoring_ablation_results.md`:
  opens score reachability 5.8% → 42% above 0.8 and improves cross-sender
  threshold consistency without hurting AUC).
- `PrototypicalHead` computes/refreshes calibration lazily after `fit()`;
  `score_raw()` now also returns `z_scale`; `score_fn: z_persender_sigmoid`
  is selectable from YAML. `CentroidProbe` evaluates calibrated fns alongside
  the plain registry (added to `eval_score_fns` in the v7 configs), and the
  probe raw dump rows are now `(cos_sim, spread, z_scale)` —
  `analyze_thresholds.py` handles both old and new formats.

**Checkpoint monitor switched to the low-FPR tail**

- v7/smoke configs now monitor `pauc/genuine_vs_synthetic_5pct` (mode=max)
  instead of `auc/genuine_vs_synthetic`, whose epoch-7 peak made
  `checkpoint_best.pt` name the wrong file (the documented V7.3 footgun).
- Removed the misleading hard-coded "New best val/loss=…" log line inside
  `_save_best_checkpoint` (it printed val/loss as "best" whatever the
  configured monitor was).

**CentroidProbe expanded ~3× and made configurable**

- New `probe:` config section (`ProbeConfig`); defaults raised from
  30×4 genuine / 200 other / 200 synthetic to **60 senders × 6 queries
  (360 genuine) / 600 / 600** (capped by availability). The old probe's
  bootstrap CIs (±0.13 on TPR@1%) were wider than most candidate
  improvements. `ablate_adaptive_scorers.py` CLI defaults raised to match.

See `docs/robustness_mechanisms.md` (new) for the design directions these
changes feed into: low-K and short-email robustness as modeling problems
rather than abstain gates.

---

### Diagnosis (2026-06-03)

Full failure-mode audit of the v7 model on 2 000 test pairs revealed five
root causes:

| Mode | Rate | Root cause |
|---|---|---|
| False positives | 30.0 % | Diff-author scores cluster near 0.5; short corporate prose is stylistically flat |
| False negatives | 8.1 % | Same author switches register (formal job app → casual text); model confuses style shift with author change |
| Score separation | 1.23× gap/std | Class distributions barely separated; threshold placement is fragile |
| Synthetic lag | AUC_syn 4.5 % below AUC_oth | LLM impostor prose lands closer to target centroid than random human prose |
| Short-email signal poverty | 3.3 % of emails < 10 words | Near-zero stylometric signal; drives both FP and FN |

Key data point: **74.1 % of same-author test pairs span different topic
domains** (admin↔personal, personal↔trading/ops, etc.).  The synthetic
training data prior to this change contained zero cross-register positive
examples — every generated email was stored as a hard negative — so the
encoder had no training signal for topic/register invariance within a sender.

---

### Changed — `scripts/generate_synthetic_emails.py`

**Cross-register positive generation** (new primary change, addresses #1 FN
failure mode)

- Added `--cross-register-fraction` arg (default `0.4`).  40 % of emails
  generated per sender are now *cross-register positives* stored under the
  real `sender_id` (no `__syn` suffix), so SupConLoss treats them as genuine
  positives.  The remaining 60 % are hard negatives stored as `sid__syn`
  (unchanged behaviour).
- Added `_build_cross_register_prompt()`: shows style-context examples from
  one register (formal or casual) and explicitly instructs the LLM to write
  as the same person in the *opposite* register.  The instruction names the
  invariant — "preserve their sentence rhythm, punctuation habits,
  characteristic phrases" — which is exactly what the encoder needs to learn.
- Added `_detect_register()` and `_partition_by_register()`: lightweight
  keyword-based register classifier that buckets each sender's real emails
  into `formal / casual / terse`.  Used to select the right style-context
  pool (e.g. formal examples → casual output) so cross-register prompts are
  grounded, not hypothetical.

**Topic pool expansion** (addresses FP failure mode — domain-generic prose)

- Split `_TOPICS` into three named pools:
  - `_BUSINESS_TOPICS` (24 items, expanded from the original 20): professional
    email topics used for hard-neg generation and casual→formal cross-register.
  - `_PERSONAL_TOPICS` (15 new items): casual, personal, and social topics used
    for formal→casual cross-register generation.  These topics were entirely
    absent before this change.
  - `_TERSE_TOPICS` (5 items): very short-form topics (confirmations, forwards,
    OOO notes) that mirror the short-email sub-distribution in the test set.

**Per-generation example variation** (bug fix)

- Previously, the same `n_examples` style-context emails were reused for *all*
  `n_per_sender` generations from a sender.  Every prompt for a sender was
  therefore identical up to the topic string.
- Now `_plan_sender_jobs()` samples fresh style-context examples independently
  for each generation job.  This increases prompt diversity and reduces
  redundancy in the output embeddings.

**Quality filter** (new, reduces near-copy contamination)

- Added `_quality_ok()` function.  Applied after preprocessing, it rejects:
  - Outputs shorter than `--min-words` (default 15 words).
  - Outputs where Jaccard overlap with any style-context example exceeds
    `--max-overlap` (default 0.40) — catches near-copies and paraphrase
    regurgitation.
- Acceptance statistics are now reported separately for `hard_neg` and
  `cross_register` modes so generation quality can be monitored per mode.

**Output schema additions**

The Arrow dataset now includes three additional columns:
- `generation_mode`: `"hard_neg"` | `"cross_register"`
- `topic`: the topic string passed to the LLM
- `context_register`: register of style-context examples (`"formal"` | `"casual"` | `"mixed"`)

Existing columns (`text`, `sender_id`, `source_sender_id`) are unchanged.
Backward compatibility: any code that only reads `text` + `sender_id` is
unaffected.

**CLI defaults updated**

- `--n-per-sender` default: `10` → `15` (to maintain roughly the same number
  of hard negatives after the cross-register split).
- `--n-examples` default: `5` (unchanged).
- Style-context example truncation: `600 chars` for hard-neg, `400 chars` for
  cross-register (shorter to leave room for the register instruction in context).

---

### Changed — `src/email_fraud/data/synthetic.py`

- Rewrote module docstring to document both generation modes and explain the
  routing logic (`__syn` suffix → hard negative / SyntheticBalancedSampler;
  no suffix → positive pool / PKSampler).
- Updated `SyntheticAugmentedDataset` docstring to note that the Arrow dataset
  may now contain both `hard_neg` and `cross_register` rows, and that both are
  handled correctly without code changes (routing is entirely determined by
  whether `sender_id` ends with `__syn`).
- No logic changes.  The existing concatenation of `text` + `sender_id` already
  routes cross-register positives correctly because they carry the real sender ID.

---

### Unchanged (deferred)

- `SyntheticBalancedSampler`: no changes needed.  Cross-register rows (real
  `sender_id`) are transparently merged into the real sender's positive pool and
  sampled by PKSampler.  The guaranteed real/syn pairing logic is unaffected.
- `SupConLoss`, `TripletLoss`: no changes.  Cross-register positives are
  attracted to same-sender anchors automatically through sender_id equality.
- Training configs: no changes.  Re-generate the synthetic dataset with the
  revised script and point `augmentation.synthetic_path` at the new output.

---

### Added — `scripts/diagnose_synthetic_quality.py`

Pre-training quality gate for cross-register synthetic emails.

**Problem it solves:** `SupConLoss` is more sensitive to positive-label noise
than negative-label noise.  If the LLM produces generic prose instead of
capturing the author's voice, the cross-register email lands in the wrong
region of embedding space.  Training with it as a positive then *drags the
real centroid toward that wrong region* — making performance worse, not better.

**What it measures** (using the existing checkpoint, no retraining required):

| Metric | What it means |
|---|---|
| `rank_1_rate` | Fraction of cross-register emails whose true sender centroid is the nearest among all profiled senders. **Primary go/no-go metric.** |
| `mean_sim_to_own` | Average cosine similarity to the claimed sender's centroid. |
| `mean_delta` | `sim_to_own − mean(sim_to_all_others)`. Positive = good positive; negative = bad positive. |
| `frac_neg_delta` | Fraction of cross-register emails that are *closer to a different sender* than to their own. Should be low. |

All three metrics are compared side-by-side against:
- **Real emails** — the ceiling (what a perfect positive looks like).
- **Hard-neg synthetics** — the floor (these should rank low, confirming the
  centroid pool is discriminative).

**Decision thresholds** (built into the verdict output):

| Condition | Action |
|---|---|
| rank_1_rate ≥ 0.55 AND mean_delta ≥ 0.05 | Safe to train as-is |
| rank_1_rate 0.40–0.55 OR mean_delta 0.00–0.05 | Filter with `--sim-threshold 0.05` |
| rank_1_rate < 0.40 OR mean_delta < 0.00 | Quality too low; reduce `--cross-register-fraction` or retune generation |

**Optional filtering (`--save-filtered`):** writes a new Arrow dataset that
keeps all `hard_neg` rows unchanged and removes `cross_register` rows below
`--sim-threshold` (default 0.0).  The training config then points at this
filtered path instead of the raw synthetic output.

**Typical workflow:**
```bash
# Step 1 — generate
python scripts/generate_synthetic_emails.py \
    --config configs/experiments/v7_luar_lora_syn_mahal_eval.yaml \
    --n-per-sender 15 --cross-register-fraction 0.4 \
    --output data/synthetic/enron_synthetic_v2 --load-in-4bit

# Step 2 — diagnose (no GPU training, just inference)
python scripts/diagnose_synthetic_quality.py \
    --config  configs/experiments/v7_luar_lora_syn_mahal_eval.yaml \
    --checkpoint runs/v7_luar_lora_syn_mahal/<ts>/checkpoint_epoch_150.pt \
    --synthetic data/synthetic/enron_synthetic_v2 \
    --data-dir  data/processed/enron \
    --out-json  results/synthetic_quality_v2.json

# Step 3a — if verdict is SAFE, train directly
# Step 3b — if verdict is BORDERLINE, filter first then train
python scripts/diagnose_synthetic_quality.py ... \
    --save-filtered data/synthetic/enron_synthetic_v2_filtered \
    --sim-threshold 0.05

# Step 4 — retrain pointing at filtered dataset
# (update augmentation.synthetic_path in your experiment config)

# Step 5 — compare FN rate on cross-domain pairs with error_analysis.py
```

---

### Next steps

1. Re-run `generate_synthetic_emails.py` with `--cross-register-fraction 0.4`
   to produce a new synthetic dataset.
2. Run `diagnose_synthetic_quality.py` on the output and read the verdict
   before committing to retraining.
3. If verdict is SAFE or BORDERLINE (with filtering), retrain v7 / launch v8.
4. Re-run `error_analysis.py` and compare FN rate — particularly the
   formal-job-application vs casual-text pairs that were the top FN failures.
5. Consider adding a `--min-words` gate to the *evaluation* pipeline to skip
   pairs where either email is under 15 words (abstain rather than misclassify).
6. Investigate threshold calibration: shift the cosine decision boundary from
   0.5 toward ~0.62 to reduce the 30 % FP rate at acceptable FN cost.
