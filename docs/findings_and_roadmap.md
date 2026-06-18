# Stylometry — Findings & Roadmap (post-v12)

*2026-06-18. Strategic synthesis of the v11→v12 cycle and the prioritized path
forward. Companion to the detailed memos: `docs/v11_results_analysis_memo.md`
(v11 ablation), `docs/v12_multigen_results.md` (v12 numbers),
`docs/EXPERIMENT_STATUS.md` (lineage history). Results in `results/v11/`,
`results/v12/`; eval data in `data/ood/` and `data/synthetic/`.*

---

## 1. What we are optimizing for (and why)

**Objective:** flag an email that *claims* to be from a profiled sender but was
not written by them — both **LLM-generated forgery** and **real-human impostors**
(BEC, wrong sender, compromised thread). We build a per-sender style centroid and
score how far a new email sits from it.

**North-star metric:**

> Maximize **TPR@1% FPR on forgeries** (catch rate at a 1% false-alarm budget),
> subject to the guardrail **FPR_other ≤ ~0.10** (wrong-human leak) at that
> threshold — measured against a **held-out, multi-generator** adversary and,
> increasingly, **off-domain** text.

**Why this exact framing (the lesson of this cycle):** a single full-range AUC on
a single generator hid two failures at once. (a) The deployment decision happens
in the low-FPR tail, not mid-ROC — so TPR@1% / pAUC@5%, not AUC, is the number.
(b) A detector tuned on one LLM (Mistral) does **not** generalize to others. If
the eval doesn't force *held-out generators* and *low-FPR operating points*, the
headline number lies. Everything below is built to keep the eval honest.

---

## 2. The three robustness axes (current state)

The model has to generalize along three independent axes. We now measure all three.

| Axis | Question | Harness | v11-lora | v12-lora | Status |
|---|---|---|---|---|---|
| **`gen:*`** | Detect forgeries from **unseen generators**? | `build_ood_eval.py` gen slices + `eval_ood.py` | mean pAUC@5% **0.041** | **0.142** | **Improved — primary win this cycle** |
| **`domain:*`** | Verify authorship **off-email** (PAN, blogs)? | `prepare_external_pairs.py` + `--extra-pairs` | PAN AUC 0.79 / blog 0.84 | *not yet re-run* | Wired; v12 retest pending |
| **`len/register`** | Robust to **length/register** shifts? | `build_ood_eval.py` len/register slices | `lenmix` ~0.64 | ~unchanged | **Length still unsolved** |

---

## 3. Key findings this cycle

### 3.1 The headline LLM-detection number was a single-generator mirage
v11-lora's reported TPR@1% of 0.875 was measured only against **Mistral-7B**, the
generator it trained on. On the *same probe/senders/scorer*, swapping in held-out
frontier generators collapses it:

| Adversary | AUC g/syn | TPR@1% FPR_syn |
|---|---|---|
| Mistral-7B (trained on) | ~0.96 | **0.91** |
| Held-out GPT-4o-mini / Claude / Llama / Gemini (pooled) | 0.75 | **0.045** |

Per-generator, the stronger/newer the model, the more it evades: Gemini-2.5-Flash
(AUC 0.614) ≈ Claude-3.5-Haiku (0.639) < GPT-4o-mini (0.709) < Llama-3.1-70B
(0.735). "Reject LLM text" was a Mistral fingerprint, not a general skill.

### 3.2 Multi-generator training closes most of the gap (v12)
v12 = the v11-lora recipe (episodic + short-mail + crop, LUAR+LoRA) retrained on a
**diverse** adversary (GPT-4o-mini + Llama-3.1-70B), with Claude + Gemini **held
out** for eval, and the **anti-Goodhart `min(other, synthetic)` monitor**.

On the held-out Claude+Gemini adversaries (never seen in training):

| Metric | v11-lora | **v12-lora** |
|---|---|---|
| Claude-3.5-Haiku AUC | 0.639 | **0.829** |
| Gemini-2.5-Flash AUC | 0.614 | **0.746** |
| `gen` axis mean pAUC@5% | 0.041 | **0.142** (3.4×) |
| held-out pool TPR@1% | 0.027 | **0.129** (≈5×) |
| held-out pool AUC | 0.696 | **0.866** |

`len`/`register` authorship slices are unchanged → no regression. **The thesis is
proven: diversity in the training adversary transfers to unseen generators.** The
150-epoch run confirmed the monitor-selected `checkpoint_best` (epoch 72) is the
model; training past it plateaus and slightly overfits to the train generators.

### 3.3 Authorship partially generalizes off-domain
On PAN cross-topic fanfiction and the Blog Authorship Corpus (real-human vs
real-human, no synthetics), v11-lora holds up on blogs (AUC 0.84) but cross-topic
collapses at the operating point (PAN pAUC@5% 0.22, TPR@1% 0.16) — the model leans
partly on topic/domain regularities, not pure style. *Caveat:* LUAR-MUD was
pretrained on Reddit, so PAN/blogs are the cleaner OOD tests (Reddit would
flatter).

### 3.4 Cross-length verification remains the clearest unsolved problem
`lenmix:short↔long` is near-chance (~0.50–0.64) for every model. Crop augmentation
bought same-length-short, not true length invariance.

---

## 4. Current best model & how to use it

- **Most robust model:** `runs/v12/lora/checkpoint_best.pt` (epoch 72). Use this
  when the threat model includes diverse/unknown LLM forgers (the realistic case).
- **Prior best (Mistral-tuned):** `runs/v11/lora/checkpoint_best.pt` — retained as
  the baseline. Higher on Mistral-only evals, but brittle off-generator; do not
  trust its synthetic numbers as a general guarantee.
- **Scorer:** `baseline_linear_z3`, K=8 (no alternative beats it with a CI
  excluding 0 on this probe).
- **Deployment threshold:** anchor on the synthetic pool to hold FPR_syn at target;
  always **report FPR_other** at that threshold (the wrong-human leak the synthetic
  axis hides — the v9@ep10 lesson).

---

## 5. Roadmap — prioritized, with rationale

> **UPDATE 2026-06-18 (v13 cycle — see `docs/v13_results.md`):** P1 was run by adding
> a 3rd vendor (DeepSeek). Result: **vendor count plateaued** — no significant change
> on held-out Claude+Gemini (Gemini 0.746→0.755, pool TPR@1% 0.129→0.155, all inside
> wide CIs). **Caveat:** this did *not* add volume — the train split has only 44 source
> senders, so 3 vendors at equal budget meant *fewer* rows/vendor (522→340). Revised
> priority below: test the **volume / source-sender** lever, NOT a 4th vendor. P2 and
> P3 also ran — P2 confirmed domain is a separate lever (unchanged); P3
> (length-matched enrollment) was **refuted** (it hurts).

### P1 — Harden multi-generator detection further *(highest leverage)*
**Why:** v12 closed most of the gap but **Gemini-2.5-Flash is still weak (AUC
0.746)** and the strongest closed models remain the realistic adversary. This is
the axis that most directly governs production trustworthiness.
**Do (REVISED post-v13 — vendor count has plateaued, do not add a 4th vendor):**
- Test the **volume / source-sender** lever: the Enron train split yields only ~44
  senders post-filter, capping every synthetic set at ~1045 rows regardless of
  `--n-per-sender`. Pull more Enron senders into the train split or add a second
  enrollment corpus, holding the vendor set fixed.
- **Precondition:** enlarge the held-out probe (currently 41–44 senders → ±0.12
  TPR@1% CIs swamp any v12↔v13 delta). Without this, further P1 claims aren't
  measurable. Generate via OpenRouter (key in `.env`).
- Keep a **rotating held-out vendor** so the generalization claim stays honest.
- Re-measure with the same `_v12_heldout_eval.yaml` path.

### P2 — Re-test the `domain:*` axis on v12 *(cheap, high information)*
**Why:** multi-generator training reshaped the embedding; it may have helped (or
hurt) off-domain authorship. We have the slices already; this is one eval command.
**Do:** `eval_ood.py --checkpoint runs/v12/lora/checkpoint_best.pt --extra-pairs
domain:pan20_xtopic=data/ood/pan20_xtopic_pairs.jsonl --extra-pairs
domain:blog=data/ood/blog_pairs.jsonl`. If cross-topic is still ~0.79, authorship
robustness is a *separate* lever from generator robustness and earns its own line
item.

### P3 — Attack length invariance *(the concrete unsolved failure)*
**Why:** `lenmix` near-chance is the clearest, most reproducible weakness, and
production email is length-heterogeneous. Cheapest first, scoring-side, no retrain:
**length-matched enrollment** (score short queries against the sender's short
enrollment subset) or **length-conditioned thresholds**. If that's insufficient,
add `cross_length` hard negatives (`--cross-length-fraction`) and retrain.

### P4 — Eval rigor *(de-risk the conclusions)*
**Why:** the held-out test split is only **6 senders** (noisy), and several of this
cycle's per-generator slices used *seen* train senders (inflated `len/register`).
**Do:** enlarge the unseen-sender probe (more held-out identities), and add a
small `held_out_generators` filter to `eval_v7_scoring._build_probe` +
`ProbeConfig` so the bootstrap-CI ablation can report TPR/FPR_other per held-out
generator (the `gen:*` OOD slices already do the slice-level version).

### Deferred (lower leverage, by design)
- **Pushing Mistral-only numbers higher** — Goodharts a narrow benchmark.
- **Two-model fusion (v10×v9)** — superseded for the single-model objective by the
  v11/v12 lora line; revisit only if a future adversary breaks the single-embedding
  tie.
- **`lora_supcon` 150-epoch retrain** — only tidies an ablation cell.

---

## 6. Reproduce / key artifacts

| Thing | Path |
|---|---|
| v12 model (best, ep72) | `runs/v12/lora/checkpoint_best.pt` |
| v12 train config | `configs/experiments/v12_multigen_lora.yaml` |
| Held-out eval config | `configs/experiments/_v12_heldout_eval.yaml` |
| Multi-gen train set (GPT+Llama) | `data/synthetic/enron_synthetic_v12_train` |
| Held-out set (Claude+Gemini) | `data/synthetic/enron_synthetic_v12_heldout` |
| Domain-OOD pairs | `data/ood/pan20_xtopic_pairs.jsonl`, `data/ood/blog_pairs.jsonl` |
| Results | `results/v12/*.json`, `docs/v12_multigen_results.md` |
| Orchestration scripts used | generate → `build_ood_eval.py` → `eval_ood.py` / `ablate_adaptive_scorers.py` |

LLM adversaries via OpenRouter: set `OPENROUTER_API_KEY` in `.env` (auto-loaded;
`generate_synthetic_emails.py` was patched to load it). Valid slugs:
`openai/gpt-4o-mini`, `anthropic/claude-3.5-haiku`,
`meta-llama/llama-3.1-70b-instruct`, `google/gemini-2.5-flash` (note: `gemini-flash-1.5`
does **not** exist on OpenRouter).

---

## 7. Caveats & gotchas (read before trusting a number)

- **`FPR_other ≈ 0` can be an artifact.** When the model can't separate a hard
  synthetic pool, the synthetic-anchored threshold is pushed so high it would
  reject genuine mail too — so the wrong-human leak *looks* zero. Read it together
  with TPR (low TPR + zero FPR_other = useless threshold, not a good model).
- **Seen vs unseen senders.** `len/register` OOD slices built on *train* senders
  (AUC ~0.95) are not comparable to the unseen-test-sender base eval (~0.64–0.81).
  Only compare like-with-like.
- **Disk quota.** The pod has a per-user MooseFS quota (~tens of GB), well below the
  756 T mount `df` reports. A 150-epoch run writes 336 MB checkpoints; keep
  `keep_last_n: 1` and clear scratch (`data/raw/*` extracts, smoke runs) before
  long runs. The v6–v9 `lineage`/`lineage_v2`/`v7_*` checkpoints were deleted this
  cycle for space (user-authorized); their `results/` analyses remain.
- **Monitor choice matters.** Use `pauc/min_other_synthetic_5pct` (anti-Goodhart),
  not `pauc/genuine_vs_synthetic_5pct` — the latter selects early checkpoints that
  ace synthetics while leaking wrong-human mail (the v9@ep10 / detector@ep11 trap).
