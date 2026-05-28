# Email Fraud Detection — Repository Reference

> **Living document.** Update this file as new experiments are run, features are added, and results improve.

---

## The Product Idea

Given any company's email archive, the system:

1. **Profile phase** — ingests historical emails, builds a per-sender "style fingerprint" (a centroid vector in embedding space).
2. **Inference phase** — when a new email arrives claiming to be from a given sender, encodes it and compares it against that sender's profile. Returns a score in [0, 1] and a verdict: genuine, suspicious, or abstain (too little history to judge).

The target threat is **impersonation / BEC fraud**: an attacker sends an email that looks legitimate but was not actually written by the claimed sender. The system catches this because the attacker's writing style does not match the claimed sender's learned profile, even if the email address and content are convincing.

Critically, the system is **sender-agnostic at training time**. The model is trained on the Enron corpus (150+ employees), but at deployment it builds profiles for whoever is in the customer's company. It never needs to retrain — it just needs to enroll.

---

## The Two Phases in Detail

### Phase 1 — Enrollment (profile building)

```
Company email archive (historical, verified-genuine emails)
    ↓  preprocessing.py
    Cleaned email bodies (quotes/signatures stripped, entities masked)
    ↓  HFEncoder.encode()
    L2-normalized embedding vectors  (one per email)
    ↓  PrototypicalHead.fit()  or  SenderProfileStore.upsert()
    Per-sender profile: { centroid: ndarray, spread: float, k: int }
    ↓  head.save()  /  store.save()
    Profiles persisted to disk (pickle or JSON)
```

The encoder never updates during enrollment — only the profiles do. This means enrollment is fast (just a forward pass per email), and adding a new sender or new emails for an existing sender is O(1).

### Phase 2 — Scoring (runtime)

```
Incoming email  +  claimed sender ID
    ↓  ScoringPipeline.score()
    preprocess → tokenize → encode → head.score(embedding, sender_id)
    ↓
    ScoringResult:
        .score     float [0, 1]    higher = more consistent with claimed sender
        .tier      "low" / "medium" / "high" / "very_high"
        .abstain   True if k < 5  (not enough history to trust the score)
```

No model weights are updated at inference time. The entire runtime cost is a single encoder forward pass plus a centroid cosine similarity lookup.

---

## How the Repository Implements This

```
email_fraud_detection/
├── src/email_fraud/
│   ├── config.py            ← all hyperparameters, YAML-driven
│   ├── registry.py          ← plugin pattern (encoder / loss / head / dataset)
│   ├── encoders/            ← HFEncoder: RoBERTa, LUAR-MUD, MPNet, etc.
│   ├── losses/              ← SupConLoss, TripletLoss (contrastive training)
│   ├── heads/               ← PrototypicalHead (centroid scoring)
│   ├── data/                ← EnronDataset, PKSampler, preprocessing
│   ├── profiles/            ← SenderProfileStore (EWMA online updates)
│   ├── scoring/             ← ScoringPipeline (end-to-end inference)
│   └── training/            ← Trainer (training loop, checkpointing, wandb)
├── scripts/
│   ├── prepare_data.py      ← raw Enron maildir → Arrow dataset splits
│   ├── train.py             ← wires config → encoder + loss + head → Trainer
│   ├── evaluate.py          ← loads checkpoint → PAN metrics on test pairs
│   └── generate_synthetic_emails.py  ← LLM hard-negative generation
├── configs/
│   ├── base.yaml            ← global defaults
│   └── experiments/         ← 18 experiment-specific overrides
└── results/                 ← WandB CSV exports
```

### Component Map

| Use-case step | Repository component |
|---|---|
| Clean incoming email | `data/preprocessing.py` — strips reply chains, signatures, masks URLs/emails/phones |
| Encode text to a style vector | `encoders/hf_encoder.py` — `HFEncoder`, any HuggingFace AutoModel |
| Build per-sender profile | `heads/prototypical.py` — `PrototypicalHead.fit()` |
| Store & update profiles online | `profiles/store.py` — `SenderProfileStore`, EWMA centroid updates |
| Score an incoming email | `scoring/pipeline.py` — `ScoringPipeline.score()` |
| Train the encoder | `training/trainer.py` — contrastive loop with PKSampler |
| Choose the right backbone/loss | `configs/experiments/*.yaml` — swap without code changes |
| Benchmark the model | `scripts/evaluate.py` — PAN metrics (AUC, EER, c@1, F0.5u) |

### Training vs. Deployment Relationship

Training teaches the encoder to **arrange emails in embedding space by style, not by topic**. The model learns: two emails from Alice should be closer together than Alice's email and Bob's email, regardless of what they're talking about. This is what makes the centroid profiles meaningful.

Deployment then just:
1. Runs historical emails through the trained encoder to build centroids
2. Scores new emails against those centroids

The training data (Enron) is never seen again. The company's email senders are completely new to the model — the representations generalize because the model learned a universal notion of stylometric consistency.

---

## Experimental Results (as of May 2026)

All experiments run 20 epochs on the Enron corpus (sender-disjoint train/val/test). "v2" series adds a linear projection head after the backbone. Metrics at epoch 20.

### Key metric: `val/auroc`

AUROC measures "does same-sender pair score higher than different-sender pair?" — the core ranking ability. 0.5 = random, 1.0 = perfect.

| Experiment | val/auroc | val/loss | val/intra_cos_sim | val/inter_cos_sim |
|---|---|---|---|---|
| **v2_luar_lora** | **0.9556** | 0.8976 | 0.453 | 0.198 |
| **v2_luar_frozen_proj** | **0.9333** | 0.8711 | 0.456 | 0.142 |
| v2_roberta_frozen_proj | 0.5369 | 4.297 | 0.818 | 0.807 |
| v2_roberta_lora_proj | 0.5257 | 4.356 | 0.768 | 0.762 |
| v2_mpnet_frozen_proj | 0.5341 | 4.364 | 0.853 | 0.841 |
| v2_mpnet_lora_proj | 0.4853 | 4.221 | 0.840 | 0.839 |

### Key Finding

**LUAR-MUD is the clear winner.** LUAR (Language-Use Authorship Representation) was pretrained specifically for authorship tasks — its embeddings already encode stylometric structure before any fine-tuning. RoBERTa and MPNet, despite being stronger general language models, have AUROC near 0.5 (random) because their representations are dominated by semantic/topical content rather than style.

The high `intra_cos_sim` for RoBERTa/MPNet is a **red flag, not a positive signal** — it means same-sender emails are close together but so are different-sender emails (high `inter_cos_sim` too), so the model has learned nothing useful for discrimination.

LUAR's low cosine similarities (0.45 intra, 0.20 inter) reflect genuine spread in embedding space — the gap between intra and inter is real signal.

### LoRA vs. Frozen

For LUAR: LoRA fine-tuning (`v2_luar_lora`, AUROC 0.956) outperforms frozen backbone + projection (`v2_luar_frozen_proj`, AUROC 0.933). The 2-point gain suggests the backbone can be nudged to be more style-aware, but the frozen version is already strong and much cheaper to train.

For RoBERTa/MPNet: LoRA made no difference — the backbone representations are not the right inductive bias for stylometry and fine-tuning does not fix that.

---

## What Is and Isn't Done

### Working

- Full training pipeline: data prep → PKSampler → contrastive loss → checkpoint
- Two production-grade losses: SupConLoss and batch-hard TripletLoss
- HFEncoder with LoRA support, mean/CLS/LUAR pooling, optional projection
- PrototypicalHead: centroid fitting, cosine z-score scoring, confidence tiers, abstain flag
- SenderProfileStore: EWMA online updates, JSON persistence
- ScoringPipeline: end-to-end raw text → anomaly score
- Preprocessing: quote/sig stripping, entity masking, Unicode normalization (ftfy)
- Config system: Pydantic v2 + YAML deep-merge, typo-safe
- Registry: decorator-based plugin pattern, no if/elif chains
- wandb integration: run resumption, loss/AUROC/KNN metrics per epoch
- Synthetic hard-negative generation (LLM impersonation emails)
- 18 experiment configs covering major backbone/loss combinations

### Gaps (Stubs / TODOs)

| Gap | File | Impact |
|---|---|---|
| Mahalanobis scoring | `heads/prototypical.py`, `profiles/store.py` | Accounts for per-sender embedding shape, not just centroid distance — known to improve anomaly detection |
| Cross-encoder reranker | `heads/cross_encoder.py` | Slower but more accurate: compare query directly against each enrollment email |
| PAN metrics in Trainer validation | `training/trainer.py` | Currently only val/loss; full AUC/EER/c@1/F0.5u only in `evaluate.py` |
| Test-set evaluation on best checkpoint | `scripts/evaluate.py` | Best LUAR LoRA checkpoint not yet benchmarked on held-out test pairs |
| pgvector backend for profiles | `profiles/store.py` | Production scale: profiles currently in-memory dict, need persistent vector DB |

---

## Next Experiments (Priority Order)

1. **Run `evaluate.py` on best LUAR LoRA checkpoint** — get final PAN numbers (AUC, EER, c@1, F0.5u) on the held-out test split. These are the reportable numbers.

2. **Fix LoRA hyperparameters for RoBERTa/MPNet** — current failure is likely LR too high (`2e-4`) and too few epochs. Try `lr=5e-5`, 30-50 epochs, `r=8`. Might still underperform LUAR but worth confirming.

3. **Increase training senders** — currently 100. Enron has ~150+ usable senders. More senders = more diverse contrastive negatives = better generalization.

4. **Synthetic hard-negative training (V4 series)** — use LLM-generated impersonation emails as hard negatives in the contrastive batches. Directly trains the model to detect LLM mimicry — the actual threat case. Track `val/centroid/auroc_genuine_vs_synthetic` to measure effect.

5. **Mahalanobis scoring** — implement Ledoit-Wolf covariance in `PrototypicalHead`. Accounts for embedding cluster shape, not just centroid distance. Expected to improve precision at high-confidence operating points.

6. **Per-sender threshold calibration** — some senders write more consistently than others. A global threshold will underperform. Calibrate per-sender thresholds on held-out enrollment emails.

7. **Cross-domain evaluation** — test LUAR LoRA zero-shot on PAN 2020/2021 authorship verification benchmarks (fanfiction/news). If it transfers, strong evidence of genuine stylometric generalization.

8. **Ensemble with classical features** — character n-gram frequencies, function word distributions, punctuation patterns. These are orthogonal to neural representations and can improve robustness against adversarial paraphrasing.

---

## Running the System

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Credentials
cp .env.example .env
# set WANDB_API_KEY and HF_TOKEN

# 3. Prepare data (run once)
python scripts/prepare_data.py --enron-dir data/raw/enron --out-dir data/processed/enron

# 4. Train
python scripts/train.py --config configs/experiments/v2_luar_lora.yaml

# 5. Evaluate (PAN metrics on test set)
python scripts/evaluate.py \
    --config configs/experiments/v2_luar_lora.yaml \
    --checkpoint runs/v2_luar_lora/best.pt \
    --data-dir data/processed/enron

# 6. Score a single email at deployment
from email_fraud.scoring.pipeline import ScoringPipeline
result = pipeline.score(email_body, claimed_sender="cfo@company.com")
# result.score, result.tier, result.abstain
```

---

## Confidence Tier Reference

| Profile emails (k) | Tier | Behavior |
|---|---|---|
| 1–4 | `low` | `abstain=True` — not enough history |
| 5–9 | `medium` | Score with caution |
| 10–24 | `high` | Reliable |
| ≥ 25 | `very_high` | High confidence |

---

*Last updated: 2026-05-28 (V7)*

## V7 — Mahalanobis scoring + retrained encoder (2026-05-28)

See `experiments/v7/CHANGELOG_V7.md` for the full experimental log. Headline:

- **V7.0**: per-sender Ledoit-Wolf Mahalanobis distance, computed from the
  stored enrollment embeddings, replaces (or complements) the cosine
  z-score in `PrototypicalHead`. Wins by **+1-3 AUC pp on g/synthetic and
  +6-9 pp on TPR@5%FPR** across the K=8..25 range over the v6 encoder.
- **V7.2**: the win widens with enrollment size — at K=16 Mahalanobis adds
  **+3.1 AUC pp** and **+9.2 TPR@5%FPR pp** over cosine. Recommended
  production scoring is `adaptive_k` (cosine for k<5, Mahalanobis for k≥5)
  — implemented and tested in `src/email_fraud/heads/prototypical.py`.
- **V7.3**: retrained the encoder with `n_syn=4` (vs 2), `temperature=0.05`
  (vs 0.07), and an added LoRA target (key). Config:
  `configs/experiments/v7_luar_lora_syn_mahal.yaml`. Run with
  `python scripts/train.py --config configs/experiments/v7_luar_lora_syn_mahal.yaml`.
  Evaluate with `python scripts/eval_v7_full.py --v7-checkpoint <path>`.
