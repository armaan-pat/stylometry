# Email Fraud Detection — Sender Fingerprinting via Contrastive Learning

Builds per-sender behavioral profiles via a contrastive encoder and flags emails
that deviate from a sender's established style. Config-driven via YAML — swapping
encoders, losses, and heads requires no source changes.

## Architecture

```
raw email → Preprocessing → Encoder (HFEncoder) → (B, d) embeddings
                                   │
                          Training ┤ Inference
                          Loss     │ ScoringPipeline → SenderProfileStore
                          (SupCon) │   head.score()     (EWMA centroid)
                          Head.fit()
```

## Components

**Registry** (`src/email_fraud/registry.py`) — every class self-registers with
`@register(kind, name)` and is resolved by name at runtime from YAML.
Kinds: `"encoder"`, `"loss"`, `"head"`, `"dataset"`.

**Config** (`src/email_fraud/config.py`) — Pydantic v2 with `extra="forbid"`;
typos in YAML are caught at load time. `load_config(path)` merges `base.yaml`
with the experiment file.

**Encoders** (`src/email_fraud/encoders/`) — `HFEncoder` wraps any AutoModel.
Pooling: `"mean"` (default), `"cls"`, `"luar_episode"` (Rivera et al. EMNLP 2021).
Add a new encoder with `@register("encoder", "key")` and import it in `encoders/__init__.py`.

**Losses** (`src/email_fraud/losses/`) — `SupConLoss` (Khosla et al. NeurIPS 2020),
`TripletLoss` (Hermans et al.). Both require PKSampler.

**Heads** (`src/email_fraud/heads/`) — `PrototypicalHead`: centroid + cosine z-score,
online `fit()`. `CrossEncoderHead`: reranker stub, not yet implemented.

**Data** (`src/email_fraud/data/`) — `PKSampler` produces P×K batches;
`SyntheticBalancedSampler` guarantees n_syn synthetic–real pairs per batch.

**Profile store** (`src/email_fraud/profiles/store.py`) — in-memory EWMA centroid store.
`store.upsert(sid, embedding)`, `store.get_profile(sid)`, `store.save/load`.

**Scoring pipeline** (`src/email_fraud/scoring/pipeline.py`) — wires encoder → head → score.
`pipeline.score(email_text, claimed_sender)` → `ScoringResult(score, tier, abstain, ...)`.

## Quick start

```bash
pip install -e ".[dev]"
cp .env.example .env          # add WANDB_API_KEY, HF_TOKEN

python scripts/train.py --config configs/experiments/example_luar.yaml

python scripts/evaluate.py \
    --config configs/experiments/example_luar.yaml \
    --checkpoint runs/<run>/checkpoint_best.pt \
    --data-dir data/processed/pan
```

To add a new experiment, copy an existing YAML under `configs/experiments/` and
override only the fields that differ from `base.yaml`.

## Confidence tiers

| k (emails seen) | Tier | Notes |
|-----------------|------|-------|
| 1–4 | `low` | `abstain=True` |
| 5–9 | `medium` | |
| 10–24 | `high` | |
| 25+ | `very_high` | |

## TODOs

| File | What's missing |
|------|----------------|
| `encoders/hf_encoder.py` | luar_episode path needs testing with a real LUAR checkpoint |
| `heads/prototypical.py` | Ledoit-Wolf Mahalanobis scoring |
| `heads/cross_encoder.py` | Full reranker |
| `profiles/store.py` | Mahalanobis with pgvector backend |
