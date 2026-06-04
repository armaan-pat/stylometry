# Architecture — Email Fraud Detection

How the system works end-to-end, from raw emails to a fraud score. For a fast
orientation see the root [README](../README.md); for experimental numbers see
[results.md](results.md); for the W&B dashboard see [metrics.md](metrics.md).

---

## The product idea

Given any company's email archive, the system runs in two phases:

1. **Enrollment** — ingest historical, verified-genuine emails and build a
   per-sender "style fingerprint" (a centroid vector in embedding space).
2. **Scoring** — when a new email arrives claiming to be from a given sender,
   encode it and compare against that sender's profile. Return a score in
   `[0, 1]` and a verdict: genuine, suspicious, or abstain (too little history).

The target threat is **impersonation / BEC fraud**: an attacker sends an email
that looks legitimate but was not written by the claimed sender. The writing
style fails to match the learned profile even when the address and content are
convincing.

The system is **sender-agnostic at training time**. The encoder is trained once
on the Enron corpus (150+ employees); at deployment it builds profiles for
whoever is in the customer's company. It never retrains — it only enrolls. The
representations generalize because the model learned a universal notion of
stylometric consistency, not the specific Enron senders.

### Phase 1 — Enrollment

```
Company email archive (historical, verified-genuine emails)
    ↓  preprocessing.py        strip quotes/signatures, mask entities
    ↓  HFEncoder.encode()      L2-normalized embedding per email
    ↓  PrototypicalHead.fit()  or SenderProfileStore.upsert()
    Per-sender profile: { centroid, spread, k }
    ↓  head.save() / store.save()
    Profiles persisted to disk
```

The encoder never updates during enrollment — only the profiles do. Enrollment
is one forward pass per email, and adding a sender or new emails is O(1).

### Phase 2 — Scoring

```
Incoming email + claimed sender ID
    ↓  ScoringPipeline.score()
    preprocess → tokenize → encode → head.score(embedding, sender_id)
    ↓
    ScoringResult:
        .score    float [0,1]   higher = more consistent with claimed sender
        .tier     low / medium / high / very_high
        .abstain  True if k < 5  (not enough history)
```

No weights update at inference. Runtime cost is one encoder forward pass plus a
centroid lookup.

---

## Repository layout

```
email_fraud_detection/
├── src/email_fraud/
│   ├── config.py            all hyperparameters, YAML-driven (Pydantic v2)
│   ├── registry.py          plugin pattern (encoder / loss / head / dataset)
│   ├── encoders/            HFEncoder: RoBERTa, LUAR-MUD, MPNet, …
│   ├── losses/              SupConLoss, TripletLoss, ContrastiveLoss
│   ├── heads/               PrototypicalHead (centroid scoring)
│   ├── data/                EnronDataset, PKSampler, preprocessing, synthetic
│   ├── profiles/            SenderProfileStore (EWMA online updates)
│   ├── scoring/             ScoringPipeline, centroid probe, metrics, score fns
│   └── training/            Trainer (loop, checkpointing, W&B)
├── scripts/
│   ├── prepare_data.py      raw Enron maildir → Arrow dataset splits
│   ├── train.py             config → encoder + loss + head → Trainer
│   ├── evaluate.py          checkpoint → PAN metrics on test pairs
│   ├── score_centroids.py   deployment-style centroid scoring validation
│   ├── generate_synthetic_emails.py   LLM hard-negative generation
│   └── eval_v7_*.py, plot_*.py        experiment-specific analysis & plots
├── configs/
│   ├── base.yaml            global defaults
│   └── experiments/         per-experiment overrides
└── results/                 W&B exports, confusion matrices, sweep outputs
```

### Component map

| Use-case step | Component |
|---|---|
| Clean an incoming email | `data/preprocessing.py` — strip reply chains/signatures, mask URLs/emails/phones |
| Encode text to a style vector | `encoders/hf_encoder.py` — `HFEncoder`, any HuggingFace AutoModel |
| Build a per-sender profile | `heads/prototypical.py` — `PrototypicalHead.fit()` |
| Store & update profiles online | `profiles/store.py` — `SenderProfileStore`, EWMA centroid updates |
| Score an incoming email | `scoring/pipeline.py` — `ScoringPipeline.score()` |
| Train the encoder | `training/trainer.py` — contrastive loop with PKSampler |
| Swap backbone / loss / head | `configs/experiments/*.yaml` — no code changes |
| Benchmark the model | `scripts/evaluate.py` — PAN metrics (AUC, EER, c@1, F0.5u) |

The **registry** (`registry.py`) lets every class self-register with
`@register(kind, name)` and resolve by name at runtime from YAML — no if/elif
chains. The **config** (`config.py`) is Pydantic v2 with `extra="forbid"`, so a
typo in a YAML key fails at load time. `load_config(path)` deep-merges
`base.yaml` with the experiment file.

---

## Training internals

### Batch construction (P, K, n_syn)

Three numbers control each training batch:

- **P** — distinct senders per batch
- **K** — emails per sender per batch
- **batch size** = P × K

`PKSampler` lays out K consecutive rows per sender (sender 0's K emails, then
sender 1's, …). This layout is not cosmetic — the LUAR reshape depends on it.

`SyntheticBalancedSampler` adds **n_syn**: the number of (real sender, synthetic
sender) **pairs** per batch. Each pair takes 2 of the P slots — one real sender
and one LLM imitation of that same sender. The remaining `P - 2·n_syn` slots are
ordinary unpaired real senders.

```
P=16, K=4, n_syn=2  →  batch size 64
Slot 0  alice@enron.com        real  ┐ pair 1
Slot 1  alice@enron.com__syn   imit. ┘
Slot 2  bob@enron.com          real  ┐ pair 2
Slot 3  bob@enron.com__syn     imit. ┘
Slot 4… carol, dave, …         12 unpaired real senders
```

### LUAR episode pooling

Most encoders (RoBERTa, MPNet) take one email in and produce one embedding out;
their masked-LM pretraining makes those embeddings reflect topic.

LUAR was pretrained for authorship: it takes **K emails from one author** as an
"episode" and produces **one embedding** for the episode. Each episode must come
from a single sender.

The encoder receives the flat `(P·K, L)` batch and reshapes it to
`(P·K/episode_k, episode_k, L)` — a stack of single-sender episodes. Slicing
into chunks of `episode_k` always stays within one sender because PKSampler
placed K consecutive rows per sender. LUAR processes each episode independently
(no cross-episode attention) and outputs one L2-normalized style embedding each.

**Hard constraint:** `K % episode_k == 0` and `K / episode_k ≥ 2`. If
`K = episode_k` each sender yields only one embedding → no positives → SupCon is
undefined.

### Positives, negatives, and SupCon

After encoding, the batch holds `N = P·K/episode_k` embeddings (or `P·K` for
non-LUAR), each labeled by integer sender ID.

- **Positive pair** — two embeddings with the same label.
- **Negative pair** — two embeddings with different labels.

Crucially, with the synthetic sampler `alice@enron.com` and
`alice@enron.com__syn` get **different labels**, so they are negatives for each
other. The loss is forced to push real-Alice and LLM-Alice apart — the model
must find the stylistic features the LLM could not replicate.

`SupConLoss` (Khosla et al., NeurIPS 2020), for each anchor `i`:

```
L_i = -(1/|P(i)|) Σ_{p∈P(i)} log[ sim(i,p) / Σ_{j≠i} sim(i,j) ]
      where sim(i,j) = exp(cosine(i,j) / τ)
```

Temperature **τ** sharpens the softmax: lower τ gives hard negatives (close
embeddings from different senders) a much stronger gradient. Synthetic
imitations are the hardest negatives and receive the largest push — exactly the
pressure fraud detection needs. `TripletLoss` (batch-hard mining) is the
alternative; both require PKSampler.

### Training loop

```
For each epoch, each batch from the sampler:
  1. episode_collate() → EpisodeBatch(texts, labels)
  2. tokenize(texts)                → (P·K, max_length)
  3. encode()                       → (P·K/episode_k, d) L2-normalized  [LUAR]
  4. stride labels: labels[::episode_k]
  5. SupConLoss(embeddings, labels) → scalar
  6. backward()                     → grads into LoRA adapters (backbone frozen)
  7. clip at 1.0; optimizer.step(); scheduler.step()  (cosine LR, per batch)
```

The trainer runs mixed-precision AMP, gradient clipping, warmup +
cosine/linear/constant schedules, W&B logging, and checkpoint save/resume
(`checkpoint_epoch_XXX.pt`, `checkpoint_last.pt`, `checkpoint_best.pt`).

---

## Inference

### Enrollment (once per sender, offline)

```
For each sender:
  collect known-genuine historical emails
  → preprocess each (strip quotes/sigs, mask entities)
  → encode each through the trained encoder
  → centroid = mean(embeddings)
  → spread   = mean(1 - cosine(email, centroid))
  → store { centroid, spread, k }
```

`spread` measures how consistently a sender writes: always-the-same → low
spread; context-dependent → high spread.

### Scoring (per incoming email)

```
1. preprocess(text)
2. tokenize → encode → (1, d) L2-normalized
3. load profile { centroid, spread, k } for the claimed sender
4. cosine_distance = 1 - cosine(query, centroid)
5. z = cosine_distance / spread
6. score = max(0, 1 - z/3)        z=0 → 1.0, z=1 → 0.67, z≥3 → 0.0
7. tier from k (see below)
```

A genuine email lands near the centroid; an impersonation (human or LLM) lands
farther away because the style does not match the learned pattern. From V7 the
default scorer is `adaptive_k`: cosine for `k < 5`, per-sender Ledoit-Wolf
Mahalanobis for `k ≥ 5` (see [results.md](results.md)).

---

## Why LUAR outperforms RoBERTa

|  | RoBERTa / MPNet | LUAR |
|---|---|---|
| Pretraining task | predict masked tokens | distinguish authors |
| Embeddings encode | semantic topic | stylometric fingerprint |
| Two emails, same topic | close together | depends on author, not topic |
| Fine-tuning with SupCon | wrong inductive bias | builds on stylometric reps |
| `val/auroc` after 20 epochs | ~0.53 (near random) | ~0.95 |

High `intra_cos_sim` for RoBERTa/MPNet is a red flag, not a signal: same-sender
emails are close, but so are different-sender emails (`inter_cos_sim` also high),
so nothing useful was learned for discrimination.

---

## Configuration quick reference

| Parameter | Controls |
|---|---|
| `P` | senders per batch — more = more diverse negatives per step |
| `K` | emails per sender per batch — must be ≥ 2·episode_k |
| `episode_k` | emails LUAR pools into one style embedding |
| `n_syn` | (real, synthetic) pairs per batch — each uses 2 of P slots |
| `τ` | SupCon sharpness; 0.1 standard, lower = harder negatives weighted more |

**Constraints:** `K % episode_k == 0`, `K / episode_k ≥ 2`, `2·n_syn ≤ P`.

### Confidence tiers

| Profile emails (k) | Tier | Behavior |
|---|---|---|
| 1–4 | `low` | `abstain=True` — not enough history |
| 5–9 | `medium` | score with caution |
| 10–24 | `high` | reliable |
| ≥ 25 | `very_high` | high confidence |
