# scoring/

End-to-end inference pipeline and PAN evaluation metrics.

---

## Files

### `pipeline.py` — `ScoringPipeline` and `ScoringResult`

Connects encoder → head → anomaly score for a claimed sender.

#### `ScoringResult`

```python
@dataclass
class ScoringResult:
    sender_id: str          # claimed sender
    score:     float        # [0, 1] — higher = more consistent with profile
    tier:      str          # "low" / "medium" / "high" / "very_high"
    abstain:   bool         # True when profile is too sparse to trust
    embedding: Tensor       # (d,) query embedding, CPU, detached
    raw:       dict         # full dict from head.score() for logging/debugging
```

#### `ScoringPipeline`

```python
pipeline = ScoringPipeline(
    encoder,          # trained BaseEncoder (put in eval mode before passing)
    head,             # fitted BaseHead with profiles loaded
    store,            # SenderProfileStore (for tier lookup and online updates)
    preprocessing,    # PreprocessingConfig — controls text cleaning
    device,           # "cpu" or "cuda"
    update_on_score,  # if True, upsert each scored email into the store
)
```

#### Single-email scoring

```python
result = pipeline.score(email_text, claimed_sender)
```

Internally:
1. `preprocess(email_text, config)` — clean the email body
2. `encoder.tokenize([cleaned])` — tokenize
3. `encoder.encode(**tokens)` → `(1, d)` → squeeze to `(d,)`
4. `head.score(query, claimed_sender)` → `{score, tier, abstain}`
5. Optionally `store.upsert(claimed_sender, query.numpy())`
6. Return `ScoringResult`

#### Batch scoring

```python
results = pipeline.score_batch(email_texts, claimed_senders)
```

All texts are encoded in a single forward pass for efficiency. Each email is then scored independently against its claimed sender.

#### `from_config` classmethod

Convenience constructor that reads `preprocessing` from an `ExperimentConfig`:

```python
pipeline = ScoringPipeline.from_config(config, encoder, head, store, device)
```

#### Online update mode

When `update_on_score=True`, each scored query is upserted into the store after scoring. This allows sender profiles to adapt to new emails over time. Disable for forensic/audit use cases where you want a frozen reference profile.

---

### `metrics.py`

PAN Author Verification evaluation metrics. All metrics take:
- `labels: np.ndarray` — binary (1 = authentic, 0 = fraud)
- `scores: np.ndarray` — continuous in `[0, 1]` (higher = more authentic)

| Function | Returns | Description |
|----------|---------|-------------|
| `compute_auc(labels, scores)` | float | ROC-AUC — threshold-independent ranking quality |
| `compute_eer(labels, scores)` | float | Equal Error Rate — where FAR == FRR |
| `compute_c_at_1(labels, scores, threshold)` | float | PAN 2011 — rewards abstaining over guessing wrong |
| `compute_f05u(labels, scores, threshold)` | float | PAN 2019 — precision-weighted F with abstain penalty |
| `compute_verification_metrics(labels, scores)` | dict | All four metrics in one call |

#### EER

The threshold where False Accept Rate (FAR) = False Reject Rate (FRR). A perfect system has EER=0; random has EER=0.5.

#### c@1 (Peñas & Rodrigo, 2011)

Treats scores exactly at `threshold` as "unanswered". Unanswered questions receive partial credit equal to the accuracy on answered questions. Encourages conservative systems that say "I don't know" rather than guessing.

#### F0.5u (Bevendorff et al., 2019)

F-beta with β=0.5 (precision > recall), where unanswered items are counted as false negatives. Penalizes systems that abstain too much.

---

## Full inference workflow

```
scripts/evaluate.py
       │
       ├─ load_config(config_path)
       ├─ load checkpoint_best.pt → encoder weights
       ├─ EnronDataset(config, split="test")
       │
       ├─ Build profiles from training data:
       │     for batch in profile_loader:
       │         embeddings = encoder.encode(...)
       │         head.fit(embeddings, sender_ids)
       │
       └─ Score test set:
             for (email, true_sender) in test_set:
                 result = pipeline.score(email, claimed_sender)
                 # compare result.score against ground truth label
             compute_verification_metrics(labels, scores)
```
