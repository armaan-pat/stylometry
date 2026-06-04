# heads/

Anomaly-scoring heads that compare a query email's embedding against a stored per-sender profile and return a calibrated fraud score.

---

## Role in the pipeline

After training, the encoder is fixed. The head is responsible for:

1. **Profile building** (`fit`): ingest labeled embeddings from a reference corpus and build a compact per-sender profile (e.g. a centroid).
2. **Scoring** (`score`): given a query embedding and a claimed sender id, return a score in `[0, 1]` where higher = more consistent with the claimed sender.
3. **Persistence** (`save`/`load`): serialize profiles to disk so they can be updated incrementally without retraining.

---

## Files

### `base.py` — `BaseHead`

Abstract base class. All heads must implement:

| Method | Signature | Purpose |
|--------|-----------|---------|
| `fit` | `(embeddings: Tensor, sender_ids: list[str]) → None` | Build/update per-sender profiles |
| `score` | `(query: Tensor, sender_id: str) → dict` | Return `{score, tier, abstain}` |
| `save` | `(path: str) → None` | Persist profiles to disk |
| `load` | `(path: str) → None` | Restore profiles from disk |

The `score` return dict always has:
- `"score"`: float in `[0, 1]` — higher = more authentic
- `"tier"`: str — confidence tier based on how many emails the profile was built from
- `"abstain"`: bool — True when the profile is too sparse to trust

---

### `prototypical.py` — `PrototypicalHead`

**Centroid-based anomaly head.** The primary head, registered as `"prototypical"`.

Inspired by Snell et al. Prototypical Networks (NeurIPS 2017), adapted from N-way classification to open-set sender verification.

#### Per-sender profile

```python
{
    "centroid": Tensor(d,),   # mean embedding of all seen emails
    "spread":   float,        # mean cosine distance from emails to centroid
    "k":        int,          # number of emails incorporated
}
```

#### fit() — online centroid update

For new senders: centroid = batch mean.  
For known senders: running average (equal-weight):

```
centroid_new = (centroid_old * k_old + sum(batch_embs)) / k_new
```

Spread is recomputed from the incoming batch against the updated centroid.

#### score() — z-score anomaly detection

```
cos_dist = 1 - cosine_similarity(query, centroid)
z        = cos_dist / spread
score    = max(0, 1 - z/3)      # z=0 → 1.0, z=3 → 0.0, z>3 → 0.0
```

A score near `1.0` means the query is as close to the centroid as a typical email from that sender. A score near `0.0` means the query is more than 3 standard deviations away — a strong anomaly signal.

#### Confidence tiers

| k (emails seen) | Tier | Abstain? |
|-----------------|------|----------|
| 1–4 | low | Yes |
| 5–9 | medium | No |
| 10–24 | high | No |
| 25+ | very_high | No |

When `abstain=True`, the scoring pipeline should not make a hard fraud decision.

#### Serialization

Profiles are saved as a pickle file (centroid stored as numpy array for portability).

#### `mahalanobis_score()` — TODO

A stub for future Mahalanobis distance scoring with Ledoit-Wolf covariance shrinkage. More statistically rigorous than z-score but requires storing all per-sender embeddings (not just the centroid).

---

## Configuration reference

```yaml
head:
  name: prototypical    # prototypical (only implemented head)
  distance: cosine      # cosine (only implemented option)
  shrinkage: ledoit_wolf  # for future Mahalanobis support
```
