# losses/

Contrastive loss functions that train the encoder to pull same-sender emails together and push different-sender emails apart in the embedding space.

---

## How contrastive training works

For a batch of N emails (each a `(d,)` L2-normalized embedding), the loss function:
1. Uses sender labels to identify **positive pairs** (same sender) and **negative pairs** (different senders)
2. Computes a scalar loss that is small when positives are close and negatives are far apart
3. Gradients flow back through the encoder, adjusting weights to improve clustering

All losses operate on the unit hypersphere (L2-normalized embeddings), so distances are bounded: cosine similarity ∈ [-1, 1], L2 distance ∈ [0, 2].

---

## Files

### `base.py` — `BaseLoss`

Abstract base class. All losses must:
- Implement `forward(embeddings, labels) → scalar Tensor`
- Declare `requires_pk_sampler` (True for all three current losses)

`requires_pk_sampler = True` tells the training script to use `PKSampler` so every batch contains `P` senders × `K` emails each — guaranteeing at least one positive pair per anchor.

---

### `supcon.py` — `SupConLoss`

**Supervised Contrastive Loss** (Khosla et al., NeurIPS 2020, arXiv:2004.11362).

Registered as `"supcon"`. **Recommended default.**

#### Formula (L_sup_out variant)

For each anchor `i`:

```
L_i = -(1/|P(i)|) * Σ_{p ∈ P(i)} log [ exp(z_i · z_p / τ) / Σ_{a ≠ i} exp(z_i · z_a / τ) ]
```

where `P(i)` = in-batch emails sharing `i`'s sender label, `τ` = temperature.

#### Intuition
- Maximizes the probability of drawing a positive when sampling uniformly from the rest of the batch
- Temperature `τ` controls the sharpness: lower → harder negatives dominate more

#### Key parameter
| Parameter | Default | Effect |
|-----------|---------|--------|
| `temperature` | 0.1 | Lower = sharper contrastive distribution |

---

### `triplet.py` — `TripletLoss`

**Triplet loss with batch-hard mining** (Hermans et al., arXiv:1703.07737).

Registered as `"triplet"`.

#### Formula

For each anchor `i`:
```
L_i = relu(d(i, hardest_pos) - d(i, hardest_neg) + margin)
```

where `d` is squared Euclidean distance (= `2 - 2·cos` for unit vectors).

#### Mining strategies

| Strategy | Description |
|----------|-------------|
| `"batch_hard"` | Hardest positive (max distance) + hardest negative (min distance) per anchor |
| `"all"` | All valid (anchor, positive, negative) triplets — O(N³), use for small batches |

#### Key parameters
| Parameter | Default | Effect |
|-----------|---------|--------|
| `margin` | 0.3 | Minimum required gap between positive and negative distances |
| `mining` | `"batch_hard"` | Triplet selection strategy |

---

### `contrastive.py` — `ContrastiveLoss`

**Pairwise contrastive loss** (Hadsell et al., CVPR 2006). The original contrastive loss.

Registered as `"contrastive"`.

#### Formula

```
L = (1/|pairs|) * Σ_{i<j} [ y·d² + (1-y)·relu(m - d)² ]
```

where `y=1` for same-sender pairs, `y=0` for different-sender pairs, `d` = L2 distance.

#### Mining strategies

| Strategy | Description |
|----------|-------------|
| `"all"` | All positive and negative pairs in the batch |
| `"semi_hard"` | Negatives farther than the hardest positive but inside the margin |

Semi-hard mining (Schroff et al. FaceNet 2015) avoids trivially easy negatives that are already well-separated.

#### Key parameters
| Parameter | Default | Effect |
|-----------|---------|--------|
| `margin` | 1.0 | Distance at which negatives stop contributing to loss |
| `mining` | `"all"` | Pair selection strategy |

---

### `episodic.py` — `EpisodicPrototypeLoss`

**Episodic, variable-K prototypical loss** (Snell et al., NeurIPS 2017, arXiv:1703.05175 — adapted; see `docs/robustness_mechanisms.md` §A1).

Registered as `"episodic"`.

Trains the way the system infers: per batch, each sender's embeddings are split into a *support set* of size K′ ~ uniform{`support_k_min`..`support_k_max`} and a *query set*. Prototypes are (renormalized) support means; each query is classified against all in-batch prototypes with the deployed cosine distance via cross-entropy. Sampling K′ small explicitly optimizes the encoder so that the mean of a few embeddings is already a stable description of the sender — low-K robustness in the representation.

Synthetic hard negatives (`sender_id` ending in `__syn`) never form prototypes; they stay in the query pool and are repelled from their mimicked sender's prototype via `-log(1 − p(mimicked))`. This requires the raw sender-id strings, so the loss sets `requires_sender_ids = True` and the Trainer passes `sender_ids=` alongside the labels.

SupCon is kept as an auxiliary term: `L = L_proto + supcon_weight · L_supcon`.

#### Key parameters
| Parameter | Default | Effect |
|-----------|---------|--------|
| `temperature` | 0.05 | Softmax sharpness over query→prototype cosines (shared with the SupCon aux term) |
| `support_k_min` / `support_k_max` | 2 / 6 | Range of the per-sender support size K′ (capped so ≥1 query remains) |
| `supcon_weight` | 0.5 | Weight of the auxiliary SupCon term (0 disables) |

---

## Comparison

| Loss | Objective | Complexity | Best for |
|------|-----------|------------|----------|
| SupCon | Maximize similarity to all positives jointly | O(N²) | Large K, rich positive signal |
| Triplet (batch-hard) | Separate hardest positive from hardest negative | O(N²) | When convergence is slow with SupCon |
| Contrastive | Push all pairs together / apart by margin | O(N²) | Simpler baseline; pairs are cheaper than triplets |
| Episodic | Make small-sample centroids discriminative (the deployed objective) | O(N²) | Low-K enrollment robustness; matches inference |

---

## Configuration reference

```yaml
loss:
  name: supcon        # supcon | triplet | contrastive | episodic
  temperature: 0.1    # SupConLoss / EpisodicPrototypeLoss
  margin: 0.3         # TripletLoss / ContrastiveLoss
  mining: batch_hard  # batch_hard | all (triplet) / all | semi_hard (contrastive)
  support_k_min: 2    # EpisodicPrototypeLoss only
  support_k_max: 6    # EpisodicPrototypeLoss only
  supcon_weight: 0.5  # EpisodicPrototypeLoss only
```
