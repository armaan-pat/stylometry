"""Offline hard negative mining for PKSampler and SyntheticBalancedSampler.

Encodes all training emails, computes per-sender centroids, and returns the
top-n most confusable (different-sender) pairs ranked by centroid cosine similarity.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def mine_hard_pairs(
    model,
    texts: list[str],
    sender_ids: list[str],
    device: str,
    n_pairs: int = 20,
    batch_size: int = 64,
) -> list[tuple[str, str]]:
    """Return the top-n most confusable sender pairs by centroid similarity.

    Only real senders should be passed (strip __syn entries before calling).
    """
    was_training = model.training
    model.eval()

    saved_k: int | None = None
    if hasattr(model, "config") and hasattr(model.config, "episode_k"):
        saved_k = model.config.episode_k
        model.config.episode_k = 1

    all_embs: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            tok = model.tokenize(batch)
            tok = {k: v.to(device) for k, v in tok.items()}
            all_embs.append(model.encode(**tok).cpu())

    if saved_k is not None:
        model.config.episode_k = saved_k
    if was_training:
        model.train()

    embs = torch.cat(all_embs, dim=0)

    unique_senders = sorted(set(sender_ids))
    s2i = {s: i for i, s in enumerate(unique_senders)}
    n_senders = len(unique_senders)
    dim = embs.shape[1]

    centroids = torch.zeros(n_senders, dim)
    counts = torch.zeros(n_senders)
    for emb, sid in zip(embs, sender_ids):
        i = s2i[sid]
        centroids[i] += emb
        counts[i] += 1
    centroids /= counts.unsqueeze(1).clamp(min=1e-9)

    c_norm = F.normalize(centroids, dim=1)
    sim = c_norm @ c_norm.T
    sim.fill_diagonal_(-2.0)

    flat = sim.reshape(-1)
    k = min(n_pairs * 4, n_senders * (n_senders - 1))
    topk_vals, topk_idx = flat.topk(k)

    pairs: list[tuple[str, str]] = []
    seen: set[frozenset] = set()
    for idx, val in zip(topk_idx.tolist(), topk_vals.tolist()):
        r, c = divmod(idx, n_senders)
        key: frozenset = frozenset([r, c])
        if r == c or key in seen:
            continue
        seen.add(key)
        pairs.append((unique_senders[r], unique_senders[c]))
        if len(pairs) >= n_pairs:
            break

    if pairs:
        top_sim = sim[s2i[pairs[0][0]], s2i[pairs[0][1]]].item()
        logger.info(
            "Mined %d hard pairs; top centroid sim=%.3f (%s vs %s)",
            len(pairs), top_sim, pairs[0][0], pairs[0][1],
        )
    return pairs
