"""Reusable pairwise verification scoring — load a checkpoint, score (a, b) pairs.

The whole evaluation stack reduces to: embed two texts, take cosine similarity,
map to [0, 1] (higher = more likely same author / authentic). This module
factors that out so the OOD harness (scripts/eval_ood.py) and any other
evaluator share one implementation instead of re-deriving it.

It is deliberately dependency-light and slice-agnostic: callers pass a flat list
of (text_a, text_b) pairs and get back a numpy array of scores in the same
order. Grouping pairs into OOD slices and computing metrics per slice is the
caller's job (see compute_verification_metrics in scoring.metrics).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_encoder(checkpoint_path: str | Path, cfg, device: str):
    """Instantiate the encoder from cfg and load weights from a .pt checkpoint.

    cfg is a loaded Config (see config.load_config). The encoder class is
    resolved from the component registry by cfg.encoder.name, matching how
    training and the other eval scripts build it.
    """
    import torch

    from email_fraud.registry import resolve

    EncoderClass = resolve("encoder", cfg.encoder.name)
    encoder = EncoderClass(cfg.encoder)
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    encoder.load_state_dict(ckpt["model_state_dict"])
    encoder.eval()
    encoder.to(device)
    return encoder, ckpt


def embed_texts(
    encoder,
    texts: list[str],
    device: str,
    batch_size: int = 64,
    show_progress: bool = True,
) -> "np.ndarray":
    """Encode a list of texts → (N, d) float32 numpy array (on CPU)."""
    import torch

    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    out: list = []
    rng = range(0, len(texts), batch_size)
    if show_progress:
        from tqdm import tqdm

        rng = tqdm(rng, desc="Encoding", unit="batch",
                   total=(len(texts) + batch_size - 1) // batch_size)
    with torch.no_grad():
        for start in rng:
            batch = texts[start : start + batch_size]
            token_dict = encoder.tokenize(batch)
            token_dict = {k: v.to(device) for k, v in token_dict.items()}
            embs = encoder.encode(**token_dict)
            out.append(embs.cpu())
    return torch.cat(out, dim=0).float().numpy()


def score_pairs(
    encoder,
    pairs: list[tuple[str, str]],
    device: str,
    batch_size: int = 64,
    show_progress: bool = True,
) -> "np.ndarray":
    """Return per-pair similarity scores in [0, 1] (higher = same author).

    Deduplicates identical texts before encoding so a text appearing in many
    pairs (common when one genuine email anchors several fraud comparisons) is
    embedded once.
    """
    if not pairs:
        return np.empty(0, dtype=np.float64)

    # Dedup: map each unique text to a row index.
    uniq: dict[str, int] = {}
    flat: list[str] = []
    for a, b in pairs:
        for t in (a, b):
            if t not in uniq:
                uniq[t] = len(flat)
                flat.append(t)

    embs = embed_texts(encoder, flat, device, batch_size, show_progress)
    # L2-normalise so dot product == cosine similarity.
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / np.clip(norms, 1e-12, None)

    scores = np.empty(len(pairs), dtype=np.float64)
    for i, (a, b) in enumerate(pairs):
        sim = float(np.dot(embs[uniq[a]], embs[uniq[b]]))
        scores[i] = (sim + 1.0) / 2.0  # [-1, 1] → [0, 1]
    return scores
