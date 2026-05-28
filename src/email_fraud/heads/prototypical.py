"""Prototypical head: centroid-based per-sender profiling.

Reference: Snell, Swersky, Zemel "Prototypical Networks for Few-Shot Learning"
           NeurIPS 2017, arXiv:1703.05175 (centroid idea);
           adapted here for open-set sender verification rather than N-way
           classification.

How scoring works
-----------------
Each sender profile stores:
  - centroid : the mean embedding of all seen emails (updated online).
  - spread   : mean cosine distance of emails from the centroid — measures
               how consistent the sender's writing style is.
  - k        : how many emails have been incorporated.

At query time:
  1. Compute cosine distance from query to centroid.
  2. Express as a z-score: deviation / spread.
  3. Map z-score to [0, 1]: score = max(0, 1 - z/3).
     z < 0 → more similar than average → score near 1.
     z = 3 → 3 standard deviations away → score ≈ 0.
     This is a simple linear normalization, not a calibrated probability.

Confidence tiers
----------------
Tiers are based on k, the number of emails in the profile.  A profile built
from 1–4 emails is "low" confidence; we mark these as abstain=True to prevent
premature fraud flags.  As k grows, the centroid and spread estimates stabilize.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from email_fraud.heads.base import BaseHead

# @register("head", "prototypical") — Snell et al. NeurIPS 2017 (arXiv:1703.05175)
from email_fraud.registry import register


@register("head", "prototypical")
class PrototypicalHead(BaseHead):
    """Centroid-based anomaly head with cosine + z-score deviation scoring.

    Per-sender profile structure::

        {
            sender_id: {
                "centroid": Tensor (d,),
                "spread":   float,   # mean cosine distance to centroid
                "k":        int,     # number of emails seen
            }
        }

    Confidence tiers are determined by k and looked up from the config's
    confidence_tiers dict (e.g. {"1-4": "low", "5-9": "medium", ...}).
    """

    def __init__(
        self,
        confidence_tiers: dict[str, str] | None = None,
        distance: str = "cosine",
        score_fn: str = "linear_z3",
    ) -> None:
        super().__init__()
        self.confidence_tiers = confidence_tiers or {
            "1-4": "low",
            "5-9": "medium",
            "10-24": "high",
            "25+": "very_high",
        }
        self.distance = distance
        # Resolve once at construction so .score() is a tight inner loop. The
        # registry validates the name (KeyError if unknown).
        from email_fraud.scoring.score_functions import resolve as _resolve_score_fn
        self.score_fn_name = score_fn
        self._score_fn = _resolve_score_fn(score_fn)
        # In-memory dict of profiles; keyed by sender_id string.
        self._profiles: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # BaseHead interface
    # ------------------------------------------------------------------

    def fit(
        self,
        embeddings: torch.Tensor,
        sender_ids: list[str],
    ) -> None:
        """Update per-sender centroid + spread from a batch of embeddings.

        For a new sender: initialise centroid as the batch mean.
        For a known sender: online update using a running average.

        The online average formula ensures the centroid is the true mean of
        all emails seen so far, regardless of batch size:
            centroid_new = (centroid_old * k_old + sum(batch_embs)) / k_new
        """
        # Detach and move to CPU — profiles are CPU tensors; embedding
        # computation happens on GPU but profile arithmetic is lightweight.
        embeddings = embeddings.detach().cpu()
        # dict.fromkeys preserves first-appearance order while deduplicating.
        unique_senders = list(dict.fromkeys(sender_ids))

        for sid in unique_senders:
            # Gather indices of this sender's emails in the current batch.
            idx = [i for i, s in enumerate(sender_ids) if s == sid]
            embs = embeddings[idx]  # (k, d)

            if sid not in self._profiles:
                # First time seeing this sender: initialise from batch mean.
                centroid = embs.mean(dim=0)
                # Spread = mean cosine distance from individual emails to centroid.
                # cosine_similarity returns values in [-1, 1]; 1 - sim gives distance in [0, 2].
                sims = F.cosine_similarity(embs, centroid.unsqueeze(0))
                spread = float((1.0 - sims).mean())
                self._profiles[sid] = {
                    "centroid": centroid,
                    "spread": spread,
                    "k": len(idx),
                }
            else:
                # Incremental update: merge new batch with existing profile.
                prof = self._profiles[sid]
                old_k = prof["k"]
                new_k = old_k + len(idx)
                # Weighted average: old centroid represents old_k emails;
                # new batch contributes len(idx) emails.
                prof["centroid"] = (
                    prof["centroid"] * old_k + embs.sum(dim=0)
                ) / new_k
                # Recompute spread only from the incoming batch against the updated
                # centroid (not a full recomputation — approximation is acceptable).
                all_embs = embs
                sims = F.cosine_similarity(
                    all_embs, prof["centroid"].unsqueeze(0)
                )
                prof["spread"] = float((1.0 - sims).mean())
                prof["k"] = new_k

    def score_raw(
        self,
        query: torch.Tensor,
        sender_id: str,
    ) -> dict[str, object]:
        """Return the underlying geometry: cos_sim, spread, tier.

        Lets callers swap score functions without touching the head — useful
        for the centroid probe, which evaluates multiple scoring variants per
        run, and for offline replay in analyze_thresholds.py.
        """
        query = query.detach().cpu().squeeze()  # ensure (d,) shape

        if sender_id not in self._profiles:
            return {
                "cos_sim": float("nan"),
                "spread": float("nan"),
                "tier": "unknown",
                "abstain": True,
            }

        prof = self._profiles[sender_id]
        centroid: torch.Tensor = prof["centroid"]
        spread: float = prof["spread"]
        k: int = prof["k"]

        cos_sim = float(F.cosine_similarity(query.unsqueeze(0), centroid.unsqueeze(0)))
        tier = self._k_to_tier(k)
        return {
            "cos_sim": cos_sim,
            "spread": spread,
            "tier": tier,
            "abstain": tier == "low",
        }

    def score(
        self,
        query: torch.Tensor,
        sender_id: str,
    ) -> dict[str, object]:
        """Return score under the configured score_fn, tier, and abstain flag.

        See email_fraud.scoring.score_functions for the variants. The default
        (linear_z3) preserves the historical "1.0 = typical, 0.0 = z≥3" mapping.
        """
        raw = self.score_raw(query, sender_id)
        if raw["tier"] == "unknown":
            return {"score": 0.0, "tier": "unknown", "abstain": True}
        normalized_score = self._score_fn(float(raw["cos_sim"]), float(raw["spread"]))
        return {
            "score": normalized_score,
            "tier": raw["tier"],
            "abstain": raw["abstain"],
        }

    def save(self, path: str) -> None:
        # Convert tensors to numpy for pickle portability across PyTorch versions.
        payload = {
            sid: {
                "centroid": prof["centroid"].numpy(),
                "spread": prof["spread"],
                "k": prof["k"],
            }
            for sid, prof in self._profiles.items()
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)

    def load(self, path: str) -> None:
        import numpy as np

        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        # Restore numpy arrays back to PyTorch tensors.
        self._profiles = {
            sid: {
                "centroid": torch.from_numpy(np.array(data["centroid"])),
                "spread": data["spread"],
                "k": data["k"],
            }
            for sid, data in payload.items()
        }

    # ------------------------------------------------------------------
    # Mahalanobis stub
    # ------------------------------------------------------------------

    def mahalanobis_score(
        self,
        query: torch.Tensor,
        sender_id: str,
    ) -> float:
        # TODO: add tied covariance + Ledoit-Wolf shrinkage estimation.
        # Full implementation should:
        #   1. Collect all stored embeddings for sender_id.
        #   2. Estimate covariance with sklearn LedoitWolf (or OAS).
        #      Ledoit-Wolf is preferred over raw MLE covariance because it
        #      regularizes the estimate for high-dimensional embeddings (d >> k).
        #   3. Compute Mahalanobis distance to centroid.
        #   4. Return distance (lower = more in-distribution).
        raise NotImplementedError(
            "Mahalanobis scoring not yet implemented. "
            "See TODO comment in mahalanobis_score()."
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _k_to_tier(self, k: int) -> str:
        """Map email count k to a confidence tier label.

        Iterates over the confidence_tiers dict (e.g. {"1-4": "low", "25+": "very_high"}).
        Range strings are either "lo-hi" or "lo+" (unbounded upper end).
        """
        for range_str, label in self.confidence_tiers.items():
            if range_str.endswith("+"):
                # Unbounded upper range (e.g. "25+")
                lo = int(range_str[:-1])
                if k >= lo:
                    return label
            else:
                # Bounded range (e.g. "5-9")
                lo, hi = (int(x) for x in range_str.split("-"))
                if lo <= k <= hi:
                    return label
        return "unknown"
