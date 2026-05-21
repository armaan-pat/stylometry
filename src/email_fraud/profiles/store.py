"""SenderProfileStore — in-memory per-sender embedding profiles.

Profiles are updated incrementally via EWMA centroid updates so live traffic
can refine a sender's representation without full recomputation.

EWMA centroid update
--------------------
For each incoming embedding e:
    centroid_new = (1 - α) * centroid_old + α * e

then re-normalised to the unit sphere.  α=0.1 weights recent emails lightly
so a single outlier email does not drastically shift the profile centroid.
Compare to the PrototypicalHead's online averaging which gives equal weight
to all emails seen so far — use the store when you want recency bias,
use the head's fit() when you want equal weight.

TODO: swap dict for Postgres pgvector for production deployments where
      profiles need to be shared across inference workers and persisted
      across restarts (e.g. pgvector + asyncpg).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


class SenderProfileStore:
    """Lightweight in-memory store for per-sender contrastive embeddings.

    The store maintains an EWMA centroid per sender rather than all raw
    embeddings to keep memory bounded.  The Mahalanobis path is stubbed;
    the cosine-distance path (via centroid + spread) is fully functional.

    Args:
        ewma_alpha:        Smoothing factor for centroid updates (0 = frozen,
                           1 = replace).  Default 0.1 weights recent emails
                           more than the stored centroid.
        confidence_tiers:  k-range → tier label mapping from ExperimentConfig.
    """

    def __init__(
        self,
        ewma_alpha: float = 0.1,
        confidence_tiers: dict[str, str] | None = None,
    ) -> None:
        self.ewma_alpha = ewma_alpha
        self.confidence_tiers = confidence_tiers or {
            "1-4": "low",
            "5-9": "medium",
            "10-24": "high",
            "25+": "very_high",
        }
        # TODO: swap dict for Postgres pgvector
        self._profiles: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def upsert(
        self,
        sender_id: str,
        embedding: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Insert or update a sender's profile with a new embedding.

        Uses EWMA for centroid so individual noisy emails have limited impact.

        Args:
            sender_id:  Unique sender identifier string.
            embedding:  (d,) float32 numpy array, L2-normalized.
            metadata:   Arbitrary key-value extras stored alongside the profile.
        """
        embedding = embedding.astype(np.float32)

        if sender_id not in self._profiles:
            # First email for this sender — initialise centroid directly.
            # spread=0 because we have no distribution yet.
            self._profiles[sender_id] = {
                "centroid": embedding.copy(),
                "spread": 0.0,
                "k": 1,
                "metadata": metadata or {},
            }
            return

        prof = self._profiles[sender_id]
        alpha = self.ewma_alpha

        # EWMA centroid update: blend old centroid with new embedding.
        # α=0.1 means the centroid moves only 10% toward the new email.
        old_centroid: np.ndarray = prof["centroid"]
        new_centroid = (1 - alpha) * old_centroid + alpha * embedding

        # Re-normalise to unit sphere after blending (blending breaks unit norm).
        norm = np.linalg.norm(new_centroid)
        if norm > 1e-9:
            new_centroid = new_centroid / norm

        # EWMA of cosine distance from new embedding to updated centroid.
        # This tracks how "scattered" this sender's emails are around the centroid.
        cos_dist = float(1.0 - np.dot(embedding, new_centroid))
        prof["spread"] = (1 - alpha) * prof["spread"] + alpha * cos_dist

        prof["centroid"] = new_centroid
        prof["k"] = prof["k"] + 1
        if metadata:
            prof["metadata"].update(metadata)

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def get_profile(self, sender_id: str) -> dict[str, Any] | None:
        """Return the stored profile dict or None if sender is unknown."""
        return self._profiles.get(sender_id)

    def mahalanobis_score(
        self,
        sender_id: str,
        query_embedding: np.ndarray,
    ) -> float:
        """Compute Mahalanobis distance from query to sender's profile.

        TODO: implement full covariance estimation with Ledoit-Wolf shrinkage.
              Currently returns 0.0 as a placeholder so the pipeline runs.
        """
        # TODO: collect stored per-sender embeddings, estimate covariance via
        #       sklearn.covariance.LedoitWolf, compute proper Mahalanobis distance.
        raise NotImplementedError(
            "Mahalanobis scoring not yet implemented. "
            "See TODO comment in mahalanobis_score()."
        )

    def confidence_tier(self, sender_id: str) -> str:
        """Return the confidence tier string based on how many emails are stored."""
        prof = self.get_profile(sender_id)
        if prof is None:
            return "unknown"
        return self._k_to_tier(prof["k"])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise all profiles to a JSON file (centroids as lists)."""
        serialisable = {
            sid: {
                "centroid": prof["centroid"].tolist(),
                "spread": prof["spread"],
                "k": prof["k"],
                "metadata": prof["metadata"],
            }
            for sid, prof in self._profiles.items()
        }
        with open(path, "w") as fh:
            json.dump(serialisable, fh)

    def load(self, path: str) -> None:
        """Restore profiles from a JSON file produced by save()."""
        with open(path) as fh:
            raw = json.load(fh)
        self._profiles = {
            sid: {
                "centroid": np.array(data["centroid"], dtype=np.float32),
                "spread": data["spread"],
                "k": data["k"],
                "metadata": data.get("metadata", {}),
            }
            for sid, data in raw.items()
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _k_to_tier(self, k: int) -> str:
        for range_str, label in self.confidence_tiers.items():
            if range_str.endswith("+"):
                lo = int(range_str[:-1])
                if k >= lo:
                    return label
            else:
                lo, hi = (int(x) for x in range_str.split("-"))
                if lo <= k <= hi:
                    return label
        return "unknown"

    def __len__(self) -> int:
        return len(self._profiles)

    def __contains__(self, sender_id: str) -> bool:
        return sender_id in self._profiles
