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

import numpy as np
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
        store_embeddings: bool = True,
        mahalanobis_min_k: int = 5,
        ridge: float = 1e-4,
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
        # Mahalanobis-flavor and calibrated names aren't in the (cos_sim,
        # spread) registry — they're dispatched directly in .score() below.
        # For backward-compat we still pre-resolve the legacy cos_sim/spread
        # fns so existing YAMLs work without change.
        from email_fraud.scoring.score_functions import CALIBRATED_SCORE_FNS
        self._mahalanobis_mode = score_fn in {"mahalanobis", "adaptive_k"}
        self._calibrated_mode = score_fn in CALIBRATED_SCORE_FNS
        if not self._mahalanobis_mode and not self._calibrated_mode:
            self._score_fn = _resolve_score_fn(score_fn)
        else:
            self._score_fn = None  # type: ignore[assignment]
        # Whether to retain raw enrollment embeddings on the profile. Required
        # for mahalanobis scoring and the K-conditional adaptive fallback.
        # Defaults on so changes are non-breaking; turn off only if memory matters.
        self.store_embeddings = (
            store_embeddings or self._mahalanobis_mode or self._calibrated_mode
        )
        # Below this k we fall back to cosine even in mahalanobis modes —
        # per-sender LW shrinkage of a rank-(k-1) sample covariance is too
        # noisy until k≈5 (K-sweep ablation: per-sender LW shrinkage is rank-deficient below this).
        self.mahalanobis_min_k = mahalanobis_min_k
        self.ridge = ridge
        # In-memory dict of profiles; keyed by sender_id string.
        self._profiles: dict[str, dict[str, Any]] = {}
        # Per-sender z calibration is bank-level state (the shrinkage prior
        # pools across all profiles), refreshed lazily after any fit().
        self._cal_dirty = True
        self._global_z_scale: float | None = None

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
        embeddings = embeddings.detach().cpu()  # profiles live on CPU
        unique_senders = list(dict.fromkeys(sender_ids))

        for sid in unique_senders:
            # Gather indices of this sender's emails in the current batch.
            idx = [i for i, s in enumerate(sender_ids) if s == sid]
            embs = embeddings[idx]  # (k, d)

            if sid not in self._profiles:
                centroid = embs.mean(dim=0)
                sims = F.cosine_similarity(embs, centroid.unsqueeze(0))
                spread = float((1.0 - sims).mean())
                self._profiles[sid] = {
                    "centroid": centroid,
                    "spread": spread,
                    "k": len(idx),
                }
                if self.store_embeddings:
                    # Keep raw enrollment embeddings on the profile so we can
                    # (re-)fit a per-sender Ledoit-Wolf covariance for Mahalanobis.
                    # CPU storage is fine — these are queried per-sender, not in tight loops.
                    self._profiles[sid]["embs"] = embs.clone()
                    self._profiles[sid]["_prec_dirty"] = True
                    self._profiles[sid]["_loo_dirty"] = True
                    self._cal_dirty = True
            else:
                prof = self._profiles[sid]
                old_k = prof["k"]
                new_k = old_k + len(idx)
                prof["centroid"] = (
                    prof["centroid"] * old_k + embs.sum(dim=0)
                ) / new_k
                # Spread approximated from the incoming batch only (not a full recomputation).
                all_embs = embs
                sims = F.cosine_similarity(
                    all_embs, prof["centroid"].unsqueeze(0)
                )
                prof["spread"] = float((1.0 - sims).mean())
                prof["k"] = new_k
                if self.store_embeddings:
                    # Append; covariance will be re-fit lazily on next mahalanobis call.
                    prof["embs"] = torch.cat([prof["embs"], embs], dim=0)
                    prof["_prec_dirty"] = True
                    prof["_loo_dirty"] = True
                    self._cal_dirty = True

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
                "z_scale": float("nan"),
                "tier": "unknown",
                "abstain": True,
            }

        self._refresh_calibration()
        prof = self._profiles[sender_id]
        centroid: torch.Tensor = prof["centroid"]
        spread: float = prof["spread"]
        k: int = prof["k"]

        cos_sim = float(F.cosine_similarity(query.unsqueeze(0), centroid.unsqueeze(0)))
        tier = self._k_to_tier(k)
        return {
            "cos_sim": cos_sim,
            "spread": spread,
            "z_scale": float(prof.get("z_scale", float("nan"))),
            "tier": tier,
            "abstain": tier == "low",
        }

    def score(
        self,
        query: torch.Tensor,
        sender_id: str,
    ) -> dict[str, object]:
        """Return score under the configured score_fn, tier, and abstain flag.

        Score-fn dispatch:
            "mahalanobis" — per-sender Ledoit-Wolf Mahalanobis distance, flipped
              so higher = more genuine. Wins by +3 AUC pp on g/syn at K=16-25
              (V7.2 K-sweep: +3 AUC pp on g/syn at K=16-25). Falls back to
              `linear_z3` when k < mahalanobis_min_k.
            "adaptive_k" — cosine for k < mahalanobis_min_k (Σ too unreliable),
              mahalanobis otherwise. The recommended production default.
            calibrated names (CALIBRATED_SCORE_FNS, e.g. "z_persender_sigmoid")
              — per-sender shrunk-z-scale calibration, computed from the
              profile's leave-one-out genuine z distribution.
            anything else — passes through to the (cos_sim, spread) registry.
        """
        raw = self.score_raw(query, sender_id)
        if raw["tier"] == "unknown":
            return {"score": 0.0, "tier": "unknown", "abstain": True}

        if self._calibrated_mode:
            from email_fraud.scoring.score_functions import resolve_calibrated
            fn = resolve_calibrated(self.score_fn_name)
            score_val = fn(
                float(raw["cos_sim"]), float(raw["spread"]), float(raw["z_scale"])
            )
        elif self._mahalanobis_mode:
            prof = self._profiles[sender_id]
            k = int(prof["k"])
            if k >= self.mahalanobis_min_k and "embs" in prof:
                mahal_dist = self._mahalanobis_distance(query, prof)
                score_val = -mahal_dist
            else:
                # Either k too small for a reliable Σ or embeddings weren't
                # stored (legacy profile). Fall back to cosine in both modes.
                from email_fraud.scoring.score_functions import cosine as _cosine
                score_val = _cosine(float(raw["cos_sim"]), float(raw["spread"]))
        else:
            score_val = self._score_fn(float(raw["cos_sim"]), float(raw["spread"]))

        return {
            "score": float(score_val),
            "tier": raw["tier"],
            "abstain": raw["abstain"],
        }

    # ------------------------------------------------------------------
    # Mahalanobis path
    # ------------------------------------------------------------------

    def _refresh_precision(self, prof: dict[str, Any]) -> None:
        """Refit the Ledoit-Wolf shrinkage covariance and cache its inverse.

        Done lazily on the first mahalanobis query after an upsert so we don't
        pay the cost during enrollment. Stored fields:
            _prec      — d×d precision matrix (regularised inverse of Σ)
            _shrinkage — α from LW (informational; should be in (0, 1))
        """
        from sklearn.covariance import LedoitWolf

        embs = prof["embs"].cpu().numpy().astype(np.float64)
        if embs.shape[0] < 2:
            d = embs.shape[1]
            prof["_prec"] = np.eye(d, dtype=np.float64)
            prof["_shrinkage"] = 1.0
        else:
            lw = LedoitWolf().fit(embs)
            cov = lw.covariance_
            d = cov.shape[0]
            scale = float(np.trace(cov)) / d
            cov_reg = cov + self.ridge * scale * np.eye(d, dtype=np.float64)
            prof["_prec"] = np.linalg.inv(cov_reg)
            prof["_shrinkage"] = float(lw.shrinkage_)
        prof["_prec_dirty"] = False

    # ------------------------------------------------------------------
    # Per-sender z calibration (z_persender_sigmoid et al.)
    # ------------------------------------------------------------------

    def _refresh_calibration(self) -> None:
        """Recompute per-sender z scales (lazy; no-op unless a fit() dirtied us).

        Two passes, mirroring scoring/adaptive.py::ProfileBank.fit:
          1. Per sender: leave-one-out genuine z distribution from the stored
             enrollment embeddings (honest — scoring an email against a
             centroid it helped define is optimistic, and at small k the bias
             is large).
          2. Global prior = pooled p90 of all LOO z; each sender's scale is
             the p90 of its own LOO z shrunk toward the prior with
             pseudo-count CAL_SHRINK_N0, so sparse senders trust the
             population and dense senders trust themselves.

        Profiles without stored embeddings get the global scale (or the
        CAL_DEFAULT_Z_SCALE fallback when no profile has embeddings).
        """
        if not self._cal_dirty:
            return
        from email_fraud.scoring.score_functions import (
            CAL_DEFAULT_Z_SCALE,
            CAL_Q_PCT,
            CAL_SHRINK_N0,
        )

        pooled: list[np.ndarray] = []
        for prof in self._profiles.values():
            if "embs" not in prof:
                continue
            if prof.get("_loo_dirty", True) or "_loo_z" not in prof:
                prof["_loo_z"] = self._leave_one_out_z(prof["embs"])
                prof["_loo_dirty"] = False
            pooled.append(prof["_loo_z"])

        if pooled:
            all_z = np.concatenate(pooled)
            self._global_z_scale = float(
                max(np.percentile(all_z, CAL_Q_PCT), 1e-9)
            )
        else:
            self._global_z_scale = CAL_DEFAULT_Z_SCALE

        for prof in self._profiles.values():
            loo = prof.get("_loo_z")
            if loo is None or len(loo) == 0:
                prof["z_scale"] = self._global_z_scale
                continue
            persender = float(np.percentile(loo, CAL_Q_PCT))
            k = int(prof["k"])
            prof["z_scale"] = float(
                max(
                    (k * persender + CAL_SHRINK_N0 * self._global_z_scale)
                    / (k + CAL_SHRINK_N0),
                    1e-9,
                )
            )
        self._cal_dirty = False

    @staticmethod
    def _leave_one_out_z(embs: torch.Tensor) -> np.ndarray:
        """Honest within-sender z for each enrollment email vs the rest.

        Uses the same geometry as score_raw (cosine distance to centroid,
        normalised by the mean cosine distance of the defining emails), so the
        resulting scale is directly comparable to query-time z values.
        """
        k = embs.shape[0]
        if k < 3:
            # Not enough to leave one out meaningfully; fall back to in-sample.
            c = embs.mean(dim=0)
            sims = F.cosine_similarity(embs, c.unsqueeze(0))
            spread = float((1.0 - sims).mean())
            return ((1.0 - sims) / max(spread, 1e-9)).numpy().astype(np.float64)

        total = embs.sum(dim=0)
        out = np.empty(k, dtype=np.float64)
        for i in range(k):
            mask = torch.arange(k) != i
            rest = embs[mask]
            c = (total - embs[i]) / (k - 1)
            rest_sims = F.cosine_similarity(rest, c.unsqueeze(0))
            spread = float((1.0 - rest_sims).mean())
            sim_i = float(F.cosine_similarity(embs[i].unsqueeze(0), c.unsqueeze(0)))
            out[i] = (1.0 - sim_i) / max(spread, 1e-9)
        return out

    def _mahalanobis_distance(self, query: torch.Tensor, prof: dict[str, Any]) -> float:
        if prof.get("_prec_dirty", True):
            self._refresh_precision(prof)
        q = query.cpu().numpy().astype(np.float64)
        mu = prof["centroid"].cpu().numpy().astype(np.float64)
        diff = q - mu
        val = float(diff @ prof["_prec"] @ diff)
        return float(np.sqrt(max(val, 0.0)))

    def save(self, path: str) -> None:
        # Convert tensors to numpy for pickle portability across PyTorch versions.
        payload = {}
        for sid, prof in self._profiles.items():
            entry: dict[str, Any] = {
                "centroid": prof["centroid"].numpy(),
                "spread": prof["spread"],
                "k": prof["k"],
            }
            if "embs" in prof:
                entry["embs"] = prof["embs"].numpy()
            payload[sid] = entry
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)

    def load(self, path: str) -> None:
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        # Restore numpy arrays back to PyTorch tensors. Embeddings are optional
        # (older profile files don't have them); mark precision dirty so it
        # refits lazily on the first mahalanobis call.
        self._profiles = {}
        for sid, data in payload.items():
            prof: dict[str, Any] = {
                "centroid": torch.from_numpy(np.array(data["centroid"])),
                "spread": data["spread"],
                "k": data["k"],
            }
            if "embs" in data:
                prof["embs"] = torch.from_numpy(np.array(data["embs"]))
                prof["_prec_dirty"] = True
                prof["_loo_dirty"] = True
            self._profiles[sid] = prof
        self._cal_dirty = True

    # ------------------------------------------------------------------
    # Mahalanobis stub
    # ------------------------------------------------------------------

    def mahalanobis_score(
        self,
        query: torch.Tensor,
        sender_id: str,
    ) -> float:
        """Per-sender Ledoit-Wolf Mahalanobis distance to the centroid.

        Lower = more in-distribution (raw distance, not flipped). Returns
        +inf if the sender is unknown, raises if the profile was loaded
        without embeddings.
        """
        if sender_id not in self._profiles:
            return float("inf")
        prof = self._profiles[sender_id]
        if "embs" not in prof:
            raise RuntimeError(
                f"Cannot compute Mahalanobis for sender {sender_id!r}: "
                "profile was built/loaded without raw embeddings. "
                "Re-fit with store_embeddings=True (default)."
            )
        return self._mahalanobis_distance(query.detach().cpu().squeeze(), prof)

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
