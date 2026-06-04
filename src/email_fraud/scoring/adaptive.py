"""Data-driven (adaptive) scoring prototypes.

The production head (`PrototypicalHead`) scores with global constants: the z
divisor is a hard-coded `/3`, the EWMA `alpha` is a fixed `0.1`, the
cosine->Mahalanobis switch is a hard cliff at `k=5`. This module prototypes the
"make every parameter a function of the data" direction from
`docs/scoring_explained.md` so the choices can be ablated head-to-head.

Design
------
A single `ProfileBank` is fit ONCE on (enrollment_embeddings, sender_ids) and
caches every per-sender statistic any scorer might need (centroids, spread,
leave-one-out genuine z distribution, a data-driven per-sender z scale, and a
lazily-fit Ledoit-Wolf precision). Each *scorer* is then a cheap, stateless
function `(bank, query, sender_id) -> float` (higher = more genuine), so the
ablation pays the encode + covariance cost once and sweeps many scorers.

Scorers (see SCORERS at the bottom)
-----------------------------------
baseline_linear_z3   max(0, 1 - z/3)            current production default
baseline_cosine      (cos+1)/2                  spread-free baseline
ewma_centroid_z3     z3 but on a fixed-alpha EWMA centroid (recency baseline)
z_global_cal         divisor = pooled p90 genuine z   (data-driven, global)
z_persender_cal      divisor = per-sender shrunk p90 genuine z  ← headline
z_persender_sigmoid  smooth calibrated variant of the above
mahalanobis          per-sender Ledoit-Wolf distance, k-gated -> cosine
mahal_blend          smooth cosine<->Mahalanobis blend in PRECISION space
tier_switch          cosine for k<5 else mahalanobis (tier-conditional)

Everything operates on L2-normalized embeddings, so cosine similarity is just a
dot product and squared Euclidean distance is `2(1 - cos)`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

_EPS = 1e-9

# Calibration knobs for the data-driven z scorers. These are themselves
# defaults, not magic constants baked into the score: Q_PCT is which genuine
# quantile we anchor the divisor on, S_TARGET is the score we want a genuine
# email AT that quantile to receive, and SHRINK_N0 is the pseudo-count pulling a
# sparse sender's scale toward the global prior (James-Stein style).
Q_PCT = 90.0
S_TARGET = 0.30
SHRINK_N0 = 8.0

# Mahalanobis blend schedule: below MIN_K we are pure cosine, by MIN_K+WINDOW we
# are pure Mahalanobis, linearly in between.
MAHAL_MIN_K = 5
MAHAL_WINDOW = 10
RIDGE = 1e-4


# ---------------------------------------------------------------------------
# Per-sender statistics
# ---------------------------------------------------------------------------


@dataclass
class SenderStats:
    sid: str
    embs: np.ndarray                 # (k, d) L2-normalized
    centroid: np.ndarray             # equal-weight mean, renormalized (d,)
    centroid_ewma: np.ndarray        # fixed-alpha EWMA centroid, renormalized
    spread: float                    # mean cosine distance to equal centroid
    spread_ewma: float
    k: int
    loo_z: np.ndarray                # leave-one-out genuine z-scores (k,)
    z_scale: float = 0.0             # data-driven per-sender divisor (filled in pass 2)
    _prec: np.ndarray | None = field(default=None, repr=False)  # lazy LW precision


class ProfileBank:
    """Fit-once container of per-sender stats; scorers read from it."""

    def __init__(self, ewma_alpha: float = 0.1) -> None:
        self.ewma_alpha = ewma_alpha
        self.stats: dict[str, SenderStats] = {}
        self.global_z_scale: float = 1.0   # pooled p90 of all genuine LOO z
        self._d: int | None = None

    # -- fit -------------------------------------------------------------

    def fit(self, embeddings: np.ndarray, sender_ids: list[str]) -> "ProfileBank":
        embeddings = _l2norm(np.asarray(embeddings, dtype=np.float64))
        self._d = embeddings.shape[1]

        by_sender: dict[str, list[int]] = {}
        for i, s in enumerate(sender_ids):
            by_sender.setdefault(s, []).append(i)

        # Pass 1: per-sender geometry + leave-one-out genuine z distribution.
        for sid, idx in by_sender.items():
            embs = embeddings[idx]                       # (k, d)
            k = embs.shape[0]
            centroid = _l2norm(embs.mean(axis=0))
            spread = float((1.0 - embs @ centroid).mean())
            centroid_ewma, spread_ewma = self._ewma_centroid(embs)
            loo_z = self._leave_one_out_z(embs)
            self.stats[sid] = SenderStats(
                sid=sid, embs=embs, centroid=centroid,
                centroid_ewma=centroid_ewma, spread=spread,
                spread_ewma=spread_ewma, k=k, loo_z=loo_z,
            )

        # Pass 2: global prior, then shrink each sender's divisor toward it.
        all_z = np.concatenate([s.loo_z for s in self.stats.values()]) if self.stats else np.array([1.0])
        self.global_z_scale = float(max(np.percentile(all_z, Q_PCT), _EPS))
        for s in self.stats.values():
            persender = float(np.percentile(s.loo_z, Q_PCT)) if len(s.loo_z) else self.global_z_scale
            # Shrinkage toward the global prior: sparse senders trust the prior,
            # dense senders trust themselves.
            s.z_scale = float(
                (s.k * persender + SHRINK_N0 * self.global_z_scale) / (s.k + SHRINK_N0)
            )
            s.z_scale = max(s.z_scale, _EPS)
        return self

    def _ewma_centroid(self, embs: np.ndarray) -> tuple[np.ndarray, float]:
        """Stream embeddings through a fixed-alpha EWMA (the store's update)."""
        a = self.ewma_alpha
        c = embs[0].copy()
        spread = 0.0
        for e in embs[1:]:
            c = (1 - a) * c + a * e
            n = np.linalg.norm(c)
            if n > _EPS:
                c = c / n
            spread = (1 - a) * spread + a * float(1.0 - e @ c)
        return _l2norm(c), float(spread)

    @staticmethod
    def _leave_one_out_z(embs: np.ndarray) -> np.ndarray:
        """Honest within-sender z for each enrollment email vs the rest.

        Holding email i out removes the self-similarity bias you'd get scoring an
        email against a centroid it helped define. With small k this is the
        difference between a calibrated scale and an optimistic one.
        """
        k = embs.shape[0]
        if k < 3:
            # Not enough to leave one out meaningfully; fall back to in-sample.
            c = _l2norm(embs.mean(axis=0))
            spread = float((1.0 - embs @ c).mean())
            return (1.0 - embs @ c) / max(spread, _EPS)
        total = embs.sum(axis=0)
        out = np.empty(k, dtype=np.float64)
        for i in range(k):
            rest = embs[np.arange(k) != i]
            c = _l2norm((total - embs[i]) / (k - 1))
            spread = float((1.0 - rest @ c).mean())
            out[i] = (1.0 - float(embs[i] @ c)) / max(spread, _EPS)
        return out

    # -- query-time helpers ---------------------------------------------

    def cos(self, query: np.ndarray, sid: str, which: str = "mean") -> float:
        s = self.stats[sid]
        c = s.centroid if which == "mean" else s.centroid_ewma
        return float(query @ c)

    def z(self, query: np.ndarray, sid: str, which: str = "mean") -> float:
        s = self.stats[sid]
        spread = s.spread if which == "mean" else s.spread_ewma
        return (1.0 - self.cos(query, sid, which)) / max(spread, _EPS)

    def precision(self, sid: str) -> np.ndarray:
        """Lazily fit + cache a Ledoit-Wolf precision, trace-normalized to ~I."""
        s = self.stats[sid]
        if s._prec is not None:
            return s._prec
        from sklearn.covariance import LedoitWolf

        d = self._d or s.embs.shape[1]
        if s.k < 2:
            s._prec = np.eye(d)
            return s._prec
        cov = LedoitWolf().fit(s.embs).covariance_
        scale = float(np.trace(cov)) / d
        cov_reg = cov + RIDGE * scale * np.eye(d)
        prec = np.linalg.inv(cov_reg)
        # Normalize so mean diagonal ~ 1, making it blendable with the identity.
        prec *= d / max(float(np.trace(prec)), _EPS)
        s._prec = prec
        return prec

    def mahalanobis(self, query: np.ndarray, sid: str, prec: np.ndarray | None = None) -> float:
        s = self.stats[sid]
        P = prec if prec is not None else self.precision(sid)
        diff = query - s.centroid
        return float(np.sqrt(max(diff @ P @ diff, 0.0)))


def _l2norm(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        n = np.linalg.norm(x)
        return x / n if n > _EPS else x
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, _EPS)


# ---------------------------------------------------------------------------
# Scorers — (bank, query, sid) -> float, higher = more genuine
# ---------------------------------------------------------------------------


def _s_linear_z3(bank: ProfileBank, q: np.ndarray, sid: str) -> float:
    return max(0.0, 1.0 - bank.z(q, sid) / 3.0)


def _s_cosine(bank: ProfileBank, q: np.ndarray, sid: str) -> float:
    return (bank.cos(q, sid) + 1.0) / 2.0


def _s_ewma_z3(bank: ProfileBank, q: np.ndarray, sid: str) -> float:
    return max(0.0, 1.0 - bank.z(q, sid, which="ewma") / 3.0)


def _calibrated(z: float, scale: float) -> float:
    """Map z to [0,1] so that z==scale -> S_TARGET, z==0 -> 1, monotone down."""
    return float(np.clip(1.0 - (1.0 - S_TARGET) * z / max(scale, _EPS), 0.0, 1.0))


def _s_z_global(bank: ProfileBank, q: np.ndarray, sid: str) -> float:
    return _calibrated(bank.z(q, sid), bank.global_z_scale)


def _s_z_persender(bank: ProfileBank, q: np.ndarray, sid: str) -> float:
    return _calibrated(bank.z(q, sid), bank.stats[sid].z_scale)


def _s_z_persender_sigmoid(bank: ProfileBank, q: np.ndarray, sid: str) -> float:
    # Smooth version: width = scale/3 so the knee sits near the genuine quantile.
    scale = bank.stats[sid].z_scale
    w = max(scale / 3.0, _EPS)
    return float(1.0 / (1.0 + np.exp((bank.z(q, sid) - scale) / w)))


def _s_mahalanobis(bank: ProfileBank, q: np.ndarray, sid: str) -> float:
    s = bank.stats[sid]
    if s.k < MAHAL_MIN_K:
        return _s_cosine(bank, q, sid)        # too few emails for a stable Sigma
    return -bank.mahalanobis(q, sid)


def _s_mahal_blend(bank: ProfileBank, q: np.ndarray, sid: str) -> float:
    """Interpolate cosine<->Mahalanobis in PRECISION space (no score-scale mixing).

    w * precision + (1-w) * I. At w=0 the metric is the identity, whose
    -distance is monotone in cosine on the unit sphere, so the blend degrades
    gracefully to the cosine ranking for cold-start senders.
    """
    s = bank.stats[sid]
    w = float(np.clip((s.k - MAHAL_MIN_K) / MAHAL_WINDOW, 0.0, 1.0))
    d = bank._d or s.embs.shape[1]
    P = w * bank.precision(sid) + (1.0 - w) * np.eye(d)
    diff = q - s.centroid
    return -float(np.sqrt(max(diff @ P @ diff, 0.0)))


def _s_tier_switch(bank: ProfileBank, q: np.ndarray, sid: str) -> float:
    s = bank.stats[sid]
    return _s_cosine(bank, q, sid) if s.k < MAHAL_MIN_K else -bank.mahalanobis(q, sid)


SCORERS: dict[str, Callable[[ProfileBank, np.ndarray, str], float]] = {
    "baseline_linear_z3": _s_linear_z3,
    "baseline_cosine": _s_cosine,
    "ewma_centroid_z3": _s_ewma_z3,
    "z_global_cal": _s_z_global,
    "z_persender_cal": _s_z_persender,
    "z_persender_sigmoid": _s_z_persender_sigmoid,
    "mahalanobis": _s_mahalanobis,
    "mahal_blend": _s_mahal_blend,
    "tier_switch": _s_tier_switch,
}

# The production default, used as the paired baseline in the ablation.
BASELINE = "baseline_linear_z3"


def score_pool(
    bank: ProfileBank,
    queries: np.ndarray,
    sender_ids: list[str],
    scorer: str,
) -> np.ndarray:
    """Apply one scorer to a pool of (query, claimed_sender) pairs."""
    fn = SCORERS[scorer]
    queries = _l2norm(np.asarray(queries, dtype=np.float64))
    return np.array([fn(bank, queries[i], sender_ids[i]) for i in range(len(sender_ids))])
