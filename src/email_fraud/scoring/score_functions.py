"""Score functions: map (cos_sim, spread) → score in some convenient range.

A *score function* is a pure transformation of the geometric quantities the
PrototypicalHead computes (cosine similarity to centroid and the centroid's
spread). They share the same fitted profile — what changes is how that
(cos_sim, spread) pair is projected into a number suitable for thresholding,
logging, or display.

This file is the single source of truth. PrototypicalHead, CentroidProbe, and
scripts/analyze_thresholds.py all import from here so the names mean the same
thing everywhere. To add a new variant: add it to SCORE_FNS and (optionally)
to DEFAULT_SCORE_FNS.

Variants
--------
linear_z3   max(0, 1 - z/3)  — current default. Bounded [0, 1]. Score > 0.95
            is unreachable in practice (would need z < 0.15, i.e. query 6x
            closer to centroid than the average enrollment email).

linear_z2   max(0, 1 - z/2)  — sharper version; score > 0.95 needs z < 0.10.
            Even harder to reach but stretches the typical-genuine range.

cosine      (cos_sim + 1) / 2  — raw cosine mapped to [0, 1]. Ignores spread
            entirely, so it doesn't account for per-sender stylistic variance,
            but the score range is *actually* usable across the full [0, 1].

sigmoid_z   sigmoid(1 - z)  — calibrated-ish; saturates smoothly near 0 and 1
            instead of clamping. z=0 → 0.73, z=1 → 0.50, z=3 → 0.12. Useful if
            you want differentiable confidence.

neg_z       -z (unbounded below; closer is higher)  — for AUC/ranking only.
            Not a probability; threshold values are arbitrary. The cleanest
            signal for ROC analysis since it's monotone in cos_sim/spread.
"""

from __future__ import annotations

import math
from typing import Callable

_EPS = 1e-9


def _z(cos_sim: float, spread: float) -> float:
    return (1.0 - cos_sim) / max(spread, _EPS)


def linear_z3(cos_sim: float, spread: float) -> float:
    return max(0.0, 1.0 - _z(cos_sim, spread) / 3.0)


def linear_z2(cos_sim: float, spread: float) -> float:
    return max(0.0, 1.0 - _z(cos_sim, spread) / 2.0)


def cosine(cos_sim: float, spread: float) -> float:
    return (cos_sim + 1.0) / 2.0


def sigmoid_z(cos_sim: float, spread: float) -> float:
    return 1.0 / (1.0 + math.exp(_z(cos_sim, spread) - 1.0))


def neg_z(cos_sim: float, spread: float) -> float:
    return -_z(cos_sim, spread)


SCORE_FNS: dict[str, Callable[[float, float], float]] = {
    "linear_z3": linear_z3,
    "linear_z2": linear_z2,
    "cosine": cosine,
    "sigmoid_z": sigmoid_z,
    "neg_z": neg_z,
}

# The sweep used when no explicit list is given. linear_z3 first so it remains
# the canonical "score" everywhere downstream.
DEFAULT_SCORE_FNS: tuple[str, ...] = ("linear_z3",)
ALL_SCORE_FNS: tuple[str, ...] = tuple(SCORE_FNS.keys())


def resolve(name: str) -> Callable[[float, float], float]:
    if name not in SCORE_FNS:
        raise KeyError(
            f"Unknown score_fn '{name}'. Available: {sorted(SCORE_FNS)}"
        )
    return SCORE_FNS[name]
