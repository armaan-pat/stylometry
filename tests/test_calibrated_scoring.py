"""Tests for the per-sender calibrated scoring path (z_persender_sigmoid).

Covers:
  - score_raw() exposes a finite positive z_scale once a profile has embeddings
  - score(score_fn="z_persender_sigmoid") is in (0, 1), higher for in-cluster
    queries than for far-away ones
  - shrinkage: a sparse sender's z_scale sits near the global prior; a dense
    sender with a much wider genuine z distribution pulls away from it
  - the raw score function falls back to the default scale on NaN z_scale
  - fit() after scoring re-dirties calibration (lazy refresh actually refreshes)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from email_fraud.heads.prototypical import PrototypicalHead
from email_fraud.scoring.score_functions import (
    CAL_DEFAULT_Z_SCALE,
    z_persender_sigmoid,
)


def _cluster(center: torch.Tensor, n: int, noise: float, seed: int) -> torch.Tensor:
    """n unit vectors scattered around a unit center with given noise."""
    g = torch.Generator().manual_seed(seed)
    x = center.unsqueeze(0) + noise * torch.randn(n, center.shape[0], generator=g)
    return x / x.norm(dim=1, keepdim=True)


def _unit(d: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(d, generator=g)
    return x / x.norm()


def test_score_raw_exposes_z_scale() -> None:
    head = PrototypicalHead(score_fn="z_persender_sigmoid")
    embs = _cluster(_unit(32, 1), 10, 0.3, seed=2)
    head.fit(embs, ["s0"] * 10)
    r = head.score_raw(_unit(32, 3), "s0")
    assert np.isfinite(r["z_scale"])
    assert r["z_scale"] > 0.0


def test_calibrated_score_orders_near_vs_far() -> None:
    head = PrototypicalHead(score_fn="z_persender_sigmoid")
    center = _unit(32, 1)
    embs = _cluster(center, 12, 0.2, seed=2)
    head.fit(embs, ["s0"] * 12)

    near = _cluster(center, 1, 0.2, seed=7)[0]
    far = -center  # antipodal: maximally distant on the sphere
    r_near = head.score(near, "s0")
    r_far = head.score(far, "s0")
    assert 0.0 <= r_far["score"] <= r_near["score"] <= 1.0
    assert r_near["score"] > 0.5  # in-cluster genuine should clear the knee


def test_shrinkage_pulls_sparse_senders_to_prior() -> None:
    head = PrototypicalHead(score_fn="z_persender_sigmoid")
    # Dense, deliberately heterogeneous sender (wide z distribution) plus a
    # sparse k=2 sender; the sparse one must sit close to the pooled prior.
    c0, c1 = _unit(32, 1), _unit(32, 2)
    head.fit(_cluster(c0, 25, 0.6, seed=3), ["dense"] * 25)
    head.fit(_cluster(c1, 2, 0.1, seed=4), ["sparse"] * 2)

    head._refresh_calibration()
    g = head._global_z_scale
    sparse = head._profiles["sparse"]
    raw_p90 = float(np.percentile(sparse["_loo_z"], 90.0))
    # k=2 with n0=8 → the shrunk scale keeps only k/(k+n0)=20% of the gap
    # between the sender's own p90 and the pooled prior.
    expected = (2 * raw_p90 + 8 * g) / 10
    assert sparse["z_scale"] == pytest.approx(expected)
    assert abs(sparse["z_scale"] - g) <= 0.2 * abs(raw_p90 - g) + 1e-9
    assert sparse["z_scale"] > 0.0


def test_raw_fn_nan_scale_falls_back_to_default() -> None:
    s_nan = z_persender_sigmoid(0.9, 0.1, float("nan"))
    s_def = z_persender_sigmoid(0.9, 0.1, CAL_DEFAULT_Z_SCALE)
    assert s_nan == s_def
    assert 0.0 <= s_nan <= 1.0


def test_fit_after_score_refreshes_calibration() -> None:
    head = PrototypicalHead(score_fn="z_persender_sigmoid")
    c = _unit(32, 1)
    head.fit(_cluster(c, 8, 0.2, seed=5), ["s0"] * 8)
    _ = head.score(_unit(32, 6), "s0")
    assert head._cal_dirty is False
    head.fit(_cluster(c, 8, 0.2, seed=9), ["s0"] * 8)
    assert head._cal_dirty is True
    _ = head.score(_unit(32, 6), "s0")
    assert head._cal_dirty is False
