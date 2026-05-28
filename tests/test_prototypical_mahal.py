"""Tests for PrototypicalHead Mahalanobis path (V7 addition).

Covers:
  - mahalanobis_score() returns a finite distance for a sender with embeddings
  - mahalanobis_score() falls back / errors for unknown / stripped profiles
  - score(score_fn="mahalanobis") flips sign so higher = more genuine
  - score(score_fn="adaptive_k") uses cosine when k < mahalanobis_min_k
    and Mahalanobis when k ≥ min_k
  - save() / load() round-trips embeddings
  - The legacy linear_z3 path is unchanged (backward compat)
"""

from __future__ import annotations

import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from email_fraud.heads.prototypical import PrototypicalHead


def _make_unit_embs(n: int, d: int, seed: int = 0) -> torch.Tensor:
    """n random L2-normalized d-dim vectors."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    return x / x.norm(dim=1, keepdim=True)


def test_legacy_linear_z3_unchanged() -> None:
    """The default path must still produce a score in [0, 1] and abstain at low k."""
    head = PrototypicalHead()
    embs = _make_unit_embs(8, 32)
    head.fit(embs, ["s0"] * 4 + ["s1"] * 4)
    q = _make_unit_embs(1, 32)[0]
    r = head.score(q, "s0")
    assert 0.0 <= r["score"] <= 1.0
    assert r["tier"] == "low"          # k=4 → low tier
    assert r["abstain"] is True


def test_mahalanobis_score_runs_and_is_finite() -> None:
    """Raw mahalanobis_score() should return a finite non-negative distance."""
    head = PrototypicalHead(score_fn="mahalanobis")
    embs = _make_unit_embs(8, 32)
    head.fit(embs, ["s0"] * 8)
    q = _make_unit_embs(1, 32)[0]
    d = head.mahalanobis_score(q, "s0")
    assert np.isfinite(d)
    assert d >= 0.0


def test_mahalanobis_unknown_sender_is_inf() -> None:
    head = PrototypicalHead(score_fn="mahalanobis")
    embs = _make_unit_embs(8, 32)
    head.fit(embs, ["s0"] * 8)
    assert head.mahalanobis_score(_make_unit_embs(1, 32)[0], "missing") == float("inf")


def test_mahalanobis_score_flips_sign() -> None:
    """score(score_fn=mahalanobis) should be negative (= -distance) so higher means more genuine."""
    head = PrototypicalHead(score_fn="mahalanobis", mahalanobis_min_k=2)
    embs = _make_unit_embs(8, 32)
    head.fit(embs, ["s0"] * 8)
    q = _make_unit_embs(1, 32)[0]
    r = head.score(q, "s0")
    assert r["score"] <= 0.0


def test_adaptive_k_dispatches_on_k() -> None:
    """adaptive_k uses cosine below min_k and mahalanobis at/above min_k."""
    head = PrototypicalHead(score_fn="adaptive_k", mahalanobis_min_k=5)
    embs = _make_unit_embs(8, 32)
    head.fit(embs, ["s0"] * 4)        # k = 4 → cosine path, score in [0, 1]
    q = _make_unit_embs(1, 32)[0]
    r_low = head.score(q, "s0")
    assert 0.0 <= r_low["score"] <= 1.0   # cosine fallback

    head.fit(embs, ["s0"] * 4)        # k now 8 → mahal path
    r_high = head.score(q, "s0")
    assert r_high["score"] <= 0.0         # mahal flipped


def test_save_load_round_trips_embeddings() -> None:
    """After save → load, mahalanobis_score should yield the same value."""
    head = PrototypicalHead(score_fn="mahalanobis", mahalanobis_min_k=2)
    embs = _make_unit_embs(8, 32)
    head.fit(embs, ["s0"] * 8)
    q = _make_unit_embs(1, 32)[0]
    d1 = head.mahalanobis_score(q, "s0")

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "prof.pkl"
        head.save(str(path))

        head2 = PrototypicalHead(score_fn="mahalanobis", mahalanobis_min_k=2)
        head2.load(str(path))
        d2 = head2.mahalanobis_score(q, "s0")
        # Allow tiny FP drift from re-fitting LW on the reloaded numpy array
        # (sklearn uses double precision; the re-fit is deterministic so we
        # expect exact agreement here).
        assert abs(d1 - d2) < 1e-9


def test_legacy_profile_without_embeddings_still_loads() -> None:
    """A pickle saved by the pre-V7 head (no 'embs' field) should still load."""
    legacy = {"s0": {
        "centroid": np.random.randn(32).astype(np.float32),
        "spread": 0.1,
        "k": 8,
    }}
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "legacy.pkl"
        with open(path, "wb") as fh:
            pickle.dump(legacy, fh)

        head = PrototypicalHead()
        head.load(str(path))
        q = torch.from_numpy(np.random.randn(32).astype(np.float32))
        q = q / q.norm()
        r = head.score(q, "s0")
        assert "score" in r and "tier" in r
