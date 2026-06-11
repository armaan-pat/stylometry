"""Tests for EpisodicPrototypeLoss (variable-K' prototypical episodes).

Covers:
  - separated sender clusters score lower than shuffled-label noise
  - gradients flow to the embeddings
  - synthetic __syn embeddings act as repulsion-only queries: a synthetic
    near its mimicked sender's prototype costs more than one far away
  - classes too small for an episode are skipped without error
  - sender_ids=None treats everything as real (no crash, finite loss)
  - supcon_weight=0 disables the aux term
  - support split respects support_k_min/max bounds (loss stays finite over
    many resamples — the split is stochastic)
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from email_fraud.losses.episodic import EpisodicPrototypeLoss


def _clusters(
    n_classes: int, per_class: int, d: int = 16, noise: float = 0.05, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    centers = F.normalize(torch.randn(n_classes, d, generator=g), dim=1)
    embs, labels = [], []
    for c in range(n_classes):
        x = centers[c] + noise * torch.randn(per_class, d, generator=g)
        embs.append(F.normalize(x, dim=1))
        labels.extend([c] * per_class)
    return torch.cat(embs), torch.tensor(labels)


def test_separated_clusters_beat_shuffled_labels() -> None:
    loss_fn = EpisodicPrototypeLoss(temperature=0.05, supcon_weight=0.0)
    embs, labels = _clusters(4, 8)
    torch.manual_seed(0)
    good = loss_fn(embs, labels).item()
    torch.manual_seed(0)
    shuffled = labels[torch.randperm(len(labels), generator=torch.Generator().manual_seed(7))]
    bad = loss_fn(embs, shuffled).item()
    assert good < bad


def test_gradient_flows() -> None:
    loss_fn = EpisodicPrototypeLoss(supcon_weight=0.5)
    embs, labels = _clusters(3, 6)
    embs = embs.clone().requires_grad_(True)
    loss = loss_fn(embs, labels)
    loss.backward()
    assert embs.grad is not None
    assert torch.isfinite(embs.grad).all()


def test_synthetic_repulsion_direction() -> None:
    """A __syn query sitting ON the mimicked prototype must cost more than one far away."""
    loss_fn = EpisodicPrototypeLoss(temperature=0.05, supcon_weight=0.0, support_k_min=2)
    embs, labels = _clusters(2, 6, noise=0.01, seed=1)
    alice_center = F.normalize(embs[:6].mean(0), dim=0)

    # Two synthetic-Alice emails appended as class 2.
    far = F.normalize(-alice_center + 0.01 * torch.randn(2, 16), dim=1)
    near = F.normalize(
        alice_center.unsqueeze(0) + 0.01 * torch.randn(2, 16), dim=1
    )
    labels_full = torch.cat([labels, torch.tensor([2, 2])])
    sender_ids = ["alice"] * 6 + ["bob"] * 6 + ["alice__syn"] * 2

    torch.manual_seed(0)
    loss_near = loss_fn(torch.cat([embs, near]), labels_full, sender_ids=sender_ids).item()
    torch.manual_seed(0)
    loss_far = loss_fn(torch.cat([embs, far]), labels_full, sender_ids=sender_ids).item()
    assert loss_near > loss_far


def test_small_classes_skipped() -> None:
    # Every class has 2 embeddings < support_k_min + min_queries = 3 → no
    # episodes; loss degenerates to zero (supcon off) without raising.
    loss_fn = EpisodicPrototypeLoss(support_k_min=2, supcon_weight=0.0)
    embs, labels = _clusters(4, 2)
    loss = loss_fn(embs, labels)
    assert loss.item() == 0.0


def test_sender_ids_none_is_all_real() -> None:
    loss_fn = EpisodicPrototypeLoss()
    embs, labels = _clusters(3, 8)
    loss = loss_fn(embs, labels, sender_ids=None)
    assert torch.isfinite(loss)


def test_supcon_weight_zero_differs() -> None:
    embs, labels = _clusters(3, 8, noise=0.3)
    torch.manual_seed(0)
    with_aux = EpisodicPrototypeLoss(supcon_weight=0.5)(embs, labels).item()
    torch.manual_seed(0)
    without = EpisodicPrototypeLoss(supcon_weight=0.0)(embs, labels).item()
    assert with_aux != without


def test_stochastic_split_stays_finite() -> None:
    loss_fn = EpisodicPrototypeLoss(support_k_min=2, support_k_max=6)
    embs, labels = _clusters(5, 8, noise=0.2)
    for _ in range(50):
        assert torch.isfinite(loss_fn(embs, labels))


def test_requires_flags() -> None:
    loss_fn = EpisodicPrototypeLoss()
    assert loss_fn.requires_pk_sampler
    assert loss_fn.requires_sender_ids
