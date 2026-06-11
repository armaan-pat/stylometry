"""Tests for LLMDetectorLoss (binary human-vs-LLM classification head + BCE).

Covers:
  - the "__syn" suffix is the binary target (synthetics separable from humans
    drives the loss down vs an embedding that mixes them)
  - gradients flow to BOTH the embeddings and the classification head params
  - pos_weight / BCE stays finite on all-human and all-synthetic batches
    (no division-by-zero when one class is absent)
  - sender_ids=None does not crash and yields a finite loss
  - supcon_weight=0 disables the aux term (changes the value)
  - required flags
  - embedding_dim validation
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from email_fraud.losses.llm_detector import LLMDetectorLoss

D = 16


def _two_groups(n_human: int, n_syn: int, separable: bool, seed: int = 0):
    """Return (embeddings, labels, sender_ids) with n_human real + n_syn __syn.

    separable=True puts humans and synthetics on opposite poles (a perfect
    detector should drive BCE low); separable=False overlaps them.
    """
    g = torch.Generator().manual_seed(seed)
    human_center = F.normalize(torch.randn(D, generator=g), dim=0)
    syn_center = -human_center if separable else human_center
    humans = F.normalize(human_center + 0.05 * torch.randn(n_human, D, generator=g), dim=1)
    syns = F.normalize(syn_center + 0.05 * torch.randn(n_syn, D, generator=g), dim=1)
    embs = torch.cat([humans, syns])
    # Integer sender labels: spread humans across a few senders, synthetics too,
    # so the SupCon aux has positives.
    labels = torch.tensor([i % 3 for i in range(n_human)] + [100 + (i % 2) for i in range(n_syn)])
    sender_ids = [f"u{i % 3}" for i in range(n_human)] + [f"u{i % 2}__syn" for i in range(n_syn)]
    return embs, labels, sender_ids


def test_separable_beats_overlapping() -> None:
    loss_fn = LLMDetectorLoss(embedding_dim=D, supcon_weight=0.0)
    sep_e, sep_l, sep_s = _two_groups(12, 4, separable=True)
    ov_e, ov_l, ov_s = _two_groups(12, 4, separable=False)
    # Same head; train it briefly so the comparison reflects the data geometry,
    # not the random init. Use a couple of gradient steps on each.
    opt = torch.optim.Adam(loss_fn.parameters(), lr=0.1)
    for _ in range(50):
        opt.zero_grad()
        loss_fn(sep_e, sep_l, sender_ids=sep_s).backward()
        opt.step()
    sep = loss_fn(sep_e, sep_l, sender_ids=sep_s).item()

    loss_fn2 = LLMDetectorLoss(embedding_dim=D, supcon_weight=0.0)
    opt2 = torch.optim.Adam(loss_fn2.parameters(), lr=0.1)
    for _ in range(50):
        opt2.zero_grad()
        loss_fn2(ov_e, ov_l, sender_ids=ov_s).backward()
        opt2.step()
    ov = loss_fn2(ov_e, ov_l, sender_ids=ov_s).item()
    assert sep < ov


def test_gradient_flows_to_embeddings_and_head() -> None:
    loss_fn = LLMDetectorLoss(embedding_dim=D, supcon_weight=0.5)
    embs, labels, sids = _two_groups(12, 4, separable=True)
    embs = embs.clone().requires_grad_(True)
    loss = loss_fn(embs, labels, sender_ids=sids)
    loss.backward()
    assert embs.grad is not None and torch.isfinite(embs.grad).all()
    # The classification head must receive gradient — otherwise the optimizer
    # change in the trainer is moot.
    assert loss_fn.classifier.weight.grad is not None
    assert torch.isfinite(loss_fn.classifier.weight.grad).all()
    assert loss_fn.classifier.weight.grad.abs().sum() > 0


def test_all_human_batch_is_finite() -> None:
    loss_fn = LLMDetectorLoss(embedding_dim=D, supcon_weight=0.0)
    embs, labels, sids = _two_groups(16, 0, separable=True)
    loss = loss_fn(embs, labels, sender_ids=sids)
    assert torch.isfinite(loss)


def test_all_synthetic_batch_is_finite() -> None:
    loss_fn = LLMDetectorLoss(embedding_dim=D, supcon_weight=0.0)
    embs, labels, sids = _two_groups(0, 16, separable=True)
    loss = loss_fn(embs, labels, sender_ids=sids)
    assert torch.isfinite(loss)


def test_sender_ids_none_is_finite() -> None:
    loss_fn = LLMDetectorLoss(embedding_dim=D, supcon_weight=0.0)
    embs, labels, _ = _two_groups(12, 4, separable=True)
    loss = loss_fn(embs, labels, sender_ids=None)
    assert torch.isfinite(loss)


def test_supcon_weight_zero_differs() -> None:
    embs, labels, sids = _two_groups(12, 4, separable=False, seed=3)
    torch.manual_seed(0)
    with_aux = LLMDetectorLoss(embedding_dim=D, supcon_weight=0.5)(embs, labels, sender_ids=sids).item()
    torch.manual_seed(0)
    without = LLMDetectorLoss(embedding_dim=D, supcon_weight=0.0)(embs, labels, sender_ids=sids).item()
    assert with_aux != without


def test_requires_flags() -> None:
    loss_fn = LLMDetectorLoss(embedding_dim=D)
    assert loss_fn.requires_pk_sampler
    assert loss_fn.requires_sender_ids


def test_bad_embedding_dim_raises() -> None:
    with pytest.raises(ValueError):
        LLMDetectorLoss(embedding_dim=0)
    with pytest.raises(ValueError):
        LLMDetectorLoss(embedding_dim=None)
