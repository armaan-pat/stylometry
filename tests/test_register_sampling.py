"""Tests for the LLM-free register/length generalization mechanisms (B).

Covers:
  - register.detect_register buckets formal / casual / terse correctly
  - _register_stratified_pick maximises distinct-register coverage and degrades
    gracefully for single-register senders and small pools
  - PKSampler / SyntheticBalancedSampler build register-spanning episodes when
    register_labels are supplied, and validate label/sender_id length
  - SyntheticAugmentedDataset drops LLM positives (rows under a real sender_id)
    by default and keeps them only under the explicit llm_negatives_only=False
    ablation — enforcing "LLM text is a hard negative, never a positive"
"""

from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from email_fraud.data.base import BaseDataset
from email_fraud.data.register import detect_register, partition_by_register
from email_fraud.data.samplers import (
    PKSampler,
    SyntheticBalancedSampler,
    _register_stratified_pick,
)
from email_fraud.data.synthetic import SyntheticAugmentedDataset


class _ToyReal(BaseDataset):
    def __init__(self, texts: list[str], senders: list[str]) -> None:
        self._texts = texts
        self._sender_ids_list = senders

    def __len__(self) -> int:
        return len(self._texts)

    def __getitem__(self, index: int) -> tuple[str, str]:
        return self._texts[index], self._sender_ids_list[index]

    @property
    def sender_ids(self) -> list[str]:
        return self._sender_ids_list


# --------------------------------------------------------------------------- #
# register detection
# --------------------------------------------------------------------------- #

def test_detect_register_terse_on_short_text():
    assert detect_register("ok thanks") == "terse"
    assert detect_register("") == "terse"


def test_detect_register_casual_vs_formal():
    casual = (
        "hey man wanna grab lunch this weekend? haha that place was awesome, "
        "thanks again for the rec, hope the kids are doing great"
    )
    formal = (
        "Please review the attached contract regarding the budget proposal and "
        "confirm approval before the compliance committee meeting deadline so that "
        "the department can forward the signed agreement to management for the "
        "scheduled transaction review next week."
    )
    assert detect_register(casual) == "casual"
    assert detect_register(formal) == "formal"


def test_partition_by_register_groups_all():
    texts = ["ok", "x " * 30]  # terse + a long low-signal (formal default)
    buckets = partition_by_register(texts)
    assert sum(len(v) for v in buckets.values()) == 2
    assert set(buckets) == {"formal", "casual", "terse"}


# --------------------------------------------------------------------------- #
# stratified pick
# --------------------------------------------------------------------------- #

def test_stratified_pick_covers_all_registers():
    rng = random.Random(0)
    pool = list(range(7))
    reg = ["formal", "formal", "formal", "casual", "casual", "casual", "terse"]
    picks = _register_stratified_pick(pool, 4, reg, rng)
    assert len(picks) == 4
    assert {reg[i] for i in picks} == {"formal", "casual", "terse"}


def test_stratified_pick_single_register_returns_available():
    rng = random.Random(0)
    picks = _register_stratified_pick([0, 1, 2], 4, ["formal"] * 3, rng)
    assert sorted(picks) == [0, 1, 2]  # only 3 available, asked for 4


def test_stratified_pick_no_duplicates():
    rng = random.Random(3)
    reg = ["formal", "casual", "terse", "formal", "casual"]
    picks = _register_stratified_pick(list(range(5)), 5, reg, rng)
    assert sorted(picks) == [0, 1, 2, 3, 4]


# --------------------------------------------------------------------------- #
# samplers with register labels
# --------------------------------------------------------------------------- #

def _toy_senders(n_senders: int, n_each: int):
    sids, regs = [], []
    cycle = ["formal", "casual", "terse"]
    for s in range(n_senders):
        for j in range(n_each):
            sids.append(f"s{s}")
            regs.append(cycle[j % 3])
    return sids, regs


def test_pksampler_builds_register_spanning_episodes():
    sids, regs = _toy_senders(6, 9)
    pk = PKSampler(sids, p=3, k=3, seed=1, register_labels=regs)
    batch = next(iter(pk))
    by_sender: dict[str, set[str]] = {}
    for idx in batch:
        by_sender.setdefault(sids[idx], set()).add(regs[idx])
    # k=3 and 3 registers available → every sender's episode spans all three
    assert all(rs == {"formal", "casual", "terse"} for rs in by_sender.values())


def test_pksampler_rejects_mismatched_register_labels():
    sids, regs = _toy_senders(4, 4)
    with pytest.raises(ValueError, match="register_labels length"):
        PKSampler(sids, p=2, k=2, register_labels=regs[:-1])


def test_synthetic_balanced_sampler_accepts_register_labels():
    sids, regs = _toy_senders(6, 8)
    # add __syn twins for two senders
    for s in range(2):
        for j in range(8):
            sids.append(f"s{s}__syn")
            regs.append(["formal", "casual", "terse"][j % 3])
    sb = SyntheticBalancedSampler(
        sids, p=4, k=4, n_syn=1, seed=2, register_labels=regs
    )
    batches = list(sb)
    assert batches and all(len(b) == 4 * 4 for b in batches)


# --------------------------------------------------------------------------- #
# LLM-positive enforcement at consumption time
# --------------------------------------------------------------------------- #

def _write_syn(tmp: str, sender_ids: list[str]):
    from datasets import Dataset

    Dataset.from_dict({
        "text": [f"t{i}" for i in range(len(sender_ids))],
        "sender_id": sender_ids,
    }).save_to_disk(tmp)


def test_synthetic_dataset_drops_llm_positives_by_default():
    real = _ToyReal(["r0", "r1"], ["alice", "bob"])
    with tempfile.TemporaryDirectory() as tmp:
        _write_syn(tmp, ["alice__syn", "bob__syn", "alice", "bob"])
        ds = SyntheticAugmentedDataset(real, tmp)
    # real rows + only the __syn synthetic rows survive
    assert ds.sender_ids == ["alice", "bob", "alice__syn", "bob__syn"]
    assert not any(s == "alice" or s == "bob" for s in ds.sender_ids[2:])


def test_synthetic_dataset_ablation_keeps_llm_positives():
    real = _ToyReal(["r0", "r1"], ["alice", "bob"])
    with tempfile.TemporaryDirectory() as tmp:
        _write_syn(tmp, ["alice__syn", "bob__syn", "alice", "bob"])
        ds = SyntheticAugmentedDataset(real, tmp, llm_negatives_only=False)
    assert len(ds.sender_ids) == 6  # nothing dropped
