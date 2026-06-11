"""Tests for crop augmentation and the (newly enforced) preprocessing floors.

Covers:
  - random_word_crop length bounds, contiguity (crop is a substring of the
    original, so newlines/whitespace are preserved), and the short-text no-op
  - CropAugmentedDataset: identity at crop_prob=0, cropping at crop_prob=1,
    sender ids untouched, uncropped ._texts passthrough for probe/mining
  - preprocessing._is_usable now enforces min_body_words and min_alnum_ratio
    (previously declared-but-dead config) and the lowered char floor
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from email_fraud.config import PreprocessingConfig
from email_fraud.data.augment import CropAugmentedDataset, random_word_crop
from email_fraud.data.base import BaseDataset
from email_fraud.data.preprocessing import preprocess


class _ToyDataset(BaseDataset):
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


_LONG_TEXT = "\n".join(
    f"Line {i} of the email body has exactly eight words here." for i in range(20)
)  # 200 words across 20 lines


def test_crop_length_bounds() -> None:
    rng = random.Random(0)
    for _ in range(100):
        out = random_word_crop(_LONG_TEXT, rng, min_words=5, max_words=60)
        n = len(out.split())
        assert 5 <= n <= 60


def test_crop_is_contiguous_substring() -> None:
    rng = random.Random(1)
    for _ in range(50):
        out = random_word_crop(_LONG_TEXT, rng, min_words=5, max_words=60)
        assert out in _LONG_TEXT  # preserves newlines and spacing verbatim


def test_short_text_unchanged() -> None:
    rng = random.Random(2)
    short = "ok thanks bye"
    assert random_word_crop(short, rng, min_words=5, max_words=60) == short


def test_dataset_identity_at_zero_prob() -> None:
    ds = _ToyDataset([_LONG_TEXT, "hello there everyone again"], ["a", "b"])
    wrapped = CropAugmentedDataset(ds, crop_prob=0.0, seed=0)
    assert wrapped[0] == ds[0]
    assert wrapped[1] == ds[1]
    assert len(wrapped) == 2


def test_dataset_crops_at_full_prob() -> None:
    ds = _ToyDataset([_LONG_TEXT], ["a"])
    wrapped = CropAugmentedDataset(ds, crop_prob=1.0, min_words=5, max_words=30, seed=3)
    text, sid = wrapped[0]
    assert sid == "a"
    assert len(text.split()) <= 30
    assert text in _LONG_TEXT


def test_dataset_exposes_uncropped_texts() -> None:
    ds = _ToyDataset([_LONG_TEXT], ["a"])
    wrapped = CropAugmentedDataset(ds, crop_prob=1.0, seed=4)
    # Probe / hard-negative mining read ._texts directly — must be full emails.
    assert wrapped._texts[0] == _LONG_TEXT
    assert wrapped.sender_ids == ["a"]


def test_word_floor_enforced() -> None:
    cfg = PreprocessingConfig(min_body_chars=5, min_body_words=5)
    assert preprocess("too few words here", cfg) is None  # 4 words
    assert preprocess("just enough words right here", cfg) is not None  # 5 words


def test_alnum_ratio_enforced() -> None:
    cfg = PreprocessingConfig(min_body_chars=5, min_body_words=2, min_alnum_ratio=0.6)
    assert preprocess("$$ %% ## @@ !! ^^ && **", cfg) is None
    assert preprocess("perfectly normal words in a sentence", cfg) is not None


def test_lowered_char_floor_keeps_short_email() -> None:
    cfg = PreprocessingConfig()  # new defaults: 20 chars / 5 words
    assert preprocess("Sounds good, see you at noon.", cfg) is not None
