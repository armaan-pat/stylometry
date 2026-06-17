"""Train-time text augmentation: random contiguous crops.

Production must score short emails (3.3% of traffic is under 10 words) but the
training corpus under-represents them. Cropping a training email to a random
contiguous span — labeled as the same sender — makes the SupCon/episodic loss
explicitly demand that a 15-word fragment of an Alice email embeds near full
Alice emails, i.e. length-invariant style features. It also manufactures
unlimited short positives without new data.

Crops slice the *original string* between word boundaries, so line breaks and
intra-span whitespace (stylometric signal) are preserved verbatim.
"""

from __future__ import annotations

import random
import re

from email_fraud.data.base import BaseDataset

_RE_WORD = re.compile(r"\S+")


def random_word_crop(
    text: str,
    rng: random.Random,
    min_words: int = 5,
    max_words: int = 60,
) -> str:
    """Return a random contiguous span of *text*, min_words..max_words long.

    Half the crops are anchored at the start of the email (greetings carry a
    large share of the stylometric signal in short mail); the rest start at a
    uniform random word. Texts with <= min_words words are returned unchanged —
    cropping an already-short email would destroy what little signal it has.
    """
    spans = [m.span() for m in _RE_WORD.finditer(text)]
    n = len(spans)
    if n <= min_words:
        return text
    hi = min(max_words, n - 1)  # strictly shorter than the original
    length = rng.randint(min_words, hi)
    start = 0 if rng.random() < 0.5 else rng.randint(0, n - length)
    return text[spans[start][0] : spans[start + length - 1][1]]


class CropAugmentedDataset(BaseDataset):
    """Wraps a train dataset; __getitem__ crops each text with probability crop_prob.

    The underlying *uncropped* ``_texts`` / ``_sender_ids_list`` are exposed
    unchanged because the Trainer (hard-negative mining) and the CentroidProbe
    read them directly — both are inference-style consumers that must see full
    emails. Only the per-item training stream is augmented.

    With seed=None the module-level ``random`` RNG is used, which PyTorch
    seeds differently per DataLoader worker — crops then vary across workers
    and epochs. Pass a seed only for reproducible single-process debugging.
    """

    def __init__(
        self,
        base: BaseDataset,
        crop_prob: float,
        min_words: int = 5,
        max_words: int = 60,
        seed: int | None = None,
    ) -> None:
        if not 0.0 <= crop_prob <= 1.0:
            raise ValueError(f"crop_prob must be in [0, 1], got {crop_prob}")
        if min_words < 1 or max_words < min_words:
            raise ValueError(
                f"need 1 <= min_words <= max_words, got {min_words}..{max_words}"
            )
        self._base = base
        self._texts = base._texts
        self._sender_ids_list = base._sender_ids_list
        self.crop_prob = crop_prob
        self.min_words = min_words
        self.max_words = max_words
        # random.Random and the random module share the needed API surface.
        self._rng: random.Random = random.Random(seed) if seed is not None else random  # type: ignore[assignment]

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, index: int) -> tuple[str, str]:
        text, sender_id = self._base[index]
        if self._rng.random() < self.crop_prob:
            text = random_word_crop(text, self._rng, self.min_words, self.max_words)
        return text, sender_id

    @property
    def sender_ids(self) -> list[str]:
        return self._sender_ids_list
