"""BaseLoss ABC.

All contrastive losses operate on a batch of L2-normalized embeddings and an
integer label tensor.  The requires_pk_sampler flag signals to the training
script whether a PKSampler must be used to construct batches — SupCon and
batch-hard triplet both require it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseLoss(nn.Module, ABC):
    """Abstract base for contrastive training objectives.

    Subclasses implement forward() which consumes a batch of embeddings and
    their sender-id integer labels and returns a scalar loss tensor.

    The requires_pk_sampler property must return True when the loss function
    depends on a specific P×K batch structure (e.g. SupCon, batch-hard
    triplet) so the training script can enforce correct DataLoader setup.

    Why inherit from nn.Module?
    ---------------------------
    Losses may contain learnable parameters (e.g. a temperature parameter that
    should be optimized).  Inheriting from nn.Module lets them participate in
    model.parameters() and be saved in checkpoints if needed.  Even losses
    with no learnable params use nn.Module so the call signature is uniform.
    """

    @abstractmethod
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the contrastive loss.

        Args:
            embeddings: (N, d) float tensor, L2-normalized.
                        N = P*K when using PKSampler (P senders × K emails).
            labels:     (N,) long tensor of integer sender ids.
                        Assigned by episode_collate() in order of first appearance.

        Returns:
            Scalar loss tensor (with grad_fn so backward() works).
        """
        ...

    @property
    def requires_sender_ids(self) -> bool:
        """True if forward() needs the raw sender-id strings as a kwarg.

        Default False — most losses only need integer labels. Losses that
        must distinguish synthetic hard negatives (sender ids ending in
        "__syn") from real senders override this; the Trainer then passes
        sender_ids=<list[str]> aligned with the (episode-strided) labels.
        """
        return False

    @property
    @abstractmethod
    def requires_pk_sampler(self) -> bool:
        """True if this loss requires a PK-structured batch.

        When True, the training script must use PKSampler with matching P and K
        values so each batch contains exactly P senders × K emails each.
        Losses like SupCon and batch-hard Triplet need this guarantee to have
        enough positives per anchor.  A plain random-batch loader would
        frequently produce batches with singletons (only one email per sender),
        giving zero positives and a zero loss — no learning signal.
        """
        ...
