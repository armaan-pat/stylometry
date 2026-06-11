"""Discriminative LLM-vs-human loss — a binary classification head + BCE.

Motivation (docs/v10_two_model_memo.md §2): "LLM-ness" is an *easy, global*
axis — one generator (Mistral-7B) wrote every synthetic impostor and a frozen
linear probe separates its text from human text with ~100% accuracy. The
metric-learning objectives (supcon / episodic) optimize per-sender clustering
and only acquire that axis *incidentally*, as a side effect that peaks early
(v9 genuine-vs-synthetic AUC peaks at epoch 10, then decays). This loss
optimizes the axis *directly*: it attaches a linear classification head to the
pooled embedding and trains it with binary cross-entropy to predict
human (0) vs LLM-generated (1), where the label is read straight off the
``__syn`` suffix every synthetic hard negative already carries.

Why a classification head beats metric learning *for detection*:
  - BCE maximizes margin on the one decision we care about, instead of
    diffusing capacity across P-choose-2 sender contrasts.
  - It contrasts *all* synthetics against *all* humans in the batch, so every
    real email is gradient signal for the boundary — far more than the
    episodic loss's handful of repulsion-only ``__syn`` queries.
  - It is sender-agnostic: the decision boundary does not depend on any
    enrolled centroid, so it generalizes to senders never seen at enrollment.

The head is a single ``nn.Linear(embedding_dim, 1)`` whose parameters live on
this module; the Trainer adds ``loss_fn.parameters()`` to the optimizer and
persists ``loss_state_dict`` in the checkpoint (see training/trainer.py).

A small SupCon auxiliary term (``supcon_weight``, default 0.3) is kept so the
embedding still clusters per-sender enough for centroid enrollment — that is
what keeps the deployed ``genuine_vs_synthetic`` CentroidProbe (and the
``pauc/genuine_vs_synthetic_5pct`` checkpoint monitor) meaningful and
comparable to the rest of the lineage. Set ``supcon_weight=0`` for a pure
detector and score off the classifier logit instead.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from email_fraud.losses.base import BaseLoss
from email_fraud.losses.supcon import SupConLoss
from email_fraud.registry import register

_SYN_SUFFIX = "__syn"


@register("loss", "llm_detector")
class LLMDetectorLoss(BaseLoss):
    """Binary human-vs-LLM classification head trained with BCE (+ SupCon aux).

    Args:
        embedding_dim: width of the pooled embedding the head reads
                       (= encoder.projection_dim; passed in by train.py).
        temperature:   temperature for the auxiliary SupCon term.
        supcon_weight: weight of the auxiliary SupCon term (0 disables it,
                       giving a pure discriminative detector).
    """

    def __init__(
        self,
        embedding_dim: int,
        temperature: float = 0.05,
        supcon_weight: float = 0.3,
    ) -> None:
        super().__init__()
        if embedding_dim is None or embedding_dim < 1:
            raise ValueError(
                f"embedding_dim must be a positive int, got {embedding_dim!r}. "
                "It is derived from encoder.projection_dim in train.py."
            )
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.embedding_dim = embedding_dim
        self.supcon_weight = supcon_weight
        # The discriminative head: pooled embedding -> single LLM-vs-human logit.
        self.classifier = nn.Linear(embedding_dim, 1)
        self._supcon = SupConLoss(temperature=temperature) if supcon_weight > 0 else None

    @property
    def requires_pk_sampler(self) -> bool:
        # The SyntheticBalancedSampler (PK + n_syn) guarantees both classes are
        # present every batch, and the SupCon aux needs the PK structure.
        return True

    @property
    def requires_sender_ids(self) -> bool:
        # The human/LLM label is read off the "__syn" suffix of each sender id.
        return True

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        sender_ids: list[str] | None = None,
    ) -> torch.Tensor:
        """Args: embeddings (N, d) L2-normalized, labels (N,) int sender ids,
        sender_ids raw id strings — the ``__syn`` suffix is the binary target.

        Returns the BCE detector loss (+ supcon_weight * SupCon aux). With no
        synthetic sender ids present the BCE term degenerates to all-negative;
        ``pos_weight`` is only applied when both classes are in the batch.
        """
        device = embeddings.device

        if sender_ids is None:
            # No labels to discriminate on — return a zero-with-grad BCE term so
            # the aux (if any) still trains and the call never crashes.
            y = torch.zeros(embeddings.size(0), device=device)
        else:
            y = torch.tensor(
                [1.0 if sid.endswith(_SYN_SUFFIX) else 0.0 for sid in sender_ids],
                device=device,
            )

        logits = self.classifier(embeddings).squeeze(-1)

        n_pos = float(y.sum().item())
        n_neg = float(y.numel() - n_pos)
        # Synthetics are the batch minority (n_syn=4 of P=16); upweight the
        # positive class so the boundary isn't dragged toward "always human".
        # Only meaningful when both classes are present this batch.
        if n_pos > 0 and n_neg > 0:
            pos_weight = torch.tensor(n_neg / n_pos, device=device)
        else:
            pos_weight = None

        bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)

        if self._supcon is not None:
            return bce + self.supcon_weight * self._supcon(embeddings, labels)
        return bce
