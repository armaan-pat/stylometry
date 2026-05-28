"""Supervised Contrastive Loss, L_sup_out variant (Khosla et al. NeurIPS 2020, arXiv:2004.11362).

L_sup_out places 1/|P(i)| outside the log (eq. 4), which is more numerically stable
than L_sup_in. Denominator uses log-sum-exp to avoid overflow.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from email_fraud.losses.base import BaseLoss

from email_fraud.registry import register


@register("loss", "supcon")
class SupConLoss(BaseLoss):
    """Supervised contrastive loss (L_sup_out).

    For each anchor i, averages -log P(z_p | z_i) over in-batch positives P(i).
    Denominator sums over all A(i) = {1..N} \\ {i}.
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.temperature = temperature

    @property
    def requires_pk_sampler(self) -> bool:
        # Random batches frequently produce K=1 per sender → no positives → zero loss.
        return True

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Args: embeddings (N, d) L2-normalized, labels (N,) int sender ids."""
        device = embeddings.device
        n = embeddings.size(0)

        sim = torch.mm(embeddings, embeddings.T) / self.temperature

        labels_col = labels.unsqueeze(1)
        labels_row = labels.unsqueeze(0)
        pos_mask = (labels_col == labels_row).float()
        diag_mask = torch.eye(n, device=device)
        pos_mask = pos_mask - diag_mask

        num_pos = pos_mask.sum(dim=1)
        valid = num_pos > 0

        if not valid.any():
            return embeddings.sum() * 0.0

        self_mask = 1.0 - diag_mask
        log_denom = torch.log(
            (self_mask * torch.exp(sim)).sum(dim=1, keepdim=True).clamp(min=1e-9)
        )

        log_prob = sim - log_denom
        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / num_pos.clamp(min=1)
        return -mean_log_prob_pos[valid].mean()
