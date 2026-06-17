"""Episodic, variable-K prototypical loss (Snell et al. NeurIPS 2017, arXiv:1703.05175).

Trains the way we infer: each batch, every sender's embeddings are split into a
*support set* of size K' ~ uniform{support_k_min..support_k_max} and a *query
set* of the remainder. Prototypes are support means; queries are classified
against all in-batch prototypes with the deployed distance (cosine), and the
loss is cross-entropy on the sender identity. Because K' is sampled small
during training, the encoder is explicitly optimized so that the mean of a few
of a sender's embeddings is already a stable, discriminative description of
them — low-K robustness baked into the representation rather than patched at
scoring time.

Synthetic hard negatives (sender_id ending in "__syn") never form prototypes —
at deployment, profiles are only ever built from genuine enrollment. They stay
in the query pool as "not-this-sender" targets: each synthetic query is pushed
*away* from its mimicked sender's prototype via -log(1 - p(mimicked sender)).
Cross-register synthetic positives carry the real sender_id, so they
participate as ordinary support/query members of that sender.

SupCon is kept as an auxiliary term so pairwise structure isn't lost:
L = L_proto + supcon_weight * L_supcon.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from email_fraud.losses.base import BaseLoss
from email_fraud.losses.supcon import SupConLoss
from email_fraud.registry import register

_SYN_SUFFIX = "__syn"


@register("loss", "episodic")
class EpisodicPrototypeLoss(BaseLoss):
    """Prototypical-network episode loss with variable support size K'.

    Args:
        temperature:   softmax temperature over query→prototype cosine
                       similarities (also passed to the SupCon aux term).
        support_k_min: smallest support size sampled per sender per batch.
        support_k_max: largest support size sampled (capped per sender at
                       n_emails - min_queries so at least min_queries remain).
        supcon_weight: weight of the auxiliary SupCon term (0 disables it).
        min_queries:   minimum queries left after the support split.
    """

    def __init__(
        self,
        temperature: float = 0.05,
        support_k_min: int = 2,
        support_k_max: int = 6,
        supcon_weight: float = 0.5,
        min_queries: int = 1,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if support_k_min < 1:
            raise ValueError(f"support_k_min must be >= 1, got {support_k_min}")
        if support_k_max < support_k_min:
            raise ValueError(
                f"support_k_max ({support_k_max}) must be >= support_k_min ({support_k_min})"
            )
        if min_queries < 1:
            raise ValueError(f"min_queries must be >= 1, got {min_queries}")
        self.temperature = temperature
        self.support_k_min = support_k_min
        self.support_k_max = support_k_max
        self.supcon_weight = supcon_weight
        self.min_queries = min_queries
        self._supcon = SupConLoss(temperature=temperature) if supcon_weight > 0 else None

    @property
    def requires_pk_sampler(self) -> bool:
        # Each sender needs >= support_k_min + min_queries embeddings in-batch.
        return True

    @property
    def requires_sender_ids(self) -> bool:
        # Needed to route __syn embeddings into the repulsion-only query pool.
        return True

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        sender_ids: list[str] | None = None,
    ) -> torch.Tensor:
        """Args: embeddings (N, d) L2-normalized, labels (N,) int sender ids,
        sender_ids optional raw id strings (None → all treated as real)."""
        device = embeddings.device
        n = embeddings.size(0)

        # Which batch-local classes are synthetic hard negatives?
        syn_classes: set[int] = set()
        if sender_ids is not None:
            for lbl, sid in zip(labels.tolist(), sender_ids):
                if sid.endswith(_SYN_SUFFIX):
                    syn_classes.add(lbl)

        # --- Episode construction: support/query split per real sender -----
        prototypes: list[torch.Tensor] = []
        class_to_proto_row: dict[int, int] = {}
        query_idx: list[int] = []        # real queries
        query_target: list[int] = []     # prototype row of their own sender
        syn_query_idx: list[int] = []    # synthetic queries (repulsion only)
        syn_query_class: list[int] = []  # batch-local class of the mimicked sender

        for cls in labels.unique().tolist():
            cls_idx = (labels == cls).nonzero(as_tuple=True)[0]
            if cls in syn_classes:
                syn_query_idx.extend(cls_idx.tolist())
                continue
            n_c = cls_idx.numel()
            if n_c < self.support_k_min + self.min_queries:
                # Too few embeddings for an episode; SupCon aux still sees them.
                continue
            k_hi = min(self.support_k_max, n_c - self.min_queries)
            k_prime = int(
                torch.randint(self.support_k_min, k_hi + 1, (1,)).item()
            )
            perm = cls_idx[torch.randperm(n_c, device=device)]
            support, queries = perm[:k_prime], perm[k_prime:]
            class_to_proto_row[cls] = len(prototypes)
            prototypes.append(embeddings[support].mean(dim=0))
            query_idx.extend(queries.tolist())
            query_target.extend([class_to_proto_row[cls]] * queries.numel())

        # Map each synthetic query to its mimicked sender's prototype (if present).
        if syn_query_idx and sender_ids is not None and class_to_proto_row:
            sender_to_class = {sid: int(lbl) for sid, lbl in zip(sender_ids, labels.tolist())}
            kept_idx: list[int] = []
            for i in syn_query_idx:
                real_sid = sender_ids[i][: -len(_SYN_SUFFIX)]
                real_cls = sender_to_class.get(real_sid)
                if real_cls is not None and real_cls in class_to_proto_row:
                    kept_idx.append(i)
                    syn_query_class.append(class_to_proto_row[real_cls])
            syn_query_idx = kept_idx
        else:
            syn_query_idx = []

        # --- Loss terms -----------------------------------------------------
        terms: list[torch.Tensor] = []
        if prototypes and (query_idx or syn_query_idx):
            # Cosine to the (renormalized) support mean == cosine to the
            # deployed centroid, since embeddings are L2-normalized.
            protos = F.normalize(torch.stack(prototypes), p=2, dim=-1)

            if query_idx:
                q = embeddings[torch.tensor(query_idx, device=device)]
                logits = (q @ protos.T) / self.temperature
                targets = torch.tensor(query_target, device=device)
                terms.append(F.cross_entropy(logits, targets, reduction="none"))

            if syn_query_idx:
                q_syn = embeddings[torch.tensor(syn_query_idx, device=device)]
                p_syn = F.softmax((q_syn @ protos.T) / self.temperature, dim=-1)
                rows = torch.arange(len(syn_query_idx), device=device)
                cols = torch.tensor(syn_query_class, device=device)
                p_mimic = p_syn[rows, cols].clamp(max=1.0 - 1e-6)
                terms.append(-torch.log1p(-p_mimic))

        if terms:
            proto_loss = torch.cat(terms).mean()
        else:
            # No class big enough for an episode this batch — zero with grad.
            proto_loss = embeddings.sum() * 0.0

        if self._supcon is not None:
            return proto_loss + self.supcon_weight * self._supcon(embeddings, labels)
        return proto_loss
