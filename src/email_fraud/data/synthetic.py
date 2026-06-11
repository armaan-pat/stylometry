"""SyntheticAugmentedDataset — merges real emails with LLM-generated synthetics.

The synthetic dataset is produced by scripts/generate_synthetic_emails.py, which
emits two distinct kinds of synthetic email:

hard_neg  (sender_id ends with ``__syn``)
    The LLM mimics the sender's style on an arbitrary business topic.
    Stored under "alice@enron.com__syn" so SupConLoss / PKSampler treat it as a
    *different* class from real-Alice.  SyntheticBalancedSampler guarantees one
    real/synthetic contrast pair per batch.  Training signal: separate real-Alice
    from LLM-Alice (adversarial impostor resistance).

cross_register  (sender_id == the real sender, no suffix)
    The LLM writes as the same person but in the *opposite* register — e.g. formal
    examples → casual output, or casual examples → formal output.  Stored under the
    real sender_id so the loss treats them as *positive* examples for that sender.
    Training signal: same-author embeddings should cluster even when register shifts
    hard, directly addressing the dominant false-negative failure mode (74 % of
    same-author test pairs cross topic domains).

Routing summary
    __syn  suffix present  →  ends up in SyntheticBalancedSampler's synthetic pool
    no suffix              →  merged into the real sender's positive pool for PKSampler

Usage with PKSampler (random pairing — simple):
    real_ds = EnronDataset(cfg.data, split="train")
    full_ds = SyntheticAugmentedDataset(real_ds, cfg.data.augmentation.synthetic_path)
    sampler = PKSampler(full_ds.sender_ids, p=cfg.p, k=cfg.k)

Usage with SyntheticBalancedSampler (guaranteed real/syn pairing — recommended):
    sampler = SyntheticBalancedSampler(full_ds.sender_ids, p=P, k=K, n_syn=2)
"""

from __future__ import annotations

from email_fraud.data.base import BaseDataset

SYN_SUFFIX = "__syn"


class SyntheticAugmentedDataset(BaseDataset):
    """Concatenates a real BaseDataset with an Arrow-backed synthetic dataset.

    The combined dataset exposes the same ``(text, sender_id)`` interface as
    EnronDataset so it works transparently with PKSampler and episode_collate.

    The synthetic Arrow dataset may contain both ``hard_neg`` rows (sender_id ends
    with ``__syn``) and ``cross_register`` rows (sender_id == real sender).  Both
    are loaded here; downstream samplers route them automatically based on the
    ``__syn`` suffix convention.

    Args:
        real_dataset: Any BaseDataset (typically EnronDataset for the train split).
        synthetic_path: Path to the Arrow dataset produced by
                        scripts/generate_synthetic_emails.py.
        llm_negatives_only: When True (default), DROP every synthetic row whose
                        sender_id does not end with ``__syn`` — i.e. any LLM text
                        stored under a real sender_id (cross_register / cross_length
                        "positives").  This enforces the project invariant that
                        LLM-generated text is ONLY ever a hard negative, never a
                        positive, regardless of how the on-disk dataset was
                        generated.  An LLM impersonation that lands in a real
                        author's positive cluster teaches the encoder that a
                        convincing forgery *is* the author — directly opposing the
                        fraud signal.  Set False only for a deliberate ablation.
    """

    def __init__(
        self,
        real_dataset: BaseDataset,
        synthetic_path: str,
        llm_negatives_only: bool = True,
    ) -> None:
        from datasets import load_from_disk

        syn_ds = load_from_disk(synthetic_path)
        syn_texts = list(syn_ds["text"])
        syn_sids = list(syn_ds["sender_id"])

        if llm_negatives_only:
            kept = [
                (t, s) for t, s in zip(syn_texts, syn_sids) if s.endswith(SYN_SUFFIX)
            ]
            n_dropped = len(syn_sids) - len(kept)
            if n_dropped:
                import logging

                logging.getLogger(__name__).warning(
                    "SyntheticAugmentedDataset: dropped %d/%d synthetic rows stored "
                    "under a real sender_id (LLM positives) to enforce "
                    "llm_negatives_only; kept %d hard-negative (__syn) rows. "
                    "Regenerate with --llm-positives exclude to silence this.",
                    n_dropped, len(syn_sids), len(kept),
                )
            syn_texts = [t for t, _ in kept]
            syn_sids = [s for _, s in kept]

        self._texts: list[str] = list(real_dataset._texts) + syn_texts
        self._sender_ids_list: list[str] = (
            list(real_dataset._sender_ids_list) + syn_sids
        )

    def __len__(self) -> int:
        return len(self._texts)

    def __getitem__(self, index: int) -> tuple[str, str]:
        return self._texts[index], self._sender_ids_list[index]

    @property
    def sender_ids(self) -> list[str]:
        return self._sender_ids_list
