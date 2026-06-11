"""P×K batch sampler for contrastive training.

Each yielded index list contains exactly P senders × K emails, giving every anchor
at least K-1 guaranteed positives and (P-1)×K negatives per batch.
Senders with fewer than K emails are excluded.
"""

from __future__ import annotations

import random
from collections import defaultdict

from torch.utils.data import Sampler


def _weighted_sample_no_replace(
    items: list, weights: list[float], k: int, rng: random.Random
) -> list:
    """Weighted sampling without replacement via Efraimidis-Spirakis."""
    if not items or k <= 0:
        return []
    keys = [rng.random() ** (1.0 / max(w, 1e-9)) for w in weights]
    scored = sorted(zip(keys, items), reverse=True)
    return [item for _, item in scored[:k]]


def _register_stratified_pick(
    pool: list[int],
    k: int,
    register_of: list[str],
    rng: random.Random,
) -> list[int]:
    """Pick k indices from *pool*, maximising distinct-register coverage.

    This is the LLM-free replacement for the old ``cross_register`` LLM
    positives.  Instead of asking an LLM to fake a register shift and storing the
    forgery under the real sender_id, we deliberately co-sample a sender's *real*
    formal, casual and terse emails into the same episode.  The K-1 in-episode
    positives for an anchor then span registers, so SupCon/episodic loss is
    forced to pull genuine cross-register same-author pairs together — the exact
    invariance the LLM positives were standing in for, learned from real text.

    We round-robin across the registers present in *pool* (shuffled within each),
    so the first picks cover one email per register before any register is
    sampled twice.  Falls back to plain shuffle behaviour when the sender writes
    in only one register.
    """
    by_reg: dict[str, list[int]] = defaultdict(list)
    for idx in pool:
        by_reg[register_of[idx]].append(idx)
    for lst in by_reg.values():
        rng.shuffle(lst)

    regs = list(by_reg.keys())
    rng.shuffle(regs)
    chosen: list[int] = []
    while len(chosen) < k:
        progressed = False
        for r in regs:
            if by_reg[r]:
                chosen.append(by_reg[r].pop())
                progressed = True
                if len(chosen) == k:
                    break
        if not progressed:  # pool exhausted (fewer than k emails available)
            break
    return chosen


class PKSampler(Sampler[list[int]]):
    """Sample P senders × K emails per batch, without intra-batch replacement."""

    def __init__(
        self,
        sender_ids: list[str],
        p: int,
        k: int,
        drop_last: bool = True,
        seed: int | None = None,
        register_labels: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.p = p
        self.k = k
        self.drop_last = drop_last
        # Using random.Random (not torch.Generator) so the sampler's state is
        # independent of PyTorch's global RNG and doesn't interfere with model
        # weight initialization or data augmentation.
        self._rng = random.Random(seed)
        # Optional per-index register label ("formal"/"casual"/"terse"), aligned
        # to sender_ids. When present, each sender's K emails are picked to cover
        # as many registers as possible (register-stratified episodes).
        self._register_of = register_labels
        if register_labels is not None and len(register_labels) != len(sender_ids):
            raise ValueError(
                f"register_labels length ({len(register_labels)}) must match "
                f"sender_ids length ({len(sender_ids)})."
            )

        self._sender_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx, sid in enumerate(sender_ids):
            self._sender_to_indices[sid].append(idx)

        self._eligible_senders: list[str] = [
            sid
            for sid, indices in self._sender_to_indices.items()
            if len(indices) >= k
        ]

        # Hard fail early rather than producing silent empty batches.
        if len(self._eligible_senders) < p:
            raise ValueError(
                f"PKSampler requires at least p={p} senders with >= k={k} emails each; "
                f"only {len(self._eligible_senders)} qualify."
            )

        self._hard_pairs: list[tuple[str, str]] = []
        self._hard_fraction: float = 0.0

    def set_hard_pairs(
        self, pairs: list[tuple[str, str]], hard_fraction: float = 0.5
    ) -> None:
        """Update confusable pairs for hard-batch construction (called each epoch by Trainer)."""
        eligible_set = set(self._eligible_senders)
        self._hard_pairs = [(a, b) for a, b in pairs if a in eligible_set and b in eligible_set]
        self._hard_fraction = hard_fraction

    def _build_batch_indices(self, batch_senders: list[str]) -> list[int]:
        indices: list[int] = []
        for sid in batch_senders:
            pool = list(self._sender_to_indices[sid])
            if self._register_of is not None:
                indices.extend(
                    _register_stratified_pick(pool, self.k, self._register_of, self._rng)
                )
            else:
                self._rng.shuffle(pool)
                indices.extend(pool[: self.k])
        return indices

    def __iter__(self):
        senders = list(self._eligible_senders)
        self._rng.shuffle(senders)

        if not self._hard_pairs:
            n_batches = len(senders) // self.p
            if not self.drop_last and len(senders) % self.p:
                n_batches += 1
            for batch_idx in range(n_batches):
                batch_senders = senders[batch_idx * self.p : (batch_idx + 1) * self.p]
                if len(batch_senders) < self.p:
                    continue
                yield self._build_batch_indices(batch_senders)
            return

        # Hard-batch mode: explicitly co-sample confusable pairs.
        n_total = len(senders) // self.p
        n_hard = max(1, int(n_total * self._hard_fraction))

        pairs_shuffled = list(self._hard_pairs)
        self._rng.shuffle(pairs_shuffled)

        used: set[str] = set()
        hard_batches: list[list[str]] = []

        for a, b in pairs_shuffled:
            if len(hard_batches) >= n_hard:
                break
            if a in used or b in used:
                continue
            fillers = [s for s in senders if s not in used and s != a and s != b]
            if len(fillers) < self.p - 2:
                continue
            self._rng.shuffle(fillers)
            batch = [a, b] + fillers[: self.p - 2]
            hard_batches.append(batch)
            used.update(batch)

        remaining = [s for s in senders if s not in used]
        n_random = len(remaining) // self.p
        random_batches = [
            remaining[i * self.p : (i + 1) * self.p] for i in range(n_random)
        ]

        all_batches = hard_batches + random_batches
        self._rng.shuffle(all_batches)

        for batch_senders in all_batches:
            if len(batch_senders) < self.p:
                continue
            yield self._build_batch_indices(batch_senders)

    def __len__(self) -> int:
        return len(self._eligible_senders) // self.p

_SYN_SUFFIX = "__syn"


class SyntheticBalancedSampler(Sampler[list[int]]):
    """PKSampler that guarantees n_syn synthetic–real sender pairs per batch.

    Each batch: n_syn synthetic senders + n_syn real counterparts + (P - 2*n_syn) fillers.
    Synthetic senders must end with "__syn"; their real counterpart must also be present.
    """

    def __init__(
        self,
        sender_ids: list[str],
        p: int,
        k: int,
        n_syn: int = 2,
        drop_last: bool = True,
        seed: int | None = None,
        register_labels: list[str] | None = None,
    ) -> None:
        super().__init__()
        if 2 * n_syn > p:
            raise ValueError(
                f"2 * n_syn ({2 * n_syn}) must be <= p ({p}); "
                "each synthetic sender occupies one slot and its real counterpart another."
            )
        self.p = p
        self.k = k
        self.n_syn = n_syn
        self.drop_last = drop_last
        self._rng = random.Random(seed)
        # Per-index register label, aligned to sender_ids; drives register-
        # stratified K-selection (real cross-register positives). See
        # _register_stratified_pick.
        self._register_of = register_labels
        if register_labels is not None and len(register_labels) != len(sender_ids):
            raise ValueError(
                f"register_labels length ({len(register_labels)}) must match "
                f"sender_ids length ({len(sender_ids)})."
            )

        sender_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx, sid in enumerate(sender_ids):
            sender_to_indices[sid].append(idx)

        def _eligible(sid: str) -> bool:
            return len(sender_to_indices[sid]) >= k

        syn_senders = {sid for sid in sender_to_indices if sid.endswith(_SYN_SUFFIX)}
        real_senders = {sid for sid in sender_to_indices if not sid.endswith(_SYN_SUFFIX)}

        self._eligible_pairs: list[tuple[str, str]] = []
        for syn_sid in syn_senders:
            real_sid = syn_sid[: -len(_SYN_SUFFIX)]
            if real_sid in real_senders and _eligible(real_sid) and _eligible(syn_sid):
                self._eligible_pairs.append((real_sid, syn_sid))

        paired_real = {real for real, _ in self._eligible_pairs}
        self._eligible_real_only: list[str] = [
            sid for sid in real_senders if sid not in paired_real and _eligible(sid)
        ]
        # When real-only senders run out, paired senders can fill the remaining slots
        # solo (without their __syn twin). Tracked separately for W&B telemetry.
        self._eligible_paired_real: list[str] = sorted(paired_real)

        slots_real_only = p - 2 * n_syn
        if len(self._eligible_pairs) < n_syn:
            raise ValueError(
                f"SyntheticBalancedSampler requires at least n_syn={n_syn} eligible "
                f"synthetic–real pairs; only {len(self._eligible_pairs)} qualify "
                f"(both sender and its __syn counterpart need >= k={k} emails)."
            )
        # Total filler-eligible senders = real-only + paired-real (used as solo fallback).
        # Each batch consumes n_syn pairs, so up to n_syn paired senders are already
        # spoken for and cannot also serve as fillers in that batch. We need enough
        # remaining capacity to fill slots_real_only.
        total_filler_pool = len(self._eligible_real_only) + len(self._eligible_paired_real)
        if total_filler_pool < slots_real_only + n_syn:
            raise ValueError(
                f"SyntheticBalancedSampler needs at least {slots_real_only + n_syn} "
                f"filler-eligible senders (slots_real_only={slots_real_only} + "
                f"n_syn={n_syn} reserved as pairs); only {total_filler_pool} qualify."
            )

        self._sender_to_indices = dict(sender_to_indices)
        self._real_only_pool_size = len(self._eligible_real_only)
        self._paired_real_pool_size = len(self._eligible_paired_real)
        self._epoch_stats: dict[str, float] = {}
        self._hard_sender_weights: dict[str, float] = {}
        self._hard_fraction: float = 0.0

    def set_hard_pairs(
        self, pairs: list[tuple[str, str]], hard_fraction: float = 0.5
    ) -> None:
        """Update per-sender weights derived from confusable pairs for filler selection."""
        weights: dict[str, float] = {}
        for a, b in pairs:
            weights[a] = weights.get(a, 0.0) + 1.0
            weights[b] = weights.get(b, 0.0) + 1.0
        max_w = max(weights.values(), default=1.0)
        self._hard_sender_weights = {s: w / max_w for s, w in weights.items()}
        self._hard_fraction = hard_fraction

    def __iter__(self):
        pairs = list(self._eligible_pairs)
        self._rng.shuffle(pairs)

        slots_real_only = self.p - 2 * self.n_syn
        n_batches = len(pairs) // self.n_syn if self.n_syn > 0 else 0

        real_only_set = set(self._eligible_real_only)
        n_filler_real_only = 0
        n_filler_paired_solo = 0
        n_filler_total = 0
        from collections import Counter as _Counter
        exposure_as_pair: _Counter = _Counter()
        exposure_as_filler: _Counter = _Counter()

        for i in range(n_batches):
            batch_pairs = pairs[i * self.n_syn : (i + 1) * self.n_syn]
            chosen_pair_reals = {real for real, _ in batch_pairs}

            filler_candidates = list(self._eligible_real_only) + [
                sid for sid in self._eligible_paired_real
                if sid not in chosen_pair_reals
            ]
            if self._hard_sender_weights:
                filler_weights = [
                    self._hard_sender_weights.get(s, 1e-2) for s in filler_candidates
                ]
                batch_real = _weighted_sample_no_replace(
                    filler_candidates, filler_weights, slots_real_only, self._rng
                )
            else:
                self._rng.shuffle(filler_candidates)
                batch_real = filler_candidates[:slots_real_only]
            if len(batch_real) < slots_real_only:
                # Should be impossible given the constructor check, but guard anyway.
                break

            for sid in batch_real:
                if sid in real_only_set:
                    n_filler_real_only += 1
                else:
                    n_filler_paired_solo += 1
                exposure_as_filler[sid] += 1
            n_filler_total += len(batch_real)
            for real_sid, _syn_sid in batch_pairs:
                exposure_as_pair[real_sid] += 1

            batch_senders: list[str] = []
            for real_sid, syn_sid in batch_pairs:
                batch_senders.extend([real_sid, syn_sid])
            batch_senders.extend(batch_real)

            indices: list[int] = []
            for sid in batch_senders:
                pool = list(self._sender_to_indices[sid])
                if self._register_of is not None:
                    indices.extend(
                        _register_stratified_pick(pool, self.k, self._register_of, self._rng)
                    )
                else:
                    self._rng.shuffle(pool)
                    indices.extend(pool[: self.k])

            yield indices

        fallback_frac = n_filler_paired_solo / n_filler_total if n_filler_total else 0.0
        pair_only_senders = sum(
            1 for s in exposure_as_pair if exposure_as_filler.get(s, 0) == 0
        )
        self._epoch_stats = {
            "train/sampler/n_batches": float(n_batches),
            "train/sampler/n_filler_real_only": float(n_filler_real_only),
            "train/sampler/n_filler_paired_solo": float(n_filler_paired_solo),
            "train/sampler/filler_fallback_fraction": float(fallback_frac),
            "train/sampler/pool_real_only": float(self._real_only_pool_size),
            "train/sampler/pool_paired_real": float(self._paired_real_pool_size),
            "train/sampler/pair_only_senders": float(pair_only_senders),
        }

    def pop_epoch_stats(self) -> dict[str, float]:
        """Return epoch composition stats and reset; called by Trainer after each epoch."""
        stats = dict(self._epoch_stats)
        self._epoch_stats = {}
        return stats

    def __len__(self) -> int:
        return len(self._eligible_pairs) // self.n_syn if self.n_syn > 0 else 0
