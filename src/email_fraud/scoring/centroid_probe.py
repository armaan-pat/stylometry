"""CentroidProbe — fixed enrollment/query probe set for inference-style validation.

Mimics the deployment scenario:
    1. Each profiled sender is enrolled with N held-out emails → centroid.
    2. Queries are scored against centroids via PrototypicalHead z-score.
    3. AUROCs are computed for three discrimination tasks:
         - genuine  vs other-sender  (easy — different person entirely)
         - genuine  vs synthetic     (hard — same person's style, LLM-written)
         - genuine  vs impostor-pool (genuine vs other ∪ synthetic)

This is built once before training (texts are fixed; only the encoder changes
between epochs) and re-evaluated each validation by re-encoding everything
through the *current* encoder weights.

Why profile the train senders rather than test senders?
-------------------------------------------------------
The encoder weights have learned the train senders' style during training; at
deployment time the encoder is frozen and used to enroll *new* senders by
averaging a handful of embeddings — there is no further fine-tuning.  Profiling
train senders here measures: given that the encoder generalises stylometric
signal, how well do the resulting centroids separate genuine emails from
{other senders, LLM imitations} for *the senders we know about*.  The
"never-seen sender" generalisation question is answered separately by the
sender-disjoint pair-cosine eval on the test split (see scripts/evaluate.py).
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch

from email_fraud.heads.prototypical import PrototypicalHead

logger = logging.getLogger(__name__)


@dataclass
class _ProbeData:
    enrollment_texts: list[str]
    enrollment_senders: list[str]
    genuine_texts: list[str]
    genuine_senders: list[str]
    other_texts: list[str]            # impostors from senders NOT in profile pool
    synthetic_texts: list[str]
    synthetic_source_senders: list[str]


class CentroidProbe:
    """Fixed probe set for inference-style centroid evaluation.

    Args:
        n_profile_senders:    Senders to profile from the training pool.
        n_enroll_per_sender:  Emails reserved per sender for centroid enrollment.
        n_query_per_sender:   Genuine queries per sender (held-out from same person).
        n_other_queries:      Total impostor queries drawn from non-profiled senders.
        n_synthetic_queries:  Total synthetic queries (capped by availability).
        confidence_tiers:     Passed through to PrototypicalHead.
        seed:                 RNG seed for reproducible probe sampling.
    """

    def __init__(
        self,
        train_texts: list[str],
        train_senders: list[str],
        other_texts: list[str],
        other_senders: list[str],
        synthetic_texts: list[str] | None = None,
        synthetic_source_senders: list[str] | None = None,
        n_profile_senders: int = 30,
        n_enroll_per_sender: int = 8,
        n_query_per_sender: int = 4,
        n_other_queries: int = 200,
        n_synthetic_queries: int = 200,
        confidence_tiers: dict[str, str] | None = None,
        seed: int = 0,
    ) -> None:
        self.confidence_tiers = confidence_tiers
        self._seed = seed
        rng = random.Random(seed)

        sender_to_texts: dict[str, list[str]] = defaultdict(list)
        for t, s in zip(train_texts, train_senders):
            sender_to_texts[s].append(t)

        min_needed = n_enroll_per_sender + n_query_per_sender
        eligible = [s for s, ts in sender_to_texts.items() if len(ts) >= min_needed]
        if len(eligible) < n_profile_senders:
            logger.warning(
                "CentroidProbe: only %d senders have >= %d emails "
                "(needed %d); using all eligible.",
                len(eligible), min_needed, n_profile_senders,
            )
            n_profile_senders = len(eligible)
        chosen = rng.sample(eligible, n_profile_senders)

        enr_texts: list[str] = []
        enr_senders: list[str] = []
        gen_texts: list[str] = []
        gen_senders: list[str] = []
        for sid in chosen:
            ts = list(sender_to_texts[sid])
            rng.shuffle(ts)
            enr_texts.extend(ts[:n_enroll_per_sender])
            enr_senders.extend([sid] * n_enroll_per_sender)
            gen_texts.extend(ts[n_enroll_per_sender : n_enroll_per_sender + n_query_per_sender])
            gen_senders.extend([sid] * n_query_per_sender)

        # Impostor texts: drawn from the other-pool (typically validation senders)
        # so the encoder hasn't memorised these specific texts during training.
        if len(other_texts) == 0:
            logger.warning("CentroidProbe: empty other-pool — impostor probes will be skipped.")
            other_idx: list[int] = []
        else:
            n_other_pick = min(n_other_queries, len(other_texts))
            other_idx = rng.sample(range(len(other_texts)), n_other_pick)
        other_q_texts = [other_texts[i] for i in other_idx]

        # Synthetic queries: only those whose source_sender_id is in the profile pool.
        syn_q_texts: list[str] = []
        syn_q_sources: list[str] = []
        if synthetic_texts and synthetic_source_senders:
            profile_set = set(chosen)
            pairs = [
                (t, s) for t, s in zip(synthetic_texts, synthetic_source_senders)
                if s in profile_set
            ]
            if not pairs:
                logger.warning(
                    "CentroidProbe: no synthetic emails match any profiled sender."
                )
            else:
                rng.shuffle(pairs)
                for t, s in pairs[:n_synthetic_queries]:
                    syn_q_texts.append(t)
                    syn_q_sources.append(s)

        self._data = _ProbeData(
            enrollment_texts=enr_texts,
            enrollment_senders=enr_senders,
            genuine_texts=gen_texts,
            genuine_senders=gen_senders,
            other_texts=other_q_texts,
            synthetic_texts=syn_q_texts,
            synthetic_source_senders=syn_q_sources,
        )
        self._profile_senders = chosen

        logger.info(
            "CentroidProbe ready: %d profiles × %d enroll  |  %d genuine, "
            "%d impostor, %d synthetic queries",
            len(chosen), n_enroll_per_sender,
            len(gen_texts), len(other_q_texts), len(syn_q_texts),
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, encoder, device: str, batch_size: int = 32) -> dict[str, float]:
        """Re-encode the fixed probe set with current encoder weights.

        Returns a dict of metrics keyed for direct W&B logging. Keys use top-level
        prefixes per concept so W&B's sidebar splits them into separate panels:

            auc/genuine_vs_other      AUROC: real emails vs random-other-sender
            auc/genuine_vs_synthetic  AUROC: real emails vs LLM imitation (hard)
            auc/genuine_vs_all        AUROC: real vs pooled impostors

            score/mean_genuine        Mean centroid score per pool
            score/mean_other
            score/mean_synthetic
            score/gap_other           mean_genuine - mean_other
            score/gap_synthetic       mean_genuine - mean_synthetic
            score/synthetic_harder_than_other  gap_other - gap_synthetic
              positive ⇒ synthetics are harder than other-sender emails (good)
              negative ⇒ synthetics are EASIER (LLM artifacts dominate, suspicious)

            probe/n_genuine_queries   Diagnostic query counts
            probe/n_other_queries
            probe/n_synthetic_queries

        Threshold-band metrics live under threshold_{τ}/ and coverage metrics
        under coverage/, populated by the helpers below.
        """
        from sklearn.metrics import roc_auc_score

        was_training = encoder.training
        encoder.eval()

        d = self._data
        enrol_emb = _encode(encoder, d.enrollment_texts, device, batch_size)
        gen_emb = _encode(encoder, d.genuine_texts, device, batch_size)
        oth_emb = _encode(encoder, d.other_texts, device, batch_size) if d.other_texts else None
        syn_emb = _encode(encoder, d.synthetic_texts, device, batch_size) if d.synthetic_texts else None

        head = PrototypicalHead(confidence_tiers=self.confidence_tiers)
        head.fit(enrol_emb, d.enrollment_senders)

        # Genuine: score each held-out email against its true sender's centroid.
        genuine_scores = np.array([
            head.score(emb, sid)["score"]
            for emb, sid in zip(gen_emb, d.genuine_senders)
        ])
        # Other-sender impostors: score each impostor email against a *random*
        # profiled sender (we know they're not that person — splits are
        # sender-disjoint).  This is the easy negative.
        rng = random.Random(self._seed)
        other_scores = np.array([])
        if oth_emb is not None and len(oth_emb) > 0:
            assigned = [rng.choice(self._profile_senders) for _ in range(len(oth_emb))]
            other_scores = np.array([
                head.score(emb, sid)["score"] for emb, sid in zip(oth_emb, assigned)
            ])
        # Synthetic hard negatives: score against the *real* source sender's
        # centroid (the LLM was prompted to imitate that person).  The model
        # passes if it scores synthetic-Alice low against Alice's centroid.
        syn_scores = np.array([])
        if syn_emb is not None and len(syn_emb) > 0:
            syn_scores = np.array([
                head.score(emb, sid)["score"]
                for emb, sid in zip(syn_emb, d.synthetic_source_senders)
            ])

        if was_training:
            encoder.train()

        out: dict[str, float] = {
            "score/mean_genuine": float(genuine_scores.mean()) if len(genuine_scores) else 0.0,
            "score/mean_other": float(other_scores.mean()) if len(other_scores) else 0.0,
            "score/mean_synthetic": float(syn_scores.mean()) if len(syn_scores) else 0.0,
            "probe/n_genuine_queries": float(len(genuine_scores)),
            "probe/n_other_queries": float(len(other_scores)),
            "probe/n_synthetic_queries": float(len(syn_scores)),
        }

        # AUROC: genuine (label=1) vs other (label=0). Higher score = more genuine.
        if len(genuine_scores) and len(other_scores):
            labels = np.concatenate([np.ones_like(genuine_scores), np.zeros_like(other_scores)])
            scores = np.concatenate([genuine_scores, other_scores])
            out["auc/genuine_vs_other"] = float(roc_auc_score(labels, scores))
            out["score/gap_other"] = float(genuine_scores.mean() - other_scores.mean())

        if len(genuine_scores) and len(syn_scores):
            labels = np.concatenate([np.ones_like(genuine_scores), np.zeros_like(syn_scores)])
            scores = np.concatenate([genuine_scores, syn_scores])
            out["auc/genuine_vs_synthetic"] = float(roc_auc_score(labels, scores))
            out["score/gap_synthetic"] = float(genuine_scores.mean() - syn_scores.mean())

        # Combined: genuine vs (other ∪ synthetic) — the "all impostors" pool.
        if len(genuine_scores) and (len(other_scores) or len(syn_scores)):
            neg = np.concatenate([other_scores, syn_scores])
            labels = np.concatenate([np.ones_like(genuine_scores), np.zeros_like(neg)])
            scores = np.concatenate([genuine_scores, neg])
            out["auc/genuine_vs_all"] = float(roc_auc_score(labels, scores))

        # Difficulty differential: positive ⇒ synthetic is harder than other-sender,
        # which is what we want from a useful hard-negative augmentation.
        if "score/gap_other" in out and "score/gap_synthetic" in out:
            out["score/synthetic_harder_than_other"] = (
                out["score/gap_other"] - out["score/gap_synthetic"]
            )

        # Operating-point reports: at score thresholds 0.5 / 0.8 / 0.95, how
        # the model behaves when used as a "report iff score > τ" verdict
        # system.  Higher τ ⇒ more confident report, lower coverage.
        out.update(_threshold_band_metrics(genuine_scores, other_scores, syn_scores))

        # Coverage at accuracy targets: largest fraction of the query stream we
        # can "report" on (top-k by confidence) while keeping accuracy ≥ target.
        out.update(_coverage_at_accuracy(genuine_scores, other_scores, syn_scores))

        return out


_THRESHOLDS = (0.5, 0.8, 0.95)


def _threshold_band_metrics(
    genuine: np.ndarray,
    other: np.ndarray,
    synthetic: np.ndarray,
) -> dict[str, float]:
    """Per-threshold report/precision/recall/FPR breakdown.

    For each τ in {0.5, 0.8, 0.95} a positive prediction is "score > τ"
    (the model commits a verdict that the email is genuine).  We log:

        report_rate@τ        fraction of all queries the system reports on
        precision@τ          P(true genuine | reported)
        recall@τ             P(reported | true genuine)            == TPR
        fpr_other@τ          P(reported | other-sender impostor)
        fpr_synthetic@τ      P(reported | synthetic impostor)
        fpr_overall@τ        P(reported | any impostor)

    A useful operating point has high precision, non-trivial recall, and
    low fpr_synthetic@τ at high τ — that's where the boss's >0.95 band
    matters most: if synthetics still slip through at >0.95 confidence
    the model is not fraud-resistant in production.
    """
    out: dict[str, float] = {}
    impostors = np.concatenate([other, synthetic]) if (len(other) or len(synthetic)) else np.array([])

    for tau in _THRESHOLDS:
        # Zero-padded so panel sections sort naturally: threshold_0.50, _0.80, _0.95.
        # (Previous "0.5"/"0.8"/"0.95" sorted lexicographically and looked random.)
        group = f"threshold_{tau:.2f}"

        # report_rate over the full query stream (genuine + impostors).
        all_scores = np.concatenate([genuine, impostors]) if len(impostors) else genuine
        if len(all_scores):
            out[f"{group}/report_rate"] = float((all_scores > tau).mean())

        # Fraction of each pool above the threshold — the boss's "report >X%" view.
        # (Dropped the duplicate "genuine_above@τ" key: it was identical to recall.)
        if len(genuine):
            out[f"{group}/recall"] = float((genuine > tau).mean())
        if len(other):
            out[f"{group}/fpr_other"] = float((other > tau).mean())
        if len(synthetic):
            out[f"{group}/fpr_synthetic"] = float((synthetic > tau).mean())
        if len(impostors):
            out[f"{group}/fpr_overall"] = float((impostors > tau).mean())

        # Precision against the pooled impostor set.
        if len(genuine) and len(impostors):
            tp = float((genuine > tau).sum())
            fp = float((impostors > tau).sum())
            out[f"{group}/precision"] = (
                tp / (tp + fp) if (tp + fp) > 0 else 0.0
            )

        # Accuracy treating "score > τ" as the genuine prediction.
        if len(genuine) and len(impostors):
            n_g = len(genuine)
            n_i = len(impostors)
            correct = float((genuine > tau).sum()) + float((impostors <= tau).sum())
            out[f"{group}/accuracy"] = correct / (n_g + n_i)

    return out


_ACCURACY_TARGETS = (0.5, 0.8, 0.95)


def _coverage_at_accuracy(
    genuine: np.ndarray,
    other: np.ndarray,
    synthetic: np.ndarray,
) -> dict[str, float]:
    """Selective-classifier coverage at fixed accuracy targets.

    Treat the head's score as a confidence signal.  Convert it to a confidence
    magnitude c = |score - 0.5| (distance from the indecision midpoint) and a
    prediction p = (score > 0.5) → 1 else 0.  Sort queries by confidence
    descending; for each prefix length k define:

        coverage(k) = k / N
        accuracy(k) = (# correct in top-k) / k

    For each target T in {0.5, 0.8, 0.95} we report:

        coverage_at_acc@T  = max coverage with accuracy ≥ T
                             (NaN if no prefix achieves the target)

    This is the question "what fraction of decisions can we make while keeping
    accuracy ≥ T?" — distinct from the threshold view above, which fixes τ on
    the raw score instead of on confidence.
    """
    out: dict[str, float] = {}
    impostors = np.concatenate([other, synthetic]) if (len(other) or len(synthetic)) else np.array([])
    if len(genuine) == 0 or len(impostors) == 0:
        return out

    scores = np.concatenate([genuine, impostors])
    truth = np.concatenate([
        np.ones_like(genuine, dtype=np.int64),
        np.zeros_like(impostors, dtype=np.int64),
    ])
    pred = (scores > 0.5).astype(np.int64)
    confidence = np.abs(scores - 0.5)
    correct = (pred == truth).astype(np.int64)

    order = np.argsort(-confidence)             # most confident first
    correct_sorted = correct[order]
    cum_correct = np.cumsum(correct_sorted)
    ks = np.arange(1, len(correct_sorted) + 1)
    running_acc = cum_correct / ks
    coverage = ks / len(correct_sorted)

    for target in _ACCURACY_TARGETS:
        # Zero-padded so coverage/at_acc_0.50 sorts before _0.80 and _0.95.
        key = f"coverage/at_acc_{target:.2f}"
        mask = running_acc >= target
        out[key] = float(coverage[mask].max()) if mask.any() else float("nan")

    return out


def _encode(encoder, texts: list[str], device: str, batch_size: int) -> torch.Tensor:
    """Tokenize + forward a list of texts through the encoder; return CPU embeddings.

    For luar_episode encoders we want one embedding per text (the probe iterates
    text-by-text), so we override episode_k=1 for the duration of probe encoding.
    Training-time episode_k is restored afterward.
    """
    episode_k = getattr(encoder, "episode_k", None)
    saved_k: int | None = None
    if episode_k is not None:
        saved_k = encoder.config.episode_k
        encoder.config.episode_k = 1
    try:
        out: list[torch.Tensor] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            tok = encoder.tokenize(batch)
            tok = {k: v.to(device) for k, v in tok.items()}
            embs = encoder.encode(**tok)
            out.append(embs.detach().cpu())
        return torch.cat(out, dim=0) if out else torch.empty(0)
    finally:
        if saved_k is not None:
            encoder.config.episode_k = saved_k
