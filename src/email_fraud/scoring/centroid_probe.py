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
        score_fns: list[str] | None = None,
        seed: int = 0,
    ) -> None:
        from email_fraud.scoring.score_functions import (
            DEFAULT_SCORE_FNS,
            SCORE_FNS,
        )

        self.confidence_tiers = confidence_tiers
        self._seed = seed
        # List of score functions to evaluate every step. The first one wins
        # the un-prefixed metric names (auc/genuine_vs_all, etc.) so existing
        # dashboards keep working unchanged.
        fns = list(score_fns) if score_fns else list(DEFAULT_SCORE_FNS)
        unknown = [f for f in fns if f not in SCORE_FNS]
        if unknown:
            raise KeyError(
                f"Unknown score_fns {unknown}. Available: {sorted(SCORE_FNS)}"
            )
        self._score_fns = fns
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
        from email_fraud.scoring.score_functions import resolve as _resolve_score_fn

        was_training = encoder.training
        encoder.eval()

        d = self._data
        enrol_emb = _encode(encoder, d.enrollment_texts, device, batch_size)
        gen_emb = _encode(encoder, d.genuine_texts, device, batch_size)
        oth_emb = _encode(encoder, d.other_texts, device, batch_size) if d.other_texts else None
        syn_emb = _encode(encoder, d.synthetic_texts, device, batch_size) if d.synthetic_texts else None

        head = PrototypicalHead(confidence_tiers=self.confidence_tiers)
        head.fit(enrol_emb, d.enrollment_senders)

        # Pull raw (cos_sim, spread) per query once; each score_fn is applied
        # in a tight inner loop below so we don't pay the encoding cost N times.
        def _raw_pairs(embs, sender_iter):
            out = []
            for emb, sid in zip(embs, sender_iter):
                r = head.score_raw(emb, sid)
                out.append((float(r["cos_sim"]), float(r["spread"])))
            return out

        gen_raw = _raw_pairs(gen_emb, d.genuine_senders)

        rng = random.Random(self._seed)
        oth_raw: list[tuple[float, float]] = []
        if oth_emb is not None and len(oth_emb) > 0:
            assigned = [rng.choice(self._profile_senders) for _ in range(len(oth_emb))]
            oth_raw = _raw_pairs(oth_emb, assigned)

        syn_raw: list[tuple[float, float]] = []
        if syn_emb is not None and len(syn_emb) > 0:
            syn_raw = _raw_pairs(syn_emb, d.synthetic_source_senders)

        if was_training:
            encoder.train()

        # Stash raw geometry so the trainer can dump it for offline score-fn
        # experimentation in scripts/analyze_thresholds.py.
        self._last_raw = {
            "genuine": gen_raw,
            "other": oth_raw,
            "synthetic": syn_raw,
            "score_fns_evaluated": list(self._score_fns),
        }

        # Apply each configured score function. The first one gets the
        # un-prefixed metric names so legacy dashboards keep working; the rest
        # get f"{fn}/" prefixes.
        out: dict[str, float] = {}
        per_fn_scores: dict[str, dict[str, np.ndarray]] = {}
        for i, fn_name in enumerate(self._score_fns):
            fn = _resolve_score_fn(fn_name)
            g = np.array([fn(c, s) for c, s in gen_raw])
            o = np.array([fn(c, s) for c, s in oth_raw])
            sn = np.array([fn(c, s) for c, s in syn_raw])
            per_fn_scores[fn_name] = {"genuine": g, "other": o, "synthetic": sn}
            prefix = "" if i == 0 else f"{fn_name}/"
            out.update(_metrics_for_score_set(g, o, sn, prefix=prefix))

        # Keep the canonical (first score_fn) arrays available to the trainer's
        # final dump for backward compatibility with scores_final.json consumers.
        canonical = per_fn_scores[self._score_fns[0]]
        genuine_scores = canonical["genuine"]
        other_scores = canonical["other"]
        syn_scores = canonical["synthetic"]

        out["probe/n_genuine_queries"] = float(len(genuine_scores))
        out["probe/n_other_queries"] = float(len(other_scores))
        out["probe/n_synthetic_queries"] = float(len(syn_scores))

        # Stash canonical-fn raw scores (legacy consumers) + per-fn scores for
        # offline replay. analyze_thresholds.py reads scores_per_fn if present.
        self._last_scores = {
            "genuine": genuine_scores.tolist(),
            "other": other_scores.tolist(),
            "synthetic": syn_scores.tolist(),
            "score_fn": self._score_fns[0],
            "scores_per_fn": {
                name: {k: v.tolist() for k, v in pool.items()}
                for name, pool in per_fn_scores.items()
            },
        }

        return out

    @torch.no_grad()
    def diagnose(self, encoder, device: str, batch_size: int = 32) -> list[dict]:
        """Return per-query records for failure-mode analysis.

        Each record contains: type (genuine/other/synthetic), label (1/0),
        actual_sender, target_sender (centroid scored against), text,
        word_count, char_count, score, tier, abstain, cos_sim, spread, k.
        Save to CSV and pass to scripts/analyze_failures.py.
        """
        from email_fraud.scoring.score_functions import resolve as _resolve_fn

        was_training = encoder.training
        encoder.eval()
        d = self._data

        enrol_emb = _encode(encoder, d.enrollment_texts, device, batch_size)
        gen_emb   = _encode(encoder, d.genuine_texts,    device, batch_size)
        oth_emb   = _encode(encoder, d.other_texts,      device, batch_size) if d.other_texts   else None
        syn_emb   = _encode(encoder, d.synthetic_texts,  device, batch_size) if d.synthetic_texts else None

        head = PrototypicalHead(confidence_tiers=self.confidence_tiers)
        head.fit(enrol_emb, d.enrollment_senders)

        score_fn = _resolve_fn(self._score_fns[0])
        rng = random.Random(self._seed)

        records: list[dict] = []

        for emb, text, sid in zip(gen_emb, d.genuine_texts, d.genuine_senders):
            raw = head.score_raw(emb, sid)
            cos_sim = float(raw["cos_sim"])
            spread  = float(raw["spread"])
            score   = score_fn(cos_sim, spread)
            records.append({
                "type": "genuine", "label": 1,
                "actual_sender": sid, "target_sender": sid,
                "text": text, "word_count": len(text.split()), "char_count": len(text),
                "score": score, "tier": raw["tier"], "abstain": raw["abstain"],
                "cos_sim": cos_sim, "spread": spread, "k": head._profiles.get(sid, {}).get("k"),
            })

        if oth_emb is not None and len(oth_emb) > 0:
            assigned = [rng.choice(self._profile_senders) for _ in range(len(oth_emb))]
            for emb, text, target in zip(oth_emb, d.other_texts, assigned):
                raw = head.score_raw(emb, target)
                cos_sim = float(raw["cos_sim"])
                spread  = float(raw["spread"])
                score   = score_fn(cos_sim, spread)
                records.append({
                    "type": "other", "label": 0,
                    "actual_sender": "?", "target_sender": target,
                    "text": text, "word_count": len(text.split()), "char_count": len(text),
                    "score": score, "tier": raw["tier"], "abstain": raw["abstain"],
                    "cos_sim": cos_sim, "spread": spread, "k": head._profiles.get(target, {}).get("k"),
                })

        if syn_emb is not None and len(syn_emb) > 0:
            for emb, text, sid in zip(syn_emb, d.synthetic_texts, d.synthetic_source_senders):
                raw = head.score_raw(emb, sid)
                cos_sim = float(raw["cos_sim"])
                spread  = float(raw["spread"])
                score   = score_fn(cos_sim, spread)
                records.append({
                    "type": "synthetic", "label": 0,
                    "actual_sender": f"{sid}__syn", "target_sender": sid,
                    "text": text, "word_count": len(text.split()), "char_count": len(text),
                    "score": score, "tier": raw["tier"], "abstain": raw["abstain"],
                    "cos_sim": cos_sim, "spread": spread, "k": head._profiles.get(sid, {}).get("k"),
                })

        if was_training:
            encoder.train()
        return records


def _metrics_for_score_set(
    genuine: np.ndarray,
    other: np.ndarray,
    synthetic: np.ndarray,
    prefix: str = "",
) -> dict[str, float]:
    """Compute the full metric bundle for one (genuine, other, synth) score set.

    Pulled out of evaluate() so the same bundle can be computed for every
    score function configured on the probe. `prefix` is prepended to every key
    (e.g. "sigmoid_z/") so multiple score functions don't collide on the same
    metric name in the W&B log.
    """
    from sklearn.metrics import roc_auc_score
    from email_fraud.scoring.metrics import compute_pauc, compute_tpr_at_fpr

    out: dict[str, float] = {}
    out[f"{prefix}score/mean_genuine"] = float(genuine.mean()) if len(genuine) else 0.0
    out[f"{prefix}score/mean_other"] = float(other.mean()) if len(other) else 0.0
    out[f"{prefix}score/mean_synthetic"] = float(synthetic.mean()) if len(synthetic) else 0.0

    if len(genuine) and len(other):
        labels = np.concatenate([np.ones_like(genuine), np.zeros_like(other)])
        scores = np.concatenate([genuine, other])
        out[f"{prefix}auc/genuine_vs_other"] = float(roc_auc_score(labels, scores))
        out[f"{prefix}score/gap_other"] = float(genuine.mean() - other.mean())

    if len(genuine) and len(synthetic):
        labels = np.concatenate([np.ones_like(genuine), np.zeros_like(synthetic)])
        scores = np.concatenate([genuine, synthetic])
        out[f"{prefix}auc/genuine_vs_synthetic"] = float(roc_auc_score(labels, scores))
        out[f"{prefix}score/gap_synthetic"] = float(genuine.mean() - synthetic.mean())

    if len(genuine) and (len(other) or len(synthetic)):
        neg = np.concatenate([other, synthetic])
        labels = np.concatenate([np.ones_like(genuine), np.zeros_like(neg)])
        scores = np.concatenate([genuine, neg])
        out[f"{prefix}auc/genuine_vs_all"] = float(roc_auc_score(labels, scores))

    if f"{prefix}score/gap_other" in out and f"{prefix}score/gap_synthetic" in out:
        out[f"{prefix}score/synthetic_harder_than_other"] = (
            out[f"{prefix}score/gap_other"] - out[f"{prefix}score/gap_synthetic"]
        )

    if len(genuine) and len(synthetic):
        labels = np.concatenate([np.ones_like(genuine), np.zeros_like(synthetic)])
        scores = np.concatenate([genuine, synthetic])
        out[f"{prefix}pauc/genuine_vs_synthetic_5pct"] = compute_pauc(labels, scores, max_fpr=0.05)
        out[f"{prefix}pauc/genuine_vs_synthetic_10pct"] = compute_pauc(labels, scores, max_fpr=0.10)
        out[f"{prefix}tpr_at_fpr/synthetic_1pct"] = compute_tpr_at_fpr(labels, scores, target_fpr=0.01)
        out[f"{prefix}tpr_at_fpr/synthetic_5pct"] = compute_tpr_at_fpr(labels, scores, target_fpr=0.05)

    if len(genuine) and (len(other) or len(synthetic)):
        neg = np.concatenate([other, synthetic])
        labels = np.concatenate([np.ones_like(genuine), np.zeros_like(neg)])
        scores = np.concatenate([genuine, neg])
        out[f"{prefix}pauc/genuine_vs_all_5pct"] = compute_pauc(labels, scores, max_fpr=0.05)
        out[f"{prefix}pauc/genuine_vs_all_10pct"] = compute_pauc(labels, scores, max_fpr=0.10)
        out[f"{prefix}tpr_at_fpr/all_1pct"] = compute_tpr_at_fpr(labels, scores, target_fpr=0.01)
        out[f"{prefix}tpr_at_fpr/all_5pct"] = compute_tpr_at_fpr(labels, scores, target_fpr=0.05)

    # Fixed-threshold band metrics (mostly diagnostic — see _fpr_anchored_thresholds
    # for the operationally meaningful version).
    band = _threshold_band_metrics(genuine, other, synthetic)
    out.update({f"{prefix}{k}": v for k, v in band.items()})

    # FPR-anchored operating points (the deployment-relevant numbers).
    fpr_op = _fpr_anchored_thresholds(genuine, other, synthetic)
    out.update({f"{prefix}{k}": v for k, v in fpr_op.items()})

    # Coverage-at-accuracy (score-fn-dependent because it uses the 0.5
    # midpoint as the "indecision" point — only meaningful for [0, 1] fns).
    cov = _coverage_at_accuracy(genuine, other, synthetic)
    out.update({f"{prefix}{k}": v for k, v in cov.items()})

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


_FPR_TARGETS = (0.01, 0.05, 0.10)


def _fpr_anchored_thresholds(
    genuine: np.ndarray,
    other: np.ndarray,
    synthetic: np.ndarray,
) -> dict[str, float]:
    """Find τ such that FPR == target, report recall/precision/τ at that point.

    Replaces the fixed-score 0.5/0.8/0.95 view, which is broken for the current
    PrototypicalHead score = max(0, 1 - z/3): score > 0.95 requires z < 0.15,
    i.e. the query has to be ~6x closer to its centroid than the average
    enrollment email — statistically impossible, so threshold_0.95/* is always
    zero regardless of training quality.

    For each target FPR we find τ via the (n+1)-th impostor quantile (so that
    exactly n impostors exceed τ, giving the target rate) and report the
    operating point: threshold value, recall, precision, FPR_synthetic,
    FPR_other. Two anchor sets:

        fpr_overall/* — τ from the pooled impostor distribution
        fpr_synthetic/* — τ from the synthetic-only distribution (hardest)
    """
    out: dict[str, float] = {}
    if len(genuine) == 0:
        return out

    impostors = np.concatenate([other, synthetic]) if (len(other) or len(synthetic)) else np.array([])

    def _report(prefix: str, neg: np.ndarray) -> None:
        if len(neg) == 0:
            return
        # Score above which exactly target_fpr fraction of impostors fall.
        # quantile(neg, 1-fpr) gives that boundary, exclusive — guarantees we
        # never report > target FPR on this sample, at the cost of slightly
        # under-shooting on small samples.
        for fpr in _FPR_TARGETS:
            tau = float(np.quantile(neg, 1.0 - fpr))
            group = f"{prefix}/fpr_{fpr:.2f}"
            recall = float((genuine > tau).mean())
            tp = float((genuine > tau).sum())
            fp = float((neg > tau).sum())
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            out[f"{group}/threshold"] = tau
            out[f"{group}/recall"] = recall
            out[f"{group}/precision"] = precision
            if len(other):
                out[f"{group}/fpr_other"] = float((other > tau).mean())
            if len(synthetic):
                out[f"{group}/fpr_synthetic"] = float((synthetic > tau).mean())

    _report("op/all", impostors)
    if len(synthetic):
        _report("op/synthetic", synthetic)
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
