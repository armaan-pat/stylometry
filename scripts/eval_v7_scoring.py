"""V7 scoring sweep — same v6 encoder, multiple scoring methods side-by-side.

Why this script exists
----------------------
The v6 best checkpoint (LUAR-MUD + LoRA + synthetic hard negatives) already
gives a reasonably separable embedding space. The win for v7 isn't to retrain
yet — it's to test whether the *centroid-based scorer* itself is the
bottleneck.

The current `PrototypicalHead` uses a single number per sender to summarize
their stylistic variance: `spread = mean(1 - cos(emb, centroid))`. That's a
1-D approximation of what's really a 128-D covariance structure. If a sender
varies a lot along one stylistic axis (say email length) but not at all along
another (say greeting habits), cosine distance can't tell those axes apart,
and the z-score is overly forgiving in directions the sender actually never
varies in. **That's exactly where a Mahalanobis distance with a Ledoit-Wolf
covariance estimate should beat cosine z-score.**

This script enrolls each profiled sender by storing **all** of their
enrollment embeddings (not just centroid+spread), then evaluates several
scoring functions on the same probe set in one pass. No model retraining;
this isolates the scorer.

How to read the output
----------------------
Every row in the printed table is a scoring function. Columns:
  AUC[g/syn]   — AUROC, genuine queries vs synthetic impostors (HARDEST).
  AUC[g/oth]   — AUROC, genuine vs other-sender impostors (easier).
  AUC[g/all]   — AUROC, genuine vs pooled impostors.
  TPR@5%_syn   — fraction of genuines admitted at the τ where 5 % of
                 synthetics pass. Operationally the most meaningful single
                 number for fraud detection: "of legit mail we'd block at the
                 5 % synthetic-pass rate, what's our recall?"
  TPR@1%_syn   — same at 1 % FPR.
  EER_syn      — equal error rate on g-vs-syn (lower is better; we report
                 1−EER so the column reads "higher is better" alongside AUCs).
  gap_syn      — mean(genuine score) − mean(synthetic score). Reflects how
                 *spread out* the two pools are in raw-score units.
  gap_oth      — same for other-sender impostors.

What to expect
--------------
- `cosine` (no spread normalization) should be ~AUC 0.87-0.89 on g/syn —
  the encoder already separates real from LLM-imitations even without
  per-sender adjustment.
- `linear_z3` (v6 default) usually ties cosine on AUROC but the *raw score
  distribution* is more discriminative (bigger `gap_syn`).
- `mahal_per_sender` (Ledoit-Wolf): expected to **improve g/syn AUROC by
  +1-3 points** if per-sender style covariance is informative; if K=8 is
  too small to estimate Σ reliably, it can be *worse* than the tied variant.
- `mahal_tied` (pooled within-sender covariance): expected to be the most
  stable Mahalanobis option — Σ is estimated from 30 senders × 7 within-
  sender deviations = 210 samples in 128 dims, which Ledoit-Wolf handles
  comfortably. **This is the candidate I expect to win on AUC[g/syn].**
- `*_snorm` (cohort score normalization): expected to help most on
  TPR@1%FPR (it sharpens the low-FPR tail) rather than full AUROC.

If Mahalanobis *doesn't* beat cosine here, it tells us per-sender covariance
isn't carrying useful signal at K=8 — and the next move is on the training
side (more enrollment-aware loss) or on enrollment (require K≥25 in real
deployment).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")

import email_fraud.data.enron  # noqa: F401
import email_fraud.encoders    # noqa: F401
import email_fraud.heads       # noqa: F401
import email_fraud.losses      # noqa: F401

from email_fraud.config import load_config
from email_fraud.data.enron import EnronDataset
from email_fraud.registry import resolve as resolve_component
from email_fraud.utils.logging import setup_logging

logger = logging.getLogger(__name__)


# =============================================================================
# Profile + scoring methods
# =============================================================================


def _l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x)
    return x if n < 1e-9 else x / n


def _ledoit_wolf_cov(embs: np.ndarray, min_samples: int = 2) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf shrinkage covariance estimate.

    Returns (covariance_matrix, shrinkage_coefficient).

    LW shrinks the sample covariance toward a scaled identity:
        Σ̂ = (1 − α) · S + α · μ · I
    where S is the sample covariance, μ = trace(S)/d, and α ∈ [0, 1] is chosen
    analytically to minimise MSE. This handles the d >> k regime gracefully —
    with K=8 enrollment embeddings in d=128, the raw sample covariance is rank
    7 and useless; LW typically picks α ≈ 0.3-0.7 here and gives a stable
    invertible estimate.
    """
    from sklearn.covariance import LedoitWolf

    if embs.shape[0] < min_samples:
        # Too few samples to estimate anything meaningful — fall back to identity.
        d = embs.shape[1]
        return np.eye(d, dtype=np.float64), 1.0

    lw = LedoitWolf().fit(embs.astype(np.float64))
    return lw.covariance_, float(lw.shrinkage_)


def _precision_from_cov(cov: np.ndarray, ridge: float = 1e-4) -> np.ndarray:
    """Invert covariance with a small ridge for numerical safety.

    We add `ridge · trace(Σ)/d · I` before inverting — invariant to the
    overall scale of Σ — so two senders with very different stylistic spreads
    get the same *relative* regularization.
    """
    d = cov.shape[0]
    scale = float(np.trace(cov)) / d
    cov_reg = cov + ridge * scale * np.eye(d, dtype=np.float64)
    return np.linalg.inv(cov_reg)


def _mahal_distance(q: np.ndarray, mu: np.ndarray, prec: np.ndarray) -> float:
    """sqrt((q − μ)ᵀ · Σ⁻¹ · (q − μ)), aka the Mahalanobis distance."""
    diff = (q - mu).astype(np.float64)
    val = float(diff @ prec @ diff)
    # Numerical clip — a tiny negative due to floating point is fine.
    return float(np.sqrt(max(val, 0.0)))


class SenderProfile:
    """Rich per-sender profile: stores raw embeddings plus derived stats.

    Stored:
        embs      — (K, d) L2-normalized enrollment embeddings
        centroid  — mean(embs) then re-normalised to unit length
        medoid    — embs row closest (cosine) to centroid; a robust "typical email"
        spread    — mean(1 − cos(emb, centroid))   (v6 default)
        spread_med — median(1 − cos(emb, centroid)) (robust to outlier emails)
        cov       — Ledoit-Wolf covariance over the K embeddings (per-sender)
        prec      — its regularised inverse, for Mahalanobis
        shrinkage — α from Ledoit-Wolf (sanity check: should be in (0, 1))
    """

    __slots__ = (
        "sid", "embs", "centroid", "medoid", "spread", "spread_med",
        "cov", "prec", "shrinkage", "k",
    )

    def __init__(self, sid: str, embs: np.ndarray) -> None:
        self.sid = sid
        self.k = embs.shape[0]
        self.embs = embs.astype(np.float64)
        c = self.embs.mean(axis=0)
        self.centroid = _l2norm(c)
        sims = self.embs @ self.centroid
        self.spread = float(np.mean(1.0 - sims))
        self.spread_med = float(np.median(1.0 - sims))
        self.medoid = self.embs[int(np.argmax(sims))]
        self.cov, self.shrinkage = _ledoit_wolf_cov(self.embs)
        self.prec = _precision_from_cov(self.cov)


# -----------------------------------------------------------------------------
# Score functions: each takes (query_emb, profile, profile_dict_for_global_stats)
# and returns a scalar where HIGHER means MORE GENUINE.
# -----------------------------------------------------------------------------


def score_cosine(q: np.ndarray, p: SenderProfile, _ctx: dict) -> float:
    """Raw cosine similarity to the L2-normalized centroid. In [-1, 1]."""
    return float(q @ p.centroid)


def score_linear_z3(q: np.ndarray, p: SenderProfile, _ctx: dict) -> float:
    """v6 default: 1 - (1-cos)/spread / 3, clamped to ≥0."""
    cos = float(q @ p.centroid)
    z = (1.0 - cos) / max(p.spread, 1e-9)
    return max(0.0, 1.0 - z / 3.0)


def score_linear_z3_median(q: np.ndarray, p: SenderProfile, _ctx: dict) -> float:
    """Same as linear_z3 but with median spread — robust to a single outlier
    enrollment email that would otherwise inflate `spread` and make the
    z-score too forgiving."""
    cos = float(q @ p.centroid)
    z = (1.0 - cos) / max(p.spread_med, 1e-9)
    return max(0.0, 1.0 - z / 3.0)


def score_sigmoid_z(q: np.ndarray, p: SenderProfile, _ctx: dict) -> float:
    cos = float(q @ p.centroid)
    z = (1.0 - cos) / max(p.spread, 1e-9)
    return 1.0 / (1.0 + np.exp(z - 1.0))


def score_medoid_cosine(q: np.ndarray, p: SenderProfile, _ctx: dict) -> float:
    """Cosine to the medoid (most central real email) instead of the mean
    centroid. Sometimes more robust than the mean when the K enrollment
    emails contain an outlier — the medoid is constrained to be a real email."""
    return float(q @ p.medoid)


def score_mahal_per_sender(q: np.ndarray, p: SenderProfile, _ctx: dict) -> float:
    """−1 × Mahalanobis distance using per-sender Ledoit-Wolf covariance.
    Negated so higher = more genuine, matching the other scorers."""
    return -_mahal_distance(q, p.centroid, p.prec)


def score_mahal_tied(q: np.ndarray, p: SenderProfile, ctx: dict) -> float:
    """Mahalanobis with a SHARED within-sender covariance pooled across all
    profiled senders. Much more sample-efficient than per-sender Σ when K is
    small — the pooled estimate uses K·N residuals instead of K."""
    prec = ctx["tied_prec"]
    return -_mahal_distance(q, p.centroid, prec)


def _cohort_snorm(
    base_scores_by_sid: dict[str, np.ndarray],
    query_scores: dict[tuple[str, str], float],
) -> dict[tuple[str, str], float]:
    """S-norm: for each (query, claimed_sender) pair, normalize the raw score
    by the impostor cohort's score against the SAME sender.

        z = (s − μ_impostor[sid]) / σ_impostor[sid]

    `base_scores_by_sid[sid]` is the array of raw scores the impostor cohort
    produced when scored against that sender. `query_scores[(sid, qid)]` is
    the raw score of the actual query.

    Returns the same dict but with normalized scores. This is the standard
    score normalization used in speaker verification (Auckenthaler et al.
    2000), and it helps most at low FPR because per-sender impostor
    distributions vary a lot.
    """
    stats = {
        sid: (float(np.mean(arr)), float(np.std(arr)) + 1e-9)
        for sid, arr in base_scores_by_sid.items()
    }
    out: dict[tuple[str, str], float] = {}
    for key, s in query_scores.items():
        sid = key[0]
        mu, sd = stats[sid]
        out[key] = (s - mu) / sd
    return out


# =============================================================================
# Pipeline
# =============================================================================


def _encode_texts(encoder, texts: list[str], device: str, batch_size: int = 32) -> np.ndarray:
    """Forward `texts` through the encoder, returning a (N, d) numpy array.

    luar_episode encoders normally pool K texts into one embedding; we
    override episode_k=1 so each text gets its own embedding, which is what
    we want at inference / probe time.
    """
    episode_k_attr = getattr(encoder, "episode_k", None)
    saved_k = None
    if episode_k_attr is not None:
        saved_k = encoder.config.episode_k
        encoder.config.episode_k = 1
    try:
        outs = []
        encoder.eval()
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                tok = encoder.tokenize(batch)
                tok = {k: v.to(device) for k, v in tok.items()}
                emb = encoder.encode(**tok).detach().cpu().numpy()
                outs.append(emb)
        return np.concatenate(outs, axis=0) if outs else np.empty((0, 0))
    finally:
        if saved_k is not None:
            encoder.config.episode_k = saved_k


def _build_probe(
    train_dataset: EnronDataset,
    val_dataset: EnronDataset,
    syn_path: str | None,
    n_profile_senders: int,
    n_enroll: int,
    n_query: int,
    n_other: int,
    n_synth: int,
    seed: int = 0,
) -> dict:
    """Mirror CentroidProbe sampling so we evaluate on the same set as v6's
    in-training probe. Returns text + sender_id lists for each pool."""
    rng = random.Random(seed)

    # Strip synthetic-suffixed senders from the train pool — only real emails
    # form profiles.
    real_train = [
        (t, s) for t, s in zip(train_dataset._texts, train_dataset._sender_ids_list)
        if not s.endswith("__syn")
    ]
    sender_to_texts: dict[str, list[str]] = defaultdict(list)
    for t, s in real_train:
        sender_to_texts[s].append(t)

    min_needed = n_enroll + n_query
    eligible = [s for s, ts in sender_to_texts.items() if len(ts) >= min_needed]
    if len(eligible) < n_profile_senders:
        logger.warning(
            "Only %d eligible senders (need %d); using all.",
            len(eligible), n_profile_senders,
        )
        n_profile_senders = len(eligible)
    chosen = rng.sample(eligible, n_profile_senders)

    enroll_texts, enroll_sids = [], []
    gen_texts, gen_sids = [], []
    for sid in chosen:
        ts = list(sender_to_texts[sid])
        rng.shuffle(ts)
        enroll_texts.extend(ts[:n_enroll])
        enroll_sids.extend([sid] * n_enroll)
        gen_texts.extend(ts[n_enroll : n_enroll + n_query])
        gen_sids.extend([sid] * n_query)

    val_texts = list(val_dataset._texts)
    n_other_pick = min(n_other, len(val_texts))
    other_idx = rng.sample(range(len(val_texts)), n_other_pick)
    other_texts = [val_texts[i] for i in other_idx]

    syn_texts, syn_src = [], []
    if syn_path:
        from datasets import load_from_disk
        syn = load_from_disk(syn_path)
        syn_pairs = [
            (t, s) for t, s in zip(syn["text"], syn["source_sender_id"])
            if s in set(chosen)
        ]
        rng.shuffle(syn_pairs)
        for t, s in syn_pairs[:n_synth]:
            syn_texts.append(t)
            syn_src.append(s)

    return {
        "chosen_senders": chosen,
        "enroll_texts": enroll_texts,
        "enroll_sids": enroll_sids,
        "gen_texts": gen_texts,
        "gen_sids": gen_sids,
        "other_texts": other_texts,
        "syn_texts": syn_texts,
        "syn_sids": syn_src,
    }


def _compute_tied_precision(profiles: dict[str, SenderProfile]) -> np.ndarray:
    """Pool within-sender deviations across all senders and fit Ledoit-Wolf.

        residuals_sid = embs_sid − centroid_sid
        Σ_tied = LedoitWolf(stack(all residuals))

    Then return Σ⁻¹ with the same ridge regularization as the per-sender path.
    """
    all_resid = []
    for p in profiles.values():
        all_resid.append(p.embs - p.centroid)
    R = np.concatenate(all_resid, axis=0)
    cov, _shr = _ledoit_wolf_cov(R)
    return _precision_from_cov(cov)


# -----------------------------------------------------------------------------
# Metric helpers
# -----------------------------------------------------------------------------


def _auc(y_true, y_score) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y_true, y_score))


def _tpr_at_fpr(y_true, y_score, fpr_target: float) -> float:
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.searchsorted(fpr, fpr_target, side="right") - 1
    return float(tpr[max(idx, 0)])


def _eer(y_true, y_score) -> float:
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def _metrics_block(gen, oth, syn) -> dict:
    """Compute the metric bundle for one scoring function."""
    out: dict[str, float] = {}
    out["mean_gen"] = float(np.mean(gen)) if len(gen) else float("nan")
    out["mean_oth"] = float(np.mean(oth)) if len(oth) else float("nan")
    out["mean_syn"] = float(np.mean(syn)) if len(syn) else float("nan")
    out["gap_oth"] = out["mean_gen"] - out["mean_oth"]
    out["gap_syn"] = out["mean_gen"] - out["mean_syn"]

    if len(gen) and len(syn):
        y = np.concatenate([np.ones_like(gen), np.zeros_like(syn)])
        s = np.concatenate([gen, syn])
        out["auc_g_syn"] = _auc(y, s)
        out["tpr@5pct_syn"] = _tpr_at_fpr(y, s, 0.05)
        out["tpr@1pct_syn"] = _tpr_at_fpr(y, s, 0.01)
        out["eer_syn"] = _eer(y, s)
    if len(gen) and len(oth):
        y = np.concatenate([np.ones_like(gen), np.zeros_like(oth)])
        s = np.concatenate([gen, oth])
        out["auc_g_oth"] = _auc(y, s)
        out["tpr@5pct_oth"] = _tpr_at_fpr(y, s, 0.05)
    if len(gen) and (len(syn) or len(oth)):
        neg = np.concatenate([oth, syn])
        y = np.concatenate([np.ones_like(gen), np.zeros_like(neg)])
        s = np.concatenate([gen, neg])
        out["auc_g_all"] = _auc(y, s)
        out["tpr@5pct_all"] = _tpr_at_fpr(y, s, 0.05)
    return out


# =============================================================================
# Main
# =============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiments/v6_luar_lora_syn.yaml")
    p.add_argument("--checkpoint", default="runs/v6_luar_lora_syn/2026-05-26_19-09-22/checkpoint_best.pt")
    p.add_argument("--out-dir", default="results/v7")
    p.add_argument("--tag", default="v7_0_scoring_sweep",
                   help="Filename prefix for outputs.")
    p.add_argument("--n-profile-senders", type=int, default=30)
    p.add_argument("--n-enroll", type=int, default=8)
    p.add_argument("--n-query", type=int, default=4)
    p.add_argument("--n-other", type=int, default=200)
    p.add_argument("--n-synth", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()

    cfg_path = _PROJECT_ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    cfg = load_config(str(cfg_path))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading encoder + checkpoint")
    EncoderClass = resolve_component("encoder", cfg.encoder.name)
    encoder = EncoderClass(cfg.encoder)
    ckpt_path = _PROJECT_ROOT / args.checkpoint if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    payload = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    encoder.load_state_dict(payload["model_state_dict"])
    encoder.to(device)
    encoder.eval()
    logger.info("Loaded epoch %s from %s", payload.get("epoch"), ckpt_path)

    logger.info("Loading datasets")
    train_ds = EnronDataset(cfg.data, split="train")
    val_ds = EnronDataset(cfg.data, split="validation")

    probe = _build_probe(
        train_ds, val_ds,
        syn_path=cfg.data.augmentation.synthetic_path,
        n_profile_senders=args.n_profile_senders,
        n_enroll=args.n_enroll,
        n_query=args.n_query,
        n_other=args.n_other,
        n_synth=args.n_synth,
        seed=args.seed,
    )
    logger.info(
        "Probe: %d senders × %d enroll, %d genuine, %d other, %d synthetic",
        len(probe["chosen_senders"]), args.n_enroll,
        len(probe["gen_texts"]), len(probe["other_texts"]), len(probe["syn_texts"]),
    )

    # ---- encode every pool exactly once ----
    logger.info("Encoding enrollment pool")
    enroll_emb = _encode_texts(encoder, probe["enroll_texts"], device)
    logger.info("Encoding genuine queries")
    gen_emb = _encode_texts(encoder, probe["gen_texts"], device)
    logger.info("Encoding other-sender impostors")
    oth_emb = _encode_texts(encoder, probe["other_texts"], device)
    logger.info("Encoding synthetic impostors")
    syn_emb = _encode_texts(encoder, probe["syn_texts"], device) if probe["syn_texts"] else np.empty((0, gen_emb.shape[1]))

    # ---- build rich profiles ----
    sid_to_idx = defaultdict(list)
    for i, sid in enumerate(probe["enroll_sids"]):
        sid_to_idx[sid].append(i)
    profiles: dict[str, SenderProfile] = {}
    for sid, idxs in sid_to_idx.items():
        profiles[sid] = SenderProfile(sid, enroll_emb[idxs])
    avg_shrinkage = float(np.mean([p.shrinkage for p in profiles.values()]))
    logger.info("Built %d profiles. Mean per-sender LW shrinkage α = %.3f",
                len(profiles), avg_shrinkage)

    # Pooled within-sender Ledoit-Wolf precision matrix for the tied variants.
    tied_prec = _compute_tied_precision(profiles)
    ctx = {"tied_prec": tied_prec}

    # Each other-sender impostor is assigned a random claimed sender (mirrors
    # CentroidProbe.evaluate()) so we have a defined (query, sender) tuple.
    rng = random.Random(args.seed)
    chosen = probe["chosen_senders"]
    oth_claimed = [rng.choice(chosen) for _ in range(len(oth_emb))]
    syn_claimed = list(probe["syn_sids"])
    gen_claimed = list(probe["gen_sids"])

    # ---- score every pool under every score function ----
    SCORERS = {
        "cosine":             score_cosine,
        "linear_z3":          score_linear_z3,
        "linear_z3_median":   score_linear_z3_median,
        "sigmoid_z":          score_sigmoid_z,
        "medoid_cosine":      score_medoid_cosine,
        "mahal_per_sender":   score_mahal_per_sender,
        "mahal_tied":         score_mahal_tied,
    }

    # Pre-compute the impostor cohort scores per (sid) for S-norm. We use the
    # cosine-to-centroid distribution of the `other_emb` pool for each sender
    # — this is the cohort "what does a typical impostor score against sender
    # S look like?" We then z-normalize the actual (gen, oth, syn) scores by
    # that distribution.
    cohort_cos_by_sid: dict[str, np.ndarray] = {}
    cohort_mahal_tied_by_sid: dict[str, np.ndarray] = {}
    for sid, p in profiles.items():
        cohort_cos_by_sid[sid] = oth_emb @ p.centroid
        cohort_mahal_tied_by_sid[sid] = np.array([
            -_mahal_distance(q, p.centroid, tied_prec) for q in oth_emb
        ])

    def _score_pool(scorer, emb, sids):
        return np.array([scorer(emb[i], profiles[s], ctx) for i, s in enumerate(sids)])

    rows = []

    for name, fn in SCORERS.items():
        gen = _score_pool(fn, gen_emb, gen_claimed)
        oth = _score_pool(fn, oth_emb, oth_claimed)
        syn = _score_pool(fn, syn_emb, syn_claimed) if len(syn_emb) else np.array([])
        m = _metrics_block(gen, oth, syn)
        rows.append({"score_fn": name, **m})

    # ---- S-norm variants (built atop cosine and mahal_tied) ----
    def _snorm(raw_scores, sids, cohort):
        stats = {sid: (float(np.mean(arr)), float(np.std(arr)) + 1e-9)
                 for sid, arr in cohort.items()}
        return np.array([
            (raw_scores[i] - stats[s][0]) / stats[s][1]
            for i, s in enumerate(sids)
        ])

    gen_cos = _score_pool(score_cosine, gen_emb, gen_claimed)
    oth_cos = _score_pool(score_cosine, oth_emb, oth_claimed)
    syn_cos = _score_pool(score_cosine, syn_emb, syn_claimed) if len(syn_emb) else np.array([])
    gen_cos_z = _snorm(gen_cos, gen_claimed, cohort_cos_by_sid)
    oth_cos_z = _snorm(oth_cos, oth_claimed, cohort_cos_by_sid)
    syn_cos_z = _snorm(syn_cos, syn_claimed, cohort_cos_by_sid) if len(syn_cos) else np.array([])
    rows.append({"score_fn": "cosine_snorm", **_metrics_block(gen_cos_z, oth_cos_z, syn_cos_z)})

    gen_mt = _score_pool(score_mahal_tied, gen_emb, gen_claimed)
    oth_mt = _score_pool(score_mahal_tied, oth_emb, oth_claimed)
    syn_mt = _score_pool(score_mahal_tied, syn_emb, syn_claimed) if len(syn_emb) else np.array([])
    gen_mt_z = _snorm(gen_mt, gen_claimed, cohort_mahal_tied_by_sid)
    oth_mt_z = _snorm(oth_mt, oth_claimed, cohort_mahal_tied_by_sid)
    syn_mt_z = _snorm(syn_mt, syn_claimed, cohort_mahal_tied_by_sid) if len(syn_mt) else np.array([])
    rows.append({"score_fn": "mahal_tied_snorm", **_metrics_block(gen_mt_z, oth_mt_z, syn_mt_z)})

    # ---- write outputs ----
    out_dir = _PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{args.tag}.json"
    csv_path = out_dir / f"{args.tag}.csv"

    summary = {
        "checkpoint": str(ckpt_path),
        "config": str(cfg_path),
        "probe": {
            "n_profile_senders": len(profiles),
            "n_enroll": args.n_enroll,
            "n_query": args.n_query,
            "n_other": len(oth_emb),
            "n_synth": int(len(syn_emb)),
            "seed": args.seed,
        },
        "per_sender_lw_shrinkage_mean": avg_shrinkage,
        "rows": rows,
    }
    with json_path.open("w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Saved JSON → %s", json_path)

    # CSV: rows × columns
    keys = ["score_fn", "auc_g_syn", "auc_g_oth", "auc_g_all",
            "tpr@5pct_syn", "tpr@1pct_syn", "eer_syn",
            "tpr@5pct_oth", "tpr@5pct_all",
            "gap_syn", "gap_oth", "mean_gen", "mean_syn", "mean_oth"]
    with csv_path.open("w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(f"{r.get(k, ''):.4f}" if isinstance(r.get(k), float) else str(r.get(k, ""))
                              for k in keys) + "\n")
    logger.info("Saved CSV  → %s", csv_path)

    # Pretty print
    print()
    print(f"V7 scoring sweep — checkpoint: {ckpt_path.name}, profiled senders: {len(profiles)}, "
          f"K_enroll={args.n_enroll}, mean LW shrinkage α={avg_shrinkage:.3f}")
    print()
    hdr = f"{'score_fn':22s} {'AUC[g/syn]':>10s} {'AUC[g/oth]':>10s} {'AUC[g/all]':>10s} "
    hdr += f"{'TPR@5%_syn':>11s} {'TPR@1%_syn':>11s} {'1-EER_syn':>10s} {'gap_syn':>9s} {'gap_oth':>9s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        line = f"{r['score_fn']:22s} "
        line += f"{r.get('auc_g_syn', float('nan')):>10.4f} "
        line += f"{r.get('auc_g_oth', float('nan')):>10.4f} "
        line += f"{r.get('auc_g_all', float('nan')):>10.4f} "
        line += f"{r.get('tpr@5pct_syn', float('nan')):>11.4f} "
        line += f"{r.get('tpr@1pct_syn', float('nan')):>11.4f} "
        line += f"{1.0 - r.get('eer_syn', float('nan')):>10.4f} "
        line += f"{r.get('gap_syn', float('nan')):>+9.4f} "
        line += f"{r.get('gap_oth', float('nan')):>+9.4f}"
        print(line)


if __name__ == "__main__":
    main()
