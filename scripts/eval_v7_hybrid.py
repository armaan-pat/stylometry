"""V7.1 — hybrid (cosine + Mahalanobis) scoring + honest S-norm.

What's new vs eval_v7_scoring.py
--------------------------------
1. Hybrid scorer: blends cosine-to-centroid with per-sender Mahalanobis
   (Ledoit-Wolf), sweeping the mixture weight α ∈ {0.0, 0.1, …, 1.0}. The
   two terms are individually z-normalised so they live on the same scale
   before mixing.
2. Honest S-norm: the impostor cohort is drawn from train senders that are
   **NOT** in the profile pool (14 of the 44 train senders) so the test
   impostors never contaminate the cohort statistics.

Interpretation
--------------
- α=0 reproduces cosine.
- α=1 reproduces pure Mahalanobis (z-scored).
- A sweet spot in between would say "the two scorers are complementary —
  cosine handles obvious-impostor case well, Mahalanobis handles same-style
  imitation, and a weighted mix beats either alone."
- A monotone curve means one of the two dominates and the other only adds
  noise.
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
sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")

import email_fraud.data.enron  # noqa
import email_fraud.encoders    # noqa
import email_fraud.heads       # noqa
import email_fraud.losses      # noqa
from email_fraud.config import load_config
from email_fraud.data.enron import EnronDataset
from email_fraud.registry import resolve as resolve_component
from email_fraud.utils.logging import setup_logging

# Reuse the rich-profile machinery from eval_v7_scoring.py.
from scripts.eval_v7_scoring import (  # noqa: E402
    SenderProfile, _encode_texts, _mahal_distance,
    _auc, _tpr_at_fpr, _eer, _metrics_block,
)

logger = logging.getLogger(__name__)


def _build_probe_with_cohort(
    train_dataset: EnronDataset,
    val_dataset: EnronDataset,
    syn_path: str | None,
    n_profile_senders: int,
    n_enroll: int,
    n_query: int,
    n_other: int,
    n_synth: int,
    n_cohort: int = 200,
    seed: int = 0,
) -> dict:
    """Build the V7.0 probe and ALSO carve out a disjoint cohort pool.

    Profile pool: `n_profile_senders` train senders (>= n_enroll+n_query emails each).
    Cohort pool : up to n_cohort emails drawn from the remaining train senders.
                  These are the "out-of-profile real users" we use to estimate
                  the impostor cohort distribution per profiled sender.
    Other pool  : val senders (sender-disjoint from train), same as V7.0.
    Synth pool  : LLM imitations of the profiled senders, same as V7.0.
    """
    rng = random.Random(seed)

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
        n_profile_senders = len(eligible)
    chosen = rng.sample(eligible, n_profile_senders)
    chosen_set = set(chosen)

    enroll_texts, enroll_sids, gen_texts, gen_sids = [], [], [], []
    for sid in chosen:
        ts = list(sender_to_texts[sid])
        rng.shuffle(ts)
        enroll_texts.extend(ts[:n_enroll])
        enroll_sids.extend([sid] * n_enroll)
        gen_texts.extend(ts[n_enroll : n_enroll + n_query])
        gen_sids.extend([sid] * n_query)

    # Cohort = remaining train senders, capped at n_cohort total emails.
    cohort_pool: list[str] = []
    for sid, ts in sender_to_texts.items():
        if sid in chosen_set:
            continue
        cohort_pool.extend(ts)
    rng.shuffle(cohort_pool)
    cohort_texts = cohort_pool[:n_cohort]

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
            if s in chosen_set
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
        "cohort_texts": cohort_texts,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiments/v6_luar_lora_syn.yaml")
    p.add_argument("--checkpoint", default="runs/v6_luar_lora_syn/2026-05-26_19-09-22/checkpoint_best.pt")
    p.add_argument("--out-dir", default="results/v7")
    p.add_argument("--tag", default="v7_1_hybrid")
    p.add_argument("--n-profile-senders", type=int, default=30)
    p.add_argument("--n-enroll", type=int, default=8)
    p.add_argument("--n-query", type=int, default=4)
    p.add_argument("--n-other", type=int, default=200)
    p.add_argument("--n-synth", type=int, default=200)
    p.add_argument("--n-cohort", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()

    cfg_path = _PROJECT_ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    cfg = load_config(str(cfg_path))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    EncoderClass = resolve_component("encoder", cfg.encoder.name)
    encoder = EncoderClass(cfg.encoder)
    ckpt_path = _PROJECT_ROOT / args.checkpoint if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    payload = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    encoder.load_state_dict(payload["model_state_dict"])
    encoder.to(device).eval()
    logger.info("Loaded epoch %s", payload.get("epoch"))

    train_ds = EnronDataset(cfg.data, split="train")
    val_ds = EnronDataset(cfg.data, split="validation")

    probe = _build_probe_with_cohort(
        train_ds, val_ds,
        syn_path=cfg.data.augmentation.synthetic_path,
        n_profile_senders=args.n_profile_senders,
        n_enroll=args.n_enroll,
        n_query=args.n_query,
        n_other=args.n_other,
        n_synth=args.n_synth,
        n_cohort=args.n_cohort,
        seed=args.seed,
    )
    logger.info(
        "Probe: %d senders × %d enroll, %d genuine, %d other (val), %d synth, %d cohort (off-profile train)",
        len(probe["chosen_senders"]), args.n_enroll,
        len(probe["gen_texts"]), len(probe["other_texts"]),
        len(probe["syn_texts"]), len(probe["cohort_texts"]),
    )

    enroll_emb = _encode_texts(encoder, probe["enroll_texts"], device)
    gen_emb = _encode_texts(encoder, probe["gen_texts"], device)
    oth_emb = _encode_texts(encoder, probe["other_texts"], device)
    syn_emb = _encode_texts(encoder, probe["syn_texts"], device) if probe["syn_texts"] else np.empty((0, gen_emb.shape[1]))
    coh_emb = _encode_texts(encoder, probe["cohort_texts"], device)

    sid_to_idx = defaultdict(list)
    for i, sid in enumerate(probe["enroll_sids"]):
        sid_to_idx[sid].append(i)
    profiles: dict[str, SenderProfile] = {
        sid: SenderProfile(sid, enroll_emb[idxs]) for sid, idxs in sid_to_idx.items()
    }

    # Random claimed-sender assignment for non-target pools (mirrors V7.0).
    rng = random.Random(args.seed)
    chosen = probe["chosen_senders"]
    oth_claimed = [rng.choice(chosen) for _ in range(len(oth_emb))]
    syn_claimed = list(probe["syn_sids"])
    gen_claimed = list(probe["gen_sids"])

    # ---- raw geometry per (query, sender) ----
    def _pool_raw(emb_pool, sids):
        cos = np.array([float(emb_pool[i] @ profiles[s].centroid)
                        for i, s in enumerate(sids)])
        mahal = np.array([-_mahal_distance(emb_pool[i], profiles[s].centroid, profiles[s].prec)
                          for i, s in enumerate(sids)])
        return cos, mahal

    gen_cos, gen_mah = _pool_raw(gen_emb, gen_claimed)
    oth_cos, oth_mah = _pool_raw(oth_emb, oth_claimed)
    syn_cos, syn_mah = _pool_raw(syn_emb, syn_claimed) if len(syn_emb) else (np.array([]), np.array([]))

    # ---- cohort statistics per profiled sender (honest S-norm) ----
    cohort_cos_by_sid: dict[str, np.ndarray] = {}
    cohort_mah_by_sid: dict[str, np.ndarray] = {}
    for sid, p in profiles.items():
        cohort_cos_by_sid[sid] = coh_emb @ p.centroid
        cohort_mah_by_sid[sid] = np.array([
            -_mahal_distance(q, p.centroid, p.prec) for q in coh_emb
        ])

    def _snorm(raw, sids, cohort):
        stats = {sid: (float(np.mean(arr)), float(np.std(arr)) + 1e-9)
                 for sid, arr in cohort.items()}
        return np.array([(raw[i] - stats[s][0]) / stats[s][1]
                         for i, s in enumerate(sids)])

    # Per-sender z-scored cosine and Mahalanobis using the OFF-PROFILE cohort.
    gen_cos_z = _snorm(gen_cos, gen_claimed, cohort_cos_by_sid)
    oth_cos_z = _snorm(oth_cos, oth_claimed, cohort_cos_by_sid)
    syn_cos_z = _snorm(syn_cos, syn_claimed, cohort_cos_by_sid) if len(syn_cos) else np.array([])

    gen_mah_z = _snorm(gen_mah, gen_claimed, cohort_mah_by_sid)
    oth_mah_z = _snorm(oth_mah, oth_claimed, cohort_mah_by_sid)
    syn_mah_z = _snorm(syn_mah, syn_claimed, cohort_mah_by_sid) if len(syn_mah) else np.array([])

    rows = []

    # Honest S-norm baselines (raw and hybrid).
    rows.append({"score_fn": "cosine_snorm_honest", **_metrics_block(gen_cos_z, oth_cos_z, syn_cos_z)})
    rows.append({"score_fn": "mahal_per_sender_snorm_honest",
                 **_metrics_block(gen_mah_z, oth_mah_z, syn_mah_z)})

    # ---- Hybrid sweep ----
    # The blend uses the z-normalised cosine and z-normalised Mahalanobis so
    # both terms have approximately mean 0 / std 1 under the cohort —
    # that puts α on a meaningful scale.
    best_alpha = None
    best_auc = -1.0
    for alpha in [round(0.1 * i, 1) for i in range(0, 11)]:
        gen = (1.0 - alpha) * gen_cos_z + alpha * gen_mah_z
        oth = (1.0 - alpha) * oth_cos_z + alpha * oth_mah_z
        syn = (1.0 - alpha) * syn_cos_z + alpha * syn_mah_z if len(syn_cos_z) else np.array([])
        m = _metrics_block(gen, oth, syn)
        rows.append({"score_fn": f"hybrid_alpha_{alpha:.1f}", **m})
        if m.get("auc_g_syn", 0.0) > best_auc:
            best_auc = m["auc_g_syn"]
            best_alpha = alpha

    # ---- write outputs ----
    out_dir = _PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.tag}.json"
    csv_path = out_dir / f"{args.tag}.csv"

    with json_path.open("w") as fh:
        json.dump({
            "checkpoint": str(ckpt_path),
            "config": str(cfg_path),
            "probe": {
                "n_profile_senders": len(profiles),
                "n_enroll": args.n_enroll,
                "n_query": args.n_query,
                "n_other": len(oth_emb),
                "n_synth": int(len(syn_emb)),
                "n_cohort": len(coh_emb),
                "seed": args.seed,
            },
            "best_hybrid_alpha": best_alpha,
            "rows": rows,
        }, fh, indent=2)

    keys = ["score_fn", "auc_g_syn", "auc_g_oth", "auc_g_all",
            "tpr@5pct_syn", "tpr@1pct_syn", "eer_syn",
            "tpr@5pct_oth", "tpr@5pct_all",
            "gap_syn", "gap_oth"]
    with csv_path.open("w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(
                f"{r.get(k, ''):.4f}" if isinstance(r.get(k), float) else str(r.get(k, ""))
                for k in keys) + "\n")

    print()
    print(f"V7.1 hybrid + honest S-norm — checkpoint: {ckpt_path.name}, cohort={len(coh_emb)} off-profile train emails")
    print()
    hdr = f"{'score_fn':30s} {'AUC[g/syn]':>10s} {'AUC[g/oth]':>10s} {'AUC[g/all]':>10s} "
    hdr += f"{'TPR@5%_syn':>11s} {'TPR@1%_syn':>11s} {'1-EER_syn':>10s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        line = f"{r['score_fn']:30s} "
        line += f"{r.get('auc_g_syn', float('nan')):>10.4f} "
        line += f"{r.get('auc_g_oth', float('nan')):>10.4f} "
        line += f"{r.get('auc_g_all', float('nan')):>10.4f} "
        line += f"{r.get('tpr@5pct_syn', float('nan')):>11.4f} "
        line += f"{r.get('tpr@1pct_syn', float('nan')):>11.4f} "
        line += f"{1.0 - r.get('eer_syn', float('nan')):>10.4f}"
        print(line)
    print()
    print(f"Best hybrid α by AUC[g/syn] = {best_alpha} (AUC = {best_auc:.4f})")


if __name__ == "__main__":
    main()
