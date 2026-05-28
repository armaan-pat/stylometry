"""V7.2 — Enrollment-K sweep for Mahalanobis vs cosine.

For each K in {4, 8, 16, 25, 40} we rebuild the probe with that many
enrollment emails per sender and score the same three pools under three
scoring functions.

Why this matters
----------------
The product spec says profiles below k=5 abstain, k≥25 is "very_high"
confidence. Mahalanobis distance is fundamentally a *covariance-aware*
distance, and the per-sender covariance estimate quality scales with K.
At K=8 the per-sender Σ is rank-7 (so LW shrinks heavily); at K=25 we have
a much more honest 25-sample estimate and LW shrinkage drops, freeing the
Σ to carry more directional information.

Expected pattern: cosine should be roughly flat across K (the centroid
estimate stabilises quickly), while `mahal_per_sender` should improve
faster — the gap should *widen* with K. If the gap *narrows* at large K
that's a sign the encoder's embedding space isn't actually elliptical and
the Mahalanobis win at K=8 was just noise/regularisation.
"""

from __future__ import annotations

import argparse
import json
import logging
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

from scripts.eval_v7_scoring import (
    SenderProfile, _encode_texts,
    score_cosine, score_linear_z3, score_mahal_per_sender, score_mahal_tied,
    _compute_tied_precision, _metrics_block,
    _build_probe,
)

logger = logging.getLogger(__name__)


def _evaluate_k(encoder, train_ds, val_ds, syn_path, K, n_query, n_other, n_synth,
                n_profile_senders, device, seed=0) -> dict:
    probe = _build_probe(
        train_ds, val_ds, syn_path,
        n_profile_senders=n_profile_senders,
        n_enroll=K,
        n_query=n_query,
        n_other=n_other,
        n_synth=n_synth,
        seed=seed,
    )
    enroll_emb = _encode_texts(encoder, probe["enroll_texts"], device)
    gen_emb = _encode_texts(encoder, probe["gen_texts"], device)
    oth_emb = _encode_texts(encoder, probe["other_texts"], device)
    syn_emb = _encode_texts(encoder, probe["syn_texts"], device) if probe["syn_texts"] else np.empty((0, gen_emb.shape[1]))

    sid_to_idx = defaultdict(list)
    for i, sid in enumerate(probe["enroll_sids"]):
        sid_to_idx[sid].append(i)
    profiles = {sid: SenderProfile(sid, enroll_emb[idxs]) for sid, idxs in sid_to_idx.items()}

    avg_shrinkage = float(np.mean([p.shrinkage for p in profiles.values()]))
    tied_prec = _compute_tied_precision(profiles)
    ctx = {"tied_prec": tied_prec}

    import random
    rng = random.Random(seed)
    chosen = probe["chosen_senders"]
    oth_claimed = [rng.choice(chosen) for _ in range(len(oth_emb))]
    syn_claimed = list(probe["syn_sids"])
    gen_claimed = list(probe["gen_sids"])

    def _pool(scorer, emb, sids):
        return np.array([scorer(emb[i], profiles[s], ctx) for i, s in enumerate(sids)])

    SCORERS = {
        "cosine":          score_cosine,
        "linear_z3":       score_linear_z3,
        "mahal_per_sender": score_mahal_per_sender,
        "mahal_tied":      score_mahal_tied,
    }
    out = {}
    for name, fn in SCORERS.items():
        gen = _pool(fn, gen_emb, gen_claimed)
        oth = _pool(fn, oth_emb, oth_claimed)
        syn = _pool(fn, syn_emb, syn_claimed) if len(syn_emb) else np.array([])
        out[name] = _metrics_block(gen, oth, syn)

    return {
        "K": K,
        "n_profile_senders_used": len(profiles),
        "mean_lw_shrinkage": avg_shrinkage,
        "scores": out,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiments/v6_luar_lora_syn.yaml")
    p.add_argument("--checkpoint", default="runs/v6_luar_lora_syn/2026-05-26_19-09-22/checkpoint_best.pt")
    p.add_argument("--out-dir", default="results/v7")
    p.add_argument("--tag", default="v7_2_k_sweep")
    p.add_argument("--ks", nargs="+", type=int, default=[4, 8, 16, 25, 40])
    p.add_argument("--n-profile-senders", type=int, default=30)
    p.add_argument("--n-query", type=int, default=4)
    p.add_argument("--n-other", type=int, default=200)
    p.add_argument("--n-synth", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()

    cfg_path = _PROJECT_ROOT / args.config
    cfg = load_config(str(cfg_path))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    EncoderClass = resolve_component("encoder", cfg.encoder.name)
    encoder = EncoderClass(cfg.encoder)
    ckpt_path = _PROJECT_ROOT / args.checkpoint
    payload = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    encoder.load_state_dict(payload["model_state_dict"])
    encoder.to(device).eval()
    logger.info("Loaded epoch %s", payload.get("epoch"))

    train_ds = EnronDataset(cfg.data, split="train")
    val_ds = EnronDataset(cfg.data, split="validation")

    all_rows = []
    for K in args.ks:
        logger.info("Running K = %d", K)
        res = _evaluate_k(
            encoder, train_ds, val_ds,
            cfg.data.augmentation.synthetic_path,
            K=K,
            n_query=args.n_query,
            n_other=args.n_other,
            n_synth=args.n_synth,
            n_profile_senders=args.n_profile_senders,
            device=device,
            seed=args.seed,
        )
        all_rows.append(res)

    out_dir = _PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.tag}.json"
    with json_path.open("w") as fh:
        json.dump({"rows": all_rows}, fh, indent=2)
    logger.info("Saved JSON → %s", json_path)

    print()
    print(f"V7.2 — Enrollment-K sweep ({len(all_rows)} K values × 4 scorers)")
    print()
    hdr = f"{'K':>4s} {'n_send':>7s} {'shrinkα':>8s} {'scorer':22s} "
    hdr += f"{'AUC[g/syn]':>10s} {'AUC[g/oth]':>10s} {'AUC[g/all]':>10s} {'TPR@5%_syn':>11s} {'TPR@1%_syn':>11s} {'1-EER':>9s}"
    print(hdr)
    print("-" * len(hdr))
    for row in all_rows:
        K = row["K"]
        nsend = row["n_profile_senders_used"]
        shrink = row["mean_lw_shrinkage"]
        for scorer, m in row["scores"].items():
            line = f"{K:>4d} {nsend:>7d} {shrink:>8.3f} {scorer:22s} "
            line += f"{m.get('auc_g_syn', float('nan')):>10.4f} "
            line += f"{m.get('auc_g_oth', float('nan')):>10.4f} "
            line += f"{m.get('auc_g_all', float('nan')):>10.4f} "
            line += f"{m.get('tpr@5pct_syn', float('nan')):>11.4f} "
            line += f"{m.get('tpr@1pct_syn', float('nan')):>11.4f} "
            line += f"{1 - m.get('eer_syn', float('nan')):>9.4f}"
            print(line)
        print()


if __name__ == "__main__":
    main()
