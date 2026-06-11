"""Checkpoint soup: interpolate two checkpoints and trace the Pareto curve.

The 2026-06-10 lineage run showed v9's two objectives mature at opposite ends
of training: the ep-10 `checkpoint_best` aces genuine-vs-synthetic but accepts
59.5% of wrong-sender real mail, while ep-150 `checkpoint_last` is the best
human-impostor verifier we have but its synthetic tail collapses
(docs/v9_lineage_results_analysis.md §3, §7.2). LoRA deltas usually
interpolate well, so a weight-space blend may dominate both endpoints.

For each α in --alphas, build w = (1-α)·A + α·B, evaluate on the same probe
the scorer ablation uses (identical sampling, seed 0), and report:

    auc_g_syn / tpr1 / tpr5        genuine vs synthetic (the v1 headline)
    auc_g_oth                      genuine vs other-sender
    fpr_other_at_{1,5}             other accept rate at syn-anchored thresholds
    min_auc                        min(auc_g_syn, auc_g_oth) — selection metric

Saves a JSON table; with --save-best-to also writes the α maximizing min_auc
as a loadable checkpoint.

Usage:
    python scripts/eval_checkpoint_soup.py \
        --config runs/_lineage/eval_cfgs/v9_common.yaml \
        --ckpt-a runs/lineage/v9/checkpoint_best.pt \
        --ckpt-b runs/lineage/v9/checkpoint_last.pt \
        --alphas 0,0.25,0.5,0.75,1 \
        --out results/lineage_v2/soup_v9.json \
        --save-best-to runs/lineage_v2/v9/checkpoint_soup.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")

import email_fraud.data.enron  # noqa: F401
import email_fraud.encoders    # noqa: F401
import email_fraud.heads       # noqa: F401
import email_fraud.losses      # noqa: F401

from email_fraud.config import load_config
from email_fraud.data.enron import EnronDataset
from email_fraud.registry import resolve as resolve_component
from email_fraud.scoring.adaptive import ProfileBank, score_pool
from email_fraud.scoring.metrics import compute_auc, compute_tpr_at_fpr
from email_fraud.utils.logging import setup_logging

from eval_v7_scoring import _build_probe, _encode_texts  # type: ignore

logger = logging.getLogger(__name__)

SCORER = "baseline_linear_z3"


def blend_state_dicts(a: dict, b: dict, alpha: float) -> dict:
    out = {}
    for k, va in a.items():
        vb = b[k]
        if torch.is_floating_point(va):
            out[k] = (1.0 - alpha) * va + alpha * vb.to(va.dtype)
        else:
            out[k] = va
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt-a", required=True, help="α=0 endpoint")
    p.add_argument("--ckpt-b", required=True, help="α=1 endpoint")
    p.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    p.add_argument("--out", required=True)
    p.add_argument("--save-best-to", default=None,
                   help="Write the blended checkpoint with the best min_auc here.")
    p.add_argument("--n-profile-senders", type=int, default=60)
    p.add_argument("--n-enroll", type=int, default=8)
    p.add_argument("--n-query", type=int, default=6)
    p.add_argument("--n-other", type=int, default=600)
    p.add_argument("--n-synth", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_config(str(_PROJECT_ROOT / args.config)
                      if not Path(args.config).is_absolute() else args.config)
    pay_a = torch.load(args.ckpt_a, map_location=device, weights_only=False)
    pay_b = torch.load(args.ckpt_b, map_location=device, weights_only=False)
    state_a, state_b = pay_a["model_state_dict"], pay_b["model_state_dict"]
    logger.info("Soup endpoints: A=%s (ep %s)  B=%s (ep %s)",
                args.ckpt_a, pay_a.get("epoch"), args.ckpt_b, pay_b.get("epoch"))

    EncoderClass = resolve_component("encoder", cfg.encoder.name)
    encoder = EncoderClass(cfg.encoder)
    encoder.to(device)

    train_ds = EnronDataset(cfg.data, split="train")
    val_ds = EnronDataset(cfg.data, split="validation")
    probe = _build_probe(
        train_ds, val_ds, syn_path=cfg.data.augmentation.synthetic_path,
        n_profile_senders=args.n_profile_senders, n_enroll=args.n_enroll,
        n_query=args.n_query, n_other=args.n_other, n_synth=args.n_synth,
        seed=args.seed,
    )
    rng = np.random.default_rng(args.seed)
    chosen = probe["chosen_senders"]
    oth_claimed = [chosen[i] for i in rng.integers(0, len(chosen), len(probe["other_texts"]))]

    rows = []
    for alpha in [float(x) for x in args.alphas.split(",")]:
        encoder.load_state_dict(blend_state_dicts(state_a, state_b, alpha))
        encoder.eval()
        e = _encode_texts(encoder, probe["enroll_texts"], device)
        g = _encode_texts(encoder, probe["gen_texts"], device)
        o = _encode_texts(encoder, probe["other_texts"], device)
        s = _encode_texts(encoder, probe["syn_texts"], device)
        bank = ProfileBank(ewma_alpha=0.1).fit(e, probe["enroll_sids"])
        gen = score_pool(bank, g, list(probe["gen_sids"]), SCORER)
        oth = score_pool(bank, o, oth_claimed, SCORER)
        syn = score_pool(bank, s, list(probe["syn_sids"]), SCORER)

        y_s = np.concatenate([np.ones_like(gen), np.zeros_like(syn)])
        sc_s = np.concatenate([gen, syn])
        y_o = np.concatenate([np.ones_like(gen), np.zeros_like(oth)])
        sc_o = np.concatenate([gen, oth])
        row = {
            "alpha": alpha,
            "auc_g_syn": float(compute_auc(y_s, sc_s)),
            "tpr1": float(compute_tpr_at_fpr(y_s, sc_s, 0.01)),
            "tpr5": float(compute_tpr_at_fpr(y_s, sc_s, 0.05)),
            "auc_g_oth": float(compute_auc(y_o, sc_o)),
            "fpr_other_at_1": float((oth >= np.quantile(syn, 0.99, method="higher")).mean()),
            "fpr_other_at_5": float((oth >= np.quantile(syn, 0.95, method="higher")).mean()),
        }
        row["min_auc"] = min(row["auc_g_syn"], row["auc_g_oth"])
        rows.append(row)
        logger.info("α=%.2f  auc_syn=%.3f auc_oth=%.3f tpr1=%.3f tpr5=%.3f "
                    "fpr_oth@5=%.3f min_auc=%.3f", alpha, row["auc_g_syn"],
                    row["auc_g_oth"], row["tpr1"], row["tpr5"],
                    row["fpr_other_at_5"], row["min_auc"])

    best = max(rows, key=lambda r: r["min_auc"])
    result = {
        "ckpt_a": args.ckpt_a, "epoch_a": pay_a.get("epoch"),
        "ckpt_b": args.ckpt_b, "epoch_b": pay_b.get("epoch"),
        "scorer": SCORER, "selection": "max min(auc_g_syn, auc_g_oth)",
        "best_alpha": best["alpha"], "rows": rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    logger.info("Saved %s (best α=%.2f, min_auc=%.3f)",
                out_path, best["alpha"], best["min_auc"])

    if args.save_best_to:
        encoder_state = blend_state_dicts(state_a, state_b, best["alpha"])
        save_path = Path(args.save_best_to)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model_state_dict": encoder_state,
             "epoch": f"soup(a={pay_a.get('epoch')},b={pay_b.get('epoch')},alpha={best['alpha']})",
             "soup": result},
            str(save_path),
        )
        logger.info("Saved best blend → %s", save_path)

    print("\nalpha  auc_syn  auc_oth  tpr1   tpr5   fpr_oth@5  min_auc")
    for r in rows:
        marker = " ←best" if r is best else ""
        print(f"{r['alpha']:5.2f}  {r['auc_g_syn']:7.3f}  {r['auc_g_oth']:7.3f}  "
              f"{r['tpr1']:5.3f}  {r['tpr5']:5.3f}  {r['fpr_other_at_5']:9.3f}  "
              f"{r['min_auc']:7.3f}{marker}")


if __name__ == "__main__":
    main()
