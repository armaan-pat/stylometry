"""Two-model fusion: authorship specialist (v10) × LLM-detector specialist (v9).

The 2026-06-10 lineage run showed one embedding geometry cannot hold both
objectives at once (docs/v9_lineage_results_analysis.md §3): the v9 recipe's
synthetic separation peaks at epoch ~10 while human-impostor discrimination
matures at ~150. So we deploy two checkpoints of the same recipe, each frozen
at its own peak, and fuse:

    authorship model (v10): "is this email by the claimed human?"
    detector model   (v9):  "is this email LLM-written mimicry?"

Both models score every query against the claimed sender's centroid in their
OWN embedding space (each model enrolls its own ProfileBank). Fusion rules:

  AND-gate  accept iff s_auth >= tau_auth AND s_det >= tau_det, where
            tau_det is anchored on the detector's synthetic pool (FPR_syn
            target) and tau_auth on the authorship model's other pool
            (FPR_other target). Each model gates the adversary it owns.
  soft-min  fused = min(rank_auth, rank_det), ranks computed against each
            model's pooled impostor distribution — a threshold-free score
            for AUC comparison against single models.

Outputs a JSON with single-model + fused operating points and the fused
confusion matrix (genuine / other / synthetic × accept / reject).

Usage:
    python scripts/eval_two_model_fusion.py \
        --config runs/_lineage/eval_cfgs/v9_common.yaml \
        --authorship-ckpt runs/lineage_v2/v10/checkpoint_best.pt \
        --detector-ckpt runs/lineage/v9/checkpoint_best.pt \
        --out results/lineage_v2/fusion_v10xv9.json
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
FPR_TARGETS = (0.01, 0.05)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="Common eval config (probe corpus).")
    p.add_argument("--authorship-ckpt", required=True)
    p.add_argument("--detector-ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-profile-senders", type=int, default=60)
    p.add_argument("--n-enroll", type=int, default=8)
    p.add_argument("--n-query", type=int, default=6)
    p.add_argument("--n-other", type=int, default=600)
    p.add_argument("--n-synth", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    return p.parse_args()


def _model_scores(cfg, ckpt: str, device: str, probe: dict, oth_claimed) -> dict:
    EncoderClass = resolve_component("encoder", cfg.encoder.name)
    encoder = EncoderClass(cfg.encoder)
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    encoder.load_state_dict(payload["model_state_dict"])
    encoder.to(device)
    encoder.eval()
    logger.info("%s: epoch %s", ckpt, payload.get("epoch"))
    e = _encode_texts(encoder, probe["enroll_texts"], device)
    g = _encode_texts(encoder, probe["gen_texts"], device)
    o = _encode_texts(encoder, probe["other_texts"], device)
    s = _encode_texts(encoder, probe["syn_texts"], device)
    bank = ProfileBank(ewma_alpha=0.1).fit(e, probe["enroll_sids"])
    out = {
        "epoch": payload.get("epoch"),
        "genuine": score_pool(bank, g, list(probe["gen_sids"]), SCORER),
        "other": score_pool(bank, o, oth_claimed, SCORER),
        "synthetic": score_pool(bank, s, list(probe["syn_sids"]), SCORER),
    }
    del encoder
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def _single_summary(sc: dict) -> dict:
    gen, oth, syn = sc["genuine"], sc["other"], sc["synthetic"]
    y_s = np.concatenate([np.ones_like(gen), np.zeros_like(syn)])
    v_s = np.concatenate([gen, syn])
    y_o = np.concatenate([np.ones_like(gen), np.zeros_like(oth)])
    v_o = np.concatenate([gen, oth])
    out = {
        "auc_g_syn": float(compute_auc(y_s, v_s)),
        "auc_g_oth": float(compute_auc(y_o, v_o)),
        "tpr1_syn": float(compute_tpr_at_fpr(y_s, v_s, 0.01)),
        "tpr5_syn": float(compute_tpr_at_fpr(y_s, v_s, 0.05)),
    }
    for fpr in FPR_TARGETS:
        thr = np.quantile(syn, 1.0 - fpr, method="higher")
        out[f"fpr_other_at_syn{int(fpr*100)}"] = float((oth >= thr).mean())
    return out


def _rank_normalize(sc: dict) -> dict:
    """Map each pool's scores to [0,1] ranks against the pooled impostors."""
    ref = np.sort(np.concatenate([sc["other"], sc["synthetic"]]))
    return {
        pool: np.searchsorted(ref, sc[pool], side="right") / len(ref)
        for pool in ("genuine", "other", "synthetic")
    }


def main() -> None:
    args = parse_args()
    setup_logging()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(str(_PROJECT_ROOT / args.config)
                      if not Path(args.config).is_absolute() else args.config)

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

    auth = _model_scores(cfg, args.authorship_ckpt, device, probe, oth_claimed)
    det = _model_scores(cfg, args.detector_ckpt, device, probe, oth_claimed)

    result: dict = {
        "config": args.config,
        "authorship_ckpt": args.authorship_ckpt, "authorship_epoch": auth["epoch"],
        "detector_ckpt": args.detector_ckpt, "detector_epoch": det["epoch"],
        "scorer": SCORER,
        "authorship_alone": _single_summary(auth),
        "detector_alone": _single_summary(det),
    }

    # ---- AND-gate fusion: each model gates its own adversary ----
    and_gate = {}
    for fpr in FPR_TARGETS:
        tag = f"fpr{int(fpr*100)}"
        tau_det = np.quantile(det["synthetic"], 1.0 - fpr, method="higher")
        tau_auth = np.quantile(auth["other"], 1.0 - fpr, method="higher")
        cm = {}
        for pool in ("genuine", "other", "synthetic"):
            acc = (auth[pool] >= tau_auth) & (det[pool] >= tau_det)
            cm[pool] = {"accept": int(acc.sum()), "reject": int((~acc).sum()),
                        "accept_rate": float(acc.mean())}
        and_gate[tag] = {
            "tau_auth": float(tau_auth), "tau_det": float(tau_det),
            "tpr": cm["genuine"]["accept_rate"],
            "fpr_other": cm["other"]["accept_rate"],
            "fpr_syn": cm["synthetic"]["accept_rate"],
            "confusion": cm,
        }
        logger.info("AND@%s: TPR=%.3f FPR_other=%.3f FPR_syn=%.3f", tag,
                    and_gate[tag]["tpr"], and_gate[tag]["fpr_other"],
                    and_gate[tag]["fpr_syn"])
    result["and_gate"] = and_gate

    # ---- soft-min fusion: threshold-free comparison ----
    ra, rd = _rank_normalize(auth), _rank_normalize(det)
    fused = {p: np.minimum(ra[p], rd[p]) for p in ra}
    result["soft_min"] = _single_summary(fused)
    logger.info("soft-min: auc_syn=%.3f auc_oth=%.3f tpr1=%.3f tpr5=%.3f",
                result["soft_min"]["auc_g_syn"], result["soft_min"]["auc_g_oth"],
                result["soft_min"]["tpr1_syn"], result["soft_min"]["tpr5_syn"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    logger.info("Saved %s", out_path)

    print("\nmodel              auc_syn  auc_oth  tpr1   tpr5   fpr_oth@syn5")
    for label, s in [("authorship alone", result["authorship_alone"]),
                     ("detector alone", result["detector_alone"]),
                     ("fused (soft-min)", result["soft_min"])]:
        print(f"{label:18s} {s['auc_g_syn']:7.3f}  {s['auc_g_oth']:7.3f}  "
              f"{s['tpr1_syn']:5.3f}  {s['tpr5_syn']:5.3f}  "
              f"{s['fpr_other_at_syn5']:12.3f}")
    print("\nAND-gate (each model holds its own adversary at the target FPR):")
    for tag, g in and_gate.items():
        print(f"  {tag}: TPR={g['tpr']:.3f}  FPR_other={g['fpr_other']:.3f}  "
              f"FPR_syn={g['fpr_syn']:.3f}")


if __name__ == "__main__":
    main()
