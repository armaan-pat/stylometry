"""Ablation: which data-dependent scorer should the PrototypicalHead use?

Avenue 2 from the V7 roadmap ("make every scoring parameter a function of the
data") is already PROTOTYPED in `src/email_fraud/scoring/adaptive.py` — per-sender
shrunk-p90 z-calibration (`z_persender_cal`), a smooth cosine<->Mahalanobis
precision-space blend (`mahal_blend`), a global p90 calibration (`z_global_cal`),
the tier switch, etc. But none of those are wired into the production head or
the CentroidProbe, so they've never been compared head-to-head against the
shipped baselines (`baseline_linear_z3`, `baseline_cosine`, raw `mahalanobis`).

This script does that comparison on a frozen encoder checkpoint, with the
statistical rigor needed to actually DECIDE:

  1. Build the same probe set CentroidProbe/eval_v7_scoring use (genuine /
     other-sender / synthetic pools), encode every pool once.
  2. Fit one `ProfileBank` (pays the covariance/LOO cost once) and score every
     pool under every scorer in `adaptive.SCORERS`.
  3. Compute the operating-point metrics that matter for fraud detection:
     AUROC, pAUC@5%, TPR@1%FPR, TPR@5%FPR, 1-EER — on the HARD genuine-vs-
     synthetic split and on genuine-vs-all.
  4. **Paired bootstrap** over the query pools: resample once per replicate and
     apply the SAME indices to every scorer, so the delta vs the production
     baseline is measured on identical samples. Report a 95% CI on each
     scorer's metric AND on its delta-vs-baseline. A scorer is only declared a
     winner if the delta CI excludes 0 — that's the guard against chasing the
     ~0.02 AUROC sampling noise.
  5. Optional K-sweep (`--k-sweep 4,8,16,25`) reproduces the V7.2 enrollment
     curve as point estimates so you can see where each scorer's advantage
     kicks in.

Outputs a ranked console table, a CSV, and a JSON with full CIs, plus a single
RECOMMENDATION line naming the scorer to adopt for the chosen target metric.

Usage
-----
    python scripts/ablate_adaptive_scorers.py \
        --config configs/experiments/v7_luar_lora_syn_mahal_eval.yaml \
        --checkpoint runs/v7_luar_lora_syn_mahal/<ts>/checkpoint_epoch_150.pt \
        --target auc_g_syn --rank-by tpr1_syn --bootstrap 1000

    # Where does each scorer's advantage start (no bootstrap, fast):
    python scripts/ablate_adaptive_scorers.py --checkpoint <ckpt> \
        --k-sweep 4,8,16,25 --bootstrap 0
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
from email_fraud.scoring.adaptive import ProfileBank, SCORERS, BASELINE, score_pool
from email_fraud.scoring.metrics import (
    compute_auc,
    compute_eer,
    compute_pauc,
    compute_tpr_at_fpr,
)
from email_fraud.utils.logging import setup_logging

# Reuse the probe sampler + encoder helper from the existing sweep so we
# evaluate on exactly the same probe construction.
from eval_v7_scoring import _build_probe, _encode_texts  # type: ignore

logger = logging.getLogger(__name__)


# =============================================================================
# Metrics
# =============================================================================

# Metric key -> (callable(labels, scores) -> float, higher_is_better).
# All are rank-based, so cross-scorer comparison is valid even though scorers
# live on different scales (cosine in [0,1], Mahalanobis is an unbounded -dist).
_METRICS: dict[str, tuple] = {
    "auc":   (compute_auc, True),
    "pauc5": (lambda y, s: compute_pauc(y, s, max_fpr=0.05), True),
    "tpr1":  (lambda y, s: compute_tpr_at_fpr(y, s, 0.01), True),
    "tpr5":  (lambda y, s: compute_tpr_at_fpr(y, s, 0.05), True),
    "eer":   (compute_eer, False),  # lower is better; we report 1-EER in display
}


def _labels_scores(gen: np.ndarray, neg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.concatenate([np.ones_like(gen), np.zeros_like(neg)])
    scores = np.concatenate([gen, neg])
    return labels, scores


def _metric_on(gen: np.ndarray, neg: np.ndarray, key: str) -> float:
    if len(gen) == 0 or len(neg) == 0:
        return float("nan")
    fn, _ = _METRICS[key]
    y, s = _labels_scores(gen, neg)
    return float(fn(y, s))


# =============================================================================
# Bootstrap
# =============================================================================


def _paired_bootstrap(
    scorer_scores: dict[str, dict[str, np.ndarray]],
    split: str,
    metric_key: str,
    baseline: str,
    n_boot: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Paired bootstrap of one metric on one split across all scorers.

    `scorer_scores[name] = {"genuine": (G,), "other": (O,), "synthetic": (S,)}`.
    `split` selects the negative pool ("synthetic" or "other"); "all" pools both.

    For each replicate we draw ONE set of genuine indices and ONE set of
    negative indices and reuse them for every scorer, so deltas are measured on
    identical resamples (variance-reduced, the right way to compare scorers).

    Returns per-scorer: point, lo, hi (95% CI on the metric) and dlo, dhi, p_win
    (95% CI on metric - baseline_metric, and the bootstrap probability the
    scorer beats the baseline).
    """
    # Build the (genuine, negative) arrays per scorer for this split.
    pools: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, sc in scorer_scores.items():
        gen = sc["genuine"]
        if split == "synthetic":
            neg = sc["synthetic"]
        elif split == "other":
            neg = sc["other"]
        else:  # all
            neg = np.concatenate([sc["other"], sc["synthetic"]])
        pools[name] = (gen, neg)

    _, higher_better = _METRICS[metric_key]
    g_n = len(next(iter(pools.values()))[0])
    n_n = len(next(iter(pools.values()))[1])
    point = {name: _metric_on(g, n, metric_key) for name, (g, n) in pools.items()}

    out: dict[str, dict[str, float]] = {
        name: {"point": point[name]} for name in pools
    }
    if n_boot <= 0 or g_n == 0 or n_n == 0:
        for name in pools:
            out[name].update(lo=float("nan"), hi=float("nan"),
                             dlo=float("nan"), dhi=float("nan"), p_win=float("nan"))
        return out

    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {name: [] for name in pools}
    deltas: dict[str, list[float]] = {name: [] for name in pools}
    base_g, base_n = pools[baseline]

    for _ in range(n_boot):
        gi = rng.integers(0, g_n, g_n)
        ni = rng.integers(0, n_n, n_n)
        base_val = _metric_on(base_g[gi], base_n[ni], metric_key)
        for name, (g, n) in pools.items():
            v = _metric_on(g[gi], n[ni], metric_key)
            samples[name].append(v)
            # Delta oriented so "positive = better than baseline" regardless of
            # whether the metric is maximised (auc) or minimised (eer).
            d = (v - base_val) if higher_better else (base_val - v)
            deltas[name].append(d)

    for name in pools:
        arr = np.array(samples[name])
        darr = np.array(deltas[name])
        out[name].update(
            lo=float(np.percentile(arr, 2.5)),
            hi=float(np.percentile(arr, 97.5)),
            dlo=float(np.percentile(darr, 2.5)),
            dhi=float(np.percentile(darr, 97.5)),
            p_win=float((darr > 0).mean()),
        )
    return out


# =============================================================================
# Core: encode + fit bank + score every pool
# =============================================================================


def _score_all(
    encoder, device: str, probe: dict, seed: int, ewma_alpha: float,
    crop_syn: bool = False,
) -> tuple[dict[str, dict[str, np.ndarray]], float]:
    """Encode pools, fit a ProfileBank, return per-scorer pool scores + mean LW-α.

    scorer_scores[name] = {"genuine", "other", "synthetic"} arrays of scores.
    With crop_syn=True an extra "syn_crop" pool is scored: each synthetic
    impostor cropped to a random 5–60-word span (mirroring train-time crop
    augmentation). The standard synthetic pool has no emails under 26 words,
    so without this pool short-query forgery is unmeasured
    (docs/v9_lineage_results_analysis.md §4).
    """
    enroll_emb = _encode_texts(encoder, probe["enroll_texts"], device)
    gen_emb = _encode_texts(encoder, probe["gen_texts"], device)
    oth_emb = _encode_texts(encoder, probe["other_texts"], device)
    syn_emb = (
        _encode_texts(encoder, probe["syn_texts"], device)
        if probe["syn_texts"] else np.empty((0, gen_emb.shape[1]))
    )
    syn_crop_emb = np.empty((0, gen_emb.shape[1]))
    if crop_syn and probe["syn_texts"]:
        import random as _random
        from email_fraud.data.augment import random_word_crop
        crng = _random.Random(seed)
        cropped = [random_word_crop(t, crng, 5, 60) for t in probe["syn_texts"]]
        syn_crop_emb = _encode_texts(encoder, cropped, device)

    bank = ProfileBank(ewma_alpha=ewma_alpha).fit(enroll_emb, probe["enroll_sids"])
    mean_lw_alpha = _mean_lw_shrinkage(bank)

    # Each other-sender impostor gets a random claimed profiled sender, mirroring
    # CentroidProbe.evaluate so the (query, claimed_sender) tuple is defined.
    rng = np.random.default_rng(seed)
    chosen = probe["chosen_senders"]
    oth_claimed = [chosen[i] for i in rng.integers(0, len(chosen), len(oth_emb))]
    gen_claimed = list(probe["gen_sids"])
    syn_claimed = list(probe["syn_sids"])

    scorer_scores: dict[str, dict[str, np.ndarray]] = {}
    for name in SCORERS:
        scorer_scores[name] = {
            "genuine": score_pool(bank, gen_emb, gen_claimed, name),
            "other": score_pool(bank, oth_emb, oth_claimed, name),
            "synthetic": (
                score_pool(bank, syn_emb, syn_claimed, name)
                if len(syn_emb) else np.array([])
            ),
        }
        if len(syn_crop_emb):
            # Crops keep the mimicked sender id — claimed senders unchanged.
            scorer_scores[name]["syn_crop"] = score_pool(
                bank, syn_crop_emb, syn_claimed, name
            )
    return scorer_scores, mean_lw_alpha


def _operating_point_extras(sc: dict[str, np.ndarray]) -> dict[str, float]:
    """Deployment-relevant point estimates the rank metrics hide.

    fpr_other_at_{1,5}: other-sender (real human, wrong claimed sender) accept
    rate when the threshold is anchored at 1%/5% FPR on the synthetic pool —
    the cost the genuine-vs-synthetic split never shows (the 2026-06-10 v9
    checkpoint had tpr5=0.837 *and* fpr_other_at_5=0.595).
    auc_g_other: genuine-vs-other ranking quality.
    {auc,tpr1,tpr5}_crop: same-tail metrics against cropped (short) synthetics.
    """
    out: dict[str, float] = {}
    gen, oth, syn = sc["genuine"], sc["other"], sc["synthetic"]
    if len(syn) and len(oth):
        for fpr, key in [(0.01, "fpr_other_at_1"), (0.05, "fpr_other_at_5")]:
            thr = np.quantile(syn, 1.0 - fpr, method="higher")
            out[key] = float((oth >= thr).mean())
    if len(oth):
        out["auc_g_other"] = _metric_on(gen, oth, "auc")
    crop = sc.get("syn_crop")
    if crop is not None and len(crop):
        out["auc_crop"] = _metric_on(gen, crop, "auc")
        out["tpr1_crop"] = _metric_on(gen, crop, "tpr1")
        out["tpr5_crop"] = _metric_on(gen, crop, "tpr5")
    return out


def _mean_lw_shrinkage(bank: ProfileBank) -> float:
    """Force LW precision to be fit for every sender and average the shrinkage.

    ProfileBank fits precision lazily and doesn't keep the shrinkage coefficient,
    so we refit cheaply here purely for the diagnostic (matches V7.2's LW-α col).
    """
    from sklearn.covariance import LedoitWolf

    alphas = []
    for s in bank.stats.values():
        if s.k >= 2:
            alphas.append(float(LedoitWolf().fit(s.embs).shrinkage_))
    return float(np.mean(alphas)) if alphas else float("nan")


# =============================================================================
# Main
# =============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/experiments/v7_luar_lora_syn_mahal_eval.yaml")
    p.add_argument("--checkpoint", required=True, help="Path to a trained checkpoint .pt")
    p.add_argument("--out-dir", default="results/v7")
    p.add_argument("--tag", default="adaptive_scorer_ablation")
    p.add_argument("--split", default="synthetic", choices=["synthetic", "other", "all"],
                   help="Negative pool the bootstrap + ranking use (synthetic = the hard BEC case).")
    p.add_argument("--rank-by", default="tpr1", choices=list(_METRICS),
                   help="Metric to rank/recommend on (tpr1 = TPR@1%%FPR, the conservative held-FPR goal).")
    p.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap replicates (0 disables CIs).")
    p.add_argument("--ewma-alpha", type=float, default=0.1, help="EWMA alpha for the ewma_* scorers.")
    p.add_argument("--crop-syn", action="store_true",
                   help="Also score a short-impostor pool: each synthetic email cropped "
                        "to a random 5-60-word span (the standard pool has no email "
                        "under 26 words, so short-query forgery is otherwise unmeasured).")
    p.add_argument("--k-sweep", default=None,
                   help="Comma-separated enroll sizes for a point-estimate sweep, e.g. 4,8,16,25.")
    # Expanded 2026-06-09 to match the bigger CentroidProbe (configs/base.yaml
    # `probe:`): the old 30×4/200/200 probe gave ±0.13-wide tpr1 CIs — wider
    # than most candidate-scorer deltas. Builders cap to availability, so these
    # degrade gracefully on small splits.
    p.add_argument("--n-profile-senders", type=int, default=60)
    p.add_argument("--n-enroll", type=int, default=8)
    p.add_argument("--n-query", type=int, default=6)
    p.add_argument("--n-other", type=int, default=600)
    p.add_argument("--n-synth", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--wandb", action="store_true",
        help="Log the scorer table + recommendation to W&B. Honors WANDB_RUN_GROUP / "
             "WANDB_JOB_TYPE / WANDB_NAME env vars for sectioning.",
    )
    return p.parse_args()


def _load_encoder(cfg, checkpoint: str, device: str):
    EncoderClass = resolve_component("encoder", cfg.encoder.name)
    encoder = EncoderClass(cfg.encoder)
    ckpt = Path(checkpoint)
    if not ckpt.is_absolute():
        ckpt = _PROJECT_ROOT / ckpt
    payload = torch.load(str(ckpt), map_location=device, weights_only=False)
    encoder.load_state_dict(payload["model_state_dict"])
    encoder.to(device)
    encoder.eval()
    logger.info("Loaded epoch %s from %s", payload.get("epoch"), ckpt)
    return encoder


def main() -> None:
    args = parse_args()
    setup_logging()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = _PROJECT_ROOT / cfg_path
    cfg = load_config(str(cfg_path))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    encoder = _load_encoder(cfg, args.checkpoint, device)
    train_ds = EnronDataset(cfg.data, split="train")
    val_ds = EnronDataset(cfg.data, split="validation")

    def build_probe(n_enroll: int) -> dict:
        return _build_probe(
            train_ds, val_ds,
            syn_path=cfg.data.augmentation.synthetic_path,
            n_profile_senders=args.n_profile_senders,
            n_enroll=n_enroll, n_query=args.n_query,
            n_other=args.n_other, n_synth=args.n_synth, seed=args.seed,
        )

    # ---- Primary run at --n-enroll, with bootstrap CIs ----
    probe = build_probe(args.n_enroll)
    logger.info(
        "Probe: %d senders × %d enroll | %d genuine, %d other, %d synthetic",
        len(probe["chosen_senders"]), args.n_enroll,
        len(probe["gen_texts"]), len(probe["other_texts"]), len(probe["syn_texts"]),
    )
    scorer_scores, lw_alpha = _score_all(
        encoder, device, probe, args.seed, args.ewma_alpha, crop_syn=args.crop_syn,
    )
    extras = {name: _operating_point_extras(scorer_scores[name]) for name in SCORERS}

    # Bootstrap every metric on the chosen split.
    boot: dict[str, dict[str, dict[str, float]]] = {}
    for mkey in _METRICS:
        boot[mkey] = _paired_bootstrap(
            scorer_scores, args.split, mkey, BASELINE, args.bootstrap, args.seed,
        )

    rank_metric = args.rank_by
    _, higher_better = _METRICS[rank_metric]
    # Rank by the bootstrap point estimate of the ranking metric.
    ranked = sorted(
        SCORERS,
        key=lambda n: boot[rank_metric][n]["point"],
        reverse=higher_better,
    )

    _print_table(args, boot, ranked, lw_alpha, probe, extras)
    recommendation = _recommend(boot, ranked, rank_metric, BASELINE)

    # ---- Optional K-sweep (point estimates only) ----
    k_sweep_rows = []
    if args.k_sweep:
        ks = [int(x) for x in args.k_sweep.split(",")]
        logger.info("K-sweep over %s (point estimates, no bootstrap)", ks)
        print("\nK-sweep — %s on genuine-vs-%s (point estimate):" % (rank_metric, args.split))
        header = f"{'scorer':22s}" + "".join(f"{('K=%d' % k):>10s}" for k in ks)
        print(header)
        print("-" * len(header))
        per_k: dict[int, dict[str, dict[str, np.ndarray]]] = {}
        for k in ks:
            pk = build_probe(k)
            ss, _ = _score_all(encoder, device, pk, args.seed, args.ewma_alpha)
            per_k[k] = ss
        for name in ranked:
            vals = {}
            line = f"{name:22s}"
            for k in ks:
                ss = per_k[k]
                neg = (ss[name]["synthetic"] if args.split == "synthetic"
                       else ss[name]["other"] if args.split == "other"
                       else np.concatenate([ss[name]["other"], ss[name]["synthetic"]]))
                v = _metric_on(ss[name]["genuine"], neg, rank_metric)
                vals[k] = v
                line += f"{v:>10.4f}"
            print(line)
            k_sweep_rows.append({"scorer": name, **{f"K={k}": vals[k] for k in ks}})

    summary = _write_outputs(
        args, cfg_path, boot, ranked, lw_alpha, probe, recommendation, k_sweep_rows,
        extras,
    )

    if args.wandb:
        _log_wandb(cfg, args, summary, ranked)

    print("\n" + "=" * 78)
    print("RECOMMENDATION:", recommendation["text"])
    print("=" * 78)


def _print_table(args, boot, ranked, lw_alpha, probe, extras) -> None:
    print()
    print(f"Adaptive-scorer ablation — split=genuine-vs-{args.split}, "
          f"K_enroll={args.n_enroll}, {len(probe['chosen_senders'])} senders, "
          f"mean LW-α={lw_alpha:.3f}, bootstrap={args.bootstrap}")
    print(f"(baseline = {BASELINE}; Δ columns are vs that baseline on the ranking split)\n")
    rb = args.rank_by
    has_crop = any("tpr1_crop" in extras[n] for n in ranked)
    hdr = (f"{'scorer':22s} {'AUC':>16s} {'pAUC5':>8s} {'TPR@1%':>8s} "
           f"{'TPR@5%':>8s} {'1-EER':>7s} {'FPRoth@5':>9s}"
           + (f" {'TPR1crop':>9s}" if has_crop else "")
           + f" {('Δ'+rb+' [95% CI]'):>22s} {'P(win)':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for name in ranked:
        auc = boot["auc"][name]
        auc_ci = f"{auc['point']:.3f} [{auc['lo']:.3f},{auc['hi']:.3f}]"
        eer = boot["eer"][name]["point"]
        d = boot[rb][name]
        dci = f"{_signed(d['dlo'])},{_signed(d['dhi'])}"
        star = "" if name == BASELINE else (" *" if (d["dlo"] > 0) else "")
        ex = extras[name]
        fpr_oth = ex.get("fpr_other_at_5")
        crop_col = (f" {ex.get('tpr1_crop', float('nan')):>9.3f}" if has_crop else "")
        print(f"{name:22s} {auc_ci:>16s} "
              f"{boot['pauc5'][name]['point']:>8.3f} "
              f"{boot['tpr1'][name]['point']:>8.3f} "
              f"{boot['tpr5'][name]['point']:>8.3f} "
              f"{1.0 - eer:>7.3f} "
              f"{(f'{fpr_oth:.3f}' if fpr_oth is not None else 'nan'):>9s}"
              f"{crop_col} "
              f"{('['+dci+']'):>22s} "
              f"{d['p_win']:>7.2f}{star}")
    print("\n  *  = Δ vs baseline 95% CI excludes 0 on the ranking metric "
          f"({rb}) → significantly better.")
    print("  FPRoth@5 = other-sender accept rate at the 5% FPR_syn threshold "
          "(watch for Goodharting — see docs/v9_lineage_results_analysis.md §3).")


def _signed(x: float) -> str:
    return f"{x:+.3f}" if x == x else "  nan"


def _recommend(boot, ranked, rank_metric, baseline) -> dict:
    """Pick the scorer to adopt: best ranking-metric point estimate whose delta
    vs baseline CI excludes 0. If none is significant, keep the baseline."""
    best = ranked[0]
    d = boot[rank_metric][best]
    significant = d["dlo"] == d["dlo"] and d["dlo"] > 0  # not-nan and >0
    if best == baseline or not significant:
        if best == baseline:
            text = (f"keep '{baseline}' — it ranks #1 on {rank_metric} and no other "
                    f"scorer significantly beats it.")
        else:
            text = (f"keep '{baseline}' — top scorer '{best}' leads on {rank_metric} "
                    f"point estimate (Δ={d['point'] - boot[rank_metric][baseline]['point']:+.3f}) "
                    f"but its Δ 95% CI [{d['dlo']:+.3f},{d['dhi']:+.3f}] includes 0 "
                    f"(P(win)={d['p_win']:.2f}) — not distinguishable from noise.")
        winner = baseline
    else:
        text = (f"adopt '{best}' — best {rank_metric} ({d['point']:.3f}) with a Δ-vs-"
                f"baseline 95% CI of [{d['dlo']:+.3f},{d['dhi']:+.3f}] (P(win)={d['p_win']:.2f}), "
                f"i.e. significantly better than '{baseline}'.")
        winner = best
    return {"winner": winner, "rank_metric": rank_metric, "text": text}


def _write_outputs(args, cfg_path, boot, ranked, lw_alpha, probe,
                   recommendation, k_sweep_rows, extras) -> None:
    out_dir = _PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in ranked:
        row = {"scorer": name, "is_baseline": name == BASELINE}
        for mkey in _METRICS:
            b = boot[mkey][name]
            row[f"{mkey}"] = b["point"]
            row[f"{mkey}_lo"] = b["lo"]
            row[f"{mkey}_hi"] = b["hi"]
            row[f"{mkey}_dlo"] = b["dlo"]
            row[f"{mkey}_dhi"] = b["dhi"]
            row[f"{mkey}_pwin"] = b["p_win"]
        row.update(extras[name])  # fpr_other_at_*, auc_g_other, *_crop
        rows.append(row)

    summary = {
        "config": str(cfg_path),
        "checkpoint": args.checkpoint,
        "split": args.split,
        "rank_by": args.rank_by,
        "bootstrap": args.bootstrap,
        "baseline": BASELINE,
        "mean_lw_shrinkage": lw_alpha,
        "probe": {
            "n_profile_senders": len(probe["chosen_senders"]),
            "n_enroll": args.n_enroll,
            "n_genuine": len(probe["gen_texts"]),
            "n_other": len(probe["other_texts"]),
            "n_synthetic": len(probe["syn_texts"]),
            "seed": args.seed,
        },
        "recommendation": recommendation,
        "rows": rows,
        "k_sweep": k_sweep_rows,
    }
    json_path = out_dir / f"{args.tag}.json"
    with json_path.open("w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Saved JSON → %s", json_path)

    _EXTRA_KEYS = ("fpr_other_at_1", "fpr_other_at_5", "auc_g_other",
                   "auc_crop", "tpr1_crop", "tpr5_crop")
    csv_keys = (["scorer", "is_baseline"]
                + [f"{m}{suf}" for m in _METRICS
                   for suf in ("", "_lo", "_hi", "_dlo", "_dhi", "_pwin")]
                + [k for k in _EXTRA_KEYS if any(k in r for r in rows)])
    csv_path = out_dir / f"{args.tag}.csv"
    with csv_path.open("w") as fh:
        fh.write(",".join(csv_keys) + "\n")
        for r in rows:
            fh.write(",".join(
                f"{r[k]:.4f}" if isinstance(r.get(k), float) else str(r.get(k, ""))
                for k in csv_keys) + "\n")
    logger.info("Saved CSV  → %s", csv_path)
    return summary


def _log_wandb(cfg, args, summary, ranked) -> None:
    """Log the scorer ablation as one W&B run: a per-scorer Table plus summary
    metrics for the baseline and the winner.

    Sectioning (group / job_type / display name) is taken from the
    WANDB_RUN_GROUP / WANDB_JOB_TYPE / WANDB_NAME environment variables — we do
    not pass those kwargs, so wandb picks them up natively. Tags come from the
    --config's wandb.tags (carrying the dataset-version tag, e.g. syn-v2) plus a
    "scorer-ablation" marker so the run is filterable.
    """
    import wandb

    rows = summary["rows"]
    rec = summary["recommendation"]
    baseline = summary["baseline"]
    winner = rec["winner"]

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        tags=[*cfg.wandb.tags, "scorer-ablation", f"split-{args.split}"],
        notes=f"Data-dependent vs fixed scorer ablation on {args.checkpoint}",
        config={
            "checkpoint": args.checkpoint,
            "split": args.split,
            "rank_by": args.rank_by,
            "bootstrap": args.bootstrap,
            "baseline": baseline,
            "n_profile_senders": summary["probe"]["n_profile_senders"],
            "n_enroll": summary["probe"]["n_enroll"],
        },
    )

    # Full per-scorer comparison table (data-dependent rows vs the fixed baseline).
    cols = ["scorer", "is_baseline", "auc", "pauc5", "tpr1", "tpr5", "eer",
            "tpr1_dlo", "tpr1_dhi", "tpr1_pwin"]
    table = wandb.Table(columns=cols)
    by_name = {r["scorer"]: r for r in rows}
    for name in ranked:
        r = by_name[name]
        table.add_data(*[r.get(c) for c in cols])
    wandb.log({"scorer_ablation": table})

    # Headline scalars so the runs are comparable at a glance across datasets.
    rank = args.rank_by
    wandb.summary.update({
        "winner": winner,
        "winner_is_data_dependent": winner != baseline,
        f"baseline/{rank}": by_name[baseline][rank],
        f"winner/{rank}": by_name[winner][rank],
        f"delta_{rank}_winner_vs_baseline": by_name[winner][rank] - by_name[baseline][rank],
        "winner_delta_ci_lo": by_name[winner].get(f"{rank}_dlo"),
        "winner_delta_ci_hi": by_name[winner].get(f"{rank}_dhi"),
        "winner_p_win": by_name[winner].get(f"{rank}_pwin"),
        "recommendation": rec["text"],
    })
    run.finish()


if __name__ == "__main__":
    main()
