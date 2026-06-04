"""Ablation: which scoring rule separates genuine from impostor best?

Compares the production score (`baseline_linear_z3`) against the data-driven
prototypes in `email_fraud.scoring.adaptive` on a real checkpoint, on identical
embeddings, so any difference is attributable to the *scoring rule* alone.

What it does
------------
1. Load an encoder from a checkpoint (auto-reads the experiment config).
2. Encode a profiling split once (embeddings cached to disk by checkpoint+split).
3. Build, per profiled sender: `n_enroll` enrollment emails -> profile, and
   `n_query` held-out genuine queries. Impostors are genuine emails from
   NON-profiled senders, each assigned to a random profiled sender (the same
   protocol CentroidProbe uses).
4. Fit one ProfileBank, sweep every scorer, and report AUC / pAUC@5% /
   TPR@1%FPR / EER with bootstrap 95% CIs, plus a paired bootstrap test of
   delta-AUC vs the baseline (the number that tells you if a change is real).
5. Optional --by-tier breakdown so you can see where each scorer wins.

Usage
-----
    python scripts/ablation_scoring.py --run runs/minilm_m5/2026-04-29_19-12-10
    python scripts/ablation_scoring.py --run <dir> --split validation --by-tier
    python scripts/ablation_scoring.py --run <dir> --n-enroll 8 --n-query 4 \
        --n-profile-senders 80 --bootstrap 2000

Profiling the *validation* split (default) enrolls senders the encoder never
trained on — the honest deployment scenario.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import email_fraud.encoders  # noqa: F401,E402  trigger @register
import email_fraud.heads     # noqa: F401,E402
import email_fraud.losses    # noqa: F401,E402
from email_fraud.config import load_config  # noqa: E402
from email_fraud.registry import resolve as resolve_component  # noqa: E402
from email_fraud.scoring.adaptive import (  # noqa: E402
    BASELINE,
    SCORERS,
    ProfileBank,
    score_pool,
)
from email_fraud.scoring.metrics import (  # noqa: E402
    compute_auc,
    compute_eer,
    compute_pauc,
    compute_tpr_at_fpr,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("ablation")

_CACHE_DIR = _PROJECT_ROOT / "runs" / "_ablation_cache"


# ---------------------------------------------------------------------------
# Checkpoint / encoder
# ---------------------------------------------------------------------------


def _resolve_checkpoint(run: str | None, checkpoint: str | None) -> Path:
    if checkpoint:
        p = Path(checkpoint)
        return p if p.is_absolute() else _PROJECT_ROOT / p
    run_dir = Path(run) if Path(run).is_absolute() else _PROJECT_ROOT / run
    for name in ("checkpoint_best.pt", "checkpoint_last.pt"):
        if (run_dir / name).exists():
            return run_dir / name
    epochs = sorted(run_dir.glob("checkpoint_epoch_*.pt"))
    if not epochs:
        raise FileNotFoundError(f"No checkpoint under {run_dir}")
    return epochs[-1]


def _find_config(ckpt_path: Path, override: str | None) -> Path:
    if override:
        return Path(override) if Path(override).is_absolute() else _PROJECT_ROOT / override
    local = ckpt_path.parent / "config.yaml"
    if local.exists():
        return local
    # Fall back to the experiment config named after the run's parent dir.
    exp = ckpt_path.parent.parent.name
    guess = _PROJECT_ROOT / "configs" / "experiments" / f"{exp}.yaml"
    if guess.exists():
        logger.info("No config.yaml beside checkpoint; using %s", guess.relative_to(_PROJECT_ROOT))
        return guess
    raise FileNotFoundError(
        f"No config found. Pass --config explicitly (looked for {local} and {guess})."
    )


def _load_encoder(ckpt_path: Path, cfg, device: str):
    EncoderClass = resolve_component("encoder", cfg.encoder.name)
    encoder = EncoderClass(cfg.encoder)
    ckpt = torch.load(str(ckpt_path), map_location=device)
    encoder.load_state_dict(ckpt["model_state_dict"])
    encoder.eval().to(device)
    logger.info("Loaded %s (epoch %s)", ckpt_path.name, ckpt.get("epoch", "?"))
    return encoder


@torch.no_grad()
def _encode(encoder, texts: list[str], device: str, batch_size: int) -> np.ndarray:
    out: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        tok = encoder.tokenize(batch)
        tok = {k: v.to(device) for k, v in tok.items()}
        out.append(encoder.encode(**tok).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float64)


# ---------------------------------------------------------------------------
# Data / embedding cache
# ---------------------------------------------------------------------------


def _load_split(processed_dir: str, split: str) -> tuple[list[str], list[str]]:
    from datasets import load_from_disk

    ds = load_from_disk(processed_dir)[split]
    return list(ds["text"]), list(ds["sender_id"])


def _embed_split_cached(
    encoder, ckpt_path: Path, processed_dir: str, split: str, device: str, batch_size: int
) -> tuple[np.ndarray, list[str]]:
    """Encode an entire split once; cache (embeddings, sender_ids) to disk."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(f"{ckpt_path}|{processed_dir}|{split}".encode()).hexdigest()[:16]
    cache = _CACHE_DIR / f"{ckpt_path.parent.parent.name}_{split}_{key}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        logger.info("Loaded cached embeddings %s (%d emails)", cache.name, len(z["sender_ids"]))
        return z["embeddings"], list(z["sender_ids"])
    texts, senders = _load_split(processed_dir, split)
    logger.info("Encoding %d emails from split '%s' ...", len(texts), split)
    embs = _encode(encoder, texts, device, batch_size)
    np.savez_compressed(cache, embeddings=embs, sender_ids=np.array(senders, dtype=object))
    logger.info("Cached -> %s", cache.name)
    return embs, senders


# ---------------------------------------------------------------------------
# Probe construction
# ---------------------------------------------------------------------------


def build_probe(
    embs: np.ndarray,
    senders: list[str],
    n_profile_senders: int,
    n_enroll: int,
    n_query: int,
    n_impostor: int,
    seed: int,
    vary_enroll: bool = False,
    max_enroll: int = 30,
):
    """Return (enroll_embs, enroll_sids, gen_embs, gen_sids, imp_embs, imp_sids).

    With vary_enroll, each sender's enrollment count is sampled from
    [2, min(available - n_query, max_enroll)] so profiles span confidence tiers
    (low/med/high/vhigh) — needed to exercise the k-dependent adaptive scorers.
    """
    rng = random.Random(seed)
    by_sender: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(senders):
        by_sender[s].append(i)

    need = (2 if vary_enroll else n_enroll) + n_query
    eligible = [s for s, idx in by_sender.items() if len(idx) >= need]
    if len(eligible) < n_profile_senders:
        logger.warning(
            "Only %d senders have >= %d emails (wanted %d profiles); using all eligible.",
            len(eligible), need, n_profile_senders,
        )
        n_profile_senders = len(eligible)
    profiled = set(rng.sample(eligible, n_profile_senders))

    enr_e, enr_s, gen_e, gen_s = [], [], [], []
    for sid in profiled:
        idx = by_sender[sid][:]
        rng.shuffle(idx)
        if vary_enroll:
            hi = min(len(idx) - n_query, max_enroll)
            k_enroll = rng.randint(2, max(2, hi))
        else:
            k_enroll = n_enroll
        for i in idx[:k_enroll]:
            enr_e.append(embs[i]); enr_s.append(sid)
        for i in idx[k_enroll : k_enroll + n_query]:
            gen_e.append(embs[i]); gen_s.append(sid)

    # Impostors: emails from NON-profiled senders, each assigned to a random
    # profiled sender so the claimed identity exists in the bank.
    other_idx = [i for i, s in enumerate(senders) if s not in profiled]
    rng.shuffle(other_idx)
    profiled_list = sorted(profiled)
    imp_e, imp_s = [], []
    for i in other_idx[:n_impostor]:
        imp_e.append(embs[i]); imp_s.append(rng.choice(profiled_list))

    enroll_mode = f"vary[2,{max_enroll}]" if vary_enroll else f"{n_enroll}"
    logger.info(
        "Probe: %d profiles x %s enroll | %d genuine, %d impostor queries",
        len(profiled), enroll_mode, len(gen_s), len(imp_s),
    )
    return (
        np.array(enr_e), enr_s,
        np.array(gen_e), gen_s,
        np.array(imp_e), imp_s,
    )


# ---------------------------------------------------------------------------
# Metrics + significance
# ---------------------------------------------------------------------------


def _metrics(genuine: np.ndarray, impostor: np.ndarray) -> dict[str, float]:
    nan = {"AUC": float("nan"), "pAUC@5%": float("nan"), "TPR@1%": float("nan"),
           "TPR@5%": float("nan"), "EER": float("nan")}
    if len(genuine) == 0 or len(impostor) == 0:
        return nan
    labels = np.concatenate([np.ones_like(genuine), np.zeros_like(impostor)])
    scores = np.concatenate([genuine, impostor])
    return {
        "AUC": compute_auc(labels, scores),
        "pAUC@5%": compute_pauc(labels, scores, max_fpr=0.05),
        "TPR@1%": compute_tpr_at_fpr(labels, scores, target_fpr=0.01),
        "TPR@5%": compute_tpr_at_fpr(labels, scores, target_fpr=0.05),
        "EER": compute_eer(labels, scores),
    }


def _bootstrap_auc(
    genuine: np.ndarray, impostor: np.ndarray, n: int, seed: int
) -> tuple[float, float]:
    """Percentile 95% CI for AUC by resampling each pool with replacement."""
    rng = np.random.default_rng(seed)
    ng, ni = len(genuine), len(impostor)
    aucs = np.empty(n)
    labels = np.concatenate([np.ones(ng), np.zeros(ni)])
    for b in range(n):
        gi = rng.integers(0, ng, ng)
        ii = rng.integers(0, ni, ni)
        scores = np.concatenate([genuine[gi], impostor[ii]])
        aucs[b] = compute_auc(labels, scores)
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def _paired_delta_auc(
    g_base: np.ndarray, i_base: np.ndarray,
    g_new: np.ndarray, i_new: np.ndarray,
    n: int, seed: int,
) -> tuple[float, float, float]:
    """Paired bootstrap of AUC(new) - AUC(base) on the SAME resampled queries.

    Returns (mean_delta, ci_lo, ci_hi). CI excluding 0 => the change is real.
    """
    rng = np.random.default_rng(seed)
    ng, ni = len(g_base), len(i_base)
    labels = np.concatenate([np.ones(ng), np.zeros(ni)])
    deltas = np.empty(n)
    for b in range(n):
        gi = rng.integers(0, ng, ng)
        ii = rng.integers(0, ni, ni)
        a_base = compute_auc(labels, np.concatenate([g_base[gi], i_base[ii]]))
        a_new = compute_auc(labels, np.concatenate([g_new[gi], i_new[ii]]))
        deltas[b] = a_new - a_base
    return float(deltas.mean()), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _tier_of(k: int) -> str:
    if k <= 4:
        return "low(1-4)"
    if k <= 9:
        return "med(5-9)"
    if k <= 24:
        return "high(10-24)"
    return "vhigh(25+)"


def run_ablation(args) -> None:
    ckpt_path = _resolve_checkpoint(args.run, args.checkpoint)
    cfg = load_config(str(_find_config(ckpt_path, args.config)))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    encoder = _load_encoder(ckpt_path, cfg, device)
    processed_dir = args.data_dir or cfg.data.processed_dir
    embs, senders = _embed_split_cached(
        encoder, ckpt_path, processed_dir, args.split, device, args.batch_size
    )

    enr_e, enr_s, gen_e, gen_s, imp_e, imp_s = build_probe(
        embs, senders,
        n_profile_senders=args.n_profile_senders,
        n_enroll=args.n_enroll, n_query=args.n_query,
        n_impostor=args.n_impostor, seed=args.seed,
        vary_enroll=args.vary_enroll, max_enroll=args.max_enroll,
    )

    bank = ProfileBank(ewma_alpha=args.ewma_alpha).fit(enr_e, enr_s)

    # Score every pool with every scorer (embeddings encoded once; cheap).
    scorer_names = list(SCORERS.keys())
    gen_scores = {s: score_pool(bank, gen_e, gen_s, s) for s in scorer_names}
    imp_scores = {s: score_pool(bank, imp_e, imp_s, s) for s in scorer_names}

    # ---- main table ----
    print("\n" + "=" * 96)
    print(f"  SCORING ABLATION  |  {ckpt_path.parent.parent.name}  |  split={args.split}  "
          f"|  {len(gen_s)} genuine / {len(imp_s)} impostor")
    print("=" * 96)
    header = f"  {'scorer':<22} {'AUC':>7} {'95% CI':>17} {'pAUC@5%':>8} {'TPR@1%':>8} {'TPR@5%':>8} {'EER':>7}"
    print(header)
    print("  " + "-" * 92)

    rows = []
    for name in scorer_names:
        m = _metrics(gen_scores[name], imp_scores[name])
        lo, hi = _bootstrap_auc(gen_scores[name], imp_scores[name], args.bootstrap, args.seed)
        rows.append((name, m, (lo, hi)))
    rows.sort(key=lambda r: r[1]["AUC"], reverse=True)
    for name, m, (lo, hi) in rows:
        star = "  *" if name == BASELINE else ""
        print(f"  {name:<22} {m['AUC']:>7.4f} [{lo:>6.4f},{hi:>6.4f}] "
              f"{m['pAUC@5%']:>8.4f} {m['TPR@1%']:>8.4f} {m['TPR@5%']:>8.4f} {m['EER']:>7.4f}{star}")
    print("  " + "-" * 92)
    print(f"  (* = production baseline '{BASELINE}'; rows sorted by AUC)")

    # ---- paired significance vs baseline ----
    print("\n  Paired bootstrap delta-AUC vs baseline (CI excluding 0 = significant):")
    g_b, i_b = gen_scores[BASELINE], imp_scores[BASELINE]
    deltas = []
    for name in scorer_names:
        if name == BASELINE:
            continue
        d, dlo, dhi = _paired_delta_auc(g_b, i_b, gen_scores[name], imp_scores[name], args.bootstrap, args.seed)
        deltas.append((name, d, dlo, dhi))
    deltas.sort(key=lambda r: r[1], reverse=True)
    for name, d, dlo, dhi in deltas:
        sig = "SIG" if (dlo > 0 or dhi < 0) else "  -"
        print(f"    {name:<22} dAUC={d:>+8.4f}  [{dlo:>+7.4f}, {dhi:>+7.4f}]  {sig}")

    # ---- calibration / reachability (what AUC can't see) ----
    # Per-sender calibration barely moves AUC by construction: AUC is a ranking
    # metric. Its real payoff is making the score USE the [0,1] range and mean
    # the same thing across senders (so one global threshold transfers). These
    # diagnostics are only meaningful for [0,1]-bounded scorers.
    bounded = ["baseline_linear_z3", "baseline_cosine", "z_global_cal",
               "z_persender_cal", "z_persender_sigmoid"]
    print("\n  Score reachability on the GENUINE pool (calibration health, [0,1] scorers):")
    print(f"    {'scorer':<22} {'mean':>7} {'p50':>7} {'p95':>7} {'%>0.8':>7} {'%>0.95':>8}")
    for name in bounded:
        g = gen_scores[name]
        print(f"    {name:<22} {g.mean():>7.3f} {np.percentile(g,50):>7.3f} "
              f"{np.percentile(g,95):>7.3f} {100*(g>0.8).mean():>6.1f}% {100*(g>0.95).mean():>7.1f}%")
    print("    (linear_z3's >0.95 is structurally ~0; a calibrated scorer should open up the upper range)")

    # Cross-sender threshold transfer: pick the per-sender-optimal-ish global
    # threshold and measure how consistent the genuine-acceptance is ACROSS
    # senders. Lower spread = the score means the same thing per sender.
    print("\n  Cross-sender consistency of genuine scores (std of per-sender mean; lower=better):")
    for name in bounded:
        per_sender_means = []
        gs = np.array(gen_s)
        for sid in set(gen_s):
            vals = gen_scores[name][gs == sid]
            if len(vals):
                per_sender_means.append(vals.mean())
        print(f"    {name:<22} std={np.std(per_sender_means):.4f}")

    # ---- optional per-tier breakdown ----
    if args.by_tier:
        print("\n  Per-tier AUC (genuine vs impostor, grouped by claimed sender's k):")
        # Each genuine/impostor query has a claimed sender -> tier.
        gen_tier = np.array([_tier_of(bank.stats[s].k) for s in gen_s])
        imp_tier = np.array([_tier_of(bank.stats[s].k) for s in imp_s])
        tiers = ["low(1-4)", "med(5-9)", "high(10-24)", "vhigh(25+)"]
        present = [t for t in tiers if (gen_tier == t).any() and (imp_tier == t).any()]
        head = "    " + f"{'scorer':<22}" + "".join(f"{t:>14}" for t in present)
        print(head)
        for name in scorer_names:
            cells = []
            for t in present:
                g = gen_scores[name][gen_tier == t]
                im = imp_scores[name][imp_tier == t]
                if len(g) and len(im):
                    cells.append(f"{_metrics(g, im)['AUC']:>14.4f}")
                else:
                    cells.append(f"{'n/a':>14}")
            print(f"    {name:<22}" + "".join(cells))
        mode = f"--vary-enroll [2,{args.max_enroll}]" if args.vary_enroll else f"fixed n_enroll={args.n_enroll}"
        print(f"    (tier from enrollment k; enrollment mode: {mode})")

    print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", help="Run directory; uses checkpoint_best.pt.")
    src.add_argument("--checkpoint", help="Path to a specific .pt checkpoint.")
    p.add_argument("--config", default=None, help="Override config path.")
    p.add_argument("--data-dir", default=None, help="Override processed dataset dir.")
    p.add_argument("--split", default="train", choices=["train", "validation", "test"],
                   help="Split to profile/enroll from (default: train; it has the most senders, "
                        "so the impostor pool is non-empty. validation/test have too few senders).")
    p.add_argument("--n-profile-senders", type=int, default=60)
    p.add_argument("--n-enroll", type=int, default=8, help="Enrollment emails per sender (fixed mode).")
    p.add_argument("--vary-enroll", action="store_true",
                   help="Randomize enrollment k per sender so profiles span confidence tiers.")
    p.add_argument("--max-enroll", type=int, default=30, help="Cap on enrollment k under --vary-enroll.")
    p.add_argument("--n-query", type=int, default=4, help="Held-out genuine queries per sender.")
    p.add_argument("--n-impostor", type=int, default=400, help="Total impostor queries.")
    p.add_argument("--ewma-alpha", type=float, default=0.1)
    p.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap resamples for CIs.")
    p.add_argument("--by-tier", action="store_true", help="Print per-tier AUC breakdown.")
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    run_ablation(parse_args())
