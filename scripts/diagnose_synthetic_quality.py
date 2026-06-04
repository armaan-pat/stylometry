"""Diagnose cross-register synthetic email quality before retraining.

The key question this script answers:
    Do the LLM-generated cross-register emails actually land near their
    claimed sender's centroid — or do they wander into generic space?

Why this matters
----------------
Cross-register synthetics are stored under the real sender_id so SupConLoss
treats them as positives.  If the LLM produced generic prose instead of
capturing the author's voice, those embeddings land far from the real centroid.
The loss then drags the centroid toward that wrong region — corrupting the
training signal rather than helping it.

This script uses the CURRENT checkpoint (no retraining) to measure three
things per cross-register synthetic email:

  rank_1        True if the true sender's centroid is the nearest among all
                profiled senders.  The go/no-go metric.

  sim_to_own    Cosine similarity to the true sender's centroid.

  delta         sim_to_own − mean(sim_to_all_other_centroids).
                Positive = good positive.  Negative = bad positive.

A cross-register email with rank_1=False and delta<0 should be filtered out
before training.  Use --save-filtered to write a cleaned Arrow dataset.

Rule of thumb (from SupConLoss sensitivity analysis):
  rank_1_rate >= 0.55 AND mean_delta >= 0.05  →  safe to train with
  rank_1_rate 0.40-0.55 OR mean_delta 0.00-0.05  →  use --sim-threshold 0.X
  rank_1_rate < 0.40 OR mean_delta < 0.00  →  cross-register quality too low;
                                               set --cross-register-fraction 0
                                               until generation is re-tuned

Reports are printed to stdout and optionally saved as JSON.

Usage
-----
# Diagnose only
python scripts/diagnose_synthetic_quality.py \\
    --config  configs/experiments/v7_luar_lora_syn_mahal_eval.yaml \\
    --checkpoint runs/v7_luar_lora_syn_mahal/<ts>/checkpoint_epoch_150.pt \\
    --synthetic data/synthetic/enron_synthetic_v2 \\
    --data-dir  data/processed/enron

# Diagnose + write filtered dataset
python scripts/diagnose_synthetic_quality.py \\
    --config  configs/experiments/v7_luar_lora_syn_mahal_eval.yaml \\
    --checkpoint runs/v7_luar_lora_syn_mahal/<ts>/checkpoint_epoch_150.pt \\
    --synthetic data/synthetic/enron_synthetic_v2 \\
    --data-dir  data/processed/enron \\
    --save-filtered data/synthetic/enron_synthetic_v2_filtered \\
    --sim-threshold 0.10
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")

import email_fraud.encoders  # noqa: F401
import email_fraud.heads     # noqa: F401
import email_fraud.losses    # noqa: F401

from email_fraud.config import load_config
from email_fraud.registry import resolve


# ---------------------------------------------------------------------------
# Encoder helpers
# ---------------------------------------------------------------------------

def _load_encoder(cfg, checkpoint_path: str, device: str):
    EncoderClass = resolve("encoder", cfg.encoder.name)
    encoder = EncoderClass(cfg.encoder)
    ckpt = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(ckpt["model_state_dict"])
    encoder.eval().to(device)
    print(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')})")
    return encoder


@torch.no_grad()
def _encode_texts(
    encoder,
    texts: list[str],
    device: str,
    batch_size: int = 64,
) -> np.ndarray:
    """Encode a list of texts; returns (N, d) float32 numpy array (L2-normalised)."""
    saved_k = None
    if hasattr(encoder, "config") and hasattr(encoder.config, "episode_k"):
        saved_k = encoder.config.episode_k
        encoder.config.episode_k = 1

    all_embs = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        tok = encoder.tokenize(batch)
        tok = {k: v.to(device) for k, v in tok.items()}
        emb = encoder.encode(**tok)
        emb = F.normalize(emb, dim=-1)
        all_embs.append(emb.cpu())

    if saved_k is not None:
        encoder.config.episode_k = saved_k

    return torch.cat(all_embs, dim=0).numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Centroid building
# ---------------------------------------------------------------------------

def _build_centroids(
    embs: np.ndarray,
    sender_ids: list[str],
) -> dict[str, np.ndarray]:
    """Return L2-normalised per-sender centroid from email embeddings."""
    acc: dict[str, list[np.ndarray]] = defaultdict(list)
    for emb, sid in zip(embs, sender_ids):
        acc[sid].append(emb)

    centroids: dict[str, np.ndarray] = {}
    for sid, es in acc.items():
        c = np.mean(es, axis=0)
        norm = np.linalg.norm(c)
        centroids[sid] = c / norm if norm > 1e-9 else c
    return centroids


# ---------------------------------------------------------------------------
# Per-email quality metrics
# ---------------------------------------------------------------------------

def _score_against_centroids(
    emb: np.ndarray,
    true_sid: str,
    centroid_mat: np.ndarray,   # (n_senders, d)
    centroid_sids: list[str],
) -> dict:
    """Compute rank, sim_to_own, and delta for one embedding."""
    sims = centroid_mat @ emb  # (n_senders,)
    true_idx = centroid_sids.index(true_sid) if true_sid in centroid_sids else -1

    if true_idx == -1:
        return {
            "rank_1": False,
            "rank": None,
            "sim_to_own": float("nan"),
            "mean_sim_others": float("nan"),
            "delta": float("nan"),
            "in_centroid_pool": False,
        }

    sim_to_own = float(sims[true_idx])
    other_sims = np.delete(sims, true_idx)
    mean_sim_others = float(other_sims.mean())
    rank = int((sims > sim_to_own).sum()) + 1  # 1 = best

    return {
        "rank_1": rank == 1,
        "rank": rank,
        "sim_to_own": sim_to_own,
        "mean_sim_others": mean_sim_others,
        "delta": sim_to_own - mean_sim_others,
        "in_centroid_pool": True,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_header(title: str, width: int = 72) -> None:
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def _report_pool(
    label: str,
    results: list[dict],
    n_senders_in_pool: int,
) -> dict:
    """Print and return aggregate stats for a pool of emails."""
    in_pool = [r for r in results if r.get("in_centroid_pool", False)]
    if not in_pool:
        print(f"  {label}: no emails with known centroids.")
        return {}

    rank1_rate  = np.mean([r["rank_1"]      for r in in_pool])
    mean_sim    = np.mean([r["sim_to_own"]   for r in in_pool])
    mean_others = np.mean([r["mean_sim_others"] for r in in_pool])
    mean_delta  = np.mean([r["delta"]        for r in in_pool])
    p25_delta   = np.percentile([r["delta"]  for r in in_pool], 25)
    frac_neg    = np.mean([r["delta"] < 0    for r in in_pool])

    print(f"\n  {label}  (n={len(in_pool)}, senders in centroid pool={n_senders_in_pool})")
    print(f"    rank-1 rate          {rank1_rate:.3f}  ← go/no-go threshold ≥ 0.55")
    print(f"    mean sim to own      {mean_sim:.4f}")
    print(f"    mean sim to others   {mean_others:.4f}")
    print(f"    mean delta           {mean_delta:+.4f}  ← target ≥ +0.05")
    print(f"    25th-pct delta       {p25_delta:+.4f}")
    print(f"    frac with delta < 0  {frac_neg:.3f}  ← these are bad positives")

    return {
        "n": len(in_pool),
        "rank_1_rate": float(rank1_rate),
        "mean_sim_to_own": float(mean_sim),
        "mean_sim_others": float(mean_others),
        "mean_delta": float(mean_delta),
        "p25_delta": float(p25_delta),
        "frac_neg_delta": float(frac_neg),
    }


def _verdict(rank1_rate: float, mean_delta: float) -> str:
    if rank1_rate >= 0.55 and mean_delta >= 0.05:
        return "SAFE  — cross-register positives look good; proceed with training."
    elif rank1_rate >= 0.40 and mean_delta >= 0.00:
        return (
            "BORDERLINE  — acceptable but noisy; use --sim-threshold 0.05 to filter "
            "the worst positives before training."
        )
    else:
        return (
            "RISKY  — cross-register emails don't resemble their claimed sender. "
            "Filter with --sim-threshold 0.10 or lower --cross-register-fraction before training."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose cross-register synthetic quality before retraining"
    )
    parser.add_argument("--config",       required=True)
    parser.add_argument("--checkpoint",   required=True)
    parser.add_argument("--synthetic",    required=True,
                        help="Path to Arrow dataset produced by generate_synthetic_emails.py")
    parser.add_argument("--data-dir",     default="data/processed/enron",
                        help="Processed Enron Arrow DatasetDict directory")
    parser.add_argument("--n-senders",    type=int, default=50,
                        help="Max senders to build centroids from (default 50)")
    parser.add_argument("--n-real-per-sender", type=int, default=None,
                        help="Real emails per sender used to build centroid "
                             "(default: all available)")
    parser.add_argument("--batch-size",   type=int, default=64)
    parser.add_argument("--out-json",     default=None,
                        help="Save full report as JSON")
    parser.add_argument("--save-filtered", default=None,
                        help="Path to save filtered Arrow dataset "
                             "(keeps all hard_neg + cross_register rows above --sim-threshold)")
    parser.add_argument("--sim-threshold", type=float, default=0.0,
                        help="Minimum sim_to_own to keep a cross-register email "
                             "when --save-filtered is set (default 0.0 = keep all)")
    args = parser.parse_args()

    cfg    = load_config(str(_PROJECT_ROOT / args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ----------------------------------------------------------------
    # 1. Load training data → build centroids from real emails only
    # ----------------------------------------------------------------
    data_dir = _PROJECT_ROOT / args.data_dir
    print(f"\nLoading training split from {data_dir} ...")
    ds_dict = load_from_disk(str(data_dir))
    train_ds = ds_dict["train"]

    sender_to_texts: dict[str, list[str]] = defaultdict(list)
    for text, sid in zip(train_ds["text"], train_ds["sender_id"]):
        # Only real emails — skip any __syn entries if they crept in
        if not sid.endswith("__syn"):
            sender_to_texts[sid].append(text)

    all_senders = sorted(sender_to_texts.keys())[:args.n_senders]
    centroid_texts:   list[str] = []
    centroid_sids:    list[str] = []
    for sid in all_senders:
        pool = sender_to_texts[sid]
        if args.n_real_per_sender:
            pool = pool[:args.n_real_per_sender]
        centroid_texts.extend(pool)
        centroid_sids.extend([sid] * len(pool))

    print(f"Building centroids for {len(all_senders)} senders "
          f"({len(centroid_texts)} real emails total) ...")

    # ----------------------------------------------------------------
    # 2. Load checkpoint + encode
    # ----------------------------------------------------------------
    encoder = _load_encoder(cfg, str(_PROJECT_ROOT / args.checkpoint), device)

    from email_fraud.data.preprocessing import preprocess
    pp = cfg.data.preprocessing

    def _pp(texts: list[str]) -> list[str]:
        return [preprocess(t, pp) or t for t in texts]

    centroid_embs = _encode_texts(encoder, _pp(centroid_texts), device, args.batch_size)
    centroids     = _build_centroids(centroid_embs, centroid_sids)

    centroid_sid_list = list(centroids.keys())
    centroid_mat      = np.stack([centroids[s] for s in centroid_sid_list])  # (S, d)
    print(f"Centroid matrix: {centroid_mat.shape}")

    # ----------------------------------------------------------------
    # 3. Load synthetic dataset
    # ----------------------------------------------------------------
    syn_path = _PROJECT_ROOT / args.synthetic
    print(f"\nLoading synthetic dataset from {syn_path} ...")
    syn_ds = load_from_disk(str(syn_path))
    print(f"  Total rows: {len(syn_ds)}")

    has_mode_col = "generation_mode" in syn_ds.column_names
    has_ctx_col  = "context_register" in syn_ds.column_names

    if has_mode_col:
        modes = syn_ds["generation_mode"]
        cross_mask = [m == "cross_register" for m in modes]
        hard_mask  = [m == "hard_neg"       for m in modes]
    else:
        # Older dataset without generation_mode column: infer from sender_id suffix
        cross_mask = [not sid.endswith("__syn") for sid in syn_ds["sender_id"]]
        hard_mask  = [sid.endswith("__syn")     for sid in syn_ds["sender_id"]]
        print("  (No generation_mode column — inferring mode from sender_id suffix)")

    n_cross = sum(cross_mask)
    n_hard  = sum(hard_mask)
    print(f"  cross_register rows: {n_cross}")
    print(f"  hard_neg rows:       {n_hard}")

    if n_cross == 0:
        print("\nNo cross-register rows found.  Nothing to diagnose.")
        print("Re-generate with --cross-register-fraction > 0 and run this script again.")
        return

    # ----------------------------------------------------------------
    # 4. Encode cross-register and hard-neg emails
    # ----------------------------------------------------------------
    cross_indices = [i for i, m in enumerate(cross_mask) if m]
    hard_indices  = [i for i, m in enumerate(hard_mask)  if m]

    def _get_texts(indices: list[int]) -> list[str]:
        return _pp([syn_ds[i]["text"] for i in indices])

    def _get_senders(indices: list[int]) -> list[str]:
        return [syn_ds[i]["source_sender_id"] if "source_sender_id" in syn_ds.column_names
                else syn_ds[i]["sender_id"].replace("__syn", "")
                for i in indices]

    print(f"\nEncoding {n_cross} cross-register emails ...")
    cross_texts   = _get_texts(cross_indices)
    cross_senders = _get_senders(cross_indices)
    cross_embs    = _encode_texts(encoder, cross_texts, device, args.batch_size)

    print(f"Encoding {min(n_hard, n_cross)} hard-neg emails (matched sample) ...")
    hard_sample = hard_indices[:n_cross]  # compare equal-sized pools
    hard_texts   = _get_texts(hard_sample)
    hard_senders = _get_senders(hard_sample)
    hard_embs    = _encode_texts(encoder, hard_texts, device, args.batch_size)

    # Also encode a sample of real emails as the "ceiling" reference
    n_real_sample = min(n_cross, len(centroid_texts))
    import random
    rng = random.Random(42)
    real_sample_idx = rng.sample(range(len(centroid_texts)), n_real_sample)
    real_sample_texts  = _pp([centroid_texts[i] for i in real_sample_idx])
    real_sample_sids   = [centroid_sids[i] for i in real_sample_idx]
    print(f"Encoding {n_real_sample} real emails as reference ceiling ...")
    real_embs = _encode_texts(encoder, real_sample_texts, device, args.batch_size)

    # ----------------------------------------------------------------
    # 5. Score each email against all centroids
    # ----------------------------------------------------------------
    def _score_pool(embs, senders):
        return [
            _score_against_centroids(emb, sid, centroid_mat, centroid_sid_list)
            for emb, sid in zip(embs, senders)
        ]

    cross_results = _score_pool(cross_embs, cross_senders)
    hard_results  = _score_pool(hard_embs,  hard_senders)
    real_results  = _score_pool(real_embs,  real_sample_sids)

    # ----------------------------------------------------------------
    # 6. Report
    # ----------------------------------------------------------------
    n_s = len(centroid_sid_list)

    _print_header("CROSS-REGISTER SYNTHETIC QUALITY REPORT")
    print(f"\n  Centroid pool: {n_s} senders")

    real_stats   = _report_pool("Real emails (reference ceiling)",   real_results,   n_s)
    hard_stats   = _report_pool("Hard-neg synthetics (should be LOW)", hard_results, n_s)
    cross_stats  = _report_pool("Cross-register synthetics (need HIGH)", cross_results, n_s)

    # Per-register breakdown if available
    if has_ctx_col and has_mode_col:
        ctx_regs = syn_ds["context_register"]
        for ctx in ("formal", "casual", "mixed"):
            sub_idx = [i for j, i in enumerate(cross_indices)
                       if ctx_regs[i] == ctx]
            if not sub_idx:
                continue
            sub_results = [cross_results[j] for j, i in enumerate(cross_indices)
                           if ctx_regs[i] == ctx]
            _report_pool(f"Cross-register (context={ctx})", sub_results, n_s)

    # Verdict
    _print_header("VERDICT")
    r1  = cross_stats.get("rank_1_rate", 0.0)
    dlt = cross_stats.get("mean_delta",  -999.0)
    print(f"\n  {_verdict(r1, dlt)}")

    # Filtering recommendation
    _print_header("FILTERING RECOMMENDATION")
    frac_would_keep_0 = np.mean([r["delta"] >= 0.00 for r in cross_results
                                  if r.get("in_centroid_pool")])
    frac_would_keep_5 = np.mean([r["delta"] >= 0.05 for r in cross_results
                                  if r.get("in_centroid_pool")])
    frac_would_keep_1 = np.mean([r["sim_to_own"] >= 0.10 for r in cross_results
                                  if r.get("in_centroid_pool")])
    print(f"\n  Keeping cross-register rows where delta >= 0.00: {frac_would_keep_0:.1%} of rows")
    print(f"  Keeping cross-register rows where delta >= 0.05: {frac_would_keep_5:.1%} of rows")
    print(f"  Keeping cross-register rows where sim_to_own >= 0.10: {frac_would_keep_1:.1%} of rows")

    # ----------------------------------------------------------------
    # 7. Optionally save JSON report
    # ----------------------------------------------------------------
    report = {
        "checkpoint": args.checkpoint,
        "synthetic":  args.synthetic,
        "centroid_pool_size": n_s,
        "real_reference":    real_stats,
        "hard_neg":          hard_stats,
        "cross_register":    cross_stats,
        "filter_retention": {
            "delta_gte_0.00": float(frac_would_keep_0),
            "delta_gte_0.05": float(frac_would_keep_5),
            "sim_gte_0.10":   float(frac_would_keep_1),
        },
        "verdict": _verdict(r1, dlt),
    }
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"\nReport saved to {out}")

    # ----------------------------------------------------------------
    # 8. Optionally save filtered dataset
    # ----------------------------------------------------------------
    if args.save_filtered:
        threshold = args.sim_threshold
        print(f"\nFiltering: keeping hard_neg rows (all) + cross_register rows "
              f"where sim_to_own >= {threshold:.3f} ...")

        keep_mask: list[bool] = []
        cross_ptr = 0
        for i in range(len(syn_ds)):
            is_cross = cross_mask[i]
            if not is_cross:
                keep_mask.append(True)
            else:
                r = cross_results[cross_ptr]
                cross_ptr += 1
                sim = r.get("sim_to_own", float("nan"))
                keep_mask.append(not (sim != sim) and sim >= threshold)

        n_keep = sum(keep_mask)
        n_cross_kept  = sum(k for k, c in zip(keep_mask, cross_mask) if c)
        n_cross_total = sum(cross_mask)
        print(f"  Kept {n_keep}/{len(syn_ds)} rows total "
              f"({n_cross_kept}/{n_cross_total} cross-register, all {n_hard} hard_neg)")

        filtered_ds = syn_ds.filter(lambda _, i: keep_mask[i], with_indices=True)
        out_path = _PROJECT_ROOT / args.save_filtered
        out_path.parent.mkdir(parents=True, exist_ok=True)
        filtered_ds.save_to_disk(str(out_path))
        print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
