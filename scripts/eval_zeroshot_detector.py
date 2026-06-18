"""Stage A (v14 validation) — does a FROZEN zero-shot MGT detector beat our
trained generator-classifier head on HELD-OUT generators, for free?

Motivation (docs/research_synthesis_v14_strategy.md): v13 showed a *trained* LLM-text
head plateaus cross-generator (Gemini held-out AUC ~0.75). Zero-shot detectors
(Fast-DetectGPT, Binoculars) never train on generators, so they have no moving-target
failure mode. This script measures Fast-DetectGPT's genuine-vs-synthetic separation on
the SAME held-out Claude+Gemini set v12/v13 were scored on, with the SAME metrics, so
the numbers are directly comparable to results/v13/heldoutCG_v13lora.json.

Method: Fast-DetectGPT analytic "conditional probability curvature" (Bao et al. 2024,
arXiv:2310.05130), single-model variant. For passage x under reference LM p:
  d(x) = (logp_obs - mu) / sqrt(sigma2)
where per token i over the full vocab,
  mu_i     = sum_v softmax(z_i)_v * logsoftmax(z_i)_v        (expected sampled log-prob)
  sigma2_i = sum_v softmax(z_i)_v * logsoftmax(z_i)_v^2 - mu_i^2
  logp_obs = logsoftmax(z_i)[x_{i+1}]
Higher d => more machine-like. We report authenticity = -d (higher = genuine/human) so
labels follow the repo convention (1 = genuine) and metrics match the ablation harness.

Eval-only; no training; frozen reference model (default EleutherAI/gpt-neo-1.3B).

Usage:
    python scripts/eval_zeroshot_detector.py \
        --config configs/experiments/_v12_heldout_eval.yaml \
        --synthetic-path data/synthetic/enron_synthetic_v12_heldout \
        --ref-model EleutherAI/gpt-neo-1.3B \
        --out results/v14/zeroshot_fastdetectgpt_heldoutCG.json
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

from email_fraud.config import load_config
from email_fraud.scoring.metrics import compute_verification_metrics

logger = logging.getLogger(__name__)


@torch.no_grad()
def binoculars_scores(
    texts: list[str], m_obs, m_perf, tokenizer, device: str, max_length: int = 512,
) -> np.ndarray:
    """Binoculars score B(x)=log_ppl_obs(x)/xppl(obs,perf) (Hans et al. 2024,
    arXiv:2401.12070). LOWER B = more machine; we return B so higher = more human
    (matches authenticity convention). Observer+performer share a tokenizer."""
    out = np.empty(len(texts), dtype=np.float64)
    for i, text in enumerate(texts):
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        ids = enc["input_ids"]
        if ids.shape[1] < 3:
            out[i] = 1.0
            continue
        zo = m_obs(**enc).logits[0, :-1].float()         # (T-1, V) observer
        zp = m_perf(**enc).logits[0, :-1].float()        # (T-1, V) performer
        tgt = ids[0, 1:]
        logp_o = torch.log_softmax(zo, dim=-1)
        # observer log-perplexity = mean token NLL under observer
        nll = -logp_o.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean()
        # cross-perplexity = mean over tokens of CE(observer_dist, performer_dist)
        p_o = logp_o.exp()
        logp_p = torch.log_softmax(zp, dim=-1)
        xce = -(p_o * logp_p).sum(-1).mean()
        out[i] = float(nll / xce.clamp_min(1e-12))
    return out


@torch.no_grad()
def fastdetect_scores(
    texts: list[str], model, tokenizer, device: str, max_length: int = 512,
) -> np.ndarray:
    """Analytic Fast-DetectGPT discrepancy d(x) per text (higher = more machine)."""
    out = np.empty(len(texts), dtype=np.float64)
    for i, text in enumerate(texts):
        enc = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length,
        ).to(device)
        ids = enc["input_ids"]
        if ids.shape[1] < 3:  # too short to score; neutral
            out[i] = 0.0
            continue
        logits = model(**enc).logits[0].float()          # (T, V)
        # Align: predict token t+1 from position t.
        z = logits[:-1]                                  # (T-1, V)
        tgt = ids[0, 1:]                                 # (T-1,)
        logp = torch.log_softmax(z, dim=-1)              # (T-1, V)
        p = logp.exp()
        mu = (p * logp).sum(-1)                           # (T-1,) expected sampled logprob
        e_lp2 = (p * logp * logp).sum(-1)
        sigma2 = (e_lp2 - mu * mu).clamp_min(1e-12)       # (T-1,)
        lp_obs = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # (T-1,)
        num = (lp_obs - mu).sum()
        den = sigma2.sum().sqrt()
        out[i] = float(num / den)
    return out


def _word_lens(texts: list[str]) -> np.ndarray:
    return np.array([len(t.split()) for t in texts], dtype=float)


def _len_matched_idx(lens_a: np.ndarray, lens_b: np.ndarray, seed: int = 0):
    """Subsample the larger-distribution pool so both pools have similar length
    histograms (20-word bins) — controls the 'LLM emails are just longer' confound."""
    rng = np.random.default_rng(seed)
    bins = np.arange(0, 600, 20)
    ba, bb = np.digitize(lens_a, bins), np.digitize(lens_b, bins)
    keep_a, keep_b = [], []
    for b in np.unique(np.concatenate([ba, bb])):
        ia = np.where(ba == b)[0]
        ib = np.where(bb == b)[0]
        n = min(len(ia), len(ib))
        if n == 0:
            continue
        keep_a.extend(rng.choice(ia, n, replace=False).tolist())
        keep_b.extend(rng.choice(ib, n, replace=False).tolist())
    return np.array(sorted(keep_a)), np.array(sorted(keep_b))


def _metrics_block(gen_auth: np.ndarray, syn_auth: np.ndarray) -> dict:
    labels = np.concatenate([np.ones_like(gen_auth), np.zeros_like(syn_auth)])
    scores = np.concatenate([gen_auth, syn_auth])
    return compute_verification_metrics(labels, scores)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/experiments/_v12_heldout_eval.yaml")
    p.add_argument("--synthetic-path", default="data/synthetic/enron_synthetic_v12_heldout")
    p.add_argument("--detector", default="fastdetect", choices=["fastdetect", "binoculars"])
    p.add_argument("--ref-model", default="EleutherAI/gpt-neo-1.3B",
                   help="fastdetect reference LM.")
    p.add_argument("--obs-model", default="Qwen/Qwen2.5-1.5B", help="binoculars observer.")
    p.add_argument("--perf-model", default="Qwen/Qwen2.5-1.5B-Instruct", help="binoculars performer.")
    p.add_argument("--genuine-split", default="train",
                   help="Synthetics imitate --from-split train senders; match genuine pool to them.")
    p.add_argument("--max-genuine", type=int, default=1000)
    p.add_argument("--max-per-generator", type=int, default=400)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/v14/zeroshot_fastdetectgpt_heldoutCG.json")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(str(_PROJECT_ROOT / args.config) if not Path(args.config).is_absolute() else args.config)

    from datasets import load_from_disk
    # --- Synthetic (held-out generators), grouped by generator ---
    syn = load_from_disk(str(_PROJECT_ROOT / args.synthetic_path)
                         if not Path(args.synthetic_path).is_absolute() else args.synthetic_path)
    syn_texts = syn["text"]
    syn_gen = syn["generator"]
    syn_src = syn["source_sender_id"] if "source_sender_id" in syn.column_names else [None] * len(syn_texts)
    by_gen: dict[str, list[str]] = defaultdict(list)
    src_senders = set()
    for t, g, s in zip(syn_texts, syn_gen, syn_src):
        by_gen[g].append(t)
        if s is not None:
            src_senders.add(s)

    # --- Genuine pool: real emails from the SAME source senders (topic-matched) ---
    proc_dir = cfg.data.processed_dir
    dd = load_from_disk(str(_PROJECT_ROOT / proc_dir) if not Path(proc_dir).is_absolute() else proc_dir)
    split = args.genuine_split if args.genuine_split in dd else list(dd.keys())[0]
    g_texts_all = dd[split]["text"]
    g_sids_all = dd[split]["sender_id"]
    rng = np.random.default_rng(args.seed)
    matched = [t for t, s in zip(g_texts_all, g_sids_all) if (not src_senders) or s in src_senders]
    if len(matched) < 50:  # fallback: any genuine from the split
        logger.warning("Only %d sender-matched genuine emails; using all genuine in split.", len(matched))
        matched = list(g_texts_all)
    if len(matched) > args.max_genuine:
        matched = [matched[i] for i in rng.choice(len(matched), args.max_genuine, replace=False)]
    genuine_texts = matched
    logger.info("Genuine pool: %d emails (split=%s, sender-matched=%s)", len(genuine_texts), split, bool(src_senders))
    for g in by_gen:
        if len(by_gen[g]) > args.max_per_generator:
            idx = rng.choice(len(by_gen[g]), args.max_per_generator, replace=False)
            by_gen[g] = [by_gen[g][i] for i in idx]
        logger.info("Generator %s: %d synthetics", g, len(by_gen[g]))

    # --- Load frozen detector model(s) + define scoring fn (higher = more human) ---
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = torch.float16 if device == "cuda" else torch.float32
    if args.detector == "binoculars":
        logger.info("Loading Binoculars pair obs=%s perf=%s on %s ...",
                    args.obs_model, args.perf_model, device)
        tok = AutoTokenizer.from_pretrained(args.obs_model)
        m_obs = AutoModelForCausalLM.from_pretrained(args.obs_model, torch_dtype=dtype).to(device).eval()
        m_perf = AutoModelForCausalLM.from_pretrained(args.perf_model, torch_dtype=dtype).to(device).eval()
        detector_name = f"binoculars[{args.obs_model.split('/')[-1]}/{args.perf_model.split('/')[-1]}]"
        score_fn = lambda ts: binoculars_scores(ts, m_obs, m_perf, tok, device, args.max_length)
        sign = 1.0   # Binoculars already returns higher = more human
    else:
        logger.info("Loading reference LM %s on %s ...", args.ref_model, device)
        tok = AutoTokenizer.from_pretrained(args.ref_model)
        model = AutoModelForCausalLM.from_pretrained(args.ref_model, torch_dtype=dtype).to(device).eval()
        detector_name = f"fast-detectgpt-analytic[{args.ref_model.split('/')[-1]}]"
        score_fn = lambda ts: fastdetect_scores(ts, model, tok, device, args.max_length)
        sign = -1.0  # discrepancy: higher = more machine -> negate for authenticity

    # --- Score everything (authenticity: higher = more human/genuine) ---
    logger.info("Scoring genuine ...")
    gen_auth = sign * score_fn(genuine_texts)
    syn_auth_by_gen = {}
    for g, ts in by_gen.items():
        logger.info("Scoring %s ...", g)
        syn_auth_by_gen[g] = sign * score_fn(ts)

    g_len = _word_lens(genuine_texts)
    results = {
        "detector": detector_name,
        "ref_model": args.ref_model if args.detector == "fastdetect" else f"{args.obs_model}|{args.perf_model}",
        "genuine_n": len(genuine_texts),
        "genuine_word_len": {"mean": float(g_len.mean()), "p50": float(np.median(g_len))},
        "per_generator": {},
        "pooled": {},
        "length_matched": {},
    }

    pooled_syn = []
    for g, sa in syn_auth_by_gen.items():
        results["per_generator"][g] = _metrics_block(gen_auth, sa)
        s_len = _word_lens(by_gen[g])
        results["per_generator"][g]["syn_word_len_mean"] = float(s_len.mean())
        results["per_generator"][g]["n_syn"] = int(len(sa))
        pooled_syn.append(sa)
    pooled_syn = np.concatenate(pooled_syn)
    results["pooled"] = _metrics_block(gen_auth, pooled_syn)
    results["pooled"]["n_syn"] = int(len(pooled_syn))

    # --- Length-matched pooled (confound control) ---
    pooled_syn_texts = [t for g in by_gen for t in by_gen[g]]
    s_len_all = _word_lens(pooled_syn_texts)
    gi, si = _len_matched_idx(g_len, s_len_all, args.seed)
    if len(gi) > 20 and len(si) > 20:
        lm = _metrics_block(gen_auth[gi], pooled_syn[si])
        lm["n_genuine"], lm["n_syn"] = int(len(gi)), int(len(si))
        results["length_matched"]["pooled"] = lm
        logger.info("Length-matched pooled: AUC=%.3f TPR@1%%=%.3f (n=%d/%d)",
                    lm["AUC"], lm["TPR@FPR=1%"], len(gi), len(si))

    out_path = _PROJECT_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("Saved → %s", out_path)

    print("\n=== Zero-shot detector %s on held-out generators ===" % detector_name)
    print(f"{'pool':28} {'AUC':>7} {'pAUC@5%':>8} {'TPR@1%':>7} {'TPR@5%':>7}")
    for g, m in results["per_generator"].items():
        print(f"{g.split('/')[-1]:28} {m['AUC']:>7.3f} {m['pAUC@5%']:>8.3f} {m['TPR@FPR=1%']:>7.3f} {m['TPR@FPR=5%']:>7.3f}")
    m = results["pooled"]
    print(f"{'POOLED':28} {m['AUC']:>7.3f} {m['pAUC@5%']:>8.3f} {m['TPR@FPR=1%']:>7.3f} {m['TPR@FPR=5%']:>7.3f}")
    if "pooled" in results["length_matched"]:
        m = results["length_matched"]["pooled"]
        print(f"{'POOLED (len-matched)':28} {m['AUC']:>7.3f} {m['pAUC@5%']:>8.3f} {m['TPR@FPR=1%']:>7.3f} {m['TPR@FPR=5%']:>7.3f}")
    print("\nCompare to v13 trained head (results/v13): held-out pool AUC 0.859, TPR@1% 0.155;")
    print("per-gen AUC Claude 0.830 / Gemini 0.755. Zero-shot WINS if it materially exceeds these.")


if __name__ == "__main__":
    main()
