"""Stage B (v14 validation) — does a CONTENT-INDEPENDENT style embedding fix our
domain/length OOD failures vs LUAR, with no training?

Motivation (docs/research_synthesis_v14_strategy.md §1.4): the StyleDistance paper
(arXiv:2410.12757, hand-verified) shows LUAR "considers both style and content, hence
confounding" them. That is the suspected root cause of our PAN cross-topic collapse
(LUAR v12 AUC 0.779) and length/register fragility. This script runs an off-the-shelf
content-independent style embedding (default StyleDistance) over the EXACT SAME OOD pair
files v12 scored with LUAR, with the SAME per-slice metrics, so the numbers are directly
comparable to results/v12/ood_domains_v12.json and results/v12/ood_v12*_heldoutCG.json.

Pairwise verification: score(a,b) = (cos(emb_a, emb_b)+1)/2; label `same` (1=same author).
Higher = same author. Metrics per slice via the repo's compute_verification_metrics.

NOTE on fairness: StyleDistance is OFF-THE-SHELF (never finetuned on Enron), so on the
domain/cross-topic slices (also OOD for LUAR) this is a fair zero-shot-transfer contest.
On the gen:* held-out slices LUAR has a finetuning advantage, so read those as a floor.

Usage:
    python scripts/eval_style_embedding.py \
        --model StyleDistance/styledistance \
        --pairs results/v12/ood_v12heldout_pairs.jsonl \
        --pairs data/ood/pan20_xtopic_pairs.jsonl \
        --pairs data/ood/blog_pairs.jsonl \
        --out results/v14/styledistance_ood.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from email_fraud.scoring.metrics import compute_verification_metrics

logger = logging.getLogger(__name__)

# v12 LUAR reference numbers (AUC) for the same pair files — for side-by-side print.
_LUAR_V12_AUC = {
    "domain:pan20_xtopic": 0.779, "domain:blog": 0.839,
    "gen:openrouter:anthropic/claude-3.5-haiku": 0.829,
    "gen:openrouter:google/gemini-2.5-flash": 0.746,
    "len:short": 0.931, "len:medium": 0.971, "len:long": 0.95,
    "lenmix:short_long": 0.931, "register:cross": 0.955, "register:same": 0.962,
}


def _load_pairs(paths: list[str]) -> tuple[list[tuple[str, str]], np.ndarray, list[str]]:
    pairs, labels, slices = [], [], []
    for p in paths:
        fp = _PROJECT_ROOT / p if not Path(p).is_absolute() else Path(p)
        with fp.open() as fh:
            for line in fh:
                d = json.loads(line)
                a, b = d["pair"]
                pairs.append((a, b))
                labels.append(int(d.get("same", d.get("label"))))
                slices.append(d["slice"])
    return pairs, np.array(labels), slices


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="StyleDistance/styledistance")
    p.add_argument("--pairs", action="append", required=True, help="Pair JSONL (repeatable).")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-chars", type=int, default=2000)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/v14/styledistance_ood.json")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import torch
    from sentence_transformers import SentenceTransformer
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    pairs, labels, slices = _load_pairs(args.pairs)
    logger.info("Loaded %d pairs across %d slices", len(pairs), len(set(slices)))

    # Dedup unique texts → encode once.
    uniq: dict[str, int] = {}
    for a, b in pairs:
        for t in (a, b):
            if t not in uniq:
                uniq[t] = len(uniq)
    texts = [None] * len(uniq)
    for t, i in uniq.items():
        texts[i] = t[: args.max_chars]
    logger.info("Encoding %d unique texts with %s ...", len(texts), args.model)

    model = SentenceTransformer(args.model, device=device)
    emb = model.encode(texts, batch_size=args.batch_size, convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=False)

    scores = np.empty(len(pairs), dtype=np.float64)
    for i, (a, b) in enumerate(pairs):
        sim = float(np.dot(emb[uniq[a]], emb[uniq[b]]))
        scores[i] = (sim + 1.0) / 2.0

    by_slice: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(slices):
        by_slice[s].append(i)

    results = {"model": args.model, "per_slice": {}}
    for s, idx in sorted(by_slice.items()):
        idx = np.array(idx)
        y, sc = labels[idx], scores[idx]
        if len(np.unique(y)) < 2:
            continue
        m = compute_verification_metrics(y, sc)
        m["n"] = int(len(idx))
        results["per_slice"][s] = m

    out_path = _PROJECT_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("Saved → %s", out_path)

    print(f"\n=== {args.model} on OOD pairs (vs LUAR v12) ===")
    print(f"{'slice':42} {'StyleDist AUC':>13} {'LUAR v12':>9} {'Δ':>7} {'pAUC@5%':>8} {'TPR@1%':>7}")
    for s, m in results["per_slice"].items():
        luar = _LUAR_V12_AUC.get(s)
        delta = f"{m['AUC']-luar:+.3f}" if luar is not None else "   -"
        luar_s = f"{luar:.3f}" if luar is not None else "  -"
        print(f"{s:42} {m['AUC']:>13.3f} {luar_s:>9} {delta:>7} {m['pAUC@5%']:>8.3f} {m['TPR@FPR=1%']:>7.3f}")
    print("\nWIN for content-independent embedding: domain:pan20_xtopic AUC materially > 0.779,")
    print("and no collapse on len/register. (gen:* slices: LUAR has the finetuning edge — floor.)")


if __name__ == "__main__":
    main()
