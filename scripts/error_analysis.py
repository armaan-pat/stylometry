"""High-confidence mistake analysis.

Loads a checkpoint, scores all test pairs, and surfaces the most confident
wrong predictions in both directions:
  - High-confidence false positives: model says "same author" but they're different
  - High-confidence false negatives: model says "different author" but they're the same

Usage:
    python scripts/error_analysis.py \
        --config configs/experiments/v6_luar_lora_syn.yaml \
        --checkpoint runs/v6_luar_lora_syn/<timestamp>/checkpoint_best.pt \
        --data-dir data/processed \
        --top-k 10
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from dotenv import load_dotenv
load_dotenv()

import email_fraud.encoders  # noqa: F401
import email_fraud.heads     # noqa: F401
import email_fraud.losses    # noqa: F401
from email_fraud.config import load_config
from email_fraud.registry import resolve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir",   default="data/processed")
    parser.add_argument("--top-k",      type=int, default=10)
    parser.add_argument("--threshold",  type=float, default=0.5)
    parser.add_argument("--out",        default=None,
                        help="Optional path to save report as .txt")
    return parser.parse_args()


def load_pairs(data_dir: str) -> list[dict]:
    root = Path(data_dir)
    pairs = []
    for path in sorted(root.rglob("test_pairs.jsonl")):
        with path.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if "pair" in rec:
                    text1, text2 = rec["pair"]
                else:
                    text1 = rec.get("text1") or rec.get("text_a", "")
                    text2 = rec.get("text2") or rec.get("text_b", "")
                label = int(bool(rec.get("same", rec.get("label", 0))))
                pairs.append({"text1": text1, "text2": text2, "label": label})
    return pairs


def encode_all(encoder, texts: list[str], device: str, batch_size: int = 64) -> torch.Tensor:
    # Force episode_k=1 for LUAR so each text gets its own embedding.
    saved_k = None
    if hasattr(encoder, "config") and hasattr(encoder.config, "episode_k"):
        saved_k = encoder.config.episode_k
        encoder.config.episode_k = 1

    all_embs = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            tok = encoder.tokenize(batch)
            tok = {k: v.to(device) for k, v in tok.items()}
            all_embs.append(encoder.encode(**tok).cpu())

    if saved_k is not None:
        encoder.config.episode_k = saved_k

    return torch.cat(all_embs, dim=0)


def _wrap(text: str, width: int = 100, max_lines: int = 6) -> str:
    lines = textwrap.wrap(text[:1000], width=width)
    truncated = lines[:max_lines]
    suffix = "  [...]" if len(lines) > max_lines else ""
    return "\n    ".join(truncated) + suffix


def format_mistake(rank: int, score: float, label: int, pair: dict, threshold: float) -> str:
    pred = "SAME" if score > threshold else "DIFF"
    actual = "SAME" if label == 1 else "DIFF"
    confidence = abs(score - 0.5) * 2  # rescale to [0, 1]
    lines = [
        f"#{rank}  score={score:.4f}  pred={pred}  actual={actual}  confidence={confidence:.2f}",
        f"  TEXT 1:",
        f"    {_wrap(pair['text1'])}",
        f"  TEXT 2:",
        f"    {_wrap(pair['text2'])}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    cfg    = load_config(str(_PROJECT_ROOT / args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    EncoderClass = resolve("encoder", cfg.encoder.name)
    encoder = EncoderClass(cfg.encoder)
    ckpt = torch.load(str(_PROJECT_ROOT / args.checkpoint), map_location=device)
    encoder.load_state_dict(ckpt["model_state_dict"])
    encoder.eval().to(device)
    print(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')})")

    pairs = load_pairs(str(_PROJECT_ROOT / args.data_dir))
    print(f"Loaded {len(pairs)} test pairs")

    from email_fraud.data.preprocessing import preprocess
    pp = cfg.data.preprocessing
    flat_texts = [
        preprocess(t, pp) or t
        for p in pairs
        for t in [p["text1"], p["text2"]]
    ]
    embs = encode_all(encoder, flat_texts, device)

    scores = []
    for i in range(0, len(embs), 2):
        sim = F.cosine_similarity(embs[i].unsqueeze(0), embs[i + 1].unsqueeze(0)).item()
        scores.append((sim + 1.0) / 2.0)

    scores  = np.array(scores)
    labels  = np.array([p["label"] for p in pairs])
    correct = ((scores > args.threshold) == labels.astype(bool))

    # High-confidence false positives: score high, label=0
    fp_mask = (~correct) & (labels == 0)
    fp_idx  = np.where(fp_mask)[0]
    fp_idx  = fp_idx[np.argsort(-scores[fp_idx])][:args.top_k]

    # High-confidence false negatives: score low, label=1
    fn_mask = (~correct) & (labels == 1)
    fn_idx  = np.where(fn_mask)[0]
    fn_idx  = fn_idx[np.argsort(scores[fn_idx])][:args.top_k]

    lines = []

    lines.append("=" * 80)
    lines.append(f"HIGH-CONFIDENCE FALSE POSITIVES (top {len(fp_idx)})")
    lines.append("Model said SAME AUTHOR — actually DIFFERENT")
    lines.append("=" * 80)
    for rank, idx in enumerate(fp_idx, 1):
        lines.append(format_mistake(rank, scores[idx], labels[idx], pairs[idx], args.threshold))

    lines.append("=" * 80)
    lines.append(f"HIGH-CONFIDENCE FALSE NEGATIVES (top {len(fn_idx)})")
    lines.append("Model said DIFFERENT AUTHOR — actually SAME")
    lines.append("=" * 80)
    for rank, idx in enumerate(fn_idx, 1):
        lines.append(format_mistake(rank, scores[idx], labels[idx], pairs[idx], args.threshold))

    # Summary stats
    lines.append("=" * 80)
    lines.append("SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Total pairs:           {len(pairs)}")
    lines.append(f"Threshold:             {args.threshold}")
    lines.append(f"Overall accuracy:      {correct.mean():.4f}")
    lines.append(f"False positives:       {fp_mask.sum()} ({fp_mask.mean()*100:.1f}%)")
    lines.append(f"False negatives:       {fn_mask.sum()} ({fn_mask.mean()*100:.1f}%)")
    lines.append(f"Mean score (same):     {scores[labels==1].mean():.4f}")
    lines.append(f"Mean score (diff):     {scores[labels==0].mean():.4f}")
    lines.append(f"Score std (same):      {scores[labels==1].std():.4f}")
    lines.append(f"Score std (diff):      {scores[labels==0].std():.4f}")

    report = "\n".join(lines)
    print(report)

    out = Path(args.out) if args.out else _PROJECT_ROOT / "results" / "error_analysis.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
