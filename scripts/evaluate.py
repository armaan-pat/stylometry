"""PAN-style evaluation script.

Computes AUC, c@1, F0.5u, and EER for a trained model on a held-out
verification dataset and can optionally report the results to W&B.

Usage::

    python scripts/evaluate.py \
        --config configs/experiments/example_luar.yaml \
        --checkpoint checkpoints/<run_id>/epoch_010.pt \
        --data-dir data/processed/pan

Metrics follow the PAN authorship verification shared task definitions:
  - AUC     : area under the ROC curve.
  - c@1     : PAN's accuracy-with-abstention metric (Peñas & Rodrigo, 2011).
  - F0.5u   : F-score weighted toward precision (PAN 2020 primary metric).
  - EER     : equal error rate (threshold-free).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()

import email_fraud.encoders  # noqa: F401  — trigger @register
import email_fraud.heads     # noqa: F401
import email_fraud.losses    # noqa: F401
from email_fraud.config import load_config
from email_fraud.scoring.metrics import compute_verification_metrics
from email_fraud.utils.logging import setup_logging

logger = logging.getLogger(__name__)


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve(p: str) -> Path:
    """Resolve a path: absolute if given, else relative to CWD, else relative to project root."""
    candidate = Path(p)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    # Try relative to project root (scripts run from any directory)
    from_root = _PROJECT_ROOT / candidate
    if from_root.exists():
        return from_root
    # Return CWD-relative so the downstream error message is meaningful
    return candidate.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained model (PAN metrics).")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file.")
    parser.add_argument("--data-dir", required=True, help="Directory with eval pairs (.jsonl).")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Texts per encoding batch (default: 64).")
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log the computed metrics to the W&B project defined in the config.",
    )
    return parser.parse_args()


def load_eval_pairs(data_dir: str) -> list[tuple[str, str, int]]:
    """Load verification pairs from JSONL files under *data_dir*."""
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Evaluation directory not found: {root}")

    jsonl_files = sorted(root.rglob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files found under {root}")

    pairs: list[tuple[str, str, int]] = []
    for path in tqdm(jsonl_files, desc="Loading pairs", unit="file"):
        with path.open() as fh:
            for line_no, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)

                if "pair" in record:
                    text1, text2 = record["pair"]
                else:
                    text1 = record.get("text1") or record.get("text_a")
                    text2 = record.get("text2") or record.get("text_b")
                    if text1 is None or text2 is None:
                        raise ValueError(
                            f"Could not find a pair in {path}:{line_no}. "
                            "Expected 'pair' or ('text1', 'text2')."
                        )

                if "same" in record:
                    label = int(bool(record["same"]))
                elif "label" in record:
                    label = int(record["label"])
                else:
                    raise ValueError(
                        f"Could not find a label in {path}:{line_no}. "
                        "Expected 'same' or 'label'."
                    )

                pairs.append((str(text1), str(text2), label))

    if not pairs:
        raise ValueError(f"No evaluation pairs found under {root}")

    logger.info("Loaded %d pairs from %d file(s)", len(pairs), len(jsonl_files))
    return pairs


def score_pairs(
    encoder,
    pairs: list[tuple[str, str, int]],
    device: str,
    batch_size: int = 64,
) -> np.ndarray:
    """Return normalized cosine similarity scores in [0, 1] for each pair."""
    import torch

    flat_texts = [text for pair in pairs for text in pair[:2]]
    all_embeddings: list = []

    with torch.no_grad():
        for start in tqdm(
            range(0, len(flat_texts), batch_size),
            desc="Encoding",
            unit="batch",
            total=(len(flat_texts) + batch_size - 1) // batch_size,
        ):
            batch = flat_texts[start : start + batch_size]
            token_dict = encoder.tokenize(batch)
            token_dict = {k: v.to(device) for k, v in token_dict.items()}
            embeddings = encoder.encode(**token_dict)
            all_embeddings.append(embeddings.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0)

    scores: list[float] = []
    for idx in range(0, all_embeddings.size(0), 2):
        pair_score = F.cosine_similarity(
            all_embeddings[idx].unsqueeze(0),
            all_embeddings[idx + 1].unsqueeze(0),
        ).item()
        scores.append((pair_score + 1.0) / 2.0)

    return np.array(scores, dtype=np.float64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    setup_logging()

    config_path     = _resolve(args.config)
    checkpoint_path = _resolve(args.checkpoint)
    data_dir        = _resolve(args.data_dir)

    cfg = load_config(str(config_path))

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    from email_fraud.registry import resolve

    EncoderClass = resolve("encoder", cfg.encoder.name)

    encoder = EncoderClass(cfg.encoder)
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    encoder.load_state_dict(checkpoint["model_state_dict"])
    encoder.eval()
    logger.info("Loaded checkpoint from epoch %d", checkpoint.get("epoch", "?"))

    eval_pairs = load_eval_pairs(str(data_dir))
    labels = np.array([label for _, _, label in eval_pairs], dtype=np.int64)
    scores = score_pairs(encoder, eval_pairs, device, batch_size=args.batch_size)
    metrics = compute_verification_metrics(labels, scores)

    logger.info("Evaluation metrics:")
    for key, value in metrics.items():
        logger.info("  %s = %.4f", key, value)

    print("\n".join(f"{key:8s} {value:.4f}" for key, value in metrics.items()))

    if args.wandb:
        import wandb

        run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            tags=cfg.wandb.tags,
            notes=cfg.wandb.notes,
            dir=str(checkpoint_path.parent),
            config={
                "config": str(config_path),
                "checkpoint": str(checkpoint_path),
                "data_dir": str(data_dir),
                "device": device,
            },
            resume="allow",
        )
        wandb.log({f"test/{key}": value for key, value in metrics.items()})
        wandb.summary.update({f"test/{key}": value for key, value in metrics.items()})
        run.finish()


if __name__ == "__main__":
    main()
