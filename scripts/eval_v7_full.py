"""V7 — run the full scoring + K-sweep evaluation suite on any checkpoint.

Convenience wrapper that re-runs:
  - eval_v7_scoring.py    (V7.0 scoring sweep)
  - eval_v7_k_sweep.py    (V7.2 enrollment-K sweep)

on a checkpoint of your choice, and writes the results to
results/v7/<tag>_scoring_sweep.{json,csv} and <tag>_k_sweep.json.

Default behaviour: evaluate the *new* v7 checkpoint and the old v6
checkpoint and print a side-by-side delta table.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=_PROJECT_ROOT, env={
        **__import__("os").environ,
        "PYTHONPATH": f"{_PROJECT_ROOT}:{_PROJECT_ROOT}/src",
    })
    if res.returncode != 0:
        sys.exit(res.returncode)


def _eval_one(checkpoint: str, config: str, tag: str) -> None:
    _run([
        "python", "scripts/eval_v7_scoring.py",
        "--config", config,
        "--checkpoint", checkpoint,
        "--tag", f"{tag}_scoring_sweep",
    ])
    _run([
        "python", "scripts/eval_v7_k_sweep.py",
        "--config", config,
        "--checkpoint", checkpoint,
        "--tag", f"{tag}_k_sweep",
    ])


def _load_k_sweep(path: Path) -> dict[int, dict[str, dict]]:
    with path.open() as fh:
        raw = json.load(fh)
    out = {}
    for row in raw["rows"]:
        out[row["K"]] = row["scores"]
    return out


def _print_delta_table(v6_path: Path, v7_path: Path) -> None:
    a = _load_k_sweep(v6_path)
    b = _load_k_sweep(v7_path)
    print()
    print(f"{'K':>4s}  {'scorer':22s}  {'v6 AUC[g/syn]':>15s}  {'v7 AUC[g/syn]':>15s}  {'Δ AUC':>8s}  "
          f"{'v6 TPR@5%':>10s}  {'v7 TPR@5%':>10s}  {'Δ':>8s}")
    print("-" * 110)
    for K in sorted(set(a) & set(b)):
        for sc in a[K]:
            if sc not in b[K]:
                continue
            a_auc = a[K][sc].get("auc_g_syn", float("nan"))
            b_auc = b[K][sc].get("auc_g_syn", float("nan"))
            a_tpr = a[K][sc].get("tpr@5pct_syn", float("nan"))
            b_tpr = b[K][sc].get("tpr@5pct_syn", float("nan"))
            print(f"{K:>4d}  {sc:22s}  {a_auc:>15.4f}  {b_auc:>15.4f}  {b_auc - a_auc:>+8.4f}  "
                  f"{a_tpr:>10.4f}  {b_tpr:>10.4f}  {b_tpr - a_tpr:>+8.4f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--v6-checkpoint", default="runs/v6_luar_lora_syn/2026-05-26_19-09-22/checkpoint_best.pt")
    p.add_argument("--v6-config", default="configs/experiments/v6_luar_lora_syn.yaml")
    p.add_argument("--v7-checkpoint", required=True,
                   help="Path to the v7 checkpoint_best.pt to evaluate.")
    p.add_argument("--v7-config", default="configs/experiments/v7_luar_lora_syn_mahal.yaml")
    p.add_argument("--skip-v6", action="store_true",
                   help="Skip re-running v6 (use existing results files).")
    args = p.parse_args()

    if not args.skip_v6:
        _eval_one(args.v6_checkpoint, args.v6_config, tag="v6")
    _eval_one(args.v7_checkpoint, args.v7_config, tag="v7")

    v6_k = _PROJECT_ROOT / "results" / "v7" / "v6_k_sweep.json"
    v7_k = _PROJECT_ROOT / "results" / "v7" / "v7_k_sweep.json"
    if v6_k.exists() and v7_k.exists():
        _print_delta_table(v6_k, v7_k)


if __name__ == "__main__":
    main()
