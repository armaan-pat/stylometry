#!/usr/bin/env python3
"""Aggregate every result artifact under results/ into one digest.

This is a *read-only* reporting tool. It scans the JSON/CSV files produced by
the various eval scripts (scoring sweeps, K-sweeps, adaptive-scorer ablations,
authenticity probes) and prints:

  1. A per-experiment leaderboard (best scorer + headline metrics).
  2. The single best operating point found anywhere (by TPR@1% and TPR@5%).
  3. A side-by-side of the syn-v1 vs syn-v2 arms (the latest A/B).
  4. A "what was tested / what won / what didn't" matrix.

It understands four file shapes, dispatched by their keys:

  * scoring sweep  -> top-level {"rows": [{"score_fn", "auc_g_syn", ...}]}
  * K sweep        -> {"rows": [{"K", "scores": {scorer: {metrics}}}]}
  * scorer ablation-> {"recommendation", "rows": [{"scorer", "auc", "tpr1", ...}]}
  * authenticity probe -> flat {"probe_frozen/roc_auc": ..., ...}

Usage:
    python scripts/summarize_results.py                 # scan results/, print
    python scripts/summarize_results.py --results-dir results/v8
    python scripts/summarize_results.py --md results/SUMMARY.md   # also write markdown
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Normalised record: every parsed row collapses to this shape so the leaderboard
# logic does not need to know which file it came from.
# --------------------------------------------------------------------------- #
@dataclass
class ScoreRow:
    source: str          # file the row came from (relative path)
    experiment: str      # logical experiment id, e.g. "v7_3_scoring_sweep_ep150"
    scorer: str          # score function / scorer name
    k: int | None        # enrollment K if the file is K-aware, else None
    auc_syn: float | None    # AUROC genuine-vs-synthetic (the headline number)
    tpr5: float | None       # TPR @ 5% synthetic FPR
    tpr1: float | None       # TPR @ 1% synthetic FPR
    eer: float | None        # equal-error rate (lower better), stored as raw EER
    auc_oth: float | None    # AUROC genuine-vs-other (easy case)
    extra: dict[str, Any] = field(default_factory=dict)


def _f(d: dict, *keys: str) -> float | None:
    """First present key, coerced to float, else None."""
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return float(d[k])
            except (TypeError, ValueError):
                return None
    return None


# --------------------------------------------------------------------------- #
# Parsers — one per file shape. Each returns (rows, meta).
# --------------------------------------------------------------------------- #
def parse_scoring_sweep(stem: str, src: str, d: dict) -> tuple[list[ScoreRow], dict]:
    rows = []
    for r in d.get("rows", []):
        rows.append(ScoreRow(
            source=src, experiment=stem, scorer=r.get("score_fn", "?"), k=None,
            auc_syn=_f(r, "auc_g_syn"), tpr5=_f(r, "tpr@5pct_syn"),
            tpr1=_f(r, "tpr@1pct_syn"), eer=_f(r, "eer_syn"),
            auc_oth=_f(r, "auc_g_oth"),
        ))
    meta = {k: d[k] for k in ("checkpoint", "config", "per_sender_lw_shrinkage_mean") if k in d}
    return rows, meta


def parse_k_sweep(stem: str, src: str, d: dict) -> tuple[list[ScoreRow], dict]:
    rows = []
    for r in d.get("rows", []):
        K = r.get("K")
        for scorer, m in (r.get("scores") or {}).items():
            if not isinstance(m, dict):
                continue
            rows.append(ScoreRow(
                source=src, experiment=stem, scorer=scorer, k=K,
                auc_syn=_f(m, "auc_g_syn"), tpr5=_f(m, "tpr@5pct_syn"),
                tpr1=_f(m, "tpr@1pct_syn"), eer=_f(m, "eer_syn"),
                auc_oth=_f(m, "auc_g_oth"),
            ))
    return rows, {}


def parse_ablation(stem: str, src: str, d: dict) -> tuple[list[ScoreRow], dict]:
    rows = []
    K = (d.get("probe") or {}).get("n_enroll")
    for r in d.get("rows", []):
        # eer stored as 1-EER here; convert back so it is comparable
        one_minus_eer = _f(r, "one_minus_eer", "1-eer")
        eer = (1.0 - one_minus_eer) if one_minus_eer is not None else None
        rows.append(ScoreRow(
            source=src, experiment=stem, scorer=r.get("scorer", "?"), k=K,
            auc_syn=_f(r, "auc"), tpr5=_f(r, "tpr5"), tpr1=_f(r, "tpr1"),
            eer=eer, auc_oth=None,
            extra={"is_baseline": r.get("is_baseline"),
                   "tpr1_pwin": _f(r, "tpr1_pwin")},
        ))
    meta = {"recommendation": d.get("recommendation"),
            "mean_lw_shrinkage": d.get("mean_lw_shrinkage"),
            "checkpoint": d.get("checkpoint")}
    return rows, meta


def parse_probe(stem: str, src: str, d: dict) -> tuple[list[ScoreRow], dict]:
    # Probes are classification heads, not scorers; surface as a meta block only.
    meta = {}
    for mode in ("probe_frozen", "probe_finetune"):
        if f"{mode}/roc_auc" in d:
            meta[mode] = {
                "roc_auc": d.get(f"{mode}/roc_auc"),
                "accuracy": d.get(f"{mode}/accuracy"),
                "precision": d.get(f"{mode}/precision"),
                "recall": d.get(f"{mode}/recall"),
                "fp": d.get(f"{mode}/fp"), "fn": d.get(f"{mode}/fn"),
                "n_test": d.get(f"{mode}/n_test"),
            }
    return [], meta


def classify_and_parse(path: Path, src: str) -> tuple[str, list[ScoreRow], dict]:
    """Return (kind, rows, meta) for a results JSON file."""
    try:
        d = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return "unreadable", [], {}
    stem = path.stem
    if any(k.startswith("probe_frozen/") or k.startswith("probe_finetune/") for k in d):
        return "probe", *parse_probe(stem, src, d)
    if "recommendation" in d or (d.get("rows") and "scorer" in d["rows"][0]):
        return "ablation", *parse_ablation(stem, src, d)
    if d.get("rows") and "K" in d["rows"][0] and "scores" in d["rows"][0]:
        return "k_sweep", *parse_k_sweep(stem, src, d)
    if d.get("rows") and "score_fn" in d["rows"][0]:
        return "scoring_sweep", *parse_scoring_sweep(stem, src, d)
    return "unknown", [], {}


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _fmt(x: float | None, p: int = 3) -> str:
    return f"{x:.{p}f}" if isinstance(x, (int, float)) else "  -  "


def build_report(results_dir: Path) -> str:
    files = sorted(results_dir.rglob("*.json"))
    all_rows: list[ScoreRow] = []
    probes: dict[str, dict] = {}
    ablation_meta: dict[str, dict] = {}
    sweep_meta: dict[str, dict] = {}
    skipped: list[str] = []

    for f in files:
        src = str(f.relative_to(_PROJECT_ROOT)) if f.is_relative_to(_PROJECT_ROOT) else str(f)
        kind, rows, meta = classify_and_parse(f, src)
        if kind in ("unknown", "unreadable"):
            skipped.append(f"{src} [{kind}]")
            continue
        all_rows.extend(rows)
        if kind == "probe":
            probes[src] = meta
        elif kind == "ablation":
            ablation_meta[src] = meta
        elif kind in ("scoring_sweep",):
            sweep_meta[src] = meta

    out: list[str] = []
    w = out.append
    w("# Results digest\n")
    w(f"Scanned `{results_dir.relative_to(_PROJECT_ROOT) if results_dir.is_relative_to(_PROJECT_ROOT) else results_dir}` "
      f"— {len(files)} JSON files, {len(all_rows)} scorer rows parsed.\n")

    # 1. Global best operating points (require a synthetic AUC to be meaningful)
    scored = [r for r in all_rows if r.auc_syn is not None]
    w("## 1. Best operating points found (genuine-vs-synthetic)\n")
    if scored:
        for label, key in (("AUC[g/syn]", "auc_syn"), ("TPR@5%FPR_syn", "tpr5"),
                           ("TPR@1%FPR_syn", "tpr1")):
            cand = [r for r in scored if getattr(r, key) is not None]
            best = max(cand, key=lambda r: getattr(r, key)) if cand else None
            if best:
                w(f"- **Best {label}: {_fmt(getattr(best, key))}** "
                  f"— `{best.scorer}`"
                  f"{f' @ K={best.k}' if best.k is not None else ''} "
                  f"(AUC[g/syn]={_fmt(best.auc_syn)}, TPR@5%={_fmt(best.tpr5)}, "
                  f"TPR@1%={_fmt(best.tpr1)}) — _{best.experiment}_")
        w("")

    # 2. Per-experiment leaderboard (best row per experiment by tpr1 then auc)
    w("## 2. Per-experiment leaderboard (best scorer in each file)\n")
    w("| experiment | K | best scorer | AUC[g/syn] | TPR@5% | TPR@1% | EER | source |")
    w("|---|---|---|---|---|---|---|---|")
    by_exp: dict[str, list[ScoreRow]] = {}
    for r in scored:
        by_exp.setdefault(r.experiment, []).append(r)
    for exp in sorted(by_exp):
        rows = by_exp[exp]
        # rank by tpr1, fall back to auc
        best = max(rows, key=lambda r: (r.tpr1 or -1, r.auc_syn or -1))
        w(f"| {exp} | {best.k if best.k is not None else '-'} | `{best.scorer}` | "
          f"{_fmt(best.auc_syn)} | {_fmt(best.tpr5)} | {_fmt(best.tpr1)} | "
          f"{_fmt(best.eer)} | {best.source} |")
    w("")

    # 3. K-sweep view (how the best scorer scales with enrollment)
    ksweep_rows = [r for r in scored if r.k is not None]
    if ksweep_rows:
        w("## 3. Enrollment-K scaling (best scorer per K)\n")
        w("| experiment | K | best scorer | AUC[g/syn] | TPR@5% | TPR@1% |")
        w("|---|---|---|---|---|---|")
        seen = {}
        for r in ksweep_rows:
            seen.setdefault((r.experiment, r.k), []).append(r)
        for (exp, k) in sorted(seen, key=lambda t: (t[0], t[1] or 0)):
            rows = seen[(exp, k)]
            best = max(rows, key=lambda r: (r.tpr1 or -1, r.auc_syn or -1))
            w(f"| {exp} | {k} | `{best.scorer}` | {_fmt(best.auc_syn)} | "
              f"{_fmt(best.tpr5)} | {_fmt(best.tpr1)} |")
        w("")

    # 4. Ablation recommendations (the statistically-vetted verdicts)
    if ablation_meta:
        w("## 4. Adaptive-scorer ablation verdicts (bootstrapped)\n")
        for src, meta in sorted(ablation_meta.items()):
            rec = meta.get("recommendation") or {}
            w(f"- **{src}**")
            if meta.get("mean_lw_shrinkage") is not None:
                w(f"  - mean Ledoit-Wolf shrinkage α = {_fmt(meta['mean_lw_shrinkage'])}")
            if rec:
                w(f"  - winner: **{rec.get('winner')}** (rank by {rec.get('rank_metric')})")
                if rec.get("text"):
                    w(f"  - _{rec['text']}_")
        w("")

    # 5. Authenticity probes
    if probes:
        w("## 5. Authenticity probes (genuine-vs-synthetic classifier; +=synthetic)\n")
        w("| source | mode | ROC-AUC | acc | precision | recall | FP | FN | n |")
        w("|---|---|---|---|---|---|---|---|---|")
        for src, meta in sorted(probes.items()):
            for mode, m in meta.items():
                w(f"| {src} | {mode.replace('probe_','')} | {_fmt(m['roc_auc'])} | "
                  f"{_fmt(m['accuracy'])} | {_fmt(m['precision'])} | {_fmt(m['recall'])} | "
                  f"{int(m['fp']) if m['fp'] is not None else '-'} | "
                  f"{int(m['fn']) if m['fn'] is not None else '-'} | "
                  f"{int(m['n_test']) if m['n_test'] is not None else '-'} |")
        w("")

    if skipped:
        w("## Skipped files\n")
        for s in skipped:
            w(f"- {s}")
        w("")

    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=str(_PROJECT_ROOT / "results"),
                    help="Directory to scan recursively for result JSONs.")
    ap.add_argument("--md", default=None,
                    help="Optional path to also write the report as markdown.")
    args = ap.parse_args()

    results_dir = Path(args.results_dir).resolve()
    if not results_dir.exists():
        raise SystemExit(f"results dir not found: {results_dir}")

    report = build_report(results_dir)
    print(report)
    if args.md:
        out = Path(args.md).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report)
        print(f"\n[written] {out}")


if __name__ == "__main__":
    main()
