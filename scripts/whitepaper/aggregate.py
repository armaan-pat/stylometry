#!/usr/bin/env python3
"""Aggregate all whitepaper experiment results into reproducible tables.

Reads the committed result JSONs and prints (and writes) whitepaper-ready tables:
  1. Multi-seed held-out Claude+Gemini eval, mean +/- std across probe seeds,
     for the deployment-relevant scorers (baseline_linear_z3 + mahalanobis).
  2. Novel-vendor spot-check (Qwen + DeepSeek), v12 vs v14b.
  3. Identity x synthetic 2x2 factorial.

Tolerant of partial results so it can be run while jobs are still in flight.
"""
from __future__ import annotations
import glob, json, math, os, statistics
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MS = os.path.join(ROOT, "results/whitepaper/multiseed")
NV = os.path.join(ROOT, "results/whitepaper/novelvendor")

SCORERS = ["baseline_linear_z3", "mahalanobis"]
METRICS = ["auc", "tpr5", "tpr1", "fpr_other_at_5", "auc_g_other"]


def row_for(path, scorer):
    try:
        d = json.load(open(path))
    except Exception:
        return None
    for r in d.get("rows", []):
        if r.get("scorer") == scorer:
            return r
    return None


def msd(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    if len(xs) == 1:
        return (xs[0], 0.0, 1)
    return (statistics.mean(xs), statistics.pstdev(xs), len(xs))


def fmt(m):
    if m is None:
        return "   -   "
    mean, sd, n = m
    return f"{mean:.3f}±{sd:.3f}"


def collect_multiseed(tags):
    """tags: dict label -> filename stem prefix (e.g. 'v14b' matches heldoutCG_v14b_s*.json)."""
    out = {}
    for label, stem in tags.items():
        per_scorer = {}
        for sc in SCORERS:
            vals = defaultdict(list)
            files = sorted(glob.glob(os.path.join(MS, f"heldoutCG_{stem}_s[0-9]*.json")))
            for f in files:
                r = row_for(f, sc)
                if not r:
                    continue
                for mk in METRICS:
                    vals[mk].append(r.get(mk))
            per_scorer[sc] = {mk: msd(vals[mk]) for mk in METRICS}, len(files)
        out[label] = per_scorer
    return out


def print_multiseed():
    tags = {
        "v11": "v11", "v12": "v12", "v13": "v13", "v14": "v14", "v14b": "v14b",
        "v14b_seed1": "v14b_seed1", "enron44_nosyn": "enron44_nosyn",
    }
    data = collect_multiseed(tags)
    lines = []
    for sc in SCORERS:
        lines.append(f"\n### Multi-seed held-out Claude+Gemini — scorer = {sc}")
        lines.append(f"{'version':16} {'n':>2}  {'AUC':>11} {'TPR@5%':>11} {'TPR@1%':>11} {'FPRoth@5':>11} {'AUCg/oth':>11}")
        for label in tags:
            ps = data.get(label, {})
            if sc not in ps:
                continue
            mets, n = ps[sc]
            if n == 0:
                continue
            lines.append(
                f"{label:16} {n:>2}  {fmt(mets['auc']):>11} {fmt(mets['tpr5']):>11} "
                f"{fmt(mets['tpr1']):>11} {fmt(mets['fpr_other_at_5']):>11} {fmt(mets['auc_g_other']):>11}"
            )
    return "\n".join(lines)


def print_novelvendor():
    lines = ["\n### Novel-vendor spot-check (Qwen-2.5-72B + DeepSeek-V3; never in train OR eval)"]
    lines.append("Pooled (mahalanobis deploy scorer), mean across probe seeds.")
    lines.append("NOTE: AUC + AUCg/oth are the generalization evidence; FPRoth@5 is a")
    lines.append("threshold-placement artifact when synthetics are near-separable (see writeup).")
    lines.append(f"{'version':10} {'n':>2}  {'AUC':>11} {'TPR@5%':>11} {'TPR@1%':>11} {'AUCg/oth':>11} {'FPRoth@5':>11}")
    for ver in ["v12", "v14b", "v14b_seed1"]:
        files = sorted(glob.glob(os.path.join(NV, f"novelvendor_{ver}_s[0-9]*.json")))
        vals = defaultdict(list)
        for f in files:
            r = row_for(f, "mahalanobis")
            if r:
                for mk in ["auc", "tpr5", "tpr1", "auc_g_other", "fpr_other_at_5"]:
                    vals[mk].append(r.get(mk))
        if not files:
            continue
        lines.append(
            f"{ver:10} {len(files):>2}  {fmt(msd(vals['auc'])):>11} {fmt(msd(vals['tpr5'])):>11} "
            f"{fmt(msd(vals['tpr1'])):>11} {fmt(msd(vals['auc_g_other'])):>11} {fmt(msd(vals['fpr_other_at_5'])):>11}"
        )
    return "\n".join(lines)


def print_2x2():
    """Identity x synthetic factorial, AUC (mahalanobis), seed-0 point + multiseed mean."""
    lines = ["\n### Identity x synthetic 2x2 (held-out Claude+Gemini, mahalanobis AUC, multiseed mean)"]
    grid = {
        ("44 authors", "no-syn"): "enron44_nosyn",
        ("44 authors", "+syn"):   "v12",
        ("844 authors", "no-syn"): "v14",
        ("844 authors", "+syn"):   "v14b",
    }
    for (authors, syn), stem in grid.items():
        files = sorted(glob.glob(os.path.join(MS, f"heldoutCG_{stem}_s[0-9]*.json")))
        vals = [row_for(f, "mahalanobis").get("auc") for f in files if row_for(f, "mahalanobis")]
        m = msd(vals)
        lines.append(f"  {authors:12} {syn:8} -> {fmt(m)}  (n={len(files)}, {stem})")
    return "\n".join(lines)


if __name__ == "__main__":
    parts = [
        "# Whitepaper results aggregate",
        print_multiseed(),
        print_2x2(),
        print_novelvendor(),
    ]
    text = "\n".join(parts)
    print(text)
    outp = os.path.join(ROOT, "results/whitepaper/AGGREGATE.md")
    with open(outp, "w") as f:
        f.write(text + "\n")
    print(f"\n[written] {outp}")
