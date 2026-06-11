"""Tests for scripts/prepare_external_pairs.py (domain-OOD pair conversion).

Covers the PAN pairs+truth join and the generic {text, author} balanced-pair
builder, plus the slice-tagging convention eval_ood relies on.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

import prepare_external_pairs as P


def _args(**kw):
    """Build an argparse-like namespace with the converter's defaults."""
    import argparse

    base = dict(
        format="pan", output="out.jsonl", slice="domain:test", config="configs/base.yaml",
        preprocess=False, max_chars=4000, seed=123,
        pairs_file=None, truth_file=None, max_pairs=None,
        input=None, text_field="text", author_field="author",
        min_words=5, n_per_class=10,
    )
    base.update(kw)
    return argparse.Namespace(**base)


def test_pan_join_on_id(tmp_path):
    pairs = tmp_path / "pairs.jsonl"
    truth = tmp_path / "truth.jsonl"
    pairs.write_text("\n".join(json.dumps(r) for r in [
        {"id": "1", "pair": ["alice one", "alice two"]},
        {"id": "2", "pair": ["bob here", "carol there"]},
        {"id": "3", "pair": ["x", "y"]},  # no truth → dropped
    ]))
    truth.write_text("\n".join(json.dumps(r) for r in [
        {"id": "1", "same": True},
        {"id": "2", "same": False},
    ]))
    recs = P._build_pan(
        _args(format="pan", pairs_file=str(pairs), truth_file=str(truth), slice="domain:pan"),
        cfg=None,
    )
    assert len(recs) == 2
    assert all(set(r) == {"pair", "same", "slice"} for r in recs)
    assert all(r["slice"] == "domain:pan" for r in recs)
    assert {r["same"] for r in recs} == {True, False}


def test_pan_max_pairs_stays_balanced(tmp_path):
    pairs = tmp_path / "pairs.jsonl"
    truth = tmp_path / "truth.jsonl"
    prs, trs = [], []
    for i in range(20):
        same = i % 2 == 0
        prs.append({"id": str(i), "pair": [f"text a {i}", f"text b {i}"]})
        trs.append({"id": str(i), "same": same})
    pairs.write_text("\n".join(json.dumps(r) for r in prs))
    truth.write_text("\n".join(json.dumps(r) for r in trs))
    recs = P._build_pan(
        _args(pairs_file=str(pairs), truth_file=str(truth), max_pairs=6),
        cfg=None,
    )
    c = Counter(r["same"] for r in recs)
    assert c[True] == 3 and c[False] == 3


def test_authors_balanced_pairs(tmp_path):
    inp = tmp_path / "authors.jsonl"
    lines = []
    for a in range(4):
        for d in range(4):
            lines.append({"text": f"author {a} doc {d} with plenty of words to pass the floor here",
                          "id": f"a{a}"})
    inp.write_text("\n".join(json.dumps(x) for x in lines))
    recs = P._build_authors(
        _args(format="authors", input=str(inp), author_field="id",
              n_per_class=10, slice="domain:blog"),
        cfg=None,
    )
    c = Counter(r["same"] for r in recs)
    assert c[True] > 0 and c[False] > 0
    assert all(r["slice"] == "domain:blog" for r in recs)


def test_clean_text_truncates():
    long = "x" * 5000
    out = P._clean_text(long, _args(max_chars=100, preprocess=False), cfg=None)
    assert out is not None and len(out) == 100
