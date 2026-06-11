"""Tests for the OOD evaluation harness (pairwise scoring + builder + evaluator).

Covers:
  - scoring.pairwise.score_pairs: identical text scores 1.0, range [0,1], dedup
  - build_ood_eval length buckets and balanced length/register slices on a
    realistic multi-email-per-sender fixture
  - build_ood_eval generator/impersonation slices from a synthetic Arrow set
  - eval_ood._load_sliced_pairs parsing and _metrics_table rendering
"""

from __future__ import annotations

import json
import random
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

import build_ood_eval as B
import eval_ood as E
from email_fraud.scoring.pairwise import score_pairs


class _FakeEncoder:
    """Deterministic 3-d embedding keyed on the first character — no model load."""

    def tokenize(self, texts):
        return {"ids": torch.tensor([[ord(t[0]) if t else 0] for t in texts])}

    def encode(self, ids=None):
        rows = [[float(c[0] % 5), float((c[0] // 5) % 5), 1.0] for c in ids.tolist()]
        return torch.tensor(rows)


# --------------------------------------------------------------------------- #
# pairwise scoring
# --------------------------------------------------------------------------- #

def test_score_pairs_identical_is_one_and_in_range():
    enc = _FakeEncoder()
    pairs = [("apple", "apple"), ("apple", "banana"), ("cherry", "date")]
    s = score_pairs(enc, pairs, "cpu", batch_size=2, show_progress=False)
    assert s.shape == (3,)
    assert abs(s[0] - 1.0) < 1e-5  # cosine(self) = 1 → mapped to 1.0
    assert np.all((s >= 0.0) & (s <= 1.0))


def test_score_pairs_empty():
    enc = _FakeEncoder()
    s = score_pairs(enc, [], "cpu", show_progress=False)
    assert len(s) == 0


# --------------------------------------------------------------------------- #
# builder: length buckets + slices
# --------------------------------------------------------------------------- #

def test_len_bucket_thresholds():
    assert B._len_bucket("ok thanks") == "short"
    assert B._len_bucket("word " * 50) == "medium"
    assert B._len_bucket("word " * 120) == "long"


def _realistic_senders(n: int = 10) -> dict[str, list[str]]:
    cas = "hey wanna grab lunch this weekend haha that place was awesome thanks again hope the kids "
    formal = "Please confirm the budget report and forward the signed agreement to management for review. "
    long = "Please review the attached contract regarding the quarterly budget proposal and confirm approval. "
    out = {}
    for k in range(n):
        out[f"s{k}"] = [
            f"ok thanks {k}", f"sounds good {k}", f"see you soon {k}",   # short ×3
            (cas * 2) + f"note {k}", (cas * 2) + f"update {k}",          # casual medium ×2
            (formal * 2) + f"{k}",                                       # formal medium
            (long * 6) + f"v1 {k}", (long * 6) + f"v2 {k}",              # long ×2
        ]
    return out


def test_length_and_register_slices_are_balanced():
    by_sender = _realistic_senders()
    rec: list[dict] = []
    B._build_length_slices(rec, by_sender, 15, random.Random(0))
    B._build_register_slices(rec, by_sender, 15, random.Random(0))

    cls: dict[str, Counter] = defaultdict(Counter)
    for r in rec:
        cls[r["slice"]][r["same"]] += 1

    for expected in ("len:short", "len:medium", "len:long", "lenmix:short_long",
                     "register:cross", "register:same"):
        assert expected in cls, f"missing slice {expected}"
        assert cls[expected][True] > 0 and cls[expected][False] > 0, (expected, cls[expected])

    # lenmix really mixes a short and a long email
    for r in rec:
        if r["slice"] == "lenmix:short_long":
            assert {B._len_bucket(r["pair"][0]), B._len_bucket(r["pair"][1])} == {"short", "long"}
    # register:cross really crosses formal/casual
    for r in rec:
        if r["slice"] == "register:cross":
            from email_fraud.data.register import detect_register
            assert {detect_register(r["pair"][0]), detect_register(r["pair"][1])} == {"formal", "casual"}


def test_generator_slices_from_synthetic(tmp_path):
    from datasets import Dataset

    syn_dir = str(tmp_path / "syn")
    Dataset.from_dict({
        "text": ["imp A1", "imp A2", "imp B1"],
        "sender_id": ["s0__syn", "s0__syn", "s1__syn"],
        "source_sender_id": ["s0", "s0", "s1"],
        "generator": ["groq:llamaX", "groq:llamaX", "gemini:flash"],
    }).save_to_disk(syn_dir)

    by_sender = _realistic_senders()
    rec: list[dict] = []
    B._build_generator_slices(rec, by_sender, syn_dir, 5, random.Random(1))

    cls: dict[str, Counter] = defaultdict(Counter)
    for r in rec:
        cls[r["slice"]][r["same"]] += 1
    assert {"gen:groq:llamaX", "gen:gemini:flash"} <= set(cls)
    for sl, c in cls.items():
        assert c[True] > 0 and c[False] > 0, (sl, c)


# --------------------------------------------------------------------------- #
# evaluator: loading + table
# --------------------------------------------------------------------------- #

def test_load_sliced_pairs_formats(tmp_path):
    p = tmp_path / "pairs.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in [
        {"pair": ["a", "b"], "same": True, "slice": "len:short"},
        {"text1": "c", "text2": "d", "label": 0, "slice": "gen:x"},
        {"pair": ["e", "f"], "same": False},  # no slice → default
    ]))
    a, b, labels, slices = E._load_sliced_pairs(p, default_slice="dom:pan")
    assert a == ["a", "c", "e"] and b == ["b", "d", "f"]
    assert labels == [1, 0, 0]
    assert slices == ["len:short", "gen:x", "dom:pan"]


def test_load_all_splits_merges_disjoint(tmp_path):
    from datasets import Dataset, DatasetDict

    proc = str(tmp_path / "proc")
    DatasetDict({
        "train": Dataset.from_dict({"text": ["a", "b"], "sender_id": ["tr0", "tr1"]}),
        "test": Dataset.from_dict({"text": ["c", "d"], "sender_id": ["te0", "te1"]}),
    }).save_to_disk(proc)
    merged = B._load_all_splits(proc)
    assert set(merged) == {"tr0", "tr1", "te0", "te1"}


def test_generator_slices_with_test_sender_synthetics(tmp_path):
    """(a) Impersonation slices work when synthetics come from the test split."""
    from datasets import Dataset

    real = _realistic_senders()                 # acts as the merged real lookup
    src = next(iter(real))                       # an existing sender id
    syn_dir = str(tmp_path / "syn")
    Dataset.from_dict({
        "text": ["impostor one", "impostor two"],
        "sender_id": [f"{src}__syn", f"{src}__syn"],
        "source_sender_id": [src, src],
        "generator": ["groq:x", "groq:x"],
        "source_split": ["test", "test"],
    }).save_to_disk(syn_dir)

    B._report_synthetic_provenance(syn_dir)      # should not raise
    rec: list[dict] = []
    B._build_generator_slices(rec, real, syn_dir, 4, random.Random(0))
    cls: dict[str, Counter] = defaultdict(Counter)
    for r in rec:
        cls[r["slice"]][r["same"]] += 1
    assert cls["gen:groq:x"][True] > 0 and cls["gen:groq:x"][False] > 0


def test_axis_of_and_aggregates():
    assert E._axis_of("len:short") == "len"
    assert E._axis_of("gen:groq:llama-3.3") == "gen"
    assert E._axis_of("overall") == "overall"

    def row(auc, pauc):
        return {"AUC": auc, "pAUC@5%": pauc, "TPR@FPR=1%": 0.1, "EER": 0.3}

    rows = {
        "len:short": row(0.6, 0.10),
        "len:long": row(0.8, 0.30),
        "gen:groq:x": row(0.7, 0.20),
        "overall": row(0.72, 0.19),  # excluded from aggregates
    }
    agg = E._axis_aggregates(rows)
    assert set(agg) == {"len", "gen"}                 # 'overall' not an axis
    assert agg["len"]["n_slices"] == 2
    assert abs(agg["len"]["pAUC@5%_mean"] - 0.20) < 1e-9
    assert abs(agg["len"]["AUC_mean"] - 0.70) < 1e-9


def test_metrics_table_renders():
    rows = {
        "len:short": {"AUC": 0.6, "pAUC@5%": 0.1, "TPR@FPR=1%": 0.05,
                      "TPR@FPR=5%": 0.2, "EER": 0.4, "c@1": 0.55, "n": 60.0},
        "overall": {"AUC": 0.8, "pAUC@5%": 0.3, "TPR@FPR=1%": 0.2,
                    "TPR@FPR=5%": 0.5, "EER": 0.25, "c@1": 0.7, "n": 120.0},
    }
    table = E._metrics_table(rows)
    assert "len:short" in table and "overall" in table
    # overall is rendered last
    assert table.index("len:short") < table.index("overall")
