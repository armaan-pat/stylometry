"""Build a large multi-author held-out verification probe from blog_authors test split.

Two slices, both as {"pair":[a,b], "same":bool, "slice":str} (eval_ood.py format):
  blog:random  — standard: same-author (any topic) positives + different-author
                 (any topic) negatives. Many authors → tight CIs.
  blog:sametopic — CONTENT-CONTROL test: same-author positives + different-author
                 SAME-TOPIC negatives. (The blog corpus `topic` is an author-level
                 industry label, so negatives are two bloggers in the same industry —
                 they share topical vocabulary, so a model leaning on topic/content
                 (the LUAR content-confound, docs/v14_validation_results.md) can't use
                 it to separate them. The gap between blog:random AUC and blog:sametopic
                 AUC measures how much the model relies on topic vs idiolect.)

Usage:
    python scripts/build_blog_probe.py --split test \
        --n-per-slice 3000 --output data/ood/blog_heldout_pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--processed-dir", default="data/processed/blog_authors")
    p.add_argument("--split", default="test")
    p.add_argument("--n-per-slice", type=int, default=3000,
                   help="Same-author positives (and equal #negatives) per slice.")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--output", default="data/ood/blog_heldout_pairs.jsonl")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import numpy as np
    from datasets import load_from_disk

    dd = load_from_disk(args.processed_dir)
    ds = dd[args.split]
    texts, sids, topics = ds["text"], ds["sender_id"], ds["topic"]
    rng = np.random.default_rng(args.seed)

    # Index by author and by (author, topic).
    by_author: dict[str, list[int]] = defaultdict(list)
    by_topic: dict[str, list[int]] = defaultdict(list)
    for i, (s, t) in enumerate(zip(sids, topics)):
        by_author[s].append(i)
        by_topic[t].append(i)
    authors = [a for a, idx in by_author.items() if len(idx) >= 2]
    logger.info("Probe authors: %d | topics: %d | posts: %d", len(authors), len(by_topic), len(texts))

    def author_of(i): return sids[i]
    def topic_of(i): return topics[i]

    pairs = []

    # ---- blog:random ----
    n = args.n_per_slice
    pos = 0
    while pos < n:
        a = authors[rng.integers(0, len(authors))]
        idx = by_author[a]
        i, j = rng.choice(idx, 2, replace=False)
        pairs.append({"pair": [texts[int(i)], texts[int(j)]], "same": True, "slice": "blog:random"})
        pos += 1
    neg = 0
    while neg < n:
        a1, a2 = rng.choice(len(authors), 2, replace=False)
        i = by_author[authors[a1]][rng.integers(0, len(by_author[authors[a1]]))]
        j = by_author[authors[a2]][rng.integers(0, len(by_author[authors[a2]]))]
        pairs.append({"pair": [texts[int(i)], texts[int(j)]], "same": False, "slice": "blog:random"})
        neg += 1

    # ---- blog:sametopic (content-control) ----
    # same-author positives (topic shared by construction since topic is author-level)
    pos = 0
    while pos < n:
        a = authors[rng.integers(0, len(authors))]
        idx = by_author[a]
        i, j = rng.choice(idx, 2, replace=False)
        pairs.append({"pair": [texts[int(i)], texts[int(j)]], "same": True, "slice": "blog:sametopic"})
        pos += 1
    # different-author SAME-topic (same industry) negatives — share topical vocabulary
    topics_multi = [t for t, idx in by_topic.items() if len({sids[k] for k in idx}) >= 2]
    neg = 0
    tries = 0
    while neg < n and tries < n * 50:
        tries += 1
        t = topics_multi[rng.integers(0, len(topics_multi))]
        idx = by_topic[t]
        i, j = rng.choice(idx, 2, replace=False)
        if author_of(int(i)) != author_of(int(j)):
            pairs.append({"pair": [texts[int(i)], texts[int(j)]], "same": False, "slice": "blog:sametopic"})
            neg += 1
    logger.info("sametopic: %d same-author pos, %d diff-author-same-topic neg", pos, neg)

    rng.shuffle(pairs)
    from pathlib import Path
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for r in pairs:
            fh.write(json.dumps(r) + "\n")
    from collections import Counter
    logger.info("Wrote %d pairs → %s | %s", len(pairs), out, dict(Counter(r["slice"] for r in pairs)))


if __name__ == "__main__":
    main()
