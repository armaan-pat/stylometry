"""Build an authorship dataset from the Blog Authorship Corpus — many identities.

Motivation (docs/v14_validation_results.md): our Enron processed corpus has only
44 train / 6 val / 6 test senders. That tiny identity count is the root bottleneck —
it caps content-invariance (too few authors × topics to decorrelate style from topic)
and makes low-FPR eval untrustworthy (±0.12 TPR@1% CIs on 6 test senders). The Blog
Authorship Corpus (paoramen/blog-authorship-corpus, ~19k authors, 681k posts, WITH a
`topic` column) fixes both: identity diversity for training, and a large held-out probe.

Outputs an Arrow DatasetDict matching EnronDataset's contract ({text, sender_id}, plus
`topic` kept for content-controlled eval). Sender-disjoint train/val/test. sender_ids are
namespaced `blog:<id>` so this can be merged with Enron (`enron:<id>`) without collision.

Usage:
    python scripts/prepare_blog_authors.py \
        --output data/processed/blog_authors \
        --min-posts 24 --max-posts 60 --min-words 10 \
        --n-train-authors 800 --n-val-authors 150 --n-test-authors 150 \
        --merge-enron data/processed/enron_shortmail --merged-output data/processed/enron_blog
"""

from __future__ import annotations

import argparse
import logging
import re
from collections import defaultdict

logger = logging.getLogger(__name__)
_WS = re.compile(r"\s+")


def _clean(text: str) -> str:
    try:
        import ftfy
        text = ftfy.fix_text(text)
    except Exception:
        pass
    return _WS.sub(" ", text).strip()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf-dataset", default="paoramen/blog-authorship-corpus")
    p.add_argument("--output", default="data/processed/blog_authors")
    p.add_argument("--min-posts", type=int, default=24, help="Min qualifying posts to keep an author.")
    p.add_argument("--max-posts", type=int, default=60, help="Cap posts/author (balance + bound size).")
    p.add_argument("--min-words", type=int, default=10)
    p.add_argument("--max-words", type=int, default=400, help="Truncate very long posts to this many words.")
    p.add_argument("--n-train-authors", type=int, default=800)
    p.add_argument("--n-val-authors", type=int, default=150)
    p.add_argument("--n-test-authors", type=int, default=150)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--merge-enron", default=None, help="Path to Enron processed dir to merge with.")
    p.add_argument("--merged-output", default="data/processed/enron_blog")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import numpy as np
    from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

    logger.info("Loading %s ...", args.hf_dataset)
    ds = load_dataset(args.hf_dataset, split="train")

    # Single pass: accumulate up to max_posts qualifying posts per author.
    by_author: dict[str, list[dict]] = defaultdict(list)
    ids, texts, topics = ds["id"], ds["text"], ds["topic"]
    for aid, txt, top in zip(ids, texts, topics):
        bucket = by_author[aid]
        if len(bucket) >= args.max_posts:
            continue
        c = _clean(txt)
        w = c.split()
        if len(w) < args.min_words:
            continue
        if len(w) > args.max_words:
            c = " ".join(w[: args.max_words])
        bucket.append({"text": c, "sender_id": f"blog:{aid}", "topic": str(top)})

    authors = [a for a, posts in by_author.items() if len(posts) >= args.min_posts]
    logger.info("Authors with >=%d qualifying posts: %d", args.min_posts, len(authors))

    rng = np.random.default_rng(args.seed)
    rng.shuffle(authors)
    n_tr, n_va, n_te = args.n_train_authors, args.n_val_authors, args.n_test_authors
    need = n_tr + n_va + n_te
    if len(authors) < need:
        raise SystemExit(f"Only {len(authors)} eligible authors < requested {need}. Lower --min-posts/caps.")
    train_a = set(authors[:n_tr])
    val_a = set(authors[n_tr:n_tr + n_va])
    test_a = set(authors[n_tr + n_va:n_tr + n_va + n_te])

    def recs(author_set: set[str]) -> list[dict]:
        out = []
        for a in author_set:
            out.extend(by_author[a])
        rng.shuffle(out)
        return out

    train_recs, val_recs, test_recs = recs(train_a), recs(val_a), recs(test_a)
    logger.info("Blog split — train %d posts/%d authors | val %d/%d | test %d/%d",
                len(train_recs), len(train_a), len(val_recs), len(val_a), len(test_recs), len(test_a))

    blog_dd = DatasetDict({
        "train": Dataset.from_list(train_recs),
        "validation": Dataset.from_list(val_recs),
        "test": Dataset.from_list(test_recs),
    })
    blog_dd.save_to_disk(args.output)
    logger.info("Saved blog dataset → %s", args.output)

    # --- Optional merge with Enron (namespaced) for v14 training ---
    if args.merge_enron:
        logger.info("Merging with Enron at %s ...", args.merge_enron)
        enron = load_from_disk(args.merge_enron)
        merged = {}
        for split in ["train", "validation", "test"]:
            e = enron[split]
            e_recs = [{"text": t, "sender_id": f"enron:{s}", "topic": "enron"}
                      for t, s in zip(e["text"], e["sender_id"])]
            b_recs = blog_dd[split].to_list()
            allr = e_recs + b_recs
            rng.shuffle(allr)
            merged[split] = Dataset.from_list(allr)
            ns = len({r["sender_id"] for r in allr})
            logger.info("  merged %s: %d rows, %d senders", split, len(allr), ns)
        DatasetDict(merged).save_to_disk(args.merged_output)
        logger.info("Saved merged Enron+blog → %s", args.merged_output)


if __name__ == "__main__":
    main()
