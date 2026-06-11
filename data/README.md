# Dataset Guide

Place raw corpora under `data/raw/<dataset-name>/` (gitignored).
Processed/tokenised splits go under `data/processed/<dataset-name>/` (also gitignored).

---

## 1. Enron Email Corpus

**Purpose:** Primary pretraining corpus + per-sender profile construction.

**Source:** <https://www.cs.cmu.edu/~enron/>  
**Size:** ~500k emails from ~150 employees.

**Download:**
```bash
curl -O https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz
tar --no-same-owner -xzf enron_mail_20150507.tar.gz -C data/raw/enron/
```

**Expected structure:**
```
data/raw/enron/
└── maildir/
    ├── allen-p/
    │   ├── inbox/
    │   └── sent/
    └── ...
```

**Notes:**
- Use `sent/` folders for clean single-author attribution.
- Filter to senders with ≥ 25 emails for profile training.
- Apply `preprocess()` (strip quotes, signatures) before tokenisation.

---

## 2. PAN Authorship Verification Datasets

**Purpose:** Authorship verification (AV) benchmarking — AUC, c@1, F0.5u, EER.

**Source:** <https://pan.webis.de/> — PAN 2020, 2021, 2022 shared tasks.

**Download:** Register at <https://zenodo.org/communities/pan-webis/> and download
the relevant year's dataset.  Place at:
```
data/raw/pan/
├── pan20-authorship-verification-training-small/
├── pan20-authorship-verification-training-large/
└── pan21-authorship-verification/
```

**Format:** JSONL files with `{id, pair: [text1, text2], same: bool}`.

**Notes:**
- PAN 2020 fanfiction domain; PAN 2021 cross-topic.
- Use `scripts/evaluate.py` to compute all PAN metrics automatically.

---

## 3. Phishing Corpus (Figshare + SpamAssassin)

**Purpose:** Fraud-side negatives — emails written to *mimic* a legitimate sender.

**Sources:**
- Figshare phishing: <https://figshare.com/articles/dataset/Phishing_Email_Dataset/19148858>
- SpamAssassin public corpus: <https://spamassassin.apache.org/old/publiccorpus/>

**Download:**
```bash
# SpamAssassin
curl -O https://spamassassin.apache.org/old/publiccorpus/20050311_spam_2.tar.bz2
tar -xjf 20050311_spam_2.tar.bz2 -C data/raw/spamassassin/
```

**Expected structure:**
```
data/raw/phishing/
├── figshare_phishing.csv    # columns: subject, body, label
└── spamassassin/
    ├── spam/
    └── ham/
```

**Notes:**
- Label all phishing emails with a synthetic `sender_id = "__phishing__"`.
- Used as hard negatives during contrastive training to push fraud embeddings
  away from legitimate sender clusters.

---

## 4. LLM-Impersonation Examples

**Purpose:** Hard negatives — LLM-generated emails designed to mimic a specific sender.

**Source:** Derived from the dataset accompanying:  
> "Can LLMs Impersonate Authors in Written Communication? A Study on Stylistic Mimicry"  
> arXiv:2509.00005

**Download:** See the paper's GitHub / Zenodo link for the release.

**Expected structure:**
```
data/raw/llm_impersonation/
├── impersonation_pairs.jsonl   # {target_sender, generated_text, source_text}
└── metadata.json
```

**Notes:**
- These are the *hardest* negatives: same topic, same claimed sender, different model.
- Use entity masking (`entity_masking: true` in preprocessing config) when ablating
  to ensure the model cannot cheat by memorising names.
- Recommend 20% of each training batch be LLM-impersonation negatives once the
  base model has converged on Enron.

---

## 5. Out-of-Distribution (OOD) Evaluation

The headline test AUC hides where the model actually fails. Build a **sliced**
OOD pair-set and report metrics per failure axis (short emails, register shifts,
unseen impersonator LLMs, new domains).

**In-distribution + impersonation slices** (length / register / generator), from
the processed Enron splits + the generator-tagged synthetic set:
```bash
python scripts/build_ood_eval.py --config configs/experiments/v9_episodic_shortmail.yaml \
    --output data/ood/enron_ood_pairs.jsonl
python scripts/eval_ood.py --run runs/<run_dir> --pairs data/ood/enron_ood_pairs.jsonl
```

**Unseen-sender impersonation:** generate impostors from the *test* split, then
build slices against them (eval only — never train on these):
```bash
python scripts/generate_synthetic_emails.py --config <cfg> --from-split test \
    --generators groq:llama-3.3-70b-versatile gemini:gemini-1.5-flash \
    --output data/synthetic/enron_test_impostors
python scripts/build_ood_eval.py --config <cfg> \
    --synthetic-path data/synthetic/enron_test_impostors --output data/ood/unseen_pairs.jsonl
```

**Domain OOD** (real text from another domain — the one axis Enron can't
synthesise). Download PAN (§2) or any `{text, author}` corpus, convert, fold in:
```bash
# PAN authorship-verification pairs+truth → one domain slice
python scripts/prepare_external_pairs.py --format pan \
    --pairs-file data/raw/pan/pan21/pairs.jsonl \
    --truth-file data/raw/pan/pan21/truth.jsonl \
    --slice domain:pan21 --output data/ood/pan21_pairs.jsonl
# Generic {text, author} corpus (Blog Authorship, Reddit, Amazon, …)
python scripts/prepare_external_pairs.py --format authors \
    --input data/raw/blog/blogs.jsonl --author-field id \
    --slice domain:blog --output data/ood/blog_pairs.jsonl

python scripts/eval_ood.py --run runs/<run_dir> --pairs data/ood/enron_ood_pairs.jsonl \
    --extra-pairs domain:pan21=data/ood/pan21_pairs.jsonl \
    --extra-pairs domain:blog=data/ood/blog_pairs.jsonl
```
