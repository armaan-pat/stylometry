# Results & Roadmap — Email Fraud Detection

Experimental findings, current gaps, and the prioritized next steps. For how the
system works see [architecture.md](architecture.md); for metric definitions see
[metrics.md](metrics.md). Detailed research logs live in
[v6_vs_v7_memo.md](v6_vs_v7_memo.md) and
[experiments/v7/CHANGELOG_V7.md](../experiments/v7/CHANGELOG_V7.md).

*Last updated: 2026-05-28 (V7).*

---

## V2 — backbone comparison

All V2 experiments run 20 epochs on Enron (sender-disjoint train/val/test). The
"v2" series adds a linear projection head after the backbone. Metrics at epoch
20. The key metric is `val/auroc`: "does a same-sender pair score higher than a
different-sender pair?" — 0.5 = random, 1.0 = perfect.

| Experiment | val/auroc | val/loss | val/intra_cos | val/inter_cos |
|---|---|---|---|---|
| **v2_luar_lora** | **0.9556** | 0.8976 | 0.453 | 0.198 |
| **v2_luar_frozen_proj** | **0.9333** | 0.8711 | 0.456 | 0.142 |
| v2_roberta_frozen_proj | 0.5369 | 4.297 | 0.818 | 0.807 |
| v2_roberta_lora_proj | 0.5257 | 4.356 | 0.768 | 0.762 |
| v2_mpnet_frozen_proj | 0.5341 | 4.364 | 0.853 | 0.841 |
| v2_mpnet_lora_proj | 0.4853 | 4.221 | 0.840 | 0.839 |

**LUAR-MUD is the clear winner.** LUAR was pretrained specifically for authorship
— its embeddings already encode stylometric structure before fine-tuning.
RoBERTa and MPNet sit near 0.5 because their representations are dominated by
topic, not style. (See [architecture.md](architecture.md#why-luar-outperforms-roberta).)

**LoRA vs. frozen.** For LUAR, LoRA (0.956) edges out frozen + projection
(0.933) — the 2-point gain suggests the backbone can be nudged to be more
style-aware, though frozen is already strong and much cheaper. For RoBERTa/MPNet
LoRA made no difference: fine-tuning does not fix the wrong inductive bias.

---

## V7 — Mahalanobis scoring + retrained encoder (2026-05-28)

Full log: [experiments/v7/CHANGELOG_V7.md](../experiments/v7/CHANGELOG_V7.md).

- **V7.0** — per-sender Ledoit-Wolf Mahalanobis distance, computed from the
  stored enrollment embeddings, replaces (or complements) the cosine z-score in
  `PrototypicalHead`. Wins **+1–3 AUC pp** on genuine-vs-synthetic and **+6–9 pp
  on TPR@5%FPR** across K = 8…25 over the v6 encoder.
- **V7.2** — the win widens with enrollment size: at K=16, Mahalanobis adds
  **+3.1 AUC pp** and **+9.2 TPR@5%FPR pp** over cosine. Recommended production
  scorer is `adaptive_k` (cosine for k<5, Mahalanobis for k≥5), implemented and
  tested in `src/email_fraud/heads/prototypical.py`.
- **V7.3** — retrained the encoder with `n_syn=4` (vs 2), `temperature=0.05`
  (vs 0.07), and an added LoRA target (key). Config:
  `configs/experiments/v7_luar_lora_syn_mahal.yaml`.

```bash
python scripts/train.py --config configs/experiments/v7_luar_lora_syn_mahal.yaml
python scripts/eval_v7_full.py --v7-checkpoint <path>
```

---

## What's done

- Full training pipeline: data prep → PKSampler → contrastive loss → checkpoint
- Two production losses: `SupConLoss` and batch-hard `TripletLoss`
- `HFEncoder` with LoRA, mean/CLS/LUAR pooling, optional projection
- `PrototypicalHead`: centroid fitting, cosine z-score **and** per-sender
  Ledoit-Wolf Mahalanobis (V7), confidence tiers, abstain flag
- `SenderProfileStore`: EWMA online updates, JSON persistence
- `ScoringPipeline`: end-to-end raw text → anomaly score
- Preprocessing: quote/sig stripping, entity masking, Unicode normalization
- Config system (Pydantic v2 + YAML deep-merge, typo-safe) and decorator registry
- W&B integration: run resumption, loss/AUROC/KNN/probe metrics per epoch
- Synthetic LLM hard-negative generation and synthetic-balanced training
- Experiment configs covering the major backbone/loss/scoring combinations

## Gaps

| Gap | File | Impact |
|---|---|---|
| PAN metrics in Trainer validation | `training/trainer.py` | Validation logs probe AUROC; full AUC/EER/c@1/F0.5u only in `evaluate.py` |
| Test-set eval on best checkpoint | `scripts/evaluate.py` | Best LUAR-LoRA checkpoint not yet benchmarked on held-out test pairs |
| pgvector backend for profiles | `profiles/store.py` | Profiles are an in-memory dict; production scale needs a persistent vector DB |

---

## Next experiments (priority order)

1. **Run `evaluate.py` on the best LUAR-LoRA checkpoint** — final PAN numbers
   (AUC, EER, c@1, F0.5u) on the held-out test split. These are the reportable
   numbers.
2. **Increase training senders** — currently 100; Enron has 150+ usable senders.
   More senders = more diverse negatives = better generalization.
3. **Per-sender threshold calibration** — some senders write more consistently
   than others; a global threshold underperforms. Calibrate per-sender on
   held-out enrollment emails.
4. **Cross-domain transfer** — evaluate the best checkpoint zero-shot on PAN
   2020/2021 authorship verification (fanfiction/news). Transfer would be strong
   evidence of genuine stylometric generalization.
5. **Ensemble with classical features** — character n-grams, function-word
   distributions, punctuation patterns. Orthogonal to neural reps and more
   robust to adversarial paraphrasing.
6. **(If revisited) fix LoRA for RoBERTa/MPNet** — likely `lr=2e-4` too high and
   too few epochs; try `lr=5e-5`, 30–50 epochs, `r=8`. May still trail LUAR.
