"""Post-hoc lineage analysis: confusion matrices + paired cross-arm deltas.

The lineage benchmark (scripts/run_lineage_v6_v9.sh) reports rank metrics with
within-arm scorer bootstraps, but never materializes (a) confusion matrices at
the deployed operating points, (b) length-stratified accept rates (the v9
hypothesis is specifically about short emails), or (c) PAIRED arm-vs-arm
deltas — which are valid here because every arm's common-corpus probe is built
from the same dataset with the same seed, i.e. the identical texts.

For each arm (v6..v9) on the COMMON corpus (enron_shortmail + syn-v2):
  1. Rebuild the probe exactly as ablate_adaptive_scorers.py does
     (n_profile_senders=60→caps, n_enroll=8, n_query=6, 600 other, 600 syn,
     seed=0), encode, fit one ProfileBank, score genuine/other/synthetic pools
     with baseline_linear_z3 and baseline_cosine.
  2. Confusion matrices at three operating points: FPR_syn=1%, FPR_syn=5%
     (threshold = quantile of the synthetic pool), and global EER. The "other"
     (wrong-sender, real-human) pool is scored at the same thresholds so
     FPR_other is visible next to FPR_syn.
  3. Accept rates stratified by query length (<10 / 10-25 / 26-60 / >60 words).
  4. Paired bootstrap of tpr1/tpr5/auc between consecutive arms and v7-vs-v9
     (same resample indices applied to both arms' score arrays).

Outputs: results/lineage/confusion_report.json, scores_<arm>.npz, and figures
under results/lineage/figures/.

Usage:  python scripts/lineage_confusion_report.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")

import email_fraud.data.enron  # noqa: F401
import email_fraud.encoders    # noqa: F401
import email_fraud.heads       # noqa: F401
import email_fraud.losses      # noqa: F401

from email_fraud.config import load_config
from email_fraud.data.enron import EnronDataset
from email_fraud.registry import resolve as resolve_component
from email_fraud.scoring.adaptive import ProfileBank, score_pool
from email_fraud.scoring.metrics import compute_auc, compute_eer, compute_tpr_at_fpr
from email_fraud.utils.logging import setup_logging

from eval_v7_scoring import _build_probe, _encode_texts  # type: ignore

logger = logging.getLogger(__name__)

ARMS = ["v6", "v7", "v8", "v9"]
SCORER_NAMES = ["baseline_linear_z3", "baseline_cosine"]
LEN_BUCKETS = [(0, 9), (10, 25), (26, 60), (61, 10**9)]
LEN_LABELS = ["<10w", "10-25w", "26-60w", ">60w"]
RESDIR = _PROJECT_ROOT / "results" / "lineage"
FIGDIR = RESDIR / "figures"


def wc(texts: list[str]) -> np.ndarray:
    return np.array([len(t.split()) for t in texts])


def threshold_at_syn_fpr(syn_scores: np.ndarray, fpr: float) -> float:
    # Accept iff score >= thr; thr chosen so ~fpr of synthetic pool is accepted.
    return float(np.quantile(syn_scores, 1.0 - fpr, method="higher"))


def eer_threshold(gen: np.ndarray, syn: np.ndarray) -> float:
    from sklearn.metrics import roc_curve
    y = np.concatenate([np.ones_like(gen), np.zeros_like(syn)])
    s = np.concatenate([gen, syn])
    fpr, tpr, thr = roc_curve(y, s)
    idx = int(np.nanargmin(np.abs((1 - tpr) - fpr)))
    return float(thr[idx])


def confusion_at(thr: float, gen, oth, syn) -> dict:
    return {
        "threshold": thr,
        "genuine":   {"accept": int((gen >= thr).sum()), "reject": int((gen < thr).sum()), "n": len(gen)},
        "other":     {"accept": int((oth >= thr).sum()), "reject": int((oth < thr).sum()), "n": len(oth)},
        "synthetic": {"accept": int((syn >= thr).sum()), "reject": int((syn < thr).sum()), "n": len(syn)},
        "tpr": float((gen >= thr).mean()),
        "fpr_other": float((oth >= thr).mean()),
        "fpr_syn": float((syn >= thr).mean()),
    }


def strata_rates(scores: np.ndarray, words: np.ndarray, thr: float) -> dict:
    out = {}
    for (lo, hi), lab in zip(LEN_BUCKETS, LEN_LABELS):
        m = (words >= lo) & (words <= hi)
        out[lab] = {"n": int(m.sum()),
                    "accept_rate": float((scores[m] >= thr).mean()) if m.any() else None}
    return out


def paired_arm_bootstrap(a: dict, b: dict, n_boot=2000, seed=0) -> dict:
    """95% CI on metric(b) - metric(a), same resample for both arms."""
    rng = np.random.default_rng(seed)
    ga, sa, gb, sb = a["genuine"], a["synthetic"], b["genuine"], b["synthetic"]
    assert len(ga) == len(gb) and len(sa) == len(sb)
    metrics = {
        "tpr1": lambda g, s: compute_tpr_at_fpr(
            np.concatenate([np.ones_like(g), np.zeros_like(s)]), np.concatenate([g, s]), 0.01),
        "tpr5": lambda g, s: compute_tpr_at_fpr(
            np.concatenate([np.ones_like(g), np.zeros_like(s)]), np.concatenate([g, s]), 0.05),
        "auc": lambda g, s: compute_auc(
            np.concatenate([np.ones_like(g), np.zeros_like(s)]), np.concatenate([g, s])),
    }
    out = {}
    deltas = {k: [] for k in metrics}
    for _ in range(n_boot):
        gi = rng.integers(0, len(ga), len(ga))
        si = rng.integers(0, len(sa), len(sa))
        for k, fn in metrics.items():
            deltas[k].append(fn(gb[gi], sb[si]) - fn(ga[gi], sa[si]))
    for k in metrics:
        d = np.array(deltas[k])
        out[k] = {
            "point": float(metrics[k](gb, sb) - metrics[k](ga, sa)),
            "dlo": float(np.percentile(d, 2.5)),
            "dhi": float(np.percentile(d, 97.5)),
            "p_b_wins": float((d > 0).mean()),
        }
    return out


def load_encoder(cfg, checkpoint: Path, device: str):
    EncoderClass = resolve_component("encoder", cfg.encoder.name)
    encoder = EncoderClass(cfg.encoder)
    payload = torch.load(str(checkpoint), map_location=device, weights_only=False)
    encoder.load_state_dict(payload["model_state_dict"])
    encoder.to(device)
    encoder.eval()
    logger.info("%s: loaded epoch %s", checkpoint, payload.get("epoch"))
    return encoder, payload.get("epoch")


def main() -> None:
    setup_logging()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    FIGDIR.mkdir(parents=True, exist_ok=True)

    report: dict = {"arms": {}, "paired_deltas": {}}
    arm_scores: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    probe_words = None

    for arm in ARMS:
        cfg_path = _PROJECT_ROOT / f"runs/_lineage/eval_cfgs/{arm}_common.yaml"
        ckpt = _PROJECT_ROOT / f"runs/lineage/{arm}/checkpoint_best.pt"
        if not ckpt.exists():
            ckpt = _PROJECT_ROOT / f"runs/lineage/{arm}/checkpoint_last.pt"
        cfg = load_config(str(cfg_path))
        encoder, epoch = load_encoder(cfg, ckpt, device)

        train_ds = EnronDataset(cfg.data, split="train")
        val_ds = EnronDataset(cfg.data, split="validation")
        probe = _build_probe(
            train_ds, val_ds, syn_path=cfg.data.augmentation.synthetic_path,
            n_profile_senders=60, n_enroll=8, n_query=6,
            n_other=600, n_synth=600, seed=0,
        )
        if probe_words is None:
            probe_words = {
                "genuine": wc(probe["gen_texts"]),
                "other": wc(probe["other_texts"]),
                "synthetic": wc(probe["syn_texts"]),
            }

        enroll_emb = _encode_texts(encoder, probe["enroll_texts"], device)
        gen_emb = _encode_texts(encoder, probe["gen_texts"], device)
        oth_emb = _encode_texts(encoder, probe["other_texts"], device)
        syn_emb = _encode_texts(encoder, probe["syn_texts"], device)

        bank = ProfileBank(ewma_alpha=0.1).fit(enroll_emb, probe["enroll_sids"])
        rng = np.random.default_rng(0)
        chosen = probe["chosen_senders"]
        oth_claimed = [chosen[i] for i in rng.integers(0, len(chosen), len(oth_emb))]

        arm_scores[arm] = {}
        arm_report = {"checkpoint": str(ckpt.relative_to(_PROJECT_ROOT)), "epoch": epoch,
                      "n_senders": len(chosen), "scorers": {}}
        for sname in SCORER_NAMES:
            gen = score_pool(bank, gen_emb, list(probe["gen_sids"]), sname)
            oth = score_pool(bank, oth_emb, oth_claimed, sname)
            syn = score_pool(bank, syn_emb, list(probe["syn_sids"]), sname)
            arm_scores[arm][sname] = {"genuine": gen, "other": oth, "synthetic": syn}

            ops = {}
            for label, thr in [
                ("fpr_syn_1pct", threshold_at_syn_fpr(syn, 0.01)),
                ("fpr_syn_5pct", threshold_at_syn_fpr(syn, 0.05)),
                ("eer", eer_threshold(gen, syn)),
            ]:
                cm = confusion_at(thr, gen, oth, syn)
                cm["genuine_by_len"] = strata_rates(gen, probe_words["genuine"], thr)
                cm["synthetic_by_len"] = strata_rates(syn, probe_words["synthetic"], thr)
                cm["other_by_len"] = strata_rates(oth, probe_words["other"], thr)
                ops[label] = cm
            y = np.concatenate([np.ones_like(gen), np.zeros_like(syn)])
            s = np.concatenate([gen, syn])
            arm_report["scorers"][sname] = {
                "auc_g_syn": compute_auc(y, s),
                "eer_g_syn": compute_eer(y, s),
                "operating_points": ops,
            }
            logger.info("%s/%s: tpr@5%%fprsyn=%.3f fpr_other@same=%.3f", arm, sname,
                        ops["fpr_syn_5pct"]["tpr"], ops["fpr_syn_5pct"]["fpr_other"])

        np.savez(RESDIR / f"scores_{arm}.npz",
                 **{f"{sn}_{pool}": arr for sn, pools in arm_scores[arm].items()
                    for pool, arr in pools.items()},
                 words_genuine=probe_words["genuine"], words_other=probe_words["other"],
                 words_synthetic=probe_words["synthetic"])
        report["arms"][arm] = arm_report
        del encoder
        torch.cuda.empty_cache()

    # Paired cross-arm deltas on the linear_z3 scorer (identical probe texts).
    for a, b in [("v6", "v7"), ("v7", "v8"), ("v8", "v9"), ("v7", "v9"), ("v6", "v9")]:
        report["paired_deltas"][f"{a}->{b}"] = paired_arm_bootstrap(
            arm_scores[a]["baseline_linear_z3"], arm_scores[b]["baseline_linear_z3"])
        logger.info("paired %s->%s done", a, b)

    with (RESDIR / "confusion_report.json").open("w") as fh:
        json.dump(report, fh, indent=2)
    logger.info("Saved %s", RESDIR / "confusion_report.json")

    make_figures(arm_scores, probe_words, report)


def make_figures(arm_scores, probe_words, report) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    sn = "baseline_linear_z3"

    # --- 1. Confusion-matrix grid (genuine vs synthetic + other row) ---
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))
    for j, arm in enumerate(ARMS):
        for i, op in enumerate(["fpr_syn_1pct", "fpr_syn_5pct"]):
            cm = report["arms"][arm]["scorers"][sn]["operating_points"][op]
            mat = np.array([
                [cm["genuine"]["accept"], cm["genuine"]["reject"]],
                [cm["other"]["accept"], cm["other"]["reject"]],
                [cm["synthetic"]["accept"], cm["synthetic"]["reject"]],
            ], dtype=float)
            rates = mat / mat.sum(axis=1, keepdims=True)
            ax = axes[i, j]
            ax.imshow(rates, cmap="Blues", vmin=0, vmax=1)
            for r in range(3):
                for c in range(2):
                    ax.text(c, r, f"{int(mat[r, c])}\n({rates[r, c]:.0%})",
                            ha="center", va="center",
                            color="white" if rates[r, c] > 0.6 else "black", fontsize=10)
            ax.set_xticks([0, 1], ["accept", "reject"])
            ax.set_yticks([0, 1, 2], ["genuine", "other-sender", "synthetic"])
            ax.set_title(f"{arm} @ {'1%' if '1' in op else '5%'} FPR$_{{syn}}$ "
                         f"(thr={cm['threshold']:.3f})", fontsize=11)
    fig.suptitle("Confusion matrices, common corpus (enron_shortmail + syn-v2), "
                 "baseline_linear_z3, K=8", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGDIR / "confusion_grid.png", dpi=130)
    plt.close(fig)

    # --- 2. Score distributions ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.2), sharey=False)
    for j, arm in enumerate(ARMS):
        ax = axes[j]
        sc = arm_scores[arm][sn]
        bins = np.linspace(min(sc["synthetic"].min(), sc["genuine"].min(), sc["other"].min()),
                           max(sc["genuine"].max(), sc["other"].max(), sc["synthetic"].max()), 50)
        ax.hist(sc["genuine"], bins=bins, alpha=0.55, label="genuine", density=True, color="tab:green")
        ax.hist(sc["other"], bins=bins, alpha=0.5, label="other-sender", density=True, color="tab:orange")
        ax.hist(sc["synthetic"], bins=bins, alpha=0.5, label="synthetic", density=True, color="tab:red")
        thr5 = report["arms"][arm]["scorers"][sn]["operating_points"]["fpr_syn_5pct"]["threshold"]
        thr1 = report["arms"][arm]["scorers"][sn]["operating_points"]["fpr_syn_1pct"]["threshold"]
        ax.axvline(thr5, color="k", ls="--", lw=1, label="thr @5% FPR$_{syn}$")
        ax.axvline(thr1, color="k", ls=":", lw=1, label="thr @1% FPR$_{syn}$")
        ax.set_title(arm)
        ax.set_xlabel("score (linear_z3)")
        if j == 0:
            ax.set_ylabel("density")
            ax.legend(fontsize=8)
    fig.suptitle("Score distributions on the common corpus (higher = accepted as claimed sender)")
    fig.tight_layout()
    fig.savefig(FIGDIR / "score_distributions.png", dpi=130)
    plt.close(fig)

    # --- 3. ROC (genuine vs synthetic), log-x ---
    fig, ax = plt.subplots(figsize=(7, 6))
    for arm, color in zip(ARMS, ["tab:gray", "tab:blue", "tab:orange", "tab:green"]):
        sc = arm_scores[arm][sn]
        y = np.concatenate([np.ones_like(sc["genuine"]), np.zeros_like(sc["synthetic"])])
        s = np.concatenate([sc["genuine"], sc["synthetic"]])
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, label=arm, color=color)
    ax.set_xscale("log")
    ax.set_xlim(5e-4, 1)
    ax.axvline(0.01, color="k", ls=":", lw=1)
    ax.axvline(0.05, color="k", ls="--", lw=1)
    ax.set_xlabel("FPR (synthetic impostor accepted) — log scale")
    ax.set_ylabel("TPR (genuine accepted)")
    ax.set_title("ROC, genuine vs synthetic — common corpus, linear_z3, K=8")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "roc_log.png", dpi=130)
    plt.close(fig)

    # --- 4. Length-stratified genuine accept rate at 5% FPR_syn ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    x = np.arange(len(LEN_LABELS))
    width = 0.2
    for k, arm in enumerate(ARMS):
        cm = report["arms"][arm]["scorers"][sn]["operating_points"]["fpr_syn_5pct"]
        gvals = [cm["genuine_by_len"][lab]["accept_rate"] for lab in LEN_LABELS]
        svals = [cm["synthetic_by_len"][lab]["accept_rate"] for lab in LEN_LABELS]
        axes[0].bar(x + (k - 1.5) * width, [v if v is not None else 0 for v in gvals],
                    width, label=arm)
        axes[1].bar(x + (k - 1.5) * width, [v if v is not None else 0 for v in svals],
                    width, label=arm)
    ns_g = [report["arms"]["v9"]["scorers"][sn]["operating_points"]["fpr_syn_5pct"]
            ["genuine_by_len"][lab]["n"] for lab in LEN_LABELS]
    ns_s = [report["arms"]["v9"]["scorers"][sn]["operating_points"]["fpr_syn_5pct"]
            ["synthetic_by_len"][lab]["n"] for lab in LEN_LABELS]
    axes[0].set_title("Genuine accept rate by query length @5% FPR$_{syn}$ (higher = better)")
    axes[1].set_title("Synthetic accept rate by length @ same threshold (lower = better)")
    for ax, ns in zip(axes, [ns_g, ns_s]):
        ax.set_xticks(x, [f"{lab}\n(n={n})" for lab, n in zip(LEN_LABELS, ns)])
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGDIR / "length_strata.png", dpi=130)
    plt.close(fig)

    logger.info("Figures saved under %s", FIGDIR)


if __name__ == "__main__":
    main()
