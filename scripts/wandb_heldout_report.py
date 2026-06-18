#!/usr/bin/env python
"""Publish a W&B summary run that surfaces the HELD-OUT eval numbers (results/),
not the misleading in-distribution training metrics.

It creates one run `heldout-eval-summary` in klconvergence/email-fraud-detection with:
  - a comparison Table (v11..v14b: held-out pool AUC / TPR@1% / TPR@5% / FPR_other,
    per-generator AUC, AND the in-distribution val tpr_at_fpr/all_1pct for contrast),
  - bar-chart panels for each headline metric,
  - the docs/figures/*.png embedded as images.

Then it tries to assemble those panels into a W&B Report (best-effort; if the Reports
API isn't available in this wandb build, the run + panels are still fully usable and can
be pinned into a report from the UI).

Run: WANDB_MODE=online python scripts/wandb_heldout_report.py
"""
from __future__ import annotations
import json, os, pathlib
import wandb

ROOT = pathlib.Path(__file__).resolve().parents[1]
R = ROOT / "results"
FIGS = ROOT / "docs" / "figures"
ENTITY, PROJECT = "klconvergence", "email-fraud-detection"
SCORER = "mahalanobis"  # deployment scorer; guardrail-satisfying; fair across versions

# in-distribution val metric (the "trap"): read straight from each training run's summary
VAL_RUN_IDS = {"v11": "xn2f9pao", "v12": "cfm18bjw", "v13": "cu10v51w",
               "v14": "xfnd378v", "v14b": "0hg6m78r"}
CLAUDE = "gen:openrouter:anthropic/claude-3.5-haiku"
GEMINI = "gen:openrouter:google/gemini-2.5-flash"


def L(p):
    p = R / p
    return json.loads(p.read_text()) if p.exists() else None

def prow(d, scorer=SCORER):
    if not d or "rows" not in d:
        return None
    return next((r for r in d["rows"] if r["scorer"] == scorer), None)

def sauc(d, s):
    return round(d[s]["AUC"], 4) if d and s in d and "AUC" in d[s] else None


ABL = {
    "v11": L("v12/heldoutCG_v11lora.json"),
    "v12": L("v12/heldoutCG_v12lora_final.json"),
    "v13": L("v13/heldoutCG_v13lora.json"),
    "v14": L("v14/heldoutCG_v14lora.json"),
    "v14b": L("v14/heldoutCG_v14blora.json"),
}
OOD = {
    "v11": L("v12/ood_v11lora_heldoutCG.json"),
    "v12": L("v12/ood_v12lora_final_heldoutCG.json"),
    "v13": L("v13/ood_v13lora_heldoutCG.json"),
    "v14": L("v14/ood_v14lora_heldoutCG.json"),
    "v14b": L("v14/ood_v14blora_heldoutCG.json"),
}
VERS = ["v11", "v12", "v13", "v14", "v14b"]

# --- pull the in-distribution val metric from each training run (authoritative) ---
api = wandb.Api()
val_all1 = {}
for v, rid in VAL_RUN_IDS.items():
    try:
        s = api.run(f"{ENTITY}/{PROJECT}/{rid}").summary
        val_all1[v] = round(float(s["tpr_at_fpr/all_1pct"]), 4) if "tpr_at_fpr/all_1pct" in s else None
    except Exception:
        val_all1[v] = None

# --- assemble rows ---
rows = []
for v in VERS:
    r = prow(ABL[v])
    rows.append({
        "version": v,
        "pool_AUC_heldout": round(r["auc"], 4) if r else None,
        "TPR@1%_heldout": round(r["tpr1"], 4) if r else None,
        "TPR@5%_heldout": round(r["tpr5"], 4) if r else None,
        "FPR_other@5 (<=0.10)": round(r.get("fpr_other_at_5"), 4) if r else None,
        "AUC_Claude_heldout": sauc(OOD[v], CLAUDE),
        "AUC_Gemini_heldout": sauc(OOD[v], GEMINI),
        "val_all_1pct (in-distribution, NOT comparable)": val_all1[v],
    })

run = wandb.init(
    entity=ENTITY, project=PROJECT, name="heldout-eval-summary",
    job_type="eval-summary", tags=["summary", "held-out", "cross-generator", "report"],
    notes=(f"Apples-to-apples held-out Claude+Gemini eval ({SCORER} scorer) for v11..v14b. "
           "Surfaces results/ numbers, NOT the in-distribution training metrics. "
           "val_all_1pct shown only to demonstrate the trap: it is per-run in-distribution "
           "and not comparable across versions."),
    config={"scorer": SCORER, "eval": "held-out Claude+Gemini pool",
            "pool": "264 genuine / 327 LLM-forgery / 600 wrong-human"},
)

# --- comparison table ---
cols = list(rows[0].keys())
tbl = wandb.Table(columns=cols, data=[[r[c] for c in cols] for r in rows])
run.log({"heldout_comparison": tbl})

# --- per-metric bar charts (each a real W&B panel) ---
def bar(metric_key, title):
    t = wandb.Table(data=[[r["version"], r[metric_key]] for r in rows if r[metric_key] is not None],
                    columns=["version", metric_key])
    return wandb.plot.bar(t, "version", metric_key, title=title)

run.log({
    "chart/pool_AUC_heldout": bar("pool_AUC_heldout", "Held-out pool AUC (mahalanobis)"),
    "chart/TPR_at_5pct_heldout": bar("TPR@5%_heldout", "Forgeries caught @5% FPR (held-out)"),
    "chart/TPR_at_1pct_heldout": bar("TPR@1%_heldout", "TPR @1% FPR (held-out)"),
    "chart/FPR_other_at_5": bar("FPR_other@5 (<=0.10)", "Wrong-human leak @5% (guardrail <=0.10)"),
    "chart/AUC_Gemini_heldout": bar("AUC_Gemini_heldout", "Held-out Gemini AUC"),
    "chart/AUC_Claude_heldout": bar("AUC_Claude_heldout", "Held-out Claude AUC"),
    "chart/val_all_1pct_TRAP": bar("val_all_1pct (in-distribution, NOT comparable)",
                                   "TRAP: in-distribution val all_1pct (do NOT compare across runs)"),
})

# --- summary scalars (for the run overview / sortable table) ---
best = rows[-1]
run.summary.update({
    "best_model": "v14b",
    "v14b/pool_AUC_heldout": best["pool_AUC_heldout"],
    "v14b/TPR@5%_heldout": best["TPR@5%_heldout"],
    "v14b/FPR_other@5": best["FPR_other@5 (<=0.10)"],
    "v11/pool_AUC_heldout": rows[0]["pool_AUC_heldout"],
    "v11/TPR@5%_heldout": rows[0]["TPR@5%_heldout"],
})

# --- embed the presentation figures ---
imgs = {}
for p in sorted(FIGS.glob("*.png")):
    imgs[f"figure/{p.stem}"] = wandb.Image(str(p))
if imgs:
    run.log(imgs)

run_url = run.url
run.finish()
print("Summary run:", run_url)

# --- best-effort: assemble a W&B Report ---
report_url = None
try:
    import wandb_workspaces.reports.v2 as wr
    report = wr.Report(
        entity=ENTITY, project=PROJECT,
        title="Held-out cross-generator eval (v11 -> v14b)",
        description=("The metrics that match the task: detect forgeries from UNSEEN generators "
                     "against a sender's average-style centroid. Numbers from results/, "
                     f"{SCORER} scorer. The in-distribution training panels (tpr_at_fpr/*) are "
                     "NOT comparable across runs — see the TRAP chart."),
        blocks=[
            wr.H1("Which model is actually best at the task"),
            wr.P("Held-out Claude+Gemini pool (never trained on). Higher AUC/TPR is better; "
                 "FPR_other must stay <= 0.10."),
            wr.PanelGrid(
                runsets=[wr.Runset(ENTITY, PROJECT, name="eval-summary",
                                   filters={"display_name": "heldout-eval-summary"})],
                panels=[
                    wr.BarPlot(title="Held-out pool AUC", metrics=["chart/pool_AUC_heldout"]),
                    wr.BarPlot(title="Forgeries caught @5%", metrics=["chart/TPR_at_5pct_heldout"]),
                    wr.BarPlot(title="Wrong-human leak @5% (<=0.10)", metrics=["chart/FPR_other_at_5"]),
                    wr.BarPlot(title="TRAP: in-distribution val all_1pct", metrics=["chart/val_all_1pct_TRAP"]),
                ],
            ),
        ],
    )
    report.save()
    report_url = report.url
    print("Report:", report_url)
except Exception as e:
    print(f"(Report API unavailable: {type(e).__name__}: {e})")
    print("-> The summary run above already carries every panel; pin them into a report from the UI.")
