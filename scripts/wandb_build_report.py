#!/usr/bin/env python
"""Assemble a W&B Report for the held-out cross-generator eval (v11 -> v14b).

Requires `wandb-workspaces`. Reads the held-out numbers from results/ (the same
source as docs/figures), references the `heldout-eval-summary` run for the figure
images, and shows the in-distribution 'trap' metric across the training runs.

Run: WANDB_MODE=online python scripts/wandb_build_report.py
"""
from __future__ import annotations
import json, pathlib
import wandb_workspaces.reports.v2 as wr

ROOT = pathlib.Path(__file__).resolve().parents[1]
R = ROOT / "results"
ENTITY, PROJECT = "klconvergence", "email-fraud-detection"
SCORER = "mahalanobis"
SUMMARY_RUN = "heldout-eval-summary"
CLAUDE = "gen:openrouter:anthropic/claude-3.5-haiku"
GEMINI = "gen:openrouter:google/gemini-2.5-flash"
VERS = ["v11", "v12", "v13", "v14", "v14b"]


def L(p):
    p = R / p
    return json.loads(p.read_text()) if p.exists() else None

def prow(d):
    return next((r for r in d["rows"] if r["scorer"] == SCORER), None) if d and "rows" in d else None

def sauc(d, s):
    return d[s]["AUC"] if d and s in d and "AUC" in d[s] else None

ABL = {"v11": L("v12/heldoutCG_v11lora.json"), "v12": L("v12/heldoutCG_v12lora_final.json"),
       "v13": L("v13/heldoutCG_v13lora.json"), "v14": L("v14/heldoutCG_v14lora.json"),
       "v14b": L("v14/heldoutCG_v14blora.json")}
OOD = {"v11": L("v12/ood_v11lora_heldoutCG.json"), "v12": L("v12/ood_v12lora_final_heldoutCG.json"),
       "v13": L("v13/ood_v13lora_heldoutCG.json"), "v14": L("v14/ood_v14lora_heldoutCG.json"),
       "v14b": L("v14/ood_v14blora_heldoutCG.json")}

def cell(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else "—"

# --- markdown held-out comparison table (exact numbers, always renders) ---
lines = ["| version | pool AUC | TPR@1% | **TPR@5%** | FPR_other@5 (≤0.10) | Claude AUC | Gemini AUC |",
         "|---|---|---|---|---|---|---|"]
for v in VERS:
    r = prow(ABL[v])
    lines.append(
        f"| {'**v14b**' if v=='v14b' else v} | {cell(r and r['auc'])} | {cell(r and r['tpr1'])} | "
        f"{cell(r and r['tpr5'])} | {cell(r and r.get('fpr_other_at_5'))} | "
        f"{cell(sauc(OOD[v], CLAUDE))} | {cell(sauc(OOD[v], GEMINI))} |")
table_md = "\n".join(lines)

report = wr.Report(
    entity=ENTITY, project=PROJECT,
    title="Held-out cross-generator eval (v11 → v14b)",
    width="fluid",
    description=("The metrics that match the task — detect forgeries from generators NEVER seen "
                 "in training, scored against a sender's average-style centroid. All numbers from "
                 f"results/ ({SCORER} scorer)."),
    blocks=[
        wr.H1("Which model is actually best at the task"),
        wr.P("Apples-to-apples: same held-out Claude+Gemini pool (264 genuine / 327 LLM-forgery / "
             "600 wrong-human) for every model. Higher AUC/TPR is better; FPR_other must stay ≤ 0.10."),
        wr.MarkdownBlock(table_md),
        wr.CalloutBlock(
            "v14b is the best model: pool AUC 0.975, catches 88% of forgeries at a 5% false-alarm "
            "budget (6× v11's 14%), and holds the wrong-human guardrail (FPR_other 0.083 ≤ 0.10)."),
        wr.H1("⚠️ Why the training panels mislead (the trap)"),
        wr.P("`tpr_at_fpr/all_1pct` is computed on each run's OWN validation split, whose synthetic "
             "negatives come from THAT run's training generators. v11 (single-generator) scores high "
             "there because it is good at catching its own in-distribution generator — the exact "
             "shortcut that fails on unseen models. It is in-distribution and NOT comparable across "
             "runs. Use the held-out table above instead. The plots below show this metric across the "
             "training runs (v11 looks strong) — contrast it with v11's 0.144 held-out TPR@5%."),
        wr.PanelGrid(
            runsets=[wr.Runset(entity=ENTITY, project=PROJECT, name="training runs (lineage)")],
            panels=[
                wr.LinePlot(title="TRAP: in-distribution val tpr_at_fpr/all_1pct",
                            y=["tpr_at_fpr/all_1pct"], max_runs_to_show=12),
                wr.LinePlot(title="Anti-Goodhart monitor pauc/min_other_synthetic_5pct",
                            y=["pauc/min_other_synthetic_5pct"], max_runs_to_show=12),
            ],
        ),
        wr.H1("Presentation figures"),
        wr.P("Rendered from the held-out results/ numbers (logged to the heldout-eval-summary run)."),
        wr.PanelGrid(
            runsets=[wr.Runset(entity=ENTITY, project=PROJECT, name="eval summary",
                               filters=f'display_name == "{SUMMARY_RUN}"')],
            panels=[wr.MediaBrowser(media_keys=["figure/fig1_cross_generator_auc",
                                                "figure/fig4_confusion_v12_vs_v14b",
                                                "figure/fig3_split_of_effects",
                                                "figure/fig2_pool_progression",
                                                "figure/fig5_guardrail_scorer_choice"],
                                    num_columns=2)],
        ),
    ],
)
report.save()
print("Report URL:", report.url)
