"""Kaplan-Meier curves by tertile of the OUT-OF-FOLD risk score (from 13_cox_analysis.py).
Every score comes from a model that never saw that patient, so the grouping is not circular.
Shows censoring ticks and a number-at-risk table (curves beyond ~7 y rest on very few patients).
Also prints exact KM median survival per group and the site mix per group (a check against site confounding).
Usage: python 14_km_plot.py [model_column]        default: "B image-only"   (also: "D everything+image")"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test, multivariate_logrank_test
from config import OUT_DIR, QC_DIR

XMAX = 7.0
col = sys.argv[1] if len(sys.argv) > 1 else "B image-only"
d = pd.read_csv(OUT_DIR / "oof_cox.csv")
d["grp"] = pd.qcut(d[col], 3, labels=["low risk", "middle", "high risk"])
fmt = lambda p: "p < 0.0001" if p < 1e-4 else f"p = {p:.4f}"
fig, ax = plt.subplots(figsize=(7.2, 6.2))
kmfs = []
for g, color in zip(["low risk", "middle", "high risk"], ["#2a9d8f", "#e9c46a", "#e76f51"]):
    s = d[d["grp"] == g]
    k = KaplanMeierFitter().fit(s["time_y"], s["event"], label=f"{g} (n={len(s)}, deaths={int(s['event'].sum())})")
    k.plot_survival_function(ax=ax, ci_show=True, ci_alpha=0.12, color=color, show_censors=True,
                             censor_styles={"ms": 5, "marker": "|"})
    kmfs.append(k)
lo, hi = d[d["grp"] == "low risk"], d[d["grp"] == "high risk"]
p3 = multivariate_logrank_test(d["time_y"], d["grp"], d["event"]).p_value
p2 = logrank_test(lo["time_y"], hi["time_y"], lo["event"], hi["event"]).p_value
ax.set_xlim(0, XMAX); ax.set_ylim(0, 1.02)
ax.set_xlabel("years from diagnosis"); ax.set_ylabel("overall survival")
ax.set_title(f"{col}: tertiles of out-of-fold score\nlog-rank {fmt(p3)} (3 groups), {fmt(p2)} (low vs high)", fontsize=10)
add_at_risk_counts(*kmfs, ax=ax, rows_to_show=["At risk"])
plt.tight_layout()
out = QC_DIR / f"km_{col.split()[0]}.png"
plt.savefig(out, dpi=150)
print(f"saved {out}\nlog-rank: {fmt(p3)} (3 groups), {fmt(p2)} (low vs high)\n\nKM median survival (years):")
for k, g in zip(kmfs, ["low risk", "middle", "high risk"]):
    print(f"  {g:10s} {k.median_survival_time_:.2f}")
lab = pd.read_csv(OUT_DIR / "label_table.csv")[["patient_id", "site"]]
print("\nsite mix per risk group (rows = site):")
print(pd.crosstab(d.merge(lab, on="patient_id", how="left")["site"], d["grp"]).to_string())
