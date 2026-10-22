"""Is the image-based risk score informative WITHIN sites (not just between sites)?
Uses the out-of-fold scores saved by 13_cox_analysis.py (outputs/oof_cox.csv).
 1. C-index of the score inside each site group (bootstrap 95% CI)
 2. Site-stratified Cox model on the score: hazard ratio per 1 SD of score, with a separate baseline hazard per site
    (so differences in overall survival between sites cannot explain the result).
Sites with fewer than MIN_SITE patients are pooled as 'other'.   Usage: python 15_within_site.py"""
import warnings
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from config import OUT_DIR, SEED

warnings.filterwarnings("ignore")
MIN_SITE, N_BOOT = 15, 1000
d = pd.read_csv(OUT_DIR / "oof_cox.csv").merge(pd.read_csv(OUT_DIR / "label_table.csv", dtype={"site": str})[["patient_id", "site"]], on="patient_id", how="left")
d["site_grp"] = d["site"].where(d["site"].map(d["site"].value_counts()) >= MIN_SITE, "other")
rng = np.random.default_rng(SEED)
for col in ("B image-only", "D everything+image"):
    print("=" * 78, f"\n{col}  (score = out-of-fold risk, higher = worse)\n")
    print(f"{'site':8s} {'n':>4s} {'deaths':>6s}   C-index [95% CI]")
    for g, s in d.groupby("site_grp"):
        s = s.reset_index(drop=True)
        c = concordance_index(s["time_y"], -s[col], s["event"])
        b = []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(s), len(s))
            if s["event"].values[i].sum() > 0:
                b.append(concordance_index(s["time_y"].values[i], -s[col].values[i], s["event"].values[i]))
        print(f"{str(g):8s} {len(s):4d} {int(s['event'].sum()):6d}   {c:.3f} [{np.percentile(b, 2.5):.3f}, {np.percentile(b, 97.5):.3f}]")
    m = pd.DataFrame({"T": d["time_y"], "E": d["event"], "score": (d[col] - d[col].mean()) / d[col].std(), "site": d["site_grp"]})
    cph = CoxPHFitter().fit(m, "T", "E", strata=["site"])
    r = cph.summary.loc["score"]
    print(f"\nSite-stratified Cox: HR per 1 SD of score = {r['exp(coef)']:.2f} "
          f"(95% CI {r['exp(coef) lower 95%']:.2f}-{r['exp(coef) upper 95%']:.2f}), p = {r['p']:.4f}")
print("\nNote: model D already contains site dummies, so its score mixes between-site and within-site information; "
      "model B (image only) is the cleaner one to interpret here.")
