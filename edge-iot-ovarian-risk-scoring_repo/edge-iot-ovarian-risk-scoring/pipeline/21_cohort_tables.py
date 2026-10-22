"""Cohort-characteristics table (TRIPOD+AI items 20b/20c) and missing-data table (item 11).
Reads outputs\\ label_table.csv, outcome_table.csv, dicom_meta.csv, habitus.csv, step1_conversion_log.csv and the GDC clinical file.
Writes outputs\\table_cohort_characteristics.csv and outputs\\table_missingness.csv and prints both.
Usage: python 21_cohort_tables.py"""
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from config import ROOT, OUT_DIR

MIN_SITE = 15
log = pd.read_csv(OUT_DIR / "step1_conversion_log.csv")
log = log[log["status"] == "ok"].set_index("patient_id")
lab = pd.read_csv(OUT_DIR / "label_table.csv", dtype={"site": str}).set_index("patient_id")
out = pd.read_csv(OUT_DIR / "outcome_table.csv"); out = out.rename(columns={out.columns[0]: "patient_id"}).set_index("patient_id")
meta = pd.read_csv(OUT_DIR / "dicom_meta.csv").set_index("patient_id")
hab = pd.read_csv(OUT_DIR / "habitus.csv").set_index("patient_id")
clin_path = sorted((ROOT / "TCGA Data set").rglob("clinical.tsv"))[0]
c = pd.read_csv(clin_path, sep="\t", dtype=str).replace(["'--", "--", "Not Reported"], np.nan)
c["patient_id"] = c["cases.submitter_id"].str[:12]
age = pd.Series(np.nan, index=c.index)
if "demographic.age_at_index" in c:
    age = pd.to_numeric(c["demographic.age_at_index"], errors="coerce")
if "diagnoses.age_at_diagnosis" in c:
    age = age.fillna(pd.to_numeric(c["diagnoses.age_at_diagnosis"], errors="coerce") / 365.25)
age = age.groupby(c["patient_id"]).first()

d = pd.DataFrame(index=log.index)
d["site"] = d.index.str[5:7]
d["site_grp"] = d["site"].where(d["site"].map(d["site"].value_counts()) >= MIN_SITE, "other")
d["age"] = age.reindex(d.index)
d["stage"] = lab["stage"].reindex(d.index)
d["event"] = out["event"].reindex(d.index)
d["time_y"] = (out["time_days"] / 365.25).reindex(d.index)
d.loc[d["time_y"] <= 0, ["time_y", "event"]] = np.nan
import ast
def _lit(x):
    try: return ast.literal_eval(x)
    except Exception: return (np.nan, np.nan, np.nan)
if {"thickness", "spacing", "size", "study_date"} <= set(log.columns):      # take acquisition variables from the Step-1 log (all 141 patients)
    sp, sz = log["spacing"].map(_lit).reindex(d.index), log["size"].map(_lit).reindex(d.index)
    d["thickness_mm"] = pd.to_numeric(log["thickness"], errors="coerce").reindex(d.index)
    d["coverage_mm"] = [a[2] * b[2] for a, b in zip(sz, sp)]
    d["scan_year"] = (pd.to_numeric(log["study_date"], errors="coerce") // 10000).reindex(d.index)
else:
    for k in ("thickness_mm", "coverage_mm", "scan_year"):
        d[k] = lab[k].reindex(d.index)
mf = meta["manufacturer"].reindex(d.index).astype("string").str.upper()
d["manufacturer"] = np.where(mf.isna() | (mf.str.strip() == ""), None,
                             np.where(mf.str.contains("GE", na=False), "GE", np.where(mf.str.contains("SIEMENS", na=False), "Siemens", "Other")))
d["kvp"] = pd.to_numeric(meta["kvp"].reindex(d.index), errors="coerce")
d["contrast"] = meta["contrast_flag"].reindex(d.index).astype(str).str.lower().map({"true": 1, "false": 0})
for k in ("body_area_cm2", "width_cm", "depth_cm", "fat_frac"):
    d[k] = hab[k].reindex(d.index)
d["z_resampled"] = log["z_resampled"].reindex(d.index).astype(str).str.lower().eq("true").astype(float) if "z_resampled" in log else np.nan
d["dup_removed"] = (pd.to_numeric(log["n_removed"], errors="coerce").reindex(d.index) > 0).astype(float) if "n_removed" in log else np.nan

def med(s, q=True, nd=1):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty: return "-"
    lo, hi = (s.quantile(.25), s.quantile(.75)) if q else (s.min(), s.max())
    return f"{s.median():.{nd}f} [{lo:.{nd}f}-{hi:.{nd}f}]"
def cnt(mask, n):
    k = int(mask.sum()); return f"{k} ({100 * k / max(n, 1):.0f}%)"
def revkm(sub):
    s = sub.dropna(subset=["time_y", "event"])
    if len(s) < 5: return "-"
    m = KaplanMeierFitter().fit(s['time_y'], 1 - s['event']).median_survival_time_
    return f"{m:.1f}" if np.isfinite(m) else "not estimable"

rows = [("Patients, n", lambda s: str(len(s))),
        ("Age at diagnosis, years: median [IQR]", lambda s: med(s["age"])),
        ("FIGO stage I", lambda s: cnt(s["stage"] == 1, len(s))), ("FIGO stage II", lambda s: cnt(s["stage"] == 2, len(s))),
        ("FIGO stage III", lambda s: cnt(s["stage"] == 3, len(s))), ("FIGO stage IV", lambda s: cnt(s["stage"] == 4, len(s))),
        ("FIGO stage missing", lambda s: cnt(s["stage"].isna(), len(s))),
        ("Deaths, n (% of patients with survival data)", lambda s: cnt(s["event"] == 1, int(s["event"].notna().sum()))),
        ("Observed time, years: median [IQR]", lambda s: med(s["time_y"], nd=2)),
        ("Median follow-up, years (reverse Kaplan-Meier)", revkm),
        ("Manufacturer: GE", lambda s: cnt(s["manufacturer"] == "GE", len(s))), ("Manufacturer: Siemens", lambda s: cnt(s["manufacturer"] == "Siemens", len(s))),
        ("Manufacturer: other", lambda s: cnt(s["manufacturer"] == "Other", len(s))),
        ("Tube voltage 120 kVp", lambda s: cnt(s["kvp"].round() == 120, len(s))),
        ("Contrast-agent field present", lambda s: cnt(s["contrast"] == 1, len(s))),
        ("Slice thickness (DICOM), mm: median [IQR]", lambda s: med(s["thickness_mm"])),
        ("Slice thickness 5 mm", lambda s: cnt(s["thickness_mm"].round(1) == 5.0, len(s))),
        ("Z-coverage, mm: median [IQR]", lambda s: med(s["coverage_mm"], nd=0)),
        ("Scan year (TCIA date-shifted): median [range]", lambda s: med(s["scan_year"], q=False, nd=0)),
        ("Body area, cm2: median [IQR]", lambda s: med(s["body_area_cm2"], nd=0)), ("Body width, cm: median [IQR]", lambda s: med(s["width_cm"])),
        ("Body depth, cm: median [IQR]", lambda s: med(s["depth_cm"])), ("Fat fraction: median [IQR]", lambda s: med(s["fat_frac"], nd=2)),
        ("Volumes interpolated for irregular spacing", lambda s: cnt(s["z_resampled"] == 1, len(s))),
        ("Volumes with duplicate slices removed", lambda s: cnt(s["dup_removed"] == 1, len(s)))]
groups = [("Overall", d)] + [(f"Site {g}", d[d["site_grp"] == g]) for g in sorted(d["site_grp"].unique())]
t1 = pd.DataFrame({name: [f(sub) for _, f in rows] for name, sub in groups}, index=[r[0] for r in rows])
t1.to_csv(OUT_DIR / "table_cohort_characteristics.csv")

miss_vars = {"Age": "age", "FIGO stage": "stage", "Vital status": "event", "Follow-up time": "time_y", "Tissue-source site": "site",
             "Slice thickness": "thickness_mm", "Z-coverage": "coverage_mm", "Scan year": "scan_year", "Manufacturer": "manufacturer",
             "Tube voltage (kVp)": "kvp", "Contrast field": "contrast", "Body area": "body_area_cm2", "Body width": "width_cm",
             "Body depth": "depth_cm", "Fat fraction": "fat_frac"}
tm = pd.DataFrame({"Variable": list(miss_vars), "Missing, n": [int(d[v].isna().sum()) for v in miss_vars.values()]})
tm["Missing, %"] = (100 * tm["Missing, n"] / len(d)).round(1)
an = d.dropna(subset=["stage", "event", "time_y"])      # patients with FIGO stage AND survival data = the survival-analysis cohort
tm[f"Missing among analysed (n={len(an)})"] = [int(an[v].isna().sum()) for v in miss_vars.values()]
handling = {"age": "median imputation (whole cohort) in survival models", "stage": "excluded from FIGO analyses; median-imputed as a predictor in survival models",
            "event": "excluded from survival analyses", "time_y": "excluded from survival analyses", "site": "taken from the patient ID (never missing)",
            "manufacturer": "coded as other/missing category", "contrast": "coded as absent"}
tm["Handling"] = [handling.get(v, "median imputation (whole cohort)") for v in miss_vars.values()]
tm.to_csv(OUT_DIR / "table_missingness.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
print(t1.to_string()); print(); print(tm.to_string(index=False))
print(f"\nSurvival-analysis cohort (FIGO stage and survival data known): n = {len(an)}; excluded from all stage/survival analyses: {len(d) - len(an)}")
print("\nsaved", OUT_DIR / "table_cohort_characteristics.csv", "and", OUT_DIR / "table_missingness.csv")
