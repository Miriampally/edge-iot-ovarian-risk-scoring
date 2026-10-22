"""How many CT-cohort patients have usable survival / residual-disease / recurrence data?
Usage: python outcome_check.py            (auto-finds clinical.tsv like label_check.py)
       python outcome_check.py "D:\\path\\clinical.tsv" """
import sys
import numpy as np
import pandas as pd
from config import ROOT, OUT_DIR

path = sys.argv[1] if len(sys.argv) > 1 else str(sorted((ROOT / "TCGA Data set").rglob("clinical.tsv"))[0])
print("clinical file:", path)
c = pd.read_csv(path, sep="\t", dtype=str)
MISSING = {"'--", "--", "", "nan", "Not Reported", "Unknown", "Not Allowed To Collect", "not reported", "unknown"}
c = c.replace(list(MISSING), np.nan)
c["pid"] = c["cases.submitter_id"].str[:12]
ct = pd.read_csv(OUT_DIR / "step1_conversion_log.csv")
cohort = set(ct.loc[ct["status"] == "ok", "patient_id"])
c = c[c["pid"].isin(cohort)]
print(f"CT cohort patients: {len(cohort)}; with a clinical row: {c['pid'].nunique()}\n")

def first(col):
    return c.dropna(subset=[col]).groupby("pid")[col].first() if col in c else pd.Series(dtype=object)
def mx(col):
    return pd.to_numeric(c[col], errors="coerce").groupby(c["pid"]).max() if col in c else pd.Series(dtype=float)

vital, d_death, d_fu = first("demographic.vital_status"), mx("demographic.days_to_death"), mx("diagnoses.days_to_last_follow_up")
t = pd.DataFrame({"vital": vital, "d_death": d_death, "d_fu": d_fu}).reindex(sorted(cohort))
t["event"] = (t["vital"] == "Dead").astype(float).where(t["vital"].notna())
t["time_days"] = np.where(t["event"] == 1, t["d_death"], t["d_fu"])
ok = t.dropna(subset=["event", "time_days"])
ok = ok[ok["time_days"] > 0]
print("OVERALL SURVIVAL")
print(f"  usable (vital status + time): {len(ok)} of {len(cohort)}")
print(f"  deaths: {int(ok['event'].sum())}   censored: {int((ok['event'] == 0).sum())}")
print(f"  median follow-up (days): {ok['time_days'].median():.0f}   (years: {ok['time_days'].median() / 365.25:.1f})")
for yrs in (1, 2, 3):
    d = yrs * 365.25
    known = ((ok["event"] == 1) & (ok["time_days"] <= d)) | (ok["time_days"] >= d)
    dead = ((ok["event"] == 1) & (ok["time_days"] <= d)).sum()
    print(f"  {yrs}-year status known for {int(known.sum())}; died within {yrs} yr: {int(dead)}")

res = first("diagnoses.residual_disease")
res2 = first("treatments.residual_disease")
r = res.combine_first(res2).reindex(sorted(cohort))
print("\nRESIDUAL DISEASE (diagnoses.residual_disease, else treatments.residual_disease)")
print(f"  patients with a value: {int(r.notna().sum())} of {len(cohort)}")
print(r.value_counts().to_string())

prog = first("diagnoses.progression_or_recurrence").reindex(sorted(cohort))
print("\nPROGRESSION / RECURRENCE")
print(f"  patients with a value: {int(prog.notna().sum())}"); print(prog.value_counts().to_string())
t.assign(residual=r, progression=prog).to_csv(OUT_DIR / "outcome_table.csv")
