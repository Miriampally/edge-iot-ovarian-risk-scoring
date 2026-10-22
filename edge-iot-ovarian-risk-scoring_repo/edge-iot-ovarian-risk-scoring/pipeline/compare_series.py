"""Compare the series chosen in your earlier workflow (CAD\\series_selection.csv, selected==1)
with the series chosen by the new Step 1 (outputs\\step1_conversion_log.csv).
Match rule: same study date AND same number of DICOM files. Writes outputs\\series_comparison.csv."""
import pandas as pd
from config import DICOM_ROOT, OUT_DIR

CAD = DICOM_ROOT.parent / "CAD"
old = pd.read_csv(CAD / "series_selection.csv")
old = old[old["selected"] == 1].drop_duplicates("patient_id")
subset = pd.read_csv(CAD / "working_subset.csv")
new = pd.read_csv(OUT_DIR / "step1_conversion_log.csv")

d = lambda x: str(int(float(x))) if pd.notna(x) else ""
old["old_date"] = old["study_date"].map(d)
old = old.rename(columns={"series_description": "old_desc", "n_files": "old_n"})[["patient_id", "old_date", "old_desc", "old_n"]]
new["new_date"] = new["study_date"].map(d)
new = new.rename(columns={"series_desc": "new_desc", "n_slices_original": "new_n"})
new = new[["patient_id", "status", "new_date", "new_desc", "new_n"]]

m = subset.merge(old, on="patient_id", how="left").merge(new, on="patient_id", how="left")
def verdict(r):
    if pd.isna(r["status"]):
        return "not in new Step 1 log"
    if r["status"] != "ok":
        return f"new pipeline: {r['status']}"
    if pd.isna(r["old_n"]):
        return "no earlier selection"
    return "SAME" if (r["old_date"] == r["new_date"] and int(r["old_n"]) == int(r["new_n"])) else "DIFFERENT"
m["verdict"] = m.apply(verdict, axis=1)
m.to_csv(OUT_DIR / "series_comparison.csv", index=False)
print(m["verdict"].value_counts().to_string(), "\n")
bad = m[m["verdict"] != "SAME"]
if len(bad):
    print(bad[["patient_id", "split", "old_date", "old_desc", "old_n", "new_date", "new_desc", "new_n", "verdict"]].to_string(index=False))
