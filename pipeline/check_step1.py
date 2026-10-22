"""Quick QC summary of the Step-1 log. Usage: python check_step1.py"""
import ast
import pandas as pd
from config import OUT_DIR

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 200)
log = pd.read_csv(OUT_DIR / "step1_conversion_log.csv")
ok = log[log["status"] == "ok"].copy()
print("status counts:\n", log["status"].value_counts().to_string(), "\n")

ok["sp"] = ok["spacing"].map(lambda s: ast.literal_eval(s) if isinstance(s, str) else (None,) * 3)
ok["sz"] = ok["size"].map(lambda s: ast.literal_eval(s) if isinstance(s, str) else (None,) * 3)
ok["inplane_mm"] = ok["sp"].map(lambda t: t[0])
ok["z_mm"] = ok["sp"].map(lambda t: t[2])
ok["z_extent_mm"] = [s[2] * p[2] if s[2] else None for s, p in zip(ok["sz"], ok["sp"])]

print("slice thickness (DICOM tag) counts:\n", ok["thickness"].value_counts().head(12).to_string(), "\n")
print("z spacing after conversion (mm):\n", ok["z_mm"].round(1).value_counts().sort_index().to_string(), "\n")
print("in-plane spacing (mm):\n", ok["inplane_mm"].describe().round(3).to_string(), "\n")
print("z coverage (mm):\n", ok["z_extent_mm"].describe().round(0).to_string(), "\n")

vague = ok["series_desc"].fillna("-").str.strip().str.lower().isin(["-", "= none =", ""]) | \
    ok["series_desc"].fillna("").str.lower().str.contains("contrast/bolus|^with contrast$|^contrast -")
flag = ok[(ok["z_extent_mm"] < 250) | (ok["z_extent_mm"] > 700) | (ok["z_mm"] > 7.6) | vague]
cols = ["patient_id", "series_desc", "thickness", "study_date", "n_slices", "z_mm", "z_extent_mm"]
print(f"{len(flag)} patients to eyeball (short/long coverage, thick slices or uninformative description):")
print(flag[cols].round(1).to_string(index=False))
flag[cols].to_csv(OUT_DIR / "step1_review_list.csv", index=False)
