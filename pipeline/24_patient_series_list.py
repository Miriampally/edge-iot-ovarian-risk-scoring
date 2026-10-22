"""Write the list of patients and DICOM series used (TRIPOD+AI item 18e; needed for reproducibility).
Reads outputs\\step1_conversion_log.csv and config.py. Writes:
  outputs\\patient_series_list.csv   one row per analysed patient: patient_id, site, SeriesInstanceUID, series description, thickness,
                                     TCIA date-shifted study date, slices, whether the series was chosen manually (SERIES_OVERRIDE)
  outputs\\excluded_patients.csv     patients not analysed and why
Patient IDs and series UIDs are public TCGA/TCIA identifiers (no patient images or protected data).
Usage: python 24_patient_series_list.py"""
import pandas as pd
import config
from config import OUT_DIR

log = pd.read_csv(OUT_DIR / "step1_conversion_log.csv")
override = getattr(config, "SERIES_OVERRIDE", {})
exclude = set(getattr(config, "EXCLUDE_PATIENTS", []))
ok = log[(log["status"] == "ok") & (~log["patient_id"].isin(exclude))].copy()
ok["site"] = ok["patient_id"].str[5:7]
ok["manual_series_choice"] = ok["patient_id"].isin(override.keys())
cols = [c for c in ["patient_id", "site", "series_uid", "series_desc", "thickness", "study_date", "n_slices_original", "n_slices",
                    "z_resampled", "manual_series_choice"] if c in ok.columns]
ok[cols].sort_values("patient_id").to_csv(OUT_DIR / "patient_series_list.csv", index=False)
bad = log[(log["status"] != "ok") | (log["patient_id"].isin(exclude))][["patient_id", "status"]].copy()
reason = {"TCGA-61-2097": "no diagnostic CT (bolus-timing series only)", "TCGA-10-0935": "chest-only scan"}
bad["reason"] = bad["patient_id"].map(reason).fillna(bad["status"])
bad.to_csv(OUT_DIR / "excluded_patients.csv", index=False)
print(f"{len(ok)} patients with a selected series ({int(ok['manual_series_choice'].sum())} chosen manually); {len(bad)} excluded")
print(bad.to_string(index=False))
print("saved", OUT_DIR / "patient_series_list.csv")
