"""Read acquisition metadata (manufacturer, model, kernel, kVp, tube current, contrast, ...) from the DICOM
header of the series chosen in Step 1 for every patient -> outputs/dicom_meta.csv.
Usage: python 12_dicom_metadata.py        (a few minutes: it re-scans the DICOM folders)"""
import os
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
from config import DICOM_ROOT, OUT_DIR

TAGS = {"manufacturer": "0008|0070", "model": "0008|1090", "kernel": "0018|1210", "kvp": "0018|0060",
        "tube_current_mA": "0018|1151", "contrast_agent": "0018|0010", "slice_thickness": "0018|0050",
        "pixel_spacing": "0028|0030", "series_desc": "0008|103e", "study_date": "0008|0020"}

log = pd.read_csv(OUT_DIR / "step1_conversion_log.csv")
log = log[log["status"] == "ok"]
rows = []
for pid, sid in tqdm(list(zip(log["patient_id"], log["series_uid"])), desc="patients"):
    row = dict(patient_id=pid)
    try:
        files = None
        for dirpath, _, fs in os.walk(DICOM_ROOT / pid):
            if fs and sid in (sitk.ImageSeriesReader.GetGDCMSeriesIDs(dirpath) or []):
                files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dirpath, sid)
                break
        if files is None:
            row["note"] = "series not found"
        else:
            r = sitk.ImageFileReader()
            r.SetFileName(files[len(files) // 2])
            r.ReadImageInformation()
            for k, t in TAGS.items():
                row[k] = r.GetMetaData(t).strip() if r.HasMetaDataKey(t) else None
    except Exception as e:
        row["note"] = str(e)
    rows.append(row)
df = pd.DataFrame(rows)
df["contrast_flag"] = df["contrast_agent"].notna() & (df["contrast_agent"].fillna("").str.len() > 0)
df.to_csv(OUT_DIR / "dicom_meta.csv", index=False)
print("saved", OUT_DIR / "dicom_meta.csv")
for col in ("manufacturer", "kernel", "kvp", "contrast_flag"):
    print(f"\n{col}:\n", df[col].value_counts(dropna=False).head(8).to_string())
