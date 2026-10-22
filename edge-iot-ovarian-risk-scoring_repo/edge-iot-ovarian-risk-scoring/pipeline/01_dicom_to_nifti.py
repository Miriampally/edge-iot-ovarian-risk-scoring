"""STEP 1: TCIA DICOM folders -> one axial CT NIfTI per patient.

Usage:
  python 01_dicom_to_nifti.py        # all patients
  python 01_dicom_to_nifti.py 3      # first 3 patients only (test run)

Series selection (per patient):
  1. candidates = CT series with >= MIN_SLICES slices whose slice normal is axial
     (excludes coronal/sagittal MPR reformats), not scout/topogram/localizer/MIP/smart-prep;
  2. use the EARLIEST study date that has a candidate (staging should come from the
     baseline scan - verify study_date in the log, it is not guaranteed pre-treatment);
  3. within that study prefer ORIGINAL images, then thickness closest to 5 mm, then most slices.

Optional manual control in config.py: EXCLUDE_PATIENTS = [...], SERIES_OVERRIDE = {patient: SeriesInstanceUID}.
(Delete that patient's NIfTI in nifti_raw first so it is converted again.)

Slice handling (SimpleITK assumes uniform spacing):
  - duplicate / near-duplicate slices are removed (n_removed in the log);
  - if remaining gaps differ by more than IRREGULAR_TOL_MM, the stack is linearly
    interpolated along z onto a uniform grid using the TRUE slice positions (z_resampled).
"""
import os
import re
import sys
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.interpolate import interp1d
from tqdm import tqdm
from config import *

SERIES_OVERRIDE = globals().get("SERIES_OVERRIDE", {})   # optional, set in config.py: {"TCGA-xx-xxxx": "<SeriesInstanceUID>"}
EXCLUDE_PATIENTS = globals().get("EXCLUDE_PATIENTS", [])  # optional, set in config.py: ["TCGA-xx-xxxx", ...]
BAD_DESC = re.compile(r"scout|topogram|localizer|smart ?prep|\bmip\b|survey", re.I)
IRREGULAR_TOL_MM = 0.5


def read_info(f):
    r = sitk.ImageFileReader()
    r.SetFileName(f)
    r.ReadImageInformation()
    return r


def tag(r, key):
    return r.GetMetaData(key).strip() if r.HasMetaDataKey(key) else "-"


def normal_from(r):
    iop = [float(v) for v in tag(r, "0020|0037").split("\\")]
    return np.cross(iop[:3], iop[3:])


def candidate_series(patient_dir):
    cands = []
    for dirpath, _, files in os.walk(patient_dir):
        if not files:
            continue
        for sid in sitk.ImageSeriesReader.GetGDCMSeriesIDs(dirpath) or []:
            fn = list(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dirpath, sid))
            if len(fn) < MIN_SLICES:
                continue
            r = read_info(fn[len(fn) // 2])
            desc, itype = tag(r, "0008|103e"), tag(r, "0008|0008")
            if tag(r, "0008|0060") != "CT" or BAD_DESC.search(desc) or "LOCALIZER" in itype.upper():
                continue
            try:
                if abs(normal_from(r)[2]) < 0.9:      # not axial
                    continue
            except Exception:
                continue
            cands.append(dict(sid=sid, fnames=fn, desc=desc, itype=itype, thickness=tag(r, "0018|0050"),
                              study_date=tag(r, "0008|0020"), original="ORIGINAL" in itype.upper()))
    return cands


def choose(cands):
    if not cands:
        return None, 0
    date = lambda c: c["study_date"] if c["study_date"] != "-" else "99999999"
    dates = sorted({date(c) for c in cands})
    pool = [c for c in cands if date(c) == dates[0]]

    def key(c):
        try:
            t = float(c["thickness"])
        except ValueError:
            t = 99.0
        return (not c["original"], abs(t - 5.0), -len(c["fnames"]))
    return min(pool, key=key), len(dates)


def thin_slices(fnames):
    """Drop duplicate / near-duplicate slices; return kept files, #removed, gap stats, positions."""
    normal = normal_from(read_info(fnames[len(fnames) // 2]))
    z = np.array([np.dot([float(v) for v in tag(read_info(f), "0020|0032").split("\\")], normal)
                  for f in fnames])
    order = np.argsort(z, kind="stable")
    med = float(np.median(np.diff(z[order])))
    if med < 0.1:
        raise ValueError("degenerate slice positions (median gap ~0)")
    keep = [order[0]]
    for i in order[1:]:
        if z[i] - z[keep[-1]] >= 0.5 * med:
            keep.append(i)
    zk = z[keep]
    gaps = np.diff(zk)
    medk = float(np.median(gaps))
    stats = dict(median_gap_mm=round(medk, 3), max_gap_mm=round(float(gaps.max()), 3),
                 min_gap_mm=round(float(gaps.min()), 3),
                 irregular=bool(np.abs(gaps - medk).max() > IRREGULAR_TOL_MM))
    return [fnames[i] for i in keep], len(fnames) - len(keep), stats, zk


def make_uniform(img, zk):
    """Resample along z onto a uniform grid at the median true spacing."""
    arr = sitk.GetArrayFromImage(img)
    med = float(np.median(np.diff(zk)))
    n = int(np.floor((zk[-1] - zk[0]) / med + 1e-6)) + 1
    zu = zk[0] + med * np.arange(n)
    f = interp1d(zk, arr.astype(np.float32), axis=0, kind="linear", assume_sorted=True,
                 bounds_error=False, fill_value=(arr[0].astype(np.float32), arr[-1].astype(np.float32)))
    new = sitk.GetImageFromArray(np.rint(f(zu)).astype(arr.dtype))
    sx, sy, _ = img.GetSpacing()
    new.SetSpacing((sx, sy, med))
    new.SetOrigin(img.GetOrigin())
    new.SetDirection(img.GetDirection())
    return new


def main():
    patients = sorted(p for p in DICOM_ROOT.iterdir() if p.is_dir() and p.name.startswith("TCGA-"))
    if len(sys.argv) > 1:
        patients = patients[:int(sys.argv[1])]
    log = []
    for p in tqdm(patients, desc="patients"):
        out = NIFTI_DIR / f"{p.name}.nii.gz"
        try:
            if p.name in EXCLUDE_PATIENTS:
                log.append(dict(patient_id=p.name, status="excluded (manual)"))
                continue
            cands = candidate_series(str(p))
            sel, n_studies = choose(cands)
            if p.name in SERIES_OVERRIDE:
                match = [c for c in cands if c["sid"] == SERIES_OVERRIDE[p.name]]
                if not match:
                    raise ValueError("override series is not an axial CT candidate")
                sel = match[0]
            if sel is None:
                log.append(dict(patient_id=p.name, status="no axial CT series"))
                continue
            row = dict(patient_id=p.name, status="ok", series_uid=sel["sid"], series_desc=sel["desc"],
                       thickness=sel["thickness"], study_date=sel["study_date"], n_studies=n_studies,
                       original=sel["original"], n_slices_original=len(sel["fnames"]))
            fnames, n_removed, stats, zk = thin_slices(sel["fnames"])
            row.update(n_slices=len(fnames), n_removed=n_removed, **stats)
            if not out.exists():
                reader = sitk.ImageSeriesReader()
                reader.SetFileNames(fnames)
                img = reader.Execute()
                if stats["irregular"]:
                    img = make_uniform(img, zk)
                sitk.WriteImage(img, str(out), useCompression=True)
                row.update(spacing=str(img.GetSpacing()), size=str(img.GetSize()))
            else:                                   # already converted: read its header for the log
                hdr = sitk.ImageFileReader()
                hdr.SetFileName(str(out))
                hdr.ReadImageInformation()
                row.update(spacing=str(hdr.GetSpacing()), size=str(hdr.GetSize()))
            row["z_resampled"] = stats["irregular"]
            log.append(row)
        except Exception as e:
            log.append(dict(patient_id=p.name, status=f"error: {e}"))
    pd.DataFrame(log).to_csv(OUT_DIR / "step1_conversion_log.csv", index=False)
    print("Done. See", OUT_DIR / "step1_conversion_log.csv")


if __name__ == "__main__":
    main()
