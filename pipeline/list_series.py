"""List every series of given patients (study date, orientation, size, coverage) to pick one manually.
Usage: python list_series.py TCGA-24-0975 TCGA-24-0980 TCGA-10-0935 TCGA-25-2390"""
import os
import sys
import numpy as np
import SimpleITK as sitk
from config import DICOM_ROOT


def info(f):
    r = sitk.ImageFileReader()
    r.SetFileName(f)
    r.ReadImageInformation()
    return r


def tag(r, k):
    return r.GetMetaData(k).strip() if r.HasMetaDataKey(k) else "-"


for pid in sys.argv[1:]:
    print("=" * 100, "\n", pid)
    rows = []
    for dirpath, _, files in os.walk(DICOM_ROOT / pid):
        if not files:
            continue
        for sid in sitk.ImageSeriesReader.GetGDCMSeriesIDs(dirpath) or []:
            fn = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dirpath, sid)
            r = info(fn[len(fn) // 2])
            try:
                iop = [float(v) for v in tag(r, "0020|0037").split("\\")]
                nrm = np.cross(iop[:3], iop[3:])
                orient = ["sag", "cor", "ax"][int(np.argmax(np.abs(nrm)))]
                p0 = float(np.dot([float(v) for v in tag(info(fn[0]), "0020|0032").split("\\")], nrm))
                p1 = float(np.dot([float(v) for v in tag(info(fn[-1]), "0020|0032").split("\\")], nrm))
                ext = abs(p1 - p0)
            except Exception:
                orient, ext = "?", float("nan")
            rows.append((tag(r, "0008|0020"), tag(r, "0008|0060"), orient, len(fn), tag(r, "0018|0050"),
                         round(ext), tag(r, "0008|103e")[:34], tag(r, "0008|0008")[:22], sid))
    print("study_date  mod orient   n  thick  extent_mm  description / imagetype\n  series_uid")
    for d, m, o, n, t, e, desc, it, sid in sorted(rows):
        print(f"{d} {m:3s} {o:3s} {n:5d} {t:>6s} {e:6d}   {desc!r} {it}\n    {sid}")
