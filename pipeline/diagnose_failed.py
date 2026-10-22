"""Diagnose patients whose Step-1 conversion failed.
Usage: python diagnose_failed.py TCGA-13-1409 TCGA-13-1509 TCGA-25-1626 TCGA-61-2097
Prints one line per series, then position/duplicate details for the largest CT series."""
import os
import sys
from collections import Counter
import numpy as np
import SimpleITK as sitk
from config import *


def tag(r, key):
    return r.GetMetaData(key).strip() if r.HasMetaDataKey(key) else "-"


def info(f):
    r = sitk.ImageFileReader()
    r.SetFileName(f)
    r.ReadImageInformation()
    return r


for pid in sys.argv[1:]:
    print("=" * 70, "\n", pid)
    best = None
    for dirpath, _, files in os.walk(DICOM_ROOT / pid):
        if not files:
            continue
        for sid in sitk.ImageSeriesReader.GetGDCMSeriesIDs(dirpath) or []:
            fn = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dirpath, sid)
            r = info(fn[0])
            print(f"  series n={len(fn):4d} modality={tag(r,'0008|0060'):3s} "
                  f"desc={tag(r,'0008|103e')[:30]!r} thick={tag(r,'0018|0050')} "
                  f"folder={os.path.basename(dirpath)[:40]!r}")
            if tag(r, "0008|0060") == "CT" and (best is None or len(fn) > len(best)):
                best = fn
    if best is None:
        print("  -> no CT series found")
        continue
    z, acq, atime, itype = [], Counter(), Counter(), Counter()
    first = info(best[0])
    iop = [float(v) for v in tag(first, "0020|0037").split("\\")]
    normal = np.cross(iop[:3], iop[3:])
    for f in best:
        r = info(f)
        z.append(float(np.dot([float(v) for v in tag(r, "0020|0032").split("\\")], normal)))
        acq[tag(r, "0020|0012")] += 1
        atime[tag(r, "0008|0032")[:6]] += 1
        itype[tag(r, "0008|0008")] += 1
    z = np.round(np.array(z), 2)
    per_pos = Counter(Counter(z).values())
    print(f"  largest CT series: {len(best)} files, {len(set(z))} unique positions")
    print("  files-per-position histogram (files_per_position: n_positions):", dict(per_pos))
    print("  AcquisitionNumber values:", dict(acq))
    print("  distinct AcquisitionTime values:", len(atime))
    print("  ImageType values:", dict(itype))
