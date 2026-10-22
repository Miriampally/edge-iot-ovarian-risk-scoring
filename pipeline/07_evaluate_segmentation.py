"""STEP 6: Dice / IoU / HD95 of predicted masks vs reference masks.
Usage: python 07_evaluate_segmentation.py <pred_dir> <ref_dir>"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import binary_erosion, distance_transform_edt
from config import OUT_DIR


def hd95(a, b, sp):
    if not a.any() or not b.any():
        return np.nan
    sa, sb = a ^ binary_erosion(a), b ^ binary_erosion(b)
    da, db = distance_transform_edt(~sa, sampling=sp), distance_transform_edt(~sb, sampling=sp)
    return float(np.percentile(np.concatenate([db[sa], da[sb]]), 95))


def main(pred_dir, ref_dir):
    rows = []
    for f in sorted(Path(pred_dir).glob("*.nii.gz")):
        r = Path(ref_dir) / f.name
        if not r.exists():
            continue
        p, g = sitk.ReadImage(str(f)), sitk.ReadImage(str(r))
        g = sitk.Resample(g, p, sitk.Transform(), sitk.sitkNearestNeighbor, 0, g.GetPixelID())
        pa, ga = sitk.GetArrayFromImage(p) > 0, sitk.GetArrayFromImage(g) > 0
        inter = (pa & ga).sum()
        rows.append(dict(patient_id=f.name[:-7], dice=2 * inter / max(pa.sum() + ga.sum(), 1),
                         iou=inter / max((pa | ga).sum(), 1), hd95_mm=hd95(pa, ga, p.GetSpacing()[::-1])))
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "segmentation_metrics.csv", index=False)
    print(df.describe().round(3))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
