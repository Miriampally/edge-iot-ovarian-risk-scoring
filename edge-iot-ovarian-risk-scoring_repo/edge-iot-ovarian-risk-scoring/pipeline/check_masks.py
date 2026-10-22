"""Validate tumour masks in MASK_DIR against the Step-1 NIfTI they were drawn on.
Checks: same grid as nifti_raw, labels only {0,1}, not empty, volume (mL), number of components.
Usage: python check_masks.py   -> outputs\\mask_check.csv"""
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage as ndi
from config import MASK_DIR, NIFTI_DIR, OUT_DIR

rows = []
for mp in sorted(MASK_DIR.glob("*.nii.gz")):
    pid = mp.name[:-7]
    ip = NIFTI_DIR / mp.name
    r = dict(patient_id=pid, status="ok", tumor_mL=None, n_components=None, note="")
    try:
        if not ip.exists():
            r.update(status="FAIL", note="no matching image in nifti_raw"); rows.append(r); continue
        m, im = sitk.ReadImage(str(mp)), sitk.ReadImage(str(ip))
        a = sitk.GetArrayFromImage(m)
        notes = []
        if m.GetSize() != im.GetSize():
            notes.append(f"size {m.GetSize()} != image {im.GetSize()}")
        elif (np.abs(np.array(m.GetOrigin()) - np.array(im.GetOrigin())).max() > 1e-2 or
              np.abs(np.array(m.GetSpacing()) - np.array(im.GetSpacing())).max() > 1e-3):
            notes.append("origin/spacing differs from image")
        vals = set(np.unique(a).tolist())
        if not vals <= {0, 1}:
            notes.append(f"labels {sorted(vals)} (expected 0/1)")
        b = a > 0
        if not b.any():
            notes.append("EMPTY mask")
        else:
            vox = float(np.prod(m.GetSpacing())) / 1000.0
            r["tumor_mL"] = round(float(b.sum() * vox), 1)
            r["n_components"] = int(ndi.label(b)[1])
            if r["tumor_mL"] < 5:
                notes.append("very small (<5 mL)")
        if notes:
            r.update(status="CHECK", note="; ".join(notes))
    except Exception as e:
        r.update(status="FAIL", note=str(e))
    rows.append(r)
df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "mask_check.csv", index=False)
print(df.to_string(index=False) if len(df) else "No masks found in " + str(MASK_DIR))
