"""STEP 4: PyRadiomics features + tumour-burden features from masks in MASK_DIR."""
import logging
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage as ndi
from tqdm import tqdm
from radiomics import featureextractor
from config import *

logging.getLogger("radiomics").setLevel(logging.ERROR)


def main():
    ext = featureextractor.RadiomicsFeatureExtractor(
        binWidth=RADIOMICS_BINWIDTH, label=1, normalize=False, correctMask=True,
        force2D=False, minimumROIDimensions=3, minimumROISize=50)
    ext.disableAllImageTypes()
    ext.enableImageTypeByName("Original")      # keep it small for small cohorts; add "LoG"/"Wavelet" later if needed
    rows, failed = [], []
    voxel_ml = (TARGET_SPACING[0] * TARGET_SPACING[1] * TARGET_SPACING[2]) / 1000.0
    for mp in tqdm(sorted(MASK_DIR.glob("*.nii.gz"))):
        pid = mp.name.replace(".nii.gz", "")
        ip = PREPROC_DIR / mp.name
        if not ip.exists():
            failed.append((pid, "no preprocessed image")); continue
        try:
            img = sitk.ReadImage(str(ip))
            m = sitk.ReadImage(str(mp))
            m = sitk.Resample(m, img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, m.GetPixelID())
            m = sitk.Cast(m > 0, sitk.sitkUInt8)
            marr = sitk.GetArrayFromImage(m)
            if marr.sum() < 50:
                failed.append((pid, "empty mask")); continue
            res = ext.execute(img, m)
            feats = {k: float(v) for k, v in res.items() if not k.startswith("diagnostics_")}
            _, n_comp = ndi.label(marr)
            feats["burden_volume_ml"] = float(marr.sum() * voxel_ml)
            feats["burden_n_lesions"] = int(n_comp)
            rows.append(dict(patient_id=pid, **feats))
        except Exception as e:
            failed.append((pid, str(e)))
    pd.DataFrame(rows).to_csv(FEATURE_CSV, index=False)
    pd.DataFrame(failed, columns=["patient_id", "reason"]).to_csv(OUT_DIR / "step4_failed.csv", index=False)
    print(f"{len(rows)} patients with features, {len(failed)} failed -> {FEATURE_CSV}")


if __name__ == "__main__":
    main()
