"""STEP 2: resample to common spacing + clip to soft-tissue HU window + QC picture."""
import SimpleITK as sitk
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import *


def resample(img, spacing, interp=sitk.sitkLinear, default=-1024):
    new_size = [int(round(sz * sp / nsp)) for sz, sp, nsp in zip(img.GetSize(), img.GetSpacing(), spacing)]
    f = sitk.ResampleImageFilter()
    f.SetOutputSpacing(spacing)
    f.SetSize(new_size)
    f.SetOutputDirection(img.GetDirection())
    f.SetOutputOrigin(img.GetOrigin())
    f.SetTransform(sitk.Transform())
    f.SetDefaultPixelValue(default)
    f.SetInterpolator(interp)
    return f.Execute(img)


def main():
    for f in tqdm(sorted(NIFTI_DIR.glob("*.nii.gz"))):
        pid = f.name.replace(".nii.gz", "")
        out = PREPROC_DIR / f.name
        if out.exists():
            continue
        img = sitk.Cast(sitk.ReadImage(str(f)), sitk.sitkFloat32)
        img = resample(img, TARGET_SPACING)
        img = sitk.Clamp(img, sitk.sitkFloat32, HU_MIN, HU_MAX)
        sitk.WriteImage(img, str(out), useCompression=True)

        arr = sitk.GetArrayFromImage(img)
        fig, ax = plt.subplots(1, 3, figsize=(10, 3.4))
        for a, z in zip(ax, [arr.shape[0] // 4, arr.shape[0] // 2, 3 * arr.shape[0] // 4]):
            a.imshow(arr[z], cmap="gray", vmin=HU_MIN, vmax=HU_MAX)
            a.set_title(f"{pid} z={z}", fontsize=8)
            a.axis("off")
        plt.tight_layout()
        plt.savefig(QC_DIR / f"{pid}_preproc.png", dpi=80)
        plt.close(fig)
    print("Done. Check a few PNGs in", QC_DIR)


if __name__ == "__main__":
    main()
