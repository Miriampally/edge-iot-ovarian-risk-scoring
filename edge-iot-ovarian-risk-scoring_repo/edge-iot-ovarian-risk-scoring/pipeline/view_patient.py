"""Larger QC views for chosen patients.
Usage: python view_patient.py TCGA-13-1509 [...]
Writes qc_png/view_<id>.png         (coronal + sagittal projections + 8 axial slices, whole scan)
       qc_png/pelvis_<id>.png       (12 axial slices through the lower ~55 % of the scan, newest first = top)
Each axial panel is labelled with S = the slice position in mm, the same number 3D Slicer shows as 'S:'."""
import sys
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import PREPROC_DIR, QC_DIR, HU_MIN, HU_MAX

for pid in sys.argv[1:]:
    img = sitk.ReadImage(str(PREPROC_DIR / f"{pid}.nii.gz"))
    a = sitk.GetArrayFromImage(img)                       # z, y, x
    sx, sy, sz = img.GetSpacing()
    n = a.shape[0]
    S = lambda k: img.TransformIndexToPhysicalPoint((0, 0, int(k)))[2]
    ascending = S(n - 1) > S(0)                           # True if index 0 is the most inferior slice
    flip = (lambda v: v[::-1]) if ascending else (lambda v: v)   # put feet at the bottom of projections

    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(2, 6, width_ratios=[1.3, 1.3, 1, 1, 1, 1])
    ax = fig.add_subplot(gs[:, 0]); ax.imshow(flip(a.mean(1)), cmap="gray", vmin=HU_MIN, vmax=HU_MAX, aspect=sz / sx)
    ax.set_title("coronal"); ax.axis("off")
    ax = fig.add_subplot(gs[:, 1]); ax.imshow(flip(a.mean(2)), cmap="gray", vmin=HU_MIN, vmax=HU_MAX, aspect=sz / sy)
    ax.set_title("sagittal"); ax.axis("off")
    zs = np.linspace(2, n - 3, 8).astype(int)
    zs = zs[::-1] if ascending else zs                    # top of the figure = most superior
    for k, z in enumerate(zs):
        ax = fig.add_subplot(gs[k // 4, 2 + k % 4])
        ax.imshow(a[z], cmap="gray", vmin=HU_MIN, vmax=HU_MAX); ax.set_title(f"S = {S(z):.0f} mm", fontsize=9); ax.axis("off")
    fig.suptitle(f"{pid}   {n * sz:.0f} mm coverage", fontsize=11)
    plt.tight_layout(); plt.savefig(QC_DIR / f"view_{pid}.png", dpi=75); plt.close(fig)

    lo, hi = (0, int(0.55 * n)) if ascending else (int(0.45 * n), n - 1)
    zs = np.linspace(lo, hi, 12).astype(int)
    zs = zs[::-1] if ascending else zs                    # superior first
    fig, axes = plt.subplots(3, 4, figsize=(15, 11))
    for ax, z in zip(axes.ravel(), zs):
        ax.imshow(a[z], cmap="gray", vmin=HU_MIN, vmax=HU_MAX); ax.set_title(f"S = {S(z):.0f} mm", fontsize=10); ax.axis("off")
    fig.suptitle(f"{pid}: lower scan, axial slices from top-left (higher) to bottom-right (lower)", fontsize=11)
    plt.tight_layout(); plt.savefig(QC_DIR / f"pelvis_{pid}.png", dpi=70); plt.close(fig)
    print("saved", QC_DIR / f"view_{pid}.png", "and", QC_DIR / f"pelvis_{pid}.png")
