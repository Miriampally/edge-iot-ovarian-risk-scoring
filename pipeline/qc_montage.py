"""QC contact sheets: one coronal 'mean projection' thumbnail per patient (like a plain X-ray),
so chest-only / legs-only / partial scans stand out at a glance.
Usage: python qc_montage.py        -> qc_png/montage_page1.png, montage_page2.png, ..."""
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import PREPROC_DIR, QC_DIR, HU_MIN, HU_MAX

COLS, ROWS = 6, 3
files = sorted(PREPROC_DIR.glob("*.nii.gz"))
per_page = COLS * ROWS
for page in range(0, len(files), per_page):
    fig, axes = plt.subplots(ROWS, COLS, figsize=(COLS * 2.4, ROWS * 3.6))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, f in zip(axes.ravel(), files[page:page + per_page]):
        img = sitk.ReadImage(str(f))
        arr = sitk.GetArrayFromImage(img)                 # z, y, x
        proj = arr.mean(axis=1)[::-1]                     # coronal mean projection, feet at bottom if z ascends
        sx, _, sz = img.GetSpacing()
        ax.imshow(proj, cmap="gray", vmin=HU_MIN, vmax=HU_MAX, aspect=sz / sx)
        ax.set_title(f"{f.name[:-7]}\n{arr.shape[0] * sz:.0f} mm", fontsize=7)
    plt.tight_layout()
    out = QC_DIR / f"montage_page{page // per_page + 1}.png"
    plt.savefig(out, dpi=70)
    plt.close(fig)
    print("saved", out)
