"""STEP 3a: classical Fuzzy C-Means BASELINE (for your ablation table / rough pseudo-masks).

WARNING: intensity-only. It will also catch bowel, uterus, muscle etc. Do NOT report it as
tumour segmentation without manual verification. Real masks come from Step 3b (3D Slicer +
nnU-Net)."""
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from tqdm import tqdm
from config import *


def fcm_1d(x, c=4, m=2.0, iters=60, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.sort(rng.choice(x, c, replace=False)).astype(np.float64)
    for _ in range(iters):
        d = np.abs(x[:, None] - centers[None, :]) + 1e-6
        inv = d ** (-2.0 / (m - 1))
        u = inv / inv.sum(1, keepdims=True)
        um = u ** m
        new = (um * x[:, None]).sum(0) / um.sum(0)
        done = np.abs(new - centers).max() < 1e-3
        centers = new
        if done:
            break
    return np.sort(centers)


def body_mask(arr):
    body = arr > (HU_MIN + 10)
    body = ndi.binary_opening(body, iterations=2)
    lab, n = ndi.label(body)
    if n == 0:
        return body
    sizes = ndi.sum(body, lab, range(1, n + 1))
    body = lab == (np.argmax(sizes) + 1)
    for z in range(body.shape[0]):
        body[z] = ndi.binary_fill_holes(body[z])
    return ndi.binary_erosion(body, iterations=3)


def z_limits(img, n):
    if ROI_Z_FRACTION is None:
        return 0, n
    z0 = img.TransformIndexToPhysicalPoint((0, 0, 0))[2]
    z1 = img.TransformIndexToPhysicalPoint((0, 0, n - 1))[2]
    lo, hi = ROI_Z_FRACTION
    if z0 < z1:      # index 0 is inferior
        return int(lo * n), int(hi * n)
    return int((1 - hi) * n), int((1 - lo) * n)


def main():
    voxel_ml = (TARGET_SPACING[0] * TARGET_SPACING[1] * TARGET_SPACING[2]) / 1000.0
    for f in tqdm(sorted(PREPROC_DIR.glob("*.nii.gz"))):
        pid = f.name.replace(".nii.gz", "")
        out = BASELINE_MASK_DIR / f.name
        if out.exists():
            continue
        img = sitk.ReadImage(str(f))
        arr = sitk.GetArrayFromImage(img)
        body = body_mask(arr)
        z_lo, z_hi = z_limits(img, arr.shape[0])
        roi = np.zeros_like(body)
        roi[z_lo:z_hi] = body[z_lo:z_hi]

        vals = arr[roi]
        if vals.size < 1000:
            continue
        sample = np.random.default_rng(SEED).choice(vals, min(200000, vals.size), replace=False)
        centers = fcm_1d(sample, c=FCM_CLUSTERS)
        label = np.abs(arr[..., None] - centers[None, None, None, :]).argmin(-1)
        good = [i for i, c in enumerate(centers) if SOFT_TISSUE_HU[0] <= c <= SOFT_TISSUE_HU[1]]
        cand = np.isin(label, good) & roi
        cand = ndi.binary_opening(cand, iterations=2)

        lab, n = ndi.label(cand)
        keep = np.zeros_like(cand)
        if n > 0:
            sizes = np.array(ndi.sum(cand, lab, range(1, n + 1))) * voxel_ml
            order = np.argsort(sizes)[::-1][:KEEP_LARGEST_N]
            for i in order:
                if sizes[i] >= MIN_COMPONENT_ML:
                    keep |= lab == (i + 1)

        m = sitk.GetImageFromArray(keep.astype(np.uint8))
        m.CopyInformation(img)
        sitk.WriteImage(m, str(out), useCompression=True)

        if keep.any():
            z = int(np.argmax(keep.sum((1, 2))))
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(arr[z], cmap="gray", vmin=HU_MIN, vmax=HU_MAX)
            ax.contour(keep[z], colors="r", linewidths=0.8)
            ax.set_title(f"{pid} z={z} centers={np.round(centers).astype(int).tolist()}", fontsize=7)
            ax.axis("off")
            plt.savefig(QC_DIR / f"{pid}_fcm_overlay.png", dpi=80)
            plt.close(fig)
    print("Done. Inspect overlays in", QC_DIR)


if __name__ == "__main__":
    main()
