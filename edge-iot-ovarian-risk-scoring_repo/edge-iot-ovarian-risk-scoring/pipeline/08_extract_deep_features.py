"""STEP B (mask-free): EfficientNetB0 slice embeddings for each patient's abdomen-pelvis slices.

For every preprocessed CT (nifti_preproc):
  * per axial slice: body mask (largest component, holes filled); a slice is 'lung-like' if its interior
    air fraction is high AND the air forms >= 2 separate large regions (left + right lung; one big midline
    air collection such as free abdominal gas is NOT lung). The chest block = everything from the top of
    the scan down to the end of the first lung run (only if it starts in the upper 35 % of the scan);
    check outputs/deep_summary.csv;
  * every SLICE_STRIDE-th remaining slice is cropped to the body box, padded square, resized to 224
    and passed through ImageNet EfficientNetB0 (global-average-pooled, 1280-d);
  * saved to outputs/deep/<patient>.npz (embeddings + slice positions). Finished patients are skipped.
Usage:  python 08_extract_deep_features.py          (all)
        python 08_extract_deep_features.py 3        (first 3 patients, test)
Set environment variable EFFNET_WEIGHTS=none to skip the ImageNet download (testing only)."""
import os
import sys
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import pandas as pd
import SimpleITK as sitk
import tensorflow as tf
from scipy import ndimage as ndi
from tqdm import tqdm
from config import PREPROC_DIR, OUT_DIR, HU_MIN, HU_MAX

DEEP_DIR = OUT_DIR / "deep"
DEEP_DIR.mkdir(parents=True, exist_ok=True)
SLICE_STRIDE = 2            # every 2nd slice (= 10 mm at 5 mm spacing)
LUNG_FRACTION = 0.15        # interior air / body area above this (and 2 big air regions) => lung-like slice
BIG_REGION = 0.03           # an air region counts as 'big' if it covers >= 3 % of the body area
MAX_GAP = 2                 # lung run may be interrupted by up to this many slices
LUNG_START_MAX = 0.35       # first lung slice must lie within the top 35 % of the scan
MIN_KEEP = 8                # if fewer slices survive, fall back to all slices
IMG = 224
WEIGHTS = None if os.environ.get("EFFNET_WEIGHTS", "imagenet").lower() == "none" else "imagenet"


def analyse_slice(sl):
    """Return (body_mask, air_fraction, n_big_air_regions, bbox) for one HU-clipped slice."""
    m = ndi.binary_opening(sl > HU_MIN + 10, iterations=2)
    lab, n = ndi.label(m)
    if n == 0:
        return None, 0.0, 0, None
    sizes = ndi.sum(m, lab, range(1, n + 1))
    m = lab == (int(np.argmax(sizes)) + 1)
    filled = ndi.binary_fill_holes(m)
    area = filled.sum()
    airmask = filled & (sl <= HU_MIN + 10)
    alab, an = ndi.label(airmask)
    n_big = int((np.array(ndi.sum(airmask, alab, range(1, an + 1))) >= BIG_REGION * area).sum()) if an else 0
    ys, xs = np.where(filled)
    return filled, float(airmask.sum()) / max(area, 1), n_big, (ys.min(), ys.max(), xs.min(), xs.max(), area)


def chest_block(lung_like, z_mm):
    """Boolean array: True for slices from the top of the scan to the end of the first lung run."""
    n = len(lung_like)
    order = np.argsort(-z_mm)                      # superior -> inferior
    seq = lung_like[order]
    idx = np.where(seq)[0]
    chest = np.zeros(n, bool)
    if len(idx) == 0 or idx[0] > LUNG_START_MAX * n:
        return chest
    end = idx[0]
    for i in idx[1:]:
        if i - end - 1 <= MAX_GAP:
            end = i
        else:
            break
    chest[order[:end + 1]] = True
    return chest


def prepare(sl, bbox):
    y0, y1, x0, x1, _ = bbox
    pad = 6
    crop = sl[max(y0 - pad, 0):y1 + pad + 1, max(x0 - pad, 0):x1 + pad + 1]
    h, w = crop.shape
    s = max(h, w)
    sq = np.full((s, s), HU_MIN, np.float32)
    sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = crop
    sq = (np.clip(sq, HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN) * 255.0
    return tf.image.resize(sq[..., None], (IMG, IMG)).numpy().repeat(3, axis=-1)


def main():
    model = tf.keras.applications.EfficientNetB0(include_top=False, weights=WEIGHTS, pooling="avg",
                                                  input_shape=(IMG, IMG, 3))
    files = sorted(PREPROC_DIR.glob("*.nii.gz"))
    if len(sys.argv) > 1:
        files = files[:int(sys.argv[1])]
    summary = []
    for f in tqdm(files, desc="patients"):
        pid = f.name[:-7]
        out = DEEP_DIR / f"{pid}.npz"
        img = sitk.ReadImage(str(f))
        a = sitk.GetArrayFromImage(img)
        n = a.shape[0]
        info = [analyse_slice(a[k]) for k in range(n)]
        z_mm = np.array([img.TransformIndexToPhysicalPoint((0, 0, int(k)))[2] for k in range(n)])
        ok = np.array([i[0] is not None for i in info])
        lung_like = np.array([i[0] is not None and i[1] > LUNG_FRACTION and i[2] >= 2 for i in info])
        chest = chest_block(lung_like, z_mm)
        keep = [k for k in range(n) if ok[k] and not chest[k]]
        fallback = len(keep) < MIN_KEEP
        if fallback:
            keep = [k for k in range(n) if ok[k]]
        keep = keep[::SLICE_STRIDE]
        summary.append(dict(patient_id=pid, n_slices=n, n_chest_dropped=int((chest & ok).sum()),
                            n_embedded=len(keep), kept_z_min=round(float(z_mm[keep].min()), 1),
                            kept_z_max=round(float(z_mm[keep].max()), 1), fallback_all_slices=fallback))
        if out.exists():
            continue
        batch = np.stack([prepare(a[k], info[k][3]) for k in keep]).astype(np.float32)
        emb = model.predict(batch, batch_size=32, verbose=0)
        np.savez_compressed(out, emb=emb.astype(np.float16), z_mm=z_mm[keep], idx=np.array(keep))
    pd.DataFrame(summary).to_csv(OUT_DIR / "deep_summary.csv", index=False)
    print("Done. Embeddings in", DEEP_DIR, "| summary:", OUT_DIR / "deep_summary.csv")


if __name__ == "__main__":
    main()
