"""Simple body-habitus measurements from the preprocessed CT (no deep learning), on the same abdomen-pelvis
slices used for the image embeddings (idx saved by 08_extract_deep_features.py):
  body_area_cm2, width_cm, depth_cm (left-right and front-back extent), fat_frac (share of body pixels at -150..-30 HU).
Median over slices -> outputs/habitus.csv.   Usage: python 19_habitus_features.py
Limits: no muscle/ascites segmentation (needs a proper segmenter); fat_frac depends a little on contrast phase."""
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage as ndi
from tqdm import tqdm
from config import PREPROC_DIR, OUT_DIR, HU_MIN

rows = []
for f in tqdm(sorted(PREPROC_DIR.glob("*.nii.gz")), desc="patients"):
    pid = f.name[:-7]
    npz = OUT_DIR / "deep" / f"{pid}.npz"
    if not npz.exists():
        continue
    img = sitk.ReadImage(str(f)); a = sitk.GetArrayFromImage(img); sx, sy, _ = img.GetSpacing()
    vals = []
    for k in np.load(npz)["idx"]:
        sl = a[int(k)]
        m = ndi.binary_opening(sl > HU_MIN + 10, iterations=2)
        lab, n = ndi.label(m)
        if n == 0:
            continue
        m = lab == (int(np.argmax(ndi.sum(m, lab, range(1, n + 1)))) + 1)
        body = ndi.binary_fill_holes(m)
        ys, xs = np.where(body)
        fat = ((sl >= -150) & (sl <= -30) & body).sum() / max(body.sum(), 1)
        vals.append((body.sum() * sx * sy / 100.0, (xs.max() - xs.min() + 1) * sx / 10.0, (ys.max() - ys.min() + 1) * sy / 10.0, fat))
    if vals:
        v = np.median(np.array(vals), axis=0)
        rows.append(dict(patient_id=pid, body_area_cm2=v[0], width_cm=v[1], depth_cm=v[2], fat_frac=v[3]))
df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "habitus.csv", index=False)
print(df.describe().round(2).to_string())
print("saved", OUT_DIR / "habitus.csv")
