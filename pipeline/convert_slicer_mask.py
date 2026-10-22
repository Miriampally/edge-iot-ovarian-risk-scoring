"""Convert a mask exported from 3D Slicer (NRRD labelmap) into masks\\<patient>.nii.gz on the Step-1 grid.
Usage:
  python convert_slicer_mask.py TCGA-13-1509
  python convert_slicer_mask.py TCGA-13-1509 "D:\\path\\to\\Segmentation-tumor-label.nrrd"
Default input: masks\\Segmentation-tumor-label.nrrd (what Slicer's Export-to-files created for segment 'tumor')."""
import sys
import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from config import MASK_DIR, NIFTI_DIR

pid = sys.argv[1]
src = sys.argv[2] if len(sys.argv) > 2 else str(MASK_DIR / "Segmentation-tumor-label.nrrd")
img_path = NIFTI_DIR / f"{pid}.nii.gz"
if not img_path.exists():
    raise SystemExit(f"No image {img_path} - check the patient ID.")
try:
    m = sitk.ReadImage(src)
except Exception as e:
    raise SystemExit(f"Cannot read {src}: {e}")
img = sitk.ReadImage(str(img_path))

arr = sitk.GetArrayFromImage(m)
vals = np.unique(arr).tolist()
b = sitk.Cast(m > 0, sitk.sitkUInt8)
if m.GetSize() != img.GetSize():
    print("Mask grid differs from image; resampling mask onto the image grid (nearest neighbour).")
b = sitk.Resample(b, img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
a = sitk.GetArrayFromImage(b)
if not a.any():
    raise SystemExit(f"The mask is EMPTY (label values in file: {vals}). Nothing was drawn/exported - "
                     "draw the tumour in Segment Editor, then export again.")
vox_ml = float(np.prod(img.GetSpacing())) / 1000.0
out = MASK_DIR / f"{pid}.nii.gz"
sitk.WriteImage(b, str(out), useCompression=True)
print(f"saved {out}\n  label values in source: {vals}\n  tumour volume: {a.sum() * vox_ml:.1f} mL"
      f"\n  components: {ndi.label(a)[1]}\n  slices touched: {int(a.any(axis=(1, 2)).sum())} of {a.shape[0]}")
