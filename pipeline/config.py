"""Central settings. Edit ONLY this file when paths/params change."""
from pathlib import Path

import os

# Folder that holds your data and receives all outputs. Set OVARIAN_DATA_ROOT, or edit this line.
ROOT = Path(os.environ.get("OVARIAN_DATA_ROOT", "data"))
PROJECT = ROOT / "ovarian_seg_pipeline_data"

# ---- INPUTS ----
DICOM_ROOT = Path(os.environ.get("TCGA_OV_DICOM_ROOT", str(ROOT / "TCGA Data set" / "TCGA-OV")))  # TCIA TCGA-OV download
CLINICAL_FILE = ROOT / "TCGA Data set" / "clinical.tsv"      # not needed until Step 5; will be updated then

# ---- OUTPUTS (created automatically) ----
NIFTI_DIR = PROJECT / "nifti_raw"            # step 1: original-resolution CT, HU preserved (nnU-Net wants THIS)
PREPROC_DIR = PROJECT / "nifti_preproc"      # step 2: resampled + HU-clipped (used for radiomics / baseline)
MASK_DIR = PROJECT / "masks"                 # final tumour masks, <patient_id>.nii.gz, label 1 (manual or nnU-Net)
BASELINE_MASK_DIR = PROJECT / "masks_baseline_fcm"
QC_DIR = PROJECT / "qc_png"
NNUNET_EXPORT = PROJECT / "nnunet_export"
OUT_DIR = PROJECT / "outputs"

# ---- Preprocessing ----
TARGET_SPACING = (1.0, 1.0, 5.0)   # mm (x, y, z); native slices are ~5 mm so z is not interpolated
HU_MIN, HU_MAX = -160, 240         # soft-tissue window, W400/L40
MIN_SLICES = 30                    # ignore scout/localizer series

# ---- Baseline FCM segmentation ----
FCM_CLUSTERS = 4
SOFT_TISSUE_HU = (0, 110)          # clusters with centre in this range are 'candidate'
MIN_COMPONENT_ML = 20
KEEP_LARGEST_N = 5
ROI_Z_FRACTION = (0.0, 0.55)       # fraction of z-extent measured from the feet (pelvis/lower abdomen); None = whole scan

# ---- Radiomics ----
RADIOMICS_BINWIDTH = 25
FEATURE_CSV = OUT_DIR / "radiomics_features.csv"

# ---- Classification ----
# "early_vs_advanced" (I-II vs III-IV) | "III_vs_IV" | "multiclass" (I-II / III / IV)
TASK = "III_vs_IV"
TOP_K_FEATURES = 20
CORR_THRESHOLD = 0.95
N_REPEATS = 5
N_SPLITS = 5
SEED = 42

for _d in (NIFTI_DIR, PREPROC_DIR, MASK_DIR, BASELINE_MASK_DIR, QC_DIR, NNUNET_EXPORT, OUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

EXCLUDE_PATIENTS = ["TCGA-61-2097"]
SERIES_OVERRIDE = {
    "TCGA-24-0975": "1.3.6.1.4.1.14519.5.2.1.7777.4007.233005020700722095334542706028",
    "TCGA-24-0980": "1.3.6.1.4.1.14519.5.2.1.7777.4007.578116533087316558618548986265",
    "TCGA-25-2390": "1.3.6.1.4.1.14519.5.2.1.3344.4007.166411449204560167879431443730",
}
EXCLUDE_PATIENTS = EXCLUDE_PATIENTS + ["TCGA-10-0935"]
