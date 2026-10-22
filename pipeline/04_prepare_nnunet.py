"""STEP 3b helper: package images+masks for nnU-Net v2 (train on Colab/Kaggle GPU), then restore names.

  python 04_prepare_nnunet.py export                 -> zip to upload
  python 04_prepare_nnunet.py restore <pred_folder>  -> copy predictions to MASK_DIR with patient ids
"""
import json, shutil, sys
import pandas as pd
import SimpleITK as sitk
from config import *

DATASET = "Dataset501_OvarianMass"


def export():
    base = NNUNET_EXPORT / DATASET
    for s in ("imagesTr", "labelsTr", "imagesTs"):
        (base / s).mkdir(parents=True, exist_ok=True)
    rows, n_train = [], 0
    images = sorted(NIFTI_DIR.glob("*.nii.gz"))
    for i, f in enumerate(images, 1):
        pid = f.name.replace(".nii.gz", "")
        case = f"OV_{i:03d}"
        mask = MASK_DIR / f.name
        if mask.exists():
            m = sitk.ReadImage(str(mask))
            im = sitk.ReadImage(str(f))
            assert m.GetSize() == im.GetSize(), f"{pid}: mask/image size differ - draw masks on the Step-1 NIfTI"
            m = sitk.Cast(m > 0, sitk.sitkUInt8)
            sitk.WriteImage(m, str(base / "labelsTr" / f"{case}.nii.gz"), useCompression=True)
            shutil.copy(f, base / "imagesTr" / f"{case}_0000.nii.gz")
            n_train += 1
            split = "train"
        else:
            shutil.copy(f, base / "imagesTs" / f"{case}_0000.nii.gz")
            split = "predict"
        rows.append(dict(patient_id=pid, case_id=case, split=split))
    json.dump({"channel_names": {"0": "CT"}, "labels": {"background": 0, "mass": 1},
               "numTraining": n_train, "file_ending": ".nii.gz"},
              open(base / "dataset.json", "w"), indent=2)
    pd.DataFrame(rows).to_csv(OUT_DIR / "nnunet_case_map.csv", index=False)
    shutil.make_archive(str(NNUNET_EXPORT / DATASET), "zip", NNUNET_EXPORT, DATASET)
    print(f"Training cases: {n_train}. Upload {NNUNET_EXPORT / (DATASET + '.zip')} to Google Drive.")


def restore(pred_dir):
    cmap = pd.read_csv(OUT_DIR / "nnunet_case_map.csv").set_index("case_id")["patient_id"]
    for f in Path(pred_dir).glob("*.nii.gz"):
        case = f.name.replace(".nii.gz", "")
        if case in cmap.index:
            shutil.copy(f, MASK_DIR / f"{cmap[case]}.nii.gz")
    print("Predictions copied into", MASK_DIR)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "export":
        export()
    elif len(sys.argv) >= 3 and sys.argv[1] == "restore":
        restore(sys.argv[2])
    else:
        print(__doc__)
