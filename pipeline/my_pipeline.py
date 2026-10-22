#!/usr/bin/env python3
"""
Per-patient edge pipeline for timing (wrapper around YOUR existing scripts).

For ONE patient's DICOM folder it runs, in memory (nothing is written to disk):
    1. dicom_to_volume  series selection + slice thinning + volume read   (01_dicom_to_nifti.py)
    2. preprocess       resample to TARGET_SPACING + HU clip              (02_preprocess.py)
    3. embed            lung-block removal, body crop, EfficientNetB0     (08_extract_deep_features.py)
    4. score            frozen scaler -> PCA -> Cox score                 (17_freeze_model.py output)

It imports and calls the functions in your own scripts, so it times exactly your code.
Your scripts are not modified.

SETUP
  Keep this file in the same folder as config.py and the risk-score scripts (in this repository: pipeline/).
  Run it with the Python environment that has TensorFlow, SimpleITK, scipy, joblib, lifelines and
  scikit-learn installed (see requirements.txt).

USAGE
  Per-stage breakdown for one patient (runs the pipeline twice: cold, then warm):
      python my_pipeline.py "<path to one patient's DICOM folder>"

  Plug into measure_inputs.py (repository root) to get mean inference time over several patients:
      python measure_inputs.py --skip-network --prefix TCGA- --study-dir "<TCGA-OV folder>" \\
          --pick-sizes 30,112,500,1500 --runs 3 --model "pipeline/my_pipeline.py:run_study"

Notes
  * Stage 4 is skipped (and said so) if you have not run 17_freeze_model.py yet; it takes
    milliseconds, so the total is barely affected.
  * Reading DICOM from disk is part of stage 1. A real edge node receives the study over the
    clinic LAN, so keep this in mind when you describe the timing.
  * The first call includes TensorFlow graph set-up; measure_inputs.py excludes it.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

FILES_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(FILES_DIR))            # so "from config import ..." works
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import pandas as pd
import SimpleITK as sitk


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, FILES_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                # type: ignore[union-attr]
    return mod


cfg = _load("pipeline_config", "config.py")
s01 = _load("s01_dicom", "01_dicom_to_nifti.py")
s02 = _load("s02_preproc", "02_preprocess.py")
s08 = _load("s08_deep", "08_extract_deep_features.py")

_MODEL = None
_FROZEN = None


def _effnet():
    global _MODEL
    if _MODEL is None:
        import tensorflow as tf
        _MODEL = tf.keras.applications.EfficientNetB0(
            include_top=False, weights=s08.WEIGHTS, pooling="avg",
            input_shape=(s08.IMG, s08.IMG, 3))
    return _MODEL


def _frozen():
    """Frozen survival model from 17_freeze_model.py, or False if it does not exist yet."""
    global _FROZEN
    if _FROZEN is None:
        d = cfg.OUT_DIR / "frozen_model"
        if (d / "model.joblib").exists() and (d / "meta.json").exists():
            _FROZEN = (joblib.load(d / "model.joblib"), json.load(open(d / "meta.json")))
        else:
            _FROZEN = False
    return _FROZEN


# ---------------------------------------------------------------- stages
def dicom_to_volume(patient_dir: Path):
    pid = patient_dir.name
    cands = s01.candidate_series(str(patient_dir))
    sel, _ = s01.choose(cands)
    override = getattr(cfg, "SERIES_OVERRIDE", {})
    if pid in override:
        match = [c for c in cands if c["sid"] == override[pid]]
        if not match:
            raise ValueError(f"{pid}: override series is not an axial CT candidate")
        sel = match[0]
    if sel is None:
        raise ValueError(f"{pid}: no axial CT series")
    fnames, _, stats, zk = s01.thin_slices(sel["fnames"])
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(fnames)
    img = reader.Execute()
    if stats["irregular"]:
        img = s01.make_uniform(img, zk)
    return img


def preprocess(img):
    img = sitk.Cast(img, sitk.sitkFloat32)
    img = s02.resample(img, cfg.TARGET_SPACING)
    return sitk.Clamp(img, sitk.sitkFloat32, cfg.HU_MIN, cfg.HU_MAX)


def embed(img):
    a = sitk.GetArrayFromImage(img)
    n = a.shape[0]
    info = [s08.analyse_slice(a[k]) for k in range(n)]
    z_mm = np.array([img.TransformIndexToPhysicalPoint((0, 0, int(k)))[2] for k in range(n)])
    ok = np.array([i[0] is not None for i in info])
    lung_like = np.array([i[0] is not None and i[1] > s08.LUNG_FRACTION and i[2] >= 2
                          for i in info])
    chest = s08.chest_block(lung_like, z_mm)
    keep = [k for k in range(n) if ok[k] and not chest[k]]
    if len(keep) < s08.MIN_KEEP:
        keep = [k for k in range(n) if ok[k]]
    keep = keep[::s08.SLICE_STRIDE]
    batch = np.stack([s08.prepare(a[k], info[k][3]) for k in keep]).astype(np.float32)
    emb = _effnet().predict(batch, batch_size=32, verbose=0)
    return emb.astype(np.float16), len(keep)      # float16, as saved by script 08


def score(emb):
    fb = _frozen()
    if not fb:
        return None
    bundle, meta = fb
    m, mm = bundle["models"]["image"], meta["image"]
    e = emb.astype(np.float32).mean(0, keepdims=True)
    pc = bundle["pca"].transform(bundle["scaler1"].transform(e))
    PC = pd.DataFrame(pc, columns=[f"pc{i}" for i in range(pc.shape[1])])
    Z = pd.DataFrame(m["scaler2"].transform(PC[m["columns"]]), columns=m["columns"])
    lp = float(m["cph"].predict_log_partial_hazard(Z).values[0])
    return {"score": lp, "score_z": (lp - mm["train_score_mean"]) / mm["train_score_sd"]}


# ---------------------------------------------------------------- entry point
def run_study(study_path: str) -> dict:
    """Whole per-patient pipeline. Returns the score and a per-stage time breakdown (s)."""
    t, p = {}, Path(study_path)
    t0 = time.perf_counter()
    img = dicom_to_volume(p);            t["dicom_to_volume"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    img = preprocess(img);               t["preprocess"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    emb, n_emb = embed(img);             t["embed"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    res = score(emb);                    t["score"] = time.perf_counter() - t0
    t["total"] = sum(t.values())
    return {"patient": p.name, "slices_embedded": n_emb, "result": res,
            "timings_s": {k: round(v, 2) for k, v in t.items()}}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("patient_dir", help="one patient's DICOM folder, e.g. ...\\TCGA-OV\\TCGA-04-1331")
    a = ap.parse_args()
    for label in ("cold (includes TensorFlow set-up)", "warm (use this one)"):
        out = run_study(a.patient_dir)
        print(f"\n[{label}] {out['patient']}: {out['slices_embedded']} slices embedded")
        for k, v in out["timings_s"].items():
            print(f"  {k:<16}{v:>8.2f} s")
        if out["result"] is None:
            print("  (score stage skipped: run 17_freeze_model.py to create the frozen model)")
        else:
            print(f"  frozen score {out['result']['score']:.3f}  (z {out['result']['score_z']:.2f})")
