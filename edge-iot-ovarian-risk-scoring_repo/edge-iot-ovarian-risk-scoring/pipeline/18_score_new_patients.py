"""Score NEW patients with the frozen model (no refitting) and, if outcomes are supplied, evaluate it.
Steps for an external cohort: run scripts 01-02 and 08 on their CTs (point config.py at their data), then:
  python 18_score_new_patients.py --deep <folder with <patient>.npz> [--outcomes new_outcomes.csv] [--model image|image_clinical]
outcomes CSV columns: patient_id, time_days, event (1 = died), and for image_clinical also age, stage.
Writes <deep>/../frozen_scores.csv. Never evaluate on the TCGA patients the model was trained on (a warning is printed if they overlap). Reported: C-index with bootstrap 95% CI, hazard ratio per 1 SD of the frozen score,
and Kaplan-Meier tertile log-rank using the TRAINING cut-points."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from lifelines import CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import concordance_index
from config import OUT_DIR

ap = argparse.ArgumentParser()
ap.add_argument("--deep", required=True); ap.add_argument("--outcomes"); ap.add_argument("--model", default="image")
a = ap.parse_args()
bundle = joblib.load(OUT_DIR / "frozen_model" / "model.joblib")
meta = json.load(open(OUT_DIR / "frozen_model" / "meta.json"))
m, mm = bundle["models"][a.model], meta[a.model]
files = sorted(Path(a.deep).glob("*.npz"))
ids = [f.name[:-4] for f in files]
EMB = np.stack([np.load(f)["emb"].astype(np.float32).mean(0) for f in files])
PC = pd.DataFrame(bundle["pca"].transform(bundle["scaler1"].transform(EMB)), columns=[f"pc{i}" for i in range(bundle["pca"].n_components_)])
out = pd.DataFrame({"patient_id": ids})
if a.outcomes:
    out = out.merge(pd.read_csv(a.outcomes), on="patient_id", how="left")
if a.model == "image_clinical":
    for k in ("age", "stage"):
        PC[k] = out[k].fillna(bundle["medians"][k]).values if k in out else bundle["medians"][k]
X = PC[m["columns"]]
Z = pd.DataFrame(m["scaler2"].transform(X), columns=m["columns"])
lp = m["cph"].predict_log_partial_hazard(Z).values
out["score"] = lp
out["score_z"] = (lp - mm["train_score_mean"]) / mm["train_score_sd"]
out["tertile"] = np.digitize(lp, mm["train_tertile_cuts"])
out.to_csv(Path(a.deep).parent / "frozen_scores.csv", index=False)
overlap = len(set(ids) & set(meta.get("train_ids", [])))
if overlap:
    print(f"WARNING: {overlap} of these {len(ids)} patients were in the TRAINING set - their results are in-sample, NOT validation.")
print(f"scored {len(out)} patients with frozen '{a.model}' model (trained on {meta['n_train']} patients, {meta['deaths_train']} deaths)")
if a.outcomes:
    d = out.dropna(subset=["time_days", "event"]); d = d[d["time_days"] > 0]
    T, E = d["time_days"].values / 365.25, d["event"].astype(int).values
    c = concordance_index(T, -d["score"].values, E)
    rng = np.random.default_rng(0)
    b = [concordance_index(T[i], -d["score"].values[i], E[i]) for i in (rng.integers(0, len(T), len(T)) for _ in range(1000)) if E[i].sum() > 0]
    hr = CoxPHFitter().fit(pd.DataFrame({"T": T, "E": E, "s": d["score_z"].values}), "T", "E").summary.loc["s"]
    print(f"n={len(d)}, deaths={E.sum()}\nC-index {c:.3f} (95% CI {np.percentile(b, 2.5):.3f}-{np.percentile(b, 97.5):.3f})   [0.5 = no information]")
    print(f"Hazard ratio per 1 SD of frozen score: {hr['exp(coef)']:.2f} (95% CI {hr['exp(coef) lower 95%']:.2f}-{hr['exp(coef) upper 95%']:.2f}), p = {hr['p']:.4f}")
    if d["tertile"].nunique() > 1:
        print(f"log-rank across training-cut-point tertiles: p = {multivariate_logrank_test(T, d['tertile'], E).p_value:.4f}; group sizes {d['tertile'].value_counts().sort_index().to_dict()}")
    if E.sum() < 50:
        print("WARNING: fewer than ~50 deaths - the interval will be wide; interpret cautiously.")
