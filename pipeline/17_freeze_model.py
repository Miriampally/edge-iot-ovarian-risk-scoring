"""Fit the FINAL portable survival models on all patients and save them, so a collaborator can score new
patients WITHOUT refitting (script 18). Only variables that can travel are used:
  image : mean-pooled EfficientNetB0 embedding -> standardise -> PCA(10) -> standardise -> penalised Cox
  image_clinical : the same + age + FIGO stage
Site/scanner variables are deliberately excluded (they did not transfer across sites).
Saves outputs/frozen_model/model.joblib + meta.json (training score mean/SD and tertile cut-points, software versions,
extraction settings). Usage: python 17_freeze_model.py"""
import json
import sys
import numpy as np
import pandas as pd
import joblib
import lifelines
import sklearn
from lifelines import CoxPHFitter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from config import ROOT, OUT_DIR, SEED

N_PCA, PEN = 10, 0.1
lab = pd.read_csv(OUT_DIR / "label_table.csv")
out = pd.read_csv(OUT_DIR / "outcome_table.csv"); out = out.rename(columns={out.columns[0]: "patient_id"})
clin_path = sorted((ROOT / "TCGA Data set").rglob("clinical.tsv"))[0]
c = pd.read_csv(clin_path, sep="\t", dtype=str).replace(["'--", "--", "Not Reported"], np.nan)
c["patient_id"] = c["cases.submitter_id"].str[:12]
age = pd.Series(np.nan, index=c.index)
if "demographic.age_at_index" in c:
    age = pd.to_numeric(c["demographic.age_at_index"], errors="coerce")
if "diagnoses.age_at_diagnosis" in c:
    age = age.fillna(pd.to_numeric(c["diagnoses.age_at_diagnosis"], errors="coerce") / 365.25)
age = age.groupby(c["patient_id"]).first().rename("age")
df = lab[["patient_id", "stage"]].merge(out[["patient_id", "event", "time_days"]], on="patient_id").merge(age, on="patient_id", how="left")
df = df[(df["time_days"] > 0) & df["event"].notna()]
df = df[df["patient_id"].map(lambda p: (OUT_DIR / "deep" / f"{p}.npz").exists())].reset_index(drop=True)
EMB = np.stack([np.load(OUT_DIR / "deep" / f"{p}.npz")["emb"].astype(np.float32).mean(0) for p in df["patient_id"]])
T, E = (df["time_days"] / 365.25).values, df["event"].astype(int).values
medians = {"age": float(df["age"].median()), "stage": float(df["stage"].median())}
df["age"] = df["age"].fillna(medians["age"]); df["stage"] = df["stage"].fillna(medians["stage"])

s1 = StandardScaler().fit(EMB)
pca = PCA(N_PCA, random_state=SEED).fit(s1.transform(EMB))
PC = pd.DataFrame(pca.transform(s1.transform(EMB)), columns=[f"pc{i}" for i in range(N_PCA)])
models, meta = {}, {}
for name, X in {"image": PC, "image_clinical": pd.concat([PC, df[["age", "stage"]]], axis=1)}.items():
    s2 = StandardScaler().fit(X)
    Z = pd.DataFrame(s2.transform(X), columns=X.columns)
    cph = CoxPHFitter(penalizer=PEN).fit(Z.assign(T=T, E=E), "T", "E")
    lp = cph.predict_log_partial_hazard(Z).values
    models[name] = {"scaler2": s2, "cph": cph, "columns": list(X.columns)}
    meta[name] = {"train_score_mean": float(lp.mean()), "train_score_sd": float(lp.std()),
                  "train_tertile_cuts": [float(q) for q in np.quantile(lp, [1 / 3, 2 / 3])]}
    print(f"{name}: in-sample score SD {lp.std():.3f}; tertile cuts {meta[name]['train_tertile_cuts']}")
(OUT_DIR / "frozen_model").mkdir(exist_ok=True)
joblib.dump({"scaler1": s1, "pca": pca, "models": models, "medians": medians}, OUT_DIR / "frozen_model" / "model.joblib")
meta.update({"train_ids": list(df["patient_id"]), "n_train": int(len(df)), "deaths_train": int(E.sum()), "n_pca": N_PCA, "penalizer": PEN, "pooling": "mean",
             "embedding_model": "keras EfficientNetB0 ImageNet, 224x224, global average pool (1280-d)",
             "extraction": "08_extract_deep_features.py: stride 2, lung block removed, body crop, window -160..240 HU, 1x1x5 mm",
             "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "sklearn": sklearn.__version__, "lifelines": lifelines.__version__},
             "note": "In-sample cut-points/SD; use only to place new patients on the training scale. Exploratory model, not clinically validated."})
json.dump(meta, open(OUT_DIR / "frozen_model" / "meta.json", "w"), indent=2)
print("saved", OUT_DIR / "frozen_model")
