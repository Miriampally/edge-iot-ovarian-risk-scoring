"""PRE-SPECIFIED analysis: does CT image content add prognostic information (death within 3 years)?
Endpoint: died within 3 years (event with time <= 3 y = 1; alive at 3 y = 0; patients censored before
3 y are excluded because their 3-year status is unknown).
Models (PCA / scaling fitted inside every CV fold, same folds for all):
  A confound-only : site, thickness, coverage, scan year
  K clinical-only : age at diagnosis, FIGO stage
  B image-only    : mean-pooled EfficientNetB0 embedding (nothing else)
  C image+clinical
Key question: C vs K (does the image add anything beyond age and stage?) and B vs A.
Run after: label_check.py, outcome_check.py, 08_extract_deep_features.py.   Usage: python 10_survival_deep_classifier.py"""
import warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from config import ROOT, OUT_DIR, SEED, N_SPLITS

warnings.filterwarnings("ignore")
HORIZON_DAYS = 3 * 365.25
N_PCA, C_IMG, N_REPEATS, N_PERM = 10, 0.1, 10, 100

lab = pd.read_csv(OUT_DIR / "label_table.csv")
out = pd.read_csv(OUT_DIR / "outcome_table.csv").rename(columns={out_col: "patient_id" for out_col in ["Unnamed: 0"]})
if "patient_id" not in out.columns:
    out = out.rename(columns={out.columns[0]: "patient_id"})
clin_path = sorted((ROOT / "TCGA Data set").rglob("clinical.tsv"))[0]
c = pd.read_csv(clin_path, sep="\t", dtype=str).replace(["'--", "--", "Not Reported"], np.nan)
c["patient_id"] = c["cases.submitter_id"].str[:12]
age = pd.Series(np.nan, index=c.index)
if "demographic.age_at_index" in c:
    age = pd.to_numeric(c["demographic.age_at_index"], errors="coerce")
if "diagnoses.age_at_diagnosis" in c:
    age = age.fillna(pd.to_numeric(c["diagnoses.age_at_diagnosis"], errors="coerce") / 365.25)
age = age.groupby(c["patient_id"]).first().rename("age")

df = lab.merge(out[["patient_id", "event", "time_days"]], on="patient_id").merge(age, on="patient_id", how="left")
died = (df["event"] == 1) & (df["time_days"] <= HORIZON_DAYS)
alive = df["time_days"] >= HORIZON_DAYS
df = df[died | alive].copy()
df["y"] = (died[df.index]).astype(int)
emb = {}
for pid in df["patient_id"]:
    p = OUT_DIR / "deep" / f"{pid}.npz"
    if p.exists():
        emb[pid] = np.load(p)["emb"].astype(np.float32).mean(0)
E = pd.DataFrame.from_dict(emb, orient="index").add_prefix("img_")
df = df.merge(E, left_on="patient_id", right_index=True).reset_index(drop=True)
y = df["y"].values
print(f"n={len(y)} patients with known 3-year status and embeddings; died within 3 y: {y.sum()} ({y.mean():.0%}); age missing: {int(df['age'].isna().sum())}")

img_cols = [c_ for c_ in df.columns if c_.startswith("img_")]
num_conf, num_clin = ["thickness_mm", "coverage_mm", "scan_year"], ["age", "stage"]
X = df[img_cols + ["site"] + num_conf + num_clin].copy()
for col in num_conf + num_clin:
    X[col] = X[col].fillna(X[col].median())
conf_tf = [("site", OneHotEncoder(handle_unknown="ignore"), ["site"]), ("cnum", StandardScaler(), num_conf)]
clin_tf = [("clin", StandardScaler(), num_clin)]
img_tf = [("img", make_pipeline(StandardScaler(), PCA(N_PCA, random_state=SEED)), img_cols)]
specs = {"A confound-only": (conf_tf, 0.5), "K clinical-only": (clin_tf, 0.5),
         "B image-only": (img_tf, C_IMG), "C image+clinical": (img_tf + clin_tf, C_IMG)}
def build(name):
    tf, C = specs[name]
    return Pipeline([("pre", ColumnTransformer(tf)), ("clf", LogisticRegression(max_iter=2000, C=C, class_weight="balanced"))])
n_split = int(min(N_SPLITS, np.bincount(y).min()))
def oof(name, yy, seed):
    o = np.zeros(len(yy))
    for tr, te in StratifiedKFold(n_split, shuffle=True, random_state=seed).split(X, yy):
        o[te] = build(name).fit(X.iloc[tr], yy[tr]).predict_proba(X.iloc[te])[:, 1]
    return o

rng = np.random.default_rng(SEED)
print(f"\n{'model':18s} {'AUC mean+/-sd':16s} {'95% CI (bootstrap)':20s} {'avg precision':14s} (prevalence {y.mean():.2f})")
pooled = {}
for name in specs:
    aucs, aps, preds = [], [], []
    for r in range(N_REPEATS):
        o = oof(name, y, SEED + r); preds.append(o)
        aucs.append(roc_auc_score(y, o)); aps.append(average_precision_score(y, o))
    pooled[name] = np.mean(preds, 0)
    b = [roc_auc_score(y[i], pooled[name][i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(1000)) if len(np.unique(y[i])) == 2]
    print(f"{name:18s} {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}   [{np.percentile(b, 2.5):.3f}, {np.percentile(b, 97.5):.3f}]       {np.mean(aps):.3f}")

d = []
for _ in range(1000):
    i = rng.integers(0, len(y), len(y))
    if len(np.unique(y[i])) == 2:
        d.append(roc_auc_score(y[i], pooled["C image+clinical"][i]) - roc_auc_score(y[i], pooled["K clinical-only"][i]))
print(f"\nAUC gain of image on top of clinical (C - K): {np.mean(d):+.3f}  95% CI [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]")
obs = roc_auc_score(y, oof("B image-only", y, SEED))
null = [roc_auc_score(yp, oof("B image-only", yp, SEED)) for yp in (rng.permutation(y) for _ in range(N_PERM))]
print(f"Model B permutation test ({N_PERM} shuffles): observed AUC {obs:.3f}, p = {(1 + sum(n >= obs for n in null)) / (N_PERM + 1):.3f}")
pd.DataFrame({"patient_id": df["patient_id"], "y": y, **pooled}).to_csv(OUT_DIR / "oof_survival_3y.csv", index=False)
