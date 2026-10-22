"""STEP C (mask-free): patient-level stage classification from pooled slice embeddings, compared with a
confound-only model under identical repeated stratified CV.

Models (all: PCA / feature selection fitted INSIDE each fold):
  A  confound-only : site, slice thickness, coverage, scan year
  B  image-only    : mean-pooled EfficientNetB0 embedding -> standardise -> PCA -> logistic regression
  C  image+confound
Reports AUC (mean +/- sd over repeats), average precision, a bootstrap 95% CI of the pooled out-of-fold AUC,
and a label-permutation p-value for model B.
Usage: python 09_train_deep_classifier.py [III_vs_IV|early_vs_advanced]     (default III_vs_IV)"""
import sys
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
from config import OUT_DIR, SEED, N_SPLITS

warnings.filterwarnings("ignore")
TASK = sys.argv[1] if len(sys.argv) > 1 else "III_vs_IV"
POOL = "mean"          # "mean" or "max" over slices
N_PCA = 10
C_IMG = 0.1
N_REPEATS = 10
N_PERM = 100

lab = pd.read_csv(OUT_DIR / "label_table.csv")
rows = []
for pid in lab["patient_id"]:
    p = OUT_DIR / "deep" / f"{pid}.npz"
    if p.exists():
        e = np.load(p)["emb"].astype(np.float32)
        rows.append((pid, e.mean(0) if POOL == "mean" else e.max(0)))
emb = pd.DataFrame([r[1] for r in rows], index=[r[0] for r in rows]).add_prefix("img_")
df = lab.merge(emb, left_on="patient_id", right_index=True)
if TASK == "early_vs_advanced":
    y = (df["stage"] >= 3).astype(int).values
else:
    df = df[df["stage"] >= 3].reset_index(drop=True); y = (df["stage"] == 4).astype(int).values
print(f"Task {TASK}: n={len(y)} patients with embeddings, class counts={np.bincount(y).tolist()}")
img_cols = [c for c in df.columns if c.startswith("img_")]
conf_cols = ["site", "thickness_mm", "coverage_mm", "scan_year"]
X = df[img_cols + conf_cols].copy()
for c in ["thickness_mm", "coverage_mm", "scan_year"]:
    X[c] = X[c].fillna(X[c].median())

conf_tf = [("cat", OneHotEncoder(handle_unknown="ignore"), ["site"]),
           ("num", StandardScaler(), ["thickness_mm", "coverage_mm", "scan_year"])]
img_tf = ("img", make_pipeline(StandardScaler(), PCA(N_PCA, random_state=SEED)), img_cols)
models = {
    "A confound-only": ColumnTransformer(conf_tf),
    "B image-only": ColumnTransformer([img_tf]),
    "C image+confound": ColumnTransformer([img_tf] + conf_tf),
}
def build(name, C):
    return Pipeline([("pre", models[name]), ("clf", LogisticRegression(max_iter=2000, C=C, class_weight="balanced"))])
Cs = {"A confound-only": 0.5, "B image-only": C_IMG, "C image+confound": C_IMG}

n_split = int(min(N_SPLITS, np.bincount(y).min()))
def cv_oof(name, yy, seed):
    oof = np.zeros(len(yy))
    for tr, te in StratifiedKFold(n_split, shuffle=True, random_state=seed).split(X, yy):
        oof[te] = build(name, Cs[name]).fit(X.iloc[tr], yy[tr]).predict_proba(X.iloc[te])[:, 1]
    return oof

rng = np.random.default_rng(SEED)
print(f"\n{'model':18s} {'AUC mean+/-sd':16s} {'95% CI (bootstrap)':20s} {'avg precision':14s}  (prevalence {y.mean():.2f})")
oofs = {}
for name in models:
    aucs, aps, preds = [], [], []
    for r in range(N_REPEATS):
        o = cv_oof(name, y, SEED + r); preds.append(o)
        aucs.append(roc_auc_score(y, o)); aps.append(average_precision_score(y, o))
    pooled = np.mean(preds, 0); oofs[name] = pooled
    boots = []
    for _ in range(1000):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            boots.append(roc_auc_score(y[i], pooled[i]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"{name:18s} {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}   [{lo:.3f}, {hi:.3f}]       {np.mean(aps):.3f}")

obs = roc_auc_score(y, cv_oof("B image-only", y, SEED))
null = [roc_auc_score(yp, cv_oof("B image-only", yp, SEED)) for yp in (rng.permutation(y) for _ in range(N_PERM))]
print(f"\nModel B permutation test ({N_PERM} label shuffles): observed AUC {obs:.3f}, "
      f"p = {(1 + sum(n >= obs for n in null)) / (N_PERM + 1):.3f}")
pd.DataFrame({"patient_id": df["patient_id"], "y": y, **{k: v for k, v in oofs.items()}}) \
    .to_csv(OUT_DIR / f"oof_deep_{TASK}.csv", index=False)
