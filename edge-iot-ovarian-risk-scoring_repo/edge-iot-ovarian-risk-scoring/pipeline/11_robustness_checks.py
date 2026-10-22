"""Robustness checks for the 3-year-survival result (script 10). Pre-specified, no tuning.
 1. Can the image embedding PREDICT the tissue-source site / scan year? (if yes, it carries a scanner/site fingerprint)
 2. Leave-one-site-out (LOSO): train on all other sites, test on the held-out site. Sites with < MIN_SITE patients
    are pooled as 'other'. Per-site AUC plus a pooled AUC over all held-out predictions.
Models as in script 10: A confound-only, K clinical-only (age, stage), B image-only, C image+clinical.
Usage: python 11_robustness_checks.py"""
import warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from config import ROOT, OUT_DIR, SEED, N_SPLITS

warnings.filterwarnings("ignore")
HORIZON_DAYS, N_PCA, C_IMG, MIN_SITE = 3 * 365.25, 10, 0.1, 15

# ---------- same data preparation as script 10 ----------
lab = pd.read_csv(OUT_DIR / "label_table.csv")
out = pd.read_csv(OUT_DIR / "outcome_table.csv")
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
df["y"] = died[df.index].astype(int)
emb = {pid: np.load(OUT_DIR / "deep" / f"{pid}.npz")["emb"].astype(np.float32).mean(0)
       for pid in df["patient_id"] if (OUT_DIR / "deep" / f"{pid}.npz").exists()}
df = df.merge(pd.DataFrame.from_dict(emb, orient="index").add_prefix("img_"), left_on="patient_id", right_index=True).reset_index(drop=True)
y = df["y"].values
img_cols = [k for k in df.columns if k.startswith("img_")]
num_conf, num_clin = ["thickness_mm", "coverage_mm", "scan_year"], ["age", "stage"]
df["site_grp"] = df["site"].where(df["site"].map(df["site"].value_counts()) >= MIN_SITE, "other")
X = df[img_cols + ["site"] + num_conf + num_clin].copy()
for col in num_conf + num_clin:
    X[col] = X[col].fillna(X[col].median())
print(f"n={len(y)}, events={y.sum()}; site groups: {df['site_grp'].value_counts().to_dict()}\n")

# ---------- 1. fingerprint check ----------
E = StandardScaler().fit_transform(df[img_cols].values)
site_y = df["site_grp"].values
if len(np.unique(site_y)) < 2:
    raise SystemExit("Fewer than 2 site groups after pooling small sites; lower MIN_SITE in the script.")
pred = cross_val_predict(make_pipeline(PCA(N_PCA, random_state=SEED), LogisticRegression(max_iter=2000, C=C_IMG)),
                         E, site_y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
maj = pd.Series(site_y).value_counts(normalize=True).iloc[0]
print("1. FINGERPRINT CHECK (image embedding only)")
print(f"   predicts site group: accuracy {np.mean(pred == site_y):.2f} (always-majority = {maj:.2f})")
yr = df["scan_year"].fillna(df["scan_year"].median()).values
pyr = cross_val_predict(make_pipeline(PCA(N_PCA, random_state=SEED), Ridge(alpha=10.0)), E, yr, cv=KFold(5, shuffle=True, random_state=SEED))
print(f"   predicts scan year: correlation r = {np.corrcoef(pyr, yr)[0, 1]:.2f}\n")

# ---------- 2. leave-one-site-out ----------
conf_tf = [("site", OneHotEncoder(handle_unknown="ignore"), ["site"]), ("cnum", StandardScaler(), num_conf)]
clin_tf = [("clin", StandardScaler(), num_clin)]
img_tf = [("img", make_pipeline(StandardScaler(), PCA(N_PCA, random_state=SEED)), img_cols)]
specs = {"A confound-only": (conf_tf, 0.5), "K clinical-only": (clin_tf, 0.5),
         "B image-only": (img_tf, C_IMG), "C image+clinical": (img_tf + clin_tf, C_IMG)}
def build(name):
    tf, C = specs[name]
    return Pipeline([("pre", ColumnTransformer(tf)), ("clf", LogisticRegression(max_iter=2000, C=C, class_weight="balanced"))])
print("2. LEAVE-ONE-SITE-OUT (train on other sites, test on the held-out site)")
rows, pooled = [], {k: np.full(len(y), np.nan) for k in specs}
for g in sorted(df["site_grp"].unique()):
    te = np.where(df["site_grp"] == g)[0]; tr = np.where(df["site_grp"] != g)[0]
    row = dict(held_out=g, n=len(te), deaths_3y=int(y[te].sum()))
    for name in specs:
        p = build(name).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
        pooled[name][te] = p
        row[name] = roc_auc_score(y[te], p) if 0 < y[te].sum() < len(te) else np.nan
    rows.append(row)
res = pd.DataFrame(rows)
print(res.round(3).to_string(index=False))
print("\n   pooled AUC over all held-out predictions:")
for name in specs:
    print(f"   {name:18s} {roc_auc_score(y, pooled[name]):.3f}")
print("   (pooled scores mix sites, so compare models, not absolute values; per-site AUCs on <20 patients are noisy)")
res.to_csv(OUT_DIR / "loso_survival_3y.csv", index=False)
