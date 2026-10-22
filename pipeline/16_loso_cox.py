"""Leave-one-site-out (LOSO) Cox validation: the closest thing to external validation available inside TCGA-OV.
For each site group (sites with < 15 patients pooled as 'other'): train on all OTHER sites, score the held-out site.
Reports per-site C-index for every model, and a site-stratified Cox hazard ratio (per 1 SD) of each model's
LOSO score, so differences in overall survival between sites cannot explain it.
Models as in 13_cox_analysis.py (A confound+acq, K clinical, E non-image, B image-only, C image+clinical, D all).
Usage: python 16_loso_cox.py          (honours N_PCA / POOL environment variables like script 13)"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from config import ROOT, OUT_DIR, SEED

warnings.filterwarnings("ignore")
N_PERM = int(sys.argv[1]) if len(sys.argv) > 1 else 100
N_PCA = int(os.environ.get("N_PCA", 10))          # sensitivity runs: set N_PCA (5/10/20) and POOL (mean/max)
POOL = os.environ.get("POOL", "mean")
PEN, N_REPEATS, N_SPLITS, MIN_CAT = 0.1, 10, 5, 8
TAG = "" if (N_PCA == 10 and POOL == "mean") else f"_pca{N_PCA}_{POOL}"

lab = pd.read_csv(OUT_DIR / "label_table.csv")
out = pd.read_csv(OUT_DIR / "outcome_table.csv"); out = out.rename(columns={out.columns[0]: "patient_id"})
meta = pd.read_csv(OUT_DIR / "dicom_meta.csv")
clin_path = sorted((ROOT / "TCGA Data set").rglob("clinical.tsv"))[0]
c = pd.read_csv(clin_path, sep="\t", dtype=str).replace(["'--", "--", "Not Reported"], np.nan)
c["patient_id"] = c["cases.submitter_id"].str[:12]
age = pd.Series(np.nan, index=c.index)
if "demographic.age_at_index" in c:
    age = pd.to_numeric(c["demographic.age_at_index"], errors="coerce")
if "diagnoses.age_at_diagnosis" in c:
    age = age.fillna(pd.to_numeric(c["diagnoses.age_at_diagnosis"], errors="coerce") / 365.25)
age = age.groupby(c["patient_id"]).first().rename("age")

df = lab.merge(out[["patient_id", "event", "time_days"]], on="patient_id").merge(age, on="patient_id", how="left") \
        .merge(meta[["patient_id", "manufacturer", "kvp", "contrast_flag"]], on="patient_id", how="left")
df = df[(df["time_days"] > 0) & df["event"].notna()].reset_index(drop=True)
pool = (lambda e: e.mean(0)) if POOL == "mean" else (lambda e: e.max(0))
emb = {p: pool(np.load(OUT_DIR / "deep" / f"{p}.npz")["emb"].astype(np.float32)) for p in df["patient_id"]
       if (OUT_DIR / "deep" / f"{p}.npz").exists()}
df = df[df["patient_id"].isin(emb)].reset_index(drop=True)
EMB = np.stack([emb[p] for p in df["patient_id"]])
T, E = (df["time_days"] / 365.25).values, df["event"].astype(int).values
print(f"settings: PCA components={N_PCA}, slice pooling={POOL}")
print(f"n={len(df)} patients, deaths={E.sum()}, censored={(E == 0).sum()}, median follow-up {np.median(T):.1f} y")

def rare(col):
    s = df[col].fillna("missing").astype(str)
    return s.where(s.map(s.value_counts()) >= MIN_CAT, "other")
num = lambda col: df[col].astype(float).fillna(df[col].astype(float).median())
conf = pd.get_dummies(rare("site"), prefix="site", drop_first=True).astype(float)
for k in ("thickness_mm", "coverage_mm", "scan_year"):
    conf[k] = num(k)
acq = pd.get_dummies(rare("manufacturer"), prefix="mfr", drop_first=True).astype(float)
acq["kvp"] = pd.to_numeric(df["kvp"], errors="coerce"); acq["kvp"] = acq["kvp"].fillna(acq["kvp"].median())
acq["contrast"] = df["contrast_flag"].astype(str).str.lower().eq("true").astype(float)
clin = pd.DataFrame({"age": num("age"), "stage": num("stage")})
blocks = {"A confound+acq": [conf, acq], "K clinical": [clin], "E clinical+conf+acq": [clin, conf, acq],
          "B image-only": [], "C image+clinical": [clin], "D everything+image": [clin, conf, acq]}
uses_img = {"B image-only", "C image+clinical", "D everything+image"}
tab = {k: (pd.concat(v, axis=1) if v else pd.DataFrame(index=df.index)) for k, v in blocks.items()}

def risk(name, tr, te, emb_matrix):
    Xtr, Xte = tab[name].iloc[tr].copy(), tab[name].iloc[te].copy()
    if name in uses_img:
        sc = StandardScaler().fit(emb_matrix[tr]); pca = PCA(N_PCA, random_state=SEED).fit(sc.transform(emb_matrix[tr]))
        cols = [f"pc{i}" for i in range(N_PCA)]
        Xtr = pd.concat([Xtr.reset_index(drop=True), pd.DataFrame(pca.transform(sc.transform(emb_matrix[tr])), columns=cols)], axis=1)
        Xte = pd.concat([Xte.reset_index(drop=True), pd.DataFrame(pca.transform(sc.transform(emb_matrix[te])), columns=cols)], axis=1)
    keep = [k for k in Xtr.columns if Xtr[k].std() > 1e-8]
    s = StandardScaler().fit(Xtr[keep])
    Ztr, Zte = pd.DataFrame(s.transform(Xtr[keep]), columns=keep), pd.DataFrame(s.transform(Xte[keep]), columns=keep)
    d = Ztr.assign(T=T[tr], E=E[tr])
    cph = CoxPHFitter(penalizer=PEN).fit(d, "T", "E")
    return cph.predict_log_partial_hazard(Zte).values


site_grp = df["site"].astype(str).where(df["site"].astype(str).map(df["site"].astype(str).value_counts()) >= 15, "other").values
models = list(tab)
score = {m: np.full(len(T), np.nan) for m in models}
for g in sorted(set(site_grp)):
    te, tr = np.where(site_grp == g)[0], np.where(site_grp != g)[0]
    for m in models:
        score[m][te] = risk(m, tr, te, EMB)
print("\nPER-SITE C-INDEX when that site was NOT used for training (0.5 = no information)")
rows = []
for g in sorted(set(site_grp)):
    te = np.where(site_grp == g)[0]
    row = {"held-out site": g, "n": len(te), "deaths": int(E[te].sum())}
    for m in models:
        row[m.split()[0]] = round(concordance_index(T[te], -score[m][te], E[te]), 3)
    rows.append(row)
res = pd.DataFrame(rows)
print(res.to_string(index=False))
w = res["n"] / res["n"].sum()
print("size-weighted mean:", {m.split()[0]: round(float((res[m.split()[0]] * w).sum()), 3) for m in models})
print("\nSITE-STRATIFIED COX on the LOSO score (hazard ratio per 1 SD; separate baseline hazard for each site)")
for m in models:
    z = (score[m] - score[m].mean()) / score[m].std()
    d = pd.DataFrame({"T": T, "E": E, "score": z, "site": site_grp})
    r = CoxPHFitter().fit(d, "T", "E", strata=["site"]).summary.loc["score"]
    print(f"  {m:22s} HR {r['exp(coef)']:.2f}  (95% CI {r['exp(coef) lower 95%']:.2f}-{r['exp(coef) upper 95%']:.2f})  p = {r['p']:.4f}")
res.to_csv(OUT_DIR / f"loso_cox{TAG}.csv", index=False)
