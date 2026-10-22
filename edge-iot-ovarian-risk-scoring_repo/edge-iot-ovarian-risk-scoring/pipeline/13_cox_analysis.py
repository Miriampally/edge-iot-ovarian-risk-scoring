"""Time-to-event (Cox) analysis on ALL patients with survival data - no 3-year dichotomisation.
Penalised Cox models, PCA fitted inside every fold, repeated stratified 5-fold CV, out-of-fold C-index.
  A  confound+acquisition : site, thickness, coverage, scan year, manufacturer, kVp, contrast given
  K  clinical             : age, FIGO stage
  E  clinical+confound+acq (everything EXCEPT the image)
  B  image-only
  C  image+clinical
  D  everything INCLUDING the image
Main question: D vs E (does the image add anything once clinical, site and scanner variables are in?),
tested by shuffling the image rows across patients (permutation).
Usage: python 13_cox_analysis.py [n_permutations]        (default 100; use 1000 for the final result)
Sensitivity (PowerShell):  $env:N_PCA=5; $env:POOL="max"; python 13_cox_analysis.py 100   (reset with Remove-Item Env:N_PCA, Env:POOL)
Needs:  pip install lifelines"""
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

def oof(name, seed, emb_matrix=EMB):
    r = np.zeros(len(T))
    for tr, te in StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed).split(df, E):
        r[te] = risk(name, tr, te, emb_matrix)
    return (r - r.mean()) / (r.std() + 1e-12)

print("\nUnivariate C-index (sanity, no CV):  age %.3f   FIGO stage %.3f   (0.5 = no information)" % (
    concordance_index(T, -clin["age"], E), concordance_index(T, -clin["stage"], E)))
rng = np.random.default_rng(SEED)
print(f"\n{'model':22s} {'C-index mean+/-sd':19s} {'95% CI (bootstrap)'}")
pooled = {}
for name in tab:
    cs, rs = [], []
    for r_ in range(N_REPEATS):
        z = oof(name, SEED + r_); rs.append(z); cs.append(concordance_index(T, -z, E))
    pooled[name] = np.mean(rs, 0)
    b = [concordance_index(T[i], -pooled[name][i], E[i]) for i in (rng.integers(0, len(T), len(T)) for _ in range(300))]
    print(f"{name:22s} {np.mean(cs):.3f} +/- {np.std(cs):.3f}      [{np.percentile(b, 2.5):.3f}, {np.percentile(b, 97.5):.3f}]")

d_obs = concordance_index(T, -oof("D everything+image", SEED), E) - concordance_index(T, -oof("E clinical+conf+acq", SEED), E)
bd = []
for i in (rng.integers(0, len(T), len(T)) for _ in range(300)):
    bd.append(concordance_index(T[i], -pooled["D everything+image"][i], E[i]) - concordance_index(T[i], -pooled["E clinical+conf+acq"][i], E[i]))
print(f"\nImage gain on top of clinical+site+scanner variables (D - E): {d_obs:+.3f}   bootstrap 95% CI "
      f"[{np.percentile(bd, 2.5):+.3f}, {np.percentile(bd, 97.5):+.3f}]")
e_base = concordance_index(T, -oof("E clinical+conf+acq", SEED), E)
null = []
for _ in range(N_PERM):
    shuffled = EMB[rng.permutation(len(T))]
    null.append(concordance_index(T, -oof("D everything+image", SEED, shuffled), E) - e_base)
print(f"Permutation test ({N_PERM} shuffles of the image rows): p = {(1 + sum(n >= d_obs for n in null)) / (N_PERM + 1):.3f}")
pd.DataFrame({"patient_id": df["patient_id"], "time_y": T, "event": E, **pooled}).to_csv(OUT_DIR / f"oof_cox{TAG}.csv", index=False)
