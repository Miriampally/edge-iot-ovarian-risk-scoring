"""Mask-free classification, step A: labels + confound check.
1. finds the GDC clinical.tsv (or pass its path), reads FIGO stage per patient;
2. tabulates stage by tissue-source site, slice thickness and scan year for the converted CT cohort;
3. trains a CONFOUND-ONLY classifier for both binary tasks (I-II vs III-IV, III vs IV) (site, thickness, coverage, scan year - no image content).
   If it already predicts stage well, an image model's score cannot be read as 'anatomy'.
Usage: python label_check.py            (auto-finds clinical.tsv)
       python label_check.py "D:\\path\\clinical.tsv" """
import ast
import re
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from config import ROOT, OUT_DIR, SEED, N_REPEATS, N_SPLITS

warnings.filterwarnings("ignore")
ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4}

if len(sys.argv) > 1:
    clin_path = sys.argv[1]
else:
    found = sorted((ROOT / "TCGA Data set").rglob("clinical.tsv"))
    if not found:
        raise SystemExit("No clinical.tsv found under 'TCGA Data set'. Run:  python label_check.py \"<full path to clinical.tsv>\"")
    clin_path = str(found[0])
print("clinical file:", clin_path)

clin = pd.read_csv(clin_path, sep="\t", dtype=str)
id_col = "cases.submitter_id" if "cases.submitter_id" in clin.columns else next(c for c in clin.columns if c.lower().endswith("submitter_id"))
stage_col = next((c for c in clin.columns if "figo_stage" in c.lower()), None)
if stage_col is None:
    raise SystemExit("No FIGO stage column. Columns:\n" + "\n".join(clin.columns))
def parse(s):
    m = re.search(r"stage\s*([ivx]+)", str(s), re.I)
    return ROMAN.get(m.group(1).upper(), np.nan) if m else np.nan
lab = pd.DataFrame({"patient_id": clin[id_col].str[:12], "stage": clin[stage_col].map(parse)}).dropna().drop_duplicates("patient_id")

log = pd.read_csv(OUT_DIR / "step1_conversion_log.csv")
log = log[log["status"] == "ok"].copy()
log["sp"] = log["spacing"].map(lambda s: ast.literal_eval(s))
log["sz"] = log["size"].map(lambda s: ast.literal_eval(s))
log["coverage_mm"] = [s[2] * p[2] for s, p in zip(log["sz"], log["sp"])]
log["thickness_mm"] = pd.to_numeric(log["thickness"], errors="coerce")
log["scan_year"] = (pd.to_numeric(log["study_date"], errors="coerce") // 10000)
log["site"] = log["patient_id"].str[5:7]
df = log[["patient_id", "site", "thickness_mm", "coverage_mm", "scan_year"]].merge(lab, on="patient_id")
print(f"\nCT patients with a FIGO stage: {len(df)} of {len(log)}")
print("\nstage counts:\n", df["stage"].astype(int).value_counts().sort_index().to_string())
df["stage_grp"] = df["stage"].map({1: "I-II", 2: "I-II", 3: "III", 4: "IV"})
print("\nsite x stage:\n", pd.crosstab(df["site"], df["stage_grp"]).to_string())
df["thick_bin"] = pd.cut(df["thickness_mm"], [0, 4, 5.5, 7.5, 100], labels=["<=4", "5", "6-7.5", ">7.5"])
print("\nthickness x stage:\n", pd.crosstab(df["thick_bin"], df["stage_grp"]).to_string())
df.to_csv(OUT_DIR / "label_table.csv", index=False)

pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), ["site"]),
                         ("num", StandardScaler(), ["thickness_mm", "coverage_mm", "scan_year"])])
model = make_pipeline(pre, LogisticRegression(max_iter=1000, C=0.5, class_weight="balanced"))
for task in ("early_vs_advanced", "III_vs_IV"):
    if task == "early_vs_advanced":
        d, y = df, (df["stage"] >= 3).astype(int)
    else:
        d = df[df["stage"] >= 3]; y = (d["stage"] == 4).astype(int)
    y = y.values
    if len(np.unique(y)) < 2 or np.bincount(y).min() < 2:
        print(f"\nTask {task}: skipped (a class has fewer than 2 patients)"); continue
    n_split = int(min(N_SPLITS, np.bincount(y).min()))
    X = d[["site", "thickness_mm", "coverage_mm", "scan_year"]].copy()
    X = X.fillna(X.median(numeric_only=True))
    aucs = []
    for r in range(N_REPEATS):
        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(n_split, shuffle=True, random_state=SEED + r).split(X, y):
            oof[te] = model.fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y, oof))
    print(f"\nTask {task}: n={len(y)}, class counts={np.bincount(y).tolist()}")
    print(f"  CONFOUND-ONLY AUC (site + thickness + coverage + scan year): {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}  (0.5 = chance)")
