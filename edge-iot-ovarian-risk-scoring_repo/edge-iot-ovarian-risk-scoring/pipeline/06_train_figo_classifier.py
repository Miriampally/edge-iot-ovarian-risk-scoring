"""STEP 5: patient-level FIGO classification. Feature selection happens INSIDE each CV fold (no leakage)."""
import re, json, warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from config import *

warnings.filterwarnings("ignore")
ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4}


class CorrPrune(BaseEstimator, TransformerMixin):
    def __init__(self, thr=0.95):
        self.thr = thr

    def fit(self, X, y=None):
        c = np.nan_to_num(np.abs(np.corrcoef(np.asarray(X), rowvar=False)))
        keep = np.ones(c.shape[0], bool)
        for i in range(len(keep)):
            if keep[i]:
                keep[i + 1:] &= ~(c[i, i + 1:] > self.thr)
        self.keep_ = keep
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.keep_]


def parse_stage(s):
    m = re.search(r"stage\s*([ivx]+)", str(s), re.I)
    return ROMAN.get(m.group(1).upper(), np.nan) if m else np.nan


def load_labels():
    clin = pd.read_csv(CLINICAL_FILE, sep="\t", dtype=str)
    id_col = "cases.submitter_id" if "cases.submitter_id" in clin.columns else next(
        (c for c in clin.columns if c.lower().endswith("submitter_id")), None)
    stage_col = next((c for c in clin.columns if "figo_stage" in c.lower()), None)
    if id_col is None or stage_col is None:
        print("Columns found:\n", list(clin.columns))
        raise SystemExit("Could not find ID / FIGO column. Tell me the column names and I will adjust.")
    df = pd.DataFrame({"patient_id": clin[id_col].str[:12], "stage": clin[stage_col].map(parse_stage)})
    return df.dropna().drop_duplicates("patient_id")


def make_target(df):
    s = df["stage"].astype(int)
    if TASK == "early_vs_advanced":
        df["y"] = (s >= 3).astype(int); names = ["I-II", "III-IV"]
    elif TASK == "III_vs_IV":
        df = df[s >= 3].copy(); df["y"] = (df["stage"] == 4).astype(int); names = ["III", "IV"]
    else:
        df["y"] = s.map({1: 0, 2: 0, 3: 1, 4: 2}); names = ["I-II", "III", "IV"]
    return df, names


def build_model(n_classes):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(1e-8)),
        ("sc", StandardScaler()),
        ("corr", CorrPrune(CORR_THRESHOLD)),
        ("sel", SelectKBest(f_classif, k=TOP_K_FEATURES)),
        ("clf", LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=7, min_child_samples=5,
                               subsample=0.8, colsample_bytree=0.8, class_weight="balanced",
                               random_state=SEED, n_jobs=4, verbose=-1)),
    ])


def score(y, proba, k):
    pred = proba.argmax(1)
    auc = roc_auc_score(y, proba[:, 1]) if k == 2 else roc_auc_score(
        y, proba, multi_class="ovr", average="macro", labels=list(range(k)))
    return dict(auc=auc, macro_f1=f1_score(y, pred, average="macro"), bal_acc=balanced_accuracy_score(y, pred))


def main():
    feats = pd.read_csv(FEATURE_CSV)
    df, names = make_target(load_labels().merge(feats, on="patient_id"))
    X = df.drop(columns=["patient_id", "stage", "y"]).values
    y = df["y"].values.astype(int)
    k = len(names)
    counts = np.bincount(y)
    print(f"Task={TASK} n={len(y)} class counts={dict(zip(names, counts))}")
    n_splits = int(min(N_SPLITS, counts.min()))
    if n_splits < 2:
        raise SystemExit("Smallest class has <2 patients. Change TASK or add masks.")

    results, oof_all = [], []
    for r in range(N_REPEATS):
        oof = np.zeros((len(y), k))
        for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=SEED + r).split(X, y):
            m = build_model(k).fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])
        results.append(score(y, oof, k)); oof_all.append(oof)
    res = pd.DataFrame(results)
    print(res.round(3)); print("\nMean +/- SD over repeats:")
    print(pd.DataFrame({"mean": res.mean(), "sd": res.std()}).round(3))

    pd.DataFrame(np.mean(oof_all, 0), columns=names).assign(patient_id=df["patient_id"].values, y=y) \
        .to_csv(OUT_DIR / f"oof_predictions_{TASK}.csv", index=False)
    res.to_csv(OUT_DIR / f"cv_metrics_{TASK}.csv", index=False)


if __name__ == "__main__":
    main()
