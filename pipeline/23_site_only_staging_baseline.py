"""Site-only (and site + acquisition) baselines for FIGO-stage tasks, under leave-one-patient-out cross-validation.
Purpose: show how much of a CT staging result could be explained without any image content. Compare these numbers with
the performance of any CT staging model evaluated on THE SAME patients and labels (for example the authors' earlier work).
Tasks: 4class (I/II/III/IV), 3class (I-II / III / IV), early_vs_advanced (I-II vs III-IV), III_vs_IV.
Models: 'site only' and 'site + acquisition' (thickness, coverage, scan year); class-balanced logistic regression.
Metrics: AUC (macro one-vs-rest for multiclass) with bootstrap 95% CI, macro-F1, balanced accuracy.
Usage: python 23_site_only_staging_baseline.py"""
import warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from config import OUT_DIR, SEED

warnings.filterwarnings("ignore")
df = pd.read_csv(OUT_DIR / "label_table.csv", dtype={"site": str})
df["stage"] = df["stage"].astype(int)
for k in ("thickness_mm", "coverage_mm", "scan_year"):
    df[k] = df[k].fillna(df[k].median())
tasks = {"4class": (df, df["stage"] - 1),
         "3class": (df, df["stage"].map({1: 0, 2: 0, 3: 1, 4: 2})),
         "early_vs_advanced": (df, (df["stage"] >= 3).astype(int)),
         "III_vs_IV": (df[df["stage"] >= 3], (df[df["stage"] >= 3]["stage"] == 4).astype(int))}
models = {"site only": ColumnTransformer([("s", OneHotEncoder(handle_unknown="ignore"), ["site"])]),
          "site + acquisition": ColumnTransformer([("s", OneHotEncoder(handle_unknown="ignore"), ["site"]),
                                                   ("n", StandardScaler(), ["thickness_mm", "coverage_mm", "scan_year"])])}
rng = np.random.default_rng(SEED)
rows = []
for tname, (d, y) in tasks.items():
    y = y.values; k = len(np.unique(y))
    for mname, pre in models.items():
        pipe = make_pipeline(pre, LogisticRegression(max_iter=2000, class_weight="balanced"))
        p = cross_val_predict(pipe, d, y, cv=LeaveOneOut(), method="predict_proba")
        auc_f = (lambda yy, pp: roc_auc_score(yy, pp[:, 1])) if k == 2 else (lambda yy, pp: roc_auc_score(yy, pp, multi_class="ovr", average="macro", labels=list(range(k))))
        auc = auc_f(y, p)
        b = []
        for _ in range(500):
            i = rng.integers(0, len(y), len(y))
            if len(np.unique(y[i])) == k:
                b.append(auc_f(y[i], p[i]))
        pred = p.argmax(1)
        rows.append(dict(task=tname, n=len(y), model=mname, AUC=round(auc, 3), CI=f"{np.percentile(b, 2.5):.3f}-{np.percentile(b, 97.5):.3f}",
                         macro_F1=round(f1_score(y, pred, average="macro"), 3), balanced_acc=round(balanced_accuracy_score(y, pred), 3)))
res = pd.DataFrame(rows)
res.to_csv(OUT_DIR / "site_only_staging_baseline.csv", index=False)
print(res.to_string(index=False))
print("\nHow to use: if a CT staging model's AUC / macro-F1 on the same patients and task is close to the site-only or site + acquisition row,\n"
      "its performance cannot be attributed to image content. Report both numbers side by side.")
