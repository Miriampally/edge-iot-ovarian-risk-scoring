"""Export the frozen model's specification as tables (TRIPOD+AI item 22): Cox coefficients, PCA variance and PCA loadings.
Writes outputs\\frozen_model\\frozen_model_cox_coefficients.csv, frozen_model_pca_variance.csv, frozen_model_pca_loadings.csv.
Cox coefficients refer to STANDARDISED inputs (each PC / age / stage scaled to mean 0, SD 1), so the hazard ratio is per 1 SD.
Usage: python 22_export_model_spec.py   (run after 17_freeze_model.py)"""
import json
import numpy as np
import pandas as pd
import joblib
from config import OUT_DIR

d = OUT_DIR / "frozen_model"
b = joblib.load(d / "model.joblib")
meta = json.load(open(d / "meta.json"))
rows = []
for name, m in b["models"].items():
    s = m["cph"].summary.reset_index()
    s = s.rename(columns={s.columns[0]: "variable"})
    s.insert(0, "model", name)
    rows.append(s[["model", "variable", "coef", "exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]])
cox = pd.concat(rows, ignore_index=True)
cox.to_csv(d / "frozen_model_cox_coefficients.csv", index=False)
pca = b["pca"]
pd.DataFrame({"component": [f"pc{i}" for i in range(pca.n_components_)], "explained_variance_ratio": pca.explained_variance_ratio_}) \
    .to_csv(d / "frozen_model_pca_variance.csv", index=False)
pd.DataFrame(pca.components_.T, columns=[f"pc{i}" for i in range(pca.n_components_)]) \
    .assign(embedding_feature=[f"emb_{i}" for i in range(pca.components_.shape[1])]).to_csv(d / "frozen_model_pca_loadings.csv", index=False)
print(f"trained on {meta['n_train']} patients, {meta['deaths_train']} deaths; PCA {pca.n_components_} components explain "
      f"{100 * pca.explained_variance_ratio_.sum():.1f}% of embedding variance")
print(cox.round(3).to_string(index=False))
print("saved tables in", d)
