# Mask-free CT deep features and overall survival in TCGA-OV (exploratory)

Code and a frozen model for an exploratory analysis of whether CT features obtained **without tumour masks** carry prognostic
information in ovarian cancer (TCGA-OV), and whether any signal survives control for site, scanner and body habitus.

> **Status: exploratory research code. Not validated, not a medical device, not for clinical use.**
> The prognostic signal is small, has no external validation, and partly reflects body composition.

## Data (not included)
- CT images: The Cancer Imaging Archive, TCGA-OV collection (doi:10.7937/K9/TCIA.2016.NDO1MDFQ).
- Clinical data: Genomic Data Commons (TCGA-OV, `clinical.tsv`).
- Check the current TCIA/TCGA data-use and citation requirements before use or publication.
- `patient_series_list.csv` (to be added): the patient identifiers and DICOM series used.

## Environment
Python 3.11, TensorFlow 2.21, scikit-learn, lifelines 0.30.3, SimpleITK, pandas, numpy 1.26. Tested versions: see `requirements.txt`.
Set the data folders with the environment variables `OVARIAN_DATA_ROOT` and `TCGA_OV_DICOM_ROOT` (or edit `config.py`), and review `EXCLUDE_PATIENTS` and `SERIES_OVERRIDE`, before running.

## Run order
| Step | Script | Purpose |
|---|---|---|
| 1 | `01_dicom_to_nifti.py` | rule-based series selection, duplicate-slice removal, spacing correction |
| 2 | `02_preprocess.py` | resample to 1 x 1 x 5 mm, window -160..240 HU |
| QC | `check_step1.py`, `qc_montage.py`, `view_patient.py` | quality control |
| 3 | `label_check.py` | FIGO labels, site/scanner confound-only baselines |
| 4 | `08_extract_deep_features.py` | lung-slice removal, EfficientNetB0 slice embeddings |
| 5 | `12_dicom_metadata.py`, `outcome_check.py`, `19_habitus_features.py` | scanner metadata, outcomes, body habitus |
| 6 | `09_...`, `10_...`, `11_...` | FIGO and 3-year analyses, robustness |
| 7 | `13_cox_analysis.py` | Cox comparison (main analysis), permutation tests |
| 8 | `14_km_plot.py`, `15_within_site.py`, `16_loso_cox.py`, `20_habitus_adjusted_cox.py` | risk groups, within-site, leave-one-site-out, habitus adjustment |
| 9 | `17_freeze_model.py`, `18_score_new_patients.py`, `22_export_model_spec.py` | frozen model, scoring of a new cohort, model specification tables |
| 10 | `21_cohort_tables.py`, `23_site_only_staging_baseline.py` | cohort and missing-data tables, site-only staging baselines |

Scripts 03-07 (FCM baseline, nnU-Net export, radiomics, mask tools) were not used in the final analysis.
See `ANALYSIS_NOTES.md` for cohort decisions, results and limitations.

## Frozen model
`frozen_model/model.joblib` is a Python pickle. **Load it only from a source you trust.**
`18_score_new_patients.py` scores a new cohort without refitting and evaluates it if outcomes are supplied.
It warns if the patients were in the training set (results would then be in-sample).

## Citation and licence
See `CITATION.cff`. Licence: MIT (see `LICENSE`).
Contact: open an issue on this repository.

## Timing wrapper used for the edge-inference measurements
`my_pipeline.py` runs the per-patient edge pipeline in memory (series selection and read, resampling, EfficientNetB0 slice embeddings, frozen scaler/PCA/Cox score) by importing `config.py`, `01_dicom_to_nifti.py`, `02_preprocess.py` and `08_extract_deep_features.py`. It needs `frozen_model/model.joblib` from `17_freeze_model.py`; without it the scoring stage is skipped. Example:

    python my_pipeline.py "<path to one patient's DICOM folder>"

To time several patients, run `measure_inputs.py` from the repository root (see the root README).
