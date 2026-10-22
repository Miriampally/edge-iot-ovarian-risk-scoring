# Mask-free CT analysis of TCGA-OV: reproducibility notes and results log

Exploratory analysis. No tumour masks, no radiologist review, no external validation yet.
Numbers below are copied from the console output of each script; re-run before quoting.

## 1. Run order
| Step | Script | Output (in `ovarian_seg_pipeline_data\outputs` unless noted) |
|---|---|---|
| 1 | `01_dicom_to_nifti.py` | `nifti_raw\*.nii.gz`, `step1_conversion_log.csv` |
| 2 | `02_preprocess.py` | `nifti_preproc\*.nii.gz` (1x1x5 mm, window -160..240 HU) |
| QC | `check_step1.py`, `qc_montage.py`, `view_patient.py`, `compare_series.py` | `qc_png\`, `series_comparison.csv` |
| 3 | `label_check.py` | `label_table.csv`, confound-only AUCs |
| 4 | `08_extract_deep_features.py` | `deep\<patient>.npz`, `deep_summary.csv` |
| 5 | `12_dicom_metadata.py`, `outcome_check.py` | `dicom_meta.csv`, `outcome_table.csv` |
| 6 | `09_train_deep_classifier.py` (FIGO), `10_survival_deep_classifier.py` (3-y death), `11_robustness_checks.py` | console |
| 7 | `13_cox_analysis.py [n_perm]` (env `N_PCA`, `POOL` for sensitivity runs) | `oof_cox*.csv` |
| 8 | `14_km_plot.py`, `15_within_site.py`, `16_loso_cox.py` | `qc_png\km_B.png`, `loso_cox*.csv` |
| 10 | `19_habitus_features.py`, `20_habitus_adjusted_cox.py [n_perm]` | `habitus.csv`; habitus-adjusted Cox comparison |
| 9 | `17_freeze_model.py`, `18_score_new_patients.py` | `frozen_model\model.joblib`, `meta.json`; scores for a new cohort without refitting |

Scripts 03-07 (FCM baseline, nnU-Net export, radiomics, mask tools) were NOT used in the final mask-free analysis.
Settings that matter are in `config.py` (`EXCLUDE_PATIENTS`, `SERIES_OVERRIDE`). Freeze versions with `pip freeze`.

## 2. Cohort decisions
- 143 TCGA-OV patient folders; series chosen by rule: earliest study with an axial CT series (>=30 slices; scouts, localizers, MIPs, coronal/sagittal reformats excluded), original images first, thickness nearest 5 mm.
- Duplicate slices removed; irregular slice gaps corrected by linear interpolation onto a uniform grid using true slice positions.
- Excluded: TCGA-61-2097 (only a Smart Prep series, no diagnostic CT); TCGA-10-0935 (chest-only scan). Remaining: 141.
- Manual series overrides: TCGA-24-0975, TCGA-24-0980, TCGA-25-2390 (earliest study lacked abdomen/pelvis coverage).
- Kept but limited coverage: TCGA-61-1737 (pelvis only), TCGA-25-1317 (partial pelvis).
- TCIA lists TCGA-OV cases with early post-operative scans; some selected scans may be post-surgical (not checked).
- 139 have FIGO stage (I 8, II 8, III 97, IV 26); 139 have survival data (87 deaths, 52 censored, median follow-up 2.9 y).
- Scanners: GE 99, Siemens 34, other 8; 124/141 scans at 120 kVp.

## 3. Features
Chest slices removed with a lung heuristic (high interior air fraction AND >=2 large air regions; chest block = top of scan to end of first lung run). Every 2nd remaining slice, cropped to body, ImageNet EfficientNetB0 (1280-d), averaged per patient. PCA fitted inside each training fold.

## 4. Results so far (all exploratory)
**FIGO stage.** Confound-only (site, thickness, coverage, scan year): AUC 0.864 for I-II vs III-IV (uninterpretable: early stages come from two sites) and 0.577 for III vs IV. Image-only III vs IV: AUC 0.541 (95% CI 0.431-0.667), permutation p = 0.198 -> no detectable signal.

**Death within 3 years** (n = 113, 49 deaths): confound-only 0.598, age+stage 0.530, image-only 0.684 (0.606-0.798), image+clinical 0.671; gain over age+stage +0.151 (0.018 to 0.280); permutation p = 0.010 (100 shuffles). Image embedding does not predict site (accuracy 0.32 vs 0.37 majority); weak link to scan year (r = 0.23).

**Cox, all 139 patients** (C-index, repeated CV): confound+scanner 0.583; age+stage 0.584; non-image model E 0.639; image-only 0.658 (0.595-0.723); image+clinical 0.652; everything incl. image D 0.688 (0.640-0.746). Gain D-E +0.034 (+0.010 to +0.096), permutation p = 0.003 (1000 shuffles).
Sensitivity (D-E): 5 PCs +0.043; 20 PCs +0.041; max pooling +0.036 (p = 0.010 each, 100 shuffles).

**Kaplan-Meier** (tertiles of out-of-fold image score): KM median survival 4.33 / 4.52 / 2.69 y (low/middle/high); deaths 25/27/35 of 46/46/47. Low and middle groups are indistinguishable; the score isolates a high-risk third.

**Within sites** (image-only score): C-index 09 0.654, 13 0.558, 25 0.642, 61 0.744, other 0.730; site-stratified HR per SD 1.74 (1.33-2.27).

**Leave-one-site-out Cox** (train on all other sites, score the held-out site; groups 09, 13, 25, 61, other).
Main setting (10 PCs, mean pooling), C-index per held-out site, image-only: 09 0.673, 13 0.552, 25 0.629, 61 0.800, other 0.762; size-weighted mean 0.664 (non-image model E 0.562; confound-only 0.441; age+stage 0.595; full model D 0.646; image+clinical 0.670). Site-stratified HR per SD of the LOSO score: image-only 1.82 (1.38-2.39); image+clinical 1.89 (1.43-2.50); full 1.66 (1.22-2.27); age+stage 1.45 (1.09-1.92); non-image E 1.24 (0.92-1.67); confound-only 0.79 (0.58-1.06). Max-pooling variant gave the same picture (image-only weighted mean 0.665, HR 1.83). Weakest in the largest site (13: 0.55); site 61 rests on 8 deaths.

**Habitus check** (body area, width, depth, fat fraction from the CT; median over abdomen-pelvis slices; cohort medians 713 cm2, 36.3 cm, 25.4 cm, 0.42): habitus alone C-index 0.532 (0.468-0.617); non-image model + habitus (E2) 0.654; everything + habitus + image (D2) 0.680. Image gain after habitus adjustment (D2 - E2) +0.022 (95% CI -0.002 to +0.064), permutation p = 0.012 (1000 shuffles; 0.020 with 100) - smaller than the +0.034 without habitus, and the interval now reaches zero. Spearman correlation of the out-of-fold image-only score with habitus: body area -0.24 (p = 0.005), width -0.22 (p = 0.009), depth -0.20 (p = 0.019), fat fraction -0.39 (p < 0.001): leaner patients tend to receive higher risk scores. So the score partly tracks body composition. Ascites, muscle mass and disease burden were NOT measured.

External validation: CMB-OV holds imaging metadata only (no outcomes, no annotations; labelled FIGO cohort of 21 patients). CPTAC-OV on TCIA is a pathology collection; its ovarian radiology overlaps TCGA-OV, so it is not independent. An independent hospital cohort with CT and survival is needed.

## 5. Limitations to state
Exploratory; survival endpoint examined after null staging analyses (not corrected for); no external validation (CMB-OV has no outcome data; CPTAC-OV not yet checked); no treatment, residual disease or recurrence data; no tumour masks and no clinician review, so the score is not tumour-specific (partly tracks body composition: leaner patients score higher risk; ascites and disease burden not measured); heterogeneous scanners/eras; series selection used a heuristic plus three manual choices; effect is small (delta C about 0.03-0.04) with wide intervals; small sites give noisy per-site estimates.
