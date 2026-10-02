# NHANES Cardiovascular ML Research

Replication and extension of Lin et al. (2026) using NHANES 1999–2004 data for 5-year all-cause mortality prediction.

## Cohort

Starting NHANES sample:

- 31,126 participants
- age 40+ → 9,977
- remove missing mortality follow-up
- final cohort → **9,957 participants**

## Current Progress

### Feature construction

Built and merged major predictor blocks including:

- demographics
- body measurements
- cardiovascular fitness
- physical activity
- A1c
- chemistry panels
- CBC
- fasting glucose, insulin, and C-peptide
- CRP
- cholesterol
- NT-proBNP
- Troponin-T
- trace elements and micronutrients

Work is continuing toward the verified **134-variable paper feature set**.

### Preprocessing / baseline

Completed:

- categorical one-hot encoding
- cleanup of NHANES nonresponse codes
- majority-class mortality baseline
- stratified 5-fold cross-validation
- fold-specific preprocessing to avoid leakage

### Baseline result

5-year mortality:

- survived / no death within 5 years: **8,664**
- died within 5 years: **1,293**

A model that predicts survival for every participant achieves:

**87.01% accuracy**

5-fold cross-validation:

**Mean accuracy: 87.01%**

This demonstrates why accuracy alone is not sufficient for evaluating an imbalanced mortality prediction problem.

## Notebooks

- `01_cohort_replication.ipynb` — reproduce the study cohort
- `02_build_features.ipynb` — construct and merge predictors
- `03_onehot_baseline_cv.ipynb` — one-hot encoding, baseline, 5-fold CV, and fold-safe preprocessing

## Stack

- Python
- pandas
- NumPy
- scikit-learn
- matplotlib
- Jupyter