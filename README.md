## Research Overview

This project investigates machine learning for mortality risk prediction using data from the National Health and Nutrition Examination Survey (NHANES).

Our current work builds on Lin et al. (2026), *Toward practical screening of mortality risk: Insights from interpretable machine learning in NHANES*.

We are also exploring a research question:

**If several machine-learning models predict mortality similarly well, do they agree on which factors are most important for the same person?**

### Dataset

We are working with NHANES 1999–2004 data.

- Initial participants: 31,126
- Final study cohort: 9,957 adults aged 40+
- Five-year mortality outcomes: 1,293 deaths and 8,664 non-deaths
- Current prediction target: 5-year all-cause mortality

The feature-engineering pipeline is still being developed toward the full 134-variable research dataset.

### Machine Learning Models

We implemented two preliminary models using three predictors:

- Age (`RIDAGEYR`)
- High-sensitivity Troponin-T (`SSTNT`)
- NT-proBNP (`SSBNP`)

Both models were evaluated using stratified 5-fold cross-validation with preprocessing fitted separately within each training fold.

### Preliminary Model Performance

| Model | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|
| Majority-class baseline | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic Regression | 0.296 | 0.771 | 0.427 | 0.816 |
| Random Forest | 0.351 | 0.707 | 0.469 | 0.833 |

Random Forest achieved higher precision, F1-score, and ROC-AUC, while Logistic Regression achieved higher recall.

### Explainable AI and Model Agreement

We are using SHAP to investigate whether different models identify the same important features for individual participants.

In a preliminary experiment involving 300 held-out participants with complete cardiac biomarker measurements:

- Top-feature agreement: 45.3%
- Complete feature-ranking agreement: 28.7%
- Average Spearman rank correlation: 0.338

These preliminary results suggest that similar predictive performance does not necessarily lead to similar feature explanations.

The analysis uses three features, one training/test split for SHAP, and uncalibrated model outputs. Additional validation is required.

### Current Research Progress

**Completed:**
- NHANES cohort construction
- Initial feature engineering
- One-hot encoding and missing-value preprocessing
- Majority-class baseline evaluation
- Stratified 5-fold cross-validation
- Logistic Regression and Random Forest evaluation
- Individual SHAP explanations
- Preliminary feature-ranking agreement analysis

**In progress:**
- Completing the full 134-feature dataset
- Obtaining and validating cause-specific mortality data
- Constructing a cardiovascular-specific mortality outcome
- Testing explanation stability across different training splits
- Expanding model and feature-importance comparisons

### Important Limitations

The current results predict **5-year all-cause mortality**, not cardiovascular-specific mortality.

The current dataset does not contain `UCOD_LEADING`, which is needed for our planned cause-specific mortality analysis.

These models are research prototypes and are not intended for clinical decision-making.

### Research Environment

Python, pandas, NumPy, scikit-learn, SHAP, Jupyter Notebook, and Git.

The notebook containing the preliminary model experiments is `notebooks/03_onehot_baseline_cv.ipynb`.

Raw NHANES data and locally generated checkpoints are excluded from GitHub.