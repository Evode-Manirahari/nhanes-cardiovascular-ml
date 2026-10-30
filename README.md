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

Our expanded experiments currently use 142 exploratory candidate predictors. The exact 134-predictor feature set described in the reference paper is still being reconstructed.

### Machine Learning Models

We first trained Logistic Regression (a binomial GLM) and Random Forest using three predictors: age (`RIDAGEYR`), high-sensitivity Troponin-T (`SSTNT`), and NT-proBNP (`SSBNP`). We then expanded both models to **142 selected NHANES predictors**: 116 numeric and 26 categorical. After one-hot encoding, these became 232 model-input columns.

The expanded set was prepared from 261 candidate columns by excluding 108 survey-weight columns and 11 other administrative, linkage, or entirely missing columns. It is an **exploratory feature set**, not the exact 134-predictor replication described in Lin et al. (2026).

Both models were evaluated with stratified five-fold cross-validation (`shuffle=True`, `random_state=42`). Median/mode imputation, one-hot encoding, and Logistic Regression scaling were fitted inside each training fold. Both classifiers used `class_weight="balanced"`.

### Preliminary Model Performance

Values are mean validation scores across five folds. The positive class means recorded death within five years.

| Model | Predictors | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Majority-class baseline | None | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic Regression | 3 | 0.296 | 0.771 | 0.427 | 0.816 |
| Random Forest | 3 | 0.351 | 0.707 | 0.469 | 0.833 |
| Logistic Regression | 142 | 0.359 | 0.770 | 0.490 | 0.857 |
| Random Forest | 142 | 0.450 | 0.668 | 0.538 | 0.868 |

Both models improved their ROC-AUC after expanding the predictors. With 142 predictors, Random Forest had higher precision, F1-score, and ROC-AUC, while Logistic Regression had higher recall.

### Individual Predictions and Explainable AI

On the same stratified 80/20 training/test split, 142-feature models were fitted on **7,965** people and evaluated on **1,992** held-out people. Their held-out ROC-AUC scores were 0.842 (Logistic Regression) and 0.865 (Random Forest). At a 0.5 threshold they disagreed on **233/1,992 participants (11.7%)**; the mean absolute difference between their uncalibrated output scores was 0.105.

The earlier three-feature SHAP experiment had **45.3% top-feature agreement** on 300 held-out participants with complete cardiac biomarker measurements (see the methodology for its setup). We subsequently compared explanations using permutation SHAP on the original 142 predictors (rather than comparing separate one-hot columns). For **30 held-out participants** with the same eight-person background sample:

- Same most important feature: **6.7%** (2/30).
- Identical full feature ranking: **0%** (0/30; very strict for 142 predictors).
- Mean Spearman correlation of absolute SHAP importance: **0.550**.
- Average top-five feature overlap: **8.0%**; average top-ten overlap: **10.7%**.

A separate stability check compared **two different 40-person SHAP backgrounds**, holding the trained models and 30 explained participants fixed. The same model retained its top feature for **50.0%** of participants with Logistic Regression and **73.3%** with Random Forest. Between-model top-feature agreement was **6.7%** and **10.0%**, respectively, under the two backgrounds.

These results are exploratory. Explanation agreement can change with the background sample; we have not established stability across training splits or a large evaluation sample. SHAP values describe model behavior, not causal medical effects.

### Current Research Progress

**Completed:** NHANES cohort construction; initial feature engineering; majority-class baseline; three-feature and 142-feature Logistic Regression/Random Forest evaluations with stratified five-fold cross-validation; individual held-out prediction comparison; preliminary SHAP agreement and background-sensitivity experiments.

**In progress:** Auditing the 142 candidate predictors and their NHANES codebooks, including special missing-value codes; reconstructing the exact 134-predictor paper feature set; validating five-year follow-up and censoring; parsing and linking the locally downloaded CDC 2019 public-use cause-of-death files; defining cardiovascular-specific mortality; and assessing explanation stability over additional training splits and participants.

### Important Limitations

- All model results shown here predict **5-year all-cause mortality**, **not** cardiovascular-specific mortality.
- The current 142-predictor experiment is exploratory and has not undergone a complete predictor/codebook audit or exact 134-feature replication.
- The preliminary binary outcome needs a follow-up/censoring audit; a zero label does not necessarily establish observation for all 60 months.
- Model output scores are **uncalibrated** (both models use class weighting) and should not be interpreted as validated personal risk probabilities.
- SHAP results are based on a single training/test split and a small participant sample; different reference/background samples can alter explanations.
- These models are research prototypes, **not** tools for clinical decisions.

### Research Environment and Reproducibility

Python, pandas, NumPy, SciPy, scikit-learn, SHAP, Jupyter Notebook, and Git.

- Experiments: [`notebooks/03_onehot_baseline_cv.ipynb`](notebooks/03_onehot_baseline_cv.ipynb)
- Detailed methodology: [`docs/methodology.md`](docs/methodology.md)
- Mortality linkage in progress: `notebooks/04_cvd_mortality_outcome.ipynb` (not yet validated)

Raw NHANES files and locally generated data checkpoints are excluded from GitHub.
