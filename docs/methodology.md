# Methodology

> **Status:** Preliminary undergraduate ML research. This document describes experiments completed so far, not a validated clinical model or a full replication of Lin et al. (2026).

## Research question

If machine-learning models predict mortality similarly well, do they agree on which factors matter most for the same person?

This initial experiment predicts **5-year all-cause mortality**, not yet cardiovascular-specific mortality. It builds on [Lin et al. (2026), *Toward practical screening of mortality risk: Insights from interpretable machine learning in NHANES*](https://pmc.ncbi.nlm.nih.gov/articles/PMC12969032/).

## Dataset and outcome

We assembled a preliminary cohort from NHANES 1999–2004 and linked mortality records:

- Initial survey records: **31,126**.
- Analytic cohort: **9,957** adults aged 40+.
- `death_5yr=1`: **1,293** deaths recorded within five years.
- `death_5yr=0`: **8,664** other participants (not necessarily all verified to have five complete years of follow-up).

The preliminary binary outcome was defined using death status and `PERMTH_INT <= 60`. **Before final analysis, we must check for right-censoring**: an individual not recorded as deceased may lack the full five years of observation.

The current models use three predictors: `RIDAGEYR` (age), `SSTNT` (high-sensitivity Troponin-T), and `SSBNP` (NT-proBNP). The intended 134-predictor replication dataset is **not yet finished**.

## Experimental design

We used stratified five-fold cross-validation (`shuffle=True`, `random_state=42`) to maintain roughly the same proportions of positive and negative outcomes across folds. Preprocessing was fitted **inside each training fold** and applied to its validation fold to avoid leakage.

- Numeric missing values: training-fold median imputation.
- Logistic Regression (GLM): imputation, then `StandardScaler`.
- Random Forest: imputation, without scaling.
- Both learned models: `class_weight="balanced"` for the imbalanced outcome.

Because class weighting was used and the models were not calibrated, their output probabilities are **not validated clinical risks**.

### Majority-class baseline

Always predicts `0` (no recorded death within five years). It has about **87.0% accuracy**, but precision, recall, and F1 are all zero for the positive class, and ROC-AUC is **0.500**.

### Logistic Regression

The model first computes a score, then passes it through the sigmoid function:

\[
z=A+Bx_1+Cx_2+Dx_3
\]

\[
P(Y=1\mid x)=\frac{1}{1+e^{-z}}
\]

A model trained on the separate stratified 80/20 split used for SHAP analysis learned the following coefficients, converted back from standardized units:

\[
z=-5.889396+0.081101(\mathrm{Age})+0.026090(\mathrm{TroponinT})+0.000182(\mathrm{NTproBNP})
\]

These numbers describe **that fitted training-split model**, not coefficients averaged across the five cross-validation folds. When an input is missing, the fitted imputer supplies its training median. The coefficients use different measurement units, so raw magnitudes should not be compared directly to determine feature importance.

Configuration: `LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)`.

### Random Forest

The Random Forest averages information from multiple decision trees. We used `RandomForestClassifier(n_estimators=200, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1)`.

For ROC-AUC, we used continuous model scores (the Random Forest's predicted probabilities), **not just binary predictions**.

## Performance metrics

Let TP = true positives, FP = false positives, FN = false negatives, and TN = true negatives, where death within five years is the positive class.

**Precision:** of those predicted to die, the fraction observed to die.

\[
\mathrm{Precision}=\frac{TP}{TP+FP}
\]

**Recall:** of those observed to die, the fraction predicted to die.

\[
\mathrm{Recall}=\frac{TP}{TP+FN}
\]

**F1-score:** the harmonic mean of precision and recall.

\[
F_1=2\frac{\mathrm{Precision}\cdot\mathrm{Recall}}{\mathrm{Precision}+\mathrm{Recall}}
\]

**Accuracy:** `(TP + TN) / (TP + TN + FP + FN)`.

**ROC-AUC:** summarizes ranking performance across classification thresholds; 0.5 indicates chance-level separation and 1.0 perfect separation on evaluated data.

### Five-fold cross-validation results

| Model | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|
| Majority-class baseline | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic Regression | 0.296 | 0.771 | 0.427 | 0.816 |
| Random Forest | 0.351 | 0.707 | 0.469 | 0.833 |

The learned-model values are means across the five validation folds.

## Comparing predictions for the same people

Using out-of-fold predictions from the same five cross-validation folds, the models gave opposite binary classifications for **1,633 of 9,957 participants (16.4%)**, at a threshold of 0.5. The mean absolute difference between their uncalibrated probability outputs was **0.148** (14.8 percentage points). This is a descriptive comparison of model scores, not a comparison of validated clinical risk probabilities.

## Explainability: SHAP

SHAP attributes a model's prediction to its input features relative to a specified reference/background distribution. A positive SHAP value raises the model prediction relative to that reference; a negative value lowers it. **These are model explanations, not evidence that the features cause mortality.**

For the explanation study, both models were fit on the same stratified training split (**7,965 participants**) and evaluated on the same held-out split (**1,992 participants**). We used the same 40-person background sample and a **permutation SHAP explainer** for each model's probability output.

For participant `i` and model `m`, the top-ranked feature is the one with the greatest absolute SHAP value:

\[
\mathrm{TopFeature}^{(m)}_i=\arg\max_j|\phi^{(m)}_{ij}|
\]

We measured three forms of agreement:

1. **Top-feature agreement:** the fraction of people for whom both models selected the same feature with the largest absolute SHAP value.
2. **Full-ranking agreement:** the fraction for whom the complete ordering of all three features by absolute SHAP values matched.
3. **Spearman rank correlation:** the correlation between the two models' feature-importance rankings within each person, then averaged across people. With only three features, this statistic takes relatively coarse values.

### Preliminary SHAP findings

| Held-out sample | Top-feature agreement | Complete-ranking agreement | Mean Spearman |
|---|---:|---:|---:|
| 100 sampled participants | 49.0% | 23.0% | 0.265 |
| 300 participants with both biomarkers observed | 45.3% | 28.7% | 0.338 |

The two exploratory samples may overlap; they are **not independent replications**. Within the 300-person complete-biomarker sample, different model architectures frequently identified different highest-impact features despite similar overall cross-validated ROC-AUC.

### Missing biomarker sensitivity analysis

In the 100-person sample:

| Biomarker measurements | Participants | Top-feature agreement | Complete-ranking agreement |
|---|---:|---:|---:|
| Both available | 84 | 42.9% | 27.4% |
| At least one missing | 16 | 81.2% | 0.0% |

The smaller subgroup should not be overinterpreted. Median imputation can affect both predictions and explanations. The subsequent 300-person analysis restricted **the explanation sample** to participants with both biomarkers observed; the models themselves were still fitted with imputation.

## Limitations and next steps

- **Outcome mismatch:** these experiments predict **all-cause**, not cardiovascular-specific, mortality. The current cohort checkpoint does not contain `UCOD_LEADING`, so we cannot yet label cause-specific deaths.
- **Follow-up and censoring:** audit the five-year mortality label and consider survival/competing-risk methods where required by the endpoint.
- **Small feature set:** only three predictors are used here; the 134-feature replication pipeline remains unfinished.
- **Explanation robustness:** examine different model training splits, different SHAP background sets, near-ties in rankings, and more held-out participants.
- **Clinical calibration:** class-balanced, uncalibrated probability outputs are not personalized clinical risk estimates.
- **Reproducibility:** document how to obtain and prepare the underlying NHANES files; raw data and intermediate local checkpoints are not stored in GitHub.

## Code and reproducibility status

The notebook for these experiments is `notebooks/03_onehot_baseline_cv.ipynb`. A cardiovascular-outcome investigation is in progress. All stated values are from preliminary notebook executions and must be independently checked before publication or clinical interpretation.
