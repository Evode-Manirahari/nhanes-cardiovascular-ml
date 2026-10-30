# Research Methodology

> **Research status:** Preliminary undergraduate machine-learning study. This document reports experiments completed so far; it does not describe a clinically validated prediction tool or a completed replication of the reference paper.

## 1. Research objective

We are investigating whether machine-learning models with similar mortality-prediction performance identify the same factors as most influential **for an individual participant**.

Our broader research question is:

> If several machine-learning models predict cardiovascular mortality similarly well, do they agree on which factors are most important for the same person?

We began with **5-year all-cause mortality** as a proof of concept. Cardiovascular-specific mortality is a planned extension, not an outcome analyzed in the results below.

This work builds on [Lin et al. (2026), *Toward practical screening of mortality risk: Insights from interpretable machine learning in NHANES*](https://pmc.ncbi.nlm.nih.gov/articles/PMC12969032/).

## 2. Dataset and cohort construction

Data were assembled from the **1999–2004 National Health and Nutrition Examination Survey (NHANES)** cycles and linked mortality records. Participant records from survey, examination, and laboratory files were joined using the NHANES participant identifier `SEQN`.

| Cohort description | Count |
|---|---:|
| Initial NHANES participants across three cycles | 31,126 |
| Adults aged 40+ with the required mortality linkage | 9,957 |
| Labeled death within five years (`death_5yr = 1`) | 1,293 |
| All other participants (`death_5yr = 0`) | 8,664 |

### Preliminary outcome definition

The notebook currently labels a participant positive when `MORTSTAT == 1` and `PERMTH_INT <= 60` (death recorded within 60 months of the interview):

```math
Y_i = \begin{cases}
1, & \text{if death is recorded within 60 months}, \\
0, & \text{otherwise.}
\end{cases}
```

**Important limitation:** A zero label does **not necessarily confirm that a person was observed alive for the entire five-year period**. Participants without a recorded death can have less than 60 months of follow-up. Right-censoring and complete five-year follow-up must be audited before final analysis; survival-analysis methods may be more appropriate for the eventual research endpoint.

### Features in the first experiments

The initial models use three numeric predictors:

| NHANES variable | Description |
|---|---|
| `RIDAGEYR` | Age in years |
| `SSTNT` | High-sensitivity cardiac Troponin-T |
| `SSBNP` | NT-proBNP |

The broader feature-engineering work aims toward the **134-predictor dataset** described in the reference research. That feature set is **not yet complete**, and these results should not be presented as the final 134-feature model.

## 3. Preprocessing and validation design

We used **stratified five-fold cross-validation** to evaluate the predictive models. The five folds preserve approximately the same positive-to-negative class proportion as the overall cohort (about 13% vs. 87%).

- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- Each fold is held out once for validation while the model is fitted on the other four folds.
- Numeric missing values are replaced using a **median fitted on the training fold only**.
- Logistic Regression additionally uses `StandardScaler`, fitted on the training fold only.
- Random Forest does not need feature scaling in this experiment.
- Both trained classifiers use `class_weight="balanced"` to address the imbalanced outcome.
- The same fold specification was used for both model types in each experiment; the initial experiment used three inputs and the expanded experiment used 142 selected inputs.

Fitting preprocessing inside the training portion of every fold helps prevent information from the validation fold leaking into model training.

**Probability caveat:** Because class weighting was used and probability calibration was not performed, reported model probabilities are **uncalibrated scores**, not validated individual clinical risk estimates.

## 4. Majority-class baseline

Our simplest baseline predicts **no death within five years** for every participant:

```math
\widehat{Y}_i = 0 \qquad \text{for every participant } i.
```

Because the outcome is imbalanced, this baseline achieves approximately **87.0% accuracy** while detecting none of the recorded five-year deaths. Its positive-class precision, recall, and F1-score are all zero, and its ROC-AUC is **0.500**.

The baseline demonstrates why accuracy alone is inadequate for this dataset.

## 5. Logistic Regression (GLM)

Logistic Regression constructs a **linear score** from the predictors and transforms that score into a number between zero and one using the **sigmoid function**.

### General formulation

```math
z = A + Bx_1 + Cx_2 + Dx_3
```

```math
P(Y = 1 \mid \mathbf{x}) = \sigma(z) = \frac{1}{1 + e^{-z}}
```

Here, $A$ is the intercept and $B$, $C$, and $D$ are learned feature coefficients.

### Coefficients from our fitted model

We also trained both models on a separate **stratified 80/20 train/test split** for the explanation study. The Logistic Regression model from that training split learned the following coefficients, converted back from standardized features to their original measurement units:

| Parameter | Predictor | Coefficient |
|---|---|---:|
| $A$ | Intercept | -5.889396 |
| $B$ | Age (`RIDAGEYR`) | 0.081101 |
| $C$ | Troponin-T (`SSTNT`) | 0.026090 |
| $D$ | NT-proBNP (`SSBNP`) | 0.000182 |

The resulting score equation is:

```math
\begin{aligned}
z ={}& -5.889396 \\
&+ 0.081101\,(\mathrm{Age}) \\
&+ 0.026090\,(\mathrm{Troponin\ T}) \\
&+ 0.000182\,(\mathrm{NTproBNP})
\end{aligned}
```

The predicted model output is obtained by substituting this score into the sigmoid equation above.

These coefficients belong to the **single fitted model used for SHAP**, not an average of the models fitted in cross-validation. Missing measurements are replaced by medians learned from its training data. Since the predictors have different units, the raw coefficients cannot be compared directly as measures of feature importance.

Model settings:

```python
LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)
```

## 6. Random Forest

Random Forest combines many decision trees. Each tree learns patterns from the training data, and the forest aggregates the trees' outputs to generate a classification or probability score.

Model settings:

```python
RandomForestClassifier(
    n_estimators=200,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
```

We used the same three predictors and stratified five-fold cross-validation as Logistic Regression. For ROC-AUC, the evaluation uses **continuous predicted scores/probabilities**, not the final zero-or-one class labels. In scikit-learn, Random Forest supplies these through `predict_proba()`.

## 7. Model evaluation metrics

For these binary outcomes, a **positive** means a recorded death within five years.

- **TP:** a death correctly predicted as a death.
- **FP:** a non-death incorrectly predicted as a death.
- **FN:** a death incorrectly predicted as a non-death.
- **TN:** a non-death correctly predicted as a non-death.

### Accuracy

The fraction of predictions that are correct:

```math
\mathrm{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
```

### Precision

Of the participants predicted to die, the fraction recorded as having died:

```math
\mathrm{Precision} = \frac{TP}{TP + FP}
```

### Recall (sensitivity)

Of the participants recorded as having died, the fraction correctly identified:

```math
\mathrm{Recall} = \frac{TP}{TP + FN}
```

### F1-score

The harmonic mean of precision and recall:

```math
F_1 = 2 \times \frac{\mathrm{Precision} \times \mathrm{Recall}}{\mathrm{Precision} + \mathrm{Recall}}
```

In our baseline calculation, undefined positive-class metrics resulting from no positive predictions are reported as zero using `zero_division=0`.

### ROC-AUC

The receiver operating characteristic (ROC) curve plots **true positive rate** against **false positive rate** as the classification threshold changes:

```math
\mathrm{TPR} = \frac{TP}{TP + FN}
```

```math
\mathrm{FPR} = \frac{FP}{FP + TN}
```

**ROC-AUC** summarizes the model's ability to rank recorded deaths above non-deaths across thresholds. A value of **0.500** indicates chance-level ranking, while **1.000** indicates perfect separation on the evaluated observations. It is **not** a measure of probability calibration.

### Five-fold validation results (initial three-feature experiment)

| Model | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|
| Majority-class baseline | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic Regression | 0.296 | 0.771 | 0.427 | 0.816 |
| Random Forest | 0.351 | 0.707 | 0.469 | 0.833 |

Values for Logistic Regression and Random Forest are **means across the five validation folds**. Random Forest had the higher precision, F1-score, and ROC-AUC; Logistic Regression had higher recall. These are preliminary results for the three-feature **all-cause mortality** experiment.

## 8. Comparing predictions for the same participants

We generated **out-of-fold predicted probability scores** with both models using the same five stratified folds. For each participant, that participant's prediction was produced by a model that did **not** train on them.

The absolute model-score difference for participant $i$ is:

```math
\Delta_i = \left|\widehat{p}_{\mathrm{GLM},i} - \widehat{p}_{\mathrm{RF},i}\right|
```

The observed mean absolute difference was **0.148**, or 14.8 percentage points. At the illustrative decision threshold of **0.5**, the models gave different binary classifications for **1,633 of 9,957 participants (16.4%)**.

| Logistic Regression | Random Forest | Participants |
|---|---|---:|
| Predicts no death | Predicts no death | 6,149 |
| Predicts no death | Predicts death | 434 |
| Predicts death | Predicts no death | 1,199 |
| Predicts death | Predicts death | 2,175 |

This comparison concerns **uncalibrated model scores** and an illustrative threshold. Disagreement about predictions does not automatically imply disagreement about the explanations for those predictions.

## 9. SHAP: explaining individual predictions

SHAP (SHapley Additive exPlanations) assigns each input feature a contribution relative to a background or reference prediction.

For the probability-output explanations used here, the additive decomposition is represented as:

```math
f(\mathbf{x}_i) \approx E[f(\mathbf{X})] + \sum_{j=1}^{3} \phi_{ij}
```

where $\phi_{ij}$ is the SHAP contribution of feature $j$ for participant $i$. Positive SHAP values raise the model output relative to its reference value, and negative values lower it. **SHAP describes model behavior; it does not demonstrate medical causation.**

### Explanation experiment setup

For this experiment, we fitted separate copies of both models on the same stratified training set and explained predictions for the same held-out participants:

| Group | Participants |
|---|---:|
| Training | 7,965 |
| Held-out test | 1,992 |

Both models' explanations used:

- The same set of three input features.
- The same **40-participant background sample** from the training set.
- `shap.Explainer(..., algorithm="permutation", seed=42)`.
- Each fitted model's **class-1 probability output** as the explained function.
- Absolute SHAP contribution magnitudes to rank features within a participant.

Using the same explainer approach and reference sample makes the two models more comparable, although results can still depend on the background choice and permutation approximation.

### Measuring explanation agreement

**Top-feature agreement** measures whether both models choose the same highest-magnitude SHAP feature for a participant:

```math
j_i^{(m)} = \underset{j}{\mathrm{argmax}}\;\left|\phi_{ij}^{(m)}\right|
```

```math
\mathrm{TopAgreement} = \frac{1}{N}\sum_{i=1}^{N}\mathbb{1}\left[j_i^{(\mathrm{GLM})}=j_i^{(\mathrm{RF})}\right]
```

**Complete ranking agreement** measures whether the full ordering of the three features by absolute SHAP value is identical in both models:

```math
\mathrm{FullAgreement} = \frac{1}{N}\sum_{i=1}^{N}\mathbb{1}\left[R_i^{(\mathrm{GLM})}=R_i^{(\mathrm{RF})}\right]
```

Here, $R_i^{(m)}$ is model $m$'s ordered list of the three feature indices for participant $i$.

**Spearman rank correlation** measures how similarly the two models order the features for an individual participant. For $K$ features with no tied ranks, it can be written as:

```math
\rho = 1 - \frac{6\sum_{j=1}^{K}d_j^2}{K(K^2-1)}
```

where $d_j$ is the difference between the two models' ranks for feature $j$. The analysis used `scipy.stats.spearmanr` on the absolute SHAP values and averaged the per-participant correlations. With only **three features**, possible correlation values are relatively coarse; ties and near-ties also deserve further study.

### Preliminary SHAP agreement results

| Held-out sample | Top-feature agreement | Complete ranking agreement | Mean Spearman correlation |
|---|---:|---:|---:|
| 20 sampled participants | 45.0% | Not evaluated | Not evaluated |
| 100 sampled participants | 49.0% | 23.0% | 0.265 |
| 300 participants with both biomarkers recorded | 45.3% | 28.7% | 0.338 |

For the **300-person complete-biomarker sample**, the two models selected different top features for **54.7%** of the participants.

These samples may overlap and are not independent replications. They also use models trained on **one 80/20 split** for SHAP analysis. The ranking differences therefore represent **preliminary exploratory evidence**, not yet a population-wide or split-stable estimate.

### Example: one held-out participant

For participant index `1721`, the feature values were age **60**, Troponin-T **5.39**, and NT-proBNP **11.44**.

| Predictor | GLM SHAP | Random Forest SHAP |
|---|---:|---:|
| Age | **-0.0934** | -0.0775 |
| Troponin-T | -0.0300 | **-0.1136** |
| NT-proBNP | -0.0065 | -0.0306 |

The GLM's largest absolute SHAP contribution came from **age**, while Random Forest's came from **Troponin-T**. Both SHAP values were negative relative to their respective reference outputs. Their uncalibrated predicted scores were **0.293** and **0.121**, respectively.

This illustrates disagreement in an **individual explanation**; it does not establish that either model is medically correct.

## 10. Missing-biomarker sensitivity analysis

Some NHANES participants have missing cardiac biomarker measurements. Our training pipelines replace missing numeric values with medians learned from the corresponding training data.

In a 100-person SHAP sample, we observed:

| Biomarker availability | Participants | Top-feature agreement | Complete-ranking agreement |
|---|---:|---:|---:|
| Both measured | 84 | 42.9% | 27.4% |
| At least one missing | 16 | 81.2% | 0.0% |

Both biomarkers were missing for the same 16 participants in that sample. These subgroup sizes are small, and no causal conclusion can be drawn from the differences.

The later 300-person SHAP study restricted the **evaluation sample** to participants with both biomarkers measured. It **did not retrain the models on complete cases**; training still used the original imputation pipelines.

## 11. Limitations

1. **Outcome definition:** The current target is 5-year *all-cause* mortality, not cardiovascular-specific mortality. The current checkpoint does not contain `UCOD_LEADING` (underlying cause of death).
2. **Censoring:** The preliminary binary label needs review to ensure participants counted as non-events had adequate follow-up. Cause-specific analysis may also require competing-risk methods.
3. **Candidate predictors:** The initial experiment used three features and the later exploratory experiment used 142 selected predictors. The paper's exact 134-predictor dataset is unfinished, and the 142 columns still need a complete variable-code and missing-value audit.
4. **Single explanation training split:** SHAP agreement has been tested with multiple backgrounds for fixed trained models, but not yet across repeated model training splits or multiple approximation seeds.
5. **Imputation and missingness:** Medians for missing laboratory measurements may affect predictions and feature attributions.
6. **Calibration:** Class-balanced predicted probabilities were not calibrated or clinically validated.
7. **Sampling and generalizability:** These are preliminary NHANES experiments. Model agreement findings should not yet be generalized beyond the evaluated data.
8. **No causal interpretation:** High model importance does not prove that a feature causes higher or lower mortality.

## 12. Next research steps

- Obtain and verify the linked cause-of-death data, including `UCOD_LEADING`, with attention to NHANES linkage and public-use coding rules.
- Audit follow-up time and define a defensible cardiovascular mortality endpoint, accounting for censoring and competing causes of death.
- Continue the exact predictor mapping and derivations needed for the planned 134-feature analysis.
- Repeat model training and explanation comparisons across different stratified splits, SHAP backgrounds, and larger held-out participant samples.
- Expand explanation agreement analysis with complete rankings, robustness to near-ties, and missing-data sensitivity checks.
- Evaluate calibration and uncertainty before treating predicted scores as individual risks.

## 13. Implementation and reproducibility

The primary experiment notebook is [`notebooks/03_onehot_baseline_cv.ipynb`](../notebooks/03_onehot_baseline_cv.ipynb). Cohort and feature-building work appears in [`01_cohort_replication.ipynb`](../notebooks/01_cohort_replication.ipynb) and [`02_build_features.ipynb`](../notebooks/02_build_features.ipynb).

The analysis uses **Python, pandas, NumPy, scikit-learn, SciPy, SHAP, and Jupyter**. The notebook loads a locally generated checkpoint at `results/cohort_features_checkpoint.pkl`. Raw NHANES files and local checkpoints are excluded from GitHub, so reproducing the results requires reconstructing the data first. A full data-download and preparation guide remains to be documented.

This document records **results observed in the working notebooks**, which should be independently reproduced and validated before publication.

## 14. Expanded 142-predictor experiments (October 2026)

This section extends the three-predictor experiments documented in Sections 3-10; it does not replace their results. The target remains preliminary **five-year all-cause mortality**.

### 14.1 Candidate predictors and preprocessing

The expanded experiment used **9,957** NHANES participants and **142 selected candidate predictors**: **116 numeric** and **26 categorical**. Starting from 261 candidate predictor columns, the notebook removed 108 survey-weight columns and 11 other columns (survey-design/administration information, mortality-linkage information, and two entirely missing variables). This is **not** yet the paper's verified 134-predictor dataset; variable definitions and nonresponse codes still need a complete audit.

The selected predictors were transformed into **232 model-input columns** after one-hot encoding. Both models used the same set of original predictors. They used training-fold median imputation for numeric variables and training-fold most-frequent imputation followed by `OneHotEncoder(handle_unknown="ignore")` for categorical variables. Logistic Regression additionally used `StandardScaler` on numeric inputs. All preprocessing was fitted separately inside each cross-validation training fold to avoid validation leakage. Some variable-specific special response codes were replaced by missing values during feature preparation; this cleaning is preliminary.

Both models used `class_weight="balanced"` and the same `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`. Logistic Regression used `max_iter=3000`. Random Forest used 200 trees and `min_samples_leaf=5` (`random_state=42`, `n_jobs=-1`). The current numeric missing-value method (median) is simpler than the iterative multivariable imputation described in the reference paper.

### 14.2 Five-fold evaluation

| Model | Predictors | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 3 | 0.296 | 0.771 | 0.427 | 0.816 |
| Random Forest | 3 | 0.351 | 0.707 | 0.469 | 0.833 |
| Logistic Regression | 142 | 0.359 | 0.770 | 0.490 | 0.857 |
| Random Forest | 142 | 0.450 | 0.668 | 0.538 | 0.868 |

Values are means of five validation folds, not scores on the training set. The expanded models had higher ROC-AUC and F1-score than their three-predictor counterparts in this experiment. Random Forest's higher precision and Logistic Regression's higher recall represent a tradeoff at the models' current decision threshold.

### 14.3 Predictions on the same held-out participants

For exploratory individual comparisons, fitted copies of both expanded pipelines used the **same stratified 80/20 split** (7,965 training and 1,992 held-out participants; `random_state=42`). This is a different evaluation design from the five-fold means above.

| Measure | Value |
|---|---:|
| Held-out Logistic Regression ROC-AUC | 0.842 |
| Held-out Random Forest ROC-AUC | 0.865 |
| Different binary classifications at 0.5 threshold | 233 / 1,992 (11.7%) |
| Mean absolute difference in predicted class-1 scores | 0.105 |

Because both models used class balancing but no probability calibration, the class-1 scores are **not validated absolute five-year risks**. A difference in score or classification is not, by itself, a difference in feature explanation.

### 14.4 Permutation SHAP on 142 original predictors

To compare interpretable variables rather than each one-hot indicator separately, permutation SHAP explained each fitted **full preprocessing-and-model pipeline as a function of the original 142 columns**. The background and explained data were converted to numeric arrays of NHANES category codes for compatibility with the SHAP masker; model predictions were checked to remain unchanged after this conversion. Each model's class-1 `predict_proba` output was explained, and predictors were ranked by absolute SHAP contributions.

The initial experiment used the **same 30 held-out participants**, the **same 8 training participants as background**, and permutation SHAP with `max_evals = 2 * 142 + 1` for both fitted models.

| Explanation comparison | Result |
|---|---:|
| Same most influential feature | 2/30 (6.7%) |
| Same complete ranking of 142 predictors | 0/30 (0.0%) |
| Average Spearman correlation of absolute SHAP values | 0.550 |
| Mean share of overlapping top-five features | 8.0% |
| Mean share of overlapping top-ten features | 10.7% |
| Participants sharing at least one top-five feature | 12/30 |

Top-five overlap for one participant means `number of shared features in the two top-five lists / 5`. Complete ranking equality is a very strict measure for 142 features; a zero result should not be interpreted as no similarity anywhere in the rankings. A correlation calculated from all features may also be influenced by many low-importance features.

### 14.5 Background sensitivity

The reference/background sample used by permutation SHAP can affect explanations. As a controlled follow-up, two different background samples of **40 training participants each** were used to explain the **same 30 held-out participants**, with the same fitted models and SHAP settings.

| Comparison | Result |
|---|---:|
| Logistic Regression: same top feature under background A and B | 15/30 (50.0%) |
| Random Forest: same top feature under background A and B | 22/30 (73.3%) |
| GLM vs. Random Forest top-feature agreement, background A | 2/30 (6.7%) |
| GLM vs. Random Forest top-feature agreement, background B | 3/30 (10.0%) |

These numbers measure **background sensitivity**, not stability across retraining. The result suggests that explanation choices can change even if the model and participant are fixed. It does not demonstrate that Random Forest is inherently more explainable or reliable. Additional sensitivity checks should vary training splits, SHAP approximation seeds, and the number and selection of held-out participants.

### 14.6 Interpretation and outstanding validation

In this preliminary experiment, Logistic Regression and Random Forest achieved relatively similar five-fold ROC-AUC values (**0.857** and **0.868**) but frequently differed in their top-ranked SHAP predictor for the same participant. The low top-feature agreement needs to be interpreted alongside the models' own SHAP background sensitivity. The experiment does **not** establish medical causation, prove one model's explanation is correct, or provide a clinically validated risk estimate.

Before making stronger claims, we need to: (1) review each of the 142 predictor definitions and code-specific missing values; (2) complete the paper's verified 134-feature mapping; (3) audit five-year follow-up and censoring; (4) validate the cardiovascular mortality outcome using the downloaded CDC mortality files; and (5) repeat model/explanation comparisons on additional held-out samples and training splits, including top-k stability and uncertainty estimates.

All expanded-model code and outputs are in [`notebooks/03_onehot_baseline_cv.ipynb`](../notebooks/03_onehot_baseline_cv.ipynb).

