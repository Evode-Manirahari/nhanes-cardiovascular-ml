> **Note added after this audit (Nov 19, 2026).** This document was written by Codex in a
> nested copy of the repository (`nhanes-cardiovascular-ml/nhanes-cardiovascular-ml/`) and
> moved here unchanged. Paths below that point into that nested copy refer to it. Changes since:
>
> - **MET score fixed** as recommended in `data_source_research_2026-11-19.md`: participants whose
>   every activity record lacks times or minutes now have a missing MET score instead of 0
>   (`sum(min_count=1)` in `src/paper134_build.py`). The non-missing count is now **4,736**
>   (paper: 4,741), so MET is no longer a Table S1 match. The `MET score` row of
>   `paper_data_comparison_2026-11-19.csv` still shows the old 4,741. The candidate dataset was
>   rebuilt in place (file name unchanged); no incomplete-activity flag was added.
> - The parent project's uncommitted work described here (stabilised imputer, results tests,
>   README, mortality notebook) is being committed together with this document.

# Verified data sources and realistic corrections

Research date: November 19, 2026.

The evidence does not support replacing the current NHANES data with an unidentified alternative dataset. Ten files downloaded directly from CDC during this investigation were byte-identical to the local copies. I found four additional public files worth inspecting, a confirmed error in the MET aggregation, and documented methodological choices that need explicit sensitivity analyses. None establishes an exact reconstruction of the authors' analytic dataset.

This is a research and correction plan. Existing datasets and model code were not changed. The earlier [134-predictor comparison](paper_data_comparison_2026-11-19.csv) remains the description of the current inputs.

## What was actually checked

1. Read the [2026 paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12969032/), its supplement already verified against the publisher, official CDC component documentation, and an earlier paper by the same three authors.
2. Downloaded ten existing source files anew: `DEMO`, `DEMO_B`, `DEMO_C`, `LAB06`, `L06_B`, `L06BMT_C`, `L06NB_C`, `L06COT_C`, `L10AM_C`, and `SSTROP_A`. Every downloaded byte matched the corresponding local file. See [URLs and SHA-256 checksums](source_research_2026-11-19/download_verification.csv).
3. Tested all seven nonempty combinations of the three survey cycles for each of nine disputed laboratory predictors. None reproduced its S1 count. See [63 comparisons](source_research_2026-11-19/cycle_combinations.csv) and [coverage by cycle](source_research_2026-11-19/coverage_by_cycle.csv).
4. Downloaded and joined the three fasting files and an alternate thyroid file by `SEQN`, checking their actual coverage in the existing cohort.
5. Recomputed MET missingness, questionnaire-code counts, several documented weight scenarios, and three eGFR equation scenarios without retraining models.
6. Searched the exact paper title, DOI and article identifier, including searches restricted to GitHub, Zenodo, Figshare and Mendeley Data. No identifiable author analysis-code repository, participant-level analytic dataset, or correction notice was located in those searches. This is a search result, not proof that none exists.

## Additional data that really exist

| Source | Local availability and observed usefulness | Appropriate use |
|---|---|---|
| Fasting questionnaires: [PH, 1999–2000](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/1999/DataFiles/PH.htm), [PH_B, 2001–2002](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2001/DataFiles/PH_B.htm), [PH_C, 2003–2004](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2003/DataFiles/PH_C.htm) | Absent from the project's raw folder; downloaded for this audit. `PHAFSTHR` and `PHAFSTMN` provide fasting duration for 8,905 of the 9,957 participants. `PHDSESN` identifies examination session; morning is code **0**. | Add to the audit/QC inputs to distinguish measured glucose or insulin from measurements meeting a specified fasting rule. These files do not supply replacement laboratory values. |
| [Surplus thyroid profile, SSNH4THY](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2001/DataFiles/SSNH4THY.htm) | Absent locally. Contains 829 TSH and 833 total-T4 observations in this cohort. Compared with the existing 2001–2002 thyroid file, it adds **one** participant with TSH and **zero** with total T4. | An alternate assay source to evaluate, not an automatic fill operation. The paper's prose mentions thyroid hormones but S1 does not include a thyroid row; adding them changes the selected feature set. |
| [LAB18T4](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/1999/DataFiles/LAB18T4.htm) and [L40T4_B](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2001/DataFiles/L40T4_B.htm) | Already local. TSH has 878 and 970 cohort observations; total T4 has 878 and 971. | Available for a clearly labelled expanded analysis. Their existence does not resolve which 134 predictors the paper actually used. |
| [DNA methylation files](https://wwwn.cdc.gov/nchs/nhanes/dnam/) | Public `dnmepi.sas7bdat` is already local. CDC separately lists normalized CpG-site data as restricted-access. | The public derived biomarkers are the relevant starting point for the current crosswalk. Restricted CpG data require an RDC proposal and are not an established solution to any discrepancy here. |
| [Linked mortality files](https://www.cdc.gov/nchs/linked-data/mortality-files/) | The project's 2019 public-use linkage is already available. CDC also offers 2022 restricted-use linkage through the RDC. | Keep the linkage version explicit. Public-use follow-up times or causes can be perturbed, although vital status is not. There is no evidence the authors used restricted data, so changing linkage is not a supported fix. |

The actual download links for the four newly inspected files are included in the checksum manifest. Downloads were kept in temporary storage; no source file was merged into the production candidate dataset.

## Corrections supported by direct evidence

### 1. Preserve unknown MET totals as missing

**Location:** `src/paper134_build.py`, `derive()`, line 146; also the corresponding candidate tests.

The current code uses:

```python
g["MET_MIN"].sum(min_count=0)
```

For **five participants**, every activity contribution is missing, but this produces a total of zero. Nine participants have at least one missing contribution; four of those also have observed contributions. CDC documents missing duration/frequency values in the [activity file](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/1999/DataFiles/PAQIAF.htm), including values set missing during editing. Missing duration is not evidence of zero activity.

The minimum correction is `sum(min_count=1)`, which leaves those five totals missing. Also add a QC flag for incomplete contributions: `min_count=1` alone still reports a partial sum for the other four participants. Whether to retain that partial sum or mark the entire total missing should be an explicit analysis rule.

This changes the nonmissing count from **4,741 to 4,736** under the minimum correction, moving it away from S1. That is a valid consequence of fixing this implementation error; it must not be hidden by retaining an unjustified zero. Tests currently expecting 4,741 should distinguish the paper's reported count from the corrected derivation and test all-missing and partially missing activity records separately.

Participants with no activity-file records are a separate case. CDC states that respondents reporting no leisure activity have no PAQIAF row, but absence from this file alone should not be treated as a verified answer for every cohort participant. Use the general PAQ responses to distinguish confirmed nonparticipation from unknown status before assigning zero.

### 2. Correct variable descriptions and make nonresponse handling explicit

**Locations:** `src/paper134_models.py`, `load()`; the feature crosswalk; `src/paper134_build.py` decision and unit metadata.

CDC identifies `MCQ160C` as coronary heart disease and `MCQ160E` as heart attack. Those are distinct variables. The model already substitutes E for C, but the stored crosswalk still describes C. Align the model's exported dictionary with its actual selection and retain the publication's main-text/S1 disagreement as unresolved. [CDC MCQ definitions](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/1999/DataFiles/MCQ.htm).

Similarly, income codes 77/99 and physical-activity codes 7/9 are refusal/unknown responses, not income brackets or activity levels. If the model is intended to impute nonresponse, recode these specific codes to missing **before** fitting preprocessing. If nonresponse is intentionally retained as a predictive category, label and document it explicitly. Neither choice is confirmed as the authors' rule. [CDC income definitions](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/1999/DataFiles/DEMO.htm); [CDC activity definitions](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/1999/DataFiles/PAQ.htm).

Recomputed counts after removing both kinds of nonresponse are 8,797 for income and 9,928 for physical activity. Removing only income code 99 would reproduce S1's count, but there is no evidence-based reason to infer that this selective rule is correct.

## Strong leads that still require an explicit analysis choice

### Heart rate recovery

The [2023 study by Yi-Ting Lin, Lian-Yu Lin and Kai-Jen Chuang](https://pmc.ncbi.nlm.nih.gov/articles/PMC10470307/) explicitly defines HRR at each minute as peak exercise heart rate minus the recovery heart rate. CDC's `CVDR1HR`, `CVDR2HR`, and `CVDR3HR` are the recovery-period heart rates themselves. [CDC CVX](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/1999/DataFiles/CVX.htm).

This is stronger support than count matching alone for using the existing `HRR_1MIN_CAND`, `HRR_2MIN_CAND`, and `HRR_3MIN_CAND` alternatives. Their counts match S1 (885, 923, 424). The place to change the selected predictors is `src/paper134_models.py:load()`, with corresponding dictionary updates. However, the 2023 definition does not verify the exact 2026 code or its peak-heart-rate construction. Label this as an evidence-supported reconstruction until confirmed.

### Insulin assay harmonization

CDC reports that the insulin assay changed in 2003 and gives cross-assay regression equations in [L10AM_C analytic notes](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2003/DataFiles/L10AM_C.htm). The current builder pools raw `LBXIN` values across cycles without a corresponding adjustment.

Create a separately named adjusted column and compare it with the raw-column analysis. Preserve the raw values, record the reference assay and conversion direction, and apply the equation only to the applicable cycle. A literal application of CDC's forward equation to the 2,793 observed 1999–2002 values yields **four negative values**. Their treatment requires a documented rule; silently clipping them to zero would introduce another unsupported decision. This adjustment will not explain the S1 coverage difference and is not a confirmed step in the paper.

### Fasting eligibility

The new PH files permit a direct fasting check. Restricting to morning-session records with 8–24 hours of reported food fasting gives:

| Laboratory variable | All observed | Morning and 8–24 hours fasting | Paper S1 |
|---|---:|---:|---:|
| Plasma glucose | 4,258 | 3,889 | 2,829 |
| C-peptide | 4,187 | 3,829 | 2,790 |
| Insulin | 4,188 | 3,831 | 2,791 |

The corresponding 1999–2002-only counts are 2,602, 2,547 and 2,549. Therefore, this documented fasting scenario also does **not** reconstruct S1. It remains useful QC. Keep fasting duration, session and the appropriate fasting weights alongside these laboratory measurements; do not select a fasting threshold merely to hit the paper's counts. [CDC fasting-weight and assay notes](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2003/DataFiles/L10AM_C.htm). These checks use the reported food-fast duration, not a full adjudication of every ingestion question.

### Survey weights and eGFR

The six-year MEC formula already implemented in `src/paper134_analysis.py:survey_weights()` agrees with [CDC's combining-cycle guidance](https://wwwn.cdc.gov/nchs/nhanes/tutorials/weighting.aspx). Do not change that formula simply because Table 1 differs.

For analyses restricted to specific laboratory subsamples, inspect their specific weights: fasting weights for fasting measures, and `WTSSCB2Y`/`WTSSCB4Y` for the stored cardiac biomarker sample. The latter are explicitly specified by the [troponin documentation](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/1999/DataFiles/SSTROP_A.htm). These weights are not interchangeable for every analysis or the full imputed cohort.

I tested MEC, interview and cardiac-subsample weighting. None reproduced the paper's age/sex/weighted-population summaries. I also tested CKD-EPI 2021, CKD-EPI 2009 and MDRD with the existing standardized creatinine. MDRD was used in the authors' 2023 study, but none of these tested scenarios reproduced the reported eGFR means. See [computed scenarios](source_research_2026-11-19/weight_comparisons.csv). This does not identify the paper's equation or weights.

During these diagnostics, SAS zero decoding artifacts below `1e-70` were normalized to zero before selecting positive-weight records. The builder already performs that normalization for raw numeric data, but the analysis weight reader does not. Adding the same handling at weight loading prevents zero-weight records from being incorrectly counted as positive-weight observations; it does not meaningfully change the weighted means shown here.

## Findings that should remain unresolved

- **Large laboratory-count differences:** all ten inspected local files are current CDC copies. No tested cycle combination matches the nine laboratory counts. The mercury target population also changes between the early cycles and 2003–2004; that is documented survey design, not evidence of a bad merge. [Earlier-cycle eligibility](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2001/DataFiles/L06_B.htm); [2003–2004 eligibility](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2003/DataFiles/L06BMT_C.htm).
- **Medication flags:** BPQ and DIQ provide alternate self-reported medication information, but these measure something different from prescription-class membership. In particular, CDC documents a DIQ050 skip-pattern correction that created missing responses. Do not turn all medication missingness into “no” to reproduce S1's zero missing counts. Review the current derivation in notebook 02, cells 233–234, against the intended definition. [CDC BPQ](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/1999/DataFiles/BPQ.htm); [CDC DIQ](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/1999/DataFiles/DIQ.htm); [CDC prescriptions](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/1999/DataFiles/RXQ_RX.htm).
- **One extra Troponin-T observation:** the fresh SSTROP_A copy is identical to the local one. No participant should be removed without an exclusion rule identifying why.
- **Heart-rate increment, MET aggregation, exact DNA feature selection and thyroid inclusion:** the publication does not uniquely establish these choices. A common or plausible formula is not proof of what the authors used.
- **Glycated albumin:** CDC supplies both concentration and percentage and recommends considering pristine-sample restriction. The code's claim that S1 count matching “resolves” the choice is too strong. Preserve assay/QC alternatives and document the choice. [CDC SSCARD_A](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/1999/DataFiles/SSCARD_A.htm).

## Practical order of work

1. Fix the verified MET missing-to-zero error in a new candidate version, add an incomplete-activity flag, and correct the expected-count tests. Retain the original candidate as provenance.
2. Add the PH files as QC inputs with download checksums. Align the actual model feature dictionary and document nonresponse categories. Normalize decoded zero weights.
3. Run separately labelled sensitivity versions for derived HRR, insulin assay harmonization, fasting eligibility and alternative medication definitions. Keep the same cohort and evaluation splits so changes can be attributed to these choices.
4. Request the authors' exact 134-variable manifest, source-file versions, cohort-selection code, Table S1 cleaning rules, derived-variable equations, and Table 1 weight/population code. Specifically ask about thyroid variables, the coronary-disease/heart-attack disagreement, the nine laboratory counts and the intermediate exclusion counts.

No author was contacted. Until that evidence is available, describe the work as a documented reconstruction with stated deviations, rather than an exact replication.
