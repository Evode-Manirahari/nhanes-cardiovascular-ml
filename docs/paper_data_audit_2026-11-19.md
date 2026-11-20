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

# Data comparison with Lin et al. 2026

Audit date: November 19, 2026.

The current dataset is **not an exact replication of the paper**. The final sample size and five-year mortality total agree, and the checked values agree with their local CDC source files. However, **22 of the 134 selected model predictors have different nonmissing counts from Supplementary Table S1**. Several definitions remain uncertain even where counts agree. Intermediate cohort counts and weighted baseline characteristics also differ.

The complete predictor comparison is in [paper_data_comparison_2026-11-19.csv](paper_data_comparison_2026-11-19.csv). These are counts before model imputation. A matching count is not proof of matching participant-level values or definitions.

## Sources and scope

- [Original paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12969032/), especially Methods 2.1–2.4 and Table 1.
- [Publisher supplement](https://ars.els-cdn.com/content/image/1-s2.0-S2772487526000255-mmc1.docx), downloaded anew during this audit. Its SHA-256 equals the local supplement: `32940aeef0175b4031161fc4bd07564d5e53b75e8ca108c8a4f4745764a51971`.
- Local CDC demographic, laboratory, questionnaire, DNA methylation, and 2019 mortality files; the stored cohort, candidate dataset, and feature crosswalk in the parent project.
- The current checkout's `src/paper134_models.py` feature selection, including the `MCQ160C` to `MCQ160E` substitution and two provisional predictors.

The current checkout is nested inside another repository. Its root is `/Users/evodemanirahari/Documents/nhanes-cardiovascular-ml/nhanes-cardiovascular-ml`; the data are in `/Users/evodemanirahari/Documents/nhanes-cardiovascular-ml/data` and `/Users/evodemanirahari/Documents/nhanes-cardiovascular-ml/results`. The candidate pickle has 9,957 rows and 171 columns, including identifiers, outcome, QC fields and alternative definitions; the model selects 134 predictors.

Candidate SHA-256: `d4f2cfcdabca305201224fd32f86c456bd04a10c2df0f5e5b1c5693d2685c01e`.

This audit compares the available data with published aggregates and definitions. The authors' participant-level analytic dataset was not available for a direct comparison. No data, model definitions or existing results were changed, and models were not retrained.

## Cohort and mortality

The demographic files and fixed-width mortality records were reread independently. Eligibility used age at least 40, mortality linkage eligibility, and observed mortality status and follow-up months. [Paper reference](https://pmc.ncbi.nlm.nih.gov/articles/PMC12969032/).

| Check | Paper | Recomputed locally | Result |
|---|---:|---:|---|
| Initial participants | 31,126 | 31,126 | Match |
| Age 40 or older before mortality exclusion | 9,977 | 9,970 | Difference of 7 |
| Excluded without mortality follow-up | 20 | 13 | Difference of 7 |
| Final cohort | 9,957 | 9,957 | Match |
| Death within 60 interview-follow-up months | 1,293 | 1,293 | Match |
| Remaining participants | 8,664 | 8,664 | Match |

There are 9,957 unique participant identifiers. The independently reconstructed eligible participant set equals the stored cohort, and all candidate death labels agree with `MORTSTAT == 1 and PERMTH_INT <= 60`. No selected participant has missing mortality status/follow-up, and no participant recorded alive has less than 60 months of follow-up. Thus, the README's early-censoring concern is not observed in this specific cohort and mortality-file version. The intermediate-count discrepancy remains unexplained; the paper could contain a reporting error, but that is not established.

| Cycle | Selected participants | Five-year deaths |
|---|---:|---:|
| 1999–2000 | 3,182 | 397 |
| 2001–2002 | 3,483 | 441 |
| 2003–2004 | 3,292 | 455 |

## Predictor coverage

Table S1 has 105 entries, including one entry representing 30 DNA methylation predictors. Expanding that entry using the project's crosswalk gives 134 rows: **112 count matches and 22 differences**. At the supplement-entry level, this is 83 matches and 22 differences out of 105. The supplement does not individually name those 30 DNA predictors, so agreement with its grouped count does not establish the authors' exact DNA feature selection.

The differences below are independently calculated from the candidate data and current model selection. Source: [Supplementary Table S1](https://ars.els-cdn.com/content/image/1-s2.0-S2772487526000255-mmc1.docx).

| Predictor | Paper nonmissing | Current nonmissing |
|---|---:|---:|
| Income | 8,818 | 8,842 |
| Heart rate recovery, 1 minute | 885 | 950 |
| Heart rate recovery, 2 minutes | 923 | 995 |
| Heart rate recovery, 3 minutes | 424 | 471 |
| Troponin-T | 7,965 | 7,966 |
| Cadmium | 2,994 | 8,721 |
| Lead | 2,994 | 8,721 |
| Red blood cell folate | 2,987 | 8,683 |
| Mercury | 354 | 3,718 |
| Vitamin B12 | 2,946 | 8,609 |
| Cotinine | 5,785 | 8,499 |
| Plasma glucose | 2,829 | 4,258 |
| C-peptide | 2,790 | 4,187 |
| Insulin | 2,791 | 4,188 |
| Hypertension history | 9,957 | 9,902 |
| Heart failure history | 9,957 | 9,956 |
| Coronary disease / heart attack history | 9,957 | 9,956 |
| Cerebrovascular event history | 9,957 | 9,956 |
| Hyperlipidemia medication | 9,957 | 9,786 |
| Hypertension medication | 9,957 | 9,806 |
| Diabetes medication | 9,957 | 9,784 |
| Diabetes history | 9,957 | 9,955 |

Larger local counts do not establish that the local data are wrong: the paper may have applied undocumented restrictions, used different source versions or reported incorrect counts. Removing valid observations merely to reproduce a count would not establish replication.

## Definitions still requiring resolution

1. **Heart rate recovery:** the active model uses raw recovery heart rates (`CVDR1HR`, `CVDR2HR`, `CVDR3HR`). Existing alternatives calculate peak exercise heart rate minus recovery heart rate. Their nonmissing counts are 885, 923 and 424, matching S1, but counts alone cannot establish the authors' formula.
2. **Heart rate increment and MET:** the model uses `HR_INCREMENT_CAND_REST` and `MET_MIN_30D_CAND`. Alternate formulas have the same observed counts (1,023 and 4,741 respectively). Neither formula is uniquely confirmed by the publication.
3. **eGFR:** the candidate uses CKD-EPI 2021 with the documented creatinine adjustment. A CKD-EPI 2009 alternative also has 8,541 observations. The paper does not specify which equation it used.
4. **Medication indicators:** the candidate carries prescription-derived flags from the master dataset. The paper does not give a reproducible derivation or missing-value rule, and S1 reports complete coverage while these columns have 151–173 missing observations.
5. **Heart disease variable:** S1 says coronary artery disease, while the main text and Table 1 say myocardial infarction/heart attack. The crosswalk uses `MCQ160C` (coronary heart disease), and the model substitutes `MCQ160E` (heart attack). Both have the same nonmissing count. This conflict in the paper needs clarification.
6. **Nonresponse codes:** the model input retains CDC refusal/unknown codes. Income has 45 records coded 77/99, including 24 coded 99; physical activity has 24 coded 7/9. The S1 income count is consistent with removing 99 alone, whereas its physical-activity count includes the nonresponse values. This is an inference from counts, not a confirmed author cleaning rule. Current preprocessing treats observed codes as categories.
7. **Source identity despite matching counts:** the existing build labels several choices as resolved using count agreement, including glycated albumin units, triglyceride assay and maximum heart rate. Such agreement supports a candidate mapping but does not prove identity of the authors' values.

## Weighted baseline characteristics

Recalculated weighted summaries do not reproduce Table 1 under the weight formula currently implemented in `src/paper134_analysis.py`: two-thirds of the four-year MEC weight for 1999–2002, and one-third of the two-year MEC weight for 2003–2004. Means and percentages below use the observed values and those weights. Standard errors and p-values were not audited. [Table 1 source](https://pmc.ncbi.nlm.nih.gov/articles/PMC12969032/).

| Measure | Paper: non-deaths / deaths | Current: non-deaths / deaths |
|---|---|---|
| Weighted population total | about 109,000,000 / 7,297,538 | 111,982,558 / 9,159,607 |
| Mean age | 55.63 / 69.39 | 55.80 / 70.60 |
| Male percentage | 46.6 / 57.9 | 46.15 / 54.33 |
| Mean eGFR | 88.54 / 77.17 | 88.29 / 70.66 |

Restricting to people with observed TNT, BNP, or both did not reproduce the paper's age/sex summaries either. These differences warrant investigation of the authors' analysis population, weights, missing-value handling and derivations. They are not evidence for any single cause.

## Validation and reproducibility

All **12 existing candidate-data tests passed**, executed from this checkout with their input paths redirected in memory to the parent project's files. They check raw-variable agreement, cycle-specific renamed variables, surplus-serum linkage, candidate derivations, cohort identity and stored checksums. An additional comparison checked all 30 selected DNA columns against the local CDC file; all matched. The tests intentionally preserve several known S1 discrepancies, so a passing test suite is not a certification of paper replication.

The fresh supplement download is byte-identical to the local copy. Both local mortality directories contain byte-identical versions of each of the three cycle files. CDC source files themselves were not all downloaded again during this audit.

The parent repository has uncommitted changes in `README.md`, two notebooks and two model scripts, plus an untracked mortality notebook and results test. Its model imputation implementation differs from this checkout. The earlier clean-Git assessment applied only to the nested current checkout; it did not cover the parent project. Parent model results should not be assumed reproducible from the nested checkout's committed source.

The current README still describes the older exploratory 142-predictor experiment and outstanding mortality work. The source and parent results contain a later candidate workflow, so that README alone is not a reliable description of the latest artifacts.

To complete an exact replication, reconcile the 22 coverage differences, establish the unresolved formulas and cleaning rules, reproduce the weighted baseline table, and record a single versioned source/data configuration. Author clarification or their analytic code is needed wherever the publication does not uniquely specify these choices.
