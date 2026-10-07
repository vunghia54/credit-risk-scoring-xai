# Local SHAP Explainability

## Scope

Six predeclared score-region examples from VALIDATION explain the frozen model's predictions. SHAP explains this model's prediction. It does not establish causality. Case aliases are not actual customer IDs.

## Frozen Model

**XGBoost Baseline / XGB-01**, calibration **none**, random_state **42**. Fit only on 104,998 TRAIN rows using `build_final_model_pipeline()`. Preprocessing, hyperparameters and thresholds 0.19/0.50 remain unchanged. No alternative model or combined TRAIN+VALIDATION fit was used.

## Case Selection Policy

Score all 22,486 validation rows. In order, choose nearest scores to the 5th, 50th and 95th percentiles (NumPy linear quantiles on the full distribution), highest score strictly below 0.19, lowest score >=0.19, then nearest score to 0.50. Ties use the smaller validation row position. Previously selected rows are skipped in each rule's eligible ordering; no criterion is relaxed. Selection fails if six eligible distinct rows cannot be found.

Deduplication occurred: **False**. The CSV/JSON stores candidate ranks and skipped counts. `validation_position` is a zero-based dataset row reference, not a customer identifier. Cases are chosen before SHAP is computed and are never replaced based on their explanations.

## Why Target Was Not Used for Case Selection

The selector accepts only predicted probabilities. Deterministic positional ordering resolves ties and duplicates. Validation targets are neither provided to this workflow nor stored in its artifacts. Local SHAP, case selection and contributor extraction use VALIDATION features and frozen model scores only. Neither validation targets nor the returned TEST partition are accessed by this workflow. The reused splitter performs its existing partition integrity checks. No final test artifact is regenerated.

## SHAP Output Space

The same TreeExplainer settings as STEP 16 are used: `model_output="raw"`, `feature_perturbation="tree_path_dependent"`, `approximate=False`. Tree path counts from training provide the reference. For binary:logistic, the raw margin is the total score before sigmoid, on a log-odds scale. Base plus all 13 SHAP contributions reconstructs this margin. Only sigmoid(total raw margin) is reconciled with predicted probability; no sigmoid is applied to an individual contribution.

Expected/base raw margin: **-2.653778315**. Scores are uncalibrated model probabilities, not regulatory PD.

## Selected Validation Cases

| Case alias | Selection rule | Predicted probability | >=0.19 | >=0.50 | Absolute reconstruction error |
|---|---|---:|---:|---:|---:|
| VAL_CASE_01_LOW | LOW_SCORE | 0.006005155 | 0 | 0 | 1.11e-06 |
| VAL_CASE_02_MEDIAN | MEDIAN_SCORE | 0.022956343 | 0 | 0 | 1.36e-06 |
| VAL_CASE_03_HIGH | HIGH_SCORE | 0.333848119 | 1 | 0 | 2.29e-07 |
| VAL_CASE_04_BELOW_019 | JUST_BELOW_0_19 | 0.189973041 | 0 | 0 | 1.42e-06 |
| VAL_CASE_05_ABOVE_019 | JUST_ABOVE_0_19 | 0.190146863 | 1 | 0 | 3.94e-07 |
| VAL_CASE_06_NEAR_050 | NEAR_0_50 | 0.500397205 | 1 | 1 | 2.13e-07 |

Classification flags only apply the frozen >= rules; their quality is not assessed here.

## Local Additivity Verification

**PASS** for all six cases. Maximum absolute error: **1.42208592e-06**; mean absolute error: **7.88569499e-07**. Criterion: absolute error <= 1e-5 + 1e-5 * abs(raw margin), matching STEP 16. Total-margin probability reconciliation: **PASS**; maximum absolute error **2.71708967e-08**.

### VAL_CASE_01_LOW

Selected by **LOW_SCORE**: criterion score 0.006005286, distance 1.30967237e-07, eligible candidate rank 1. Previously selected candidates skipped: 0.

Predicted probability **0.006005155**; prediction at 0.19: **0**; prediction at 0.50: **0**. Total raw margin: **-5.109113693**; base value: **-2.653778315**; absolute reconstruction error: **1.11179543e-06**.

**Top contributors to higher model output** (raw-margin units)

- `MonthlyIncome`: +0.033166
- `NumberRealEstateLoansOrLines`: +0.013743
- `MonthlyIncome_missing`: +0.002047

**Top contributors to lower model output** (raw-margin units)

- `RevolvingUtilizationOfUnsecuredLines`: -1.111439
- `age`: -0.455173
- `NumberOfTime30-59DaysPastDueNotWorse`: -0.312411

Together, all 13 contributions sum to **-2.455334 raw-margin units** relative to the reference. Positive entries contribute toward higher model output and negative entries toward lower model output for this observation. This is a decomposition of the model prediction, not an explanation of an observed outcome or a lending recommendation.

### VAL_CASE_02_MEDIAN

Selected by **MEDIAN_SCORE**: criterion score 0.022963078, distance 6.73439354e-06, eligible candidate rank 1. Previously selected candidates skipped: 0.

Predicted probability **0.022956343**; prediction at 0.19: **0**; prediction at 0.50: **0**. Total raw margin: **-3.750936985**; base value: **-2.653778315**; absolute reconstruction error: **1.36104063e-06**.

**Top contributors to higher model output** (raw-margin units)

- `NumberOfOpenCreditLinesAndLoans`: +0.105211
- `NumberRealEstateLoansOrLines`: +0.035681
- `age`: +0.025893

**Top contributors to lower model output** (raw-margin units)

- `RevolvingUtilizationOfUnsecuredLines`: -0.424046
- `NumberOfTime30-59DaysPastDueNotWorse`: -0.244417
- `NumberOfTimes90DaysLate`: -0.187807

Together, all 13 contributions sum to **-1.097157 raw-margin units** relative to the reference. Positive entries contribute toward higher model output and negative entries toward lower model output for this observation. This is a decomposition of the model prediction, not an explanation of an observed outcome or a lending recommendation.

### VAL_CASE_03_HIGH

Selected by **HIGH_SCORE**: criterion score 0.333821140, distance 2.69785523e-05, eligible candidate rank 1. Previously selected candidates skipped: 0.

Predicted probability **0.333848119**; prediction at 0.19: **1**; prediction at 0.50: **0**. Total raw margin: **-0.690831661**; base value: **-2.653778315**; absolute reconstruction error: **2.28945282e-07**.

**Top contributors to higher model output** (raw-margin units)

- `NumberOfTime60-89DaysPastDueNotWorse`: +1.006519
- `RevolvingUtilizationOfUnsecuredLines`: +0.910857
- `age`: +0.400472

**Top contributors to lower model output** (raw-margin units)

- `NumberOfTime30-59DaysPastDueNotWorse`: -0.226441
- `NumberOfTimes90DaysLate`: -0.186521
- `NumberOfOpenCreditLinesAndLoans`: -0.115951

Together, all 13 contributions sum to **+1.962947 raw-margin units** relative to the reference. Positive entries contribute toward higher model output and negative entries toward lower model output for this observation. This is a decomposition of the model prediction, not an explanation of an observed outcome or a lending recommendation.

### VAL_CASE_04_BELOW_019

Selected by **JUST_BELOW_0_19**: criterion score 0.190000000, distance 2.69585848e-05, eligible candidate rank 1. Previously selected candidates skipped: 0.

Predicted probability **0.189973041**; prediction at 0.19: **0**; prediction at 0.50: **0**. Total raw margin: **-1.450185299**; base value: **-2.653778315**; absolute reconstruction error: **1.42208592e-06**.

**Top contributors to higher model output** (raw-margin units)

- `RevolvingUtilizationOfUnsecuredLines`: +1.532638
- `age`: +0.435417
- `DebtRatio`: +0.177403

**Top contributors to lower model output** (raw-margin units)

- `NumberOfTime30-59DaysPastDueNotWorse`: -0.299330
- `NumberOfOpenCreditLinesAndLoans`: -0.278767
- `NumberOfTimes90DaysLate`: -0.185882

Together, all 13 contributions sum to **+1.203594 raw-margin units** relative to the reference. Positive entries contribute toward higher model output and negative entries toward lower model output for this observation. This is a decomposition of the model prediction, not an explanation of an observed outcome or a lending recommendation.

### VAL_CASE_05_ABOVE_019

Selected by **JUST_ABOVE_0_19**: criterion score 0.190000000, distance 0.000146863461, eligible candidate rank 1. Previously selected candidates skipped: 0.

Predicted probability **0.190146863**; prediction at 0.19: **1**; prediction at 0.50: **0**. Total raw margin: **-1.449056149**; base value: **-2.653778315**; absolute reconstruction error: **3.94218659e-07**.

**Top contributors to higher model output** (raw-margin units)

- `NumberRealEstateLoansOrLines`: +0.862640
- `DebtRatio`: +0.430445
- `NumberOfOpenCreditLinesAndLoans`: +0.403855

**Top contributors to lower model output** (raw-margin units)

- `NumberOfTime30-59DaysPastDueNotWorse`: -0.246083
- `NumberOfTimes90DaysLate`: -0.184250
- `NumberOfTime60-89DaysPastDueNotWorse`: -0.113947

Together, all 13 contributions sum to **+1.204723 raw-margin units** relative to the reference. Positive entries contribute toward higher model output and negative entries toward lower model output for this observation. This is a decomposition of the model prediction, not an explanation of an observed outcome or a lending recommendation.

### VAL_CASE_06_NEAR_050

Selected by **NEAR_0_50**: criterion score 0.500000000, distance 0.000397205353, eligible candidate rank 1. Previously selected candidates skipped: 0.

Predicted probability **0.500397205**; prediction at 0.19: **1**; prediction at 0.50: **1**. Total raw margin: **0.001588802**; base value: **-2.653778315**; absolute reconstruction error: **2.13331077e-07**.

**Top contributors to higher model output** (raw-margin units)

- `RevolvingUtilizationOfUnsecuredLines`: +1.476321
- `NumberOfTime60-89DaysPastDueNotWorse`: +0.843767
- `NumberOfTime30-59DaysPastDueNotWorse`: +0.750779

**Top contributors to lower model output** (raw-margin units)

- `NumberOfTimes90DaysLate`: -0.123660
- `MonthlyIncome`: -0.105757
- `DebtRatio`: -0.089958

Together, all 13 contributions sum to **+2.655367 raw-margin units** relative to the reference. Positive entries contribute toward higher model output and negative entries toward lower model output for this observation. This is a decomposition of the model prediction, not an explanation of an observed outcome or a lending recommendation.

## Interpretation Guidelines

- Contribution CSV contains 78 rows, ranked by absolute SHAP within each case. Ties retain the approved feature order.
- Contributions with absolute value <= 1e-08 are labelled Neutral and excluded from positive/negative top lists.
- `raw_value` is the original value; `transformed_value` is the actual model input. Blank raw values denote missing observations or engineered indicators without a raw source column.
- `age=0` is treated as missing and filled with its TRAIN median. Delinquency 96/98 values are marked missing in a working copy and filled with TRAIN medians.
- `MonthlyIncome_missing` and `NumberOfDependents_missing` record original missingness. `has_special_delinquency_value` records the presence of 96/98 before replacement.
- All three flags are engineered features. The source meaning of 96/98 is unknown; these are not confirmed fraud/error codes or confirmed default counts.
- Waterfalls use transformed feature values and raw-margin contributions. Probability appears only as the total model prediction in the title.

## Limitations

- SHAP explains this model's prediction. It does not establish causality.
- Contributions are raw-margin / log-odds terms, not per-feature probability percentages.
- Sigmoid is applied only to the TOTAL raw margin for reconciliation, never to individual SHAP values.
- Selected cases illustrate predeclared score regions; six examples do not establish population representativeness, fairness or performance.
- Top contributors are not legal adverse action reason codes, regulatory reason codes, or reasons why an observation experienced delinquency.
- Scores are uncalibrated XGBoost predicted probabilities, not regulatory or production-validated PD.
- Attributions depend on transformed inputs, tree-path reference counts and feature dependence; they do not establish counterfactual recourse.
- The source meaning of delinquency values 96/98 remains unknown; the special indicator is not proof of fraud, error or a confirmed delinquency count.
- No model, preprocessing, calibration, hyperparameter, feature or threshold decision is changed using these explanations.

## What These Explanations Are Not

These are not legal adverse action reason codes, regulatory reason codes, causal reasons, explanations of why a customer defaulted, or instructions to approve/reject a borrower. No local result is used to change the frozen model or thresholds. The consumed internal test is excluded from local explainability development.

Runtime (seconds): {"figure_export": 15.309565100003965, "fit_predict_select_explain": 5.194523399986792, "load_fit_predict_select_explain": 7.591496200009715, "local_shap_calculation": 0.1560413000115659}.
