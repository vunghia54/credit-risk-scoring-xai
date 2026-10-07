# Global SHAP Explainability

## Scope

Global explanation of the previously frozen final model. SHAP explains model behavior, not causal relationships. No feature selection, model change, retuning, local customer explanation or policy derivation is performed.

## Frozen Model

**XGBoost Baseline / XGB-01**, calibration **none**, audit threshold **0.50**, development operating threshold **0.19**, random_state **42**. The original `build_final_model_pipeline()` was fitted only on 104,998 TRAIN rows. The frozen configuration is checked against STEP 14; native importance also reproduces the existing STEP 10 frozen baseline audit. SHAP does not use either classification threshold to select observations.

## Explanation Dataset

All **22,486 VALIDATION rows** and all **13 transformed features** are used for global importance. No validation target or target-conditioned segmentation is used. TEST was not used for SHAP values, feature ranking, plot selection, effect patterns or any explainability decision. The analysis API receives TRAIN features/labels and VALIDATION features only, with no validation target. The existing splitter still performs its approved integrity checks on all partitions; its returned test partition is never accessed by this module.

The beeswarm and dependence plots use the same deterministic **5,000-row** validation sample to limit overplotting, drawn uniformly without replacement with seed 42. Sampling uses neither target nor model score. The bar/table and top-feature selection use full-validation SHAP values.

## SHAP Output Space

`shap.TreeExplainer(model_output="raw", feature_perturbation="tree_path_dependent")` explains XGBoost's binary:logistic margin, the score before sigmoid, on a log-odds scale. The training path counts stored in the trees provide the reference; no validation/test background is fitted. Exact Tree SHAP is requested (`approximate=False`). Positive contributions raise the model margin relative to its base value; negative contributions lower it. A SHAP contribution is not a direct probability increase, a probability percentage, or a causal effect. See the [official TreeExplainer documentation](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html).

Expected/base raw margin: **-2.653778315**. It is the explainer reference value, not validation prevalence or a calibrated PD.

## Additivity Verification

**PASS**: `expected_value + sum(SHAP values)` agrees with `predict(output_margin=True)` on 512 deterministic validation observations. Maximum absolute error: **5.42758789e-06**; mean absolute error: **1.12772221e-06**. Criterion: absolute error <= 1e-05 + 1e-05 * abs(raw margin), allowing float32 tree arithmetic rounding. The SHAP library's additivity check was also enabled for the full validation call. Failures stop report generation.

## Global Feature Importance

| Rank | Transformed feature | Mean absolute SHAP | Relative importance |
|---:|---|---:|---:|
| 1 | RevolvingUtilizationOfUnsecuredLines | 0.780963 | 34.35% |
| 2 | NumberOfTime30-59DaysPastDueNotWorse | 0.345702 | 15.20% |
| 3 | NumberOfTimes90DaysLate | 0.281675 | 12.39% |
| 4 | age | 0.216171 | 9.51% |
| 5 | NumberOfTime60-89DaysPastDueNotWorse | 0.169241 | 7.44% |
| 6 | NumberOfOpenCreditLinesAndLoans | 0.157292 | 6.92% |
| 7 | DebtRatio | 0.102398 | 4.50% |
| 8 | MonthlyIncome | 0.090989 | 4.00% |
| 9 | NumberRealEstateLoansOrLines | 0.082945 | 3.65% |
| 10 | NumberOfDependents | 0.027792 | 1.22% |
| 11 | has_special_delinquency_value | 0.014992 | 0.66% |
| 12 | MonthlyIncome_missing | 0.002096 | 0.09% |
| 13 | NumberOfDependents_missing | 0.001376 | 0.06% |

Mean absolute SHAP measures contribution magnitude, not direction. Relative importance divides each magnitude by their sum (all-zero contributions would produce zero relative values). Stable ties retain the approved feature order. These quantities are neither causal importance nor directly comparable to Logistic Regression coefficients.

## Feature Effect Patterns

The following summaries use all validation observations for the measured top five. Distinct lower/upper quartiles define low/high groups; when quartiles tie, the minimum-value group is compared with larger values. Positive/negative means describe association with higher/lower model margin relative to the reference. These observational summaries do not isolate other features or establish monotonic effects.

- **RevolvingUtilizationOfUnsecuredLines** — low: value <= 0.0305622; high: value >= 0.564023 (validation quartiles). low group: n=5,622, mean signed SHAP=-1.057415, positive SHAP fraction=0.00%; high group: n=5,622, mean signed SHAP=+0.798957, positive SHAP fraction=100.00%. Descriptive association with model raw-margin output; no simple monotonic pattern is established by these group summaries.
- **NumberOfTime30-59DaysPastDueNotWorse** — tied quartiles; low: value = 0; high: value > 0. low group: n=18,885, mean signed SHAP=-0.274257, positive SHAP fraction=0.00%; high group: n=3,601, mean signed SHAP=+0.720385, positive SHAP fraction=100.00%. Descriptive association with model raw-margin output; no simple monotonic pattern is established by these group summaries.
- **NumberOfTimes90DaysLate** — tied quartiles; low: value = 0; high: value > 0. low group: n=21,213, mean signed SHAP=-0.208808, positive SHAP fraction=0.00%; high group: n=1,273, mean signed SHAP=+1.495925, positive SHAP fraction=100.00%. Descriptive association with model raw-margin output; no simple monotonic pattern is established by these group summaries.
- **age** — low: value <= 41; high: value >= 63 (validation quartiles). low group: n=5,720, mean signed SHAP=+0.224189, positive SHAP fraction=99.72%; high group: n=5,654, mean signed SHAP=-0.407877, positive SHAP fraction=0.19%. Descriptive association with model raw-margin output; no simple monotonic pattern is established by these group summaries.
- **NumberOfTime60-89DaysPastDueNotWorse** — tied quartiles; low: value = 0; high: value > 0. low group: n=21,339, mean signed SHAP=-0.123133, positive SHAP fraction=0.00%; high group: n=1,147, mean signed SHAP=+1.027046, positive SHAP fraction=100.00%. Descriptive association with model raw-margin output; no simple monotonic pattern is established by these group summaries.

Dependence plots show transformed values against their SHAP contributions. The x-axis uses symlog with a linear region within +/-1 to show extreme values without dropping or clipping observations. Vertical spread can reflect feature interactions and other model behavior; it does not establish a causal mechanism.

## SHAP vs Native Tree Importance

Source: `reports/tree_feature_importance_audit.csv`, XGBoost entries only. Native importance is normalized gain from training tree splits. SHAP ranks mean absolute raw-margin contributions across validation observations. Differences are expected; neither ranking is ground truth.

| Feature | SHAP rank | Mean absolute SHAP | Native rank | Native gain importance |
|---|---:|---:|---:|---:|
| RevolvingUtilizationOfUnsecuredLines | 1 | 0.780963 | 2 | 0.277404 |
| NumberOfTime30-59DaysPastDueNotWorse | 2 | 0.345702 | 4 | 0.133075 |
| NumberOfTimes90DaysLate | 3 | 0.281675 | 1 | 0.289750 |
| age | 4 | 0.216171 | 7 | 0.018087 |
| NumberOfTime60-89DaysPastDueNotWorse | 5 | 0.169241 | 3 | 0.139052 |
| NumberOfOpenCreditLinesAndLoans | 6 | 0.157292 | 8 | 0.013717 |
| DebtRatio | 7 | 0.102398 | 10 | 0.011743 |
| MonthlyIncome | 8 | 0.090989 | 9 | 0.012446 |
| NumberRealEstateLoansOrLines | 9 | 0.082945 | 6 | 0.023637 |
| NumberOfDependents | 10 | 0.027792 | 11 | 0.008005 |
| has_special_delinquency_value | 11 | 0.014992 | 5 | 0.063099 |
| MonthlyIncome_missing | 12 | 0.002096 | 12 | 0.005828 |
| NumberOfDependents_missing | 13 | 0.001376 | 13 | 0.004158 |

## Interpretation Guidelines

- Interpret direction as associated with higher/lower model output, not as causing delinquency.
- `age=0` is treated as missing and imputed using the training median.
- Delinquency values 96/98 are replaced with missing in the preprocessing working copy and imputed; raw data is unchanged.
- `has_special_delinquency_value` records special-pattern presence before replacement.
- `MonthlyIncome_missing` and `NumberOfDependents_missing` are engineered model inputs. Their contributions explain missingness indicators, not observed income or dependent counts.
- All 13 approved inputs are retained. Low SHAP importance does not trigger feature removal.

## Limitations

- SHAP explains model behavior, not causal relationships or credit policy rules.
- Values are raw-margin / log-odds contributions, not direct probability changes or percentage-point PD effects.
- Attributions depend on the tree-path-dependent reference distribution and feature dependence assumptions; correlated features can share attribution.
- Importance is mean absolute contribution on validation, not statistical significance or model-selection evidence.
- Inputs include training-median imputation and engineered missing/special-value indicators; they are not untouched raw columns.
- Marginal low/high summaries and scatter plots do not establish a monotonic, causal or counterfactual response.
- Native gain importance and SHAP summarize different quantities; neither is ground truth and their numerical units differ.
- Internal validation explanations do not establish external generalization, fairness, production suitability or regulatory PD.
- The consumed internal test remains excluded from explainability development. No frozen decision is revised.

## What SHAP Does Not Establish

SHAP does not establish causality, individual recourse, lending policy, fairness, calibrated probability changes or a production/regulatory PD. No individual customer, waterfall, force plot or local reason code is selected. The internal test was consumed in STEP 15 and was not reused for this explainability analysis. No model, preprocessing, calibration, threshold or hyperparameter decision was revised.

Runtime (seconds): {"figure_export": 1.3301709000079427, "fit_explain_and_aggregate": 2.7908256000082474, "load_fit_explain_and_aggregate": 3.301873099990189, "shap_calculation": 0.500679000018863}. SHAP version: 0.52.0.
