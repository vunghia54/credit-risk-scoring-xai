# Final Development Model Selection

## Selection Scope

**Decision:** freeze XGBoost Baseline / XGB-01, without calibration, before opening the final internal test holdout. This is a user-approved development decision, not an automated metric winner selection.

**Observed evidence:** all results below come from TRAIN CV and the fixed VALIDATION partition. No test metrics are present. Source artifact SHA-256 hashes are recorded in the companion JSON.

## Candidate Models

The current candidates are XGBoost Baseline, tuned LightGBM LGBM-02, and Logistic Balanced as a linear reference. Historical XGB-07, Random Forest, SMOTE and other experiment artifacts remain unchanged.

## Ranking Evidence

| Candidate | Train CV AP | Train CV ROC-AUC | Validation AP | Validation ROC-AUC |
|---|---:|---:|---:|---:|
| XGBoost Baseline | 0.398880 | 0.863557 | 0.381284 | 0.863666 |
| LightGBM LGBM-02 | 0.397801 | 0.863313 | 0.381685 | 0.864139 |
| XGBoost XGB-07 (historical) | 0.400384 | 0.864180 | 0.382030 | 0.863488 |
| Logistic Balanced | 0.356926 | 0.819723 | 0.343405 | 0.818049 |

**Rationale:** AP was the primary development selection metric. XGBoost Baseline is slightly ahead of LGBM-02 in TRAIN CV; LGBM-02 is slightly ahead on validation. These small observed differences do not establish statistical or production superiority. XGB-07 improves AP only slightly while lowering validation ROC-AUC; the simpler baseline decision is retained.

## Threshold Trade-offs

| Threshold | Precision | Recall | F1 | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.568826 | 0.190122 | 0.284990 | 281 | 213 | 20795 | 1197 |
| 0.19 | 0.372303 | 0.502030 | 0.427542 | 742 | 1251 | 19757 | 736 |

Lowering the threshold from 0.50 to 0.19 catches more positives while increasing false positives. No financial cost matrix is available, so no bank-optimal threshold is claimed.

## Calibration Evidence

| XGBoost method | Brier | Log loss | ECE | Mean probability |
|---|---:|---:|---:|---:|
| uncalibrated | 0.049105 | 0.177687 | 0.005417 | 0.068218 |
| sigmoid | 0.050795 | 0.189396 | 0.020427 | 0.067921 |
| isotonic | 0.049070 | 0.177558 | 0.003952 | 0.068286 |

Validation prevalence: 0.065730. Mean probability alone is insufficient to establish calibration quality. STEP 13 ECE uses up to 10 quantile bins and is binning-dependent. Calibrated alternatives were five-fold ensembles, so their differences also include ensembling effects.

## Selected Model

**Frozen selection:** XGBoost Baseline / XGB-01. Pipeline: `build_tree_preprocessor()` followed by `XGBClassifier`. No scaling, class weighting, SMOTE or automatic model selection.

```json
{
  "n_estimators": 300,
  "learning_rate": 0.05,
  "max_depth": 3,
  "min_child_weight": 1,
  "subsample": 1.0,
  "colsample_bytree": 1.0,
  "reg_alpha": 0,
  "reg_lambda": 1,
  "objective": "binary:logistic",
  "eval_metric": "logloss",
  "random_state": 42,
  "n_jobs": -1,
  "tree_method": "hist"
}
```

## Selected Calibration Strategy

**Decision:** `CALIBRATION_METHOD = "none"`. Sigmoid worsens Brier, log loss and ECE. Isotonic improves Brier/log loss only marginally; the gain does not justify an additional calibration ensemble for this development decision. Retain uncalibrated scores.

## Frozen Development Threshold

`DEFAULT_AUDIT_THRESHOLD = 0.50` and `DEVELOPMENT_OPERATING_THRESHOLD = 0.19`.

The 0.19 point was selected in STEP 12 from validation under RECALL_AT_LEAST_50: maximize precision among grid thresholds with recall >= 0.50; ties use higher F1, then higher threshold. The historical grid was 0.05–0.95 in steps of 0.01. This step does not repeat the search. The prediction rule is score >= threshold.

The official name is **development operating threshold**. It is a pre-specified operating point for final internal holdout evaluation, not a production threshold or regulatory cutoff.

## Final Test Evaluation Plan

STEP 15 may evaluate only this frozen model, fitted on TRAIN with the approved preprocessing. Do not refit on TRAIN + VALIDATION for this evaluation.

- Ranking/probability metrics: ROC-AUC, Average Precision, Brier score, log loss and Gini (2 × ROC-AUC − 1).
- Calibration audit: ECE, mean predicted probability, observed test prevalence, and reliability bins using the same STEP 13 quantile definition (10 requested bins, duplicate edges removed, identical scores kept together, empty bins omitted).
- At exactly 0.50 and 0.19: precision, recall, F1, accuracy, specificity, FPR, FNR, predicted positive rate and confusion matrix ordered [[TN, FP], [FN, TP]].
- No threshold search, alternative-model comparison or calibration-method selection on test.

## Test Lock Policy

Test remains locked throughout STEP 14. The reusable splitter may execute its existing integrity checks, but this module only receives TRAIN and VALIDATION and never accesses the returned test target.

After STEP 15 opens test, do not change model, hyperparameters, calibration, preprocessing, thresholds or selected features in response to test results. Do not add SMOTE/class weights or switch to LightGBM because of test performance. Test is the final internal holdout; report lower performance honestly without retuning.

## Limitations

Development decisions reused validation across experiments, so selection optimism remains possible. Small candidate differences are not evidence of significance. Group-aware splitting reduces identical-feature leakage but does not demonstrate external, temporal or production generalization. Calibration quality has only been assessed internally. No business loss function, deployment validation or regulatory assessment has been established.

## What This Model Is Not

This is not an objectively best model, a production-approved system, a regulatory model, or a source of regulatory/production-validated PD. The operating point is not an optimal bank threshold. Model serialization, SHAP and test evaluation are outside STEP 14.

## Validation Reproduction

```json
{
  "status": "PASS",
  "fit_rows": 104998,
  "validation_rows": 22486,
  "roc_auc": 0.863666393084869,
  "average_precision": 0.38128449036768636,
  "threshold_checks": [
    {
      "threshold": 0.5,
      "precision": 0.5688259109311741,
      "recall": 0.19012178619756429,
      "f1": 0.28498985801217036,
      "specificity": 0.9898610053313024,
      "false_positive_rate": 0.010138994668697639,
      "false_negative_rate": 0.8098782138024357,
      "predicted_positive_rate": 0.02196922529573957,
      "tp": 281,
      "fp": 213,
      "tn": 20795,
      "fn": 1197,
      "balanced_accuracy": 0.5899913957644334
    },
    {
      "threshold": 0.19,
      "precision": 0.37230306071249375,
      "recall": 0.5020297699594046,
      "f1": 0.42754249495822527,
      "specificity": 0.940451256664128,
      "false_positive_rate": 0.05954874333587205,
      "false_negative_rate": 0.4979702300405954,
      "predicted_positive_rate": 0.08863292715467402,
      "tp": 742,
      "fp": 1251,
      "tn": 19757,
      "fn": 736,
      "balanced_accuracy": 0.7212405133117663
    }
  ],
  "number_of_transformed_features": 13,
  "tolerance": {
    "rtol": 1e-06,
    "atol": 1e-08
  }
}
```
