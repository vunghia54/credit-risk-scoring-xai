# Frozen Model Serialization

## Artifact Scope

Complete fitted tree preprocessor + XGBClassifier, generated locally. The binary artifact is intentionally excluded from Git.

## Frozen Model Specification

XGBoost Baseline / XGB-01; calibration **none**. Frozen thresholds: **0.50 / 0.19**, applied externally with score >= threshold.
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

## Training Policy

Fit once on **104,998 TRAIN rows only**, using `build_final_model_pipeline()`. No TRAIN+VALIDATION refit. Validation is used only for reproduction and load-back checks.

## Input Contract

A pandas DataFrame with exactly these ten named real numeric raw features:

- `RevolvingUtilizationOfUnsecuredLines`
- `age`
- `NumberOfTime30-59DaysPastDueNotWorse`
- `DebtRatio`
- `MonthlyIncome`
- `NumberOfOpenCreditLinesAndLoans`
- `NumberOfTimes90DaysLate`
- `NumberRealEstateLoansOrLines`
- `NumberOfTime60-89DaysPastDueNotWorse`
- `NumberOfDependents`

The complete pipeline creates the three indicators internally and produces 13 transformed features. Caller-supplied target, index, group IDs or indicators are rejected. Frozen missing-value rules and TRAIN medians remain inside the artifact.

## Serialization Method

`joblib.dump` with compression 3 and pickle protocol 5; atomic file replacement. Load-back uses `joblib.load` only on the trusted local artifact after SHA-256 verification. Nothing is serialized on module import.

## Validation Reproduction

**PASS** on 22,486 VALIDATION rows before serialization. ROC-AUC **0.8636663931**; Average Precision **0.3812844904**. The committed STEP 14 reference is checked with rtol=1e-6, atol=1e-8; confusion counts must match exactly.

| Threshold | Precision | Recall | F1 |
|---|---:|---:|---:|
| 0.50 | 0.568825911 | 0.190121786 | 0.284989858 |
| 0.19 | 0.372303061 | 0.502029770 | 0.427542495 |

## Load-Back Verification

**PASS** across full VALIDATION. Original/loaded probability shape: [22486, 2] / [22486, 2]. Exact equality: **True**; maximum absolute difference: **0**; mean absolute difference: **0**. Tolerance: atol=rtol=1e-12. Probabilities are finite and within [0,1].

```json
[
  {
    "threshold": 0.5,
    "status": "PASS",
    "identical_binary_predictions": true,
    "original_confusion_matrix": [
      [
        20795,
        213
      ],
      [
        1197,
        281
      ]
    ],
    "loaded_confusion_matrix": [
      [
        20795,
        213
      ],
      [
        1197,
        281
      ]
    ],
    "original_predicted_positive": 494,
    "loaded_predicted_positive": 494
  },
  {
    "threshold": 0.19,
    "status": "PASS",
    "identical_binary_predictions": true,
    "original_confusion_matrix": [
      [
        19757,
        1251
      ],
      [
        736,
        742
      ]
    ],
    "loaded_confusion_matrix": [
      [
        19757,
        1251
      ],
      [
        736,
        742
      ]
    ],
    "original_predicted_positive": 1993,
    "loaded_predicted_positive": 1993
  }
]
```

Loaded pipeline structure, all estimator parameters, feature names, class order, tree count and TRAIN medians were verified. No scaler or calibration layer is present. Thresholds are not learned estimator values.

## Artifact Integrity

- Relative path: `models/final_model.joblib`
- Size: 66547 bytes
- SHA-256: `e99eb083596beac8b6b53c8ac75d90b028dce7bf9229d759a2967709d9084590`
- Workflow version: `STEP18-v1`

```json
{
  "workflow_source_sha256": "82443c80fec4fd7b370b7a148cece050061fbc46826564efc5b59e7c8ed05254",
  "requirements_sha256": "d9a974eb2bb03c139c8889ee4df6f253181ea8ab4891e9324b2024c5016ba788",
  "head_commit": "dd79819353c76ed4bb15385e0ded01f4cb9b80c5",
  "working_tree_clean": false,
  "note": "HEAD identifies the base revision; new workflow files may be uncommitted. Source hash identifies this implementation."
}
```

## Software Environment

| Component | Installed version |
|---|---|
| Python | 3.13.14 |
| numpy | 2.5.3 |
| pandas | 3.0.6 |
| scikit-learn | 1.9.1 |
| xgboost | 3.4.1 |
| joblib | 1.6.0 |

## Security Considerations

joblib/pickle-based artifacts must only be loaded from trusted sources. Loading an untrusted pickle/joblib file may execute arbitrary code. SHA-256 detects changes only when compared with a trusted reference; it does not establish trust.

## Version Compatibility

Serialized sklearn/XGBoost artifacts may depend on library versions. Preserve requirements.txt, artifact SHA-256, metadata/software versions and project source (including src.preprocessing custom classes). The project must be importable when loading. Binary portability across arbitrary future versions is not guaranteed.

## Test-Set Policy

consumed previously; not used in serialization. Only TRAIN and VALIDATION are exposed by the reused loader; its splitter retains existing partition integrity checks. No returned test features/targets are accessed, no test predictions are computed, and no final test or SHAP reports are regenerated.

## Limitations

Round-trip equivalence confirms persistence correctness in this environment, not external performance, regulatory PD, calibration adequacy or production readiness. The model is a fresh deterministic reconstruction of the frozen TRAIN-only specification, not the historical in-memory STEP 15 object. No new test evaluation was performed. Identical predictions do not imply byte-identical artifacts across library versions. No API or deployment retraining policy is introduced.
