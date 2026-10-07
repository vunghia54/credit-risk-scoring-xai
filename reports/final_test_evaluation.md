# Final Internal Test Evaluation

## Frozen Evaluation Protocol

The internal test set was opened only after STEP 14 model freeze (commit b6efa63). The pipeline was fitted once on TRAIN only; no validation or test rows were used to fit preprocessing or the estimator. Validation reproduction passed before test prediction. The same fitted pipeline evaluated both partitions.

The approved group-aware assignment uses random_state=42 and isolates exact raw feature groups. The CLI refuses to overwrite an existing final evaluation. No threshold search was performed.

## Final Model

XGBoost Baseline / XGB-01, built by `src.final_model.build_final_model_pipeline()`. Calibration: **none / uncalibrated**. Frozen thresholds: **0.50** (default audit) and **0.19** (development operating point). The approved preprocessing and exact parameters are recorded in the JSON frozen specification.

## Test Dataset

Source: `data/raw/cs-training.csv`. TRAIN: 104,998; VALIDATION: 22,486; TEST: 22,516. Test positives: 1,487; prevalence: 6.604193%.

Target `SeriousDlqin2yrs` describes serious delinquency (90 days past due or worse) in the two-year target window; it is not a bankruptcy label.

## Ranking Performance

Test ROC-AUC: **0.869663**; Average Precision: **0.416082**; Gini: **0.739325**. AP is Average Precision, not trapezoidal PR area.

## Probability Calibration Audit

Brier score: 0.047727; log loss: 0.173379; ECE: 0.005445. Mean probability: 0.067286; observed prevalence: 0.066042; absolute mean probability gap: 0.001244.

STEP 13 quantile binning requests 10 bins and produced 10 nonempty test bins. Duplicate edges are removed, identical scores remain together, and empty bins are omitted. ECE is the count-weighted absolute observed-minus-predicted bin gap. No calibrator was fitted.

## Threshold 0.50 Results

| Metric | Test value |
|---|---:|
| precision | 0.602592 |
| recall | 0.187626 |
| f1 | 0.286154 |
| specificity | 0.991250 |
| false_positive_rate | 0.008750 |
| false_negative_rate | 0.812374 |
| predicted_positive_rate | 0.020563 |
| tp | 279 |
| fp | 184 |
| tn | 20845 |
| fn | 1208 |
| accuracy | 0.938177 |

Confusion matrix `[[TN, FP], [FN, TP]]`: `[[20845, 184], [1208, 279]]`.

## Development Operating Threshold 0.19 Results

| Metric | Test value |
|---|---:|
| precision | 0.403366 |
| recall | 0.531944 |
| f1 | 0.458817 |
| specificity | 0.944363 |
| false_positive_rate | 0.055637 |
| false_negative_rate | 0.468056 |
| predicted_positive_rate | 0.087094 |
| tp | 791 |
| fp | 1170 |
| tn | 19859 |
| fn | 696 |
| accuracy | 0.917126 |

Confusion matrix `[[TN, FP], [FN, TP]]`: `[[19859, 1170], [696, 791]]`.

## Validation vs Test Generalization

| Metric | Validation | Test | Test - validation |
|---|---:|---:|---:|
| brier_score | 0.049105 | 0.047727 | -0.001378 |
| log_loss | 0.177687 | 0.173379 | -0.004308 |
| ece | 0.005417 | 0.005445 | +0.000027 |
| mean_predicted_probability | 0.068218 | 0.067286 | -0.000932 |
| prevalence | 0.065730 | 0.066042 | +0.000312 |
| mean_probability_gap | 0.002488 | 0.001244 | -0.001244 |
| roc_auc | 0.863666 | 0.869663 | +0.005996 |
| average_precision | 0.381284 | 0.416082 | +0.034798 |
| gini | 0.727333 | 0.739325 | +0.011992 |

| Threshold | Metric | Validation | Test | Test - validation |
|---|---|---:|---:|---:|
| 0.50 | precision | 0.568826 | 0.602592 | +0.033766 |
| 0.50 | recall | 0.190122 | 0.187626 | -0.002496 |
| 0.50 | f1 | 0.284990 | 0.286154 | +0.001164 |
| 0.50 | predicted_positive_rate | 0.021969 | 0.020563 | -0.001406 |
| 0.19 | precision | 0.372303 | 0.403366 | +0.031063 |
| 0.19 | recall | 0.502030 | 0.531944 | +0.029914 |
| 0.19 | f1 | 0.427542 | 0.458817 | +0.031274 |
| 0.19 | predicted_positive_rate | 0.088633 | 0.087094 | -0.001539 |

ROC-AUC changes by +0.005996 and AP by +0.034798. These are descriptive ranking differences; this single holdout does not establish statistical stability or significance. Brier changes by -0.001378, log loss by -0.004308, and ECE by +0.000027. No unestablished small/large-gap or calibration-adequacy standard is applied.

At 0.19, test recall is 53.194351%, +3.194 percentage points relative to 50%, versus validation recall 50.202977%. Precision is 40.336563%, a change of +3.106 percentage points from validation. The development recall target is not guaranteed on unseen data.

## Error Trade-offs

At 0.50, test has 184 false positives and 1,208 false negatives. At 0.19, test has 1,170 false positives and 696 false negatives. Both points were fixed before opening test; the result does not authorize threshold changes. No business cost matrix supports an optimal financial cutoff claim.

## No-Retuning Policy

No model, hyperparameter, preprocessing, calibration, threshold or feature selection decision was changed using test results. No test threshold search or alternative-model evaluation was performed. The consumed test must not inform further development decisions.

**INTERNAL TEST SET HAS BEEN CONSUMED FOR FINAL EVALUATION.**

The test is no longer an unseen holdout. No model change followed the reported results.

## Limitations

This is an internal holdout, not external or temporal validation. Exact-feature groups are not verified borrower identifiers. Repeated development use of validation may cause selection optimism. No uncertainty interval or statistical significance claim is made. ECE depends on binning, and average probability agreement alone does not prove calibration. Uncalibrated probabilities are not regulatory or production-validated PD. The operating threshold is not a production-approved policy. No serialization, SHAP, retraining on combined partitions, or downstream deployment is part of this step.

## Final Internal Evaluation Summary

The frozen XGB-01 specification passed validation reproduction and was evaluated on 22,516 internal test observations at exactly the two approved thresholds. Results are reported unchanged regardless of direction relative to validation. The frozen model and decisions remain unchanged.
