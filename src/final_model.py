"""Frozen development specification, decided before final holdout evaluation.

The CLI fits training and verifies validation only. It does not select a model,
search thresholds, calibrate, access the returned test partition or serialize.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.baseline_model import (
    DEFAULT_REPORT_DIR, TrainingValidationData, calculate_metrics,
    load_training_validation, positive_probabilities,
)
from src.imbalance_experiments import validate_xy
from src.preprocessing import OUTPUT_FEATURE_COLUMNS, build_tree_preprocessor
from src.threshold_analysis import threshold_metrics
from src.tree_model_experiments import fit_tree

FINAL_MODEL_NAME = "XGBoost Baseline / XGB-01"
RANDOM_STATE = 42
CALIBRATION_METHOD = "none"
DEFAULT_AUDIT_THRESHOLD = 0.50
DEVELOPMENT_OPERATING_THRESHOLD = 0.19
MODEL_PARAMETERS = MappingProxyType(dict(
    n_estimators=300, learning_rate=0.05, max_depth=3, min_child_weight=1,
    subsample=1.0, colsample_bytree=1.0, reg_alpha=0, reg_lambda=1,
    objective="binary:logistic", eval_metric="logloss", random_state=RANDOM_STATE,
    n_jobs=-1, tree_method="hist",
))
SOURCE_ARTIFACTS = (
    "tree_model_experiments.json", "boosting_tuning_results.json",
    "logistic_imbalance_experiments.json", "threshold_scenarios.csv", "calibration_results.json",
)


def build_final_model_pipeline() -> Pipeline:
    """Return a fresh, unfitted tree preprocessor and immutable XGB-01 specification."""
    return Pipeline([("preprocessor", build_tree_preprocessor()),
                     ("model", XGBClassifier(**MODEL_PARAMETERS))])


def get_frozen_metadata() -> dict:
    """Return independent JSON-ready configuration and pre-specified holdout policy."""
    return {
        "final_model": FINAL_MODEL_NAME,
        "preprocessing": {"builder": "build_tree_preprocessor", "module": "src.preprocessing",
                          "transformed_features": list(OUTPUT_FEATURE_COLUMNS),
                          "fit_partition": "training only", "scaling": "none"},
        "model_parameters": dict(MODEL_PARAMETERS), "calibration_method": CALIBRATION_METHOD,
        "default_audit_threshold": DEFAULT_AUDIT_THRESHOLD,
        "development_operating_threshold": DEVELOPMENT_OPERATING_THRESHOLD,
        "development_threshold_selection_rule": {
            "source": "STEP 12 validation RECALL_AT_LEAST_50 scenario",
            "rule": "Among thresholds with recall >= 0.50, choose highest precision; ties: higher F1, then higher threshold.",
            "historical_grid": {"start": .05, "stop": .95, "step": .01},
            "status": "already selected and frozen; no search in this module",
            "positive_rule": "score >= threshold",
        },
        "final_test_metrics_to_report": {
            "threshold_independent": ["roc_auc", "average_precision", "brier_score", "log_loss", "gini"],
            "calibration_audit": ["ece", "mean_predicted_probability", "observed_test_prevalence", "reliability_bins"],
            "classification_thresholds": [DEFAULT_AUDIT_THRESHOLD, DEVELOPMENT_OPERATING_THRESHOLD],
            "classification_metrics": ["precision", "recall", "f1", "accuracy", "specificity", "false_positive_rate",
                                       "false_negative_rate", "predicted_positive_rate", "confusion_matrix"],
            "confusion_matrix_order": [["tn", "fp"], ["fn", "tp"]],
            "gini_definition": "2 * ROC-AUC - 1",
            "ece_definition": "sum(bin_count / total_count * abs(observed_positive_rate - mean_predicted_probability))",
            "reliability_definition": {"implementation": "src.calibration_analysis.reliability_bins",
                                       "n_bins": 10, "strategy": "quantile",
                                       "duplicate_policy": "remove duplicate edges; do not split identical scores; omit empty bins"},
        },
        "test_lock_policy": {
            "step_14": "locked; no test evaluation, labels, prevalence or distributions inspected by this module",
            "step_15": "evaluate only this frozen model as the final internal holdout evaluation",
            "fit_partition": "training only; do not combine training and validation for this frozen evaluation",
            "forbidden_after_test": ["model reselection", "hyperparameter changes", "calibration changes",
                                     "preprocessing changes", "threshold changes or searches", "test-driven feature selection",
                                     "adding SMOTE or class weighting", "switching to LightGBM based on test results"],
            "performance_shortfall": "Report honestly even if test performance is lower than validation; do not retune.",
        },
        "random_state": RANDOM_STATE,
        "limitations": ["Selected from development evidence, not objectively or production best.",
                        "Repeated development use of validation can cause selection optimism.",
                        "No business cost matrix supports a bank-optimal threshold claim.",
                        "Calibration has only been audited internally on validation; scores are not regulatory or production-validated PD.",
                        "Development operating threshold 0.19 is neither a production threshold nor a regulatory cutoff.",
                        "STEP 14 contains a future test evaluation policy, not observed test metrics."],
    }


def build_selection_artifact(report_dir: Path = DEFAULT_REPORT_DIR) -> dict:
    """Attach saved development evidence without making any metric-driven choice."""
    report_dir = Path(report_dir)
    def read(name: str) -> dict:
        return json.loads((report_dir / name).read_text(encoding="utf-8"))
    tree = read("tree_model_experiments.json")["tree_experiments"]["xgboost"]
    boosting = read("boosting_tuning_results.json")["selected"]
    logistic = read("logistic_imbalance_experiments.json")["experiments"]["balanced"]
    scenarios = pd.read_csv(report_dir / "threshold_scenarios.csv")
    points = []
    for scenario, threshold in [("DEFAULT", DEFAULT_AUDIT_THRESHOLD), ("RECALL_AT_LEAST_50", DEVELOPMENT_OPERATING_THRESHOLD)]:
        selected = scenarios.loc[scenarios.model.eq("XGBoost Baseline") & scenarios.scenario.eq(scenario)]
        if len(selected) != 1 or selected.iloc[0].status != "available" or selected.iloc[0].threshold != threshold:
            raise ValueError("Saved threshold evidence differs from the frozen decision.")
        row = selected.iloc[0]
        points.append({"scenario": scenario, **{k: float(row[k]) for k in (
            "threshold", "precision", "recall", "f1", "specificity", "false_positive_rate", "false_negative_rate", "predicted_positive_rate")},
                       **{k: int(row[k]) for k in ("tp", "fp", "tn", "fn")}})
    ranking = {}
    for name, entry in [("XGBoost Baseline", tree), ("LightGBM LGBM-02", boosting["lightgbm"]),
                        ("XGBoost XGB-07 (historical)", boosting["xgboost"]), ("Logistic Balanced", logistic)]:
        ranking[name] = {"train_cv": entry["cv"]["summary"], "validation": entry["validation"]}
    artifact = get_frozen_metadata()
    artifact["selection_evidence"] = {
        "ranking": ranking, "threshold_tradeoffs": points,
        "xgboost_calibration": [r for r in read("calibration_results.json")["metrics"] if r["model"] == "XGBoost Baseline"],
        "decision_basis": "User-approved development decision fixed before test; no automatic model selection.",
        "rationale": ["Baseline XGBoost exceeds LGBM-02 on train CV AP, the primary development selection metric.",
                      "LGBM-02 is marginally higher on validation AP/AUC; the observed differences are small.",
                      "XGB-07 tuning adds only a small AP improvement and slightly lowers validation AUC; retain the baseline.",
                      "XGBoost sigmoid worsens probability metrics; isotonic gains are too small to justify an extra calibration layer.",
                      "Prefer the fixed baseline without an additional tuned configuration or calibration ensemble; no superiority claim is made."],
    }
    artifact["source_artifacts_sha256"] = {
        f"reports/{name}": hashlib.sha256((report_dir / name).read_bytes()).hexdigest() for name in SOURCE_ARTIFACTS}
    return artifact


def verify_validation(data: TrainingValidationData, artifact: dict) -> dict:
    """Fit TRAIN only, then check frozen VALIDATION metrics; never expose test."""
    validate_xy(data.X_train, data.y_train)
    validate_xy(data.X_validation, data.y_validation)
    pipeline = fit_tree(build_final_model_pipeline(), data.X_train, data.y_train)
    names = pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
    if names != OUTPUT_FEATURE_COLUMNS or pipeline.named_steps["model"].n_features_in_ != 13:
        raise ValueError("Frozen model must use the 13 approved transformed features.")
    probabilities = positive_probabilities(pipeline, data.X_validation)
    actual = calculate_metrics(data.y_validation, probabilities)
    reference = artifact["selection_evidence"]["ranking"]["XGBoost Baseline"]["validation"]
    for key in ("roc_auc", "average_precision", "precision", "recall", "f1", "accuracy", "gini", "predicted_positive_rate"):
        if not np.isclose(actual[key], reference[key], rtol=1e-6, atol=1e-8):
            raise ValueError(f"Frozen validation reference mismatch: {key}")
    if actual["rows"] != reference["rows"] or actual["confusion_matrix"] != reference["confusion_matrix"]:
        raise ValueError("Frozen validation row count or default confusion matrix mismatch.")
    operating_points = []
    for reference_point in artifact["selection_evidence"]["threshold_tradeoffs"]:
        threshold = reference_point["threshold"]
        if threshold not in (DEFAULT_AUDIT_THRESHOLD, DEVELOPMENT_OPERATING_THRESHOLD):
            raise ValueError("Verification only permits the two frozen thresholds.")
        point = threshold_metrics(data.y_validation, probabilities, threshold)
        for key in ("precision", "recall", "f1", "specificity", "false_positive_rate", "false_negative_rate", "predicted_positive_rate"):
            if not np.isclose(point[key], reference_point[key], rtol=1e-6, atol=1e-8):
                raise ValueError(f"Frozen validation reference mismatch at {threshold}: {key}")
        if any(point[k] != reference_point[k] for k in ("tp", "fp", "tn", "fn")):
            raise ValueError("Frozen operating-point confusion counts mismatch.")
        operating_points.append(point)
    return {"status": "PASS", "fit_rows": len(data.X_train), "validation_rows": len(data.X_validation),
            "roc_auc": actual["roc_auc"], "average_precision": actual["average_precision"],
            "threshold_checks": operating_points, "number_of_transformed_features": 13,
            "tolerance": {"rtol": 1e-6, "atol": 1e-8}}


def render_selection_report(artifact: dict) -> str:
    """Professional English decision record derived from saved measured evidence."""
    evidence = artifact["selection_evidence"]
    ranking_rows = []
    for name, row in evidence["ranking"].items():
        ranking_rows.append(f"| {name} | {row['train_cv']['average_precision']['mean']:.6f} | {row['train_cv']['roc_auc']['mean']:.6f} | {row['validation']['average_precision']:.6f} | {row['validation']['roc_auc']:.6f} |")
    points = evidence["threshold_tradeoffs"]
    threshold_rows = [f"| {p['threshold']:.2f} | {p['precision']:.6f} | {p['recall']:.6f} | {p['f1']:.6f} | {p['tp']} | {p['fp']} | {p['tn']} | {p['fn']} |" for p in points]
    calibration_rows = [f"| {r['calibration_method']} | {r['brier_score']:.6f} | {r['log_loss']:.6f} | {r['ece']:.6f} | {r['mean_predicted_probability']:.6f} |" for r in evidence["xgboost_calibration"]]
    return "\n".join([
        "# Final Development Model Selection", "", "## Selection Scope", "",
        "**Decision:** freeze XGBoost Baseline / XGB-01, without calibration, before opening the final internal test holdout. This is a user-approved development decision, not an automated metric winner selection.", "",
        "**Observed evidence:** all results below come from TRAIN CV and the fixed VALIDATION partition. No test metrics are present. Source artifact SHA-256 hashes are recorded in the companion JSON.", "",
        "## Candidate Models", "",
        "The current candidates are XGBoost Baseline, tuned LightGBM LGBM-02, and Logistic Balanced as a linear reference. Historical XGB-07, Random Forest, SMOTE and other experiment artifacts remain unchanged.", "",
        "## Ranking Evidence", "", "| Candidate | Train CV AP | Train CV ROC-AUC | Validation AP | Validation ROC-AUC |",
        "|---|---:|---:|---:|---:|", *ranking_rows, "",
        "**Rationale:** AP was the primary development selection metric. XGBoost Baseline is slightly ahead of LGBM-02 in TRAIN CV; LGBM-02 is slightly ahead on validation. These small observed differences do not establish statistical or production superiority. XGB-07 improves AP only slightly while lowering validation ROC-AUC; the simpler baseline decision is retained.", "",
        "## Threshold Trade-offs", "", "| Threshold | Precision | Recall | F1 | TP | FP | TN | FN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|", *threshold_rows, "",
        "Lowering the threshold from 0.50 to 0.19 catches more positives while increasing false positives. No financial cost matrix is available, so no bank-optimal threshold is claimed.", "",
        "## Calibration Evidence", "", "| XGBoost method | Brier | Log loss | ECE | Mean probability |",
        "|---|---:|---:|---:|---:|", *calibration_rows, "",
        f"Validation prevalence: {evidence['xgboost_calibration'][0]['prevalence']:.6f}. Mean probability alone is insufficient to establish calibration quality. STEP 13 ECE uses up to 10 quantile bins and is binning-dependent. Calibrated alternatives were five-fold ensembles, so their differences also include ensembling effects.", "",
        "## Selected Model", "", f"**Frozen selection:** {FINAL_MODEL_NAME}. Pipeline: `build_tree_preprocessor()` followed by `XGBClassifier`. No scaling, class weighting, SMOTE or automatic model selection.", "",
        "```json", json.dumps(dict(MODEL_PARAMETERS), indent=2), "```", "",
        "## Selected Calibration Strategy", "", "**Decision:** `CALIBRATION_METHOD = \"none\"`. Sigmoid worsens Brier, log loss and ECE. Isotonic improves Brier/log loss only marginally; the gain does not justify an additional calibration ensemble for this development decision. Retain uncalibrated scores.", "",
        "## Frozen Development Threshold", "", "`DEFAULT_AUDIT_THRESHOLD = 0.50` and `DEVELOPMENT_OPERATING_THRESHOLD = 0.19`.", "",
        "The 0.19 point was selected in STEP 12 from validation under RECALL_AT_LEAST_50: maximize precision among grid thresholds with recall >= 0.50; ties use higher F1, then higher threshold. The historical grid was 0.05–0.95 in steps of 0.01. This step does not repeat the search. The prediction rule is score >= threshold.", "",
        "The official name is **development operating threshold**. It is a pre-specified operating point for final internal holdout evaluation, not a production threshold or regulatory cutoff.", "",
        "## Final Test Evaluation Plan", "",
        "STEP 15 may evaluate only this frozen model, fitted on TRAIN with the approved preprocessing. Do not refit on TRAIN + VALIDATION for this evaluation.", "",
        "- Ranking/probability metrics: ROC-AUC, Average Precision, Brier score, log loss and Gini (2 × ROC-AUC − 1).",
        "- Calibration audit: ECE, mean predicted probability, observed test prevalence, and reliability bins using the same STEP 13 quantile definition (10 requested bins, duplicate edges removed, identical scores kept together, empty bins omitted).",
        "- At exactly 0.50 and 0.19: precision, recall, F1, accuracy, specificity, FPR, FNR, predicted positive rate and confusion matrix ordered [[TN, FP], [FN, TP]].",
        "- No threshold search, alternative-model comparison or calibration-method selection on test.", "",
        "## Test Lock Policy", "",
        "Test remains locked throughout STEP 14. The reusable splitter may execute its existing integrity checks, but this module only receives TRAIN and VALIDATION and never accesses the returned test target.", "",
        "After STEP 15 opens test, do not change model, hyperparameters, calibration, preprocessing, thresholds or selected features in response to test results. Do not add SMOTE/class weights or switch to LightGBM because of test performance. Test is the final internal holdout; report lower performance honestly without retuning.", "",
        "## Limitations", "",
        "Development decisions reused validation across experiments, so selection optimism remains possible. Small candidate differences are not evidence of significance. Group-aware splitting reduces identical-feature leakage but does not demonstrate external, temporal or production generalization. Calibration quality has only been assessed internally. No business loss function, deployment validation or regulatory assessment has been established.", "",
        "## What This Model Is Not", "",
        "This is not an objectively best model, a production-approved system, a regulatory model, or a source of regulatory/production-validated PD. The operating point is not an optimal bank threshold. Model serialization, SHAP and test evaluation are outside STEP 14.", "",
        "## Validation Reproduction", "", "```json", json.dumps(artifact.get("validation_reproduction", {"status": "not run"}), indent=2), "```", "",
    ])


def main() -> None:
    artifact = build_selection_artifact()
    artifact["validation_reproduction"] = verify_validation(load_training_validation(), artifact)
    (DEFAULT_REPORT_DIR / "final_model_selection.json").write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (DEFAULT_REPORT_DIR / "final_model_selection.md").write_text(render_selection_report(artifact), encoding="utf-8")
    print(json.dumps(artifact["validation_reproduction"], indent=2))
    print(f"Frozen: {FINAL_MODEL_NAME}; calibration={CALIBRATION_METHOD}; thresholds=0.50, 0.19")
    print("Test remains locked. STEP 15 evaluates only this specification; no post-test retuning.")


if __name__ == "__main__":
    main()
