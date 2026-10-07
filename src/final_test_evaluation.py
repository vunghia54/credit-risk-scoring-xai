"""One frozen internal holdout evaluation after development reproduction.

Run explicitly with ``python -m src.final_test_evaluation``. Importing this
module never loads data, fits a model or evaluates the holdout. The CLI refuses
to overwrite an existing final evaluation; it is not a development experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from src import final_model as frozen
from src.baseline_model import DEFAULT_REPORT_DIR, positive_probabilities
from src.calibration_analysis import calibration_metrics
from src.data_split import DEFAULT_DATA_PATH, DatasetSplits, load_data, split_data
from src.preprocessing import OUTPUT_FEATURE_COLUMNS
from src.threshold_analysis import threshold_metrics
from src.tree_model_experiments import fit_tree

THRESHOLDS = (frozen.DEFAULT_AUDIT_THRESHOLD, frozen.DEVELOPMENT_OPERATING_THRESHOLD)
CONSUMPTION_STATEMENT = "INTERNAL TEST SET HAS BEEN CONSUMED FOR FINAL EVALUATION."
NO_RETUNING_STATEMENT = (
    "No model, hyperparameter, preprocessing, calibration, threshold or feature "
    "selection decision was changed using test results. No test threshold search "
    "or alternative-model evaluation was performed. The consumed test must not "
    "inform further development decisions."
)
RAW_SHA256 = "1bd46da486a5708c58c7b01a034fae2a13b327f6f7b62ea7ba4fe3b5824b24ac"
REPORT_FILES = (
    "final_test_results.json", "final_test_metrics.csv",
    "final_test_threshold_metrics.csv", "final_test_calibration_bins.csv",
    "final_test_evaluation.md", "figures/final_test_roc_curve.png",
    "figures/final_test_pr_curve.png", "figures/final_test_calibration_curve.png",
    "figures/final_test_confusion_matrices.png", "figures/validation_vs_test_comparison.png",
)


@dataclass(frozen=True)
class EvaluationResult:
    """Measured report plus in-memory test predictions used only for figures."""

    report: dict
    test_bins: pd.DataFrame
    test_labels: np.ndarray
    test_probabilities: np.ndarray


def verify_frozen_specification(reference: dict) -> dict:
    """Reject configuration drift against the committed STEP 14 specification."""
    metadata = frozen.get_frozen_metadata()
    if any(reference.get(key) != value for key, value in metadata.items()):
        raise ValueError("Frozen specification differs from the STEP 14 artifact.")
    if (metadata["final_model"] != "XGBoost Baseline / XGB-01"
            or metadata["calibration_method"] != "none"
            or THRESHOLDS != (.50, .19) or metadata["random_state"] != 42):
        raise ValueError("The approved final model, calibration or thresholds changed.")
    return metadata


def fixed_threshold_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Evaluate only an approved frozen operating point; positive means >=."""
    if threshold not in THRESHOLDS:
        raise ValueError("Only frozen thresholds 0.50 and 0.19 are permitted.")
    point = threshold_metrics(y, scores, threshold)
    point.pop("balanced_accuracy")
    total = sum(point[key] for key in ("tn", "fp", "fn", "tp"))
    point["accuracy"] = (point["tn"] + point["tp"]) / total
    point["confusion_matrix"] = [[point["tn"], point["fp"]], [point["fn"], point["tp"]]]
    return point


def evaluate_scores(y: np.ndarray, scores: np.ndarray) -> tuple[dict, pd.DataFrame]:
    """Compute prescribed ranking/probability metrics with STEP 13 quantile ECE."""
    metrics, bins = calibration_metrics(y, scores)
    metrics["gini"] = 2 * metrics["roc_auc"] - 1
    return metrics, bins


def check_validation_reproduction(metrics: dict, points: list[dict], rows: int,
                                 reference: dict) -> dict:
    """Fail closed on validation drift before the caller accesses test data."""
    evidence = reference["selection_evidence"]
    ranking = evidence["ranking"]["XGBoost Baseline"]["validation"]
    if rows != ranking["rows"]:
        raise ValueError("Validation row count differs from the frozen reference.")
    probability_reference = next(
        r for r in evidence["xgboost_calibration"] if r["calibration_method"] == "uncalibrated"
    )
    for key, expected in {**{k: ranking[k] for k in ("roc_auc", "average_precision")},
                          **{k: probability_reference[k] for k in (
                              "brier_score", "log_loss", "ece", "prevalence",
                              "mean_predicted_probability")}}.items():
        if not np.isclose(metrics[key], expected, rtol=1e-6, atol=1e-8):
            raise ValueError(f"Validation reproduction failed: {key}")
    references = {p["threshold"]: p for p in evidence["threshold_tradeoffs"]}
    if set(references) != set(THRESHOLDS) or [p["threshold"] for p in points] != list(THRESHOLDS):
        raise ValueError("Validation must reproduce exactly the two frozen thresholds.")
    for point in points:
        expected = references[point["threshold"]]
        for key in ("precision", "recall", "f1", "specificity", "false_positive_rate",
                    "false_negative_rate", "predicted_positive_rate"):
            if not np.isclose(point[key], expected[key], rtol=1e-6, atol=1e-8):
                raise ValueError(f"Validation reproduction failed: {point['threshold']} / {key}")
        if any(point[k] != expected[k] for k in ("tp", "fp", "tn", "fn")):
            raise ValueError("Validation confusion counts differ from the frozen reference.")
    return {"status": "PASS", "rows": rows, "metrics": metrics, "threshold_metrics": points,
            "tolerance": {"rtol": 1e-6, "atol": 1e-8}}


def generalization_gaps(validation: dict, test: dict) -> dict:
    """Return test minus validation for matching numeric metric dictionaries."""
    if validation.keys() != test.keys():
        raise ValueError("Generalization metrics must have identical keys.")
    return {key: float(test[key] - validation[key]) for key in validation}


def run_evaluation(splits: DatasetSplits, reference: dict) -> EvaluationResult:
    """Fit TRAIN once, gate on VALIDATION, then score TEST once without refitting.

    The approved splitter checks all partition integrity beforehand. Within this
    workflow, neither test features nor labels are accessed until the validation
    reproduction gate passes. The same fitted pipeline produces both predictions.
    """
    specification = verify_frozen_specification(reference)
    pipeline = frozen.build_final_model_pipeline()
    actual_parameters = pipeline.named_steps["model"].get_params()
    if any(actual_parameters[k] != v for k, v in specification["model_parameters"].items()):
        raise ValueError("Pipeline parameters differ from the frozen specification.")
    pipeline = fit_tree(pipeline, splits.train.X, splits.train.y)
    if (pipeline.named_steps["preprocessor"].get_feature_names_out().tolist() != OUTPUT_FEATURE_COLUMNS
            or pipeline.named_steps["model"].n_features_in_ != 13):
        raise ValueError("Frozen preprocessing must produce the 13 approved features.")
    validation_scores = positive_probabilities(pipeline, splits.validation.X)
    validation, _ = evaluate_scores(splits.validation.y, validation_scores)
    validation_points = [fixed_threshold_metrics(splits.validation.y, validation_scores, t) for t in THRESHOLDS]
    reproduction = check_validation_reproduction(validation, validation_points, len(splits.validation.X), reference)
    print("Validation reproduction PASS; evaluating the frozen internal TEST now.", flush=True)
    test_partition = splits.test
    test_scores = positive_probabilities(pipeline, test_partition.X)
    test, bins = evaluate_scores(test_partition.y, test_scores)
    test_points = [fixed_threshold_metrics(test_partition.y, test_scores, t) for t in THRESHOLDS]
    point_keys = ("precision", "recall", "f1", "accuracy", "specificity", "false_positive_rate",
                  "false_negative_rate", "predicted_positive_rate")
    report = {
        "frozen_model_specification": specification,
        "random_state": frozen.RANDOM_STATE, "fit_partition": "training only",
        "train_rows": len(splits.train.X), "validation_rows": len(splits.validation.X),
        "test_rows": len(test_partition.X), "test_positive_count": int(np.asarray(test_partition.y).sum()),
        "validation_reproduction": reproduction,
        "validation_metrics": validation, "test_metrics": test,
        "calibration_audit": {"method": "none", "n_bins": 10, "strategy": "quantile",
                              "actual_test_bins": len(bins),
                              "implementation": "src.calibration_analysis.calibration_metrics",
                              "ece_definition": "sum(count / total_count * absolute_gap)",
                              "test_reliability_bins": bins.to_dict(orient="records")},
        "validation_threshold_metrics": validation_points, "test_threshold_metrics": test_points,
        "generalization_gaps": {
            "definition": "test minus validation", "metrics": generalization_gaps(validation, test),
            "thresholds": [{"threshold": v["threshold"], **generalization_gaps(
                {k: v[k] for k in point_keys}, {k: t[k] for k in point_keys})}
                for v, t in zip(validation_points, test_points, strict=True)],
        },
        "test_consumption_status": CONSUMPTION_STATEMENT,
        "no_retuning_statement": NO_RETUNING_STATEMENT,
    }
    if frozen.get_frozen_metadata() != specification:
        raise ValueError("Frozen configuration was mutated during evaluation.")
    return EvaluationResult(report, bins, np.asarray(test_partition.y).copy(), test_scores.copy())


def report_tables(result: EvaluationResult) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the two validation/test summary tables without stored predictions."""
    report = result.report
    metrics = pd.DataFrame([{"dataset": name, **report[f"{name}_metrics"]} for name in ("validation", "test")])
    metrics = metrics.rename(columns={"mean_predicted_probability": "mean_probability"})
    rows = []
    for index, threshold in enumerate(THRESHOLDS):
        for dataset in ("validation", "test"):
            point = report[f"{dataset}_threshold_metrics"][index]
            rows.append({"dataset": dataset, **{k: v for k, v in point.items() if k != "confusion_matrix"}})
    return metrics, pd.DataFrame(rows)


def render_report(report: dict) -> str:
    """Render factual English findings without performance adequacy cutoffs."""
    val, test = report["validation_metrics"], report["test_metrics"]
    gaps = report["generalization_gaps"]["metrics"]
    audit, operating = report["test_threshold_metrics"]
    val_operating = report["validation_threshold_metrics"][1]
    rows = [f"| {k} | {val[k]:.6f} | {test[k]:.6f} | {gaps[k]:+.6f} |" for k in val]
    point_rows = []
    for v, t in zip(report["validation_threshold_metrics"], report["test_threshold_metrics"], strict=True):
        for key in ("precision", "recall", "f1", "predicted_positive_rate"):
            point_rows.append(f"| {t['threshold']:.2f} | {key} | {v[key]:.6f} | {t[key]:.6f} | {t[key]-v[key]:+.6f} |")
    def point_text(point: dict) -> str:
        return "\n".join(["| Metric | Test value |", "|---|---:|", *[
            f"| {key} | {value:.6f} |" if isinstance(value, float) else f"| {key} | {value} |"
            for key, value in point.items() if key not in ("threshold", "confusion_matrix")],
            "", f"Confusion matrix `[[TN, FP], [FN, TP]]`: `{point['confusion_matrix']}`."])
    return "\n".join([
        "# Final Internal Test Evaluation", "", "## Frozen Evaluation Protocol", "",
        "The internal test set was opened only after STEP 14 model freeze (commit b6efa63). "
        "The pipeline was fitted once on TRAIN only; no validation or test rows were used to fit preprocessing or the estimator. "
        "Validation reproduction passed before test prediction. The same fitted pipeline evaluated both partitions.", "",
        "The approved group-aware assignment uses random_state=42 and isolates exact raw feature groups. "
        "The CLI refuses to overwrite an existing final evaluation. No threshold search was performed.", "",
        "## Final Model", "", "XGBoost Baseline / XGB-01, built by `src.final_model.build_final_model_pipeline()`. "
        "Calibration: **none / uncalibrated**. Frozen thresholds: **0.50** (default audit) and **0.19** (development operating point). "
        "The approved preprocessing and exact parameters are recorded in the JSON frozen specification.", "",
        "## Test Dataset", "",
        f"Source: `data/raw/cs-training.csv`. TRAIN: {report['train_rows']:,}; VALIDATION: {report['validation_rows']:,}; "
        f"TEST: {report['test_rows']:,}. Test positives: {report['test_positive_count']:,}; prevalence: {test['prevalence']:.6%}.", "",
        "Target `SeriousDlqin2yrs` describes serious delinquency (90 days past due or worse) in the two-year target window; it is not a bankruptcy label.", "",
        "## Ranking Performance", "",
        f"Test ROC-AUC: **{test['roc_auc']:.6f}**; Average Precision: **{test['average_precision']:.6f}**; Gini: **{test['gini']:.6f}**. "
        "AP is Average Precision, not trapezoidal PR area.", "",
        "## Probability Calibration Audit", "",
        f"Brier score: {test['brier_score']:.6f}; log loss: {test['log_loss']:.6f}; ECE: {test['ece']:.6f}. "
        f"Mean probability: {test['mean_predicted_probability']:.6f}; observed prevalence: {test['prevalence']:.6f}; "
        f"absolute mean probability gap: {test['mean_probability_gap']:.6f}.", "",
        f"STEP 13 quantile binning requests 10 bins and produced {report['calibration_audit']['actual_test_bins']} nonempty test bins. "
        "Duplicate edges are removed, identical scores remain together, and empty bins are omitted. "
        "ECE is the count-weighted absolute observed-minus-predicted bin gap. No calibrator was fitted.", "",
        "## Threshold 0.50 Results", "", point_text(audit), "",
        "## Development Operating Threshold 0.19 Results", "", point_text(operating), "",
        "## Validation vs Test Generalization", "", "| Metric | Validation | Test | Test - validation |",
        "|---|---:|---:|---:|", *rows, "",
        "| Threshold | Metric | Validation | Test | Test - validation |", "|---|---|---:|---:|---:|", *point_rows, "",
        f"ROC-AUC changes by {gaps['roc_auc']:+.6f} and AP by {gaps['average_precision']:+.6f}. "
        "These are descriptive ranking differences; this single holdout does not establish statistical stability or significance. "
        f"Brier changes by {gaps['brier_score']:+.6f}, log loss by {gaps['log_loss']:+.6f}, and ECE by {gaps['ece']:+.6f}. "
        "No unestablished small/large-gap or calibration-adequacy standard is applied.", "",
        f"At 0.19, test recall is {operating['recall']:.6%}, {(operating['recall']-.5)*100:+.3f} percentage points relative to 50%, "
        f"versus validation recall {val_operating['recall']:.6%}. Precision is {operating['precision']:.6%}, "
        f"a change of {(operating['precision']-val_operating['precision'])*100:+.3f} percentage points from validation. "
        "The development recall target is not guaranteed on unseen data.", "",
        "## Error Trade-offs", "",
        f"At 0.50, test has {audit['fp']:,} false positives and {audit['fn']:,} false negatives. "
        f"At 0.19, test has {operating['fp']:,} false positives and {operating['fn']:,} false negatives. "
        "Both points were fixed before opening test; the result does not authorize threshold changes. "
        "No business cost matrix supports an optimal financial cutoff claim.", "",
        "## No-Retuning Policy", "", NO_RETUNING_STATEMENT, "", f"**{CONSUMPTION_STATEMENT}**", "",
        "The test is no longer an unseen holdout. No model change followed the reported results.", "",
        "## Limitations", "",
        "This is an internal holdout, not external or temporal validation. Exact-feature groups are not verified borrower identifiers. "
        "Repeated development use of validation may cause selection optimism. No uncertainty interval or statistical significance claim is made. "
        "ECE depends on binning, and average probability agreement alone does not prove calibration. "
        "Uncalibrated probabilities are not regulatory or production-validated PD. The operating threshold is not a production-approved policy. "
        "No serialization, SHAP, retraining on combined partitions, or downstream deployment is part of this step.", "",
        "## Final Internal Evaluation Summary", "",
        f"The frozen XGB-01 specification passed validation reproduction and was evaluated on {report['test_rows']:,} internal test observations "
        "at exactly the two approved thresholds. Results are reported unchanged regardless of direction relative to validation. "
        "The frozen model and decisions remain unchanged.", "",
    ])


def save_figures(result: EvaluationResult, directory: Path) -> None:
    """Export only the five prescribed figures with English titles and labels."""
    directory.mkdir(parents=True, exist_ok=True)
    report, labels, scores = result.report, result.test_labels, result.test_probabilities
    test = report["test_metrics"]
    def save(fig: plt.Figure, name: str) -> None:
        fig.tight_layout()
        fig.savefig(directory / name, dpi=160)
        plt.close(fig)
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = plt.subplots(figsize=(7, 5))
        fpr, tpr, _ = roc_curve(labels, scores)
        ax.plot(fpr, tpr, label=f"XGB-01 (ROC-AUC = {test['roc_auc']:.4f})")
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Diagonal reference")
        ax.set(title="Final Internal Test: ROC Curve", xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.02))
        ax.legend(loc="lower right")
        save(fig, "final_test_roc_curve.png")
        fig, ax = plt.subplots(figsize=(7, 5))
        precision, recall, _ = precision_recall_curve(labels, scores)
        ax.plot(recall, precision, label=f"XGB-01 (Average Precision = {test['average_precision']:.4f})")
        ax.axhline(test["prevalence"], linestyle="--", color="gray", label=f"Prevalence = {test['prevalence']:.4f}")
        ax.set(title="Final Internal Test: Precision-Recall Curve", xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
        ax.legend()
        save(fig, "final_test_pr_curve.png")
        fig, ax = plt.subplots(figsize=(7, 5))
        bins = result.test_bins
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
        ax.plot(bins.mean_predicted_probability, bins.observed_positive_rate, "o-", label=f"Uncalibrated XGB-01 ({len(bins)} quantile bins)")
        ax.set(title="Final Internal Test: Reliability Curve", xlabel="Mean predicted probability", ylabel="Observed positive rate", xlim=(0, 1), ylim=(0, 1))
        ax.legend()
        save(fig, "final_test_calibration_curve.png")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        matrices = [np.asarray(p["confusion_matrix"]) for p in report["test_threshold_metrics"]]
        vmax = max(int(m.max()) for m in matrices)
        for ax, matrix, threshold in zip(axes, matrices, THRESHOLDS, strict=True):
            ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax)
            for (i, j), count in np.ndenumerate(matrix):
                ax.text(j, i, f"{count:,}", ha="center", va="center", color="white" if count > vmax / 2 else "black")
            ax.set(title=f"Frozen threshold {threshold:.2f}", xlabel="Predicted label", ylabel="True label", xticks=[0, 1], yticks=[0, 1])
        fig.suptitle("Final Internal Test: Confusion Matrices")
        save(fig, "final_test_confusion_matrices.png")
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        entries = [("ROC-AUC", "roc_auc", False), ("Average Precision", "average_precision", False),
                   ("Brier Score (lower is better)", "brier_score", False), ("F1 @ 0.19", "f1", True),
                   ("Recall @ 0.19", "recall", True), ("Precision @ 0.19", "precision", True)]
        for ax, (title, key, point) in zip(axes.flat, entries, strict=True):
            values = [report[f"{name}_threshold_metrics"][1][key] if point else report[f"{name}_metrics"][key] for name in ("validation", "test")]
            bars = ax.bar(["Validation", "Test"], values, color=["#58758e", "#d48240"])
            ax.bar_label(bars, fmt="%.4f", padding=3)
            ax.set(title=title, ylim=(0, max(values) * 1.22))
        fig.suptitle("Frozen XGB-01: Validation vs Final Internal Test")
        save(fig, "validation_vs_test_comparison.png")


def ensure_fresh_output(report_dir: Path) -> None:
    """Prevent accidental reevaluation/overwrite when any final output exists."""
    existing = [name for name in REPORT_FILES if (report_dir / name).exists()]
    if existing:
        raise FileExistsError(f"Final evaluation output already exists; do not reopen test: {existing}")


def save_reports(result: EvaluationResult, report_dir: Path = DEFAULT_REPORT_DIR) -> None:
    """Write only the approved final reports/figures; never serialize the model."""
    ensure_fresh_output(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    # Persist measured results and consumption status first, before rendering.
    (report_dir / REPORT_FILES[0]).write_text(json.dumps(result.report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    metrics, thresholds = report_tables(result)
    metrics.to_csv(report_dir / "final_test_metrics.csv", index=False)
    thresholds.to_csv(report_dir / "final_test_threshold_metrics.csv", index=False)
    result.test_bins.to_csv(report_dir / "final_test_calibration_bins.csv", index=False)
    (report_dir / "final_test_evaluation.md").write_text(render_report(result.report), encoding="utf-8")
    save_figures(result, report_dir / "figures")


def main() -> None:
    """Explicit final evaluation entry point, with pre-test drift/safety gates."""
    ensure_fresh_output(DEFAULT_REPORT_DIR)
    reference_path = DEFAULT_REPORT_DIR / "final_model_selection.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    specification = verify_frozen_specification(reference)
    print("Frozen specification verified BEFORE test access:", flush=True)
    print(json.dumps(specification, indent=2), flush=True)
    source_hash = hashlib.sha256(DEFAULT_DATA_PATH.read_bytes()).hexdigest()
    if source_hash != RAW_SHA256:
        raise ValueError("Raw dataset differs from the verified source; evaluation stopped.")
    result = run_evaluation(split_data(load_data()), reference)
    result.report["provenance"] = {
        "freeze_commit": "b6efa63", "source": "data/raw/cs-training.csv", "source_sha256": source_hash,
        "frozen_selection_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
    }
    save_reports(result)
    print(json.dumps({"validation_reproduction": result.report["validation_reproduction"]["status"],
                      "test_metrics": result.report["test_metrics"],
                      "test_threshold_metrics": result.report["test_threshold_metrics"]}, indent=2))
    print(CONSUMPTION_STATEMENT)
    print(NO_RETUNING_STATEMENT)


if __name__ == "__main__":
    main()
