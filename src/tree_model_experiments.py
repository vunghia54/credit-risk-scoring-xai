"""Untuned tree baselines on train-only grouped CV and fixed validation data.

Run ``python -m src.tree_model_experiments`` for the heavy integration experiment.
No test predictions, early stopping, imbalance changes, tuning or serialization.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.baseline_model import (
    DEFAULT_REPORT_DIR, THRESHOLD, build_baseline_pipeline, calculate_metrics,
    fit_baseline, positive_probabilities,
)
from src.data_split import RANDOM_STATE, load_data, split_data
from src.imbalance_experiments import ExperimentData, fit_checked, make_cv_folds, validate_xy
from src.preprocessing import OUTPUT_FEATURE_COLUMNS, build_tree_preprocessor, validate_transformed

TREE_NAMES = ("random_forest", "xgboost", "lightgbm")
LOGISTIC_NAMES = ("logistic_baseline", "logistic_balanced")
DISPLAY_NAMES = {
    "logistic_baseline": "Logistic Baseline", "logistic_balanced": "Logistic Balanced",
    "random_forest": "Random Forest", "xgboost": "XGBoost", "lightgbm": "LightGBM",
}
TREE_CONFIGURATIONS = {
    "random_forest": dict(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1, class_weight=None),
    "xgboost": dict(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=1.0,
                    colsample_bytree=1.0, objective="binary:logistic", eval_metric="logloss",
                    random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist"),
    "lightgbm": dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
                     random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1),
}
VALIDATION_METRICS = (
    "roc_auc", "average_precision", "precision", "recall", "f1", "accuracy", "gini",
    "predicted_positive_rate",
)
COMPARISON_COLUMNS = [
    "model", "cv_roc_auc_mean", "cv_roc_auc_std", "cv_average_precision_mean",
    "cv_average_precision_std", *[f"validation_{key}" for key in VALIDATION_METRICS],
]


@dataclass(frozen=True)
class TreeExperimentResult:
    """Metrics and validation predictions, without retained fitted models."""

    report: dict
    comparison: pd.DataFrame
    feature_importances: pd.DataFrame
    validation_probabilities: dict[str, np.ndarray]


def load_tree_data() -> ExperimentData:
    """Expose train/validation and existing raw train groups; ignore test.

    The existing splitter still executes its approved integrity checks. This
    experiment never accesses its returned test partition or its labels.
    """
    splits = split_data(load_data())
    return ExperimentData(
        splits.train.X, splits.train.y, splits.train.groups,
        splits.validation.X, splits.validation.y,
    )


def build_tree_experiments() -> dict[str, Pipeline]:
    """Fresh fixed configurations with the unscaled tree preprocessor only."""
    classes = {"random_forest": RandomForestClassifier, "xgboost": XGBClassifier, "lightgbm": LGBMClassifier}
    return {name: Pipeline([
        ("preprocessor", build_tree_preprocessor()),
        ("model", classes[name](**TREE_CONFIGURATIONS[name])),
    ]) for name in TREE_NAMES}


def fit_tree(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Fit only the supplied fitting rows; no eval_set or early stopping."""
    validate_xy(X, y)
    if list(pipeline.named_steps) != ["preprocessor", "model"]:
        raise ValueError("Expected preprocessor + model pipeline.")
    pipeline.fit(X, y)
    return pipeline


def importance_audit(name: str, pipeline: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Check the named 13-feature model input and finite native importances.

    Native definitions differ: RF impurity, XGBoost gain, LightGBM split counts.
    Values are for within-model sanity checks, not causal or cross-model effects.
    """
    preprocessor, model = pipeline.named_steps["preprocessor"], pipeline.named_steps["model"]
    transformed = preprocessor.transform(X)
    validate_transformed(X, transformed)
    names = preprocessor.get_feature_names_out().tolist()
    if names != OUTPUT_FEATURE_COLUMNS or model.n_features_in_ != 13:
        raise ValueError("Tree model must receive exactly 13 approved transformed features.")
    model_names = getattr(model, "feature_names_in_", None)
    if model_names is not None and list(model_names) != names:
        raise ValueError("Tree model feature-name mapping differs from preprocessing.")
    values = np.asarray(model.feature_importances_, dtype=float)
    if values.shape != (13,) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Expected 13 finite nonnegative native feature importances.")
    if name == "random_forest" and not np.isclose(values.sum(), 1):
        raise ValueError("Random Forest importances must sum approximately to one.")
    audit = pd.DataFrame({"model": DISPLAY_NAMES[name], "feature": names, "importance": values})
    audit = audit.sort_values("importance", ascending=False, kind="stable").reset_index(drop=True)
    audit["rank"] = np.arange(1, 14)
    return audit


def tree_cv(pipeline: Pipeline, data: ExperimentData, folds: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    """Fit fresh preprocessing/models per shared fitting fold, evaluate held-out fold."""
    records = []
    for number, (fitting, evaluation) in enumerate(folds, 1):
        overlap = np.intersect1d(data.groups_train.iloc[fitting], data.groups_train.iloc[evaluation]).size
        if overlap:
            raise ValueError("CV group overlap detected.")
        started = perf_counter()
        fitted = fit_tree(clone(pipeline), data.X_train.iloc[fitting], data.y_train.iloc[fitting])
        probabilities = positive_probabilities(fitted, data.X_train.iloc[evaluation])
        metrics = calculate_metrics(data.y_train.iloc[evaluation], probabilities)
        records.append({
            "fold": number, "fitting_rows": len(fitting), "evaluation_rows": len(evaluation),
            "group_overlap": int(overlap), "roc_auc": metrics["roc_auc"],
            "average_precision": metrics["average_precision"], "runtime_seconds": perf_counter() - started,
        })
    return {
        "folds": records, "std_ddof": 0,
        "summary": {key: {"mean": float(np.mean([record[key] for record in records])),
                          "std": float(np.std([record[key] for record in records], ddof=0))}
                    for key in ("roc_auc", "average_precision")},
    }


def validate_reference(entry: dict, expected_rows: int) -> None:
    """Reject missing, invalid or inconsistent saved Logistic metrics before reuse."""
    try:
        metrics = entry["validation"]
        if metrics["rows"] != expected_rows:
            raise ValueError("Logistic reference validation row count differs.")
        for key in (*VALIDATION_METRICS, "positive_prevalence"):
            value = metrics[key]
            if not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f"Invalid Logistic reference metric: {key}.")
            if not (-1 if key == "gini" else 0) <= value <= 1:
                raise ValueError(f"Out-of-range Logistic reference metric: {key}.")
        if not np.isclose(metrics["gini"], 2 * metrics["roc_auc"] - 1):
            raise ValueError("Logistic reference Gini is inconsistent.")
        matrix = np.asarray(metrics["confusion_matrix"], dtype=float)
        if matrix.shape != (2, 2) or not np.isfinite(matrix).all() or (matrix < 0).any() or not np.equal(matrix, np.floor(matrix)).all():
            raise ValueError("Invalid Logistic reference confusion matrix.")
        if matrix.sum() != expected_rows:
            raise ValueError("Logistic reference confusion matrix does not cover validation.")
        tn, fp, fn, tp = matrix.ravel()
        expected = {
            "positive_prevalence": (tp + fn) / expected_rows,
            "predicted_positive_rate": (tp + fp) / expected_rows,
            "accuracy": (tn + tp) / expected_rows,
            "precision": tp / (tp + fp) if tp + fp else 0,
            "recall": tp / (tp + fn) if tp + fn else 0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
        }
        if any(not np.isclose(metrics[key], value) for key, value in expected.items()):
            raise ValueError("Logistic reference metrics disagree with confusion matrix.")
        cv = entry["cv"]
        if cv["std_ddof"] != 0 or len(cv["folds"]) != 5:
            raise ValueError("Logistic reference must use five folds and ddof=0.")
        for key in ("roc_auc", "average_precision"):
            values = np.asarray([fold[key] for fold in cv["folds"]], dtype=float)
            if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                raise ValueError("Invalid Logistic reference fold scores.")
            for stat, actual in (("mean", values.mean()), ("std", values.std(ddof=0))):
                saved = cv["summary"][key][stat]
                if not np.isfinite(saved) or not np.isclose(saved, actual):
                    raise ValueError("Logistic reference CV summary is inconsistent.")
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Invalid Logistic reference schema: {exc}") from exc


def load_logistic_references(expected_rows: int, report_dir: Path = DEFAULT_REPORT_DIR) -> dict:
    """Reuse STEP 9 CV/validation results, cross-checking STEP 8 baseline."""
    report_dir = Path(report_dir)
    saved = json.loads((report_dir / "logistic_imbalance_experiments.json").read_text(encoding="utf-8"))
    baseline = json.loads((report_dir / "baseline_logistic_metrics.json").read_text(encoding="utf-8"))
    if saved.get("threshold") != THRESHOLD or baseline.get("threshold") != THRESHOLD:
        raise ValueError("Logistic reference threshold must equal 0.5.")
    if saved.get("random_state") != RANDOM_STATE or saved.get("baseline_reproduction") != "PASS":
        raise ValueError("Logistic reference provenance does not match the approved baseline.")
    try:
        references = {"logistic_baseline": saved["experiments"]["baseline"],
                      "logistic_balanced": saved["experiments"]["balanced"]}
        for name, entry in references.items():
            validate_reference(entry, expected_rows)
            config = entry["configuration"]
            if config["class_weight"] != ("balanced" if name == "logistic_balanced" else None):
                raise ValueError("Unexpected Logistic reference class weight.")
            if config["C"] != 1 or config["solver"] != "lbfgs" or config["max_iter"] != 2000:
                raise ValueError("Unexpected Logistic reference configuration.")
        verify_reference_predictions(references["logistic_baseline"]["validation"], baseline["validation"])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Invalid Logistic reference schema: {exc}") from exc
    return references


def verify_reference_predictions(actual: dict, expected: dict) -> None:
    """Fail if regenerated reference curves would contradict saved metrics."""
    for key in (*VALIDATION_METRICS, "positive_prevalence", "rows"):
        if not np.isclose(actual[key], expected[key], rtol=1e-6, atol=1e-8):
            raise ValueError(f"Logistic reference reproduction failed: {key}.")
    if actual["confusion_matrix"] != expected["confusion_matrix"]:
        raise ValueError("Logistic reference confusion matrix changed.")


def logistic_curve_predictions(data: ExperimentData, references: dict) -> dict[str, np.ndarray]:
    """Regenerate two reference curves only: saved reports lack per-row scores.

    No Logistic CV is repeated. Only train is fitted, and regenerated validation
    metrics must match the accepted saved results before curves can be plotted.
    """
    predictions = {}
    for name in LOGISTIC_NAMES:
        if name == "logistic_baseline":
            pipeline = fit_baseline(data.X_train, data.y_train)
        else:
            pipeline = build_baseline_pipeline().set_params(model__class_weight="balanced")
            pipeline = fit_checked(pipeline, data.X_train, data.y_train)
        probabilities = positive_probabilities(pipeline, data.X_validation)
        verify_reference_predictions(calculate_metrics(data.y_validation, probabilities), references[name]["validation"])
        predictions[name] = probabilities
    return predictions


def comparison_table(entries: dict) -> pd.DataFrame:
    """Build the five-candidate table exclusively from measured/reused results."""
    rows = []
    for name in (*LOGISTIC_NAMES, *TREE_NAMES):
        entry = entries[name]
        row = {"model": DISPLAY_NAMES[name]}
        for key in ("roc_auc", "average_precision"):
            for stat in ("mean", "std"):
                row[f"cv_{key}_{stat}"] = entry["cv"]["summary"][key][stat]
        row.update({f"validation_{key}": entry["validation"][key] for key in VALIDATION_METRICS})
        rows.append(row)
    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)


def run_tree_experiments(data: ExperimentData, progress: Callable[[str], None] | None = None) -> TreeExperimentResult:
    """Run shared train-only CV, full-train tree fits and fixed validation evaluation."""
    started = perf_counter()
    validate_xy(data.X_validation, data.y_validation)
    references = load_logistic_references(len(data.X_validation))
    predictions = logistic_curve_predictions(data, references)
    folds = make_cv_folds(data)
    entries, audits = {}, []
    for name, prototype in build_tree_experiments().items():
        if progress:
            progress(f"{DISPLAY_NAMES[name]}: 5-fold train-only CV started")
        model_started = perf_counter()
        cv = tree_cv(prototype, data, folds)
        fit_started = perf_counter()
        fitted = fit_tree(clone(prototype), data.X_train, data.y_train)
        fit_seconds = perf_counter() - fit_started
        audit = importance_audit(name, fitted, data.X_train)
        probabilities = positive_probabilities(fitted, data.X_validation)
        model = fitted.named_steps["model"]
        if name == "random_forest":
            actual_trees = len(model.estimators_)
        elif name == "xgboost":
            actual_trees = model.get_booster().num_boosted_rounds()
        else:
            actual_trees = model.booster_.current_iteration()
        if actual_trees != 300:
            raise ValueError(f"{name}: expected 300 estimators/boosting rounds; got {actual_trees}.")
        entries[name] = {
            "configuration": TREE_CONFIGURATIONS[name], "other_parameters": "library defaults, untuned",
            "preprocessor": "build_tree_preprocessor", "number_of_features": 13,
            "cv": cv, "validation": calculate_metrics(data.y_validation, probabilities),
            "full_train_fit_seconds": fit_seconds, "runtime_seconds": perf_counter() - model_started,
            "fitted_estimators_or_rounds": int(actual_trees),
            "feature_importances": audit.to_dict(orient="records"),
        }
        predictions[name] = probabilities
        audits.append(audit)
        if progress:
            progress(f"{DISPLAY_NAMES[name]}: CV and validation complete ({entries[name]['runtime_seconds']:.1f}s)")
    all_entries = {**references, **entries}
    report = {
        "random_state": RANDOM_STATE, "threshold": THRESHOLD,
        "cv_design": {"splitter": "StratifiedGroupKFold", "n_splits": 5, "shuffle": True,
                      "std_ddof": 0, "shared_folds": True, "groups": "raw training feature-group metadata"},
        "tree_experiments": entries, "logistic_references": references,
        "logistic_curve_reproduction": "PASS", "runtime_seconds": perf_counter() - started,
        "notes": [
            "Untuned baselines; each fold fits its own tree preprocessing without StandardScaler.",
            "Same raw split, preprocessing policy and 0.5 threshold for the three tree models.",
            "Logistic metrics/CV reused from STEP 8/9. Two train-only Logistic fits regenerate validated curves; Logistic CV is not rerun here.",
            "No test evaluation, imbalance modification, early stopping, tuning, threshold selection, calibration or serialization.",
            "Probabilities are model scores, not calibrated, regulatory or production PD estimates.",
            "Native importances are audit only: RF impurity, XGBoost gain, LightGBM split counts; not causal, comparable across algorithms or a substitute for SHAP.",
            "CV standard deviation is fold variability (ddof=0), not a confidence interval. No final model is selected.",
        ],
    }
    return TreeExperimentResult(report, comparison_table(all_entries), pd.concat(audits, ignore_index=True), predictions)


def save_reports(result: TreeExperimentResult, y_validation: pd.Series, report_dir: Path = DEFAULT_REPORT_DIR) -> None:
    """Save measured JSON/CSV audit reports and validation-only figures."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from sklearn.metrics import precision_recall_curve, roc_curve

    report_dir = Path(report_dir)
    figures = report_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    (report_dir / "tree_model_experiments.json").write_text(json.dumps(result.report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    result.comparison.to_csv(report_dir / "model_baseline_comparison.csv", index=False)
    result.feature_importances.to_csv(report_dir / "tree_feature_importance_audit.csv", index=False)
    entries = {**result.report["logistic_references"], **result.report["tree_experiments"]}
    colors = ["#64748b", "#cc7a16", "#34834a", "#176b91", "#9053a0"]
    for kind in ("roc", "pr"):
        figure = Figure(figsize=(8.5, 6), layout="constrained")
        FigureCanvasAgg(figure)
        ax = figure.subplots()
        for (name, probabilities), color in zip(result.validation_probabilities.items(), colors):
            metrics = entries[name]["validation"]
            if kind == "roc":
                x, y, _ = roc_curve(y_validation, probabilities)
                label = f"{DISPLAY_NAMES[name]} (AUC {metrics['roc_auc']:.4f})"
            else:
                y, x, _ = precision_recall_curve(y_validation, probabilities)
                label = f"{DISPLAY_NAMES[name]} (AP {metrics['average_precision']:.4f})"
            ax.plot(x, y, color=color, linewidth=1.8, label=label)
        if kind == "roc":
            ax.plot([0, 1], [0, 1], "--", color="gray", label="Random ranking")
        else:
            ax.axhline(y_validation.mean(), linestyle="--", color="gray", label=f"Prevalence {y_validation.mean():.4f}")
        ax.set(xlabel="False positive rate" if kind == "roc" else "Recall",
               ylabel="True positive rate (recall)" if kind == "roc" else "Precision",
               title="Model Baselines — Validation " + ("ROC" if kind == "roc" else "Precision–Recall"),
               xlim=(0, 1), ylim=(0, 1.02))
        ax.legend(loc="lower right" if kind == "roc" else "upper right", fontsize=9)
        ax.grid(alpha=.2)
        figure.savefig(figures / f"tree_models_{kind}_comparison.png", dpi=160)
    figure = Figure(figsize=(11, 6), layout="constrained")
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    keys = ["roc_auc", "average_precision", "precision", "recall", "f1"]
    for number, (name, color) in enumerate(zip((*LOGISTIC_NAMES, *TREE_NAMES), colors)):
        values = [entries[name]["validation"][key] for key in keys]
        bars = ax.bar(np.arange(len(keys)) + (number - 2) * .16, values, .16, color=color, label=DISPLAY_NAMES[name])
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=3)
    ax.set(xticks=np.arange(len(keys)), xticklabels=["ROC-AUC", "Average Precision", "Precision", "Recall", "F1"],
           ylim=(0, 1.12), ylabel="Score", title="Model Baselines — Validation Comparison\nPrecision, recall and F1 at threshold 0.5")
    ax.legend(loc="upper center", ncol=5, fontsize=9)
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    figure.savefig(figures / "model_baseline_validation_comparison.png", dpi=160)


def main() -> None:
    data = load_tree_data()
    result = run_tree_experiments(data, progress=lambda message: print(message, flush=True))
    save_reports(result, data.y_validation)
    print(result.comparison.to_string(index=False))
    print(f"Full experiment runtime (excluding report export): {result.report['runtime_seconds']:.2f}s")


if __name__ == "__main__":
    main()
