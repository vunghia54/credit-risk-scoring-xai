"""Unweighted Logistic Regression baseline; evaluate train/validation only.

Run ``python -m src.baseline_model`` to write metrics and validation figures.
No test predictions, resampling, tuning, calibration or serialization occurs.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from src.data_split import RANDOM_STATE, load_data, split_data
from src.preprocessing import (
    OUTPUT_FEATURE_COLUMNS, build_linear_preprocessor, validate_transformed,
)

MAX_ITER = 2000
THRESHOLD = 0.5
DEFAULT_REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"


@dataclass(frozen=True)
class TrainingValidationData:
    """Explicit modeling boundary: no test features or labels are exposed."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_validation: pd.DataFrame
    y_validation: pd.Series


@dataclass(frozen=True)
class BaselineResult:
    """In-memory model, report and validation probabilities for plotting."""

    pipeline: Pipeline
    report: dict[str, object]
    validation_probabilities: np.ndarray


def load_training_validation() -> TrainingValidationData:
    """Reuse the approved splitter, then expose only train and validation.

    The splitter performs its existing integrity checks on all partitions.
    Modeling/evaluation code never accesses the returned test partition.
    """
    splits = split_data(load_data())
    return TrainingValidationData(
        splits.train.X, splits.train.y, splits.validation.X, splits.validation.y,
    )


def build_baseline_pipeline(max_iter: int = MAX_ITER) -> Pipeline:
    """Construct fresh preprocessing + default L2 Logistic Regression.

    Keep default C=1 and lbfgs. In the installed sklearn API default L2 uses
    l1_ratio=0; leaving penalty unset avoids its deprecated parameter spelling.
    max_iter is a convergence budget, not a metric-tuning parameter.
    """
    return Pipeline([
        ("preprocessor", build_linear_preprocessor()),
        ("model", LogisticRegression(
            class_weight=None, max_iter=max_iter, random_state=RANDOM_STATE,
        )),
    ])


def _validate_labels(y: pd.Series | np.ndarray, size: int) -> np.ndarray:
    labels = np.asarray(y)
    if labels.ndim != 1 or len(labels) != size:
        raise ValueError("Labels must be one-dimensional and aligned with observations.")
    if not np.isin(labels, [0, 1]).all() or set(labels.tolist()) != {0, 1}:
        raise ValueError("Both binary labels 0 and 1 are required, with no missing values.")
    return labels


def _validate_xy(X: pd.DataFrame, y: pd.Series) -> None:
    _validate_labels(y, len(X))
    if not isinstance(y, pd.Series) or not X.index.equals(y.index):
        raise ValueError("X and y must retain the same source-row index and order.")


def _validate_probabilities(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Positive-class probabilities must be a nonempty 1D array.")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Probabilities must be finite and within [0, 1].")
    return values


def predict_at_default_threshold(probabilities: np.ndarray) -> np.ndarray:
    """Classify probability >= 0.5 as positive, without threshold optimization."""
    return (_validate_probabilities(probabilities) >= THRESHOLD).astype(np.int64)


def calculate_metrics(y: pd.Series | np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    """Compute binary metrics; PR summary is Average Precision, not trapezoidal area.

    Confusion matrix order is [[TN, FP], [FN, TP]]. Undefined classification
    precision/recall/F1 use zero_division=0. Rates are fractions in [0, 1].
    """
    values = _validate_probabilities(probabilities)
    labels = _validate_labels(y, len(values))
    predicted = predict_at_default_threshold(values)
    auc = float(roc_auc_score(labels, values))
    return {
        "rows": len(labels), "roc_auc": auc,
        "average_precision": float(average_precision_score(labels, values)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "accuracy": float(accuracy_score(labels, predicted)),
        "confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
        "positive_prevalence": float(labels.mean()),
        "predicted_positive_rate": float(predicted.mean()),
        "gini": 2 * auc - 1,
    }


def fit_baseline(X_train: pd.DataFrame, y_train: pd.Series, max_iter: int = MAX_ITER) -> Pipeline:
    """Fit the entire fresh pipeline on train only; fail loudly on nonconvergence."""
    _validate_xy(X_train, y_train)
    pipeline = build_baseline_pipeline(max_iter=max_iter)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            pipeline.fit(X_train, y_train)
        except ConvergenceWarning as exc:
            raise RuntimeError(
                "Baseline failed to converge; increase max_iter only, then rerun."
            ) from exc
    audit_pipeline(pipeline, X_train)
    return pipeline


def audit_pipeline(pipeline: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Validate feature mapping and finite parameters; return coefficient audit."""
    if list(pipeline.named_steps) != ["preprocessor", "model"]:
        raise ValueError("Expected exactly preprocessor + model.")
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    check_is_fitted(model)
    names = preprocessor.get_feature_names_out().tolist()
    transformed = preprocessor.transform(X)
    validate_transformed(X, transformed)
    if names != OUTPUT_FEATURE_COLUMNS or model.n_features_in_ != 13:
        raise ValueError("Model input must be the fixed 13-feature preprocessing output.")
    if model.feature_names_in_.tolist() != names or model.classes_.tolist() != [0, 1]:
        raise ValueError("Model feature or positive-class mapping is inconsistent.")
    if model.coef_.shape != (1, 13) or not np.isfinite(model.coef_).all():
        raise ValueError("Expected 13 finite binary-model coefficients.")
    if not np.isfinite(model.intercept_).all():
        raise ValueError("Model intercept must be finite.")
    if (model.n_iter_ >= model.max_iter).any():
        raise RuntimeError("Iteration budget was exhausted; review convergence before evaluation.")
    return pd.DataFrame({"feature": names, "coefficient": model.coef_[0]})


def positive_probabilities(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Return checked predict_proba[:, 1] for the confirmed class order [0, 1]."""
    if pipeline.named_steps["model"].classes_.tolist() != [0, 1]:
        raise ValueError("Expected class order [0, 1].")
    probabilities = pipeline.predict_proba(X)
    if probabilities.shape != (len(X), 2) or not np.isfinite(probabilities).all():
        raise ValueError("Expected a finite (rows, 2) probability matrix.")
    if ((probabilities < 0) | (probabilities > 1)).any() or not np.allclose(probabilities.sum(axis=1), 1):
        raise ValueError("Invalid class probabilities.")
    return _validate_probabilities(probabilities[:, 1])


def run_baseline(data: TrainingValidationData) -> BaselineResult:
    """Fit train and evaluate train/validation only; no output files are written."""
    _validate_xy(data.X_validation, data.y_validation)
    pipeline = fit_baseline(data.X_train, data.y_train)
    train_probabilities = positive_probabilities(pipeline, data.X_train)
    validation_probabilities = positive_probabilities(pipeline, data.X_validation)
    coefficients = audit_pipeline(pipeline, data.X_train)
    model = pipeline.named_steps["model"]
    report = {
        "model": "LogisticRegression", "threshold": THRESHOLD,
        "random_state": RANDOM_STATE, "class_weight": model.class_weight,
        "regularization": "L2", "C": model.C, "solver": model.solver,
        "max_iter": model.max_iter, "number_of_features": int(model.n_features_in_),
        "feature_names": coefficients["feature"].tolist(),
        "train": calculate_metrics(data.y_train, train_probabilities),
        "validation": calculate_metrics(data.y_validation, validation_probabilities),
        "convergence": {"warning": False, "n_iter": model.n_iter_.tolist(), "converged": True},
        "coefficients": coefficients.to_dict(orient="records"),
        "intercept": float(model.intercept_[0]),
        "notes": [
            "ROC-AUC measures ranking across thresholds; Average Precision summarizes the precision-recall curve.",
            "Accuracy alone is insufficient for rare positives; compare Average Precision with prevalence.",
            "Classification metrics use the unoptimized 0.5 threshold, not a final operating threshold.",
            "Probabilities are baseline risk scores, not validated calibrated PD estimates.",
            "Coefficients are conditional log-odds associations, not causal effects or SHAP explanations.",
            "Ten numeric features are standardized; three binary indicators are unscaled, limiting magnitude comparisons.",
            "Only training data fitted preprocessing and model; test set remains locked for modeling/evaluation.",
        ],
    }
    return BaselineResult(pipeline, report, validation_probabilities)


def save_reports(
    result: BaselineResult, y_validation: pd.Series, report_dir: Path = DEFAULT_REPORT_DIR,
) -> None:
    """Write actual metrics JSON and three validation-only PNGs, no model files."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    values = _validate_probabilities(result.validation_probabilities)
    labels = _validate_labels(y_validation, len(values))
    report_dir = Path(report_dir)
    figures_dir = report_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "baseline_logistic_metrics.json").write_text(
        json.dumps(result.report, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )

    def canvas(title: str):
        figure = Figure(figsize=(7, 5.5), layout="constrained")
        FigureCanvasAgg(figure)
        ax = figure.subplots()
        ax.set_title(title, fontsize=13, pad=14)
        return figure, ax

    metrics = result.report["validation"]
    figure, ax = canvas("Baseline Logistic Regression — Validation ROC")
    fpr, tpr, _ = roc_curve(labels, values)
    ax.plot(fpr, tpr, color="#176b91", linewidth=2, label=f"ROC-AUC = {metrics['roc_auc']:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random ranking")
    ax.set(xlabel="False positive rate", ylabel="True positive rate (recall)", xlim=(0, 1), ylim=(0, 1.02))
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right")
    figure.savefig(figures_dir / "baseline_logistic_roc_curve.png", dpi=160)

    figure, ax = canvas("Baseline Logistic Regression — Validation Precision–Recall")
    precision, recall, _ = precision_recall_curve(labels, values)
    ax.plot(recall, precision, color="#176b91", linewidth=2, label=f"Average Precision = {metrics['average_precision']:.4f}")
    ax.axhline(labels.mean(), linestyle="--", color="gray", label=f"Prevalence = {labels.mean():.4f}")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")
    figure.savefig(figures_dir / "baseline_logistic_pr_curve.png", dpi=160)

    figure, ax = canvas("Baseline Logistic Regression — Validation\nConfusion Matrix (threshold = 0.5)")
    matrix = confusion_matrix(labels, predict_at_default_threshold(values), labels=[0, 1])
    picture = ax.imshow(matrix, cmap="Blues")
    figure.colorbar(picture, ax=ax, label="Observations")
    for row in range(2):
        for column in range(2):
            ax.text(column, row, f"{matrix[row, column]:,}", ha="center", va="center",
                    fontsize=15, color="white" if matrix[row, column] > matrix.max() / 2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["0", "1"], yticklabels=["0", "1"],
           xlabel="Predicted label", ylabel="Actual label")
    figure.savefig(figures_dir / "baseline_logistic_confusion_matrix.png", dpi=160)


def main() -> None:
    """Execute the fixed baseline and save its train/validation report."""
    data = load_training_validation()
    result = run_baseline(data)
    save_reports(result, data.y_validation)
    print(json.dumps(result.report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
