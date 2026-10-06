"""Fixed Logistic Regression imbalance benchmarks with train-only grouped CV.

Run ``python -m src.imbalance_experiments`` for reports and validation figures.
Test data is never exposed to the experiment API. No tuning or serialization.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from src.baseline_model import (
    DEFAULT_REPORT_DIR, THRESHOLD, build_baseline_pipeline, calculate_metrics,
    positive_probabilities,
)
from src.data_split import RANDOM_STATE, load_data, split_data
from src.preprocessing import OUTPUT_FEATURE_COLUMNS

N_SPLITS = 5
EXPERIMENT_NAMES = ("baseline", "balanced", "smote")
DISPLAY_NAMES = {"baseline": "Baseline", "balanced": "Class weight: balanced", "smote": "SMOTE"}
BASELINE_REPORT = DEFAULT_REPORT_DIR / "baseline_logistic_metrics.json"


class PreprocessorAdapter(TransformerMixin, BaseEstimator):
    """Delegate unchanged preprocessing through an imblearn-compatible transformer.

    imblearn rejects a Pipeline as an intermediate step. This adapter clones and
    fits the original sklearn pipeline on the supplied fitting rows only. It
    changes composition, not transformations, parameters or output columns.
    """

    def __init__(self, preprocessor: Pipeline):
        self.preprocessor = preprocessor

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> PreprocessorAdapter:
        self.preprocessor_ = clone(self.preprocessor).fit(X, y)
        self.n_features_in_ = self.preprocessor_.n_features_in_
        self.feature_names_in_ = self.preprocessor_.feature_names_in_
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "preprocessor_")
        return self.preprocessor_.transform(X)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, "preprocessor_")
        return self.preprocessor_.get_feature_names_out(input_features)

    @property
    def named_steps(self):
        """Expose underlying steps for fitting-statistic audits."""
        return getattr(self, "preprocessor_", self.preprocessor).named_steps


def class_counts(y: pd.Series | np.ndarray) -> dict[str, int]:
    """JSON-friendly class counts without retaining synthetic observations."""
    values, counts = np.unique(np.asarray(y), return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(values, counts)}


class AuditedSMOTE(SMOTE):
    """Standard SMOTE that records actual input/output counts only during fit."""

    def fit_resample(self, X, y, **params):
        resampled_X, resampled_y = super().fit_resample(X, y, **params)
        self.before_counts_ = class_counts(y)
        self.after_counts_ = class_counts(resampled_y)
        self.input_rows_ = len(y)
        self.output_rows_ = len(resampled_y)
        return resampled_X, resampled_y


@dataclass(frozen=True)
class ExperimentData:
    """Only training groups/labels and the fixed validation partition."""

    X_train: pd.DataFrame
    y_train: pd.Series
    groups_train: pd.Series
    X_validation: pd.DataFrame
    y_validation: pd.Series


@dataclass(frozen=True)
class ExperimentResult:
    """Reports plus validation predictions, all in memory until explicit export."""

    report: dict
    validation_probabilities: dict[str, np.ndarray]


def load_experiment_data() -> ExperimentData:
    """Reuse raw feature groups from the splitter, ignoring its test partition.

    Existing splitter integrity checks remain in place. No test target is used
    by training, CV, evaluation, reporting or plotting in this module.
    """
    splits = split_data(load_data())
    return ExperimentData(
        splits.train.X, splits.train.y, splits.train.groups,
        splits.validation.X, splits.validation.y,
    )


def build_experiments() -> dict[str, Pipeline | ImbalancedPipeline]:
    """Create exactly three fresh pipelines; only imbalance strategy differs."""
    baseline = build_baseline_pipeline()
    balanced = build_baseline_pipeline().set_params(model__class_weight="balanced")
    smote_base = build_baseline_pipeline()
    smote = ImbalancedPipeline([
        ("preprocessor", PreprocessorAdapter(smote_base.named_steps["preprocessor"])),
        ("smote", AuditedSMOTE(random_state=RANDOM_STATE)),
        ("model", smote_base.named_steps["model"]),
    ])
    return {"baseline": baseline, "balanced": balanced, "smote": smote}


def make_cv() -> StratifiedGroupKFold:
    """Fixed five-fold stratification, preserving the existing raw feature groups."""
    return StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)


def validate_xy(X: pd.DataFrame, y: pd.Series) -> None:
    if not X.index.equals(y.index) or len(X) != len(y):
        raise ValueError("Features and labels must retain source-row alignment.")
    if y.isna().any() or set(y.unique()) != {0, 1}:
        raise ValueError("Expected nonmissing binary labels with both classes present.")


def make_cv_folds(data: ExperimentData) -> list[tuple[np.ndarray, np.ndarray]]:
    """Materialize shared folds and verify group/row isolation and full coverage."""
    validate_xy(data.X_train, data.y_train)
    if not data.groups_train.index.equals(data.X_train.index) or data.groups_train.isna().any():
        raise ValueError("Training groups must be complete and aligned with raw training rows.")
    folds = list(make_cv().split(data.X_train, data.y_train, data.groups_train))
    coverage = np.zeros(len(data.X_train), dtype=int)
    for fitting, evaluation in folds:
        if np.intersect1d(fitting, evaluation).size:
            raise ValueError("CV source-row overlap detected.")
        if len(np.union1d(fitting, evaluation)) != len(coverage):
            raise ValueError("CV fold does not cover the training partition.")
        if np.intersect1d(data.groups_train.iloc[fitting], data.groups_train.iloc[evaluation]).size:
            raise ValueError("CV feature-group overlap detected.")
        coverage[evaluation] += 1
    if not np.all(coverage == 1):
        raise ValueError("Each training row must be evaluated in exactly one CV fold.")
    return folds


def fit_checked(pipeline, X: pd.DataFrame, y: pd.Series):
    """Fit a fresh pipeline on fitting rows; reject warnings or invalid parameters."""
    validate_xy(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            pipeline.fit(X, y)
        except ConvergenceWarning as exc:
            raise RuntimeError("Nonconvergence: stop comparison and review max_iter uniformly.") from exc
    model = pipeline.named_steps["model"]
    names = pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
    if names != OUTPUT_FEATURE_COLUMNS or model.n_features_in_ != 13:
        raise ValueError("Unexpected model input schema.")
    if not np.isfinite(model.coef_).all() or not np.isfinite(model.intercept_).all():
        raise ValueError("Model coefficients/intercept must be finite.")
    if (model.n_iter_ >= model.max_iter).any():
        raise RuntimeError("Iteration budget exhausted; stop comparison.")
    return pipeline


def check_baseline_reproduction(metrics: dict, reference_path: Path = BASELINE_REPORT) -> None:
    """Fail before comparison if actual baseline differs from the accepted report."""
    reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))["validation"]
    for key, expected in reference.items():
        actual = metrics[key]
        if key in ("confusion_matrix", "rows"):
            matches = actual == expected
        else:
            matches = bool(np.isclose(actual, expected, rtol=1e-6, atol=1e-8))
        if not matches:
            raise ValueError(f"STEP 8 baseline reproduction failed for {key}: {actual} != {expected}")


def run_cv(pipeline, data: ExperimentData, folds: list) -> dict:
    """Clone and fit preprocessing/resampling/model separately inside each fold."""
    fold_results = []
    for number, (fitting, evaluation) in enumerate(folds, start=1):
        fitted = fit_checked(clone(pipeline), data.X_train.iloc[fitting], data.y_train.iloc[fitting])
        probabilities = positive_probabilities(fitted, data.X_train.iloc[evaluation])
        metrics = calculate_metrics(data.y_train.iloc[evaluation], probabilities)
        record = {
            "fold": number, "fitting_rows": len(fitting), "evaluation_rows": len(evaluation),
            "group_overlap": int(np.intersect1d(data.groups_train.iloc[fitting], data.groups_train.iloc[evaluation]).size),
            "roc_auc": metrics["roc_auc"], "average_precision": metrics["average_precision"],
            "n_iter": fitted.named_steps["model"].n_iter_.tolist(), "convergence_warning": False,
        }
        if "smote" in fitted.named_steps:
            sampler = fitted.named_steps["smote"]
            record["smote"] = {"input_rows": sampler.input_rows_, "before": sampler.before_counts_, "after": sampler.after_counts_}
        fold_results.append(record)
    return {
        "folds": fold_results,
        "summary": {key: {"mean": float(np.mean([f[key] for f in fold_results])),
                          "std": float(np.std([f[key] for f in fold_results], ddof=0))}
                    for key in ("roc_auc", "average_precision")},
        "std_ddof": 0,
    }


def run_experiments(data: ExperimentData) -> ExperimentResult:
    """Reproduce baseline first, then compare identical train-only folds and holdout."""
    validate_xy(data.X_validation, data.y_validation)
    prototypes = build_experiments()
    # Gate comparison on STEP 8 reproduction before fitting any alternatives.
    baseline = fit_checked(clone(prototypes["baseline"]), data.X_train, data.y_train)
    baseline_probabilities = positive_probabilities(baseline, data.X_validation)
    baseline_metrics = calculate_metrics(data.y_validation, baseline_probabilities)
    check_baseline_reproduction(baseline_metrics)
    folds = make_cv_folds(data)
    experiments, predictions = {}, {}
    for name, prototype in prototypes.items():
        cv = run_cv(prototype, data, folds)
        fitted = baseline if name == "baseline" else fit_checked(clone(prototype), data.X_train, data.y_train)
        probabilities = baseline_probabilities if name == "baseline" else positive_probabilities(fitted, data.X_validation)
        model = fitted.named_steps["model"]
        config = {"class_weight": model.class_weight, "C": model.C, "solver": model.solver,
                  "max_iter": model.max_iter, "regularization": "L2", "random_state": RANDOM_STATE,
                  "preprocessor": "build_linear_preprocessor", "number_of_features": 13}
        entry = {"configuration": config, "cv": cv,
                 "validation": calculate_metrics(data.y_validation, probabilities),
                 "convergence": {"warning": False, "n_iter": model.n_iter_.tolist(), "converged": True}}
        if name == "smote":
            sampler = fitted.named_steps["smote"]
            config["smote"] = {"random_state": sampler.random_state, "sampling_strategy": sampler.sampling_strategy, "k_neighbors": sampler.k_neighbors}
            entry["smote_audit"] = {"before_rows": sampler.input_rows_, "after_rows": sampler.output_rows_,
                                    "before_class_counts": sampler.before_counts_, "after_class_counts": sampler.after_counts_}
        experiments[name] = entry
        predictions[name] = probabilities
    return ExperimentResult({
        "random_state": RANDOM_STATE, "threshold": THRESHOLD,
        "cv_design": {"splitter": "StratifiedGroupKFold", "n_splits": N_SPLITS, "shuffle": True,
                      "source": "training partition raw feature-group metadata", "shared_folds": True},
        "baseline_reproduction": "PASS", "experiments": experiments,
        "notes": [
            "All preprocessing and SMOTE fit only inside each fitting fold or full training partition.",
            "Validation is unchanged and never resampled. Test is not evaluated.",
            "SMOTE interpolates transformed numeric space; fractional count/binary-like values may not represent real customer profiles.",
            "SMOTE is a benchmark, not a preferred method by assumption. Synthetic samples are not exported.",
            "Threshold remains 0.5. No final strategy selection, tuning, calibration or calibrated PD claim.",
            "Average Precision is the PR summary, not trapezoidal PR area; CV std uses ddof=0 and is not a confidence interval.",
        ],
    }, predictions)


def save_reports(result: ExperimentResult, y_validation: pd.Series, report_dir: Path = DEFAULT_REPORT_DIR) -> None:
    """Write actual results and validation-only figures, never synthetic samples."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from sklearn.metrics import precision_recall_curve, roc_curve

    report_dir = Path(report_dir)
    figures = report_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    (report_dir / "logistic_imbalance_experiments.json").write_text(
        json.dumps(result.report, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    entries = result.report["experiments"]
    colors = ["#176b91", "#d67718", "#34834a"]
    for kind in ("roc", "pr"):
        figure = Figure(figsize=(8, 6), layout="constrained")
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
            ax.plot(x, y, color=color, label=label, linewidth=1.8)
        if kind == "roc":
            ax.plot([0, 1], [0, 1], "--", color="gray", label="Random ranking")
        else:
            ax.axhline(y_validation.mean(), linestyle="--", color="gray", label=f"Prevalence {y_validation.mean():.4f}")
        ax.set(xlabel="False positive rate" if kind == "roc" else "Recall",
               ylabel="True positive rate (recall)" if kind == "roc" else "Precision",
               title="Logistic Regression — Validation " + ("ROC" if kind == "roc" else "Precision–Recall"),
               xlim=(0, 1), ylim=(0, 1.02))
        ax.grid(alpha=.2)
        ax.legend(loc="lower right" if kind == "roc" else "upper right", fontsize=9)
        figure.savefig(figures / f"logistic_imbalance_{kind}_comparison.png", dpi=160)
    figure = Figure(figsize=(8, 5.5), layout="constrained")
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    keys = ["precision", "recall", "f1", "average_precision"]
    positions = np.arange(len(keys))
    for number, (name, color) in enumerate(zip(EXPERIMENT_NAMES, colors)):
        values = [entries[name]["validation"][key] for key in keys]
        bars = ax.bar(positions + (number - 1) * .25, values, .25, label=DISPLAY_NAMES[name], color=color)
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
    ax.set(xticks=positions, xticklabels=["Precision", "Recall", "F1", "Average Precision"],
           ylim=(0, 1), ylabel="Score", title="Logistic Regression — Validation Metrics\nClassification threshold = 0.5; AP uses probability rankings")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    figure.savefig(figures / "logistic_imbalance_validation_metrics.png", dpi=160)


def main() -> None:
    data = load_experiment_data()
    result = run_experiments(data)
    save_reports(result, data.y_validation)
    print(json.dumps(result.report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
