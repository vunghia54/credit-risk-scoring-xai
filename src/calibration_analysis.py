"""Train-only grouped probability calibration with fixed validation evaluation.

Run ``python -m src.calibration_analysis`` for the full integration experiment.
There is no threshold optimization, test evaluation or model serialization.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline

from src.baseline_model import (
    DEFAULT_REPORT_DIR, _validate_labels, _validate_probabilities,
)
from src.boosting_tuning import TrainingData, training_folds
from src.data_split import RANDOM_STATE
from src.imbalance_experiments import ExperimentData, fit_checked, validate_xy
from src.preprocessing import OUTPUT_FEATURE_COLUMNS
from src.threshold_analysis import MODEL_NAMES, build_candidates, candidate_definition, load_references
from src.tree_model_experiments import fit_tree, load_tree_data

CALIBRATION_STATES = ("uncalibrated", "sigmoid", "isotonic")
N_BINS = 10


@dataclass(frozen=True)
class CalibrationResult:
    """Measured aggregates and validation-only scores; no retained fitted models."""

    report: dict
    comparison: pd.DataFrame
    bins: pd.DataFrame
    validation_scores: dict[tuple[str, str], np.ndarray]


def positive_probabilities(model: Pipeline | CalibratedClassifierCV, X: pd.DataFrame) -> np.ndarray:
    """Validate either sklearn classifier API, without assuming pipeline internals."""
    if np.asarray(model.classes_).tolist() != [0, 1]:
        raise ValueError("Expected fitted class order [0, 1].")
    values = np.asarray(model.predict_proba(X), dtype=float)
    if values.shape != (len(X), 2):
        raise ValueError("Expected a two-column probability matrix aligned with X.")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Probabilities must be finite and within [0, 1].")
    if not np.allclose(values.sum(axis=1), 1, rtol=1e-6, atol=1e-8):
        raise ValueError("Class probabilities must sum to one.")
    return _validate_probabilities(values[:, 1])


def calibration_folds(train: TrainingData) -> list[tuple[np.ndarray, np.ndarray]]:
    """Reuse deterministic 5-fold SGKF and its row/group coverage checks."""
    folds = training_folds(train)
    for fitting, calibration in folds:
        if set(train.y.iloc[fitting].unique()) != {0, 1} or set(train.y.iloc[calibration].unique()) != {0, 1}:
            raise ValueError("Both classes must occur in each fitting and calibration portion.")
    return folds


def fold_audit(train: TrainingData, folds: list[tuple[np.ndarray, np.ndarray]]) -> list[dict]:
    """Record train-only row/group isolation and calibration sample counts."""
    return [{"fold": number, "fitting_rows": len(fitting), "calibration_rows": len(calibration),
             "row_overlap": int(np.intersect1d(fitting, calibration).size),
             "group_overlap": int(np.intersect1d(train.groups.iloc[fitting], train.groups.iloc[calibration]).size)}
            for number, (fitting, calibration) in enumerate(folds, 1)]


def build_calibrator(pipeline: Pipeline, method: str,
                     folds: list[tuple[np.ndarray, np.ndarray]]) -> CalibratedClassifierCV:
    """Explicit ensemble of five independently fitted base/calibrator pairs.

    Outer n_jobs=1 avoids nested parallelism; approved tree n_jobs stays -1.
    The entire preprocessing pipeline is cloned and fitted within each fold.
    """
    if method not in CALIBRATION_STATES[1:]:
        raise ValueError("Calibration method must be sigmoid or isotonic.")
    if len(folds) != 5:
        raise ValueError("Exactly five explicit calibration folds are required.")
    return CalibratedClassifierCV(estimator=pipeline, method=method, cv=folds, ensemble=True, n_jobs=1)


def fit_calibrator(pipeline: Pipeline, method: str, train: TrainingData,
                   folds: list[tuple[np.ndarray, np.ndarray]]) -> CalibratedClassifierCV:
    """Fit using training features/labels only; outer validation is not an input."""
    validate_xy(train.X, train.y)
    model = build_calibrator(pipeline, method, folds)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(train.X, train.y)
    if len(model.calibrated_classifiers_) != 5:
        raise ValueError("Expected five fitted base/calibrator pairs.")
    for pair in model.calibrated_classifiers_:
        fitted = pair.estimator
        names = fitted.named_steps["preprocessor"].get_feature_names_out().tolist()
        if names != OUTPUT_FEATURE_COLUMNS or fitted.named_steps["model"].n_features_in_ != 13:
            raise ValueError("Each calibration base estimator must use 13 approved features.")
    return model


def reliability_bins(y_validation: np.ndarray, scores: np.ndarray, n_bins: int = N_BINS) -> pd.DataFrame:
    """Quantile bins with duplicate boundaries removed and empty bins omitted.

    Identical scores always remain together. Internal edges belong to the bin
    on their left; endpoints 0/1 are supported. A constant score gives one bin.
    Count-weighted absolute gaps define this project's bin-dependent ECE.
    """
    values = _validate_probabilities(scores)
    labels = _validate_labels(y_validation, len(values))
    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer.")
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
    assignments = np.searchsorted(edges[1:-1], values, side="left")
    rows = []
    for group in np.unique(assignments):
        mask = assignments == group
        mean_probability = float(values[mask].mean())
        observed = float(labels[mask].mean())
        rows.append({"bin_id": len(rows) + 1, "count": int(mask.sum()),
                     "mean_predicted_probability": mean_probability,
                     "observed_positive_rate": observed, "absolute_gap": abs(observed - mean_probability)})
    return pd.DataFrame(rows)


def expected_calibration_error(bins: pd.DataFrame) -> float:
    """Sum count/total * absolute gap; not a universal or regulatory metric."""
    counts = bins["count"].to_numpy(dtype=float)
    gaps = bins["absolute_gap"].to_numpy(dtype=float)
    if len(counts) == 0 or not np.isfinite(counts).all() or (counts <= 0).any():
        raise ValueError("Reliability bins require positive finite counts.")
    if not np.isfinite(gaps).all() or ((gaps < 0) | (gaps > 1)).any():
        raise ValueError("Reliability gaps must be finite and in [0, 1].")
    return float(np.dot(counts / counts.sum(), gaps))


def calibration_metrics(y_validation: np.ndarray, scores: np.ndarray) -> tuple[dict, pd.DataFrame]:
    """Evaluate probability quality and ranking; no classification thresholds.

    Brier is mean squared binary probability error. Log loss uses sklearn's
    numerical clipping at floating-point endpoints. Neither is a ranking metric.
    """
    values = _validate_probabilities(scores)
    labels = _validate_labels(y_validation, len(values))
    bins = reliability_bins(labels, values)
    prevalence = float(labels.mean())
    mean_probability = float(values.mean())
    return {"brier_score": float(brier_score_loss(labels, values)),
            "log_loss": float(log_loss(labels, values, labels=[0, 1])),
            "ece": expected_calibration_error(bins),
            "mean_predicted_probability": mean_probability, "prevalence": prevalence,
            "mean_probability_gap": abs(mean_probability - prevalence),
            "roc_auc": float(roc_auc_score(labels, values)),
            "average_precision": float(average_precision_score(labels, values))}, bins


def check_reference(metrics: dict, reference: dict) -> None:
    """Stop before calibrating if uncalibrated validation ranking has drifted."""
    for metric in ("roc_auc", "average_precision"):
        if not np.isclose(metrics[metric], reference[metric], rtol=1e-6, atol=1e-8):
            raise ValueError(f"Uncalibrated reference reproduction failed: {metric}")


def run_analysis(data: ExperimentData) -> CalibrationResult:
    """Reproduce all three fixed candidates, then calibrate on training only."""
    started = perf_counter()
    train = TrainingData(data.X_train, data.y_train, data.groups_train)
    folds = calibration_folds(train)
    audits = fold_audit(train, folds)
    validate_xy(data.X_validation, data.y_validation)
    labels = data.y_validation.to_numpy(copy=True)
    candidates, references = build_candidates(), load_references()
    rows, bins_tables, scores, definitions = [], [], {}, {}
    runtimes = {name: {"uncalibrated": 0.0, "sigmoid": 0.0, "isotonic": 0.0} for name in MODEL_NAMES}

    def evaluate(name: str, method: str, model) -> dict:
        probability = positive_probabilities(model, data.X_validation)
        metrics, bins = calibration_metrics(labels, probability)
        scores[name, method] = probability
        rows.append({"model": name, "calibration_method": method, **metrics})
        bins.insert(0, "calibration_method", method)
        bins.insert(0, "model", name)
        bins_tables.append(bins)
        return metrics

    # All uncalibrated references must pass before any calibration fitting.
    for name, pipeline in candidates.items():
        tick = perf_counter()
        definitions[name] = candidate_definition(pipeline)
        fit = fit_checked if name == MODEL_NAMES[0] else fit_tree
        model = fit(pipeline, train.X, train.y)
        check_reference(evaluate(name, "uncalibrated", model), references[name])
        runtimes[name]["uncalibrated"] = perf_counter() - tick
    for name, pipeline in candidates.items():
        for method in CALIBRATION_STATES[1:]:
            tick = perf_counter()
            model = fit_calibrator(pipeline, method, train, folds)
            evaluate(name, method, model)
            runtimes[name][method] = perf_counter() - tick
    comparison = pd.DataFrame(rows)
    bins = pd.concat(bins_tables, ignore_index=True)
    report = {
        "random_state": RANDOM_STATE, "candidate_definitions": definitions,
        "calibration_states": list(CALIBRATION_STATES),
        "reference_reproduction": dict.fromkeys(MODEL_NAMES, "PASS"),
        "reference_tolerance": {"rtol": 1e-6, "atol": 1e-8},
        "calibration_design": {"splitter": "StratifiedGroupKFold", "n_splits": 5,
                               "shuffle": True, "random_state": RANDOM_STATE,
                               "groups": "raw training feature-group metadata", "explicit_cv_splits": True,
                               "ensemble": True, "outer_n_jobs": 1, "fitting_partition": "training only",
                               "prediction": "mean of five fold-specific calibrated probability estimates",
                               "calibration_input": "decision_function if available (Logistic), otherwise predict_proba (boosting)"},
        "group_leakage_checks": audits, "validation_rows": len(labels),
        "validation_prevalence": float(labels.mean()),
        "metrics": comparison.to_dict(orient="records"),
        "binning": {"strategy": "quantile", "requested_bins": N_BINS,
                    "duplicate_policy": "remove duplicate quantile edges; do not split identical scores; omit empty bins",
                    "actual_bins": bins.groupby(["model", "calibration_method"], sort=False).size().rename("count").reset_index().to_dict(orient="records"),
                    "ece_definition": "sum(bin_count / total_count * abs(observed_positive_rate - mean_predicted_probability))"},
        "runtime_seconds": {"by_model_and_state": runtimes,
                            "calibration_by_model": {name: times["sigmoid"] + times["isotonic"] for name, times in runtimes.items()},
                            "total_including_folds_references_and_evaluation": perf_counter() - started},
        "notes": ["Test remains locked; no test labels, predictions, metrics or distributions are used by this experiment.",
                  "Preprocessing and base models fit only each fitting fold; calibrators use only the disjoint training calibration portion.",
                  "Outer validation is used only for evaluation; no calibration or base model is fitted on it.",
                  "Calibrated ensemble predictions differ from the full-train uncalibrated estimator: changes include ensembling as well as calibration.",
                  "Isotonic can create ties; neither method is assumed to preserve ensemble ranking exactly.",
                  "ECE is this project's bin-dependent implementation, not a universal regulatory metric. Mean probability alone does not establish calibration quality.",
                  "No final model, production threshold, regulatory model or production PD is declared; no threshold optimization is performed.",
                  "Wall-clock times describe this environment only, exclude data loading and export, and are not standardized hardware benchmarks."],
    }
    return CalibrationResult(report, comparison, bins, scores)


def save_reports(result: CalibrationResult, report_dir: Path = DEFAULT_REPORT_DIR) -> None:
    """Export nine measured combinations, actual bins and three validation figures."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    report_dir = Path(report_dir)
    figures = report_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    result.comparison.to_csv(report_dir / "calibration_comparison.csv", index=False)
    result.bins.to_csv(report_dir / "calibration_bins.csv", index=False)
    (report_dir / "calibration_results.json").write_text(json.dumps(result.report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    colors = ["#64748b", "#176b91", "#c56136"]

    def canvas(width: float, height: float):
        figure = Figure(figsize=(width, height), layout="constrained")
        FigureCanvasAgg(figure)
        return figure

    figure = canvas(14, 4.8)
    for ax, name in zip(figure.subplots(1, 3), MODEL_NAMES):
        ax.plot([0, 1], [0, 1], "--", color="black", linewidth=1, label="Perfect calibration")
        for method, color in zip(CALIBRATION_STATES, colors):
            table = result.bins.loc[result.bins.model.eq(name) & result.bins.calibration_method.eq(method)]
            ax.plot(table.mean_predicted_probability, table.observed_positive_rate, "o-", color=color, label=method.title(), markersize=4)
        ax.set(title=name, xlabel="Mean predicted probability", ylabel="Observed positive rate", xlim=(0, 1), ylim=(0, 1))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    figure.suptitle("Validation Reliability Curves — Up to 10 Quantile Bins")
    figure.savefig(figures / "calibration_reliability_curves.png", dpi=160)

    figure = canvas(13, 5)
    for ax, metric, title in zip(figure.subplots(1, 2), ["brier_score", "log_loss"], ["Brier Score", "Log Loss"]):
        for i, (method, color) in enumerate(zip(CALIBRATION_STATES, colors)):
            table = result.comparison.loc[result.comparison.calibration_method.eq(method)].set_index("model").loc[list(MODEL_NAMES)]
            bars = ax.bar(np.arange(3) + (i - 1) * .25, table[metric], width=.25, color=color, label=method.title())
            ax.bar_label(bars, fmt="%.4f", fontsize=8, padding=3)
        ax.set(title=f"{title} (Lower Is Better)", xticks=np.arange(3), xticklabels=["Logistic Balanced", "XGBoost Baseline", "LGBM-02"], ylabel=title)
        ax.margins(y=.3)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    figure.suptitle("Validation Probability Quality — Fixed Candidates")
    figure.savefig(figures / "calibration_brier_logloss_comparison.png", dpi=160)

    figure = canvas(14, 4.8)
    for ax, name in zip(figure.subplots(1, 3), MODEL_NAMES):
        for method, color in zip(CALIBRATION_STATES, colors):
            ax.hist(result.validation_scores[name, method], bins=np.linspace(0, 1, 41), density=True,
                    histtype="step", linewidth=1.7, color=color, label=method.title())
        ax.set(title=name, xlabel="Predicted probability / score", ylabel="Density", xlim=(0, 1))
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    figure.suptitle("Validation Score Distributions — Before and After Calibration")
    figure.savefig(figures / "calibration_score_distribution.png", dpi=160)


def main() -> None:
    result = run_analysis(load_tree_data())
    save_reports(result)
    print("Uncalibrated reference reproduction: PASS for all three candidates")
    print(result.comparison.to_string(index=False))
    print(json.dumps(result.report["runtime_seconds"], indent=2))


if __name__ == "__main__":
    main()
