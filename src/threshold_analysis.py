"""Fixed candidate comparison and validation-only operating-point analysis.

Run ``python -m src.threshold_analysis`` to export reports. Test stays locked;
scenario thresholds are validation-optimized, not production decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.baseline_model import (
    DEFAULT_REPORT_DIR, TrainingValidationData, _validate_labels,
    _validate_probabilities, build_baseline_pipeline, calculate_metrics,
    load_training_validation, positive_probabilities,
)
from src.boosting_tuning import build_candidate, candidate_configurations
from src.data_split import RANDOM_STATE
from src.imbalance_experiments import fit_checked, validate_xy
from src.tree_model_experiments import fit_tree

MODEL_NAMES = ("Logistic Balanced", "XGBoost Baseline", "LightGBM Tuned (LGBM-02)")
SCENARIOS = ("DEFAULT", "MAX_F1", "RECALL_AT_LEAST_50", "RECALL_AT_LEAST_70", "PRECISION_AT_LEAST_30")
METRIC_COLUMNS = ("threshold", "precision", "recall", "f1", "specificity",
                  "false_positive_rate", "false_negative_rate", "predicted_positive_rate",
                  "tp", "fp", "tn", "fn", "balanced_accuracy")


@dataclass(frozen=True)
class ThresholdResult:
    """Only validation scores are retained for plots; no fitted model is saved."""

    report: dict
    thresholds: pd.DataFrame
    scenarios: pd.DataFrame
    validation_scores: dict[str, np.ndarray]
    validation_labels: np.ndarray


def build_candidates() -> dict[str, Pipeline]:
    """Reuse approved L2 logistic and explicit XGB-01/LGBM-02 definitions."""
    configs = {c.configuration_id: c for c in candidate_configurations()}
    return {
        MODEL_NAMES[0]: build_baseline_pipeline().set_params(model__class_weight="balanced"),
        MODEL_NAMES[1]: build_candidate(configs["XGB-01"]),
        MODEL_NAMES[2]: build_candidate(configs["LGBM-02"]),
    }


def threshold_grid() -> np.ndarray:
    """91 hundredth-spaced thresholds, including 0.50 exactly once."""
    return np.arange(5, 96, dtype=int) / 100.0


def threshold_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Classify score >= threshold; undefined precision/F1 are zero.

    Both true classes are required so specificity, sensitivity and error rates
    have nonzero denominators. No score is treated as a calibrated PD.
    """
    values = _validate_probabilities(scores)
    labels = _validate_labels(y, len(values))
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be finite and in [0, 1].")
    predicted = values >= threshold
    positive = labels == 1
    tp = int(np.sum(predicted & positive))
    fp = int(np.sum(predicted & ~positive))
    tn = int(np.sum(~predicted & ~positive))
    fn = int(np.sum(~predicted & positive))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return dict(threshold=float(threshold), precision=precision, recall=recall,
                f1=2 * tp / (2 * tp + fp + fn), specificity=specificity,
                false_positive_rate=fp / (fp + tn), false_negative_rate=fn / (fn + tp),
                predicted_positive_rate=(tp + fp) / len(labels), tp=tp, fp=fp, tn=tn, fn=fn,
                balanced_accuracy=(recall + specificity) / 2)


def analyze_thresholds(y_validation: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    """Evaluate the fixed grid using validation labels and scores only."""
    return pd.DataFrame([threshold_metrics(y_validation, scores, t) for t in threshold_grid()])


def select_scenario(table: pd.DataFrame, scenario: str) -> dict:
    """Select within the supplied grid with the exact prescribed tie-breaks."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    if table.empty or table.threshold.duplicated().any():
        raise ValueError("Scenario table must be nonempty with unique thresholds.")
    if not np.isfinite(table[list(METRIC_COLUMNS)].to_numpy(dtype=float)).all():
        raise ValueError("Scenario table must contain finite metrics.")
    eligible = table
    if scenario == "DEFAULT":
        eligible = table.loc[table.threshold.eq(.5)]
        keys, ascending = ["threshold"], [True]
    elif scenario == "MAX_F1":
        keys, ascending = ["f1", "recall", "precision", "threshold"], [False, False, False, True]
    elif scenario.startswith("RECALL_AT_LEAST"):
        minimum = .5 if scenario.endswith("50") else .7
        eligible = table.loc[table.recall.ge(minimum)]
        keys, ascending = ["precision", "f1", "threshold"], [False, False, False]
    else:
        eligible = table.loc[table.precision.ge(.3)]
        keys, ascending = ["recall", "f1", "threshold"], [False, False, True]
    if eligible.empty:
        return {"scenario": scenario, "status": "unavailable",
                "reason": "No threshold in the specified grid satisfies this scenario.",
                **dict.fromkeys(METRIC_COLUMNS)}
    row = eligible.sort_values(keys, ascending=ascending, kind="stable").iloc[0]
    metrics = {k: int(row[k]) if k in ("tp", "fp", "tn", "fn") else float(row[k]) for k in METRIC_COLUMNS}
    return {"scenario": scenario, "status": "available", "reason": None, **metrics}


def score_distribution(y_validation: np.ndarray, scores: np.ndarray) -> dict:
    """Validation score quantiles overall and separately for both true classes."""
    values = _validate_probabilities(scores)
    labels = _validate_labels(y_validation, len(values))
    result = {}
    for name, sample in [("all", values), ("target_0", values[labels == 0]), ("target_1", values[labels == 1])]:
        quantiles = np.quantile(sample, [0, .01, .05, .25, .5, .75, .95, .99, 1])
        result[name] = dict(zip(["min", "p01", "p05", "p25", "median", "p75", "p95", "p99", "max"], map(float, quantiles)))
        result[name].update(mean=float(sample.mean()), count=len(sample))
    return result


def load_references(report_dir: Path = DEFAULT_REPORT_DIR) -> dict:
    """Load actual approved validation references, never hard-coded scores."""
    def read(name: str) -> dict:
        return json.loads((Path(report_dir) / name).read_text(encoding="utf-8"))
    return {
        MODEL_NAMES[0]: read("logistic_imbalance_experiments.json")["experiments"]["balanced"]["validation"],
        MODEL_NAMES[1]: read("tree_model_experiments.json")["tree_experiments"]["xgboost"]["validation"],
        MODEL_NAMES[2]: read("boosting_tuning_results.json")["selected"]["lightgbm"]["validation"],
    }


def check_reproduction(actual: dict, reference: dict) -> None:
    """Stop on reference drift before any threshold selection or export."""
    for key in ("rows", "confusion_matrix"):
        if actual[key] != reference[key]:
            raise ValueError(f"Reference reproduction failed: {key}")
    for key in ("roc_auc", "average_precision", "precision", "recall", "f1", "accuracy",
                "gini", "predicted_positive_rate", "positive_prevalence"):
        if not np.isclose(actual[key], reference[key], rtol=1e-6, atol=1e-8):
            raise ValueError(f"Reference reproduction failed: {key}")


def candidate_definition(pipeline: Pipeline) -> dict:
    """JSON-friendly actual model parameters plus approved preprocessing identity."""
    model = pipeline.named_steps["model"]
    keys = ("class_weight", "C", "solver", "max_iter", "l1_ratio", "random_state") if hasattr(model, "solver") else (
        "n_estimators", "learning_rate", "max_depth", "min_child_weight", "min_child_samples",
        "num_leaves", "subsample", "subsample_freq", "colsample_bytree", "reg_alpha", "reg_lambda",
        "objective", "eval_metric", "tree_method", "random_state", "n_jobs", "verbosity")
    params = model.get_params(deep=False)
    return {"estimator": type(model).__name__, "parameters": {k: params[k] for k in keys if k in params},
            "preprocessor": "build_linear_preprocessor" if hasattr(model, "solver") else "build_tree_preprocessor",
            "regularization_note": "Logistic uses the existing sklearn default L2 (l1_ratio=0); deprecated penalty spelling is unchanged."}


def run_analysis(data: TrainingValidationData) -> ThresholdResult:
    """Fit full train once per fixed candidate, reproduce all three, then analyze."""
    validate_xy(data.X_train, data.y_train)
    validate_xy(data.X_validation, data.y_validation)
    references = load_references()
    scores, comparisons, definitions = {}, {}, {}
    for name, pipeline in build_candidates().items():
        definitions[name] = candidate_definition(pipeline)
        fit = fit_checked if name == MODEL_NAMES[0] else fit_tree
        fitted = fit(pipeline, data.X_train, data.y_train)
        scores[name] = positive_probabilities(fitted, data.X_validation)
        comparisons[name] = calculate_metrics(data.y_validation, scores[name])
        check_reproduction(comparisons[name], references[name])
    # No scenario selection occurs until ALL three reference checks have passed.
    tables, scenarios, distributions = [], [], {}
    labels = data.y_validation.to_numpy(copy=True)
    for name in MODEL_NAMES:
        table = analyze_thresholds(labels, scores[name])
        table.insert(0, "model", name)
        tables.append(table)
        scenarios.extend({"model": name, **select_scenario(table, s)} for s in SCENARIOS)
        distributions[name] = score_distribution(labels, scores[name])
    report = {
        "random_state": RANDOM_STATE, "candidate_definitions": definitions,
        "reference_reproduction": dict.fromkeys(MODEL_NAMES, "PASS"),
        "reference_tolerance": {"rtol": 1e-6, "atol": 1e-8},
        "candidate_comparison_at_0_5": comparisons,
        "validation_prevalence": float(labels.mean()), "validation_rows": len(labels),
        "threshold_grid": {"start": .05, "stop": .95, "step": .01, "count": 91,
                           "positive_rule": "score >= threshold"},
        "score_distributions": distributions, "selected_scenarios": scenarios,
        "notes": ["Test set remains locked; no test labels, predictions or metrics are used by this analysis.",
                  "Scores are not calibrated PD; no calibration was performed.",
                  "MAX_F1 and constrained scenarios are validation-optimized within the fixed grid, not final production thresholds.",
                  "ROC-AUC/AP evaluate ranking; threshold metrics describe operating points.",
                  "No business cost matrix is available; no financial cost or bank-optimal threshold is assumed.",
                  "Fixed candidate models fit full training only; no model selection is made from maximum F1."],
    }
    return ThresholdResult(report, pd.concat(tables, ignore_index=True), pd.DataFrame(scenarios), scores, labels)


def save_reports(result: ThresholdResult, report_dir: Path = DEFAULT_REPORT_DIR) -> None:
    """Export measured reports and four figures; never serialize models or rows."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    report_dir = Path(report_dir)
    figures = report_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    result.thresholds.to_csv(report_dir / "threshold_analysis.csv", index=False)
    result.scenarios.to_csv(report_dir / "threshold_scenarios.csv", index=False)
    (report_dir / "threshold_analysis.json").write_text(json.dumps(result.report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    colors = ["#8b5a2b", "#176b91", "#9053a0"]

    def canvas(width=12, height=5):
        fig = Figure(figsize=(width, height), layout="constrained")
        FigureCanvasAgg(fig)
        return fig

    fig = canvas()
    for ax, metric in zip(fig.subplots(1, 2), ["precision", "recall"]):
        for name, color in zip(MODEL_NAMES, colors):
            table = result.thresholds.loc[result.thresholds.model.eq(name)]
            ax.plot(table.threshold, table[metric], label=name, color=color)
        ax.set(xlabel="Threshold", ylabel=metric.title(), ylim=(0, 1.03), title=f"{metric.title()} vs Threshold")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Validation Precision–Recall Trade-offs")
    fig.savefig(figures / "threshold_precision_recall_tradeoff.png", dpi=160)

    fig = canvas(10, 5)
    ax = fig.subplots()
    for name, color in zip(MODEL_NAMES, colors):
        table = result.thresholds.loc[result.thresholds.model.eq(name)]
        point = result.scenarios.loc[result.scenarios.model.eq(name) & result.scenarios.scenario.eq("MAX_F1")].iloc[0]
        ax.plot(table.threshold, table.f1, color=color, label=f"{name}: max at {point.threshold:.2f}")
        ax.scatter([point.threshold], [point.f1], color=color, edgecolors="white", s=75, zorder=3)
    ax.set(xlabel="Threshold", ylabel="F1", ylim=(0, .6), title="Validation F1 — Grid Maximum per Candidate (Not a Final Threshold)")
    ax.grid(alpha=.2)
    ax.legend(fontsize=9)
    fig.savefig(figures / "threshold_f1_comparison.png", dpi=160)

    fig = canvas(14, 4.5)
    for ax, name in zip(fig.subplots(1, 3), MODEL_NAMES):
        for label, color in [(0, "#176b91"), (1, "#c56136")]:
            values = result.validation_scores[name][result.validation_labels == label]
            ax.hist(values, bins=np.linspace(0, 1, 41), density=True, histtype="step", linewidth=1.8,
                    color=color, label=f"Target = {label}")
        ax.set(title=name, xlabel="Uncalibrated score", ylabel="Density within true class", xlim=(0, 1))
        ax.legend(fontsize=9)
        ax.grid(alpha=.2)
    fig.suptitle("Validation Score Distributions by True Class")
    fig.savefig(figures / "validation_score_distribution.png", dpi=160)

    fig = canvas(14, 5)
    for ax, scenario in zip(fig.subplots(1, 3), SCENARIOS[:3]):
        for i, (name, color) in enumerate(zip(MODEL_NAMES, colors)):
            point = result.scenarios.loc[result.scenarios.model.eq(name) & result.scenarios.scenario.eq(scenario)].iloc[0]
            if point.status == "available":
                bars = ax.bar(np.arange(3) + (i - 1) * .25, [point.precision, point.recall, point.f1], .25,
                              color=color, label=f"{name} (t={point.threshold:.2f})")
                ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
            else:
                ax.plot([], [], color=color, label=f"{name}: unavailable")
        ax.set(title=scenario.replace("_", " "), xticks=[0, 1, 2], xticklabels=["Precision", "Recall", "F1"], ylim=(0, 1))
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle("Validation Operating Scenarios — Decision Analysis Only")
    fig.savefig(figures / "threshold_scenario_comparison.png", dpi=160)


def main() -> None:
    result = run_analysis(load_training_validation())
    save_reports(result)
    print("Reference reproduction: PASS for all three candidates")
    print(result.scenarios.to_string(index=False))


if __name__ == "__main__":
    main()
