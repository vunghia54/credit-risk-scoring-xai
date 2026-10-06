"""Sixteen predeclared boosting configurations, selected using training CV only.

Run ``python -m src.boosting_tuning`` for the heavy 16 x 5 integration search.
Validation is evaluated only after both best configurations have been frozen.
The experiment exposes no test partition and never tunes thresholds.
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
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.baseline_model import DEFAULT_REPORT_DIR, THRESHOLD, calculate_metrics, positive_probabilities
from src.data_split import RANDOM_STATE
from src.imbalance_experiments import ExperimentData, make_cv, validate_xy
from src.preprocessing import build_tree_preprocessor
from src.tree_model_experiments import (
    VALIDATION_METRICS, fit_tree, importance_audit, load_logistic_references, load_tree_data,
    validate_reference,
)

ALGORITHMS = ("xgboost", "lightgbm")
TIE_ATOL = 1e-12
BASELINE_REPORT = DEFAULT_REPORT_DIR / "tree_model_experiments.json"
FIXED_PARAMETERS = {
    "xgboost": dict(objective="binary:logistic", eval_metric="logloss", random_state=RANDOM_STATE,
                    n_jobs=-1, tree_method="hist"),
    "lightgbm": dict(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1),
}


@dataclass(frozen=True)
class Candidate:
    algorithm: str
    configuration_id: str
    parameters: dict


@dataclass(frozen=True)
class TrainingData:
    """Search input deliberately contains neither validation nor test data."""

    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series


@dataclass(frozen=True)
class CandidateCV:
    """Selector input contains only predeclared parameters and training CV scores."""

    candidate: Candidate
    cv: dict
    runtime_seconds: float


@dataclass(frozen=True)
class SearchResult:
    candidates: tuple[CandidateCV, ...]
    selected: dict[str, CandidateCV]
    ranks: dict[str, int]
    algorithm_seconds: dict[str, float]
    search_seconds: float
    baseline_reproduction: dict[str, str]


@dataclass(frozen=True)
class TuningResult:
    report: dict
    cv_table: pd.DataFrame
    comparison: pd.DataFrame
    importances: pd.DataFrame


def candidate_configurations() -> tuple[Candidate, ...]:
    """Exactly eight explicit variants per algorithm; never a Cartesian product."""
    xgb = dict(n_estimators=300, learning_rate=.05, max_depth=3, min_child_weight=1,
               subsample=1.0, colsample_bytree=1.0, reg_alpha=0, reg_lambda=1)
    lgbm = dict(n_estimators=300, learning_rate=.05, num_leaves=31, max_depth=-1,
                min_child_samples=20, subsample=1.0, subsample_freq=0,
                colsample_bytree=1.0, reg_alpha=0, reg_lambda=0)
    variants = {
        "xgboost": ({}, {"max_depth": 2}, {"max_depth": 4},
                    {"n_estimators": 500, "learning_rate": .03},
                    {"n_estimators": 200, "learning_rate": .08}, {"min_child_weight": 5},
                    {"subsample": .8, "colsample_bytree": .8}, {"reg_lambda": 5}),
        "lightgbm": ({}, {"num_leaves": 15}, {"num_leaves": 63},
                     {"n_estimators": 500, "learning_rate": .03},
                     {"n_estimators": 200, "learning_rate": .08}, {"min_child_samples": 50},
                     {"subsample": .8, "subsample_freq": 1, "colsample_bytree": .8}, {"reg_lambda": 5}),
    }
    return tuple(
        Candidate(algorithm, f"{'XGB' if algorithm == 'xgboost' else 'LGBM'}-{number:02d}",
                  {**(xgb if algorithm == "xgboost" else lgbm), **override})
        for algorithm in ALGORITHMS for number, override in enumerate(variants[algorithm], 1)
    )


def build_candidate(candidate: Candidate) -> Pipeline:
    """Fresh unscaled preprocessing and the exact specified boosting configuration."""
    if candidate.algorithm not in ALGORITHMS:
        raise ValueError(f"Unsupported tuning algorithm: {candidate.algorithm}")
    model_class = XGBClassifier if candidate.algorithm == "xgboost" else LGBMClassifier
    return Pipeline([
        ("preprocessor", build_tree_preprocessor()),
        ("model", model_class(**FIXED_PARAMETERS[candidate.algorithm], **candidate.parameters)),
    ])


def training_folds(train: TrainingData) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate shared train-only group-stratified folds and verify invariants."""
    validate_xy(train.X, train.y)
    if not train.groups.index.equals(train.X.index) or train.groups.isna().any():
        raise ValueError("Raw training groups must be complete and aligned.")
    folds = list(make_cv().split(train.X, train.y, train.groups))
    coverage = np.zeros(len(train.X), dtype=int)
    for fitting, evaluation in folds:
        if np.intersect1d(fitting, evaluation).size or len(np.union1d(fitting, evaluation)) != len(train.X):
            raise ValueError("CV row overlap or incomplete fold coverage.")
        if np.intersect1d(train.groups.iloc[fitting], train.groups.iloc[evaluation]).size:
            raise ValueError("CV raw feature-group overlap.")
        coverage[evaluation] += 1
    if len(folds) != 5 or not np.all(coverage == 1):
        raise ValueError("Expected five folds with each training row evaluated once.")
    return folds


def candidate_cv(candidate: Candidate, train: TrainingData, folds: list) -> CandidateCV:
    """Fit preprocessing separately in each fitting fold; no external holdout input."""
    started = perf_counter()
    records = []
    for number, (fitting, evaluation) in enumerate(folds, 1):
        overlap = np.intersect1d(train.groups.iloc[fitting], train.groups.iloc[evaluation]).size
        if overlap:
            raise ValueError("CV feature-group overlap.")
        pipeline = fit_tree(build_candidate(candidate), train.X.iloc[fitting], train.y.iloc[fitting])
        probabilities = positive_probabilities(pipeline, train.X.iloc[evaluation])
        metrics = calculate_metrics(train.y.iloc[evaluation], probabilities)
        records.append({"fold": number, "fitting_rows": len(fitting), "evaluation_rows": len(evaluation),
                        "group_overlap": int(overlap), "roc_auc": metrics["roc_auc"],
                        "average_precision": metrics["average_precision"]})
    cv = {"folds": records, "std_ddof": 0,
          "summary": {key: {"mean": float(np.mean([row[key] for row in records])),
                            "std": float(np.std([row[key] for row in records], ddof=0))}
                      for key in ("average_precision", "roc_auc")}}
    return CandidateCV(candidate, cv, perf_counter() - started)


def rank_candidates(results: list[CandidateCV]) -> list[CandidateCV]:
    """Rank one algorithm by CV AP, true numerical ties by AUC, then config ID.

    A tie means an absolute difference <= 1e-12, with no relative tolerance.
    Standard deviations and validation outcomes never influence selection.
    """
    if not results or len({r.candidate.algorithm for r in results}) != 1:
        raise ValueError("Rank a nonempty candidate list for exactly one algorithm.")
    if len({r.candidate.configuration_id for r in results}) != len(results):
        raise ValueError("Duplicate candidate IDs.")
    for result in results:
        for key in ("average_precision", "roc_auc"):
            score = result.cv["summary"][key]["mean"]
            if not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("CV selection scores must be finite and in [0, 1].")
    remaining, ranked = list(results), []
    while remaining:
        highest_ap = max(r.cv["summary"]["average_precision"]["mean"] for r in remaining)
        ap_ties = [r for r in remaining if abs(r.cv["summary"]["average_precision"]["mean"] - highest_ap) <= TIE_ATOL]
        highest_auc = max(r.cv["summary"]["roc_auc"]["mean"] for r in ap_ties)
        auc_ties = [r for r in ap_ties if abs(r.cv["summary"]["roc_auc"]["mean"] - highest_auc) <= TIE_ATOL]
        winner = min(auc_ties, key=lambda r: r.candidate.configuration_id)
        ranked.append(winner)
        remaining.remove(winner)
    return ranked


def load_baseline_cv(path: Path = BASELINE_REPORT) -> dict:
    """Read only the STEP 10 CV reference fields needed before selection."""
    saved = json.loads(Path(path).read_text(encoding="utf-8"))
    return {algorithm: saved["tree_experiments"][algorithm]["cv"] for algorithm in ALGORITHMS}


def check_baseline_cv(result: CandidateCV, reference: dict) -> None:
    """Stop if an explicitly specified baseline fails to reproduce STEP 10 CV."""
    for metric in ("roc_auc", "average_precision"):
        for statistic in ("mean", "std"):
            actual = result.cv["summary"][metric][statistic]
            expected = reference["summary"][metric][statistic]
            if not np.isclose(actual, expected, rtol=1e-6, atol=1e-8):
                raise ValueError(f"Baseline CV reproduction failed for {result.candidate.configuration_id}: {metric} {statistic}.")


def run_search(train: TrainingData, progress: Callable[[str], None] | None = None) -> SearchResult:
    """Evaluate exactly 16 x 5 folds and select both winners before validation."""
    started = perf_counter()
    references = load_baseline_cv()
    folds = training_folds(train)
    results, selected, ranks, runtimes, reproduction = [], {}, {}, {}, {}
    for algorithm in ALGORITHMS:
        algorithm_started = perf_counter()
        algorithm_results = []
        for candidate in candidate_configurations():
            if candidate.algorithm != algorithm:
                continue
            result = candidate_cv(candidate, train, folds)
            if candidate.configuration_id.endswith("-01"):
                check_baseline_cv(result, references[algorithm])
                reproduction[algorithm] = "PASS"
            algorithm_results.append(result)
            results.append(result)
            if progress:
                progress(f"{candidate.configuration_id}: train CV AP={result.cv['summary']['average_precision']['mean']:.6f}; AUC={result.cv['summary']['roc_auc']['mean']:.6f}")
        ordered = rank_candidates(algorithm_results)
        selected[algorithm] = ordered[0]
        ranks.update({row.candidate.configuration_id: rank for rank, row in enumerate(ordered, 1)})
        runtimes[algorithm] = perf_counter() - algorithm_started
    return SearchResult(tuple(results), selected, ranks, runtimes, perf_counter() - started, reproduction)


def cv_results_table(search: SearchResult) -> pd.DataFrame:
    rows = []
    for result in search.candidates:
        candidate = result.candidate
        row = {"algorithm": candidate.algorithm, "configuration_id": candidate.configuration_id,
               "parameters": json.dumps(candidate.parameters, sort_keys=True),
               "rank_within_algorithm": search.ranks[candidate.configuration_id],
               "runtime_seconds": result.runtime_seconds}
        for metric in ("average_precision", "roc_auc"):
            for statistic in ("mean", "std"):
                row[f"cv_{metric}_{statistic}"] = result.cv["summary"][metric][statistic]
        rows.append(row)
    return pd.DataFrame(rows)


def run_tuning(data: ExperimentData, progress: Callable[[str], None] | None = None) -> TuningResult:
    """Complete train-only selection first, then fit/evaluate the two winners once."""
    started = perf_counter()
    train = TrainingData(data.X_train, data.y_train, data.groups_train)
    search = run_search(train, progress)
    if progress:
        progress("Selection frozen from TRAIN CV: " + ", ".join(r.candidate.configuration_id for r in search.selected.values()))
    # First validation access in this function occurs AFTER both winners exist.
    validate_xy(data.X_validation, data.y_validation)
    saved = json.loads(BASELINE_REPORT.read_text(encoding="utf-8"))
    baselines = {algorithm: saved["tree_experiments"][algorithm] for algorithm in ALGORITHMS}
    for entry in baselines.values():
        validate_reference(entry, len(data.X_validation))
    logistic = load_logistic_references(len(data.X_validation))
    tuned, audits = {}, []
    for algorithm in ALGORITHMS:
        chosen = search.selected[algorithm]
        fitted = fit_tree(build_candidate(chosen.candidate), train.X, train.y)
        audit = importance_audit(algorithm, fitted, train.X)
        audit["configuration_id"] = chosen.candidate.configuration_id
        audits.append(audit)
        probabilities = positive_probabilities(fitted, data.X_validation)
        validation = calculate_metrics(data.y_validation, probabilities)
        model = fitted.named_steps["model"]
        rounds = model.get_booster().num_boosted_rounds() if algorithm == "xgboost" else model.booster_.current_iteration()
        if rounds != chosen.candidate.parameters["n_estimators"]:
            raise ValueError("Selected model did not fit the specified boosting rounds.")
        tuned[algorithm] = {
            "configuration_id": chosen.candidate.configuration_id,
            "parameters": {**FIXED_PARAMETERS[algorithm], **chosen.candidate.parameters},
            "cv": chosen.cv, "validation": validation, "number_of_features": 13,
            "fitted_rounds": int(rounds), "feature_importances": audit.to_dict(orient="records"),
            "cv_change_vs_baseline": {metric: chosen.cv["summary"][metric]["mean"] - baselines[algorithm]["cv"]["summary"][metric]["mean"]
                                      for metric in ("average_precision", "roc_auc")},
            "validation_change_vs_baseline": {metric: validation[metric] - baselines[algorithm]["validation"][metric]
                                              for metric in ("average_precision", "roc_auc")},
            "validation_minus_cv": {metric: validation[metric] - chosen.cv["summary"][metric]["mean"]
                                    for metric in ("average_precision", "roc_auc")},
        }
    rows = []
    comparison_entries = [
        ("Logistic Baseline", "STEP 8 reference", logistic["logistic_baseline"]),
        ("Logistic Balanced", "STEP 9 reference", logistic["logistic_balanced"]),
        ("XGBoost Baseline", "XGB-01", baselines["xgboost"]),
        ("XGBoost Tuned", tuned["xgboost"]["configuration_id"], tuned["xgboost"]),
        ("LightGBM Baseline", "LGBM-01", baselines["lightgbm"]),
        ("LightGBM Tuned", tuned["lightgbm"]["configuration_id"], tuned["lightgbm"]),
    ]
    for name, configuration, entry in comparison_entries:
        row = {"model": name, "configuration": configuration}
        for metric in ("roc_auc", "average_precision"):
            for statistic in ("mean", "std"):
                row[f"cv_{metric}_{statistic}"] = entry["cv"]["summary"][metric][statistic]
        row.update({f"validation_{metric}": entry["validation"][metric] for metric in VALIDATION_METRICS})
        rows.append(row)
    report = {
        "random_state": RANDOM_STATE, "threshold": THRESHOLD,
        "selection": {"primary": "CV Average Precision mean", "secondary_on_numerical_tie": "CV ROC-AUC mean",
                      "tie_atol": TIE_ATOL, "final_tiebreak": "configuration_id ascending",
                      "validation_used": False, "candidate_count": 16, "fits_in_search": 80},
        "cv_design": {"splitter": "StratifiedGroupKFold", "n_splits": 5, "shuffle": True,
                      "std_ddof": 0, "shared_folds": True, "groups": "raw training feature-group metadata"},
        "fixed_parameters": FIXED_PARAMETERS, "baseline_reproduction": search.baseline_reproduction,
        "candidates": [{"algorithm": r.candidate.algorithm, "configuration_id": r.candidate.configuration_id,
                        "parameters": r.candidate.parameters, "cv": r.cv,
                        "rank_within_algorithm": search.ranks[r.candidate.configuration_id],
                        "runtime_seconds": r.runtime_seconds} for r in search.candidates],
        "selected": tuned, "untuned_baselines": baselines, "logistic_references": logistic,
        "runtime_seconds": {**search.algorithm_seconds, "total_search": search.search_seconds,
                            "total_experiment": perf_counter() - started},
        "notes": [
            "Predeclared targeted search: eight configurations per algorithm, no Cartesian expansion.",
            "All selection uses train-only CV; validation is evaluated once per selected model after both selections are frozen.",
            "Same raw feature groups and per-fold tree preprocessing, without scaling or imbalance changes.",
            "No test evaluation, early stopping, threshold tuning, calibration, SHAP or model serialization.",
            "CV std is fold variability (ddof=0), not a confidence interval; selection among candidates can cause CV optimism.",
            "Native feature importance is audit only, not causal interpretation or SHAP. Scores are not calibrated PD.",
            "Runtimes are wall-clock measurements on this environment, not standardized hardware benchmarks; export excluded.",
            "No selected configuration is revised after observing validation; no final production model is declared.",
        ],
    }
    return TuningResult(report, cv_results_table(search), pd.DataFrame(rows), pd.concat(audits, ignore_index=True))


def save_reports(result: TuningResult, report_dir: Path = DEFAULT_REPORT_DIR) -> None:
    """Write measured reports and two figures; no models or processed samples."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    report_dir = Path(report_dir)
    figures = report_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    (report_dir / "boosting_tuning_results.json").write_text(json.dumps(result.report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    result.cv_table.to_csv(report_dir / "boosting_tuning_cv_results.csv", index=False)
    result.comparison.to_csv(report_dir / "tuned_boosting_comparison.csv", index=False)
    result.importances.to_csv(report_dir / "tuned_boosting_feature_importance.csv", index=False)
    figure = Figure(figsize=(12, 5.5), layout="constrained")
    FigureCanvasAgg(figure)
    for ax, algorithm, color in zip(figure.subplots(1, 2), ALGORITHMS, ["#176b91", "#9053a0"]):
        table = result.cv_table.loc[result.cv_table.algorithm.eq(algorithm)]
        positions = np.arange(8)
        bars = ax.bar(positions, table.cv_average_precision_mean, color=color,
                      yerr=table.cv_average_precision_std, capsize=3)
        ax.bar_label(bars, fmt="%.4f", fontsize=8, padding=10)
        ax.set(xticks=positions, xticklabels=table.configuration_id, ylim=(0, .5),
               ylabel="CV Average Precision", title="XGBoost" if algorithm == "xgboost" else "LightGBM")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    figure.suptitle("Targeted Boosting Search — Training CV\nMean ± fold standard deviation (5 group-aware folds)", fontsize=13)
    figure.savefig(figures / "boosting_tuning_cv_ap.png", dpi=160)
    figure = Figure(figsize=(10, 5.5), layout="constrained")
    FigureCanvasAgg(figure)
    for ax, algorithm in zip(figure.subplots(1, 2), ALGORITHMS):
        baseline = result.report["untuned_baselines"][algorithm]["validation"]
        tuned = result.report["selected"][algorithm]["validation"]
        keys = ["roc_auc", "average_precision"]
        for offset, label, metrics, color in [(-.18, "Baseline", baseline, "#64748b"), (.18, "CV-selected", tuned, "#176b91")]:
            bars = ax.bar(np.arange(2) + offset, [metrics[key] for key in keys], .36, label=label, color=color)
            ax.bar_label(bars, fmt="%.4f", fontsize=9, padding=3)
        ax.set(xticks=[0, 1], xticklabels=["ROC-AUC", "Average Precision"], ylim=(0, 1.05),
               title="XGBoost" if algorithm == "xgboost" else "LightGBM", ylabel="Validation score")
        ax.legend(loc="upper center", ncol=2, fontsize=9)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    figure.suptitle("Boosting Baseline vs Tuned — Fixed Validation Set", fontsize=13)
    figure.savefig(figures / "tuned_boosting_validation_comparison.png", dpi=160)


def main() -> None:
    result = run_tuning(load_tree_data(), progress=lambda message: print(message, flush=True))
    save_reports(result)
    print(result.comparison.to_string(index=False))
    print(json.dumps(result.report["runtime_seconds"], indent=2))


if __name__ == "__main__":
    main()
