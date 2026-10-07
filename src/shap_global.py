"""Global raw-margin SHAP for frozen XGB-01 on validation features only.

The CLI fits TRAIN only. No test partition or validation labels are exposed to
the analysis API. Importing this module does not read data, fit or explain.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from src import final_model as frozen
from src.baseline_model import DEFAULT_REPORT_DIR
from src.data_split import DEFAULT_DATA_PATH, load_data, split_data
from src.preprocessing import OUTPUT_FEATURE_COLUMNS, validate_transformed
from src.tree_model_experiments import fit_tree

RANDOM_STATE = 42
ADDITIVITY_SAMPLE_SIZE = 512
PLOTTING_SAMPLE_SIZE = 5_000
ADDITIVITY_ATOL = 1e-5
ADDITIVITY_RTOL = 1e-5
TEST_NOT_USED = (
    "TEST was not used for SHAP values, feature ranking, plot selection, effect "
    "patterns or any explainability decision. The analysis API receives TRAIN "
    "features/labels and VALIDATION features only, with no validation target. "
    "The existing splitter still performs its approved integrity checks on all "
    "partitions; its returned test partition is never accessed by this module."
)
LIMITATIONS = [
    "SHAP explains model behavior, not causal relationships or credit policy rules.",
    "Values are raw-margin / log-odds contributions, not direct probability changes or percentage-point PD effects.",
    "Attributions depend on the tree-path-dependent reference distribution and feature dependence assumptions; correlated features can share attribution.",
    "Importance is mean absolute contribution on validation, not statistical significance or model-selection evidence.",
    "Inputs include training-median imputation and engineered missing/special-value indicators; they are not untouched raw columns.",
    "Marginal low/high summaries and scatter plots do not establish a monotonic, causal or counterfactual response.",
    "Native gain importance and SHAP summarize different quantities; neither is ground truth and their numerical units differ.",
    "Internal validation explanations do not establish external generalization, fairness, production suitability or regulatory PD.",
    "The consumed internal test remains excluded from explainability development. No frozen decision is revised.",
]


@dataclass(frozen=True)
class ExplanationData:
    """Minimal train-fitting/validation-explanation boundary, without holdout y."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_validation: pd.DataFrame


@dataclass(frozen=True)
class GlobalShapResult:
    """Global aggregates and transient validation explanations for plotting."""

    report: dict
    importance: pd.DataFrame
    native_comparison: pd.DataFrame
    transformed_validation: pd.DataFrame
    explanation: shap.Explanation


def load_explanation_data() -> ExplanationData:
    """Reuse approved split integrity checks, expose only TRAIN and validation X."""
    splits = split_data(load_data())
    return ExplanationData(splits.train.X, splits.train.y, splits.validation.X)


def verify_frozen_specification(reference: dict) -> dict:
    """Verify metadata against the previously committed final-model decision."""
    actual = frozen.get_frozen_metadata()
    if any(reference.get(k) != v for k, v in actual.items()):
        raise ValueError("Frozen specification differs from the STEP 14 artifact.")
    if (actual["final_model"] != "XGBoost Baseline / XGB-01"
            or actual["calibration_method"] != "none"
            or (actual["default_audit_threshold"], actual["development_operating_threshold"]) != (.50, .19)
            or actual["random_state"] != RANDOM_STATE):
        raise ValueError("Approved model, calibration, thresholds or seed changed.")
    return actual


def deterministic_positions(rows: int, maximum: int) -> np.ndarray:
    """Sample positions using only population size and seed, never y or scores."""
    if rows < 1 or maximum < 1:
        raise ValueError("Row count and maximum sample size must be positive.")
    return np.sort(np.random.default_rng(RANDOM_STATE).choice(rows, size=min(rows, maximum), replace=False))


def validate_shap_output(explanation: shap.Explanation, X: pd.DataFrame,
                         expected_value: float) -> None:
    """Require binary raw-margin values aligned to all 13 named model inputs."""
    if not isinstance(X, pd.DataFrame) or X.empty or X.columns.tolist() != OUTPUT_FEATURE_COLUMNS:
        raise ValueError("SHAP inputs must retain the exact 13-feature DataFrame schema.")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Transformed SHAP inputs must be finite.")
    values = np.asarray(explanation.values)
    base = np.asarray(explanation.base_values)
    if values.shape != X.shape or not np.isfinite(values).all():
        raise ValueError("SHAP values must be a finite (rows, 13) matrix.")
    if (not np.isfinite(expected_value) or base.shape != (len(X),)
            or not np.isfinite(base).all() or not np.allclose(base, expected_value, rtol=0, atol=1e-10)):
        raise ValueError("SHAP base values must match one finite expected raw margin.")
    if list(explanation.feature_names) != OUTPUT_FEATURE_COLUMNS:
        raise ValueError("SHAP feature names or ordering differ from model inputs.")
    if not np.array_equal(np.asarray(explanation.data), X.to_numpy()):
        raise ValueError("SHAP explanation data differs from transformed validation.")


def check_additivity(values: np.ndarray, base_values: np.ndarray | float,
                     raw_margins: np.ndarray, *, atol: float = ADDITIVITY_ATOL,
                     rtol: float = ADDITIVITY_RTOL) -> dict:
    """Check base + sum(SHAP) against margins; fail before publishing on drift."""
    values = np.asarray(values, dtype=float)
    margins = np.asarray(raw_margins, dtype=float)
    base = np.asarray(base_values, dtype=float)
    if (values.ndim != 2 or values.shape[0] == 0 or margins.shape != (len(values),)
            or base.shape not in ((), (len(values),))):
        raise ValueError("Additivity arrays must have aligned row shapes.")
    if not all(np.isfinite(a).all() for a in (values, margins, base)):
        raise ValueError("Additivity inputs must be finite.")
    if not np.isfinite([atol, rtol]).all() or min(atol, rtol) < 0:
        raise ValueError("Additivity tolerances must be finite and nonnegative.")
    reconstructed = base + values.sum(axis=1)
    errors = np.abs(reconstructed - margins)
    result = {"status": "PASS", "rows": len(values),
              "max_absolute_error": float(errors.max()), "mean_absolute_error": float(errors.mean()),
              "atol": atol, "rtol": rtol,
              "criterion": "abs(base + sum(SHAP) - margin) <= atol + rtol * abs(margin)"}
    if not np.all(errors <= atol + rtol * np.abs(margins)):
        raise ValueError(f"Raw-margin additivity failed: {result}")
    return result


def global_importance(values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Rank full-validation mean |SHAP|; ties retain approved schema order."""
    values = np.asarray(values, dtype=float)
    if (feature_names != OUTPUT_FEATURE_COLUMNS or values.ndim != 2
            or values.shape[1] != len(feature_names) or values.shape[0] == 0
            or not np.isfinite(values).all()):
        raise ValueError("Importance requires finite SHAP values for exactly 13 features.")
    means = np.mean(np.abs(values), axis=0)
    total = float(means.sum())
    table = pd.DataFrame({"feature": feature_names, "mean_absolute_shap": means,
                          "relative_importance": means / total if total > 0 else np.zeros_like(means)})
    table = table.sort_values("mean_absolute_shap", ascending=False, kind="stable").reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return table


def compare_native_importance(importance: pd.DataFrame, native: pd.DataFrame,
                              fitted_importances: np.ndarray) -> pd.DataFrame:
    """Join saved STEP 10 XGBoost gain audit after checking the frozen fit matches."""
    required = {"model", "feature", "importance", "rank"}
    if not required.issubset(native.columns):
        raise ValueError("Native importance artifact schema is incomplete.")
    selected = native.loc[native.model.eq("XGBoost")].copy()
    if (len(selected) != 13 or not selected.feature.is_unique
            or set(selected.feature) != set(OUTPUT_FEATURE_COLUMNS)
            or set(selected["rank"]) != set(range(1, 14))):
        raise ValueError("Native XGBoost artifact must contain all 13 unique features/ranks.")
    saved = selected.set_index("feature").loc[OUTPUT_FEATURE_COLUMNS, "importance"].to_numpy(dtype=float)
    fitted = np.asarray(fitted_importances, dtype=float)
    if (fitted.shape != (13,) or not np.isfinite(saved).all() or (saved < 0).any()
            or not np.allclose(saved, fitted, rtol=1e-6, atol=1e-8)):
        raise ValueError("Frozen fitted native importance differs from STEP 10 artifact.")
    merged = importance.rename(columns={"rank": "shap_rank"}).merge(
        selected.rename(columns={"importance": "xgboost_native_importance", "rank": "native_rank"}),
        on="feature", validate="one_to_one")
    return merged[["feature", "shap_rank", "mean_absolute_shap", "xgboost_native_importance", "native_rank"]].sort_values("shap_rank").reset_index(drop=True)


def feature_pattern_summary(X: pd.DataFrame, values: np.ndarray,
                            importance: pd.DataFrame) -> list[dict]:
    """Describe top-five marginal low/high associations without target or causality.

    Use <=25th and >=75th percentiles when distinct. For tied quartiles, compare
    the minimum-value group with the greater-than-minimum group. These are
    descriptive groups, not business cutoffs or a monotonicity test.
    """
    rows = []
    for name in importance.feature.head(5):
        x = X[name].to_numpy(dtype=float)
        effects = values[:, X.columns.get_loc(name)]
        q25, q75 = np.quantile(x, [.25, .75])
        if q25 < q75:
            low, high = x <= q25, x >= q75
            definition = f"low: value <= {q25:.6g}; high: value >= {q75:.6g} (validation quartiles)"
        else:
            low, high = x == x.min(), x > x.min()
            definition = f"tied quartiles; low: value = {x.min():.6g}; high: value > {x.min():.6g}"
        groups = {}
        for label, mask in (("low", low), ("high", high)):
            groups[label] = {"rows": int(mask.sum()),
                             "mean_signed_shap": float(effects[mask].mean()) if mask.any() else None,
                             "positive_shap_fraction": float((effects[mask] > 0).mean()) if mask.any() else None}
        rows.append({"feature": name, "group_definition": definition, **groups,
                     "interpretation": "Descriptive association with model raw-margin output; no simple monotonic pattern is established by these group summaries."})
    return rows


def explain_validation(pipeline: Pipeline, X_validation: pd.DataFrame) -> tuple[pd.DataFrame, shap.Explanation, dict]:
    """Explain every transformed validation row without labels or fitting."""
    X = pipeline.named_steps["preprocessor"].transform(X_validation)
    validate_transformed(X_validation, X)
    model = pipeline.named_steps["model"]
    if model.n_features_in_ != 13 or list(model.feature_names_in_) != OUTPUT_FEATURE_COLUMNS:
        raise ValueError("Fitted model must retain the exact named 13-feature schema.")
    started = perf_counter()
    explainer = shap.TreeExplainer(model, model_output="raw", feature_perturbation="tree_path_dependent")
    explanation = explainer(X, check_additivity=True, approximate=False)
    shap_seconds = perf_counter() - started
    expected = np.asarray(explainer.expected_value)
    if expected.size != 1:
        raise ValueError("Expected a single binary raw-margin base value.")
    expected_value = float(expected.item())
    validate_shap_output(explanation, X, expected_value)
    positions = deterministic_positions(len(X), ADDITIVITY_SAMPLE_SIZE)
    audit = check_additivity(explanation.values[positions], explanation.base_values[positions],
                             model.predict(X.iloc[positions], output_margin=True))
    audit.update({"sample_selection": "uniform positions without replacement, sorted, seed 42",
                  "positions_sha256": hashlib.sha256(positions.astype('<i8').tobytes()).hexdigest(),
                  "library_full_validation_additivity_check": True})
    return X, explanation, {"expected_value": expected_value, "additivity": audit, "shap_seconds": shap_seconds}


def run_analysis(data: ExplanationData, reference: dict, native: pd.DataFrame) -> GlobalShapResult:
    """Fit the unchanged frozen pipeline on TRAIN and explain full validation X."""
    started = perf_counter()
    specification = verify_frozen_specification(reference)
    pipeline = frozen.build_final_model_pipeline()
    parameters = pipeline.named_steps["model"].get_params()
    if any(parameters[k] != v for k, v in specification["model_parameters"].items()):
        raise ValueError("Pipeline parameters drifted from the frozen specification.")
    pipeline = fit_tree(pipeline, data.X_train, data.y_train)
    X, explanation, audit = explain_validation(pipeline, data.X_validation)
    importance = global_importance(explanation.values, X.columns.tolist())
    comparison = compare_native_importance(importance, native, pipeline.named_steps["model"].feature_importances_)
    patterns = feature_pattern_summary(X, explanation.values, importance)
    if frozen.get_frozen_metadata() != specification:
        raise ValueError("Frozen configuration mutated during SHAP analysis.")
    report = {
        "final_model": frozen.FINAL_MODEL_NAME, "frozen_specification": specification,
        "calibration": "none", "random_state": RANDOM_STATE, "fit_partition": "training only",
        "train_rows": len(data.X_train), "shap_dataset": "validation", "validation_rows": len(X),
        "transformed_feature_count": len(X.columns), "transformed_features": X.columns.tolist(),
        "shap_output_shape": list(explanation.values.shape), "shap_output_space": "raw margin / log-odds",
        "explainer": {"class": "shap.TreeExplainer", "shap_version": shap.__version__,
                      "model_output": "raw", "feature_perturbation": "tree_path_dependent",
                      "approximate": False, "background": "training path counts recorded in the fitted trees; no external background data"},
        "expected_value": audit["expected_value"], "additivity_check": audit["additivity"],
        "global_importance": importance.to_dict(orient="records"),
        "ranking_tie_policy": "stable sort descending; ties preserve approved feature schema order",
        "native_importance_comparison": comparison.to_dict(orient="records"),
        "native_importance_definition": "normalized XGBoost gain from the existing STEP 10 audit; reproduced by the frozen fit",
        "feature_effect_patterns": patterns,
        "plotting": {"rows": min(len(X), PLOTTING_SAMPLE_SIZE), "population_rows": len(X),
                     "selection": "uniform validation positions without replacement, seed 42; independent of target and model score",
                     "global_importance_population": "full validation, no sampling",
                     "dependence_x_axis": "symlog, linthresh=1; all selected observations retained"},
        "test_not_used_statement": TEST_NOT_USED, "interpretation_limitations": LIMITATIONS.copy(),
        "no_model_changes": True, "runtime_seconds": {"shap_calculation": audit["shap_seconds"],
                                                       "fit_explain_and_aggregate": perf_counter() - started},
    }
    return GlobalShapResult(report, importance, comparison, X, explanation)


def safe_feature_name(name: str) -> str:
    """Produce a filesystem-safe suffix from an approved feature name."""
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")


def figure_names(importance: pd.DataFrame) -> list[str]:
    """Choose dependence plot features from measured global rank, never a list."""
    return ["shap_global_bar.png", "shap_global_beeswarm.png", *[
        f"shap_dependence_{i:02d}_{safe_feature_name(name)}.png"
        for i, name in enumerate(importance.feature.head(3), 1)]]


def save_figures(result: GlobalShapResult, directory: Path) -> None:
    """Save five global figures; deterministic subsampling only affects plots."""
    directory.mkdir(parents=True, exist_ok=True)
    names = figure_names(result.importance)
    positions = deterministic_positions(len(result.transformed_validation), PLOTTING_SAMPLE_SIZE)
    X = result.transformed_validation.iloc[positions]
    values = result.explanation.values[positions]
    order = np.array([X.columns.get_loc(name) for name in result.importance.feature])
    def save(fig: plt.Figure, name: str) -> None:
        fig.tight_layout()
        fig.savefig(directory / name, dpi=160, bbox_inches="tight")
        plt.close(fig)
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = plt.subplots(figsize=(12, 7))
        table = result.importance.iloc[::-1]
        ax.barh(table.feature, table.mean_absolute_shap, color="#356b8c")
        ax.set(title=f"Frozen XGB-01: Global SHAP Importance\nFull validation ({len(result.transformed_validation):,} rows)",
               xlabel="Mean absolute SHAP value (raw-margin units)")
        save(fig, names[0])
        plot_explanation = shap.Explanation(values=values, base_values=result.explanation.base_values[positions],
                                            data=X.to_numpy(), feature_names=X.columns.tolist())
        # SHAP 0.52 beeswarm jitter uses the legacy NumPy RNG; restore caller state.
        rng_state = np.random.get_state()
        try:
            np.random.seed(RANDOM_STATE)
            ax = shap.plots.beeswarm(plot_explanation, max_display=13, order=order,
                                     show=False, plot_size=(12, 8), alpha=.55, s=8)
            ax.set(title=f"Frozen XGB-01: Validation SHAP Distribution\nDeterministic plotting sample ({len(X):,} rows)",
                   xlabel="SHAP value (raw margin): lower output < 0 < higher output")
            save(ax.figure, names[1])
        finally:
            np.random.set_state(rng_state)
        for rank, (feature, name) in enumerate(zip(result.importance.feature.head(3), names[2:], strict=True), 1):
            fig, ax = plt.subplots(figsize=(9, 5.5))
            ax.scatter(X[feature], values[:, X.columns.get_loc(feature)], s=9, alpha=.3, color="#356b8c", edgecolors="none")
            ax.axhline(0, color="gray", linewidth=.8, linestyle="--")
            ax.set_xscale("symlog", linthresh=1)
            # Explicit limits avoid linear autoscale padding creating a large
            # empty negative range after switching nonnegative data to symlog.
            lower, upper = float(X[feature].min()), float(X[feature].max())
            ax.set_xlim(min(0.0, lower * 1.05), max(1.0, upper * 1.05))
            ax.set(title=f"Global Dependence {rank}: {feature}\nModel association only; not a causal effect ({len(X):,} validation rows)",
                   xlabel="Transformed feature value (symlog; linear within +/-1)", ylabel="SHAP value (raw-margin units)")
            save(fig, name)


def render_report(report: dict) -> str:
    """Render English global findings and numerical patterns without causal claims."""
    audit = report["additivity_check"]
    rows = [f"| {r['rank']} | {r['feature']} | {r['mean_absolute_shap']:.6f} | {r['relative_importance']:.2%} |" for r in report["global_importance"]]
    comparisons = [f"| {r['feature']} | {r['shap_rank']} | {r['mean_absolute_shap']:.6f} | {r['native_rank']} | {r['xgboost_native_importance']:.6f} |" for r in report["native_importance_comparison"]]
    patterns = []
    for row in report["feature_effect_patterns"]:
        descriptions = []
        for label in ("low", "high"):
            group = row[label]
            if group["rows"]:
                descriptions.append(f"{label} group: n={group['rows']:,}, mean signed SHAP={group['mean_signed_shap']:+.6f}, positive SHAP fraction={group['positive_shap_fraction']:.2%}")
            else:
                descriptions.append(f"{label} group: unavailable (constant feature)")
        patterns.append(f"- **{row['feature']}** — {row['group_definition']}. {'; '.join(descriptions)}. {row['interpretation']}")
    return "\n".join([
        "# Global SHAP Explainability", "", "## Scope", "",
        "Global explanation of the previously frozen final model. SHAP explains model behavior, not causal relationships. "
        "No feature selection, model change, retuning, local customer explanation or policy derivation is performed.", "",
        "## Frozen Model", "", f"**{report['final_model']}**, calibration **none**, audit threshold **0.50**, development operating threshold **0.19**, random_state **42**. "
        f"The original `build_final_model_pipeline()` was fitted only on {report['train_rows']:,} TRAIN rows. "
        "The frozen configuration is checked against STEP 14; native importance also reproduces the existing STEP 10 frozen baseline audit. "
        "SHAP does not use either classification threshold to select observations.", "",
        "## Explanation Dataset", "", f"All **{report['validation_rows']:,} VALIDATION rows** and all **13 transformed features** are used for global importance. "
        "No validation target or target-conditioned segmentation is used. " + TEST_NOT_USED, "",
        f"The beeswarm and dependence plots use the same deterministic **{report['plotting']['rows']:,}-row** validation sample "
        "to limit overplotting, drawn uniformly without replacement with seed 42. Sampling uses neither target nor model score. "
        "The bar/table and top-feature selection use full-validation SHAP values.", "",
        "## SHAP Output Space", "",
        "`shap.TreeExplainer(model_output=\"raw\", feature_perturbation=\"tree_path_dependent\")` explains XGBoost's binary:logistic margin, "
        "the score before sigmoid, on a log-odds scale. The training path counts stored in the trees provide the reference; no validation/test background is fitted. "
        "Exact Tree SHAP is requested (`approximate=False`). "
        "Positive contributions raise the model margin relative to its base value; negative contributions lower it. "
        "A SHAP contribution is not a direct probability increase, a probability percentage, or a causal effect. "
        "See the [official TreeExplainer documentation](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html).", "",
        f"Expected/base raw margin: **{report['expected_value']:.9f}**. It is the explainer reference value, not validation prevalence or a calibrated PD.", "",
        "## Additivity Verification", "",
        f"**{audit['status']}**: `expected_value + sum(SHAP values)` agrees with `predict(output_margin=True)` on "
        f"{audit['rows']} deterministic validation observations. Maximum absolute error: **{audit['max_absolute_error']:.9g}**; "
        f"mean absolute error: **{audit['mean_absolute_error']:.9g}**. "
        f"Criterion: absolute error <= {audit['atol']:g} + {audit['rtol']:g} * abs(raw margin), allowing float32 tree arithmetic rounding. "
        "The SHAP library's additivity check was also enabled for the full validation call. Failures stop report generation.", "",
        "## Global Feature Importance", "", "| Rank | Transformed feature | Mean absolute SHAP | Relative importance |",
        "|---:|---|---:|---:|", *rows, "",
        "Mean absolute SHAP measures contribution magnitude, not direction. Relative importance divides each magnitude by their sum "
        "(all-zero contributions would produce zero relative values). Stable ties retain the approved feature order. "
        "These quantities are neither causal importance nor directly comparable to Logistic Regression coefficients.", "",
        "## Feature Effect Patterns", "",
        "The following summaries use all validation observations for the measured top five. Distinct lower/upper quartiles define low/high groups; "
        "when quartiles tie, the minimum-value group is compared with larger values. Positive/negative means describe association with higher/lower model margin "
        "relative to the reference. These observational summaries do not isolate other features or establish monotonic effects.", "",
        *patterns, "",
        "Dependence plots show transformed values against their SHAP contributions. The x-axis uses symlog with a linear region within +/-1 "
        "to show extreme values without dropping or clipping observations. Vertical spread can reflect feature interactions and other model behavior; "
        "it does not establish a causal mechanism.", "",
        "## SHAP vs Native Tree Importance", "",
        "Source: `reports/tree_feature_importance_audit.csv`, XGBoost entries only. Native importance is normalized gain from training tree splits. "
        "SHAP ranks mean absolute raw-margin contributions across validation observations. Differences are expected; neither ranking is ground truth.", "",
        "| Feature | SHAP rank | Mean absolute SHAP | Native rank | Native gain importance |", "|---|---:|---:|---:|---:|", *comparisons, "",
        "## Interpretation Guidelines", "",
        "- Interpret direction as associated with higher/lower model output, not as causing delinquency.",
        "- `age=0` is treated as missing and imputed using the training median.",
        "- Delinquency values 96/98 are replaced with missing in the preprocessing working copy and imputed; raw data is unchanged.",
        "- `has_special_delinquency_value` records special-pattern presence before replacement.",
        "- `MonthlyIncome_missing` and `NumberOfDependents_missing` are engineered model inputs. Their contributions explain missingness indicators, not observed income or dependent counts.",
        "- All 13 approved inputs are retained. Low SHAP importance does not trigger feature removal.", "",
        "## Limitations", "", *[f"- {line}" for line in report["interpretation_limitations"]], "",
        "## What SHAP Does Not Establish", "",
        "SHAP does not establish causality, individual recourse, lending policy, fairness, calibrated probability changes or a production/regulatory PD. "
        "No individual customer, waterfall, force plot or local reason code is selected. "
        "The internal test was consumed in STEP 15 and was not reused for this explainability analysis. "
        "No model, preprocessing, calibration, threshold or hyperparameter decision was revised.", "",
        f"Runtime (seconds): {json.dumps(report['runtime_seconds'], sort_keys=True)}. SHAP version: {report['explainer']['shap_version']}.", "",
    ])


def save_reports(result: GlobalShapResult, report_dir: Path = DEFAULT_REPORT_DIR) -> None:
    """Export only new global summaries and five plots; no model or per-row dump."""
    report_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    save_figures(result, report_dir / "figures")
    result.report["runtime_seconds"]["figure_export"] = perf_counter() - started
    result.importance.to_csv(report_dir / "shap_global_importance.csv", index=False)
    result.native_comparison.to_csv(report_dir / "shap_native_importance_comparison.csv", index=False)
    (report_dir / "shap_global_results.json").write_text(json.dumps(result.report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (report_dir / "shap_global_explainability.md").write_text(render_report(result.report), encoding="utf-8")


def main() -> None:
    """Run TRAIN fitting and VALIDATION SHAP, leaving all prior artifacts intact."""
    started = perf_counter()
    reference_path = DEFAULT_REPORT_DIR / "final_model_selection.json"
    native_path = DEFAULT_REPORT_DIR / "tree_feature_importance_audit.csv"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    verify_frozen_specification(reference)
    print("Frozen XGB-01 verified: calibration=none; thresholds=0.50/0.19; TRAIN fit, VALIDATION SHAP only.", flush=True)
    result = run_analysis(load_explanation_data(), reference, pd.read_csv(native_path))
    result.report["runtime_seconds"]["load_fit_explain_and_aggregate"] = perf_counter() - started
    result.report["provenance"] = {"freeze_commit": "b6efa63", "prior_final_evaluation_commit": "025d063",
        "raw_dataset_sha256": hashlib.sha256(DEFAULT_DATA_PATH.read_bytes()).hexdigest(),
        "frozen_specification_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "native_importance_sha256": hashlib.sha256(native_path.read_bytes()).hexdigest()}
    save_reports(result)
    print(result.importance.to_string(index=False))
    print(json.dumps({"additivity": result.report["additivity_check"], "expected_value": result.report["expected_value"],
                      "runtime_seconds": result.report["runtime_seconds"], "total_wall_seconds": perf_counter() - started}, indent=2))
    print(TEST_NOT_USED)


if __name__ == "__main__":
    main()
