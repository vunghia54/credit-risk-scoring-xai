"""Deterministic local SHAP for six score-selected VALIDATION observations.

Importing this module does not fit, predict, select cases or explain. The CLI
fits the frozen pipeline on TRAIN only, then uses validation features without
validation targets or the returned test partition.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src import final_model as frozen
from src.baseline_model import DEFAULT_REPORT_DIR, positive_probabilities
from src.data_split import DEFAULT_DATA_PATH, FEATURE_COLUMNS
from src.preprocessing import OUTPUT_FEATURE_COLUMNS
from src.shap_global import (
    ExplanationData, check_additivity, explain_validation, load_explanation_data,
    verify_frozen_specification,
)
from src.tree_model_experiments import fit_tree

RANDOM_STATE = 42
NEUTRAL_ATOL = 1e-8
OPERATING_THRESHOLD = frozen.DEVELOPMENT_OPERATING_THRESHOLD
AUDIT_THRESHOLD = frozen.DEFAULT_AUDIT_THRESHOLD
CASE_IDS = (
    "VAL_CASE_01_LOW", "VAL_CASE_02_MEDIAN", "VAL_CASE_03_HIGH",
    "VAL_CASE_04_BELOW_019", "VAL_CASE_05_ABOVE_019", "VAL_CASE_06_NEAR_050",
)
SELECTION_RULES = ("LOW_SCORE", "MEDIAN_SCORE", "HIGH_SCORE", "JUST_BELOW_0_19",
                   "JUST_ABOVE_0_19", "NEAR_0_50")
WATERFALL_NAMES = (
    "shap_local_01_low.png", "shap_local_02_median.png", "shap_local_03_high.png",
    "shap_local_04_below_019.png", "shap_local_05_above_019.png", "shap_local_06_near_050.png",
)
TEST_NOT_USED = (
    "Local SHAP, case selection and contributor extraction use VALIDATION features "
    "and frozen model scores only. Neither validation targets nor the returned "
    "TEST partition are accessed by this workflow. The reused splitter performs "
    "its existing partition integrity checks. No final test artifact is regenerated."
)
SELECTION_POLICY = {
    "order": list(SELECTION_RULES),
    "quantiles": {"LOW_SCORE": .05, "MEDIAN_SCORE": .50, "HIGH_SCORE": .95},
    "quantile_method": "linear, calculated on full validation before exclusions",
    "below": "highest available score strictly below 0.19",
    "above": "lowest available score greater than or equal to 0.19",
    "near_audit": "smallest absolute distance from 0.50",
    "tie_break": "smaller zero-based validation row position",
    "deduplication": "process six rules in order; skip previously selected positions within each rule's eligible ordered candidates",
    "unavailable_rule": "raise an error if no unselected eligible observation remains; do not relax the rule",
    "identifiers": "friendly case aliases, not actual customer IDs; validation_position is only a zero-based dataset row reference",
    "target_used": False,
}
LIMITATIONS = [
    "SHAP explains this model's prediction. It does not establish causality.",
    "Contributions are raw-margin / log-odds terms, not per-feature probability percentages.",
    "Sigmoid is applied only to the TOTAL raw margin for reconciliation, never to individual SHAP values.",
    "Selected cases illustrate predeclared score regions; six examples do not establish population representativeness, fairness or performance.",
    "Top contributors are not legal adverse action reason codes, regulatory reason codes, or reasons why an observation experienced delinquency.",
    "Scores are uncalibrated XGBoost predicted probabilities, not regulatory or production-validated PD.",
    "Attributions depend on transformed inputs, tree-path reference counts and feature dependence; they do not establish counterfactual recourse.",
    "The source meaning of delinquency values 96/98 remains unknown; the special indicator is not proof of fraud, error or a confirmed delinquency count.",
    "No model, preprocessing, calibration, hyperparameter, feature or threshold decision is changed using these explanations.",
]


@dataclass(frozen=True)
class LocalShapResult:
    """Local aggregates plus transient arrays for the seven approved figures."""

    report: dict
    cases: pd.DataFrame
    contributions: pd.DataFrame
    explanation: shap.Explanation
    validation_probabilities: np.ndarray


def validate_scores(scores: np.ndarray) -> np.ndarray:
    """Require finite one-dimensional predicted probabilities, preserving order."""
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Scores must be a nonempty finite one-dimensional array.")
    if ((values < 0) | (values > 1)).any():
        raise ValueError("Predicted probabilities must be within [0, 1].")
    return values


def select_cases(scores: np.ndarray) -> pd.DataFrame:
    """Select six unique positions using only probabilities and positional ties.

    Quantile distances use the full score distribution. Below/above eligibility
    remains strict during deduplication; unavailable rules fail explicitly.
    """
    scores = validate_scores(scores)
    if len(scores) < 6:
        raise ValueError("At least six validation observations are required.")
    positions = np.arange(len(scores))
    quantiles = np.quantile(scores, [.05, .5, .95], method="linear")
    criteria = [*quantiles, OPERATING_THRESHOLD, OPERATING_THRESHOLD, AUDIT_THRESHOLD]
    selected: set[int] = set()
    rows = []
    for i, (case_id, rule, criterion) in enumerate(zip(CASE_IDS, SELECTION_RULES, criteria, strict=True)):
        eligible = positions
        if i == 3:
            eligible = positions[scores < OPERATING_THRESHOLD]
        elif i == 4:
            eligible = positions[scores >= OPERATING_THRESHOLD]
        distances = np.abs(scores[eligible] - criterion)
        ordered = eligible[np.lexsort((eligible, distances))]
        available = [(rank, int(position)) for rank, position in enumerate(ordered, 1) if int(position) not in selected]
        if not available:
            raise ValueError(f"No distinct eligible observation remains for {rule}.")
        candidate_rank, position = available[0]
        selected.add(position)
        rows.append({"case_id": case_id, "selection_rule": rule, "validation_position": position,
                     "criterion_score": float(criterion), "criterion_distance": float(abs(scores[position] - criterion)),
                     "eligible_candidate_rank": candidate_rank, "deduplication_occurred": candidate_rank > 1,
                     "skipped_selected_candidates": candidate_rank - 1, "predicted_probability": float(scores[position])})
    return pd.DataFrame(rows)


def threshold_status(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply the two frozen >= decision rules without threshold evaluation/search."""
    values = validate_scores(scores)
    return (values >= OPERATING_THRESHOLD).astype(int), (values >= AUDIT_THRESHOLD).astype(int)


def reconcile_probability(raw_margins: np.ndarray, probabilities: np.ndarray) -> dict:
    """Apply stable sigmoid only to TOTAL raw margins, checking model probabilities."""
    margins = np.asarray(raw_margins, dtype=float)
    probabilities = validate_scores(probabilities)
    if margins.shape != probabilities.shape or not np.isfinite(margins).all():
        raise ValueError("Margins must be finite and aligned with probabilities.")
    sigmoid = np.exp(-np.logaddexp(0, -margins))
    errors = np.abs(sigmoid - probabilities)
    if not np.allclose(sigmoid, probabilities, rtol=1e-6, atol=1e-7):
        raise ValueError("Total raw-margin sigmoid does not reproduce predicted probabilities.")
    return {"status": "PASS", "sigmoid_of_total_margin": sigmoid.tolist(),
            "absolute_errors": errors.tolist(), "max_absolute_error": float(errors.max()),
            "atol": 1e-7, "rtol": 1e-6}


def contribution_table(case_ids: list[str], raw: pd.DataFrame, transformed: pd.DataFrame,
                       values: np.ndarray) -> pd.DataFrame:
    """Retain every input; rank absolute SHAP with stable schema-order tie breaks.

    Blank raw_value means either a missing raw observation or an engineered
    indicator without a raw source column. The feature name distinguishes these.
    """
    values = np.asarray(values, dtype=float)
    if (raw.columns.tolist() != FEATURE_COLUMNS or transformed.columns.tolist() != OUTPUT_FEATURE_COLUMNS
            or values.shape != transformed.shape or len(raw) != len(transformed)
            or len(case_ids) != len(raw) or len(set(case_ids)) != len(case_ids)
            or not raw.index.equals(transformed.index) or not np.isfinite(values).all()
            or not np.isfinite(transformed.to_numpy(dtype=float)).all()):
        raise ValueError("Local contribution inputs must align to the raw and 13-feature schemas.")
    rows = []
    for i, case_id in enumerate(case_ids):
        order = np.argsort(-np.abs(values[i]), kind="stable")
        for rank, feature_index in enumerate(order, 1):
            feature = OUTPUT_FEATURE_COLUMNS[feature_index]
            value = float(values[i, feature_index])
            raw_value = raw.iloc[i][feature] if feature in raw.columns else None
            rows.append({"case_id": case_id, "feature": feature,
                         "raw_value": None if raw_value is None or pd.isna(raw_value) else float(raw_value),
                         "transformed_value": float(transformed.iloc[i, feature_index]),
                         "shap_value": value, "absolute_shap": abs(value),
                         "direction": "Neutral" if abs(value) <= NEUTRAL_ATOL else (
                             "Higher model output" if value > 0 else "Lower model output"),
                         "rank_within_case": rank})
    return pd.DataFrame(rows)


def top_contributors(contributions: pd.DataFrame) -> list[dict]:
    """Extract up to three strictly non-neutral contributors on either side."""
    summaries = []
    for case_id, group in contributions.groupby("case_id", sort=False):
        group = group.sort_values("rank_within_case", kind="stable")
        def side(direction: str) -> list[dict]:
            return group.loc[group.direction.eq(direction), ["feature", "shap_value"]].head(3).to_dict(orient="records")
        summaries.append({"case_id": case_id,
                          "top_contributors_to_higher_model_output": side("Higher model output"),
                          "top_contributors_to_lower_model_output": side("Lower model output")})
    return summaries


def run_analysis(data: ExplanationData, reference: dict) -> LocalShapResult:
    """Fit TRAIN once, freeze score-only selection, then explain exactly six rows."""
    started = perf_counter()
    specification = verify_frozen_specification(reference)
    pipeline = frozen.build_final_model_pipeline()
    parameters = pipeline.named_steps["model"].get_params()
    if any(parameters[k] != v for k, v in specification["model_parameters"].items()):
        raise ValueError("Pipeline parameters differ from frozen XGB-01.")
    pipeline = fit_tree(pipeline, data.X_train, data.y_train)
    scores = positive_probabilities(pipeline, data.X_validation)
    cases = select_cases(scores)
    positions = cases.validation_position.to_numpy(dtype=int)
    raw = data.X_validation.iloc[positions].copy()
    # Reuse STEP 16's exact TreeExplainer settings, shape/name and additivity checks.
    transformed, explanation, shap_audit = explain_validation(pipeline, raw)
    margins = pipeline.named_steps["model"].predict(transformed, output_margin=True).astype(float)
    additivity = check_additivity(explanation.values, explanation.base_values, margins)
    reconstruction = explanation.base_values.astype(float) + explanation.values.astype(float).sum(axis=1)
    reconciliation = reconcile_probability(margins, cases.predicted_probability.to_numpy())
    cases["raw_margin"] = margins
    cases["base_value"] = explanation.base_values.astype(float)
    cases["prediction_at_0_19"], cases["prediction_at_0_50"] = threshold_status(cases.predicted_probability.to_numpy())
    cases["reconstructed_raw_margin"] = reconstruction
    cases["additivity_error"] = np.abs(reconstruction - margins)
    cases["sigmoid_raw_margin"] = reconciliation["sigmoid_of_total_margin"]
    cases["probability_reconciliation_error"] = reconciliation["absolute_errors"]
    contributions = contribution_table(cases.case_id.tolist(), raw, transformed, explanation.values)
    summaries = top_contributors(contributions)
    if frozen.get_frozen_metadata() != specification:
        raise ValueError("Frozen configuration mutated during local SHAP.")
    # Pandas JSON conversion encodes missing raw values as JSON null, never NaN.
    contribution_records = json.loads(contributions.to_json(orient="records", double_precision=15))
    report = {
        "final_model": frozen.FINAL_MODEL_NAME, "frozen_specification": specification,
        "calibration": "none", "random_state": RANDOM_STATE, "fit_partition": "training only",
        "train_rows": len(data.X_train), "explanation_partition": "validation", "validation_rows": len(scores),
        "case_selection_policy": SELECTION_POLICY.copy(), "selected_cases": cases.to_dict(orient="records"),
        "deduplication_occurred": bool(cases.deduplication_occurred.any()),
        "thresholds": {"development_operating": OPERATING_THRESHOLD, "default_audit": AUDIT_THRESHOLD},
        "transformed_features": OUTPUT_FEATURE_COLUMNS.copy(), "shap_output_shape": list(explanation.values.shape),
        "shap_output_space": "raw margin / log-odds", "base_value": shap_audit["expected_value"],
        "explainer": {"class": "shap.TreeExplainer", "model_output": "raw", "feature_perturbation": "tree_path_dependent",
                      "approximate": False, "shap_version": shap.__version__, "background": "TRAIN path counts recorded in the fitted trees"},
        "local_additivity": additivity, "probability_reconciliation": reconciliation,
        "neutral_absolute_tolerance": NEUTRAL_ATOL, "contributions": contribution_records,
        "top_contributors": summaries, "test_not_used_statement": TEST_NOT_USED,
        "interpretation_limitations": LIMITATIONS.copy(),
        "raw_value_semantics": "Original predictive values before frozen preprocessing; null means missing raw value or no raw column for an engineered indicator.",
        "transformed_value_semantics": "Actual named model input after fixed quality rules and TRAIN-median imputation; includes three engineered indicators.",
        "runtime_seconds": {"local_shap_calculation": shap_audit["shap_seconds"],
                            "fit_predict_select_explain": perf_counter() - started},
    }
    return LocalShapResult(report, cases, contributions, explanation, scores)


def render_report(report: dict) -> str:
    """Render English case explanations with no true labels or lending advice."""
    cases = report["selected_cases"]
    audit = report["local_additivity"]
    rows = [f"| {c['case_id']} | {c['selection_rule']} | {c['predicted_probability']:.9f} | {c['prediction_at_0_19']} | {c['prediction_at_0_50']} | {c['additivity_error']:.3g} |" for c in cases]
    lines = [
        "# Local SHAP Explainability", "", "## Scope", "",
        "Six predeclared score-region examples from VALIDATION explain the frozen model's predictions. "
        "SHAP explains this model's prediction. It does not establish causality. Case aliases are not actual customer IDs.", "",
        "## Frozen Model", "", f"**{report['final_model']}**, calibration **none**, random_state **42**. "
        f"Fit only on {report['train_rows']:,} TRAIN rows using `build_final_model_pipeline()`. "
        "Preprocessing, hyperparameters and thresholds 0.19/0.50 remain unchanged. No alternative model or combined TRAIN+VALIDATION fit was used.", "",
        "## Case Selection Policy", "",
        f"Score all {report['validation_rows']:,} validation rows. In order, choose nearest scores to the 5th, 50th and 95th percentiles "
        "(NumPy linear quantiles on the full distribution), highest score strictly below 0.19, lowest score >=0.19, "
        "then nearest score to 0.50. Ties use the smaller validation row position. Previously selected rows are skipped "
        "in each rule's eligible ordering; no criterion is relaxed. Selection fails if six eligible distinct rows cannot be found.", "",
        f"Deduplication occurred: **{report['deduplication_occurred']}**. The CSV/JSON stores candidate ranks and skipped counts. "
        "`validation_position` is a zero-based dataset row reference, not a customer identifier. "
        "Cases are chosen before SHAP is computed and are never replaced based on their explanations.", "",
        "## Why Target Was Not Used for Case Selection", "",
        "The selector accepts only predicted probabilities. Deterministic positional ordering resolves ties and duplicates. "
        "Validation targets are neither provided to this workflow nor stored in its artifacts. " + TEST_NOT_USED, "",
        "## SHAP Output Space", "",
        "The same TreeExplainer settings as STEP 16 are used: `model_output=\"raw\"`, `feature_perturbation=\"tree_path_dependent\"`, "
        "`approximate=False`. Tree path counts from training provide the reference. "
        "For binary:logistic, the raw margin is the total score before sigmoid, on a log-odds scale. "
        "Base plus all 13 SHAP contributions reconstructs this margin. "
        "Only sigmoid(total raw margin) is reconciled with predicted probability; no sigmoid is applied to an individual contribution.", "",
        f"Expected/base raw margin: **{report['base_value']:.9f}**. Scores are uncalibrated model probabilities, not regulatory PD.", "",
        "## Selected Validation Cases", "",
        "| Case alias | Selection rule | Predicted probability | >=0.19 | >=0.50 | Absolute reconstruction error |",
        "|---|---|---:|---:|---:|---:|", *rows, "",
        "Classification flags only apply the frozen >= rules; their quality is not assessed here.", "",
        "## Local Additivity Verification", "",
        f"**{audit['status']}** for all six cases. Maximum absolute error: **{audit['max_absolute_error']:.9g}**; "
        f"mean absolute error: **{audit['mean_absolute_error']:.9g}**. "
        "Criterion: absolute error <= 1e-5 + 1e-5 * abs(raw margin), matching STEP 16. "
        f"Total-margin probability reconciliation: **{report['probability_reconciliation']['status']}**; "
        f"maximum absolute error **{report['probability_reconciliation']['max_absolute_error']:.9g}**.", "",
    ]
    for case, summary in zip(cases, report["top_contributors"], strict=True):
        lines.extend([f"### {case['case_id']}", "",
            f"Selected by **{case['selection_rule']}**: criterion score {case['criterion_score']:.9f}, "
            f"distance {case['criterion_distance']:.9g}, eligible candidate rank {case['eligible_candidate_rank']}. "
            f"Previously selected candidates skipped: {case['skipped_selected_candidates']}.", "",
            f"Predicted probability **{case['predicted_probability']:.9f}**; prediction at 0.19: **{case['prediction_at_0_19']}**; "
            f"prediction at 0.50: **{case['prediction_at_0_50']}**. Total raw margin: **{case['raw_margin']:.9f}**; "
            f"base value: **{case['base_value']:.9f}**; absolute reconstruction error: **{case['additivity_error']:.9g}**.", ""])
        for title, key in [("Top contributors to higher model output", "top_contributors_to_higher_model_output"),
                           ("Top contributors to lower model output", "top_contributors_to_lower_model_output")]:
            lines.extend([f"**{title}** (raw-margin units)", ""])
            lines.extend([f"- `{c['feature']}`: {c['shap_value']:+.6f}" for c in summary[key]] or ["- None beyond the neutral tolerance."])
            lines.append("")
        net = case["reconstructed_raw_margin"] - case["base_value"]
        lines.extend([f"Together, all 13 contributions sum to **{net:+.6f} raw-margin units** relative to the reference. "
                      "Positive entries contribute toward higher model output and negative entries toward lower model output for this observation. "
                      "This is a decomposition of the model prediction, not an explanation of an observed outcome or a lending recommendation.", ""])
    lines.extend([
        "## Interpretation Guidelines", "",
        "- Contribution CSV contains 78 rows, ranked by absolute SHAP within each case. Ties retain the approved feature order.",
        f"- Contributions with absolute value <= {NEUTRAL_ATOL:g} are labelled Neutral and excluded from positive/negative top lists.",
        "- `raw_value` is the original value; `transformed_value` is the actual model input. Blank raw values denote missing observations or engineered indicators without a raw source column.",
        "- `age=0` is treated as missing and filled with its TRAIN median. Delinquency 96/98 values are marked missing in a working copy and filled with TRAIN medians.",
        "- `MonthlyIncome_missing` and `NumberOfDependents_missing` record original missingness. `has_special_delinquency_value` records the presence of 96/98 before replacement.",
        "- All three flags are engineered features. The source meaning of 96/98 is unknown; these are not confirmed fraud/error codes or confirmed default counts.",
        "- Waterfalls use transformed feature values and raw-margin contributions. Probability appears only as the total model prediction in the title.", "",
        "## Limitations", "", *[f"- {line}" for line in report["interpretation_limitations"]], "",
        "## What These Explanations Are Not", "",
        "These are not legal adverse action reason codes, regulatory reason codes, causal reasons, explanations of why a customer defaulted, "
        "or instructions to approve/reject a borrower. No local result is used to change the frozen model or thresholds. "
        "The consumed internal test is excluded from local explainability development.", "",
        f"Runtime (seconds): {json.dumps(report['runtime_seconds'], sort_keys=True)}.", "",
    ])
    return "\n".join(lines)


def save_figures(result: LocalShapResult, directory: Path) -> None:
    """Export six raw-margin waterfalls and a label-free score-position figure."""
    directory.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.size": 10}):
        for i, name in enumerate(WATERFALL_NAMES):
            case = result.cases.iloc[i]
            ax = shap.plots.waterfall(result.explanation[i], max_display=13, show=False)
            fig = ax.figure
            fig.set_size_inches(15, 9)
            fig.subplots_adjust(left=.43, right=.97, top=.85, bottom=.12)
            fig.suptitle(f"{case.case_id} | Predicted probability = {case.predicted_probability:.6f}\n"
                         "Frozen XGB-01: contributions in raw-margin / log-odds units", y=.97, fontsize=13)
            fig.savefig(directory / name, dpi=150, bbox_inches="tight")
            plt.close(fig)
        fig, (distribution, selected) = plt.subplots(2, 1, figsize=(12, 9), sharex=True,
                                                     gridspec_kw={"height_ratios": [1, 1.4]})
        distribution.hist(result.validation_probabilities, bins=np.linspace(0, 1, 51), color="#547f93", edgecolor="white")
        distribution.set_yscale("log")
        distribution.set(title="Full Validation Predicted-Score Distribution", ylabel="Observation count (log scale)")
        for ax in (distribution, selected):
            ax.axvline(OPERATING_THRESHOLD, color="#cc7a00", linestyle="--", label="Frozen threshold 0.19")
            ax.axvline(AUDIT_THRESHOLD, color="#7452a2", linestyle=":", label="Frozen threshold 0.50")
            ax.set_xlim(0, 1)
        distribution.legend()
        for i, case in result.cases.iterrows():
            distribution.axvline(case.predicted_probability, color="#355e6d", alpha=.2, linewidth=.8)
            selected.hlines(i, 0, 1, color="#e5e5e5", linewidth=.8)
            selected.scatter(case.predicted_probability, i, color="#24576e", s=45, zorder=3)
            selected.annotate(f"{case.predicted_probability:.6f}", (case.predicted_probability, i), xytext=(9, 7), textcoords="offset points", fontsize=9)
        selected.set(yticks=range(6), yticklabels=result.cases.case_id.tolist(), ylim=(5.7, -.7),
                     xlabel="Frozen XGBoost predicted probability (uncalibrated)", title="Six Deterministic Validation Cases — No Target-Based Selection")
        fig.tight_layout()
        fig.savefig(directory / "shap_local_case_positions.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def save_reports(result: LocalShapResult, report_dir: Path = DEFAULT_REPORT_DIR) -> None:
    """Write only STEP 17 summaries/figures, never models or processed datasets."""
    report_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    save_figures(result, report_dir / "figures")
    result.report["runtime_seconds"]["figure_export"] = perf_counter() - started
    result.cases.to_csv(report_dir / "shap_local_cases.csv", index=False)
    result.contributions.to_csv(report_dir / "shap_local_contributions.csv", index=False)
    (report_dir / "shap_local_results.json").write_text(json.dumps(result.report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (report_dir / "shap_local_explainability.md").write_text(render_report(result.report), encoding="utf-8")


def main() -> None:
    """Run the fixed local explanation workflow and report its measured runtime."""
    started = perf_counter()
    reference_path = DEFAULT_REPORT_DIR / "final_model_selection.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    verify_frozen_specification(reference)
    print("Frozen XGB-01 verified; calibration=none; thresholds=0.19/0.50. TRAIN fit, VALIDATION local SHAP only.", flush=True)
    result = run_analysis(load_explanation_data(), reference)
    result.report["runtime_seconds"]["load_fit_predict_select_explain"] = perf_counter() - started
    result.report["provenance"] = {"freeze_commit": "b6efa63", "global_shap_commit": "b87cb43",
        "source": "data/raw/cs-training.csv", "raw_dataset_sha256": hashlib.sha256(DEFAULT_DATA_PATH.read_bytes()).hexdigest(),
        "frozen_specification_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest()}
    save_reports(result)
    print(result.cases.to_string(index=False))
    print(json.dumps({"additivity": result.report["local_additivity"],
                      "probability_reconciliation": result.report["probability_reconciliation"],
                      "deduplication_occurred": result.report["deduplication_occurred"],
                      "runtime_seconds": result.report["runtime_seconds"], "total_wall_seconds": perf_counter() - started}, indent=2))
    print(TEST_NOT_USED)


if __name__ == "__main__":
    main()
