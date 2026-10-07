"""Persist the TRAIN-only frozen pipeline and verify a VALIDATION round trip.

Importing this module never fits, reads data or writes artifacts. Joblib files
are trusted-local Python objects, not a secure or version-independent format.
"""

from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import tempfile
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBClassifier

from src import final_model as frozen
from src.baseline_model import TrainingValidationData, load_training_validation
from src.data_split import FEATURE_COLUMNS
from src.final_test_evaluation import (
    check_validation_reproduction, evaluate_scores, fixed_threshold_metrics,
    verify_frozen_specification,
)
from src.imbalance_experiments import validate_xy
from src.preprocessing import (
    DataQualityTransformer, TrainMedianImputer, OUTPUT_FEATURE_COLUMNS,
    validate_transformed,
)
from src.tree_model_experiments import fit_tree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_PATH = PROJECT_ROOT / "models/final_model.joblib"
THRESHOLDS = (frozen.DEFAULT_AUDIT_THRESHOLD, frozen.DEVELOPMENT_OPERATING_THRESHOLD)
WORKFLOW_VERSION = "STEP18-v1"
SECURITY_NOTE = (
    "joblib/pickle-based artifacts must only be loaded from trusted sources. "
    "Loading an untrusted pickle/joblib file may execute arbitrary code. "
    "SHA-256 detects changes only when compared with a trusted reference; it does not establish trust."
)
COMPATIBILITY_NOTE = (
    "Serialized sklearn/XGBoost artifacts may depend on library versions. Preserve "
    "requirements.txt, artifact SHA-256, metadata/software versions and project source "
    "(including src.preprocessing custom classes). The project must be importable when "
    "loading. Binary portability across arbitrary future versions is not guaranteed."
)
TEST_POLICY = (
    "consumed previously; not used in serialization. Only TRAIN and VALIDATION are "
    "exposed by the reused loader; its splitter retains existing partition integrity "
    "checks. No returned test features/targets are accessed, no test predictions are "
    "computed, and no final test or SHAP reports are regenerated."
)


def verify_pipeline_integrity(pipeline: Pipeline, X: pd.DataFrame) -> dict:
    """Validate exact fitted structure, complete model parameters and named inputs."""
    if type(pipeline) is not Pipeline or list(pipeline.named_steps) != ["preprocessor", "model"]:
        raise ValueError("Expected exactly a complete preprocessor + model sklearn Pipeline.")
    preprocessor, model = pipeline.named_steps["preprocessor"], pipeline.named_steps["model"]
    if (type(preprocessor) is not Pipeline or list(preprocessor.named_steps) != ["quality", "imputer"]
            or type(preprocessor.named_steps["quality"]) is not DataQualityTransformer
            or type(preprocessor.named_steps["imputer"]) is not TrainMedianImputer
            or type(model) is not XGBClassifier):
        raise ValueError("Expected frozen tree preprocessor and XGBClassifier; no scaler/calibration.")
    expected_model = frozen.build_final_model_pipeline().named_steps["model"]
    actual_parameters = model.get_params(deep=False)
    expected_parameters = expected_model.get_params(deep=False)
    # The default missing=np.nan survives serialization but NaN != NaN.
    def same_parameter(actual: object, expected: object) -> bool:
        if isinstance(actual, float) and isinstance(expected, float) and np.isnan(actual) and np.isnan(expected):
            return True
        return actual == expected
    if (actual_parameters.keys() != expected_parameters.keys()
            or any(not same_parameter(actual_parameters[k], v) for k, v in expected_parameters.items())):
        raise ValueError("Model parameters differ from exact frozen XGB-01.")
    check_is_fitted(model)
    if (pipeline.n_features_in_ != 10 or set(pipeline.feature_names_in_) != set(FEATURE_COLUMNS)
            or model.n_features_in_ != 13 or model.feature_names_in_.tolist() != OUTPUT_FEATURE_COLUMNS
            or model.classes_.tolist() != [0, 1]
            or model.get_booster().num_boosted_rounds() != frozen.MODEL_PARAMETERS["n_estimators"]):
        raise ValueError("Fitted input schema, class order or tree count differs from frozen pipeline.")
    # XGBoost has a legitimate max_cat_threshold constructor parameter. Only
    # additional threshold state is forbidden; constructor values were checked above.
    if any("threshold" in key.lower() and key not in vars(expected_model) for key in vars(model)):
        raise ValueError("Frozen decision thresholds must remain external to the fitted estimator.")
    transformed = preprocessor.transform(X)
    validate_transformed(X, transformed)
    if preprocessor.get_feature_names_out().tolist() != OUTPUT_FEATURE_COLUMNS:
        raise ValueError("Transformed schema differs from the approved 13 features.")
    return {"status": "PASS", "steps": ["preprocessor", "model"],
            "raw_input_count": 10, "transformed_feature_count": 13,
            "calibration_layer": False, "scaler": False,
            "thresholds_external_to_estimator": True}


def fit_frozen_pipeline(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Fit one fresh approved pipeline on caller-supplied TRAIN only."""
    validate_xy(X_train, y_train)
    pipeline = fit_tree(frozen.build_final_model_pipeline(), X_train, y_train)
    verify_pipeline_integrity(pipeline, X_train.iloc[:5])
    return pipeline


def validation_probabilities(pipeline: Pipeline, X_validation: pd.DataFrame) -> np.ndarray:
    """Return a checked full two-class matrix, preserving validation row order."""
    if pipeline.named_steps["model"].classes_.tolist() != [0, 1]:
        raise ValueError("Expected class order [0, 1].")
    values = np.asarray(pipeline.predict_proba(X_validation))
    if (values.shape != (len(X_validation), 2) or len(values) == 0
            or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any()
            or not np.allclose(values.sum(axis=1), 1, rtol=1e-6, atol=1e-7)):
        raise ValueError("Expected finite normalized (rows, 2) probabilities within [0, 1].")
    return values


def verify_validation_reproduction(y_validation: pd.Series, probabilities: np.ndarray,
                                   reference: dict) -> dict:
    """Reuse pure metric/reference helpers; never run the holdout workflow."""
    scores = probabilities[:, 1]
    metrics, _ = evaluate_scores(y_validation, scores)
    points = [fixed_threshold_metrics(y_validation, scores, t) for t in THRESHOLDS]
    return check_validation_reproduction(metrics, points, len(y_validation), reference)


def compute_file_sha256(path: Path) -> str:
    """Stream the actual file bytes into SHA-256 without a hard-coded digest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_model_artifact(pipeline: Pipeline, path: Path) -> Path:
    """Atomically save a complete fitted pipeline; caller owns the reproduction gate."""
    if type(pipeline) is not Pipeline or list(pipeline.named_steps) != ["preprocessor", "model"]:
        raise ValueError("Only a complete fitted pipeline may be saved.")
    check_is_fitted(pipeline.named_steps["model"])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        joblib.dump(pipeline, temporary, compress=3, protocol=5)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load_model_artifact(path: Path, expected_sha256: str | None = None) -> Pipeline:
    """Load ONLY trusted-local joblib files; deserialization can execute code.

    A supplied trusted digest is checked before loading. Type validation after
    load cannot make an untrusted pickle safe.
    """
    if expected_sha256 is not None and compute_file_sha256(path) != expected_sha256:
        raise ValueError("Artifact SHA-256 mismatch; refused before joblib.load.")
    pipeline = joblib.load(path)
    if type(pipeline) is not Pipeline:
        raise ValueError("Artifact is not a complete sklearn Pipeline.")
    return pipeline


def verify_loaded_artifact(original: Pipeline, loaded: Pipeline,
                           X_validation: pd.DataFrame, y_validation: pd.Series) -> dict:
    """Compare complete probability matrices and both fixed binary predictions."""
    validate_xy(X_validation, y_validation)
    verify_pipeline_integrity(original, X_validation)
    integrity = verify_pipeline_integrity(loaded, X_validation)
    before = validation_probabilities(original, X_validation)
    after = validation_probabilities(loaded, X_validation)
    differences = np.abs(before.astype(float) - after.astype(float))
    if not np.allclose(before, after, atol=1e-12, rtol=1e-12):
        raise ValueError("Loaded probabilities differ from in-memory pipeline.")
    checks = []
    for threshold in THRESHOLDS:
        original_labels, loaded_labels = before[:, 1] >= threshold, after[:, 1] >= threshold
        original_metrics = fixed_threshold_metrics(y_validation, before[:, 1], threshold)
        loaded_metrics = fixed_threshold_metrics(y_validation, after[:, 1], threshold)
        if (not np.array_equal(original_labels, loaded_labels)
                or original_metrics["confusion_matrix"] != loaded_metrics["confusion_matrix"]
                or int(original_labels.sum()) != int(loaded_labels.sum())):
            raise ValueError(f"Loaded threshold predictions differ at {threshold}.")
        checks.append({"threshold": threshold, "status": "PASS", "identical_binary_predictions": True,
                       "original_confusion_matrix": original_metrics["confusion_matrix"],
                       "loaded_confusion_matrix": loaded_metrics["confusion_matrix"],
                       "original_predicted_positive": int(original_labels.sum()),
                       "loaded_predicted_positive": int(loaded_labels.sum())})
    # Confirm that learned preprocessing statistics also survived.
    pd.testing.assert_series_equal(original.named_steps["preprocessor"].named_steps["imputer"].statistics_,
                                   loaded.named_steps["preprocessor"].named_steps["imputer"].statistics_)
    return {"status": "PASS", "partition": "validation", "rows": len(X_validation),
            "original_probability_shape": list(before.shape), "loaded_probability_shape": list(after.shape),
            "probabilities_finite_and_in_range": True, "exact_equality": bool(np.array_equal(before, after)),
            "max_absolute_probability_difference": float(differences.max()),
            "mean_absolute_probability_difference": float(differences.mean()),
            "atol": 1e-12, "rtol": 1e-12, "threshold_checks": checks,
            "pipeline_integrity": integrity, "train_medians_identical": True}


def collect_artifact_metadata(path: Path, training_rows: int, reproduction: dict,
                              verification: dict, provenance: dict | None = None) -> dict:
    """Record measured file/software properties and validated aggregate evidence."""
    return {
        "artifact_filename": Path(path).name, "artifact_relative_path": "models/final_model.joblib",
        "sha256": compute_file_sha256(path), "size_bytes": Path(path).stat().st_size,
        "workflow_version": WORKFLOW_VERSION, "project_provenance": provenance or {},
        "model_name": frozen.FINAL_MODEL_NAME, "calibration": frozen.CALIBRATION_METHOD,
        "default_audit_threshold": frozen.DEFAULT_AUDIT_THRESHOLD,
        "development_operating_threshold": frozen.DEVELOPMENT_OPERATING_THRESHOLD,
        "fit_partition": "TRAIN only", "training_rows": training_rows,
        "raw_input_features": list(FEATURE_COLUMNS), "transformed_feature_count": 13,
        "transformed_features": list(OUTPUT_FEATURE_COLUMNS), "model_parameters": dict(frozen.MODEL_PARAMETERS),
        "validation_reproduction": reproduction, "serialization_verification": verification,
        "software_versions": {"Python": platform.python_version(), **{
            package: version(package) for package in ("numpy", "pandas", "scikit-learn", "xgboost", "joblib")}},
        "serialization": {"library": "joblib", "compress": 3, "pickle_protocol": 5,
                          "object": "complete fitted sklearn Pipeline", "binary_committed": False},
        "security_note": SECURITY_NOTE, "version_compatibility_note": COMPATIBILITY_NOTE,
        "test_set_status": TEST_POLICY,
    }


def run_serialization(data: TrainingValidationData, reference: dict, path: Path,
                      provenance: dict | None = None) -> dict:
    """Fit once on TRAIN, gate on VALIDATION, save/load and verify without refit."""
    specification = verify_frozen_specification(reference)
    pipeline = fit_frozen_pipeline(data.X_train, data.y_train)
    validate_xy(data.X_validation, data.y_validation)
    probabilities = validation_probabilities(pipeline, data.X_validation)
    reproduction = verify_validation_reproduction(data.y_validation, probabilities, reference)
    print("Validation reproduction PASS; serializing complete frozen pipeline.", flush=True)
    save_model_artifact(pipeline, path)
    digest = compute_file_sha256(path)
    loaded = load_model_artifact(path, expected_sha256=digest)
    verification = verify_loaded_artifact(pipeline, loaded, data.X_validation, data.y_validation)
    if compute_file_sha256(path) != digest or frozen.get_frozen_metadata() != specification:
        raise ValueError("Artifact or frozen specification changed during verification.")
    return collect_artifact_metadata(path, len(data.X_train), reproduction, verification, provenance)


def project_provenance() -> dict:
    """Audit HEAD and dirty status honestly; hash the uncommitted workflow source."""
    def git(*arguments: str) -> str:
        return subprocess.run(["git", *arguments], cwd=PROJECT_ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    result = {"workflow_source_sha256": compute_file_sha256(Path(__file__)),
              "requirements_sha256": compute_file_sha256(PROJECT_ROOT / "requirements.txt")}
    try:
        result.update({"head_commit": git("rev-parse", "HEAD"),
                       "working_tree_clean": not bool(git("status", "--porcelain")),
                       "note": "HEAD identifies the base revision; new workflow files may be uncommitted. Source hash identifies this implementation."})
    except (OSError, subprocess.CalledProcessError):
        result.update({"head_commit": None, "working_tree_clean": None, "note": "Git provenance unavailable."})
    return result


def require_git_ignored(path: Path) -> str:
    """Fail closed if the binary path is tracked or not excluded by Git rules."""
    relative = Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()
    result = subprocess.run(["git", "check-ignore", "-v", "--", relative], cwd=PROJECT_ROOT,
                            capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        raise ValueError("Model binary is not Git-ignored; stop before proceeding.")
    return result.stdout.strip()


def render_report(metadata: dict) -> str:
    """Render an English integrity report without test rows or predictions."""
    verification = metadata["serialization_verification"]
    reproduction = metadata["validation_reproduction"]
    rows = [f"| {p['threshold']:.2f} | {p['precision']:.9f} | {p['recall']:.9f} | {p['f1']:.9f} |"
            for p in reproduction["threshold_metrics"]]
    return "\n".join([
        "# Frozen Model Serialization", "", "## Artifact Scope", "",
        "Complete fitted tree preprocessor + XGBClassifier, generated locally. The binary artifact is intentionally excluded from Git.", "",
        "## Frozen Model Specification", "",
        f"{metadata['model_name']}; calibration **{metadata['calibration']}**. Frozen thresholds: **0.50 / 0.19**, applied externally with score >= threshold.",
        "```json", json.dumps(metadata["model_parameters"], indent=2), "```", "",
        "## Training Policy", "",
        f"Fit once on **{metadata['training_rows']:,} TRAIN rows only**, using `build_final_model_pipeline()`. No TRAIN+VALIDATION refit. Validation is used only for reproduction and load-back checks.", "",
        "## Input Contract", "", "A pandas DataFrame with exactly these ten named real numeric raw features:", "",
        *[f"- `{name}`" for name in metadata["raw_input_features"]], "",
        "The complete pipeline creates the three indicators internally and produces 13 transformed features. Caller-supplied target, index, group IDs or indicators are rejected. Frozen missing-value rules and TRAIN medians remain inside the artifact.", "",
        "## Serialization Method", "", "`joblib.dump` with compression 3 and pickle protocol 5; atomic file replacement. Load-back uses `joblib.load` only on the trusted local artifact after SHA-256 verification. Nothing is serialized on module import.", "",
        "## Validation Reproduction", "",
        f"**{reproduction['status']}** on {reproduction['rows']:,} VALIDATION rows before serialization. "
        f"ROC-AUC **{reproduction['metrics']['roc_auc']:.10f}**; Average Precision **{reproduction['metrics']['average_precision']:.10f}**. "
        "The committed STEP 14 reference is checked with rtol=1e-6, atol=1e-8; confusion counts must match exactly.", "",
        "| Threshold | Precision | Recall | F1 |", "|---|---:|---:|---:|", *rows, "",
        "## Load-Back Verification", "",
        f"**{verification['status']}** across full VALIDATION. Original/loaded probability shape: "
        f"{verification['original_probability_shape']} / {verification['loaded_probability_shape']}. "
        f"Exact equality: **{verification['exact_equality']}**; maximum absolute difference: "
        f"**{verification['max_absolute_probability_difference']:.12g}**; mean absolute difference: "
        f"**{verification['mean_absolute_probability_difference']:.12g}**. Tolerance: atol=rtol=1e-12. Probabilities are finite and within [0,1].", "",
        "```json", json.dumps(verification["threshold_checks"], indent=2), "```", "",
        "Loaded pipeline structure, all estimator parameters, feature names, class order, tree count and TRAIN medians were verified. No scaler or calibration layer is present. Thresholds are not learned estimator values.", "",
        "## Artifact Integrity", "",
        f"- Relative path: `{metadata['artifact_relative_path']}`",
        f"- Size: {metadata['size_bytes']} bytes",
        f"- SHA-256: `{metadata['sha256']}`",
        f"- Workflow version: `{metadata['workflow_version']}`", "",
        "```json", json.dumps(metadata["project_provenance"], indent=2), "```", "",
        "## Software Environment", "", "| Component | Installed version |", "|---|---|",
        *[f"| {key} | {value} |" for key, value in metadata["software_versions"].items()], "",
        "## Security Considerations", "", SECURITY_NOTE, "",
        "## Version Compatibility", "", COMPATIBILITY_NOTE, "",
        "## Test-Set Policy", "", TEST_POLICY, "",
        "## Limitations", "",
        "Round-trip equivalence confirms persistence correctness in this environment, not external performance, regulatory PD, calibration adequacy or production readiness. "
        "The model is a fresh deterministic reconstruction of the frozen TRAIN-only specification, not the historical in-memory STEP 15 object. "
        "No new test evaluation was performed. Identical predictions do not imply byte-identical artifacts across library versions. No API or deployment retraining policy is introduced.", "",
    ])


def main() -> None:
    """Execute one real serialization workflow; write only its binary and two reports."""
    started = perf_counter()
    require_git_ignored(DEFAULT_ARTIFACT_PATH)
    reference = json.loads((PROJECT_ROOT / "reports/final_model_selection.json").read_text(encoding="utf-8"))
    verify_frozen_specification(reference)
    metadata = run_serialization(load_training_validation(), reference, DEFAULT_ARTIFACT_PATH, project_provenance())
    metadata["git_ignore_rule"] = require_git_ignored(DEFAULT_ARTIFACT_PATH)
    metadata["runtime_seconds"] = perf_counter() - started
    report_dir = PROJECT_ROOT / "reports"
    (report_dir / "final_model_artifact_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (report_dir / "model_serialization.md").write_text(render_report(metadata), encoding="utf-8")
    print(json.dumps(metadata, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
