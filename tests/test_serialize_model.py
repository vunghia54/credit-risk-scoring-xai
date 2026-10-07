"""Small-data persistence checks; all binary output is confined to tmp_path."""

import ast
from copy import deepcopy
from dataclasses import fields
import hashlib
from importlib.metadata import version
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import baseline_model, final_model as frozen, serialize_model as serialization
from src.data_split import FEATURE_COLUMNS
from src.preprocessing import OUTPUT_FEATURE_COLUMNS


@pytest.fixture(scope="module")
def small_data():
    rng = np.random.default_rng(42)
    raw = pd.DataFrame(rng.uniform(.1, 4, (240, 10)), columns=FEATURE_COLUMNS)
    raw["age"] = rng.integers(20, 80, len(raw)).astype(float)
    raw.loc[::11, "MonthlyIncome"] = np.nan
    raw.loc[::17, "NumberOfDependents"] = np.nan
    raw.loc[::19, "NumberOfTimes90DaysLate"] = 98
    raw.loc[::23, "age"] = 0
    y = pd.Series((raw.RevolvingUtilizationOfUnsecuredLines > 2).astype(int), index=raw.index)
    return baseline_model.TrainingValidationData(raw.iloc[:180], y.iloc[:180], raw.iloc[180:], y.iloc[180:])


@pytest.fixture(scope="module")
def fitted(small_data):
    before = small_data.X_train.copy(deep=True)
    pipeline = serialization.fit_frozen_pipeline(small_data.X_train, small_data.y_train)
    pd.testing.assert_frame_equal(small_data.X_train, before)
    return pipeline


@pytest.fixture(scope="module")
def reference(fitted, small_data):
    scores = serialization.validation_probabilities(fitted, small_data.X_validation)[:, 1]
    metrics, _ = serialization.evaluate_scores(small_data.y_validation, scores)
    points = [serialization.fixed_threshold_metrics(small_data.y_validation, scores, t)
              for t in serialization.THRESHOLDS]
    reference = frozen.get_frozen_metadata()
    reference["selection_evidence"] = {
        "ranking": {"XGBoost Baseline": {"validation": {**metrics, "rows": len(scores)}}},
        "xgboost_calibration": [{**metrics, "calibration_method": "uncalibrated"}],
        "threshold_tradeoffs": points,
    }
    return reference


def test_builder_reused_and_only_train_fit(monkeypatch, small_data, fitted):
    sentinel = object()
    calls = []
    def factory():
        calls.append("builder")
        return sentinel
    def fit(pipeline, X, y):
        assert pipeline is sentinel and X is small_data.X_train and y is small_data.y_train
        calls.append("fit_train")
        return fitted
    monkeypatch.setattr(frozen, "build_final_model_pipeline", factory)
    monkeypatch.setattr(serialization, "fit_tree", fit)
    monkeypatch.setattr(serialization, "verify_pipeline_integrity", lambda *args: None)
    assert serialization.fit_frozen_pipeline(small_data.X_train, small_data.y_train) is fitted
    assert calls == ["builder", "fit_train"]


def test_pipeline_integrity_exact_config_and_raw_contract(fitted, small_data):
    audit = serialization.verify_pipeline_integrity(fitted, small_data.X_validation)
    assert type(fitted) is Pipeline
    assert audit["raw_input_count"] == 10 and audit["transformed_feature_count"] == 13
    assert not audit["scaler"] and not audit["calibration_layer"]
    assert audit["thresholds_external_to_estimator"]
    assert fitted.named_steps["model"].get_params() == frozen.build_final_model_pipeline().named_steps["model"].get_params()
    assert fitted.named_steps["preprocessor"].transform(small_data.X_validation).columns.tolist() == OUTPUT_FEATURE_COLUMNS
    assert frozen.CALIBRATION_METHOD == "none" and serialization.THRESHOLDS == (.5, .19)


@pytest.mark.parametrize("column", ["SeriousDlqin2yrs", "Unnamed: 0", "group_id", "MonthlyIncome_missing", "has_special_delinquency_value"])
def test_extra_raw_fields_rejected(fitted, small_data, column):
    with pytest.raises(ValueError, match="schema"):
        fitted.predict_proba(small_data.X_validation.assign(**{column: 0}))


def test_missing_raw_field_rejected_and_reordered_columns_work(fitted, small_data):
    with pytest.raises(ValueError, match="schema"):
        fitted.predict_proba(small_data.X_validation.drop(columns=["age"]))
    np.testing.assert_array_equal(fitted.predict_proba(small_data.X_validation),
                                  fitted.predict_proba(small_data.X_validation[FEATURE_COLUMNS[::-1]]))


@pytest.mark.parametrize("drift", ["parameters", "missing", "scaler", "threshold", "steps"])
def test_integrity_rejects_pipeline_drift(fitted, small_data, drift):
    bad = deepcopy(fitted)
    if drift == "parameters":
        bad.named_steps["model"].set_params(max_depth=4)
    elif drift == "missing":
        bad.named_steps["model"].set_params(missing=0)
    elif drift == "scaler":
        bad.named_steps["preprocessor"].steps.append(("scaler", StandardScaler()))
    elif drift == "threshold":
        bad.named_steps["model"].threshold_ = .3
    else:
        bad.steps.append(("calibrator", StandardScaler()))
    with pytest.raises(ValueError):
        serialization.verify_pipeline_integrity(bad, small_data.X_validation)


def test_round_trip_complete_pipeline_and_thresholds(tmp_path, fitted, small_data):
    path = tmp_path / "final_model.joblib"
    serialization.save_model_artifact(fitted, path)
    loaded = serialization.load_model_artifact(path, serialization.compute_file_sha256(path))
    audit = serialization.verify_loaded_artifact(fitted, loaded, small_data.X_validation, small_data.y_validation)
    assert audit["exact_equality"] and audit["status"] == "PASS"
    assert audit["original_probability_shape"] == audit["loaded_probability_shape"] == [60, 2]
    assert audit["max_absolute_probability_difference"] == audit["mean_absolute_probability_difference"] == 0
    assert audit["atol"] == audit["rtol"] == 1e-12
    assert [x["threshold"] for x in audit["threshold_checks"]] == [.50, .19]
    for point in audit["threshold_checks"]:
        assert point["identical_binary_predictions"]
        assert point["original_confusion_matrix"] == point["loaded_confusion_matrix"]
        assert point["original_predicted_positive"] == point["loaded_predicted_positive"]
    assert audit["train_medians_identical"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["final_model.joblib"]


def test_digest_is_measured_deterministic_and_changes_with_content(tmp_path):
    path = tmp_path / "bytes.bin"
    path.write_bytes(b"known data")
    expected = hashlib.sha256(b"known data").hexdigest()
    assert serialization.compute_file_sha256(path) == serialization.compute_file_sha256(path) == expected
    path.write_bytes(b"changed")
    assert serialization.compute_file_sha256(path) != expected


def test_digest_mismatch_blocks_unpickling(tmp_path, monkeypatch):
    path = tmp_path / "untrusted.joblib"
    path.write_bytes(b"not a pickle")
    monkeypatch.setattr(joblib, "load", lambda *args: pytest.fail("Must not unpickle"))
    with pytest.raises(ValueError, match="SHA-256"):
        serialization.load_model_artifact(path, "0" * 64)


def test_atomic_save_failure_preserves_existing_file(tmp_path, fitted, monkeypatch):
    path = tmp_path / "final_model.joblib"
    path.write_bytes(b"previous artifact")
    def failing_dump(*args, **kwargs):
        Path(args[1]).write_bytes(b"partial")
        raise OSError("disk failure")
    monkeypatch.setattr(joblib, "dump", failing_dump)
    with pytest.raises(OSError, match="disk failure"):
        serialization.save_model_artifact(fitted, path)
    assert path.read_bytes() == b"previous artifact"
    assert list(tmp_path.iterdir()) == [path]


def test_unfitted_or_estimator_only_save_rejected(tmp_path, fitted):
    for pipeline in [frozen.build_final_model_pipeline(), fitted.named_steps["model"]]:
        with pytest.raises(ValueError):
            serialization.save_model_artifact(pipeline, tmp_path / "bad.joblib")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("kind", ["nan", "range", "shape", "normalization", "classes"])
def test_probability_contract_rejects_invalid_values(kind, small_data):
    matrix = np.tile([.6, .4], (len(small_data.X_validation), 1))
    classes = np.array([0, 1])
    if kind == "nan":
        matrix[0, 0] = np.nan
    elif kind == "range":
        matrix[0] = [-.1, 1.1]
    elif kind == "shape":
        matrix = matrix[:, 1]
    elif kind == "normalization":
        matrix[0] = [.6, .6]
    else:
        classes = np.array([1, 0])
    pipeline = SimpleNamespace(named_steps={"model": SimpleNamespace(classes_=classes)}, predict_proba=lambda X: matrix)
    with pytest.raises(ValueError):
        serialization.validation_probabilities(pipeline, small_data.X_validation)


def test_loaded_probability_drift_fails(fitted, small_data, monkeypatch):
    loaded = deepcopy(fitted)
    old_predict = loaded.predict_proba
    def changed(X):
        result = old_predict(X).astype(float)
        result[0] += [1e-6, -1e-6]
        return result
    monkeypatch.setattr(loaded, "predict_proba", changed)
    with pytest.raises(ValueError, match="probabilities differ"):
        serialization.verify_loaded_artifact(fitted, loaded, small_data.X_validation, small_data.y_validation)


def test_tolerance_does_not_allow_threshold_crossing(fitted, small_data, monkeypatch):
    loaded = deepcopy(fitted)
    before = np.tile([.81, .19], (len(small_data.X_validation), 1))
    after = before.copy()
    after[0] += [5e-14, -5e-14]
    monkeypatch.setattr(serialization, "validation_probabilities", lambda p, X: before if p is fitted else after)
    with pytest.raises(ValueError, match="threshold predictions"):
        serialization.verify_loaded_artifact(fitted, loaded, small_data.X_validation, small_data.y_validation)


def test_reproduction_failure_never_saves(tmp_path, monkeypatch, small_data, fitted, reference):
    reference = deepcopy(reference)
    reference["selection_evidence"]["ranking"]["XGBoost Baseline"]["validation"]["roc_auc"] = 0
    monkeypatch.setattr(serialization, "fit_frozen_pipeline", lambda *args: fitted)
    monkeypatch.setattr(serialization, "save_model_artifact", lambda *args: pytest.fail("Must not save drifted model"))
    with pytest.raises(ValueError, match="reproduction failed"):
        serialization.run_serialization(small_data, reference, tmp_path / "final_model.joblib")
    assert not list(tmp_path.iterdir())


def test_frozen_drift_blocks_fit(tmp_path, monkeypatch, small_data, reference):
    reference = deepcopy(reference)
    reference["calibration_method"] = "sigmoid"
    monkeypatch.setattr(serialization, "fit_frozen_pipeline", lambda *args: pytest.fail("Must not fit"))
    with pytest.raises(ValueError, match="specification"):
        serialization.run_serialization(small_data, reference, tmp_path / "final_model.joblib")


def test_workflow_metadata_report_and_train_validation_boundary(tmp_path, monkeypatch, small_data, fitted, reference):
    calls = []
    def train_only(X, y):
        assert X is small_data.X_train and y is small_data.y_train
        calls.append("fit_train")
        return fitted
    monkeypatch.setattr(serialization, "fit_frozen_pipeline", train_only)
    path = tmp_path / "final_model.joblib"
    report = serialization.run_serialization(small_data, reference, path, {"head_commit": "unit-test"})
    assert calls == ["fit_train"]
    assert report["training_rows"] == 180 and report["fit_partition"] == "TRAIN only"
    assert report["raw_input_features"] == FEATURE_COLUMNS and report["transformed_features"] == OUTPUT_FEATURE_COLUMNS
    assert report["size_bytes"] == path.stat().st_size and report["sha256"] == serialization.compute_file_sha256(path)
    assert report["validation_reproduction"]["status"] == report["serialization_verification"]["status"] == "PASS"
    assert report["software_versions"]["Python"]
    for name in ["numpy", "pandas", "scikit-learn", "xgboost", "joblib"]:
        assert report["software_versions"][name] == version(name)
    assert "trusted sources" in report["security_note"] and "arbitrary code" in report["security_note"]
    assert "not guaranteed" in report["version_compatibility_note"]
    assert "not used in serialization" in report["test_set_status"]
    json.dumps(report, allow_nan=False)
    text = serialization.render_report(report)
    for heading in ["Artifact Scope", "Frozen Model Specification", "Training Policy", "Input Contract",
                    "Serialization Method", "Validation Reproduction", "Load-Back Verification", "Artifact Integrity",
                    "Software Environment", "Security Considerations", "Version Compatibility", "Test-Set Policy", "Limitations"]:
        assert "## " + heading in text
    assert "intentionally excluded from Git" in text


def test_loader_does_not_access_test(monkeypatch):
    class Splits:
        train = SimpleNamespace(X="train X", y="train y")
        validation = SimpleNamespace(X="validation X", y="validation y")
        @property
        def test(self):
            pytest.fail("Returned test partition accessed")
    monkeypatch.setattr(baseline_model, "load_data", lambda: None)
    monkeypatch.setattr(baseline_model, "split_data", lambda _: Splits())
    data = serialization.load_training_validation()
    assert data.X_train == "train X" and data.y_validation == "validation y"
    assert [f.name for f in fields(data)] == ["X_train", "y_train", "X_validation", "y_validation"]


def test_module_has_no_holdout_access_selection_or_import_time_work():
    source = inspect.getsource(serialization)
    tree = ast.parse(source)
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert not {"test", "X_test", "y_test", "fit_resample", "run_evaluation", "GridSearchCV", "RandomizedSearchCV"} & attributes
    assert len([node for node in ast.walk(tree) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name) and node.func.id == "fit_tree"]) == 1
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            pytest.fail("Unexpected module-level workflow call")


def test_git_ignore_failure_blocks_workflow(monkeypatch):
    monkeypatch.setattr(serialization.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout=""))
    monkeypatch.setattr(serialization, "load_training_validation", lambda: pytest.fail("Must stop before data load"))
    with pytest.raises(ValueError, match="not Git-ignored"):
        serialization.main()
