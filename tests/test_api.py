"""TestClient inference checks using toy requests, never dataset/test records."""

import ast
from copy import deepcopy
import hashlib
import inspect
import json

from fastapi.testclient import TestClient
import joblib
import numpy as np
import pandas as pd
import pytest

from api import main, model_service, schemas

EXPECTED_SHA256 = "e99eb083596beac8b6b53c8ac75d90b028dce7bf9229d759a2967709d9084590"


@pytest.fixture
def payload():
    """Invented contract example, not a person or a test/validation record."""
    return {
        "RevolvingUtilizationOfUnsecuredLines": .35, "age": 45,
        "NumberOfTime30-59DaysPastDueNotWorse": 0, "DebtRatio": .3,
        "MonthlyIncome": 5000, "NumberOfOpenCreditLinesAndLoans": 6,
        "NumberOfTimes90DaysLate": 0, "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0, "NumberOfDependents": 2,
    }


@pytest.fixture(scope="module")
def client():
    """Load the approved local binary once, without training or file writes."""
    before = hashlib.sha256(model_service.DEFAULT_ARTIFACT_PATH.read_bytes()).hexdigest()
    assert before == EXPECTED_SHA256
    with TestClient(main.create_app()) as client:
        yield client
    assert hashlib.sha256(model_service.DEFAULT_ARTIFACT_PATH.read_bytes()).hexdigest() == before


@pytest.fixture(scope="module")
def direct_pipeline():
    path = model_service.DEFAULT_ARTIFACT_PATH
    assert hashlib.sha256(path.read_bytes()).hexdigest() == EXPECTED_SHA256
    return joblib.load(path)


def test_app_metadata_docs_and_exact_schema(client):
    assert client.app.title == "Credit Risk Scoring API"
    assert schemas.DISCLAIMER in client.app.description
    assert client.get("/docs").status_code == client.get("/redoc").status_code == 200
    document = client.get("/openapi.json").json()
    schema = document["components"]["schemas"]["PredictionRequest"]
    assert len(schema["properties"]) == len(schema["required"]) == 10
    assert set(schema["properties"]) == set(schemas.RAW_FEATURE_COLUMNS)
    assert schema["additionalProperties"] is False
    assert all(field.get("description") for field in schema["properties"].values())


def test_health_and_model_info(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
    response = client.get("/model-info")
    assert response.status_code == 200
    info = response.json()
    assert info["model_name"] == "XGBoost Baseline / XGB-01" and info["calibration"] == "none"
    assert info["raw_input_feature_count"] == 10 and info["transformed_feature_count"] == 13
    assert info["raw_input_features"] == list(schemas.RAW_FEATURE_COLUMNS)
    assert info["default_audit_threshold"] == .5 and info["development_operating_threshold"] == .19
    assert info["artifact_sha256"] == EXPECTED_SHA256 and info["artifact_workflow_version"] == "STEP18-v1"
    assert info["disclaimer"] == schemas.DISCLAIMER
    assert "D:\\" not in response.text and "C:\\" not in response.text
    assert not {"project_provenance", "training_rows", "test_rows", "artifact_relative_path"} & info.keys()


@pytest.mark.parametrize("updates", [{}, {"MonthlyIncome": None}, {"MonthlyIncome": 0}, {"NumberOfDependents": None},
                                     {"age": 0}, {"NumberOfTimes90DaysLate": 96},
                                     {"NumberOfTime30-59DaysPastDueNotWorse": 98},
                                     {"RevolvingUtilizationOfUnsecuredLines": 500, "DebtRatio": 100000, "age": 120}])
def test_predict_success_direct_artifact_parity(client, direct_pipeline, payload, updates):
    payload.update(updates)
    raw = pd.DataFrame([payload], columns=schemas.RAW_FEATURE_COLUMNS, dtype=float)
    expected = float(direct_pipeline.predict_proba(raw)[0, 1])
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["predicted_probability"] == expected
    assert np.isfinite(expected) and 0 <= expected <= 1
    assert result["above_development_operating_threshold"] == (expected >= .19)
    assert result["above_default_audit_threshold"] == (expected >= .5)
    assert result["calibration"] == "none" and result["model_name"] == "XGBoost Baseline / XGB-01"
    assert not {"approved", "rejected", "loan_decision", "credit_decision"} & result.keys()


def test_null_zero_semantics_reach_pipeline_unchanged(client, payload, monkeypatch):
    pipeline = client.app.state.model_service.pipeline
    predict = pipeline.predict_proba
    seen = []
    def capture(frame):
        seen.append(frame.copy())
        return predict(frame)
    monkeypatch.setattr(pipeline, "predict_proba", capture)
    for income in [None, 0]:
        payload["MonthlyIncome"] = income
        assert client.post("/predict", json=payload).status_code == 200
    assert np.isnan(seen[0].MonthlyIncome.iloc[0]) and seen[1].MonthlyIncome.iloc[0] == 0
    transform = pipeline.named_steps["preprocessor"].transform
    assert transform(seen[0]).MonthlyIncome_missing.iloc[0] == 1
    assert transform(seen[1]).MonthlyIncome_missing.iloc[0] == 0
    assert transform(seen[1]).MonthlyIncome.iloc[0] == 0
    assert seen[0].columns.tolist() == list(schemas.RAW_FEATURE_COLUMNS)


@pytest.mark.parametrize("score,operating,audit", [(.19, True, False), (np.nextafter(.19, 0), False, False),
                                                  (.50, True, True), (np.nextafter(.50, 0), True, False)])
def test_threshold_boundaries(client, score, operating, audit):
    response = model_service.prediction_response(score, client.app.state.model_service.info)
    assert response.above_development_operating_threshold is operating
    assert response.above_default_audit_threshold is audit


@pytest.mark.parametrize("column", list(schemas.RAW_FEATURE_COLUMNS))
def test_each_required_field_cannot_be_omitted(client, payload, column, monkeypatch):
    payload.pop(column)
    monkeypatch.setattr(client.app.state.model_service.pipeline, "predict_proba", lambda *_: pytest.fail("Invalid input predicted"))
    assert client.post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize("field,value", [
    ("age", -1), ("MonthlyIncome", -1), ("NumberOfDependents", -1),
    ("NumberOfTimes90DaysLate", -1), ("NumberOfOpenCreditLinesAndLoans", -1),
    ("RevolvingUtilizationOfUnsecuredLines", -1), ("DebtRatio", -1),
    ("age", "45"), ("MonthlyIncome", "5000"), ("age", 45.5), ("age", 45.0),
    ("age", True), ("DebtRatio", True), ("DebtRatio", None), ("age", None),
    ("NumberOfDependents", 1.5), ("age", 10**400),
    ("SeriousDlqin2yrs", 1), ("Unnamed: 0", 1), ("group_id", 1),
    ("MonthlyIncome_missing", 0), ("has_special_delinquency_value", 0), ("typo", 5),
])
def test_invalid_inputs_no_prediction(client, payload, field, value, monkeypatch):
    payload[field] = value
    monkeypatch.setattr(client.app.state.model_service.pipeline, "predict_proba", lambda *_: pytest.fail("Invalid input predicted"))
    response = client.post("/predict", json=payload)
    assert response.status_code == 422, response.text
    assert response.json()["detail"]
    assert all("input" not in error and "ctx" not in error for error in response.json()["detail"])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_raw_json_returns_422(client, payload, value, monkeypatch):
    payload["MonthlyIncome"] = value
    monkeypatch.setattr(client.app.state.model_service.pipeline, "predict_proba", lambda *_: pytest.fail("Invalid input predicted"))
    response = client.post("/predict", content=json.dumps(payload), headers={"content-type": "application/json"})
    assert response.status_code == 422 and response.json()["detail"]


def test_model_loaded_once_per_lifespan(payload, monkeypatch):
    calls = []
    load = model_service.joblib.load
    def count(*args, **kwargs):
        calls.append(1)
        return load(*args, **kwargs)
    monkeypatch.setattr(model_service.joblib, "load", count)
    app = main.create_app()
    assert calls == []
    with TestClient(app) as client:
        for _ in range(3):
            assert client.get("/health").status_code == 200
            assert client.post("/predict", json=payload).status_code == 200
        assert len(calls) == 1
    assert app.state.model_service is None


def test_hash_helper():
    digest = hashlib.sha256(b"trusted bytes").hexdigest()
    assert model_service.verify_artifact_hash(b"trusted bytes", digest) == digest
    with pytest.raises(model_service.ArtifactIntegrityError, match="mismatch"):
        model_service.verify_artifact_hash(b"other bytes", digest)


def test_hash_mismatch_prevents_unpickle(tmp_path, monkeypatch):
    artifact = tmp_path / "changed.joblib"
    artifact.write_bytes(b"corrupt file")
    monkeypatch.setattr(model_service.joblib, "load", lambda *_: pytest.fail("Must not unpickle before hash check"))
    with pytest.raises(model_service.ArtifactIntegrityError, match="SHA-256 mismatch"):
        with TestClient(main.create_app(artifact_path=artifact)):
            pytest.fail("Must fail startup")


def test_missing_artifact_fails_startup(tmp_path):
    with pytest.raises(model_service.ArtifactIntegrityError, match="python -m src.serialize_model"):
        with TestClient(main.create_app(artifact_path=tmp_path / "missing.joblib")):
            pytest.fail("Must fail startup")


def test_missing_metadata_fails_startup(tmp_path):
    with pytest.raises(model_service.ArtifactIntegrityError, match="metadata missing"):
        with TestClient(main.create_app(metadata_path=tmp_path / "missing.json")):
            pytest.fail("Must fail startup")


@pytest.mark.parametrize("key,value", [("calibration", "sigmoid"), ("default_audit_threshold", .49),
                                     ("development_operating_threshold", .2), ("raw_input_features", []),
                                     ("transformed_feature_count", 10), ("model_name", "Other")])
def test_metadata_drift_blocks_load(tmp_path, monkeypatch, key, value):
    metadata = json.loads(model_service.DEFAULT_METADATA_PATH.read_text())
    metadata[key] = value
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata))
    monkeypatch.setattr(model_service.joblib, "load", lambda *_: pytest.fail("Must not load drifted metadata"))
    with pytest.raises(model_service.ArtifactIntegrityError, match="frozen model contract"):
        model_service.ModelService.load(metadata_path=path)


def test_loaded_pipeline_integrity_rejects_drift(client):
    metadata = json.loads(model_service.DEFAULT_METADATA_PATH.read_text())
    with pytest.raises(model_service.ArtifactIntegrityError, match="complete"):
        model_service.validate_loaded_pipeline(object(), metadata)
    bad = deepcopy(client.app.state.model_service.pipeline)
    bad.named_steps["model"].set_params(max_depth=4)
    with pytest.raises(model_service.ArtifactIntegrityError, match="parameters"):
        model_service.validate_loaded_pipeline(bad, metadata)


def test_without_lifespan_never_claims_healthy(payload):
    client = TestClient(main.create_app())
    assert client.get("/health").status_code == 503
    assert client.post("/predict", json=payload).status_code == 503


def test_unexpected_model_error_is_generic_and_payload_not_logged(client, payload, monkeypatch, caplog):
    def fail(_):
        raise RuntimeError('D:\\private\\secret.bin applicant payload 987654')
    monkeypatch.setattr(client.app.state.model_service.pipeline, "predict_proba", fail)
    response = client.post("/predict", json=payload)
    assert response.status_code == 500
    assert response.json() == {"detail": "Model inference failed."}
    assert "secret" not in caplog.text and "987654" not in caplog.text


@pytest.mark.parametrize("matrix", [[[np.nan, .5]], [[-.1, 1.1]], [[.5]], [[.7, .7]]])
def test_invalid_model_probability_never_returned(client, payload, monkeypatch, matrix):
    monkeypatch.setattr(client.app.state.model_service.pipeline, "predict_proba", lambda _: np.array(matrix))
    response = client.post("/predict", json=payload)
    assert response.status_code == 500 and "predicted_probability" not in response.json()


def test_no_training_dataset_or_test_dependency():
    forbidden = {"fit", "fit_transform", "fit_resample", "read_csv", "load_data", "split_data", "dump",
                 "X_test", "y_test", "run_evaluation", "run_serialization", "build_final_model_pipeline"}
    for module in [main, model_service, schemas]:
        tree = ast.parse(inspect.getsource(module))
        accessed = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        accessed |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        assert not forbidden & accessed
        imports = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        assert not {"src.final_model", "src.serialize_model", "src.final_test_evaluation", "src.shap_global", "src.shap_local"} & imports
