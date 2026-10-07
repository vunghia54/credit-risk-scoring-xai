"""Load a trusted frozen artifact once and perform stateless inference.

No data loaders, fitting workflows or SHAP workflows are called here. The custom
preprocessor classes must remain importable for trusted joblib deserialization.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBClassifier

from api.schemas import DISCLAIMER, ModelInfoResponse, PredictionRequest, PredictionResponse, RAW_FEATURE_COLUMNS
from src.preprocessing import DataQualityTransformer, TrainMedianImputer, OUTPUT_FEATURE_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_PATH = PROJECT_ROOT / "models/final_model.joblib"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "reports/final_model_artifact_metadata.json"


class ArtifactIntegrityError(RuntimeError):
    """Startup cannot serve an unavailable, inconsistent or unverified model."""


def verify_artifact_hash(payload: bytes, expected: str) -> str:
    """Hash the exact bytes that will be loaded, using trusted metadata as reference."""
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ArtifactIntegrityError("Metadata must supply a valid SHA-256 digest.")
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise ArtifactIntegrityError("Artifact SHA-256 mismatch; inference startup refused.")
    return actual


def validate_metadata(metadata: dict) -> None:
    """Check the approved identity, schema and thresholds before loading the binary."""
    if (not isinstance(metadata, dict) or metadata.get("model_name") != "XGBoost Baseline / XGB-01"
            or metadata.get("calibration") != "none"
            or metadata.get("default_audit_threshold") != .50
            or metadata.get("development_operating_threshold") != .19
            or metadata.get("raw_input_features") != list(RAW_FEATURE_COLUMNS)
            or metadata.get("transformed_feature_count") != 13
            or metadata.get("transformed_features") != OUTPUT_FEATURE_COLUMNS
            or not isinstance(metadata.get("model_parameters"), dict)
            or not metadata["model_parameters"]
            or not isinstance(metadata.get("workflow_version"), str)):
        raise ArtifactIntegrityError("Metadata differs from the approved frozen model contract.")


def validate_loaded_pipeline(pipeline: Pipeline, metadata: dict) -> None:
    """Check complete fitted structure and metadata correspondence without fitting."""
    if type(pipeline) is not Pipeline or list(pipeline.named_steps) != ["preprocessor", "model"]:
        raise ArtifactIntegrityError("Expected complete preprocessor + model sklearn Pipeline.")
    preprocessor, model = pipeline.named_steps["preprocessor"], pipeline.named_steps["model"]
    if (type(model) is not XGBClassifier or type(preprocessor) is not Pipeline
            or list(preprocessor.named_steps) != ["quality", "imputer"]
            or type(preprocessor.named_steps["quality"]) is not DataQualityTransformer
            or type(preprocessor.named_steps["imputer"]) is not TrainMedianImputer):
        raise ArtifactIntegrityError("Expected frozen tree preprocessing and XGBClassifier without calibration/scaling.")
    try:
        check_is_fitted(model)
        check_is_fitted(preprocessor.named_steps["imputer"], "statistics_")
        if (pipeline.n_features_in_ != 10 or pipeline.feature_names_in_.tolist() != list(RAW_FEATURE_COLUMNS)
                or preprocessor.get_feature_names_out().tolist() != OUTPUT_FEATURE_COLUMNS
                or model.n_features_in_ != 13 or model.feature_names_in_.tolist() != OUTPUT_FEATURE_COLUMNS
                or model.classes_.tolist() != [0, 1]):
            raise ArtifactIntegrityError("Loaded feature schema or class order is inconsistent.")
        params = model.get_params()
        if any(params.get(key) != value for key, value in metadata["model_parameters"].items()):
            raise ArtifactIntegrityError("Loaded model parameters differ from frozen metadata.")
    except (AttributeError, ValueError) as exc:
        raise ArtifactIntegrityError("Loaded pipeline is not fitted with the expected contract.") from exc


def request_frame(request: PredictionRequest) -> pd.DataFrame:
    """Preserve zero/null semantics and canonical raw names; no preprocessing here."""
    return pd.DataFrame([request.model_dump(by_alias=True)], columns=RAW_FEATURE_COLUMNS, dtype=float)


def prediction_response(score: float, info: ModelInfoResponse) -> PredictionResponse:
    """Apply only the frozen inclusive thresholds; never produce lending decisions."""
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Model probability is invalid.")
    return PredictionResponse(
        model_name=info.model_name, predicted_probability=score, calibration=info.calibration,
        development_operating_threshold=info.development_operating_threshold,
        above_development_operating_threshold=score >= info.development_operating_threshold,
        default_audit_threshold=info.default_audit_threshold,
        above_default_audit_threshold=score >= info.default_audit_threshold, disclaimer=DISCLAIMER,
    )


@dataclass(frozen=True)
class ModelService:
    """Process-local fitted model with no per-request storage or learning."""

    pipeline: Pipeline
    info: ModelInfoResponse

    @classmethod
    def load(cls, artifact_path: Path = DEFAULT_ARTIFACT_PATH,
             metadata_path: Path = DEFAULT_METADATA_PATH) -> ModelService:
        """Verify and deserialize trusted bytes once; never auto-generate an artifact."""
        if not Path(artifact_path).is_file():
            raise ArtifactIntegrityError("Model artifact missing. Generate the trusted local artifact with python -m src.serialize_model before starting the API.")
        if not Path(metadata_path).is_file():
            raise ArtifactIntegrityError("Artifact metadata missing; restore the trusted serialization metadata before startup.")
        try:
            metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ArtifactIntegrityError("Artifact metadata is not valid UTF-8 JSON.") from exc
        validate_metadata(metadata)
        payload = Path(artifact_path).read_bytes()
        digest = verify_artifact_hash(payload, metadata.get("sha256"))
        if len(payload) != metadata.get("size_bytes"):
            raise ArtifactIntegrityError("Artifact size differs from metadata.")
        # Hash and load the same bytes to avoid a file replacement between checks.
        pipeline = joblib.load(io.BytesIO(payload))
        validate_loaded_pipeline(pipeline, metadata)
        info = ModelInfoResponse(
            model_name=metadata["model_name"], calibration=metadata["calibration"],
            raw_input_feature_count=10, transformed_feature_count=13,
            raw_input_features=list(RAW_FEATURE_COLUMNS), default_audit_threshold=metadata["default_audit_threshold"],
            development_operating_threshold=metadata["development_operating_threshold"], artifact_sha256=digest,
            artifact_workflow_version=metadata["workflow_version"], disclaimer=DISCLAIMER,
        )
        return cls(pipeline, info)

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Run the loaded pipeline on one raw row without fitting or retaining it."""
        values = np.asarray(self.pipeline.predict_proba(request_frame(request)))
        if (values.shape != (1, 2) or not np.isfinite(values).all()
                or ((values < 0) | (values > 1)).any()
                or not np.allclose(values.sum(axis=1), 1, rtol=1e-6, atol=1e-7)):
            raise ValueError("Model probability matrix is invalid.")
        return prediction_response(float(values[0, 1]), self.info)
