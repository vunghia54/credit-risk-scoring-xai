"""Strict raw-feature and public response contracts for inference only."""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

DISCLAIMER = "This is an internal portfolio risk-scoring model and is not a production lending decision system."
NonnegativeNumber = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
NonnegativeInteger = Annotated[int, Field(strict=True, ge=0)]


class PredictionRequest(BaseModel):
    """Exactly ten raw features; nullable fields remain required JSON keys."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    RevolvingUtilizationOfUnsecuredLines: NonnegativeNumber = Field(description="Unsecured revolving balance divided by credit limits; no upper cap is imposed.")
    age: NonnegativeInteger = Field(description="Age in years. Zero is accepted and handled by the frozen preprocessor.")
    NumberOfTime30_59DaysPastDueNotWorse: NonnegativeInteger = Field(alias="NumberOfTime30-59DaysPastDueNotWorse", description="Number of 30–59 day past-due events, not worse. Special values pass to frozen preprocessing unchanged.")
    DebtRatio: NonnegativeNumber = Field(description="Dataset debt-ratio value, passed unchanged to the frozen pipeline.")
    MonthlyIncome: NonnegativeNumber | None = Field(description="Monthly income; currency is unspecified. Null denotes missing; zero remains zero.")
    NumberOfOpenCreditLinesAndLoans: NonnegativeInteger = Field(description="Number of open credit lines and loans.")
    NumberOfTimes90DaysLate: NonnegativeInteger = Field(description="Number of 90-day-or-more past-due events; frozen preprocessing handles special values.")
    NumberRealEstateLoansOrLines: NonnegativeInteger = Field(description="Number of real-estate loans or lines, including home-equity lines.")
    NumberOfTime60_89DaysPastDueNotWorse: NonnegativeInteger = Field(alias="NumberOfTime60-89DaysPastDueNotWorse", description="Number of 60–89 day past-due events, not worse; frozen preprocessing handles special values.")
    NumberOfDependents: NonnegativeInteger | None = Field(description="Number of dependents excluding the applicant; null denotes missing.")

    @field_validator("*")
    @classmethod
    def representable_numeric(cls, value: int | float | None) -> int | float | None:
        """Reject values outside finite numeric representation, without business caps."""
        if value is not None:
            try:
                finite = math.isfinite(float(value))
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError("Value must have a finite numeric representation.")
        return value


RAW_FEATURE_COLUMNS = tuple(field.alias or name for name, field in PredictionRequest.model_fields.items())


class HealthResponse(BaseModel):
    status: Literal["ok"] = Field(description="Returned only after successful artifact startup checks.")
    model_loaded: Literal[True]


class ModelInfoResponse(BaseModel):
    model_name: str
    calibration: Literal["none"]
    raw_input_feature_count: Literal[10]
    transformed_feature_count: Literal[13]
    raw_input_features: list[str]
    default_audit_threshold: float
    development_operating_threshold: float
    artifact_sha256: str
    artifact_workflow_version: str
    disclaimer: str


class PredictionResponse(BaseModel):
    model_name: str
    predicted_probability: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] = Field(description="Uncalibrated frozen model output for class 1; not regulatory or production-validated PD.")
    calibration: Literal["none"]
    development_operating_threshold: float
    above_development_operating_threshold: bool = Field(description="True exactly when score >= development operating threshold. No lending action is implied.")
    default_audit_threshold: float
    above_default_audit_threshold: bool = Field(description="True exactly when score >= default audit threshold. No lending action is implied.")
    disclaimer: str
