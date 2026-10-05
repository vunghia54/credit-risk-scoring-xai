"""Train-fitted preprocessing for the approved raw feature schema.

Build a fresh preprocessor inside each future modeling/CV pipeline. Fit only on
training (or fitting-fold) X; transform evaluation X without refitting. The API
cannot infer a partition's provenance: callers own that boundary. No target is
used here. Run ``python -m src.preprocessing`` for an in-memory smoke audit.
"""

from __future__ import annotations

import json
from typing import Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from src.data_split import FEATURE_COLUMNS, load_data, split_data

DELINQUENCY_COLUMNS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]
SPECIAL_DELINQUENCY_VALUES = (96, 98)
MISSING_INDICATOR_SOURCES = ["MonthlyIncome", "NumberOfDependents"]
SPECIAL_INDICATOR = "has_special_delinquency_value"
INDICATOR_COLUMNS = [
    "MonthlyIncome_missing", "NumberOfDependents_missing", SPECIAL_INDICATOR,
]
IMPUTATION_COLUMNS = [
    "age", "MonthlyIncome", "NumberOfDependents", *DELINQUENCY_COLUMNS,
]
OUTPUT_FEATURE_COLUMNS = [*FEATURE_COLUMNS, *INDICATOR_COLUMNS]


class PreprocessingValidationError(ValueError):
    """Input or output violates the documented preprocessing contract."""


def _validate_frame(X: pd.DataFrame, columns: Sequence[str]) -> None:
    """Require an exact named, real numeric schema; allow NaN, reject infinity."""
    if not isinstance(X, pd.DataFrame):
        raise PreprocessingValidationError("Expected a pandas DataFrame with named features.")
    if not X.columns.is_unique:
        raise PreprocessingValidationError("Duplicate feature names are not allowed.")
    missing = [column for column in columns if column not in X.columns]
    unexpected = [column for column in X.columns if column not in columns]
    if missing or unexpected:
        raise PreprocessingValidationError(
            f"Feature schema mismatch: missing={missing}; unexpected={unexpected}."
        )
    if X.empty:
        raise PreprocessingValidationError("Feature input must contain at least one row.")
    for column in columns:
        dtype = X[column].dtype
        if not is_numeric_dtype(dtype) or is_bool_dtype(dtype) or is_complex_dtype(dtype):
            raise PreprocessingValidationError(f"{column} must have a real numeric dtype.")
        if np.isinf(X[column].to_numpy(dtype=float, na_value=np.nan)).any():
            raise PreprocessingValidationError(f"{column} contains infinity.")


def _feature_names(
    estimator: BaseEstimator, input_features: Sequence[str] | None, output: Sequence[str],
) -> np.ndarray:
    """Implement sklearn's feature-name interface, preserving canonical output."""
    check_is_fitted(estimator, "feature_names_in_")
    if input_features is not None and not np.array_equal(
        np.asarray(input_features, dtype=object), estimator.feature_names_in_
    ):
        raise ValueError("input_features must match feature_names_in_.")
    return np.asarray(output, dtype=object)


class DataQualityTransformer(TransformerMixin, BaseEstimator):
    """Apply fixed rules on a copy and append three fixed binary indicators.

    Age zero becomes missing; age above 100 is retained. The special observed
    delinquency values 96/98 become missing after recording their shared flag.
    They are not claimed to be confirmed error codes. Income zero stays zero.
    Fit records only the schema; no distribution or target is learned.
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> DataQualityTransformer:
        _validate_frame(X, FEATURE_COLUMNS)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(FEATURE_COLUMNS)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "feature_names_in_")
        _validate_frame(X, FEATURE_COLUMNS)
        result = X.loc[:, FEATURE_COLUMNS].astype(float).copy()
        for column in MISSING_INDICATOR_SOURCES:
            result[f"{column}_missing"] = X[column].isna().astype("int64")
        special = X[DELINQUENCY_COLUMNS].isin(SPECIAL_DELINQUENCY_VALUES)
        result[SPECIAL_INDICATOR] = special.any(axis=1).astype("int64")
        result["age"] = result["age"].mask(result["age"].eq(0))
        result[DELINQUENCY_COLUMNS] = result[DELINQUENCY_COLUMNS].mask(special)
        return result.loc[:, OUTPUT_FEATURE_COLUMNS]

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        return _feature_names(self, input_features, OUTPUT_FEATURE_COLUMNS)


class TrainMedianImputer(TransformerMixin, BaseEstimator):
    """Impute only the six approved columns using statistics learned in fit.

    All-missing fitting columns fail explicitly instead of being dropped or
    filled with an invented fallback. Unexpected missingness in other columns
    also fails; it requires an explicit extension of the preprocessing policy.
    """

    @staticmethod
    def _validate(X: pd.DataFrame) -> None:
        _validate_frame(X, OUTPUT_FEATURE_COLUMNS)
        for column in INDICATOR_COLUMNS:
            if not X[column].isin([0, 1]).all():
                raise PreprocessingValidationError(f"{column} must contain only 0/1.")
        untouched = [column for column in FEATURE_COLUMNS if column not in IMPUTATION_COLUMNS]
        missing = [column for column in untouched if X[column].isna().any()]
        if missing:
            raise PreprocessingValidationError(f"No imputation policy for missing values in {missing}.")

    def fit(self, X: pd.DataFrame, y: object = None) -> TrainMedianImputer:
        self._validate(X)
        all_missing = [column for column in IMPUTATION_COLUMNS if X[column].isna().all()]
        if all_missing:
            raise PreprocessingValidationError(
                f"All-missing training columns after quality rules: {all_missing}; "
                "an explicit fallback decision is required."
            )
        statistics = X[IMPUTATION_COLUMNS].median()
        if not np.isfinite(statistics.to_numpy()).all():
            raise PreprocessingValidationError("Training medians must be finite.")
        self.statistics_ = statistics.copy()
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(OUTPUT_FEATURE_COLUMNS)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "statistics_")
        self._validate(X)
        result = X.loc[:, OUTPUT_FEATURE_COLUMNS].copy()
        result[IMPUTATION_COLUMNS] = result[IMPUTATION_COLUMNS].fillna(self.statistics_)
        validate_transformed(X, result)
        return result

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        return _feature_names(self, input_features, OUTPUT_FEATURE_COLUMNS)


def build_tree_preprocessor() -> Pipeline:
    """Return a fresh, unfitted tree preprocessor, with no scaling or model."""
    return Pipeline([
        ("quality", DataQualityTransformer()),
        ("imputer", TrainMedianImputer()),
    ])


def build_linear_preprocessor() -> Pipeline:
    """Return a fresh preprocessor scaling ten numerics, passing indicators through."""
    scaling = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), FEATURE_COLUMNS),
            ("indicators", "passthrough", INDICATOR_COLUMNS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    return Pipeline([
        ("quality", DataQualityTransformer()),
        ("imputer", TrainMedianImputer()),
        ("scaling", scaling),
    ])


def validate_transformed(source: pd.DataFrame, transformed: pd.DataFrame) -> None:
    """Check output shape, order, row alignment, finite values and binary flags."""
    _validate_frame(transformed, OUTPUT_FEATURE_COLUMNS)
    if transformed.columns.tolist() != OUTPUT_FEATURE_COLUMNS:
        raise PreprocessingValidationError("Output feature order differs from the fixed schema.")
    if len(transformed) != len(source) or not transformed.index.equals(source.index):
        raise PreprocessingValidationError("Transformation changed row count or index alignment.")
    if not np.isfinite(transformed.to_numpy(dtype=float)).all():
        raise PreprocessingValidationError("Transformed features contain NaN or infinity.")
    if not transformed[INDICATOR_COLUMNS].isin([0, 1]).all().all():
        raise PreprocessingValidationError("Output indicators must contain only 0/1.")


def main() -> None:
    """Fit on train X, transform all X, audit train statistics; never export.

    The existing splitter performs its approved label-integrity checks. This
    audit accesses only the returned X partitions, never validation/test y.
    """
    splits = split_data(load_data())
    inputs = {name: part.X for name, part in splits.items()}
    audit: dict[str, object] = {}
    for name, factory in (("tree", build_tree_preprocessor), ("linear", build_linear_preprocessor)):
        preprocessor = factory().fit(inputs["train"])
        medians_before = preprocessor.named_steps["imputer"].statistics_.copy()
        scaler = (
            preprocessor.named_steps["scaling"].named_transformers_["numeric"]
            if name == "linear" else None
        )
        scaler_before = (scaler.mean_.copy(), scaler.var_.copy()) if scaler is not None else None
        checks = {}
        for partition_name, X in inputs.items():
            transformed = preprocessor.transform(X)
            validate_transformed(X, transformed)
            unscaled = transformed[FEATURE_COLUMNS].to_numpy()
            if scaler is not None:
                unscaled = scaler.inverse_transform(transformed[FEATURE_COLUMNS])
            numeric = pd.DataFrame(unscaled, index=X.index, columns=FEATURE_COLUMNS)
            expected_age = X["age"].mask(X["age"].eq(0)).fillna(medians_before["age"])
            np.testing.assert_allclose(numeric["age"], expected_age, atol=1e-10)
            for column in DELINQUENCY_COLUMNS:
                expected = X[column].mask(X[column].isin(SPECIAL_DELINQUENCY_VALUES))
                np.testing.assert_allclose(numeric[column], expected.fillna(medians_before[column]), atol=1e-10)
            for column in MISSING_INDICATOR_SOURCES:
                np.testing.assert_array_equal(transformed[f"{column}_missing"], X[column].isna().astype(int))
            np.testing.assert_array_equal(
                transformed[SPECIAL_INDICATOR],
                X[DELINQUENCY_COLUMNS].isin(SPECIAL_DELINQUENCY_VALUES).any(axis=1).astype(int),
            )
            np.testing.assert_allclose(numeric.loc[X["MonthlyIncome"].eq(0), "MonthlyIncome"], 0, atol=1e-10)
            checks[partition_name] = {"shape": list(transformed.shape), "validation": "PASS"}
        pd.testing.assert_series_equal(medians_before, preprocessor.named_steps["imputer"].statistics_)
        if scaler is not None:
            np.testing.assert_array_equal(scaler.mean_, scaler_before[0])
            np.testing.assert_array_equal(scaler.var_, scaler_before[1])
        audit[name] = {
            "train_fitted_medians": medians_before.to_dict(), "partitions": checks,
            "fitted_statistics_unchanged_after_transform": True,
        }
    print(json.dumps({"status": "PASS", "feature_names": OUTPUT_FEATURE_COLUMNS, "variants": audit}, indent=2))


if __name__ == "__main__":
    main()
