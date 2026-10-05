"""Preprocessing contracts and train-only fitting, without fitting any model.

Real partitions come from the approved splitter. Evaluation labels are never
accessed here; the splitter retains its existing integrity checks. Small rule
fixtures are in-memory copies of train rows, never exported datasets.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from src.data_split import FEATURE_COLUMNS, load_data, split_data
from src.preprocessing import (
    DELINQUENCY_COLUMNS,
    IMPUTATION_COLUMNS,
    INDICATOR_COLUMNS,
    OUTPUT_FEATURE_COLUMNS,
    SPECIAL_DELINQUENCY_VALUES,
    SPECIAL_INDICATOR,
    DataQualityTransformer,
    PreprocessingValidationError,
    TrainMedianImputer,
    build_linear_preprocessor,
    build_tree_preprocessor,
    validate_transformed,
)


@pytest.fixture(scope="module")
def inputs() -> dict[str, pd.DataFrame]:
    return {name: part.X for name, part in split_data(load_data()).items()}


@pytest.fixture(scope="module", params=["tree", "linear"])
def fitted(request: pytest.FixtureRequest, inputs: dict[str, pd.DataFrame]):
    factory = build_tree_preprocessor if request.param == "tree" else build_linear_preprocessor
    preprocessor = factory().fit(inputs["train"])
    outputs = {name: preprocessor.transform(X) for name, X in inputs.items()}
    return request.param, preprocessor, outputs


@pytest.fixture
def rule_rows(inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    X = inputs["train"].head(6).astype(float).copy()
    X["age"] = [0, 109, 21, 40, 50, 60]
    X["MonthlyIncome"] = [0, np.nan, 1000, 2000, 3000, 4000]
    X["NumberOfDependents"] = [np.nan, 0, 1, 2, 3, 4]
    X[DELINQUENCY_COLUMNS] = 0.0
    for row, column in enumerate(DELINQUENCY_COLUMNS):
        X.iloc[row, X.columns.get_loc(column)] = 96 if row != 1 else 98
    X.iloc[3, X.columns.get_loc(DELINQUENCY_COLUMNS[0])] = np.nan
    return X


@pytest.mark.parametrize("extra", ["SeriousDlqin2yrs", "Unnamed: 0", "group_id", "feature_group", "source_ids", "row_positions"])
def test_rejects_metadata_at_fit_and_transform(inputs, extra):
    X = inputs["train"].head(10)
    transformer = DataQualityTransformer().fit(X)
    invalid = X.assign(**{extra: 0})
    for method in (DataQualityTransformer().fit, transformer.transform):
        with pytest.raises(PreprocessingValidationError, match="unexpected="):
            method(invalid)


@pytest.mark.parametrize("kind", ["missing", "duplicate", "array", "nonnumeric", "infinity", "empty"])
def test_invalid_input_schema(inputs, kind):
    X = inputs["train"].head(10).copy()
    if kind == "missing":
        X = X.drop(columns="age")
    elif kind == "duplicate":
        X = pd.concat([X, X[["age"]]], axis=1)
    elif kind == "array":
        X = X.to_numpy()
    elif kind == "nonnumeric":
        X["age"] = "invalid"
    elif kind == "infinity":
        X["DebtRatio"] = -np.inf
    else:
        X = X.head(0)
    with pytest.raises(PreprocessingValidationError):
        DataQualityTransformer().fit(X)


def test_transform_requires_fit(inputs):
    with pytest.raises(NotFittedError):
        build_tree_preprocessor().transform(inputs["train"].head(5))
    with pytest.raises(NotFittedError):
        TrainMedianImputer().transform(pd.DataFrame())


def test_fixed_quality_rules_and_indicators(rule_rows):
    before = rule_rows.copy(deep=True)
    result = DataQualityTransformer().fit_transform(rule_rows)
    assert pd.isna(result["age"].iloc[0])
    assert result["age"].iloc[1] == 109
    assert result["MonthlyIncome"].iloc[0] == 0
    assert result["MonthlyIncome_missing"].tolist() == [0, 1, 0, 0, 0, 0]
    assert result["NumberOfDependents_missing"].tolist() == [1, 0, 0, 0, 0, 0]
    assert result[SPECIAL_INDICATOR].tolist() == [1, 1, 1, 0, 0, 0]
    for row, column in enumerate(DELINQUENCY_COLUMNS):
        assert pd.isna(result[column].iloc[row])
    assert not result[DELINQUENCY_COLUMNS].isin([96, 98]).any().any()
    pd.testing.assert_frame_equal(rule_rows, before)


def test_output_schema_rows_finiteness_and_metadata_exclusion(fitted, inputs):
    _, preprocessor, outputs = fitted
    assert preprocessor.get_feature_names_out().tolist() == OUTPUT_FEATURE_COLUMNS
    assert len(OUTPUT_FEATURE_COLUMNS) == 13
    for name, output in outputs.items():
        validate_transformed(inputs[name], output)
        assert output.shape == (len(inputs[name]), 13)
        assert output.columns.tolist() == OUTPUT_FEATURE_COLUMNS
        assert not output.isna().any().any()
        assert np.isfinite(output.to_numpy()).all()
        assert output[INDICATOR_COLUMNS].isin([0, 1]).all().all()
        assert {"SeriousDlqin2yrs", "Unnamed: 0", "group_id", "feature_group", "source_ids"}.isdisjoint(output.columns)


def test_column_order_is_resolved_by_name(fitted, inputs):
    _, preprocessor, outputs = fitted
    X = inputs["train"]
    pd.testing.assert_frame_equal(preprocessor.transform(X[X.columns[::-1]]), outputs["train"])
    reordered_fit = clone(preprocessor).fit(X[X.columns[::-1]])
    pd.testing.assert_frame_equal(reordered_fit.transform(X), outputs["train"])
    assert reordered_fit.get_feature_names_out().tolist() == OUTPUT_FEATURE_COLUMNS


def test_train_only_medians_for_all_six_columns(fitted, inputs):
    _, preprocessor, _ = fitted
    expected = inputs["train"][IMPUTATION_COLUMNS].copy()
    expected["age"] = expected["age"].mask(expected["age"].eq(0))
    expected[DELINQUENCY_COLUMNS] = expected[DELINQUENCY_COLUMNS].mask(
        expected[DELINQUENCY_COLUMNS].isin(SPECIAL_DELINQUENCY_VALUES)
    )
    pd.testing.assert_series_equal(preprocessor.named_steps["imputer"].statistics_, expected.median())


def test_evaluation_transform_cannot_refit_or_change_statistics(fitted, inputs):
    variant, preprocessor, _ = fitted
    imputer = preprocessor.named_steps["imputer"]
    before = imputer.statistics_.copy()
    scaler = preprocessor.named_steps["scaling"].named_transformers_["numeric"] if variant == "linear" else None
    scaler_state = {key: np.array(getattr(scaler, key), copy=True) for key in ("mean_", "var_", "scale_", "n_samples_seen_")} if scaler is not None else {}
    # A transform that accidentally calls fit or fit_transform must fail.
    from contextlib import ExitStack
    with ExitStack() as stack:
        estimators = [preprocessor, *preprocessor.named_steps.values()]
        if scaler is not None:
            estimators.append(scaler)
        for estimator in estimators:
            for method in ("fit", "fit_transform"):
                stack.enter_context(patch.object(estimator, method, side_effect=AssertionError("Evaluation refit")))
        for name in ("validation", "test"):
            preprocessor.transform(inputs[name])
    pd.testing.assert_series_equal(imputer.statistics_, before)
    for key, value in scaler_state.items():
        np.testing.assert_array_equal(getattr(scaler, key), value)


def test_tree_preserves_unaffected_values_and_has_no_scaling(inputs):
    tree = build_tree_preprocessor().fit(inputs["train"])
    assert list(tree.named_steps) == ["quality", "imputer"]
    for X in inputs.values():
        output = tree.transform(X)
        for column in FEATURE_COLUMNS:
            valid = X[column].notna()
            if column == "age":
                valid &= X[column].ne(0)
            elif column in DELINQUENCY_COLUMNS:
                valid &= ~X[column].isin(SPECIAL_DELINQUENCY_VALUES)
            np.testing.assert_array_equal(output.loc[valid, column], X.loc[valid, column])
        assert not output["age"].eq(0).any()
        assert not output[DELINQUENCY_COLUMNS].isin([96, 98]).any().any()


def test_linear_scaler_fits_only_train_and_leaves_indicators_unscaled(inputs):
    linear = build_linear_preprocessor().fit(inputs["train"])
    tree = build_tree_preprocessor().fit(inputs["train"])
    numeric_train = tree.transform(inputs["train"])[FEATURE_COLUMNS]
    scaler = linear.named_steps["scaling"].named_transformers_["numeric"]
    np.testing.assert_allclose(scaler.mean_, numeric_train.mean().to_numpy())
    np.testing.assert_allclose(scaler.var_, numeric_train.var(ddof=0).to_numpy())
    assert scaler.n_samples_seen_ == len(inputs["train"])
    for name, X in inputs.items():
        scaled = linear.transform(X)
        unscaled = tree.transform(X)
        np.testing.assert_allclose(scaled[FEATURE_COLUMNS], (unscaled[FEATURE_COLUMNS] - scaler.mean_) / scaler.scale_)
        pd.testing.assert_frame_equal(scaled[INDICATOR_COLUMNS], unscaled[INDICATOR_COLUMNS])
        if name == "train":
            np.testing.assert_allclose(scaled[FEATURE_COLUMNS].mean(), 0, atol=1e-12)
            np.testing.assert_allclose(scaled[FEATURE_COLUMNS].var(ddof=0), 1, atol=1e-12)


def test_rules_after_full_pipeline_and_income_zero_semantics(fitted, rule_rows):
    variant, preprocessor, _ = fitted
    output = preprocessor.transform(rule_rows)
    unscaled = output[FEATURE_COLUMNS]
    if variant == "linear":
        scaler = preprocessor.named_steps["scaling"].named_transformers_["numeric"]
        unscaled = pd.DataFrame(scaler.inverse_transform(unscaled), columns=FEATURE_COLUMNS, index=rule_rows.index)
    medians = preprocessor.named_steps["imputer"].statistics_
    assert unscaled["age"].iloc[0] == pytest.approx(medians["age"])
    assert unscaled["age"].iloc[1] == pytest.approx(109)
    for row, column in enumerate(DELINQUENCY_COLUMNS):
        assert unscaled[column].iloc[row] == pytest.approx(medians[column])
    assert unscaled["MonthlyIncome"].iloc[0] == pytest.approx(0, abs=1e-10)
    assert unscaled["MonthlyIncome"].iloc[1] == pytest.approx(medians["MonthlyIncome"])
    assert output["MonthlyIncome_missing"].iloc[0] == 0
    assert output["MonthlyIncome_missing"].iloc[1] == 1


def test_fixed_indicators_when_fitting_fold_has_no_missing(inputs):
    X = inputs["train"].dropna().head(100).copy()
    X["age"] = 40
    X[DELINQUENCY_COLUMNS] = 0
    tree = build_tree_preprocessor().fit(X)
    later = X.head(2).copy()
    later["MonthlyIncome"] = np.nan
    later["NumberOfDependents"] = np.nan
    later[DELINQUENCY_COLUMNS[0]] = 98
    result = tree.transform(later)
    assert result.columns.tolist() == OUTPUT_FEATURE_COLUMNS
    assert result[INDICATOR_COLUMNS].eq(1).all().all()
    assert not result.isna().any().any()


@pytest.mark.parametrize("column", IMPUTATION_COLUMNS)
def test_all_missing_training_column_fails_explicitly(inputs, column):
    X = inputs["train"].head(10).copy()
    X[column] = 0 if column == "age" else (98 if column in DELINQUENCY_COLUMNS else np.nan)
    with pytest.raises(PreprocessingValidationError, match="All-missing training columns"):
        build_tree_preprocessor().fit(X)


def test_unapproved_missingness_requires_explicit_policy(inputs):
    X = inputs["train"].head(10).copy()
    X["DebtRatio"] = np.nan
    with pytest.raises(PreprocessingValidationError, match="No imputation policy"):
        build_tree_preprocessor().fit(X)


def test_repeated_fit_deterministic_and_inputs_unmodified(fitted, inputs):
    _, preprocessor, outputs = fitted
    before = {name: X.copy(deep=True) for name, X in inputs.items()}
    repeated = clone(preprocessor).fit(inputs["train"])
    for name, X in inputs.items():
        pd.testing.assert_frame_equal(repeated.transform(X), outputs[name])
        pd.testing.assert_frame_equal(X, before[name])


@pytest.mark.parametrize("outer", [Pipeline, ImbalancedPipeline])
@pytest.mark.parametrize("factory", [build_tree_preprocessor, build_linear_preprocessor])
def test_clone_and_nested_pipeline_fold_fitting(inputs, outer, factory):
    prototype = outer([("preprocessor", factory())])
    # Fit independent clones on train-only fold subsets, without any model.
    for start in (0, 1000):
        fold = inputs["train"].iloc[start:start + 1000]
        pipeline = clone(prototype).fit(fold)
        transformer = pipeline.named_steps["preprocessor"]
        assert transformer.named_steps["imputer"].statistics_["MonthlyIncome"] == fold["MonthlyIncome"].median()
        validate_transformed(fold, pipeline.transform(fold))
    assert not hasattr(prototype.named_steps["preprocessor"].named_steps["imputer"], "statistics_")


@pytest.mark.parametrize("kind", ["nan", "infinity", "order", "rows", "indicator"])
def test_output_validator_rejects_invalid_outputs(fitted, inputs, kind):
    _, _, outputs = fitted
    X = inputs["train"].head(5)
    broken = outputs["train"].head(5).copy()
    if kind == "nan":
        broken.iloc[0, 0] = np.nan
    elif kind == "infinity":
        broken.iloc[0, 0] = np.inf
    elif kind == "order":
        broken = broken[broken.columns[::-1]]
    elif kind == "rows":
        broken = broken.iloc[::-1]
    else:
        broken[SPECIAL_INDICATOR] = 2
    with pytest.raises(PreprocessingValidationError):
        validate_transformed(X, broken)
