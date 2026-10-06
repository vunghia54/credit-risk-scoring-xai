"""Small-data tree tests; the full 105k-row tree CV runs only via the module."""

from copy import deepcopy
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data_split import FEATURE_COLUMNS, TARGET_COLUMN, build_feature_groups, load_data
from src.preprocessing import OUTPUT_FEATURE_COLUMNS
import src.tree_model_experiments as trees


@pytest.fixture(scope="module")
def small_data():
    # Existing real rows, in memory only. No new dataset or model file is saved.
    raw = load_data().head(600)
    return trees.ExperimentData(
        raw[FEATURE_COLUMNS].iloc[:400], raw[TARGET_COLUMN].iloc[:400],
        build_feature_groups(raw).iloc[:400],
        raw[FEATURE_COLUMNS].iloc[400:], raw[TARGET_COLUMN].iloc[400:],
    )


@pytest.fixture(scope="module", params=trees.TREE_NAMES)
def fitted(request, small_data):
    name = request.param
    pipeline = trees.fit_tree(trees.build_tree_experiments()[name], small_data.X_train, small_data.y_train)
    return name, pipeline


def test_exact_models_configuration_and_unscaled_preprocessing():
    experiments = trees.build_tree_experiments()
    assert tuple(experiments) == ("random_forest", "xgboost", "lightgbm")
    classes = {"random_forest": RandomForestClassifier, "xgboost": XGBClassifier, "lightgbm": LGBMClassifier}
    for name, pipeline in experiments.items():
        assert list(pipeline.named_steps) == ["preprocessor", "model"]
        assert isinstance(pipeline.named_steps["model"], classes[name])
        params = pipeline.named_steps["model"].get_params()
        for key, value in trees.TREE_CONFIGURATIONS[name].items():
            assert params[key] == value
        assert params["random_state"] == 42 and params["n_estimators"] == 300
        preprocessor = pipeline.named_steps["preprocessor"]
        assert list(preprocessor.named_steps) == ["quality", "imputer"]
        assert not any(isinstance(value, StandardScaler) for value in preprocessor.get_params(deep=True).values())
        assert params.get("class_weight") is None
        assert params.get("scale_pos_weight") is None
        assert params.get("early_stopping_rounds") is None
        assert not params.get("is_unbalance", False)
    rf = experiments["random_forest"].named_steps["model"]
    assert rf.max_depth is None and rf.min_samples_leaf == 1


def test_group_cv_uses_raw_groups_with_no_overlap(small_data):
    from src.imbalance_experiments import make_cv
    cv = make_cv()
    assert cv.__class__.__name__ == "StratifiedGroupKFold"
    assert cv.n_splits == 5 and cv.random_state == 42 and cv.shuffle
    folds = trees.make_cv_folds(small_data)
    evaluation_rows = []
    for fitting, evaluation in folds:
        assert set(small_data.groups_train.iloc[fitting]).isdisjoint(small_data.groups_train.iloc[evaluation])
        assert set(fitting).isdisjoint(evaluation)
        evaluation_rows.extend(evaluation)
    np.testing.assert_array_equal(np.sort(evaluation_rows), np.arange(400))


def test_probability_shape_finiteness_size_and_feature_schema(fitted, small_data):
    _, pipeline = fitted
    X_before = small_data.X_validation.copy(deep=True)
    y_before = small_data.y_validation.copy(deep=True)
    raw_probabilities = pipeline.predict_proba(small_data.X_validation)
    probabilities = trees.positive_probabilities(pipeline, small_data.X_validation)
    assert raw_probabilities.shape == (200, 2) and probabilities.shape == (200,)
    assert np.isfinite(raw_probabilities).all()
    assert ((raw_probabilities >= 0) & (raw_probabilities <= 1)).all()
    np.testing.assert_allclose(raw_probabilities.sum(axis=1), 1)
    preprocessor = pipeline.named_steps["preprocessor"]
    assert preprocessor.get_feature_names_out().tolist() == OUTPUT_FEATURE_COLUMNS
    assert pipeline.named_steps["model"].n_features_in_ == 13
    assert set(OUTPUT_FEATURE_COLUMNS).isdisjoint({"SeriousDlqin2yrs", "Unnamed: 0", "feature_group", "group_id"})
    with patch.object(preprocessor, "fit", side_effect=AssertionError("Evaluation refit")), patch.object(preprocessor, "fit_transform", side_effect=AssertionError("Evaluation refit")):
        trees.positive_probabilities(pipeline, small_data.X_validation)
    pd.testing.assert_frame_equal(small_data.X_validation, X_before)
    pd.testing.assert_series_equal(small_data.y_validation, y_before)


def test_native_importance_mapping_and_ranking(fitted, small_data):
    name, pipeline = fitted
    audit = trees.importance_audit(name, pipeline, small_data.X_train)
    assert audit.shape == (13, 4)
    assert audit.columns.tolist() == ["model", "feature", "importance", "rank"]
    assert set(audit.feature) == set(OUTPUT_FEATURE_COLUMNS)
    assert audit["rank"].tolist() == list(range(1, 14))
    assert audit.importance.is_monotonic_decreasing
    assert np.isfinite(audit.importance).all()
    if name == "random_forest":
        assert audit.importance.sum() == pytest.approx(1)


def test_deterministic_small_data_predictions(fitted, small_data):
    name, pipeline = fitted
    repeated = trees.fit_tree(trees.build_tree_experiments()[name], small_data.X_train, small_data.y_train)
    np.testing.assert_allclose(trees.positive_probabilities(repeated, small_data.X_validation),
                               trees.positive_probabilities(pipeline, small_data.X_validation), atol=1e-10, rtol=1e-10)


def test_fold_preprocessor_is_fitted_on_fitting_rows_only(fitted, small_data):
    name, _ = fitted
    fitting, evaluation = trees.make_cv_folds(small_data)[0]
    captured = []

    def capture_clone(prototype):
        copied = clone(prototype)
        captured.append(copied)
        return copied

    with patch.object(trees, "clone", side_effect=capture_clone):
        report = trees.tree_cv(trees.build_tree_experiments()[name], small_data, [(fitting, evaluation)])
    assert len(captured) == 1
    imputer = captured[0].named_steps["preprocessor"].named_steps["imputer"]
    assert imputer.statistics_["MonthlyIncome"] == small_data.X_train.iloc[fitting].MonthlyIncome.median()
    assert report["folds"][0]["fitting_rows"] == len(fitting)
    assert report["folds"][0]["evaluation_rows"] == len(evaluation)
    assert report["folds"][0]["group_overlap"] == 0
    assert report["std_ddof"] == 0


def test_metrics_correct_on_toy_inputs():
    metrics = trees.calculate_metrics(np.array([0, 0, 1, 1]), np.array([.1, .6, .4, .8]))
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["roc_auc"] == .75
    assert metrics["average_precision"] == pytest.approx(5 / 6)
    assert metrics["precision"] == metrics["recall"] == metrics["f1"] == .5
    assert trees.THRESHOLD == .5


def test_saved_logistic_reference_schema_and_values():
    references = trees.load_logistic_references(22486)
    assert tuple(references) == trees.LOGISTIC_NAMES
    for entry in references.values():
        trees.validate_reference(entry, 22486)


@pytest.mark.parametrize("corruption", ["missing", "nan", "range", "rows", "matrix", "cv", "gini"])
def test_invalid_logistic_artifacts_rejected(corruption):
    entry = deepcopy(trees.load_logistic_references(22486)["logistic_baseline"])
    if corruption == "missing":
        del entry["validation"]["recall"]
    elif corruption == "nan":
        entry["validation"]["roc_auc"] = np.nan
    elif corruption == "range":
        entry["validation"]["precision"] = 1.1
    elif corruption == "rows":
        entry["validation"]["rows"] = 1
    elif corruption == "matrix":
        entry["validation"]["confusion_matrix"][0][0] -= 1
    elif corruption == "cv":
        entry["cv"]["summary"]["roc_auc"]["mean"] = 0
    else:
        entry["validation"]["gini"] = 0
    with pytest.raises(ValueError):
        trees.validate_reference(entry, 22486)


def test_comparison_table_schema_and_actual_values(fitted, small_data):
    _, pipeline = fitted
    references = trees.load_logistic_references(22486)
    metrics = trees.calculate_metrics(small_data.y_validation, trees.positive_probabilities(pipeline, small_data.X_validation))
    # Controlled table-shape fixture only; these records are never exported.
    toy_entry = {"validation": metrics, "cv": {"summary": {
        key: {"mean": metrics[key], "std": 0.0} for key in ("roc_auc", "average_precision")}}}
    table = trees.comparison_table({**references, **{name: toy_entry for name in trees.TREE_NAMES}})
    assert table.columns.tolist() == trees.COMPARISON_COLUMNS
    assert table.shape == (5, len(trees.COMPARISON_COLUMNS))
    assert table.model.tolist() == list(trees.DISPLAY_NAMES.values())
    assert table.loc[0, "validation_roc_auc"] == references["logistic_baseline"]["validation"]["roc_auc"]
    assert table.loc[2, "validation_roc_auc"] == metrics["roc_auc"]
    assert np.isfinite(table.drop(columns="model").to_numpy()).all()


def test_loader_and_api_never_access_test_target(monkeypatch):
    class Locked:
        train = SimpleNamespace(X="train X", y="train y", groups="raw groups")
        validation = SimpleNamespace(X="val X", y="val y")

        @property
        def test(self):
            raise AssertionError("Test is locked")

    monkeypatch.setattr(trees, "load_data", lambda: None)
    monkeypatch.setattr(trees, "split_data", lambda raw: Locked())
    data = trees.load_tree_data()
    assert data.groups_train == "raw groups"
    assert [field.name for field in fields(data)] == ["X_train", "y_train", "groups_train", "X_validation", "y_validation"]


def test_curve_reproduction_rejects_reference_drift():
    expected = trees.load_logistic_references(22486)["logistic_baseline"]["validation"]
    actual = deepcopy(expected)
    actual["roc_auc"] -= .01
    with pytest.raises(ValueError, match="reproduction failed"):
        trees.verify_reference_predictions(actual, expected)
