"""Baseline contracts using train/validation only; no test-set evaluation."""

from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

import src.baseline_model as baseline
from src.preprocessing import OUTPUT_FEATURE_COLUMNS


@pytest.fixture(scope="module")
def data():
    return baseline.load_training_validation()


@pytest.fixture(scope="module")
def result(data):
    return baseline.run_baseline(data)


def test_unfitted_pipeline_structure_and_configuration():
    pipeline = baseline.build_baseline_pipeline()
    assert list(pipeline.named_steps) == ["preprocessor", "model"]
    model = pipeline.named_steps["model"]
    assert isinstance(model, LogisticRegression)
    assert model.class_weight is None
    assert model.C == 1 and model.solver == "lbfgs"
    assert model.l1_ratio == 0 and model.max_iter == 2000
    assert model.random_state == 42
    assert not hasattr(model, "coef_")
    assert not hasattr(pipeline.named_steps["preprocessor"].named_steps["imputer"], "statistics_")


def test_pipeline_integrity_and_finite_coefficients(result, data):
    table = baseline.audit_pipeline(result.pipeline, data.X_train)
    model = result.pipeline.named_steps["model"]
    assert table.shape == (13, 2)
    assert table.feature.tolist() == OUTPUT_FEATURE_COLUMNS
    assert model.n_features_in_ == 13
    assert model.coef_.shape == (1, 13)
    assert np.isfinite(model.coef_).all() and np.isfinite(model.intercept_).all()
    assert set(OUTPUT_FEATURE_COLUMNS).isdisjoint({"SeriousDlqin2yrs", "Unnamed: 0", "group_id", "source_ids"})
    assert result.report["convergence"]["converged"] is True
    assert (model.n_iter_ < model.max_iter).all()


def test_probabilities_shape_range_and_finiteness(result, data):
    for X in (data.X_train, data.X_validation):
        probabilities = result.pipeline.predict_proba(X)
        assert probabilities.shape == (len(X), 2)
        assert np.isfinite(probabilities).all()
        assert ((probabilities >= 0) & (probabilities <= 1)).all()
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        np.testing.assert_array_equal(baseline.positive_probabilities(result.pipeline, X), probabilities[:, 1])


def test_deterministic_training_and_predictions(result, data):
    repeated = baseline.fit_baseline(data.X_train, data.y_train)
    np.testing.assert_allclose(repeated.named_steps["model"].coef_, result.pipeline.named_steps["model"].coef_, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(baseline.positive_probabilities(repeated, data.X_validation), result.validation_probabilities, rtol=1e-12, atol=1e-12)


def test_preprocessing_fitted_inside_pipeline_only_on_train(data):
    pipeline = baseline.build_baseline_pipeline()
    preprocessor = pipeline.named_steps["preprocessor"]
    with patch.object(baseline, "build_baseline_pipeline", return_value=pipeline), patch.object(
        preprocessor, "fit_transform", wraps=preprocessor.fit_transform
    ) as fitted:
        baseline.fit_baseline(data.X_train, data.y_train)
    assert fitted.call_count == 1
    assert fitted.call_args.args[0] is data.X_train
    assert fitted.call_args.args[1] is data.y_train
    imputer = preprocessor.named_steps["imputer"]
    assert imputer.statistics_["MonthlyIncome"] == data.X_train.MonthlyIncome.median()
    scaler = preprocessor.named_steps["scaling"].named_transformers_["numeric"]
    assert scaler.n_samples_seen_ == len(data.X_train)
    before = imputer.statistics_.copy()
    with patch.object(preprocessor, "fit", side_effect=AssertionError("Refit")), patch.object(
        preprocessor, "fit_transform", side_effect=AssertionError("Refit")
    ):
        baseline.positive_probabilities(pipeline, data.X_validation)
    pd.testing.assert_series_equal(imputer.statistics_, before)


def test_default_threshold_including_boundary():
    np.testing.assert_array_equal(baseline.predict_at_default_threshold(np.array([0, .499, .5, .501, 1])), [0, 0, 1, 1, 1])


def test_metrics_on_controlled_inputs():
    metrics = baseline.calculate_metrics(np.array([0, 0, 1, 1]), np.array([.1, .6, .4, .8]))
    assert metrics["roc_auc"] == pytest.approx(.75)
    assert metrics["average_precision"] == pytest.approx(5 / 6)
    for key in ("precision", "recall", "f1", "accuracy", "positive_prevalence", "predicted_positive_rate", "gini"):
        assert metrics[key] == pytest.approx(.5)
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]


def test_metrics_with_no_positive_predictions():
    metrics = baseline.calculate_metrics(np.array([0, 1]), np.array([.1, .2]))
    assert metrics["precision"] == metrics["recall"] == metrics["f1"] == 0
    assert metrics["confusion_matrix"] == [[1, 0], [1, 0]]


@pytest.mark.parametrize("values", [[np.nan], [np.inf], [-.1], [1.1], [], [[.5]]])
def test_invalid_probabilities_rejected(values):
    with pytest.raises(ValueError):
        baseline.predict_at_default_threshold(np.asarray(values))


def test_training_validation_boundary_does_not_access_test(monkeypatch):
    class LockedSplits:
        train = SimpleNamespace(X="train X", y="train y")
        validation = SimpleNamespace(X="validation X", y="validation y")

        @property
        def test(self):
            raise AssertionError("Test partition must not be accessed by modeling")

    monkeypatch.setattr(baseline, "load_data", lambda: None)
    monkeypatch.setattr(baseline, "split_data", lambda raw: LockedSplits())
    loaded = baseline.load_training_validation()
    assert loaded.X_train == "train X" and loaded.y_validation == "validation y"
    assert [field.name for field in fields(loaded)] == ["X_train", "y_train", "X_validation", "y_validation"]


def test_report_required_keys_and_no_test_metrics(result):
    assert {"model", "threshold", "train", "validation", "convergence", "coefficients"} <= result.report.keys()
    assert "test" not in result.report
    required = {"roc_auc", "average_precision", "precision", "recall", "f1", "accuracy", "confusion_matrix", "positive_prevalence", "predicted_positive_rate"}
    for partition in ("train", "validation"):
        assert required <= result.report[partition].keys()
        for key in required - {"confusion_matrix"}:
            assert 0 <= result.report[partition][key] <= 1


def test_nonconvergence_fails_loudly(data):
    import warnings

    def fail_fit(*args, **kwargs):
        warnings.warn("Controlled convergence failure", ConvergenceWarning)

    with patch("sklearn.pipeline.Pipeline.fit", side_effect=fail_fit):
        with pytest.raises(RuntimeError, match="increase max_iter only"):
            baseline.fit_baseline(data.X_train, data.y_train)


def test_misaligned_labels_rejected(data):
    with pytest.raises(ValueError, match="source-row index"):
        baseline.fit_baseline(data.X_train, data.y_train.iloc[::-1])
