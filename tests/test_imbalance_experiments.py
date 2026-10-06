"""Train-only CV and fixed validation comparison; test partition stays locked."""

from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold

import src.imbalance_experiments as experiments


@pytest.fixture(scope="module")
def data():
    return experiments.load_experiment_data()


@pytest.fixture(scope="module")
def result(data):
    X_before = data.X_validation.copy(deep=True)
    y_before = data.y_validation.copy(deep=True)
    outcome = experiments.run_experiments(data)
    pd.testing.assert_frame_equal(data.X_validation, X_before)
    pd.testing.assert_series_equal(data.y_validation, y_before)
    return outcome


def test_exact_three_configurations_and_fair_model_parameters():
    pipelines = experiments.build_experiments()
    assert tuple(pipelines) == ("baseline", "balanced", "smote")
    reference = pipelines["baseline"].named_steps["model"].get_params()
    for name, pipeline in pipelines.items():
        params = pipeline.named_steps["model"].get_params()
        assert params["class_weight"] == ("balanced" if name == "balanced" else None)
        assert {k: v for k, v in params.items() if k != "class_weight"} == {k: v for k, v in reference.items() if k != "class_weight"}
        assert params["C"] == 1 and params["solver"] == "lbfgs" and params["max_iter"] == 2000
        assert list(pipeline.named_steps["preprocessor"].named_steps) == ["quality", "imputer", "scaling"]
    smote = pipelines["smote"]
    assert isinstance(smote, ImbalancedPipeline)
    assert list(smote.named_steps) == ["preprocessor", "smote", "model"]
    assert smote.named_steps["smote"].random_state == 42
    assert "smote" not in pipelines["balanced"].named_steps


def test_group_cv_design_is_deterministic_and_has_zero_overlap(data):
    cv = experiments.make_cv()
    assert isinstance(cv, StratifiedGroupKFold)
    assert cv.n_splits == 5 and cv.shuffle and cv.random_state == 42
    folds = experiments.make_cv_folds(data)
    repeated = experiments.make_cv_folds(data)
    evaluated = []
    for (fit, validation), (fit_again, validation_again) in zip(folds, repeated):
        np.testing.assert_array_equal(fit, fit_again)
        np.testing.assert_array_equal(validation, validation_again)
        assert set(data.groups_train.iloc[fit]).isdisjoint(data.groups_train.iloc[validation])
        assert set(fit).isdisjoint(validation)
        evaluated.extend(validation)
    np.testing.assert_array_equal(np.sort(evaluated), np.arange(len(data.X_train)))


def test_adapter_preserves_original_preprocessing_exactly(data):
    pipelines = experiments.build_experiments()
    direct = pipelines["baseline"].named_steps["preprocessor"].fit(data.X_train)
    adapted = clone(pipelines["smote"].named_steps["preprocessor"]).fit(data.X_train)
    for X in (data.X_train, data.X_validation):
        pd.testing.assert_frame_equal(direct.transform(X), adapted.transform(X))
    assert adapted.get_feature_names_out().tolist() == direct.get_feature_names_out().tolist()


def test_result_schema_cv_fold_local_smote_and_valid_probabilities(result, data):
    report = result.report
    assert report["threshold"] == .5 and report["random_state"] == 42
    assert report["baseline_reproduction"] == "PASS"
    assert "test" not in report
    assert set(report["experiments"]) == set(experiments.EXPERIMENT_NAMES)
    for name, entry in report["experiments"].items():
        assert {"configuration", "cv", "validation", "convergence"} <= entry.keys()
        assert "test" not in entry
        assert len(entry["cv"]["folds"]) == 5
        for key in ("roc_auc", "average_precision"):
            values = [fold[key] for fold in entry["cv"]["folds"]]
            assert entry["cv"]["summary"][key]["mean"] == pytest.approx(np.mean(values))
            assert entry["cv"]["summary"][key]["std"] == pytest.approx(np.std(values))
        for fold in entry["cv"]["folds"]:
            assert fold["group_overlap"] == 0
            assert fold["fitting_rows"] + fold["evaluation_rows"] == len(data.X_train)
            if name == "smote":
                audit = fold["smote"]
                assert audit["input_rows"] == fold["fitting_rows"] < len(data.X_train)
                assert sum(audit["before"].values()) == fold["fitting_rows"]
                assert audit["after"]["0"] == audit["after"]["1"]
            else:
                assert "smote" not in fold
        probabilities = result.validation_probabilities[name]
        assert probabilities.shape == (len(data.y_validation),)
        assert np.isfinite(probabilities).all()
        assert ((probabilities >= 0) & (probabilities <= 1)).all()
        assert entry["validation"]["rows"] == len(data.y_validation)
        assert entry["convergence"]["converged"] and not entry["convergence"]["warning"]


def test_baseline_reproduces_step8(result):
    experiments.check_baseline_reproduction(result.report["experiments"]["baseline"]["validation"])


def test_baseline_mismatch_stops_comparison(data):
    with patch.object(experiments, "check_baseline_reproduction", side_effect=ValueError("Mismatch")), patch.object(experiments, "run_cv") as cv:
        with pytest.raises(ValueError, match="Mismatch"):
            experiments.run_experiments(data)
        cv.assert_not_called()


def test_reproduction_check_rejects_changed_metric(result):
    metrics = dict(result.report["experiments"]["baseline"]["validation"])
    metrics["roc_auc"] = 0
    with pytest.raises(ValueError, match="reproduction failed"):
        experiments.check_baseline_reproduction(metrics)


def test_smote_audit_uses_actual_full_training_counts(result, data):
    audit = result.report["experiments"]["smote"]["smote_audit"]
    counts = experiments.class_counts(data.y_train)
    assert audit["before_rows"] == len(data.y_train)
    assert audit["before_class_counts"] == counts
    assert audit["after_class_counts"] == {"0": max(counts.values()), "1": max(counts.values())}
    assert audit["after_rows"] == 2 * max(counts.values())


def test_smote_only_fits_fitting_rows_never_evaluation(data):
    fitting, evaluation = experiments.make_cv_folds(data)[0]
    pipeline = experiments.build_experiments()["smote"]
    sampler = pipeline.named_steps["smote"]
    with patch.object(sampler, "fit_resample", wraps=sampler.fit_resample) as resample:
        experiments.fit_checked(pipeline, data.X_train.iloc[fitting], data.y_train.iloc[fitting])
        assert resample.call_count == 1
        assert len(resample.call_args.args[0]) == len(fitting)
        assert len(resample.call_args.args[1]) == len(fitting)
    preprocessor = pipeline.named_steps["preprocessor"]
    assert preprocessor.named_steps["imputer"].statistics_["MonthlyIncome"] == data.X_train.iloc[fitting].MonthlyIncome.median()
    assert preprocessor.named_steps["scaling"].named_transformers_["numeric"].n_samples_seen_ == len(fitting)
    with patch.object(sampler, "fit_resample", side_effect=AssertionError("Evaluation resampling")), patch.object(preprocessor, "fit_transform", side_effect=AssertionError("Evaluation refit")):
        for X in (data.X_train.iloc[evaluation], data.X_validation):
            assert len(experiments.positive_probabilities(pipeline, X)) == len(X)


@pytest.mark.parametrize("name", experiments.EXPERIMENT_NAMES)
def test_deterministic_full_training_predictions(name, result, data):
    fitted = experiments.fit_checked(clone(experiments.build_experiments()[name]), data.X_train, data.y_train)
    np.testing.assert_allclose(experiments.positive_probabilities(fitted, data.X_validation), result.validation_probabilities[name], atol=1e-12, rtol=1e-12)
    assert np.isfinite(fitted.named_steps["model"].coef_).all()


def test_metrics_use_fixed_threshold_on_controlled_inputs():
    metrics = experiments.calculate_metrics(np.array([0, 0, 1, 1]), np.array([.1, .6, .4, .8]))
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["precision"] == metrics["recall"] == metrics["f1"] == .5
    assert metrics["roc_auc"] == .75
    assert metrics["average_precision"] == pytest.approx(5 / 6)


def test_api_excludes_and_does_not_access_test_partition(monkeypatch):
    class Locked:
        train = SimpleNamespace(X="X", y="y", groups="raw groups")
        validation = SimpleNamespace(X="val X", y="val y")

        @property
        def test(self):
            raise AssertionError("Test is locked")

    monkeypatch.setattr(experiments, "load_data", lambda: None)
    monkeypatch.setattr(experiments, "split_data", lambda raw: Locked())
    data = experiments.load_experiment_data()
    assert data.groups_train == "raw groups"
    assert [field.name for field in fields(data)] == ["X_train", "y_train", "groups_train", "X_validation", "y_validation"]
