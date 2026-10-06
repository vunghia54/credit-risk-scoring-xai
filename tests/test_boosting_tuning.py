"""Small-data and selection-contract tests; never run the full 16 x 5 search."""

from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

import src.boosting_tuning as tuning
import src.tree_model_experiments as trees
from src.data_split import FEATURE_COLUMNS, TARGET_COLUMN, build_feature_groups, load_data
from src.preprocessing import OUTPUT_FEATURE_COLUMNS


@pytest.fixture(scope="module")
def small_train():
    raw = load_data().head(500)
    return tuning.TrainingData(raw[FEATURE_COLUMNS], raw[TARGET_COLUMN], build_feature_groups(raw))


def cv_result(identifier, ap, auc):
    algorithm = "xgboost" if identifier.startswith("XGB") else "lightgbm"
    return tuning.CandidateCV(tuning.Candidate(algorithm, identifier, {}), {
        "summary": {"average_precision": {"mean": ap, "std": .01},
                    "roc_auc": {"mean": auc, "std": .01}}, "folds": [], "std_ddof": 0,
    }, 0.0)


def test_exact_predeclared_configurations_without_cartesian_expansion():
    candidates = tuning.candidate_configurations()
    assert len(candidates) == len({c.configuration_id for c in candidates}) == 16
    assert [c.configuration_id for c in candidates] == [f"XGB-{i:02d}" for i in range(1, 9)] + [f"LGBM-{i:02d}" for i in range(1, 9)]
    xgb_keys = ["n_estimators", "learning_rate", "max_depth", "min_child_weight", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"]
    xgb_expected = [
        (300, .05, 3, 1, 1., 1., 0, 1), (300, .05, 2, 1, 1., 1., 0, 1),
        (300, .05, 4, 1, 1., 1., 0, 1), (500, .03, 3, 1, 1., 1., 0, 1),
        (200, .08, 3, 1, 1., 1., 0, 1), (300, .05, 3, 5, 1., 1., 0, 1),
        (300, .05, 3, 1, .8, .8, 0, 1), (300, .05, 3, 1, 1., 1., 0, 5),
    ]
    lgbm_keys = ["n_estimators", "learning_rate", "num_leaves", "max_depth", "min_child_samples", "subsample", "subsample_freq", "colsample_bytree", "reg_alpha", "reg_lambda"]
    lgbm_expected = [
        (300, .05, 31, -1, 20, 1., 0, 1., 0, 0), (300, .05, 15, -1, 20, 1., 0, 1., 0, 0),
        (300, .05, 63, -1, 20, 1., 0, 1., 0, 0), (500, .03, 31, -1, 20, 1., 0, 1., 0, 0),
        (200, .08, 31, -1, 20, 1., 0, 1., 0, 0), (300, .05, 31, -1, 50, 1., 0, 1., 0, 0),
        (300, .05, 31, -1, 20, .8, 1, .8, 0, 0), (300, .05, 31, -1, 20, 1., 0, 1., 0, 5),
    ]
    assert [tuple(c.parameters[k] for k in xgb_keys) for c in candidates[:8]] == xgb_expected
    assert [tuple(c.parameters[k] for k in lgbm_keys) for c in candidates[8:]] == lgbm_expected


def test_all_pipelines_have_fixed_seed_tree_preprocessor_and_no_imbalance():
    for candidate in tuning.candidate_configurations():
        pipeline = tuning.build_candidate(candidate)
        assert list(pipeline.named_steps) == ["preprocessor", "model"]
        preprocessor = pipeline.named_steps["preprocessor"]
        assert list(preprocessor.named_steps) == ["quality", "imputer"]
        assert not any(isinstance(value, StandardScaler) for value in preprocessor.get_params(deep=True).values())
        params = pipeline.named_steps["model"].get_params()
        assert params["random_state"] == 42 and params["n_jobs"] == -1
        for key, value in candidate.parameters.items():
            assert params[key] == value
        assert params.get("class_weight") is None
        assert params.get("scale_pos_weight") is None
        assert not params.get("is_unbalance", False)
        assert params.get("early_stopping_rounds") is None


def test_baseline_configurations_match_step10_starter_parameters():
    old = trees.build_tree_experiments()
    for candidate in tuning.candidate_configurations():
        if candidate.configuration_id.endswith("-01"):
            actual = tuning.build_candidate(candidate).named_steps["model"].get_params()
            for key, value in trees.TREE_CONFIGURATIONS[candidate.algorithm].items():
                assert actual[key] == value
            assert old[candidate.algorithm].named_steps["model"].get_params()["n_estimators"] == actual["n_estimators"]
            assert actual["reg_alpha"] == 0
            assert actual["reg_lambda"] == (1 if candidate.algorithm == "xgboost" else 0)


def test_group_cv_is_reproducible_and_group_disjoint(small_train):
    cv = tuning.make_cv()
    assert cv.__class__.__name__ == "StratifiedGroupKFold"
    assert cv.n_splits == 5 and cv.random_state == 42 and cv.shuffle
    folds = tuning.training_folds(small_train)
    repeated = tuning.training_folds(small_train)
    evaluation_rows = []
    for (fitting, evaluation), (fit2, eval2) in zip(folds, repeated):
        np.testing.assert_array_equal(fitting, fit2)
        np.testing.assert_array_equal(evaluation, eval2)
        assert set(small_train.groups.iloc[fitting]).isdisjoint(small_train.groups.iloc[evaluation])
        evaluation_rows.extend(evaluation)
    np.testing.assert_array_equal(np.sort(evaluation_rows), np.arange(len(small_train.X)))


def test_selection_prioritizes_ap_over_auc():
    lower_ap = cv_result("XGB-01", .4, .99)
    higher_ap = cv_result("XGB-02", .41, .7)
    assert tuning.rank_candidates([lower_ap, higher_ap])[0] is higher_ap


def test_true_numerical_tie_uses_auc_then_id_deterministically():
    a = cv_result("XGB-01", .4, .8)
    b = cv_result("XGB-02", .4 + 5e-13, .9)
    c = cv_result("XGB-03", .4, .9)
    assert [r.candidate.configuration_id for r in tuning.rank_candidates([c, b, a])] == ["XGB-02", "XGB-03", "XGB-01"]
    assert tuning.rank_candidates([a, b, c])[0] is b
    # A small but real AP improvement must not be erased by a loose tolerance.
    d = cv_result("XGB-04", .4 + 1e-8, .1)
    assert tuning.rank_candidates([b, d])[0] is d


def test_selector_has_no_validation_or_test_inputs():
    assert [field.name for field in fields(tuning.TrainingData)] == ["X", "y", "groups"]
    assert [field.name for field in fields(tuning.CandidateCV)] == ["candidate", "cv", "runtime_seconds"]
    high_ap = cv_result("XGB-01", .5, .8)
    low_ap = cv_result("XGB-02", .4, .9)
    high_ap.cv["unused_validation_score"] = 0
    low_ap.cv["unused_validation_score"] = 1
    assert tuning.rank_candidates([low_ap, high_ap])[0] is high_ap


def test_both_selections_precede_any_validation_access(monkeypatch, small_train):
    selected = {a: cv_result("XGB-01" if a == "xgboost" else "LGBM-01", .4, .8) for a in tuning.ALGORITHMS}
    events = []

    def search(train, progress):
        assert isinstance(train, tuning.TrainingData)
        events.append("both selections frozen")
        return SimpleNamespace(selected=selected)

    class GuardedData:
        X_train, y_train, groups_train = small_train.X, small_train.y, small_train.groups

        @property
        def X_validation(self):
            assert events == ["both selections frozen"]
            raise RuntimeError("Validation boundary reached after selection")

    monkeypatch.setattr(tuning, "run_search", search)
    with pytest.raises(RuntimeError, match="after selection"):
        tuning.run_tuning(GuardedData())


def test_cv_baseline_mismatch_fails_loudly():
    result = cv_result("XGB-01", .4, .8)
    reference = {"summary": {"average_precision": {"mean": .3, "std": .01}, "roc_auc": {"mean": .8, "std": .01}}}
    with pytest.raises(ValueError, match="Baseline CV reproduction failed"):
        tuning.check_baseline_cv(result, reference)


@pytest.mark.parametrize("algorithm", tuning.ALGORITHMS)
def test_small_candidate_fit_probabilities_features_and_fold_statistics(algorithm, small_train):
    candidate = next(c for c in tuning.candidate_configurations() if c.algorithm == algorithm)
    fitting, evaluation = tuning.training_folds(small_train)[0]
    pipeline = tuning.fit_tree(tuning.build_candidate(candidate), small_train.X.iloc[fitting], small_train.y.iloc[fitting])
    model = pipeline.named_steps["model"]
    assert model.n_features_in_ == 13
    preprocessor = pipeline.named_steps["preprocessor"]
    assert preprocessor.get_feature_names_out().tolist() == OUTPUT_FEATURE_COLUMNS
    assert preprocessor.named_steps["imputer"].statistics_["MonthlyIncome"] == small_train.X.iloc[fitting].MonthlyIncome.median()
    with patch.object(preprocessor, "fit_transform", side_effect=AssertionError("Evaluation fit")):
        probabilities = tuning.positive_probabilities(pipeline, small_train.X.iloc[evaluation])
    assert probabilities.shape == (len(evaluation),)
    assert np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()
    audit = tuning.importance_audit(algorithm, pipeline, small_train.X.iloc[fitting])
    assert audit.shape == (13, 4) and np.isfinite(audit.importance).all()


def test_metrics_and_cv_table_schema_on_controlled_inputs():
    metrics = tuning.calculate_metrics(np.array([0, 0, 1, 1]), np.array([.1, .6, .4, .8]))
    assert metrics["roc_auc"] == .75 and metrics["average_precision"] == pytest.approx(5 / 6)
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    candidate = cv_result("XGB-01", .4, .8)
    search = tuning.SearchResult((candidate,), {"xgboost": candidate}, {"XGB-01": 1}, {"xgboost": 0.0}, 0.0, {"xgboost": "PASS"})
    table = tuning.cv_results_table(search)
    assert {"algorithm", "configuration_id", "parameters", "rank_within_algorithm",
            "cv_average_precision_mean", "cv_average_precision_std", "cv_roc_auc_mean", "cv_roc_auc_std"} <= set(table.columns)
    assert table.loc[0, "cv_average_precision_mean"] == .4
    assert table.loc[0, "rank_within_algorithm"] == 1


def test_loader_keeps_test_locked(monkeypatch):
    class Locked:
        train = SimpleNamespace(X="train X", y="train y", groups="raw groups")
        validation = SimpleNamespace(X="validation X", y="validation y")

        @property
        def test(self):
            raise AssertionError("Test is locked")

    monkeypatch.setattr(trees, "load_data", lambda: None)
    monkeypatch.setattr(trees, "split_data", lambda raw: Locked())
    data = tuning.load_tree_data()
    assert data.groups_train == "raw groups"
    assert not any("test" in field.name for field in fields(data))
