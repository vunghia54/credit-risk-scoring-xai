"""Calibration contracts on controlled or small data; no full nine-model runs."""

from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

import src.calibration_analysis as analysis
import src.tree_model_experiments as trees
from src.data_split import FEATURE_COLUMNS, TARGET_COLUMN, build_feature_groups, load_data
from src.preprocessing import OUTPUT_FEATURE_COLUMNS


@pytest.fixture(scope="module")
def small_train():
    raw = load_data().head(600)
    return analysis.TrainingData(raw[FEATURE_COLUMNS], raw[TARGET_COLUMN], build_feature_groups(raw))


def test_exact_candidates_and_calibration_states():
    models = analysis.build_candidates()
    assert len(models) == 3 and tuple(models) == analysis.MODEL_NAMES
    assert analysis.CALIBRATION_STATES == ('uncalibrated', 'sigmoid', 'isotonic')
    expected = [
        dict(C=1., class_weight='balanced', solver='lbfgs', max_iter=2000, l1_ratio=0., random_state=42),
        dict(n_estimators=300, learning_rate=.05, max_depth=3, min_child_weight=1, subsample=1.,
             colsample_bytree=1., reg_alpha=0, reg_lambda=1, objective='binary:logistic',
             eval_metric='logloss', tree_method='hist', random_state=42, n_jobs=-1),
        dict(n_estimators=300, learning_rate=.05, num_leaves=15, max_depth=-1, min_child_samples=20,
             subsample=1., subsample_freq=0, colsample_bytree=1., reg_alpha=0, reg_lambda=0,
             random_state=42, n_jobs=-1, verbosity=-1),
    ]
    for pipeline, reference in zip(models.values(), expected):
        params = pipeline.named_steps['model'].get_params()
        assert {k: params[k] for k in reference} == reference


def test_group_folds_determinism_isolation_and_coverage(small_train):
    first = analysis.calibration_folds(small_train)
    second = analysis.calibration_folds(small_train)
    assert len(first) == 5
    for (fitting, calibration), (fit2, cal2) in zip(first, second):
        np.testing.assert_array_equal(fitting, fit2)
        np.testing.assert_array_equal(calibration, cal2)
        assert set(fitting).isdisjoint(calibration)
        assert set(small_train.groups.iloc[fitting]).isdisjoint(small_train.groups.iloc[calibration])
    np.testing.assert_array_equal(np.sort(np.concatenate([c for _, c in first])), np.arange(600))
    assert all(row['group_overlap'] == row['row_overlap'] == 0 for row in analysis.fold_audit(small_train, first))


def test_calibrator_explicit_api_and_no_prefit(small_train):
    folds = analysis.calibration_folds(small_train)
    pipeline = next(iter(analysis.build_candidates().values()))
    for method in ('sigmoid', 'isotonic'):
        model = analysis.build_calibrator(pipeline, method, folds)
        assert model.cv is folds and model.estimator is pipeline
        assert model.ensemble is True and model.n_jobs == 1 and model.method == method
    with pytest.raises(ValueError):
        analysis.build_calibrator(pipeline, 'temperature', folds)


@pytest.mark.parametrize('method', ['sigmoid', 'isotonic'])
def test_actual_calibration_fold_local_fit_and_valid_probabilities(method, small_train, monkeypatch):
    folds = analysis.calibration_folds(small_train)
    original_fit = Pipeline.fit
    fit_rows = []
    def observed_fit(self, X, y=None, **kwargs):
        if list(self.named_steps) == ['preprocessor', 'model']:
            fit_rows.append(X.index.to_numpy())
        return original_fit(self, X, y, **kwargs)
    monkeypatch.setattr(Pipeline, 'fit', observed_fit)
    pipeline = analysis.build_candidates()[analysis.MODEL_NAMES[0]]
    fitted = analysis.fit_calibrator(pipeline, method, small_train, folds)
    assert len(fit_rows) == 5
    for pair, indices, (fitting, calibration) in zip(fitted.calibrated_classifiers_, fit_rows, folds):
        np.testing.assert_array_equal(indices, small_train.X.iloc[fitting].index)
        assert set(indices).isdisjoint(small_train.X.iloc[calibration].index)
        preprocessor = pair.estimator.named_steps['preprocessor']
        assert preprocessor.get_feature_names_out().tolist() == OUTPUT_FEATURE_COLUMNS
        assert preprocessor.named_steps['imputer'].statistics_['MonthlyIncome'] == small_train.X.iloc[fitting].MonthlyIncome.median()
    probabilities = analysis.positive_probabilities(fitted, small_train.X.iloc[:25])
    assert probabilities.shape == (25,) and np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_small_sigmoid_fit_is_deterministic(small_train):
    folds = analysis.calibration_folds(small_train)
    outputs = []
    for _ in range(2):
        pipeline = analysis.build_candidates()[analysis.MODEL_NAMES[0]]
        fitted = analysis.fit_calibrator(pipeline, 'sigmoid', small_train, folds)
        outputs.append(analysis.positive_probabilities(fitted, small_train.X.iloc[:20]))
    np.testing.assert_allclose(outputs[0], outputs[1], rtol=0, atol=1e-12)


def test_brier_log_loss_and_ranking_on_controlled_inputs():
    labels = np.array([0, 0, 1, 1])
    values = np.array([.1, .4, .6, .9])
    metrics, bins = analysis.calibration_metrics(labels, values)
    assert metrics['brier_score'] == pytest.approx(.085)
    assert metrics['log_loss'] == pytest.approx(-np.log([.9, .6, .6, .9]).mean())
    assert metrics['roc_auc'] == metrics['average_precision'] == 1
    assert metrics['mean_predicted_probability'] == metrics['prevalence'] == .5
    assert metrics['mean_probability_gap'] == 0
    assert metrics['ece'] == pytest.approx(analysis.expected_calibration_error(bins))


def test_ece_count_weighted_formula():
    table = pd.DataFrame({'count': [1, 3], 'absolute_gap': [.2, .4]})
    assert analysis.expected_calibration_error(table) == pytest.approx(.35)
    bins = analysis.reliability_bins(np.array([0, 0, 1, 1]), np.array([.1, .4, .6, .9]), n_bins=2)
    assert bins['count'].tolist() == [2, 2]
    assert bins.mean_predicted_probability.tolist() == [.25, .75]
    assert bins.observed_positive_rate.tolist() == [0., 1.]
    assert analysis.expected_calibration_error(bins) == .25


def test_quantile_bins_schema_coverage_and_duplicate_collapse():
    labels = np.array([0, 1] * 10)
    values = np.array([.1] * 10 + [.9] * 10)
    bins = analysis.reliability_bins(labels, values)
    assert len(bins) == 2 and bins['count'].sum() == 20
    assert bins.bin_id.tolist() == [1, 2]
    assert set(bins) == {'bin_id', 'count', 'mean_predicted_probability', 'observed_positive_rate', 'absolute_gap'}
    constant = analysis.reliability_bins(labels, np.full(20, .2))
    assert len(constant) == 1 and constant['count'].iloc[0] == 20
    assert analysis.expected_calibration_error(constant) == pytest.approx(.3)


def test_quantile_bins_ten_distinct_and_endpoints():
    bins = analysis.reliability_bins(np.tile([0, 1], 50), np.linspace(0, 1, 100))
    assert len(bins) == 10 and bins['count'].tolist() == [10] * 10
    metrics, _ = analysis.calibration_metrics(np.array([0, 1]), np.array([0., 1.]))
    assert metrics['brier_score'] == 0 and np.isfinite(metrics['log_loss'])


@pytest.mark.parametrize('scores', [[np.nan, .2], [np.inf, .2], [-.1, .5], [.1, 1.1], [], [[.1], [.8]]])
def test_invalid_probabilities_rejected(scores):
    with pytest.raises(ValueError):
        analysis.calibration_metrics(np.array([0, 1]), np.array(scores))


@pytest.mark.parametrize('classes,probabilities', [
    ([1, 0], [[.2, .8]]), ([0, 1], [[.2, .8], [.4, .6]]),
    ([0, 1], [[np.nan, .8]]), ([0, 1], [[-.1, 1.1]]), ([0, 1], [[.2, .3]]),
])
def test_classifier_probability_contract(classes, probabilities):
    model = SimpleNamespace(classes_=np.array(classes), predict_proba=lambda X: np.array(probabilities))
    with pytest.raises(ValueError):
        analysis.positive_probabilities(model, pd.DataFrame({'x': [1]}))


@pytest.mark.parametrize('metric', ['roc_auc', 'average_precision'])
def test_reference_reproduction_and_failure(metric):
    values = dict(roc_auc=.8, average_precision=.4)
    analysis.check_reference(values, values)
    changed = {**values, metric: values[metric] + .01}
    with pytest.raises(ValueError, match='reference reproduction failed'):
        analysis.check_reference(changed, values)


def test_loader_test_target_inaccessible(monkeypatch):
    class Locked:
        train = SimpleNamespace(X='train X', y='train y', groups='raw train groups')
        validation = SimpleNamespace(X='validation X', y='validation y')

        @property
        def test(self):
            raise AssertionError('Test stays locked')
    monkeypatch.setattr(trees, 'load_data', lambda: None)
    monkeypatch.setattr(trees, 'split_data', lambda _: Locked())
    data = analysis.load_tree_data()
    assert data.groups_train == 'raw train groups'
    assert not any('test' in f.name for f in fields(data))
    assert [f.name for f in fields(analysis.TrainingData)] == ['X', 'y', 'groups']


def mock_run(monkeypatch, drift=False):
    """Spy on full orchestration without any full real model fitting."""
    X = pd.DataFrame({'value': np.arange(20)})
    y = pd.Series([0, 1] * 10)
    groups = pd.Series(np.arange(20))
    V = pd.DataFrame({'value': [20, 21, 22, 23]}, index=[20, 21, 22, 23])
    labels = pd.Series([0, 0, 1, 1], index=V.index)
    data = analysis.ExperimentData(X, y, groups, V, labels)
    probabilities = np.array([.1, .4, .6, .9])
    metrics, _ = analysis.calibration_metrics(labels, probabilities)
    if drift:
        metrics['roc_auc'] = .5
    monkeypatch.setattr(analysis, 'load_references', lambda: dict.fromkeys(analysis.MODEL_NAMES, metrics))
    events = []
    def fit(pipeline, fit_X, fit_y):
        assert fit_X is X and fit_y is y
        events.append('base fit')
        return pipeline
    def calibrate(pipeline, method, train, folds):
        assert train.X is X and train.y is y and train.groups is groups
        assert len(folds) == 5
        assert events.count('base fit') == 3
        events.append(method)
        return pipeline
    def predict(model, eval_X):
        assert eval_X is V
        events.append('validation predict')
        return probabilities
    monkeypatch.setattr(analysis, 'fit_checked', fit)
    monkeypatch.setattr(analysis, 'fit_tree', fit)
    monkeypatch.setattr(analysis, 'fit_calibrator', calibrate)
    monkeypatch.setattr(analysis, 'positive_probabilities', predict)
    return data, events


def test_validation_excluded_from_calibration_and_nine_combination_schema(monkeypatch):
    import json
    data, events = mock_run(monkeypatch)
    result = analysis.run_analysis(data)
    assert events.count('base fit') == 3
    assert events.count('sigmoid') == events.count('isotonic') == 3
    assert events.count('validation predict') == 9
    assert len(result.comparison) == 9
    assert set(zip(result.comparison.model, result.comparison.calibration_method)) == {
        (m, s) for m in analysis.MODEL_NAMES for s in analysis.CALIBRATION_STATES}
    assert set(result.comparison) == {'model', 'calibration_method', 'brier_score', 'log_loss', 'ece',
                                      'mean_predicted_probability', 'prevalence', 'mean_probability_gap', 'roc_auc', 'average_precision'}
    assert not any('threshold' in c or 'test' in c for c in result.comparison)
    assert len(result.report['group_leakage_checks']) == 5
    json.dumps(result.report, allow_nan=False)


def test_reference_failure_stops_before_any_calibration(monkeypatch):
    data, events = mock_run(monkeypatch, drift=True)
    with pytest.raises(ValueError, match='reference reproduction failed'):
        analysis.run_analysis(data)
    assert 'sigmoid' not in events and 'isotonic' not in events


def test_no_threshold_selection_calls():
    import ast
    import inspect
    source = ast.parse(inspect.getsource(analysis))
    forbidden = {'select_scenario', 'analyze_thresholds', 'threshold_metrics', 'threshold_grid'}
    called = {node.func.id for node in ast.walk(source) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert not (called & forbidden)
