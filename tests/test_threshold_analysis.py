"""Small-input contracts; full real candidate fits run only via the module CLI."""

from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

import src.baseline_model as baseline
import src.threshold_analysis as analysis
from src.boosting_tuning import FIXED_PARAMETERS, candidate_configurations


def scenario_table(rows):
    """Controlled selector scores isolate ordering from model fitting."""
    default = analysis.threshold_metrics(np.array([0, 1]), np.array([.2, .8]), .5)
    return pd.DataFrame([{**default, **row} for row in rows])


def test_exact_three_candidates_and_approved_configurations():
    candidates = analysis.build_candidates()
    assert tuple(candidates) == analysis.MODEL_NAMES and len(candidates) == 3
    logistic = candidates[analysis.MODEL_NAMES[0]]
    params = logistic.named_steps['model'].get_params()
    assert {k: params[k] for k in ['C', 'class_weight', 'solver', 'max_iter', 'random_state', 'l1_ratio']} == {
        'C': 1., 'class_weight': 'balanced', 'solver': 'lbfgs', 'max_iter': 2000, 'random_state': 42, 'l1_ratio': 0.}
    assert any(isinstance(v, StandardScaler) for v in logistic.get_params(deep=True).values())
    configs = {c.configuration_id: c for c in candidate_configurations()}
    for name, config_id in zip(analysis.MODEL_NAMES[1:], ['XGB-01', 'LGBM-02']):
        pipeline = candidates[name]
        config = configs[config_id]
        params = pipeline.named_steps['model'].get_params()
        for key, expected in {**config.parameters, **FIXED_PARAMETERS[config.algorithm]}.items():
            assert params[key] == expected
        assert list(pipeline.named_steps['preprocessor'].named_steps) == ['quality', 'imputer']
        assert not any(isinstance(v, StandardScaler) for v in pipeline.get_params(deep=True).values())
        assert params.get('early_stopping_rounds') is None
        assert params.get('class_weight') is None and params.get('scale_pos_weight') is None


def test_grid_determinism_count_and_single_default():
    grid = analysis.threshold_grid()
    np.testing.assert_array_equal(grid, analysis.threshold_grid())
    assert len(grid) == 91 and len(set(grid)) == 91
    assert grid[0] == .05 and grid[-1] == .95
    assert np.count_nonzero(grid == .5) == 1
    np.testing.assert_allclose(np.diff(grid), .01)


def test_confusion_derived_metrics_and_threshold_boundary():
    values = analysis.threshold_metrics(np.array([0, 0, 0, 1, 1]), np.array([.1, .2, .5, .4, .8]), .5)
    assert [values[k] for k in ['tn', 'fp', 'fn', 'tp']] == [2, 1, 1, 1]
    for key in ['precision', 'recall', 'f1', 'false_negative_rate']:
        assert values[key] == .5
    assert values['specificity'] == pytest.approx(2 / 3)
    assert values['false_positive_rate'] == pytest.approx(1 / 3)
    assert values['predicted_positive_rate'] == .4
    assert values['balanced_accuracy'] == pytest.approx((.5 + 2 / 3) / 2)


def test_no_positive_predictions_have_zero_precision_f1():
    result = analysis.threshold_metrics(np.array([0, 1]), np.array([.1, .2]), .5)
    assert result['precision'] == result['recall'] == result['f1'] == 0
    assert result['specificity'] == result['false_negative_rate'] == 1


@pytest.mark.parametrize('scores', [[np.nan, .5], [np.inf, .5], [-.1, .5], [1.1, .5], [[.1], [.5]], []])
def test_invalid_probabilities_rejected(scores):
    with pytest.raises(ValueError):
        analysis.threshold_metrics(np.array([0, 1]), np.array(scores), .5)


@pytest.mark.parametrize('threshold', [-.1, 1.1, np.nan])
def test_invalid_threshold_rejected(threshold):
    with pytest.raises(ValueError):
        analysis.threshold_metrics(np.array([0, 1]), np.array([.2, .8]), threshold)


def test_max_f1_uses_all_tie_breaks_deterministically():
    table = scenario_table([
        dict(threshold=.1, f1=.5, recall=.4, precision=.9),
        dict(threshold=.2, f1=.5, recall=.6, precision=.4),
        dict(threshold=.3, f1=.5, recall=.6, precision=.5),
        dict(threshold=.4, f1=.5, recall=.6, precision=.5),
        dict(threshold=.5, f1=.4, recall=.9, precision=.9),
    ])
    assert analysis.select_scenario(table, 'MAX_F1')['threshold'] == .3
    assert analysis.select_scenario(table.iloc[::-1], 'MAX_F1')['threshold'] == .3
    assert analysis.select_scenario(table, 'DEFAULT')['threshold'] == .5


@pytest.mark.parametrize('scenario,minimum', [('RECALL_AT_LEAST_50', .5), ('RECALL_AT_LEAST_70', .7)])
def test_recall_constraint_precision_f1_then_higher_threshold(scenario, minimum):
    table = scenario_table([
        dict(threshold=.1, recall=minimum - .01, precision=.9, f1=.8),
        dict(threshold=.2, recall=minimum, precision=.4, f1=.5),
        dict(threshold=.3, recall=minimum, precision=.4, f1=.6),
        dict(threshold=.4, recall=minimum, precision=.4, f1=.6),
        dict(threshold=.5, recall=minimum, precision=.3, f1=.8),
    ])
    assert analysis.select_scenario(table, scenario)['threshold'] == .4
    assert analysis.select_scenario(table.iloc[::-1], scenario)['threshold'] == .4


def test_precision_constraint_recall_f1_then_lower_threshold():
    table = scenario_table([
        dict(threshold=.1, precision=.29, recall=.99, f1=.9),
        dict(threshold=.2, precision=.3, recall=.7, f1=.5),
        dict(threshold=.3, precision=.3, recall=.7, f1=.6),
        dict(threshold=.4, precision=.3, recall=.7, f1=.6),
        dict(threshold=.5, precision=.9, recall=.6, f1=.9),
    ])
    assert analysis.select_scenario(table, 'PRECISION_AT_LEAST_30')['threshold'] == .3
    assert analysis.select_scenario(table.iloc[::-1], 'PRECISION_AT_LEAST_30')['threshold'] == .3


@pytest.mark.parametrize('scenario', ['RECALL_AT_LEAST_50', 'RECALL_AT_LEAST_70', 'PRECISION_AT_LEAST_30'])
def test_unavailable_scenario_has_nulls_and_reason(scenario):
    result = analysis.select_scenario(scenario_table([dict(threshold=.5, recall=.1, precision=.1)]), scenario)
    assert result['status'] == 'unavailable' and result['reason']
    assert all(result[key] is None for key in analysis.METRIC_COLUMNS)


def test_score_distribution_schema_values_and_class_separation():
    distribution = analysis.score_distribution(np.array([0, 0, 1, 1]), np.array([.1, .2, .7, .9]))
    assert set(distribution) == {'all', 'target_0', 'target_1'}
    assert set(distribution['all']) == {'min', 'p01', 'p05', 'p25', 'median', 'p75', 'p95', 'p99', 'max', 'mean', 'count'}
    assert distribution['all']['median'] == pytest.approx(.45)
    assert distribution['target_0']['max'] == .2 and distribution['target_1']['min'] == .7
    assert distribution['all']['count'] == 4


def test_threshold_table_schema_and_monotonic_recall():
    table = analysis.analyze_thresholds(np.array([0, 0, 1, 1]), np.array([.1, .3, .5, .9]))
    assert table.shape == (91, len(analysis.METRIC_COLUMNS))
    assert tuple(table.columns) == analysis.METRIC_COLUMNS
    assert table.recall.is_monotonic_decreasing
    assert table.predicted_positive_rate.is_monotonic_decreasing


def test_loader_exposes_train_validation_only(monkeypatch):
    class Locked:
        train = SimpleNamespace(X='train X', y='train y')
        validation = SimpleNamespace(X='validation X', y='validation y')

        @property
        def test(self):
            raise AssertionError('Test remains locked')

    monkeypatch.setattr(baseline, 'load_data', lambda: None)
    monkeypatch.setattr(baseline, 'split_data', lambda _: Locked())
    data = analysis.load_training_validation()
    assert data.y_validation == 'validation y'
    assert [f.name for f in fields(data)] == ['X_train', 'y_train', 'X_validation', 'y_validation']


def test_reference_drift_stops_before_analysis(monkeypatch):
    X = pd.DataFrame({'example': [1, 2]})
    y = pd.Series([0, 1])
    data = analysis.TrainingValidationData(X, y, X.copy(), y.copy())
    reference = baseline.calculate_metrics(y, np.array([.1, .9]))
    reference['roc_auc'] = .5
    monkeypatch.setattr(analysis, 'load_references', lambda: dict.fromkeys(analysis.MODEL_NAMES, reference))
    monkeypatch.setattr(analysis, 'fit_checked', lambda pipeline, X, y: pipeline)
    monkeypatch.setattr(analysis, 'positive_probabilities', lambda pipeline, X: np.array([.1, .9]))
    def forbidden(*args):
        raise AssertionError('Analysis must not run on reference mismatch')
    monkeypatch.setattr(analysis, 'analyze_thresholds', forbidden)
    with pytest.raises(ValueError, match='Reference reproduction failed: roc_auc'):
        analysis.run_analysis(data)


def test_run_contract_fits_train_and_predicts_validation_once(monkeypatch):
    import json
    X = pd.DataFrame({'example': [1, 2]})
    V = pd.DataFrame({'example': [3, 4]})
    y, labels = pd.Series([0, 1]), pd.Series([0, 1])
    data = analysis.TrainingValidationData(X, y, V, labels)
    probs = np.array([.1, .9])
    reference = baseline.calculate_metrics(labels, probs)
    events = []
    def fit(pipeline, fit_X, fit_y):
        assert fit_X is X and fit_y is y
        events.append('fit')
        return pipeline
    def predict(pipeline, eval_X):
        assert eval_X is V
        events.append('predict')
        return probs
    monkeypatch.setattr(analysis, 'load_references', lambda: dict.fromkeys(analysis.MODEL_NAMES, reference))
    monkeypatch.setattr(analysis, 'fit_checked', fit)
    monkeypatch.setattr(analysis, 'fit_tree', fit)
    monkeypatch.setattr(analysis, 'positive_probabilities', predict)
    result = analysis.run_analysis(data)
    assert events == ['fit', 'predict'] * 3
    assert len(result.thresholds) == 273 and len(result.scenarios) == 15
    assert set(result.report['candidate_comparison_at_0_5']) == set(analysis.MODEL_NAMES)
    assert all(v == 'PASS' for v in result.report['reference_reproduction'].values())
    json.dumps(result.report, allow_nan=False)
