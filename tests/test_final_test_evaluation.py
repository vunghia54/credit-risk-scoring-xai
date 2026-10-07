"""Toy-data checks for the final evaluation; never evaluate the real holdout."""

import ast
from copy import deepcopy
import inspect
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from src import final_model as frozen
from src import final_test_evaluation as evaluation
from src.preprocessing import OUTPUT_FEATURE_COLUMNS


@pytest.fixture
def toy():
    return np.array([0, 0, 1, 1]), np.array([.1, .4, .35, .8])


def make_reference(labels, scores):
    metrics, _ = evaluation.evaluate_scores(labels, scores)
    points = [evaluation.fixed_threshold_metrics(labels, scores, t) for t in evaluation.THRESHOLDS]
    reference = frozen.get_frozen_metadata()
    reference['selection_evidence'] = {
        'ranking': {'XGBoost Baseline': {'validation': {'rows': len(labels), **metrics}}},
        'xgboost_calibration': [{'calibration_method': 'uncalibrated', **metrics}],
        'threshold_tradeoffs': points,
    }
    return reference


@pytest.fixture
def simulated(monkeypatch, toy):
    """Mock only fitting/prediction; preserve production control flow and metrics."""
    labels, scores = toy
    events = []
    train = SimpleNamespace(X=object(), y=labels)
    validation = SimpleNamespace(X=object(), y=labels)
    test = SimpleNamespace(X=object(), y=labels)

    class Splits:
        @property
        def test(self):
            events.append('access_test')
            return test
    splits = Splits()
    # Distinct frames make accidental fit/score of the wrong partition observable.
    train.X = pd.DataFrame({'toy': range(6)})
    validation.X = pd.DataFrame({'toy': range(4)})
    test.X = pd.DataFrame({'toy': range(4)})
    splits.train, splits.validation = train, validation
    pipeline = SimpleNamespace(named_steps={
        'model': SimpleNamespace(get_params=lambda: dict(frozen.MODEL_PARAMETERS), n_features_in_=13),
        'preprocessor': SimpleNamespace(get_feature_names_out=lambda: np.array(OUTPUT_FEATURE_COLUMNS)),
    })
    def build():
        events.append('build_frozen')
        return pipeline
    def fit(actual, X, y):
        assert actual is pipeline and X is train.X and y is train.y
        events.append('fit_train')
        return actual
    def predict(actual, X):
        assert actual is pipeline
        assert X is validation.X or X is test.X
        events.append('score_validation' if X is validation.X else 'score_test')
        return scores.copy()
    monkeypatch.setattr(frozen, 'build_final_model_pipeline', build)
    monkeypatch.setattr(evaluation, 'fit_tree', fit)
    monkeypatch.setattr(evaluation, 'positive_probabilities', predict)
    return splits, make_reference(labels, scores), events


def test_frozen_specification_and_exact_config():
    expected = dict(n_estimators=300, learning_rate=.05, max_depth=3, min_child_weight=1,
                    subsample=1., colsample_bytree=1., reg_alpha=0, reg_lambda=1,
                    objective='binary:logistic', eval_metric='logloss', random_state=42,
                    n_jobs=-1, tree_method='hist')
    reference = json.loads((evaluation.DEFAULT_REPORT_DIR / 'final_model_selection.json').read_text(encoding='utf-8'))
    metadata = evaluation.verify_frozen_specification(reference)
    assert metadata['model_parameters'] == expected
    assert metadata['calibration_method'] == 'none'
    assert evaluation.THRESHOLDS == (.50, .19)
    pipeline = frozen.build_final_model_pipeline()
    assert {k: pipeline.named_steps['model'].get_params()[k] for k in expected} == expected


@pytest.mark.parametrize('key,value', [('calibration_method', 'sigmoid'), ('default_audit_threshold', .49),
                                       ('development_operating_threshold', .20), ('final_model', 'LightGBM')])
def test_configuration_drift_rejected(key, value):
    reference = frozen.get_frozen_metadata()
    reference[key] = value
    with pytest.raises(ValueError, match='specification'):
        evaluation.verify_frozen_specification(reference)


@pytest.mark.parametrize('threshold', [.18, .20, .49, 0, 1, np.nan, np.inf])
def test_unapproved_thresholds_rejected(toy, threshold):
    with pytest.raises(ValueError, match='Only frozen'):
        evaluation.fixed_threshold_metrics(*toy, threshold)


def test_confusion_metrics_both_frozen_thresholds(toy):
    audit = evaluation.fixed_threshold_metrics(*toy, .5)
    assert audit['confusion_matrix'] == [[2, 0], [1, 1]]
    assert audit['precision'] == 1 and audit['recall'] == .5
    assert audit['f1'] == pytest.approx(2 / 3)
    assert audit['accuracy'] == .75 and audit['specificity'] == 1
    assert audit['false_positive_rate'] == 0 and audit['false_negative_rate'] == .5
    assert audit['predicted_positive_rate'] == .25
    operating = evaluation.fixed_threshold_metrics(*toy, .19)
    assert operating['confusion_matrix'] == [[1, 1], [0, 2]]
    assert operating['precision'] == pytest.approx(2 / 3) and operating['recall'] == 1
    assert operating['f1'] == .8 and operating['accuracy'] == .75
    assert operating['specificity'] == .5 and operating['false_positive_rate'] == .5
    assert operating['false_negative_rate'] == 0 and operating['predicted_positive_rate'] == .75


def test_threshold_boundary_is_inclusive():
    point = evaluation.fixed_threshold_metrics(np.array([0, 1]), np.array([.19, .19]), .19)
    assert point['confusion_matrix'] == [[0, 1], [0, 1]]


def test_probability_and_ranking_metrics(toy):
    y, scores = toy
    metrics, _ = evaluation.evaluate_scores(y, scores)
    assert metrics['roc_auc'] == roc_auc_score(y, scores) == .75
    assert metrics['average_precision'] == average_precision_score(y, scores)
    assert metrics['brier_score'] == pytest.approx((.01 + .16 + .4225 + .04) / 4)
    assert metrics['log_loss'] == pytest.approx(log_loss(y, scores))
    assert metrics['gini'] == .5 and metrics['prevalence'] == .5
    assert metrics['mean_predicted_probability'] == pytest.approx(.4125)
    assert metrics['mean_probability_gap'] == pytest.approx(.0875)


def test_ece_and_bin_schema_with_tied_scores():
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([.2, .2, .2, .8, .8, .8])
    metrics, bins = evaluation.evaluate_scores(y, scores)
    assert list(bins.columns) == ['bin_id', 'count', 'mean_predicted_probability', 'observed_positive_rate', 'absolute_gap']
    assert bins['count'].tolist() == [3, 3] and bins['bin_id'].tolist() == [1, 2]
    assert bins.observed_positive_rate.tolist() == [0., 1.]
    assert metrics['ece'] == pytest.approx(.2)
    assert bins['count'].sum() == len(y)


def test_ece_constant_score_and_endpoints():
    y = np.array([0, 1, 1, 1])
    metrics, bins = evaluation.evaluate_scores(y, np.full(4, .4))
    assert len(bins) == 1 and metrics['ece'] == pytest.approx(.35)
    metrics, _ = evaluation.evaluate_scores(np.array([0, 1]), np.array([0., 1.]))
    assert metrics['brier_score'] == 0 and np.isfinite(metrics['log_loss'])


@pytest.mark.parametrize('scores', [[-.1, .8], [.2, 1.1], [np.nan, .8], [np.inf, .8], [], [[.2, .8]]])
def test_invalid_probabilities_rejected(scores):
    with pytest.raises(ValueError):
        evaluation.evaluate_scores(np.array([0, 1]), np.array(scores))


def test_invalid_labels_rejected():
    for labels in ([0, 0], [0, 2], [0, 1, 1]):
        with pytest.raises(ValueError):
            evaluation.evaluate_scores(np.array(labels), np.array([.2, .8]))


def test_generalization_gap_sign_and_schema():
    assert evaluation.generalization_gaps({'auc': .8, 'loss': .2}, {'auc': .7, 'loss': .25}) == pytest.approx({'auc': -.1, 'loss': .05})
    with pytest.raises(ValueError, match='identical keys'):
        evaluation.generalization_gaps({'auc': .8}, {'loss': .2})


def test_training_only_and_validation_before_test(simulated):
    splits, reference, events = simulated
    result = evaluation.run_evaluation(splits, reference)
    assert events == ['build_frozen', 'fit_train', 'score_validation', 'access_test', 'score_test']
    assert result.report['validation_reproduction']['status'] == 'PASS'
    assert result.report['fit_partition'] == 'training only'
    assert result.report['train_rows'] == 6 and result.report['test_rows'] == 4


@pytest.mark.parametrize('drift', ['ranking', 'calibration', 'threshold', 'counts'])
def test_reproduction_failure_never_accesses_test(simulated, drift):
    splits, reference, events = simulated
    evidence = reference['selection_evidence']
    if drift == 'ranking':
        evidence['ranking']['XGBoost Baseline']['validation']['roc_auc'] += .01
    elif drift == 'calibration':
        evidence['xgboost_calibration'][0]['ece'] += .01
    elif drift == 'threshold':
        evidence['threshold_tradeoffs'][1]['recall'] -= .01
    else:
        evidence['threshold_tradeoffs'][0]['tp'] += 1
    with pytest.raises(ValueError, match='[Vv]alidation'):
        evaluation.run_evaluation(splits, reference)
    assert events == ['build_frozen', 'fit_train', 'score_validation']


def test_evaluation_does_not_mutate_configuration_or_reference(simulated):
    splits, reference, _ = simulated
    before, metadata = deepcopy(reference), frozen.get_frozen_metadata()
    evaluation.run_evaluation(splits, reference)
    assert reference == before and frozen.get_frozen_metadata() == metadata


def test_report_tables_and_json_schema(simulated):
    splits, reference, _ = simulated
    result = evaluation.run_evaluation(splits, reference)
    report = result.report
    assert set(report) >= {'frozen_model_specification', 'train_rows', 'validation_rows', 'test_rows',
                           'validation_reproduction', 'test_metrics', 'calibration_audit',
                           'test_threshold_metrics', 'generalization_gaps', 'no_retuning_statement',
                           'random_state', 'test_consumption_status'}
    json.dumps(report, allow_nan=False)
    metrics, points = evaluation.report_tables(result)
    assert metrics.dataset.tolist() == ['validation', 'test']
    assert set(metrics.columns) == {'dataset', 'roc_auc', 'average_precision', 'brier_score', 'log_loss',
                                   'gini', 'prevalence', 'mean_probability', 'ece', 'mean_probability_gap'}
    assert list(zip(points.dataset, points.threshold)) == [('validation', .5), ('test', .5), ('validation', .19), ('test', .19)]
    assert set(points.columns) == {'dataset', 'threshold', 'precision', 'recall', 'f1', 'accuracy', 'specificity',
                                  'false_positive_rate', 'false_negative_rate', 'predicted_positive_rate', 'tp', 'fp', 'tn', 'fn'}
    assert all(value == 0 for value in report['generalization_gaps']['metrics'].values())


def test_output_files_and_english_report_in_temporary_directory(simulated, tmp_path):
    splits, reference, _ = simulated
    result = evaluation.run_evaluation(splits, reference)
    evaluation.save_reports(result, tmp_path)
    files = {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob('*') if p.is_file()}
    assert files == set(evaluation.REPORT_FILES)
    for name in evaluation.REPORT_FILES:
        if name.endswith('.png'):
            assert (tmp_path / name).read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    report = (tmp_path / 'final_test_evaluation.md').read_text(encoding='utf-8')
    for heading in ['Frozen Evaluation Protocol', 'Final Model', 'Test Dataset', 'Ranking Performance',
                    'Probability Calibration Audit', 'Threshold 0.50 Results',
                    'Development Operating Threshold 0.19 Results', 'Validation vs Test Generalization',
                    'Error Trade-offs', 'No-Retuning Policy', 'Limitations', 'Final Internal Evaluation Summary']:
        assert f'## {heading}' in report
    assert evaluation.CONSUMPTION_STATEMENT in report
    assert 'not external or temporal validation' in report and 'not regulatory or production-validated PD' in report
    assert len(pd.read_csv(tmp_path / 'final_test_calibration_bins.csv')) == len(result.test_bins)
    with pytest.raises(FileExistsError):
        evaluation.save_reports(result, tmp_path)


def test_existing_output_blocks_cli_before_data_access(monkeypatch, tmp_path):
    (tmp_path / 'final_test_results.json').write_text('{}', encoding='utf-8')
    monkeypatch.setattr(evaluation, 'DEFAULT_REPORT_DIR', tmp_path)
    def forbidden():
        pytest.fail('No data access allowed after evaluation is already recorded')
    monkeypatch.setattr(evaluation, 'load_data', forbidden)
    with pytest.raises(FileExistsError):
        evaluation.main()


def test_no_search_selection_serialization_or_import_time_evaluation():
    tree = ast.parse(inspect.getsource(evaluation))
    forbidden = {'select_scenario', 'analyze_thresholds', 'threshold_grid', 'build_candidates', 'run_search',
                 'GridSearchCV', 'RandomizedSearchCV', 'build_calibrator', 'fit_calibrator', 'rank_candidates',
                 'CalibratedClassifierCV', 'XGBClassifier', 'LGBMClassifier', 'argmax', 'argmin'}
    calls = {getattr(n.func, 'id', getattr(n.func, 'attr', '')) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert not calls.intersection(forbidden)
    imports = {alias.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for alias in n.names}
    assert not imports.intersection({'pickle', 'joblib', 'optuna'})
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.Expr)):
            names = {getattr(n.func, 'id', getattr(n.func, 'attr', '')) for n in ast.walk(node) if isinstance(n, ast.Call)}
            assert not names.intersection({'fit', 'load_data', 'split_data', 'run_evaluation', 'main'})
