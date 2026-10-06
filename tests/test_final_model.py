"""Frozen specification and one shared full-training reproduction fixture."""

import ast
from dataclasses import fields
import inspect
import json
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from xgboost import XGBClassifier

import src.baseline_model as baseline
import src.final_model as final
from src.threshold_analysis import build_candidates
from src.tree_model_experiments import TREE_CONFIGURATIONS


EXPECTED_PARAMETERS = dict(
    n_estimators=300, learning_rate=.05, max_depth=3, min_child_weight=1,
    subsample=1., colsample_bytree=1., reg_alpha=0, reg_lambda=1,
    objective='binary:logistic', eval_metric='logloss', random_state=42,
    n_jobs=-1, tree_method='hist',
)


def test_frozen_identity_calibration_and_thresholds():
    assert final.FINAL_MODEL_NAME == 'XGBoost Baseline / XGB-01'
    assert final.RANDOM_STATE == 42 and final.CALIBRATION_METHOD == 'none'
    assert final.DEFAULT_AUDIT_THRESHOLD == .50
    assert final.DEVELOPMENT_OPERATING_THRESHOLD == .19
    assert 0 <= final.DEFAULT_AUDIT_THRESHOLD <= 1
    assert 0 <= final.DEVELOPMENT_OPERATING_THRESHOLD <= 1


def test_exact_pipeline_preprocessor_and_no_scaler():
    pipeline = final.build_final_model_pipeline()
    assert type(pipeline) is Pipeline
    assert list(pipeline.named_steps) == ['preprocessor', 'model']
    assert type(pipeline.named_steps['model']) is XGBClassifier
    assert list(pipeline.named_steps['preprocessor'].named_steps) == ['quality', 'imputer']
    assert not any(isinstance(v, StandardScaler) for v in pipeline.get_params(deep=True).values())
    with pytest.raises(NotFittedError):
        check_is_fitted(pipeline.named_steps['model'])


def test_all_frozen_parameters_match_step10_and_step12():
    pipeline = final.build_final_model_pipeline()
    actual = pipeline.named_steps['model'].get_params()
    assert dict(final.MODEL_PARAMETERS) == EXPECTED_PARAMETERS
    assert {k: actual[k] for k in EXPECTED_PARAMETERS} == EXPECTED_PARAMETERS
    assert {k: actual[k] for k in TREE_CONFIGURATIONS['xgboost']} == TREE_CONFIGURATIONS['xgboost']
    approved = build_candidates()['XGBoost Baseline'].named_steps['model'].get_params()
    assert {k: approved[k] for k in EXPECTED_PARAMETERS} == EXPECTED_PARAMETERS
    assert actual.get('scale_pos_weight') is None
    assert actual.get('early_stopping_rounds') is None


def test_metadata_and_fresh_pipelines_cannot_mutate_frozen_defaults():
    with pytest.raises(TypeError):
        final.MODEL_PARAMETERS['max_depth'] = 99
    metadata = final.get_frozen_metadata()
    metadata['model_parameters']['max_depth'] = 99
    metadata['final_test_metrics_to_report']['classification_thresholds'].append(.3)
    pipeline = final.build_final_model_pipeline().set_params(model__max_depth=99)
    fresh = final.build_final_model_pipeline()
    assert fresh is not pipeline
    assert fresh.named_steps['model'].get_params()['max_depth'] == 3
    assert final.get_frozen_metadata()['model_parameters'] == EXPECTED_PARAMETERS
    assert final.get_frozen_metadata()['final_test_metrics_to_report']['classification_thresholds'] == [.5, .19]


@pytest.fixture(scope='module')
def artifact():
    return final.build_selection_artifact()


def test_selection_json_schema_and_source_evidence(artifact):
    required = {'final_model', 'preprocessing', 'model_parameters', 'calibration_method',
                'selection_evidence', 'default_audit_threshold', 'development_operating_threshold',
                'development_threshold_selection_rule', 'final_test_metrics_to_report', 'test_lock_policy', 'random_state'}
    assert required <= set(artifact)
    assert artifact['model_parameters'] == EXPECTED_PARAMETERS
    assert artifact['calibration_method'] == 'none'
    evidence = artifact['selection_evidence']
    assert evidence['ranking']['XGBoost Baseline']['train_cv']['average_precision']['mean'] == pytest.approx(.39887984341049254)
    assert evidence['ranking']['LightGBM LGBM-02']['validation']['average_precision'] == pytest.approx(.38168512498191703)
    assert len(artifact['source_artifacts_sha256']) == 5
    assert all(len(value) == 64 for value in artifact['source_artifacts_sha256'].values())
    json.dumps(artifact, allow_nan=False)


def test_future_policy_is_frozen_but_contains_no_observed_test_results(artifact):
    policy = artifact['final_test_metrics_to_report']
    assert policy['threshold_independent'] == ['roc_auc', 'average_precision', 'brier_score', 'log_loss', 'gini']
    assert policy['classification_thresholds'] == [.5, .19]
    assert policy['reliability_definition']['n_bins'] == 10
    assert policy['reliability_definition']['strategy'] == 'quantile'
    assert 'accuracy' in policy['classification_metrics']
    assert len(artifact['test_lock_policy']['forbidden_after_test']) == 8
    assert 'test_results' not in artifact and 'test_metrics' not in artifact
    assert 'test' not in artifact['selection_evidence']


@pytest.fixture(scope='module')
def verification(artifact):
    # Exactly one full real training fit is shared by the reproduction tests.
    return final.verify_validation(final.load_training_validation(), artifact)


def test_validation_ranking_reference_reproduction(verification):
    assert verification['status'] == 'PASS'
    assert verification['fit_rows'] == 104998 and verification['validation_rows'] == 22486
    assert verification['roc_auc'] == pytest.approx(.863666393084869, rel=1e-6, abs=1e-8)
    assert verification['average_precision'] == pytest.approx(.38128449036768636, rel=1e-6, abs=1e-8)
    assert verification['number_of_transformed_features'] == 13


@pytest.mark.parametrize('threshold,precision,recall,f1', [
    (.50, .5688259109311741, .19012178619756429, .28498985801217036),
    (.19, .37230306071249375, .5020297699594046, .42754249495822527),
])
def test_two_frozen_threshold_reference_metrics(verification, threshold, precision, recall, f1):
    point = next(p for p in verification['threshold_checks'] if p['threshold'] == threshold)
    assert point['precision'] == pytest.approx(precision, rel=1e-6, abs=1e-8)
    assert point['recall'] == pytest.approx(recall, rel=1e-6, abs=1e-8)
    assert point['f1'] == pytest.approx(f1, rel=1e-6, abs=1e-8)
    assert sum(point[k] for k in ['tp', 'fp', 'tn', 'fn']) == 22486


def test_small_fit_probability_contract():
    raw = baseline.load_data().head(500)
    from src.data_split import FEATURE_COLUMNS, TARGET_COLUMN
    from src.preprocessing import OUTPUT_FEATURE_COLUMNS
    pipeline = final.fit_tree(final.build_final_model_pipeline(), raw[FEATURE_COLUMNS].iloc[:400], raw[TARGET_COLUMN].iloc[:400])
    assert pipeline.named_steps['model'].n_features_in_ == 13
    assert pipeline.named_steps['preprocessor'].get_feature_names_out().tolist() == OUTPUT_FEATURE_COLUMNS
    scores = final.positive_probabilities(pipeline, raw[FEATURE_COLUMNS].iloc[400:])
    assert scores.shape == (100,)
    assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()


def test_loader_does_not_access_test_target(monkeypatch):
    class Locked:
        train = SimpleNamespace(X='train X', y='train y')
        validation = SimpleNamespace(X='validation X', y='validation y')

        @property
        def test(self):
            raise AssertionError('Test remains locked')
    monkeypatch.setattr(baseline, 'load_data', lambda: None)
    monkeypatch.setattr(baseline, 'split_data', lambda _: Locked())
    data = final.load_training_validation()
    assert data.y_train == 'train y' and data.y_validation == 'validation y'
    assert [f.name for f in fields(data)] == ['X_train', 'y_train', 'X_validation', 'y_validation']


def test_no_automatic_selection_threshold_search_or_import_time_fit():
    parsed = ast.parse(inspect.getsource(final))
    forbidden = {'rank_candidates', 'run_search', 'select_scenario', 'analyze_thresholds', 'threshold_grid', 'fit_calibrator'}
    called = {n.func.id for n in ast.walk(parsed) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert not (called & forbidden)
    for node in parsed.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.Expr)):
            direct_calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)]
            assert all(not isinstance(n.func, ast.Attribute) or n.func.attr != 'fit' for n in direct_calls)
            assert all(not isinstance(n.func, ast.Name) or n.func.id not in {'load_data', 'load_training_validation', 'verify_validation', 'build_final_model_pipeline'} for n in direct_calls)


def test_english_selection_report_required_sections(artifact):
    report = final.render_selection_report(artifact)
    for heading in ['Selection Scope', 'Candidate Models', 'Ranking Evidence', 'Threshold Trade-offs',
                    'Calibration Evidence', 'Selected Model', 'Selected Calibration Strategy',
                    'Frozen Development Threshold', 'Final Test Evaluation Plan', 'Test Lock Policy',
                    'Limitations', 'What This Model Is Not']:
        assert f'## {heading}' in report
    assert '0.19' in report and '0.50' in report
