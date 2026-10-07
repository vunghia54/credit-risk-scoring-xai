"""Small deterministic SHAP checks; no full-validation or test-set explanations."""

import ast
from copy import deepcopy
from dataclasses import fields
import inspect
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import shap

from src import final_model as frozen
from src import shap_global as global_shap
from src.data_split import FEATURE_COLUMNS
from src.preprocessing import OUTPUT_FEATURE_COLUMNS


@pytest.fixture(scope='module')
def small_fit():
    """Fit the exact frozen config once on synthetic unit-test inputs only."""
    rng = np.random.default_rng(42)
    raw = pd.DataFrame(rng.uniform(.1, 4, (180, 10)), columns=FEATURE_COLUMNS)
    raw['age'] = rng.integers(20, 80, len(raw)).astype(float)
    raw.loc[::11, 'MonthlyIncome'] = np.nan
    raw.loc[::17, 'NumberOfDependents'] = np.nan
    raw.loc[::19, 'NumberOfTimes90DaysLate'] = 96
    raw.loc[::23, 'age'] = 0
    y = pd.Series((raw['RevolvingUtilizationOfUnsecuredLines'] > 2).astype(int), index=raw.index)
    pipeline = frozen.build_final_model_pipeline().fit(raw.iloc[:140], y.iloc[:140])
    X, explanation, audit = global_shap.explain_validation(pipeline, raw.iloc[140:])
    importance = global_shap.global_importance(explanation.values, OUTPUT_FEATURE_COLUMNS)
    native = pd.DataFrame({'model': 'XGBoost', 'feature': OUTPUT_FEATURE_COLUMNS,
                           'importance': pipeline.named_steps['model'].feature_importances_})
    native = native.sort_values('importance', ascending=False, kind='stable').reset_index(drop=True)
    native['rank'] = np.arange(1, 14)
    return SimpleNamespace(pipeline=pipeline, raw=raw, y=y, X=X, explanation=explanation,
                           audit=audit, importance=importance, native=native)


def test_exact_frozen_model_configuration_and_calibration():
    reference = json.loads((global_shap.DEFAULT_REPORT_DIR / 'final_model_selection.json').read_text(encoding='utf-8'))
    metadata = global_shap.verify_frozen_specification(reference)
    expected = dict(n_estimators=300, learning_rate=.05, max_depth=3, min_child_weight=1,
                    subsample=1., colsample_bytree=1., reg_alpha=0, reg_lambda=1,
                    objective='binary:logistic', eval_metric='logloss', random_state=42,
                    n_jobs=-1, tree_method='hist')
    assert metadata['model_parameters'] == expected
    assert metadata['calibration_method'] == 'none'
    assert metadata['final_model'] == 'XGBoost Baseline / XGB-01'
    assert (metadata['default_audit_threshold'], metadata['development_operating_threshold']) == (.5, .19)


def test_configuration_drift_rejected_before_analysis():
    reference = frozen.get_frozen_metadata()
    reference['model_parameters']['max_depth'] = 4
    with pytest.raises(ValueError, match='Frozen specification'):
        global_shap.verify_frozen_specification(reference)


def test_loader_never_accesses_test_or_validation_target(monkeypatch):
    class Validation:
        X = 'validation features'

        @property
        def y(self):
            pytest.fail('Validation target must not be accessed')

    class Locked:
        train = SimpleNamespace(X='training features', y='training labels')
        validation = Validation()

        @property
        def test(self):
            pytest.fail('Test partition must not be accessed')
    monkeypatch.setattr(global_shap, 'load_data', lambda: None)
    monkeypatch.setattr(global_shap, 'split_data', lambda _: Locked())
    data = global_shap.load_explanation_data()
    assert data.X_validation == 'validation features'
    assert [f.name for f in fields(data)] == ['X_train', 'y_train', 'X_validation']
    assert list(inspect.signature(global_shap.explain_validation).parameters) == ['pipeline', 'X_validation']


def test_transformed_schema_names_and_metadata_exclusion(small_fit):
    X = small_fit.X
    assert isinstance(X, pd.DataFrame) and X.shape == (40, 13)
    assert X.columns.tolist() == OUTPUT_FEATURE_COLUMNS
    assert not {'SeriousDlqin2yrs', 'Unnamed: 0', 'group_id', 'feature_group'} & set(X.columns)
    assert X.index.equals(small_fit.raw.iloc[140:].index)
    assert np.isfinite(X.to_numpy()).all()


def test_shap_values_and_base_are_finite_and_aligned(small_fit):
    explanation = small_fit.explanation
    assert explanation.values.shape == (40, 13)
    assert np.isfinite(explanation.values).all()
    assert explanation.base_values.shape == (40,)
    assert np.isfinite(explanation.base_values).all()
    global_shap.validate_shap_output(explanation, small_fit.X, small_fit.audit['expected_value'])


@pytest.mark.parametrize('problem', ['shape', 'nan', 'base', 'names', 'data'])
def test_invalid_explanation_rejected(small_fit, problem):
    explanation = deepcopy(small_fit.explanation)
    if problem == 'shape':
        explanation.values = explanation.values[:, :-1]
    elif problem == 'nan':
        explanation.values[0, 0] = np.nan
    elif problem == 'base':
        explanation.base_values[0] = np.inf
    elif problem == 'names':
        explanation.feature_names = ['wrong'] * 13
    else:
        explanation.data = np.zeros_like(explanation.data)
    with pytest.raises(ValueError):
        global_shap.validate_shap_output(explanation, small_fit.X, small_fit.audit['expected_value'])


def test_real_tree_additivity_on_small_validation(small_fit):
    assert small_fit.audit['additivity']['status'] == 'PASS'
    assert small_fit.audit['additivity']['rows'] == 40
    margins = small_fit.pipeline.named_steps['model'].predict(small_fit.X, output_margin=True)
    np.testing.assert_allclose(small_fit.explanation.base_values + small_fit.explanation.values.sum(axis=1),
                               margins, rtol=1e-5, atol=1e-5)


def test_additivity_helper_known_values_and_failure():
    values = np.array([[1., -2.], [3., .5]])
    result = global_shap.check_additivity(values, .25, np.array([-.75, 3.75]))
    assert result['max_absolute_error'] == result['mean_absolute_error'] == 0
    with pytest.raises(ValueError, match='additivity failed'):
        global_shap.check_additivity(values, .25, np.array([-.65, 3.75]))
    with pytest.raises(ValueError, match='aligned'):
        global_shap.check_additivity(values, np.zeros(3), np.zeros(2))
    with pytest.raises(ValueError, match='finite'):
        global_shap.check_additivity(values, np.nan, np.zeros(2))


def test_global_importance_nonnegative_normalized_complete(small_fit):
    table = small_fit.importance
    assert table.shape == (13, 4) and table['rank'].tolist() == list(range(1, 14))
    assert set(table.feature) == set(OUTPUT_FEATURE_COLUMNS)
    assert (table.mean_absolute_shap >= 0).all()
    assert table.relative_importance.sum() == pytest.approx(1)
    means = np.abs(small_fit.explanation.values.astype(float)).mean(axis=0)
    np.testing.assert_allclose(table.set_index('feature').loc[OUTPUT_FEATURE_COLUMNS, 'mean_absolute_shap'], means)


def test_ranking_ties_and_zero_contributions_are_deterministic():
    equal = global_shap.global_importance(np.ones((4, 13)), OUTPUT_FEATURE_COLUMNS)
    assert equal.feature.tolist() == OUTPUT_FEATURE_COLUMNS
    assert np.allclose(equal.relative_importance, 1/13)
    empty_contributions = global_shap.global_importance(np.zeros((4, 13)), OUTPUT_FEATURE_COLUMNS)
    assert empty_contributions.feature.tolist() == OUTPUT_FEATURE_COLUMNS
    assert empty_contributions.relative_importance.sum() == 0
    again = global_shap.global_importance(np.ones((4, 13)), OUTPUT_FEATURE_COLUMNS)
    pd.testing.assert_frame_equal(equal, again)


@pytest.mark.parametrize('values', [np.zeros((4, 12)), np.zeros((0, 13)), np.full((4, 13), np.nan)])
def test_importance_rejects_invalid_arrays(values):
    with pytest.raises(ValueError):
        global_shap.global_importance(values, OUTPUT_FEATURE_COLUMNS)


def test_sampler_is_deterministic_unique_and_target_free():
    first = global_shap.deterministic_positions(100, 25)
    second = global_shap.deterministic_positions(100, 25)
    np.testing.assert_array_equal(first, second)
    assert len(np.unique(first)) == 25 and first.min() >= 0 and first.max() < 100
    np.testing.assert_array_equal(global_shap.deterministic_positions(4, 20), np.arange(4))
    assert list(inspect.signature(global_shap.deterministic_positions).parameters) == ['rows', 'maximum']
    with pytest.raises(ValueError):
        global_shap.deterministic_positions(0, 5)


def test_small_shap_repeat_preserves_inputs_and_fitted_statistics(small_fit):
    raw_before = small_fit.raw.copy(deep=True)
    medians = small_fit.pipeline.named_steps['preprocessor'].named_steps['imputer'].statistics_.copy()
    X, explanation, audit = global_shap.explain_validation(small_fit.pipeline, small_fit.raw.iloc[140:])
    np.testing.assert_array_equal(explanation.values, small_fit.explanation.values)
    np.testing.assert_array_equal(explanation.base_values, small_fit.explanation.base_values)
    pd.testing.assert_frame_equal(X, small_fit.X)
    pd.testing.assert_frame_equal(raw_before, small_fit.raw)
    pd.testing.assert_series_equal(medians, small_fit.pipeline.named_steps['preprocessor'].named_steps['imputer'].statistics_)
    assert audit['additivity']['positions_sha256'] == small_fit.audit['additivity']['positions_sha256']


def test_native_importance_join_and_mismatch(small_fit):
    fitted = small_fit.pipeline.named_steps['model'].feature_importances_
    comparison = global_shap.compare_native_importance(small_fit.importance, small_fit.native, fitted)
    assert list(comparison.columns) == ['feature', 'shap_rank', 'mean_absolute_shap', 'xgboost_native_importance', 'native_rank']
    assert len(comparison) == 13 and comparison.feature.tolist() == small_fit.importance.feature.tolist()
    with pytest.raises(ValueError, match='differs'):
        global_shap.compare_native_importance(small_fit.importance, small_fit.native, fitted + .1)
    with pytest.raises(ValueError):
        global_shap.compare_native_importance(small_fit.importance, small_fit.native.iloc[:-1], fitted)


def test_pattern_summary_and_dynamic_plot_selection(small_fit):
    patterns = global_shap.feature_pattern_summary(small_fit.X, small_fit.explanation.values, small_fit.importance)
    assert [p['feature'] for p in patterns] == small_fit.importance.feature.head(5).tolist()
    assert all('Descriptive association' in p['interpretation'] for p in patterns)
    names = global_shap.figure_names(small_fit.importance)
    assert len(names) == 5
    for i, feature in enumerate(small_fit.importance.feature.head(3), 1):
        assert names[i+1] == f'shap_dependence_{i:02d}_{global_shap.safe_feature_name(feature)}.png'
    assert global_shap.safe_feature_name('a/../b-1') == 'a_b_1'


@pytest.fixture
def result(monkeypatch, small_fit):
    """Exercise orchestration using the shared small frozen fit; never full data."""
    data = global_shap.ExplanationData(small_fit.raw.iloc[:140], small_fit.y.iloc[:140], small_fit.raw.iloc[140:])
    calls = []
    real_factory = frozen.build_final_model_pipeline
    def factory():
        calls.append('frozen_factory')
        return real_factory()
    def fit(pipeline, X, y):
        assert X is data.X_train and y is data.y_train
        calls.append('fit_train_only')
        return small_fit.pipeline
    def explain(pipeline, X):
        assert pipeline is small_fit.pipeline and X is data.X_validation
        calls.append('explain_validation_only')
        return small_fit.X, small_fit.explanation, small_fit.audit
    monkeypatch.setattr(frozen, 'build_final_model_pipeline', factory)
    monkeypatch.setattr(global_shap, 'fit_tree', fit)
    monkeypatch.setattr(global_shap, 'explain_validation', explain)
    before = frozen.get_frozen_metadata()
    result = global_shap.run_analysis(data, before, small_fit.native)
    assert calls == ['frozen_factory', 'fit_train_only', 'explain_validation_only']
    assert before == frozen.get_frozen_metadata()
    return result


def test_analysis_partition_schema_and_frozen_configuration(result):
    report = result.report
    assert report['shap_dataset'] == 'validation' and report['validation_rows'] == 40
    assert report['train_rows'] == 140 and report['fit_partition'] == 'training only'
    assert report['transformed_feature_count'] == 13 and report['shap_output_shape'] == [40, 13]
    assert report['shap_output_space'] == 'raw margin / log-odds'
    assert report['explainer']['feature_perturbation'] == 'tree_path_dependent'
    assert report['no_model_changes'] and report['random_state'] == 42
    assert report['test_not_used_statement'] == global_shap.TEST_NOT_USED
    json.dumps(report, allow_nan=False)


def test_only_expected_global_outputs_and_report(result, tmp_path):
    state = np.random.get_state()
    global_shap.save_reports(result, tmp_path)
    after = np.random.get_state()
    assert state[0] == after[0] and state[2:] == after[2:]
    np.testing.assert_array_equal(state[1], after[1])
    files = {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob('*') if p.is_file()}
    expected = {'shap_global_importance.csv', 'shap_native_importance_comparison.csv',
                'shap_global_results.json', 'shap_global_explainability.md'}
    expected.update('figures/' + n for n in global_shap.figure_names(result.importance))
    assert files == expected
    assert len(pd.read_csv(tmp_path / 'shap_global_importance.csv')) == 13
    assert len(pd.read_csv(tmp_path / 'shap_native_importance_comparison.csv')) == 13
    report = (tmp_path / 'shap_global_explainability.md').read_text(encoding='utf-8')
    for heading in ['Scope', 'Frozen Model', 'Explanation Dataset', 'SHAP Output Space', 'Additivity Verification',
                    'Global Feature Importance', 'Feature Effect Patterns', 'SHAP vs Native Tree Importance',
                    'Interpretation Guidelines', 'Limitations', 'What SHAP Does Not Establish']:
        assert f'## {heading}' in report
    assert 'not causal relationships' in report
    assert 'not a direct probability increase' in report
    for name in global_shap.figure_names(result.importance):
        assert (tmp_path / 'figures' / name).read_bytes().startswith(b'\x89PNG\r\n\x1a\n')


def test_no_selection_dropping_target_dependency_or_import_time_analysis():
    parsed = ast.parse(inspect.getsource(global_shap))
    calls = {getattr(n.func, 'id', getattr(n.func, 'attr', '')) for n in ast.walk(parsed) if isinstance(n, ast.Call)}
    assert not calls & {'GridSearchCV', 'RandomizedSearchCV', 'select_scenario', 'run_search', 'drop',
                        'SelectFromModel', 'RFE', 'build_candidates', 'fit_calibrator', 'XGBClassifier',
                        'LGBMClassifier', 'waterfall', 'force', 'predict_proba'}
    attributes = {n.attr for n in ast.walk(parsed) if isinstance(n, ast.Attribute)}
    assert not attributes & {'X_test', 'y_test', 'y_validation', 'test'}
    for node in parsed.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.Expr)):
            direct = {getattr(n.func, 'id', getattr(n.func, 'attr', '')) for n in ast.walk(node) if isinstance(n, ast.Call)}
            assert not direct & {'fit', 'run_analysis', 'load_data', 'load_explanation_data', 'TreeExplainer', 'main'}
