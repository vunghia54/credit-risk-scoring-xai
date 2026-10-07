"""Small-data local explanation checks; no real holdout or full-validation SHAP."""

import ast
from copy import deepcopy
from dataclasses import fields
import inspect
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src import final_model as frozen
from src import shap_global
from src import shap_local as local
from src.data_split import FEATURE_COLUMNS
from src.preprocessing import OUTPUT_FEATURE_COLUMNS


def test_exact_six_rules_and_friendly_case_ids():
    assert local.SELECTION_RULES == ('LOW_SCORE', 'MEDIAN_SCORE', 'HIGH_SCORE', 'JUST_BELOW_0_19', 'JUST_ABOVE_0_19', 'NEAR_0_50')
    assert local.CASE_IDS == ('VAL_CASE_01_LOW', 'VAL_CASE_02_MEDIAN', 'VAL_CASE_03_HIGH',
                              'VAL_CASE_04_BELOW_019', 'VAL_CASE_05_ABOVE_019', 'VAL_CASE_06_NEAR_050')
    assert local.RANDOM_STATE == 42


def test_selection_matches_each_criterion_without_target():
    scores = np.array([.001, .007, .02, .09, .16, .189, .19, .25, .49, .502, .7, .91, .97])
    table = local.select_cases(scores)
    assert len(table) == 6 and table.validation_position.is_unique
    assert table.case_id.tolist() == list(local.CASE_IDS)
    available = set(range(len(scores)))
    criteria = [*np.quantile(scores, [.05, .5, .95]), .19, .19, .5]
    for i, row in table.iterrows():
        eligible = [p for p in available if (i != 3 or scores[p] < .19) and (i != 4 or scores[p] >= .19)]
        best = min(eligible, key=lambda p: (abs(scores[p] - criteria[i]), p))
        assert row.validation_position == best
        assert row.predicted_probability == scores[best]
        available.remove(best)
    assert list(inspect.signature(local.select_cases).parameters) == ['scores']


def test_deterministic_deduplication_and_smaller_position_ties():
    scores = np.array([.1] * 6 + [.19] * 6 + [.5] * 6)
    first, second = local.select_cases(scores), local.select_cases(scores)
    pd.testing.assert_frame_equal(first, second)
    assert first.validation_position.tolist() == [0, 6, 12, 1, 7, 13]
    assert first.deduplication_occurred.tolist() == [False, False, False, True, True, True]
    assert first.skipped_selected_candidates.tolist() == [0, 0, 0, 1, 1, 1]
    assert first.iloc[4].predicted_probability == .19


def test_selector_does_not_modify_scores():
    scores = np.linspace(.01, .95, 40)
    before = scores.copy()
    local.select_cases(scores)
    np.testing.assert_array_equal(scores, before)


@pytest.mark.parametrize('scores', [[.1] * 5, [.1] * 9, [.9] * 9, [0, .2, .3, .4, .5, .6]])
def test_unavailable_unique_rule_fails_without_relaxing(scores):
    with pytest.raises(ValueError):
        local.select_cases(np.array(scores))


@pytest.mark.parametrize('scores', [[np.nan], [np.inf], [-.01], [1.01], [], [[.1, .2]]])
def test_invalid_scores_rejected(scores):
    with pytest.raises(ValueError):
        local.validate_scores(np.array(scores))


def test_exact_threshold_boundaries():
    operating, audit = local.threshold_status(np.array([.1899999, .19, .4999999, .5]))
    assert operating.tolist() == [0, 1, 1, 1]
    assert audit.tolist() == [0, 0, 0, 1]


def test_total_margin_reconciliation_including_extremes():
    margins = np.array([-1000., -np.log(3), 0., np.log(3), 1000.])
    result = local.reconcile_probability(margins, np.array([0., .25, .5, .75, 1.]))
    assert result['status'] == 'PASS' and result['max_absolute_error'] < 1e-15
    with pytest.raises(ValueError, match='does not reproduce'):
        local.reconcile_probability(np.array([0.]), np.array([.6]))
    with pytest.raises(ValueError, match='aligned'):
        local.reconcile_probability(np.array([0., 1.]), np.array([.5]))


def test_local_additivity_matches_step16_and_fails_on_drift():
    assert local.check_additivity is shap_global.check_additivity
    values = np.array([[.1, -.2], [.3, .4]])
    audit = local.check_additivity(values, -.5, np.array([-.6, .2]))
    assert audit['atol'] == audit['rtol'] == 1e-5
    with pytest.raises(ValueError, match='additivity failed'):
        local.check_additivity(values, -.5, np.array([0., .2]))


def test_data_boundary_has_no_validation_target_or_test(monkeypatch):
    class Validation:
        X = 'validation features'

        @property
        def y(self):
            pytest.fail('Validation target accessed')

    class Splits:
        train = SimpleNamespace(X='train features', y='train labels')
        validation = Validation()

        @property
        def test(self):
            pytest.fail('Test partition accessed')
    monkeypatch.setattr(shap_global, 'load_data', lambda: None)
    monkeypatch.setattr(shap_global, 'split_data', lambda _: Splits())
    data = local.load_explanation_data()
    assert data.X_validation == 'validation features'
    assert [f.name for f in fields(data)] == ['X_train', 'y_train', 'X_validation']


@pytest.fixture(scope='module')
def small_result():
    """One exact frozen TRAIN fit and six explanations on deterministic toy data."""
    rng = np.random.default_rng(42)
    raw = pd.DataFrame(rng.uniform(.1, 4, (240, 10)), columns=FEATURE_COLUMNS)
    raw['age'] = rng.integers(20, 80, len(raw)).astype(float)
    raw.loc[::11, 'MonthlyIncome'] = np.nan
    raw.loc[::17, 'NumberOfDependents'] = np.nan
    raw.loc[::19, 'NumberOfTimes90DaysLate'] = 98
    raw.loc[::23, 'age'] = 0
    labels = pd.Series((raw.RevolvingUtilizationOfUnsecuredLines > 2).astype(int), index=raw.index)
    data = local.ExplanationData(raw.iloc[:180], labels.iloc[:180], raw.iloc[180:])
    reference = frozen.get_frozen_metadata()
    before_raw, before_reference = raw.copy(deep=True), deepcopy(reference)
    calls = []
    factory, fit, predict, select, explain = (frozen.build_final_model_pipeline, local.fit_tree,
                                            local.positive_probabilities, local.select_cases, local.explain_validation)
    with pytest.MonkeyPatch.context() as patch:
        def build():
            calls.append('frozen_factory')
            return factory()
        def fit_train(pipeline, X, y):
            assert X is data.X_train and y is data.y_train
            calls.append('fit_train')
            return fit(pipeline, X, y)
        def predict_validation(pipeline, X):
            assert X is data.X_validation
            calls.append('predict_validation')
            return predict(pipeline, X)
        def select_scores(scores):
            calls.append('select_scores')
            return select(scores)
        def explain_six(pipeline, X):
            assert len(X) == 6
            calls.append('explain_six')
            return explain(pipeline, X)
        patch.setattr(frozen, 'build_final_model_pipeline', build)
        patch.setattr(local, 'fit_tree', fit_train)
        patch.setattr(local, 'positive_probabilities', predict_validation)
        patch.setattr(local, 'select_cases', select_scores)
        patch.setattr(local, 'explain_validation', explain_six)
        result = local.run_analysis(data, reference)
    assert calls == ['frozen_factory', 'fit_train', 'predict_validation', 'select_scores', 'explain_six']
    assert reference == before_reference == frozen.get_frozen_metadata()
    pd.testing.assert_frame_equal(raw, before_raw)
    return result


def test_exact_frozen_xgb01_and_partition(small_result):
    r = small_result.report
    expected = dict(n_estimators=300, learning_rate=.05, max_depth=3, min_child_weight=1, subsample=1.,
                    colsample_bytree=1., reg_alpha=0, reg_lambda=1, objective='binary:logistic', eval_metric='logloss',
                    random_state=42, n_jobs=-1, tree_method='hist')
    assert r['frozen_specification']['model_parameters'] == expected
    assert r['final_model'] == 'XGBoost Baseline / XGB-01' and r['calibration'] == 'none'
    assert r['thresholds'] == {'development_operating': .19, 'default_audit': .5}
    assert r['explanation_partition'] == 'validation' and r['validation_rows'] == 60 and r['train_rows'] == 180


def test_configuration_drift_blocks_fit(monkeypatch):
    reference = frozen.get_frozen_metadata()
    reference['calibration_method'] = 'sigmoid'
    monkeypatch.setattr(local, 'fit_tree', lambda *args: pytest.fail('Fitting must not occur'))
    with pytest.raises(ValueError, match='specification'):
        local.run_analysis(None, reference)


def test_local_shapes_schema_and_finite_values(small_result):
    assert small_result.explanation.values.shape == (6, 13)
    assert np.isfinite(small_result.explanation.values).all()
    assert np.isfinite(small_result.explanation.base_values).all()
    assert small_result.explanation.feature_names == OUTPUT_FEATURE_COLUMNS
    assert small_result.report['local_additivity']['status'] == 'PASS'
    assert small_result.report['probability_reconciliation']['status'] == 'PASS'
    assert len(small_result.cases) == 6 and small_result.cases.validation_position.is_unique
    assert not {'SeriousDlqin2yrs', 'Unnamed: 0', 'feature_group', 'group_id'} & set(small_result.explanation.feature_names)


def test_contribution_table_is_78_rows_and_reconciles(small_result):
    table = small_result.contributions
    assert len(table) == 78
    assert list(table.columns) == ['case_id', 'feature', 'raw_value', 'transformed_value', 'shap_value', 'absolute_shap', 'direction', 'rank_within_case']
    for _, case in small_result.cases.iterrows():
        subset = table.loc[table.case_id.eq(case.case_id)]
        assert subset.rank_within_case.tolist() == list(range(1, 14))
        assert set(subset.feature) == set(OUTPUT_FEATURE_COLUMNS)
        assert np.isclose(case.base_value + subset.shap_value.sum(), case.raw_margin, atol=1e-5, rtol=1e-5)
        assert subset.absolute_shap.is_monotonic_decreasing


def test_raw_transformed_semantics_and_neutral_tie_ranking():
    raw = pd.DataFrame(np.ones((1, 10)), columns=FEATURE_COLUMNS)
    raw.loc[0, 'MonthlyIncome'] = np.nan
    transformed = pd.DataFrame(np.ones((1, 13)), columns=OUTPUT_FEATURE_COLUMNS)
    transformed.loc[0, 'MonthlyIncome'] = 5000
    values = np.array([[1., -1., 0., 1e-10, .2, -.2, 0., 0., 0., 0., 0., 0., 0.]])
    table = local.contribution_table(['case'], raw, transformed, values)
    assert table.feature.iloc[:4].tolist() == [FEATURE_COLUMNS[0], FEATURE_COLUMNS[1], FEATURE_COLUMNS[4], FEATURE_COLUMNS[5]]
    income = table.set_index('feature').loc['MonthlyIncome']
    assert pd.isna(income.raw_value) and income.transformed_value == 5000
    assert table.loc[table.feature.str.endswith('_missing'), 'raw_value'].isna().all()
    assert table.loc[table.shap_value.abs().le(1e-8), 'direction'].eq('Neutral').all()
    again = local.contribution_table(['case'], raw, transformed, values)
    pd.testing.assert_frame_equal(table, again)
    summary = local.top_contributors(table)[0]
    assert [c['shap_value'] for c in summary['top_contributors_to_higher_model_output']] == [1., .2]
    assert [c['shap_value'] for c in summary['top_contributors_to_lower_model_output']] == [-1., -.2]


def test_top_contributors_cap_three_and_no_neutrals():
    raw = pd.DataFrame(np.ones((1, 10)), columns=FEATURE_COLUMNS)
    transformed = pd.DataFrame(np.ones((1, 13)), columns=OUTPUT_FEATURE_COLUMNS)
    values = np.array([[1., 2., 3., 4., -1., -2., -3., -4., 0., 0., 0., 0., 0.]])
    summary = local.top_contributors(local.contribution_table(['case'], raw, transformed, values))[0]
    assert [c['shap_value'] for c in summary['top_contributors_to_higher_model_output']] == [4., 3., 2.]
    assert [c['shap_value'] for c in summary['top_contributors_to_lower_model_output']] == [-4., -3., -2.]


def test_artifact_schema_and_no_targets(small_result):
    r = small_result.report
    json.dumps(r, allow_nan=False)
    assert len(r['selected_cases']) == 6 and len(r['contributions']) == 78
    assert len(r['top_contributors']) == 6 and r['case_selection_policy']['target_used'] is False
    assert r['test_not_used_statement'] == local.TEST_NOT_USED
    assert not {'target', 'y_validation', 'y_test', 'true_label', 'source_id'} & set(small_result.cases.columns)
    assert r['shap_output_space'] == 'raw margin / log-odds'
    assert r['explainer']['model_output'] == 'raw' and r['explainer']['feature_perturbation'] == 'tree_path_dependent'


def test_report_sections_and_exact_output_files(small_result, tmp_path):
    local.save_reports(small_result, tmp_path)
    expected = {'shap_local_cases.csv', 'shap_local_contributions.csv', 'shap_local_results.json', 'shap_local_explainability.md'}
    expected.update('figures/' + name for name in local.WATERFALL_NAMES)
    expected.add('figures/shap_local_case_positions.png')
    assert {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob('*') if p.is_file()} == expected
    assert len(pd.read_csv(tmp_path / 'shap_local_cases.csv')) == 6
    assert len(pd.read_csv(tmp_path / 'shap_local_contributions.csv')) == 78
    text = (tmp_path / 'shap_local_explainability.md').read_text(encoding='utf-8')
    for heading in ['Scope', 'Frozen Model', 'Case Selection Policy', 'Why Target Was Not Used for Case Selection',
                    'SHAP Output Space', 'Selected Validation Cases', 'Local Additivity Verification',
                    'Interpretation Guidelines', 'Limitations', 'What These Explanations Are Not']:
        assert '## ' + heading in text
    for case in local.CASE_IDS:
        assert '### ' + case in text
    assert "SHAP explains this model's prediction. It does not establish causality." in text
    assert 'source meaning of 96/98 is unknown' in text
    for path in (tmp_path / 'figures').glob('*.png'):
        assert path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')


def test_no_model_selection_target_dependency_or_import_time_fit():
    parsed = ast.parse(inspect.getsource(local))
    calls = {getattr(n.func, 'id', getattr(n.func, 'attr', '')) for n in ast.walk(parsed) if isinstance(n, ast.Call)}
    assert not calls & {'GridSearchCV', 'RandomizedSearchCV', 'select_scenario', 'run_search', 'build_candidates',
                        'fit_calibrator', 'XGBClassifier', 'LGBMClassifier', 'drop', 'SelectFromModel', 'RFE'}
    assert sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'fit_tree' for n in ast.walk(parsed)) == 1
    attributes = {n.attr for n in ast.walk(parsed) if isinstance(n, ast.Attribute)}
    assert not attributes & {'test', 'X_test', 'y_test', 'y_validation'}
    for node in parsed.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.Expr)):
            direct = {getattr(n.func, 'id', getattr(n.func, 'attr', '')) for n in ast.walk(node) if isinstance(n, ast.Call)}
            assert not direct & {'fit', 'run_analysis', 'load_data', 'load_explanation_data', 'TreeExplainer', 'main', 'select_cases'}
