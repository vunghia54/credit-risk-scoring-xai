"""Checks against the verified raw dataset and targeted in-memory mutations.

The raw CSV is intentionally not committed. Integration tests require the
verified local file; they fail clearly when it is missing instead of silently
skipping the exact-assignment checks. No dataset files are written.
"""

from dataclasses import replace
import hashlib
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_split import (
    DEFAULT_DATA_PATH,
    EXPECTED_ASSIGNMENT_SHA256,
    EXPECTED_COUNTS,
    FEATURE_COLUMNS,
    INDEX_COLUMN,
    RANDOM_STATE,
    TARGET_COLUMN,
    DatasetSplits,
    SplitValidationError,
    assignment_fingerprint,
    build_feature_groups,
    load_data,
    split_data,
    validate_schema,
    validate_splits,
)


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return load_data()


@pytest.fixture(scope="module")
def splits(raw: pd.DataFrame) -> DatasetSplits:
    return split_data(raw)


def test_schema_matches_dictionary(raw: pd.DataFrame) -> None:
    dictionary = pd.read_excel(DEFAULT_DATA_PATH.parent / "Data Dictionary.xls", header=1)
    declared = dictionary["Variable Name"].dropna().tolist()
    assert FEATURE_COLUMNS == [name for name in declared if name != TARGET_COLUMN]
    assert raw.columns.tolist() == [INDEX_COLUMN, *declared]
    assert len(FEATURE_COLUMNS) == 10
    validate_schema(raw)


@pytest.mark.parametrize("missing_column", [TARGET_COLUMN, INDEX_COLUMN, "MonthlyIncome"])
def test_schema_rejects_missing_columns(raw: pd.DataFrame, missing_column: str) -> None:
    with pytest.raises(SplitValidationError, match="missing="):
        validate_schema(raw.head(3).drop(columns=missing_column))


def test_schema_rejects_extra_predictive_column(raw: pd.DataFrame) -> None:
    invalid = raw.head(3).assign(feature_group=0)
    with pytest.raises(SplitValidationError, match="unexpected=.*feature_group"):
        validate_schema(invalid)


def test_schema_rejects_duplicate_column_names(raw: pd.DataFrame) -> None:
    invalid = pd.concat([raw.head(3), raw.head(3)[["age"]]], axis=1)
    with pytest.raises(SplitValidationError, match="duplicate column"):
        validate_schema(invalid)


@pytest.mark.parametrize("label", [np.nan, 2, -1])
def test_schema_rejects_invalid_target(raw: pd.DataFrame, label: float) -> None:
    invalid = raw.head(3).copy()
    invalid[TARGET_COLUMN] = invalid[TARGET_COLUMN].astype(float)
    invalid.loc[invalid.index[0], TARGET_COLUMN] = label
    with pytest.raises(SplitValidationError, match="nonmissing 0/1"):
        validate_schema(invalid)


@pytest.mark.parametrize("missing", [False, True])
def test_schema_rejects_invalid_source_index(raw: pd.DataFrame, missing: bool) -> None:
    invalid = raw.head(3).copy()
    invalid[INDEX_COLUMN] = invalid[INDEX_COLUMN].astype(float)
    invalid.iloc[0, invalid.columns.get_loc(INDEX_COLUMN)] = np.nan if missing else invalid[INDEX_COLUMN].iloc[1]
    with pytest.raises(SplitValidationError, match="unique, nonmissing"):
        validate_schema(invalid)


def test_schema_rejects_nonnumeric_feature(raw: pd.DataFrame) -> None:
    invalid = raw.head(3).copy()
    invalid["age"] = "not-a-number"
    with pytest.raises(SplitValidationError, match="age must be numeric"):
        validate_schema(invalid)


def test_schema_rejects_infinite_feature(raw: pd.DataFrame) -> None:
    invalid = raw.head(3).copy()
    invalid.loc[invalid.index[0], "DebtRatio"] = np.inf
    with pytest.raises(SplitValidationError, match="DebtRatio contains infinity"):
        validate_schema(invalid)


def test_schema_rejects_empty_data(raw: pd.DataFrame) -> None:
    with pytest.raises(SplitValidationError, match="empty"):
        validate_schema(raw.head(0))


def test_loading_missing_file_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Raw training CSV not found"):
        load_data(tmp_path / "missing.csv")


def test_default_path_is_independent_of_working_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, raw: pd.DataFrame) -> None:
    monkeypatch.chdir(tmp_path)
    pd.testing.assert_frame_equal(load_data(), raw)


def test_grouping_uses_features_only_with_consistent_nan(raw: pd.DataFrame) -> None:
    # Repeat one real raw feature vector in memory, with different labels/IDs.
    row = raw.loc[raw["MonthlyIncome"].isna()].head(1)
    example = pd.concat([row, row, row], ignore_index=True)
    example[INDEX_COLUMN] = [1, 2, 3]
    example[TARGET_COLUMN] = [0, 1, 0]
    example.loc[2, "MonthlyIncome"] = 0  # Observed zero must differ from NaN.
    groups = build_feature_groups(example)
    assert groups.iloc[0] == groups.iloc[1]
    assert groups.iloc[0] != groups.iloc[2]
    changed_metadata = example.copy()
    changed_metadata[INDEX_COLUMN] = [103, 102, 101]
    changed_metadata[TARGET_COLUMN] = 1 - changed_metadata[TARGET_COLUMN]
    pd.testing.assert_series_equal(groups, build_feature_groups(changed_metadata))


def test_expected_feature_group_population(raw: pd.DataFrame) -> None:
    sizes = build_feature_groups(raw).value_counts()
    assert len(sizes) == 149354
    assert int((sizes > 1).sum()) == 354
    assert int(sizes.loc[sizes > 1].sum()) == 1000
    assert sizes.max() == 12


@pytest.mark.parametrize("name", ["train", "validation", "test"])
def test_partition_schema_and_expected_counts(name: str, splits: DatasetSplits) -> None:
    part = getattr(splits, name)
    expected_rows, expected_positives = EXPECTED_COUNTS[name]
    assert len(part.X) == len(part.y) == expected_rows
    assert int(part.y.sum()) == expected_positives
    assert part.X.columns.tolist() == FEATURE_COLUMNS
    assert {TARGET_COLUMN, INDEX_COLUMN, "feature_group", "group_id"}.isdisjoint(part.X.columns)
    assert part.y.notna().all() and set(part.y.unique()) == {0, 1}


def test_complete_coverage_and_no_overlap(raw: pd.DataFrame, splits: DatasetSplits) -> None:
    all_positions = np.concatenate([part.row_positions for _, part in splits.items()])
    np.testing.assert_array_equal(np.sort(all_positions), np.arange(150000))
    all_ids = pd.concat([part.source_ids for _, part in splits.items()])
    assert all_ids.is_unique and set(all_ids) == set(raw[INDEX_COLUMN])
    for (_, left), (_, right) in combinations(splits.items(), 2):
        assert np.intersect1d(left.groups, right.groups).size == 0
        assert np.intersect1d(left.row_positions, right.row_positions).size == 0
        assert set(left.source_ids).isdisjoint(right.source_ids)


def test_deterministic_exact_assignment_and_no_raw_mutation(raw: pd.DataFrame, splits: DatasetSplits) -> None:
    before = raw.copy(deep=True)
    file_hash_before = hashlib.sha256(DEFAULT_DATA_PATH.read_bytes()).hexdigest()
    repeated = split_data(raw)
    assert RANDOM_STATE == 42
    for (name, left), (_, right) in zip(splits.items(), repeated.items()):
        np.testing.assert_array_equal(left.row_positions, right.row_positions, err_msg=name)
        pd.testing.assert_frame_equal(left.X, right.X)
        pd.testing.assert_series_equal(left.y, right.y)
        pd.testing.assert_series_equal(left.groups, right.groups)
        pd.testing.assert_series_equal(left.source_ids, right.source_ids)
    assert assignment_fingerprint(repeated, len(raw)) == EXPECTED_ASSIGNMENT_SHA256
    pd.testing.assert_frame_equal(raw, before)
    assert hashlib.sha256(DEFAULT_DATA_PATH.read_bytes()).hexdigest() == file_hash_before


def test_raw_values_are_preserved_in_partitions(raw: pd.DataFrame, splits: DatasetSplits) -> None:
    for _, part in splits.items():
        expected = raw.iloc[part.row_positions]
        pd.testing.assert_frame_equal(part.X, expected[FEATURE_COLUMNS])
        pd.testing.assert_series_equal(part.y, expected[TARGET_COLUMN])
    assert sum(int(part.X["age"].eq(0).sum()) for _, part in splits.items()) == 1
    assert sum(int(part.X["MonthlyIncome"].isna().sum()) for _, part in splits.items()) == 29731
    assert sum(int(part.X["NumberOfTimes90DaysLate"].isin([96, 98]).sum()) for _, part in splits.items()) == 269


def test_validation_accepts_approved_assignment(raw: pd.DataFrame, splits: DatasetSplits) -> None:
    summary = validate_splits(raw, splits)
    assert sum(part.rows for part in summary.values()) == 150000
    assert sum(part.positive_count for part in summary.values()) == 10026
    assert [part.unique_groups for part in summary.values()] == [104547, 22403, 22404]


def test_split_rejects_unapproved_dataset_size(raw: pd.DataFrame) -> None:
    with pytest.raises(SplitValidationError, match="Expected 150000 source rows"):
        split_data(raw.head(100))


def test_validation_rejects_incomplete_coverage(raw: pd.DataFrame, splits: DatasetSplits) -> None:
    broken = replace(splits, train=replace(splits.train, row_positions=splits.train.row_positions[:-1]))
    with pytest.raises(SplitValidationError, match="coverage is incomplete"):
        validate_splits(raw, broken)


def test_validation_rejects_row_overlap(raw: pd.DataFrame, splits: DatasetSplits) -> None:
    positions = splits.validation.row_positions.copy()
    positions[0] = splits.train.row_positions[0]
    broken = replace(splits, validation=replace(splits.validation, row_positions=positions))
    with pytest.raises(SplitValidationError, match="row overlap"):
        validate_splits(raw, broken)


@pytest.mark.parametrize("column", [TARGET_COLUMN, INDEX_COLUMN, "group_id"])
def test_validation_rejects_metadata_in_model_features(raw: pd.DataFrame, splits: DatasetSplits, column: str) -> None:
    broken_X = splits.train.X.assign(**{column: 0})
    broken = replace(splits, train=replace(splits.train, X=broken_X))
    with pytest.raises(SplitValidationError, match="exactly the ten predictive features"):
        validate_splits(raw, broken)


@pytest.mark.parametrize("field,message", [
    ("groups", "feature-group metadata"), ("source_ids", "source indices"),
    ("y", "expected 7061 positives"), ("X", "differs from unmodified source"),
])
def test_validation_rejects_tampered_values(raw: pd.DataFrame, splits: DatasetSplits, field: str, message: str) -> None:
    value = getattr(splits.train, field).copy()
    if field == "X":
        value.iloc[0, value.columns.get_loc("age")] += 1
    elif field == "y":
        value.iloc[0] = 1 - value.iloc[0]
    else:
        value.iloc[0] = -1
    broken = replace(splits, train=replace(splits.train, **{field: value}))
    with pytest.raises(SplitValidationError, match=message):
        validate_splits(raw, broken)
