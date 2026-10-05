"""Reproduce the approved, group-disjoint split without preprocessing or export.

Run ``python -m src.data_split`` using the project's virtual environment. The
module only reads the raw CSV and prints validation results. ``split_data``
returns ``splits.train.X``, ``splits.train.y`` and equivalent validation/test
partitions. Group IDs and source-row references are audit metadata, not inputs
to a predictive model.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
from itertools import combinations
import json
from pathlib import Path
from typing import Iterator

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.model_selection import GroupShuffleSplit

RANDOM_STATE = 42
TARGET_COLUMN = "SeriousDlqin2yrs"
INDEX_COLUMN = "Unnamed: 0"
FEATURE_COLUMNS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]
DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data/raw/cs-training.csv"
EXPECTED_TOTAL_ROWS = 150_000
EXPECTED_GROUP_COUNT = 149_354
EXPECTED_COUNTS = {"train": (104_998, 7_061), "validation": (22_486, 1_478), "test": (22_516, 1_487)}
EXPECTED_ASSIGNMENT_SHA256 = "c1061432589f1981dfce4a3bbe0ae5c55818767ea97e6b8b80962e9f9b7ec1f2"


class SplitValidationError(ValueError):
    """The input schema or an approved split invariant does not hold."""


@dataclass(frozen=True)
class DataPartition:
    """Raw model inputs and aligned audit metadata for one partition.

    Pandas row labels are preserved. ``row_positions`` are zero-based positions
    in the input frame; ``source_ids`` retain the original CSV index column.
    Frozen fields do not make the contained pandas/NumPy objects immutable:
    call ``validate_splits`` again if a caller modifies them.
    """

    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series
    source_ids: pd.Series
    row_positions: NDArray[np.int64]


@dataclass(frozen=True)
class DatasetSplits:
    """The three partitions, in the approved assignment-label order."""

    train: DataPartition
    validation: DataPartition
    test: DataPartition

    def items(self) -> Iterator[tuple[str, DataPartition]]:
        """Iterate in train=0, validation=1, test=2 order."""
        yield "train", self.train
        yield "validation", self.validation
        yield "test", self.test


@dataclass(frozen=True)
class SplitSummary:
    """Descriptive checks; percentages use observations, not group counts."""

    rows: int
    percentage: float
    positive_count: int
    positive_rate: float
    unique_groups: int


def validate_schema(data: pd.DataFrame) -> None:
    """Require the documented raw columns and valid labels/source indices.

    Missing predictive values remain allowed. No domain cleaning is performed:
    age zero, special delinquency values and extreme ratios are retained.
    """
    if not data.columns.is_unique:
        raise SplitValidationError("Raw schema contains duplicate column names.")
    expected = [INDEX_COLUMN, TARGET_COLUMN, *FEATURE_COLUMNS]
    missing = [column for column in expected if column not in data.columns]
    unexpected = [column for column in data.columns if column not in expected]
    if missing or unexpected:
        raise SplitValidationError(f"Raw schema mismatch: missing={missing}; unexpected={unexpected}.")
    if data.empty:
        raise SplitValidationError("Raw dataset is empty.")
    target = data[TARGET_COLUMN]
    if target.isna().any() or not target.isin([0, 1]).all():
        raise SplitValidationError(f"{TARGET_COLUMN} must contain only nonmissing 0/1 labels.")
    if not is_numeric_dtype(target.dtype) or is_bool_dtype(target.dtype):
        raise SplitValidationError(f"{TARGET_COLUMN} must have a numeric, non-boolean dtype.")
    source_ids = data[INDEX_COLUMN]
    if source_ids.isna().any() or not source_ids.is_unique:
        raise SplitValidationError(f"{INDEX_COLUMN} must contain unique, nonmissing source indices.")
    for column in FEATURE_COLUMNS:
        values = data[column]
        if not is_numeric_dtype(values.dtype) or is_bool_dtype(values.dtype):
            raise SplitValidationError(f"Predictive feature {column} must be numeric.")
        if np.isinf(values.to_numpy(dtype=float, na_value=np.nan)).any():
            raise SplitValidationError(f"Predictive feature {column} contains infinity.")


def load_data(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Read and validate a raw CSV without changing it.

    The default path is relative to the module's repository, not the working
    directory. Round-trip float parsing and original row order are part of the
    approved exact-equality grouping contract.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Raw training CSV not found: {source}")
    try:
        data = pd.read_csv(source, float_precision="round_trip")
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise SplitValidationError(f"Cannot parse raw training CSV {source}: {exc}") from exc
    validate_schema(data)
    return data


def build_feature_groups(data: pd.DataFrame) -> pd.Series:
    """Group exact ten-feature vectors, including matching NaN positions.

    Target and CSV index are never grouping keys. Group numbers follow first
    occurrence order, so reproducibility depends on raw content/order and the
    pinned parser/library versions. They are not borrower identifiers.
    """
    validate_schema(data)
    features = data.loc[:, FEATURE_COLUMNS]
    return features.groupby(FEATURE_COLUMNS, dropna=False, sort=False).ngroup().astype("int64").rename("feature_group")


def _partition(data: pd.DataFrame, groups: pd.Series, positions: NDArray[np.int64]) -> DataPartition:
    """Copy selected raw observations, with no derived predictive columns."""
    return DataPartition(
        X=data.iloc[positions].loc[:, FEATURE_COLUMNS].copy(),
        y=data[TARGET_COLUMN].iloc[positions].copy(),
        groups=groups.iloc[positions].copy(),
        source_ids=data[INDEX_COLUMN].iloc[positions].copy(),
        row_positions=positions.copy(),
    )


def assignment_fingerprint(splits: DatasetSplits, total_rows: int) -> str:
    """Hash complete assignment labels in source-row order; write no files."""
    assignment = np.full(total_rows, -1, dtype=np.int8)
    for label, (name, partition) in enumerate(splits.items()):
        positions = np.asarray(partition.row_positions)
        if positions.ndim != 1 or not np.issubdtype(positions.dtype, np.integer):
            raise SplitValidationError(f"{name}: row_positions must be a one-dimensional integer array.")
        if (positions < 0).any() or (positions >= total_rows).any():
            raise SplitValidationError(f"{name}: row_positions are outside the source dataset.")
        if np.unique(positions).size != positions.size or (assignment[positions] != -1).any():
            raise SplitValidationError(f"{name}: duplicate row assignments or row overlap detected.")
        assignment[positions] = label
    if (assignment == -1).any():
        raise SplitValidationError("Split coverage is incomplete: some source rows are unassigned.")
    return hashlib.sha256(assignment.tobytes()).hexdigest()


def validate_splits(data: pd.DataFrame, splits: DatasetSplits) -> dict[str, SplitSummary]:
    """Validate exact counts, content, coverage, group isolation and assignment.

    Validation is intentionally tied to the approved 150,000-row dataset and
    seed-42 assignment. A changed source requires explicit methodology review,
    not relaxed checks or a search for a different seed.
    """
    validate_schema(data)
    if len(data) != EXPECTED_TOTAL_ROWS:
        raise SplitValidationError(f"Expected {EXPECTED_TOTAL_ROWS} source rows; received {len(data)}.")
    fingerprint = assignment_fingerprint(splits, len(data))
    expected_groups = build_feature_groups(data)
    if expected_groups.nunique() != EXPECTED_GROUP_COUNT:
        raise SplitValidationError(f"Expected {EXPECTED_GROUP_COUNT} raw feature groups; received {expected_groups.nunique()}.")
    summaries: dict[str, SplitSummary] = {}
    for name, partition in splits.items():
        expected_rows, expected_positives = EXPECTED_COUNTS[name]
        if len(partition.row_positions) != expected_rows:
            raise SplitValidationError(f"{name}: expected {expected_rows} rows; received {len(partition.row_positions)}.")
        if partition.X.columns.tolist() != FEATURE_COLUMNS:
            raise SplitValidationError(f"{name}: X must contain exactly the ten predictive features in dictionary order.")
        if any(len(value) != expected_rows for value in (partition.X, partition.y, partition.groups, partition.source_ids)):
            raise SplitValidationError(f"{name}: feature, target and audit metadata lengths are not aligned.")
        if partition.y.isna().any() or not partition.y.isin([0, 1]).all():
            raise SplitValidationError(f"{name}: y must contain only nonmissing 0/1 labels.")
        if int(partition.y.sum()) != expected_positives:
            raise SplitValidationError(f"{name}: expected {expected_positives} positives; received {partition.y.sum()}.")
        source = data.iloc[partition.row_positions]
        if not partition.X.equals(source.loc[:, FEATURE_COLUMNS]):
            raise SplitValidationError(f"{name}: X differs from unmodified source features or row alignment.")
        if not partition.y.equals(source[TARGET_COLUMN]):
            raise SplitValidationError(f"{name}: target differs from the corresponding source labels.")
        if not partition.source_ids.equals(source[INDEX_COLUMN]):
            raise SplitValidationError(f"{name}: source indices are inconsistent with row positions.")
        if not partition.groups.equals(expected_groups.iloc[partition.row_positions]):
            raise SplitValidationError(f"{name}: feature-group metadata differs from raw feature-only groups.")
        summaries[name] = SplitSummary(
            rows=expected_rows, percentage=100 * expected_rows / len(data),
            positive_count=expected_positives, positive_rate=100 * float(partition.y.mean()),
            unique_groups=int(partition.groups.nunique()),
        )
    for (left_name, left), (right_name, right) in combinations(splits.items(), 2):
        pair = f"{left_name}/{right_name}"
        if np.intersect1d(left.groups, right.groups).size:
            raise SplitValidationError(f"{pair}: raw feature-group overlap detected.")
        if np.intersect1d(left.row_positions, right.row_positions).size:
            raise SplitValidationError(f"{pair}: source-row overlap detected.")
        if set(left.source_ids) & set(right.source_ids):
            raise SplitValidationError(f"{pair}: source-index overlap detected.")
    if fingerprint != EXPECTED_ASSIGNMENT_SHA256:
        raise SplitValidationError("Assignment fingerprint differs from the approved Step 5 assignment.")
    return summaries


def split_data(data: pd.DataFrame) -> DatasetSplits:
    """Create and validate the approved two-stage group-random split.

    This is not an explicitly stratified splitter. Group proportions approximate
    70/15/15 row proportions, and the exact approved label counts are checked.
    No transformation is applied before or after assignment.
    """
    validate_schema(data)
    if len(data) != EXPECTED_TOTAL_ROWS:
        raise SplitValidationError(f"Expected {EXPECTED_TOTAL_ROWS} source rows; received {len(data)}.")
    groups = build_feature_groups(data)
    rows = np.arange(len(data), dtype=np.int64)
    train, holdout = next(GroupShuffleSplit(
        n_splits=1, train_size=0.70, random_state=RANDOM_STATE
    ).split(rows, groups=groups.to_numpy()))
    validation_local, test_local = next(GroupShuffleSplit(
        n_splits=1, test_size=0.50, random_state=RANDOM_STATE
    ).split(holdout, groups=groups.iloc[holdout].to_numpy()))
    result = DatasetSplits(
        train=_partition(data, groups, train),
        validation=_partition(data, groups, holdout[validation_local]),
        test=_partition(data, groups, holdout[test_local]),
    )
    validate_splits(data, result)
    return result


def main() -> None:
    """Run a read-only split smoke check and print a JSON summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Raw training CSV path")
    args = parser.parse_args()
    data = load_data(args.data)
    splits = split_data(data)
    summary = validate_splits(data, splits)
    print(json.dumps({
        "validation": "PASS", "random_state": RANDOM_STATE,
        "total_rows": sum(part.rows for part in summary.values()),
        "assignment_sha256": assignment_fingerprint(splits, len(data)),
        "splits": {name: asdict(part) for name, part in summary.items()},
        "overlaps": {f"{a}/{b}": {
            "feature_groups": int(np.intersect1d(left.groups, right.groups).size),
            "row_positions": int(np.intersect1d(left.row_positions, right.row_positions).size),
            "source_indices": len(set(left.source_ids) & set(right.source_ids)),
        } for (a, left), (b, right) in combinations(splits.items(), 2)},
    }, indent=2))


if __name__ == "__main__":
    main()
