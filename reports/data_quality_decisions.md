# Data Quality Decisions

Status: decisions documented; split methodology validated in memory. Preprocessing and modeling are not implemented.

Date: 2026-10-05. Evidence: the executed `notebooks/01_eda.ipynb`, `data/raw/Data Dictionary.xls`, and the Step 5 assignment checks reported below.

The sole modeling source is `data/raw/cs-training.csv`: 150,000 observations, ten raw predictive features, and complete binary `SeriousDlqin2yrs` labels. There are 10,026 positive labels (6.684%). The target represents serious delinquency, not bankruptcy. The local `train.csv` / `test.csv` and Kaggle `cs-test.csv` are not the project's internal evaluation sets.

Raw CSV values remain immutable. Decisions below describe later pipeline behavior, not edits applied to the dataset. No imputer/scaler was fitted, no resampling or model training occurred, and no processed datasets or split-assignment files were exported in this step. The EDA notebook remains the historical observation record; this document records the subsequent decisions.

## Index Column

| Item | Specification |
|---|---|
| Observed fact | `Unnamed: 0` is unique and sequential from 1 to 150000, and is absent from the Data Dictionary. |
| Decision | **Exclude from predictive features.** It is also excluded from feature-group construction. |
| Rationale | An index-like field is not a documented borrower identifier or a credit-risk feature. |
| Implementation timing | Preserve in raw data and audit references; omit when selecting model inputs in a future pipeline. Do not rewrite the CSV. |

## Missing Values

| Feature | Observed fact | Decision | Rationale | Implementation timing |
|---|---|---|---|---|
| `MonthlyIncome` | 29,731 missing values (19.8207%). Missing/present event rates in the descriptive EDA are 5.6137% / 6.9486%. | Median imputation **plus a missing indicator**. Observed zero remains zero. | Missingness is associated with the target, without establishing causality; zero and unobserved income are different states. | Fit the median only on the training partition, and only the fitting fold during cross-validation. Apply those fitted statistics unchanged to validation/test. |
| `NumberOfDependents` | 3,924 missing values (2.616%). | Median imputation; retain the missing indicator when using a shared `SimpleImputer(strategy="median", add_indicator=True)` numeric branch. | A common train-fitted approach preserves observations and distinguishes missing from observed counts. | Fit on training/fitting-fold data only. Do not calculate a full-dataset median for modeling. |

The intended implementation can use `SimpleImputer(add_indicator=True)` for the appropriate numeric branch. Indicator columns must be learned/fixed from the training schema: this option only adds indicators for features missing at fit time. Test-time missingness must not trigger a refit or a changed output schema. Pipeline contracts should verify the required income indicator and stable feature names. An unexpectedly all-missing training column must be reported and handled explicitly, not silently dropped.

## Duplicate Feature Vectors

| Item | Specification |
|---|---|
| Observed fact | Excluding the index but including target: 609 excess duplicate rows in 351 repeated groups, containing 960 observations. Using only the ten features: 646 excess duplicate rows in 354 repeated groups, containing 1,000 observations. The largest group has 12 observations. There are 37 conflicting-target groups containing 145 observations. |
| Decision | **Retain every observation.** Assign identical ten-feature vectors to one internal split. |
| Rationale | Identical features do not establish shared borrower identity. Conflicting labels make automatic deduplication particularly difficult to justify. Nevertheless, splitting identical vectors across partitions introduces an avoidable dependence concern. |
| Implementation timing | Feature-only group construction is validated now in memory; recreate the same assignments in the later split implementation. Never use the group ID as a model feature. |

Group construction contract:

- Select exactly the ten raw features in the order shown in the reproduction recipe below.
- Read the verified source with `float_precision="round_trip"`; retain original row order.
- Use `X.groupby(FEATURES, dropna=False, sort=False).ngroup()`. The grouping key contains neither target nor source index. Target labels are not used to construct, order, or refine groups.
- Matching NaN positions are grouped consistently through `dropna=False`. Do not fill missing values with zero or another plausible observed value to construct keys.
- Equality is exact on the parsed raw numeric values; no rounding, imputation, capping or near-duplicate clustering is performed.
- Group numbers are encounter-order identifiers, deterministic for this file, row order and environment. They are not permanent borrower IDs. Changing the source, row order, parser or equality policy requires revalidation.
- There are **149,354 unique feature groups**, including 354 repeated groups. All groups were checked to contain just one distinct value per feature, counting NaN as a value.

## Invalid Age

| Item | Specification |
|---|---|
| Observed fact | One record has age 0; 13 records have age above 100, with a maximum of 109. |
| Decision | Treat age 0 as invalid in the future preprocessing pipeline: map it to missing in the working input, then impute using a training-fitted median. Retain ages above 100 at this stage. |
| Rationale | Zero is not a credible borrower age; very high age is not, on its own, proof of invalidity. |
| Implementation timing | After split assignment, within the pipeline. The zero-to-missing rule is fixed and stateless; the median is fitted on training/fitting-fold data only. No raw edits or EDA-driven age caps. |

## Special Delinquency Values

Affected columns:

- `NumberOfTime30-59DaysPastDueNotWorse`
- `NumberOfTime60-89DaysPastDueNotWorse`
- `NumberOfTimes90DaysLate`

**Observed fact:** 96 occurs on all three columns in five observations; 98 occurs on all three in 264 observations. The dictionary does not explain these values. They are **special values requiring investigation**, not confirmed error/missing codes and not established counts of 96 or 98 actual late-payment events.

**Decision:** retain raw values and create a future binary model input named `has_special_delinquency_value`, equal to 1 if any of the three raw columns is 96 or 98, otherwise 0. It is not created in this step and is not included in splitting groups.

**Recommended future representation:** compute the indicator first; in a separate pipeline working representation, replace 96/98 in each affected delinquency column with missing, then median-impute each column using only ordinary, observed values from training/fitting-fold data. Reuse the fitted medians for validation/test. Keep all observations and preserve genuine existing missingness. A training column with no ordinary observed values requires an explicit fallback decision rather than an invented median.

**Rationale and trade-off:** this avoids presenting undocumented special codes as very large ordinal counts to a linear model, while retaining an explicit signal that special values occurred. Imputed counts are placeholders, not recovered true counts. A shared binary flag loses the distinction between 96 and 98 and between affected columns, and may capture a source-system convention rather than transferable credit behavior. Any richer categorical/per-column representation should be compared through training/validation or group-disjoint cross-validation later, not selected using test performance. The combined flag is the planned minimal representation.

**Implementation timing:** design documented now; transformation, indicators and fitted imputation are deferred until the preprocessing step. Group identities continue to use the unmodified ten raw features.

## Extreme Ratios

| Feature | Observed fact | Decision | Rationale | Implementation timing |
|---|---|---|---|---|
| `RevolvingUtilizationOfUnsecuredLines` | 3,321 values exceed 1; maximum 50,708. The supplied variable is already a balance/credit-limit ratio over the accounts described in the dictionary. | Retain raw values; no arbitrary cap or winsorization. | Above-limit exposure is not automatically invalid. Separate balances and credit limits are unavailable. | Consider robust/log representations only through later validation, with any fitted parameters learned on training folds. Do not invent raw components. |
| `DebtRatio` | Maximum 329,664. Median is 1,159 for missing income, 930 for zero income, and about 0.292454 for positive income. | Retain raw values; no arbitrary cap/winsorization or inferred correction. | Officially, the numerator is monthly debt payments, alimony and living costs; the denominator is monthly gross income. It is **not total outstanding debt / income**. Income associations do not prove a source calculation rule. | Review representations jointly with income availability in future model comparisons; keep transformations within the training pipeline. |

## Income

**Observed fact:** 1,634 recorded incomes are zero; maximum income is 3,008,750. The dictionary does not specify currency or independently identify `MonthlyIncome` as gross/net income.

**Decision:** preserve zero as an observed value, distinct from missing. Retain high income values; do not set zeros to NaN or cap the tail automatically.

**Rationale:** changing observed zero to missing would merge different recorded states. Extreme income alone does not establish an error. Future median fitting includes valid observed zeros.

**Implementation timing:** enforce zero/missing separation when building the future missing-value branch; investigate any further treatment through validation.

Extreme count features follow the same retention policy. EDA maxima are 58 open credit lines/loans, 54 real-estate loans/lines and 20 dependents. These values remain unchanged; capping/winsorization requires later validation evidence and a documented rationale. Their presence does not authorize record removal.

## Class Imbalance

**Observed fact:** 10,026 of 150,000 observations have target 1 (6.684%).

**Decision:** compare an unresampled baseline, class weighting, and a SMOTE training-pipeline/fold option later. Never apply SMOTE before splitting and never resample validation/test. This is an experiment plan, not a claim that SMOTE is appropriate or beneficial for these mixed count/indicator inputs.

**Rationale:** preserving evaluation prevalence is necessary for meaningful risk-probability assessment. Training-fold-only resampling avoids using evaluation observations to generate training examples. See [imbalanced-learn's leakage guidance](https://imbalanced-learn.org/stable/common_pitfalls.html).

**Implementation timing:** inside later model-comparison pipelines after train-only preprocessing is defined. Synthetic fractional counts/indicators and effects on calibration must be reviewed. Model-specific class weighting, calibration and threshold selection use training/validation or group-disjoint folds, never the final test. No balancing is performed now.

## Split Strategy

### Selected method and alternatives

**Decision:** use **two-stage `GroupShuffleSplit`, with `RANDOM_STATE = 42` for each stage**, followed by predeclared size and target-distribution checks. It is group-disjoint random splitting with verified approximate class balance; it is **not an explicitly stratified algorithm**.

Stage 1 assigns 70% of unique groups to training and the remainder to a temporary holdout. Stage 2 assigns half of the holdout groups to validation and half to test. Float sizes apply to **groups**, so row counts need not be exactly 70/15/15. See [GroupShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html).

Before observing the assignment results, the engineering acceptance limits were set to:

- Absolute row-share deviation from the requested split percentage: at most **0.25 percentage points** per split.
- Absolute target-1-rate deviation from the full-data 6.684%: at most **0.25 percentage points** per split.
- Zero feature-group, row-position and source-index overlaps for every pair.
- Every observation assigned exactly once, and identical results on a repeated run with seed 42.

These are project acceptance checks, not statistical guarantees. The first seed-42 assignment passed. No alternate seeds were searched and no test predictive performance was used to select a split.

| Method | Assessment | Outcome |
|---|---|---|
| Two-stage GroupShuffleSplit + checks | Directly supports the desired group proportions and keeps each feature group intact. Does not optimize target stratification. | Selected: the observed row sizes and target rates satisfy the predeclared bounds. |
| StratifiedGroupKFold | Attempts class balance with non-overlapping groups. Combining predetermined folds, for example 14/3/3 out of 20, could approximate the requested proportions; group constraints still limit exactness. | Considered from its documented behavior, not benchmarked here. A reasonable alternative if the fixed group-random approach fails on a changed dataset. It is not needed to meet the current criteria. |
| Standard two-stage stratified row split | Gives exact row proportions and very close event rates, but does not preserve feature groups. | Tested as a comparator only; rejected for modeling because group overlap occurs. |

See [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html) for the alternative's class-balance objective. Do not describe this selected GroupShuffleSplit design as StratifiedGroupKFold or stratified sampling.

### Validated assignment results

All counts below are from in-memory assignments; no split files exist.

| Split | Rows | Share of all observations (%) | Target 1 count | Target 1 rate (%) | Unique feature groups | Event-rate difference from full data (pp) |
|---|---:|---:|---:|---:|---:|---:|
| Train | 104,998 | 69.998667 | 7,061 | 6.724890 | 104,547 | +0.040890 |
| Validation | 22,486 | 14.990667 | 1,478 | 6.572979 | 22,403 | -0.111021 |
| Test | 22,516 | 15.010667 | 1,487 | 6.604193 | 22,404 | -0.079807 |
| Total | 150,000 | 100.000000 | 10,026 | 6.684000 | 149,354 | 0.000000 |

The largest absolute size deviation is 0.010667 pp; the largest event-rate deviation is 0.111021 pp. Both are below the 0.25 pp limits.

### Reproduction contract for later implementation

Validated environment: Python 3.13.14, pandas 3.0.6, NumPy 2.5.3, scikit-learn 1.9.1. The following recipe documents the exact in-memory method; it performs no preprocessing or export. Run from the project root with its `.venv` when implementing the next step.

```python
from pathlib import Path
from itertools import combinations
import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

RANDOM_STATE = 42
path = Path("data/raw/cs-training.csv")
assert hashlib.sha256(path.read_bytes()).hexdigest() == (
    "1bd46da486a5708c58c7b01a034fae2a13b327f6f7b62ea7ba4fe3b5824b24ac"
)
df = pd.read_csv(path, float_precision="round_trip")
FEATURES = [
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
X = df.loc[:, FEATURES]
y = df["SeriousDlqin2yrs"].to_numpy()
rows = np.arange(len(df), dtype=np.int64)
groups = X.groupby(FEATURES, dropna=False, sort=False).ngroup().to_numpy(
    dtype=np.int64
)
assert len(np.unique(groups)) == 149354
train, holdout = next(GroupShuffleSplit(
    n_splits=1, train_size=0.70, random_state=RANDOM_STATE
).split(rows, groups=groups))
validation_local, test_local = next(GroupShuffleSplit(
    n_splits=1, test_size=0.50, random_state=RANDOM_STATE
).split(holdout, groups=groups[holdout]))
splits = {
    "Train": train,
    "Validation": holdout[validation_local],
    "Test": holdout[test_local],
}
assert np.array_equal(np.sort(np.concatenate(list(splits.values()))), rows)
for left, right in combinations(splits.values(), 2):
    assert np.intersect1d(left, right).size == 0
    assert np.intersect1d(groups[left], groups[right]).size == 0
    assert np.intersect1d(
        df.iloc[left]["Unnamed: 0"], df.iloc[right]["Unnamed: 0"]
    ).size == 0
for name, indices in splits.items():
    desired_pct = 70 if name == "Train" else 15
    assert abs(100 * len(indices) / len(df) - desired_pct) <= 0.25
    assert abs(100 * (y[indices].mean() - y.mean())) <= 0.25

# Fingerprint of the assignment in original row order; no file is written.
assignment = np.full(len(df), -1, dtype=np.int8)
for label, indices in enumerate(splits.values()):  # Train=0, Validation=1, Test=2
    assignment[indices] = label
assert hashlib.sha256(assignment.tobytes()).hexdigest() == (
    "c1061432589f1981dfce4a3bbe0ae5c55818767ea97e6b8b80962e9f9b7ec1f2"
)
```

The checksum, row order, grouping policy, library versions and seed jointly define reproducibility. Do not silently reuse acceptance counts for a new dataset or reroll until labels look desirable.

### Standard stratified comparison

The comparator uses `train_test_split(rows, test_size=0.30, random_state=42, stratify=y)` and then `train_test_split(holdout, test_size=0.50, random_state=42, stratify=y[holdout])`. It is not used for modeling.

| Split | Rows | Share (%) | Target 1 count | Target 1 rate (%) | Unique feature groups |
|---|---:|---:|---:|---:|---:|
| Train | 105,000 | 70.000000 | 7,018 | 6.683810 | 104,626 |
| Validation | 22,500 | 15.000000 | 1,504 | 6.684444 | 22,474 |
| Test | 22,500 | 15.000000 | 1,504 | 6.684444 | 22,473 |

| Pair | Selected method: shared feature groups | Standard stratified: shared feature groups | Shared row/source indices, both methods |
|---|---:|---:|---:|
| Train vs Validation | 0 | 102 | 0 |
| Train vs Test | 0 | 109 | 0 |
| Validation vs Test | 0 | 30 | 0 |

The standard split disperses **197 distinct groups containing 605 observations** across multiple partitions. Pairwise overlap counts must not be added to obtain a unique-group total: some groups appear in all three partitions. The selected method has zero cross-split feature groups and zero observations in such groups. Both methods assign all 150,000 row positions exactly once. This comparison establishes a difference in split integrity, not a measured change in model performance.

## Leakage Prevention

**Observed fact:** selected assignments have no feature-group, row-position or source-index overlap; group IDs are constructed without labels. Label access in this step is limited to reporting/predeclared distribution checks and stratifying the comparator, not creating groups or choosing preprocessing statistics.

**Decision and rationale:** enforce the following boundaries in future implementation:

1. Construct groups from the ten unmodified raw features before preprocessing. Keep group IDs and source indices out of model inputs.
2. Retain all observations, including inconsistent-target groups, in exactly one partition each. Training cross-validation must also preserve these feature groups; a group-disjoint outer split alone does not protect ordinary row-wise CV.
3. Fit imputation, scaling, any learned tail thresholds and resampling only on training/fitting-fold data. Apply fitted transforms to validation/test without refitting. Stateless, documented masks such as `age == 0` or 96/98 detection may be applied consistently without learning from evaluation data.
4. Do not resample validation/test. Keep their observed prevalence for evaluation. Select model settings, calibration strategy and classification thresholds through training/validation or appropriate group-disjoint CV; reserve the final test for the frozen model evaluation.
5. Verify coverage and overlap again when the split is later recreated. Keep explicit raw-row references for audit, not prediction. Future imputation can make previously distinct vectors look equal; this split guarantee concerns exact **raw** feature equality and does not identify borrower-level or approximate duplicates.
6. Protect raw files from writes and from Git tracking. Neither local `train.csv` / `test.csv` nor unlabeled Kaggle `cs-test.csv` substitutes for the internally designed labeled evaluation sets.

**Implementation timing:** the split checks are validated now; pipeline and modeling enforcement belong to later steps. No claim of universal absence of leakage is made from these checks alone.

## Limitations

- The earlier EDA examined the full labeled dataset. The planned internal test is group-disjoint, but it is not completely untouched by analyst inspection. Freeze decisions before modeling, avoid iterative decisions based on test outcomes, and disclose this limitation. An independent, previously uninspected labeled dataset would support stronger external validation.
- Feature groups are equality groups, not established borrower identities. Group isolation can be conservative when different borrowers share identical recorded attributes; neither deduplication nor borrower-level independence is established.
- GroupShuffleSplit does not guarantee stratification. Its observed event rates satisfy the declared bounds on this verified source only. A future source change requires reevaluation rather than seed hunting.
- This is a random group split, not a temporal or external validation design. No observation timestamp or documented borrower ID supports such claims here.
- No predictive model was evaluated. The standard-split comparison demonstrates feature-group overlap, not the magnitude of any resulting optimism.
- The meaning of 96/98, extreme ratios and some recording conventions remains unresolved. Pipeline representations are modeling decisions, not corrections of established source errors.
- No processed CSV, assignment array, model artifact or preprocessing object was saved. `data/processed/` still contains only `.gitkeep`. Raw source files, existing EDA outputs and dependencies were verified unchanged during this step.

Step 5 deliverable: this document only. No `git add`, commit or push was performed.
