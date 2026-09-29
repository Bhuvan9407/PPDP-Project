
"""
Reusable KC-Slice engine.

Milestone 4
-----------

This module converts the Milestone 3 Adult 1-SA execution
into a parameterized engine for the 54-run experiment matrix.

Paper-derived phases:
    1. Bucket creation using sensitive-attribute cardinalities.
    2. Sensitive-attribute privacy enforcement.
    3. Sensitive-cell grouping + SID generation.
    4. QID correlation, concatenation, and permutation.

Project operationalizations:
    - Deterministic tuple-to-bucket allocation.
    - Most-frequent-value HSA selection.
    - C-threshold suppression keeps the first allowed HSA
      occurrences in deterministic row order.
    - |Spearman| / Cramer's V / Eta association measures.
    - Correlation threshold = 0.40.
    - Greedy non-overlapping QID grouping.
    - Deterministic SID format.
    - Seeded QID-slice permutation.
    - Fractional internal k realized using balanced floor/ceil
      bucket sizes.

IMPORTANT:
    project_k_reference (default 10) is retained as the
    experiment's fixed k reference.

    KC-Slice's internal bucket size is derived from:
        internal_k = train_rows / sum(distinct SA cardinalities)

    It is NOT replaced by project_k_reference.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import math

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass(frozen=True)
class BucketSpecification:

    train_rows: int
    sensitive_attributes: Tuple[str, ...]

    sa_cardinalities: Dict[str, int]

    bucket_count: int

    internal_k: float

    base_bucket_size: int

    remainder: int

    min_bucket_size: int

    max_bucket_size: int


@dataclass
class KCSliceResult:

    parameters: Dict

    bucketed_data: pd.DataFrame

    privacy_report: pd.DataFrame

    sensitive_cells: pd.DataFrame

    row_sid_links: pd.DataFrame

    correlation_audit: pd.DataFrame

    qid_grouping: pd.DataFrame

    bqa_internal: pd.DataFrame

    permutation_log: pd.DataFrame


# ============================================================
# VALIDATION
# ============================================================

def validate_configuration(
    df: pd.DataFrame,
    sensitive_attributes: Sequence[str],
    qid_columns: Sequence[str],
) -> None:

    if df.empty:
        raise ValueError(
            "KC-Slice input dataset is empty."
        )

    if not sensitive_attributes:
        raise ValueError(
            "At least one sensitive attribute is required."
        )

    if not qid_columns:
        raise ValueError(
            "At least one QID is required."
        )

    missing_sa = [
        col
        for col in sensitive_attributes
        if col not in df.columns
    ]

    missing_qid = [
        col
        for col in qid_columns
        if col not in df.columns
    ]

    if missing_sa:
        raise ValueError(
            f"Missing sensitive attributes: {missing_sa}"
        )

    if missing_qid:
        raise ValueError(
            f"Missing QIDs: {missing_qid}"
        )

    overlap = (
        set(sensitive_attributes)
        &
        set(qid_columns)
    )

    if overlap:
        raise ValueError(
            "Sensitive/QID overlap detected: "
            f"{sorted(overlap)}"
        )

    if len(set(sensitive_attributes)) != len(
        sensitive_attributes
    ):
        raise ValueError(
            "Sensitive attributes contain duplicates."
        )

    if len(set(qid_columns)) != len(
        qid_columns
    ):
        raise ValueError(
            "QIDs contain duplicates."
        )


# ============================================================
# BUCKET PARAMETER CALCULATION
# ============================================================

def calculate_bucket_spec(
    df: pd.DataFrame,
    sensitive_attributes: Sequence[str],
) -> BucketSpecification:

    cardinalities = {
        sa: int(
            df[sa]
            .astype(str)
            .nunique()
        )
        for sa in sensitive_attributes
    }

    bucket_count = sum(
        cardinalities.values()
    )

    if bucket_count <= 0:
        raise ValueError(
            "Derived bucket count must be positive."
        )

    train_rows = len(df)

    internal_k = (
        train_rows /
        bucket_count
    )

    base_bucket_size = (
        train_rows //
        bucket_count
    )

    remainder = (
        train_rows %
        bucket_count
    )

    min_bucket_size = (
        base_bucket_size
    )

    max_bucket_size = (
        base_bucket_size
        +
        (1 if remainder else 0)
    )

    return BucketSpecification(
        train_rows=train_rows,
        sensitive_attributes=tuple(
            sensitive_attributes
        ),
        sa_cardinalities=cardinalities,
        bucket_count=bucket_count,
        internal_k=internal_k,
        base_bucket_size=base_bucket_size,
        remainder=remainder,
        min_bucket_size=min_bucket_size,
        max_bucket_size=max_bucket_size,
    )


def balanced_bucket_sizes(
    train_rows: int,
    bucket_count: int,
) -> List[int]:

    base = (
        train_rows //
        bucket_count
    )

    remainder = (
        train_rows %
        bucket_count
    )

    sizes = [
        base +
        (1 if bucket_id < remainder else 0)
        for bucket_id in range(bucket_count)
    ]

    assert sum(sizes) == train_rows

    assert (
        max(sizes) - min(sizes)
        <= 1
    )

    return sizes


# ============================================================
# HSA SELECTION
# ============================================================

def select_hsa_values(
    df: pd.DataFrame,
    sensitive_attributes: Sequence[str],
) -> Dict[str, str]:

    hsa_values = {}

    for sa in sensitive_attributes:

        series = (
            df[sa]
            .astype(str)
        )

        counts = (
            series
            .value_counts(
                sort=True
            )
        )

        if counts.empty:
            raise ValueError(
                f"No values available for SA: {sa}"
            )

        # value_counts preserves deterministic first occurrence
        # ordering for ties in the source/project environment.
        hsa_values[sa] = str(
            counts.index[0]
        )

    return hsa_values


# ============================================================
# BUCKET ASSIGNMENT
# ============================================================

def _assign_one_sa_frequency_balanced(
    df: pd.DataFrame,
    sensitive_attribute: str,
    bucket_count: int,
) -> np.ndarray:

    """
    Deterministic one-SA bucket assignment.

    Sensitive values are processed in first-appearance order.
    Within each sensitive value, records are assigned to buckets
    using one global round-robin cursor.

    This guarantees:

        1. Each sensitive value is distributed across buckets
           as evenly as possible.

        2. Overall bucket sizes differ by at most one.

    This is a project operationalization of the bucket creation
    step because the source does not specify a unique
    tuple-to-bucket assignment procedure.
    """

    if bucket_count <= 0:
        raise ValueError(
            "bucket_count must be positive."
        )

    bucket_ids = np.full(
        len(df),
        -1,
        dtype=np.int64,
    )

    work = pd.DataFrame({
        "__position": np.arange(
            len(df)
        ),
        "__value": (
            df[sensitive_attribute]
            .astype(str)
            .to_numpy()
        ),
    })

    # Preserve deterministic first-appearance ordering.
    value_order = list(
        dict.fromkeys(
            work["__value"].tolist()
        )
    )

    global_cursor = 0

    for value in value_order:

        positions = work.loc[
            work["__value"] == value,
            "__position"
        ].to_numpy()

        # Global round-robin assignment.
        for offset, position in enumerate(
            positions
        ):

            bucket_ids[
                position
            ] = (
                global_cursor
                +
                offset
            ) % bucket_count

        global_cursor = (
            global_cursor
            +
            len(positions)
        ) % bucket_count

    if (
        bucket_ids < 0
    ).any():

        raise RuntimeError(
            "Some records were not assigned "
            "to a bucket."
        )

    return bucket_ids

def _assign_multi_sa_composite_stratified(
    df: pd.DataFrame,
    sensitive_attributes: Sequence[str],
    bucket_count: int,
) -> np.ndarray:

    """
    Multi-SA deterministic extension.

    For each SA column, each value group is converted into a
    stable within-value percentile. The mean percentile across
    all SA columns gives each record a composite sensitive
    stratification score.

    Records are then sorted by that score and assigned to
    exactly balanced bucket sizes.

    This is a project operationalization because the paper
    does not prescribe a unique tuple-to-bucket assignment
    procedure.
    """

    work = pd.DataFrame(
        index=np.arange(len(df))
    )

    work["__source_position"] = np.arange(
        len(df)
    )

    scores = np.zeros(
        len(df),
        dtype=np.float64,
    )

    for sa in sensitive_attributes:

        series = (
            df[sa]
            .astype(str)
            .reset_index(drop=True)
        )

        group_size = (
            series
            .groupby(
                series,
                sort=False
            )
            .transform("size")
            .to_numpy()
        )

        position = (
            series
            .groupby(
                series,
                sort=False
            )
            .cumcount()
            .to_numpy()
        )

        denominator = np.maximum(
            group_size - 1,
            1
        )

        percentile = (
            position /
            denominator
        )

        percentile[
            group_size == 1
        ] = 0.0

        scores += percentile

    scores /= len(
        sensitive_attributes
    )

    work["__score"] = scores

    # Stable tie breaking by original row position.
    work = work.sort_values(
        [
            "__score",
            "__source_position",
        ],
        kind="mergesort",
    )

    sizes = balanced_bucket_sizes(
        len(df),
        bucket_count,
    )

    output = np.full(
        len(df),
        -1,
        dtype=np.int64,
    )

    start = 0

    for bucket_id, size in enumerate(
        sizes
    ):

        end = start + size

        positions = (
            work.iloc[
                start:end
            ]["__source_position"]
            .to_numpy()
        )

        output[
            positions
        ] = bucket_id

        start = end

    if (
        output < 0
    ).any():

        raise RuntimeError(
            "Some records were not assigned "
            "to a multi-SA bucket."
        )

    return output


def assign_buckets(
    df: pd.DataFrame,
    sensitive_attributes: Sequence[str],
    bucket_count: int,
) -> pd.Series:

    if len(sensitive_attributes) == 1:

        assignments = (
            _assign_one_sa_frequency_balanced(
                df,
                sensitive_attributes[0],
                bucket_count,
            )
        )

    else:

        assignments = (
            _assign_multi_sa_composite_stratified(
                df,
                sensitive_attributes,
                bucket_count,
            )
        )

    result = pd.Series(
        assignments,
        index=df.index,
        name="_kc_bucket",
    )

    counts = (
        result
        .value_counts()
        .sort_index()
    )

    expected_sizes = balanced_bucket_sizes(
        len(df),
        bucket_count,
    )

    actual_sizes = [
        int(
            counts.loc[bucket_id]
        )
        for bucket_id in range(
            bucket_count
        )
    ]

    # For the one-SA strategy, independent remainder
    # allocation can still produce the same balanced sizes
    # when the derived configuration permits it. We require
    # only a one-record spread.
    assert (
        max(actual_sizes) -
        min(actual_sizes)
        <= 1
    )

    assert sum(actual_sizes) == len(df)

    if (
        sorted(actual_sizes)
        !=
        sorted(expected_sizes)
    ):

        # This guard catches accidental bucket imbalance
        # without assuming which particular bucket receives
        # a remainder record.
        raise AssertionError(
            "Bucket sizes do not match the "
            "balanced floor/ceil realization."
        )

    return result


# ============================================================
# PRIVACY ENFORCEMENT
# ============================================================

def apply_privacy_check(
    df: pd.DataFrame,
    sensitive_attributes: Sequence[str],
    hsa_values: Dict[str, str],
    privacy_c: float,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    if not (
        0 < privacy_c <= 100
    ):
        raise ValueError(
            "privacy_c must be in (0, 100]."
        )

    processed = df.copy()

    records = []

    bucket_ids = sorted(
        processed["_kc_bucket"]
        .unique()
    )

    for bucket_id in bucket_ids:

        bucket_mask = (
            processed["_kc_bucket"]
            ==
            bucket_id
        )

        bucket_positions = list(
            processed.index[
                bucket_mask
            ]
        )

        bucket_size = len(
            bucket_positions
        )

        for sa in sensitive_attributes:

            hsa = str(
                hsa_values[sa]
            )

            hsa_mask = (
                processed.loc[
                    bucket_positions,
                    sa,
                ]
                .astype(str)
                .eq(hsa)
            )

            hsa_count = int(
                hsa_mask.sum()
            )

            before_percent = (
                hsa_count /
                bucket_size *
                100
            )

            # The locked Milestone-3 implementation keeps
            # floor(C% of bucket_size) HSA records.
            allowed_count = int(
                math.floor(
                    privacy_c /
                    100 *
                    bucket_size
                )
            )

            suppress_count = max(
                0,
                hsa_count -
                allowed_count,
            )

            hsa_positions = list(
                pd.Index(
                    bucket_positions
                )[hsa_mask.to_numpy()]
            )

            # Deterministic suppression:
            # retain the first allowed HSA positions and
            # suppress every subsequent HSA occurrence.
            suppressed_positions = (
                hsa_positions[
                    allowed_count:
                ]
            )

            if suppressed_positions:

                processed.loc[
                    suppressed_positions,
                    sa,
                ] = "#####"

            after_count = int(
                (
                    processed.loc[
                        bucket_positions,
                        sa,
                    ]
                    .astype(str)
                    .eq(hsa)
                ).sum()
            )

            after_percent = (
                after_count /
                bucket_size *
                100
            )

            assert (
                after_percent
                <=
                privacy_c +
                1e-12
            )

            assert (
                len(suppressed_positions)
                ==
                suppress_count
            )

            records.append({
                "bucket_id":
                    int(bucket_id),

                "bucket":
                    int(bucket_id) + 1,

                "attribute":
                    sa,

                "hsa":
                    hsa,

                "bucket_size":
                    bucket_size,

                "before_count":
                    hsa_count,

                "before_percent":
                    before_percent,

                "allowed_count":
                    allowed_count,

                "suppressed_count":
                    suppress_count,

                "after_count":
                    after_count,

                "after_percent":
                    after_percent,
            })

    privacy_report = pd.DataFrame(
        records
    )

    return (
        processed,
        privacy_report,
    )


# ============================================================
# ASSOCIATION MEASURES
# ============================================================

def cramers_v(
    x: pd.Series,
    y: pd.Series,
) -> float:

    table = pd.crosstab(
        x.astype(str),
        y.astype(str),
    )

    if table.empty:
        return 0.0

    observed = table.to_numpy(
        dtype=float
    )

    n = observed.sum()

    if n <= 0:
        return 0.0

    row_sums = observed.sum(
        axis=1
    )

    col_sums = observed.sum(
        axis=0
    )

    expected = np.outer(
        row_sums,
        col_sums,
    ) / n

    mask = (
        expected > 0
    )

    chi2 = (
        (
            observed[mask]
            -
            expected[mask]
        ) ** 2
        /
        expected[mask]
    ).sum()

    phi2 = (
        chi2 /
        n
    )

    r, k = observed.shape

    denominator = max(
        1,
        min(
            k - 1,
            r - 1,
        ),
    )

    return float(
        np.sqrt(
            phi2 /
            denominator
        )
    )


def correlation_ratio(
    categories: pd.Series,
    values: pd.Series,
) -> float:

    categories = (
        pd.Series(categories)
        .astype(str)
    )

    values = pd.to_numeric(
        pd.Series(values),
        errors="coerce",
    )

    valid = (
        categories.notna()
        &
        values.notna()
    )

    categories = categories[
        valid
    ]

    values = values[
        valid
    ]

    if len(values) == 0:
        return 0.0

    grand_mean = values.mean()

    numerator = 0.0

    for category in (
        categories.unique()
    ):

        group = values[
            categories == category
        ]

        numerator += (
            len(group)
            *
            (
                group.mean()
                -
                grand_mean
            ) ** 2
        )

    denominator = (
        (
            values
            -
            grand_mean
        ) ** 2
    ).sum()

    if denominator == 0:
        return 0.0

    return float(
        np.sqrt(
            numerator /
            denominator
        )
    )


def association(
    x: pd.Series,
    y: pd.Series,
    x_name: str,
    y_name: str,
    numeric_qids: Sequence[str],
) -> Tuple[float, str]:

    x_numeric = (
        x_name in numeric_qids
    )

    y_numeric = (
        y_name in numeric_qids
    )

    if x_numeric and y_numeric:

        x_num = pd.to_numeric(
            x,
            errors="coerce",
        )

        y_num = pd.to_numeric(
            y,
            errors="coerce",
        )

        valid = (
            x_num.notna()
            &
            y_num.notna()
        )

        if valid.sum() < 2:
            return 0.0, "Spearman"

        rho, _ = spearmanr(
            x_num[valid],
            y_num[valid],
        )

        if pd.isna(rho):
            return 0.0, "Spearman"

        return (
            abs(float(rho)),
            "Spearman",
        )

    if (
        not x_numeric
        and
        not y_numeric
    ):

        return (
            cramers_v(x, y),
            "Cramer's V",
        )

    if x_numeric:

        value = x
        category = y

    else:

        value = y
        category = x

    return (
        correlation_ratio(
            category,
            value,
        ),
        "Eta",
    )


# ============================================================
# QID CORRELATION
# ============================================================

def correlate_qids(
    df: pd.DataFrame,
    qid_columns: Sequence[str],
    numeric_qids: Sequence[str],
    threshold: float,
) -> Tuple[
    pd.DataFrame,
    Dict[int, List[List[str]]],
]:

    if not (
        0 <= threshold <= 1
    ):
        raise ValueError(
            "Correlation threshold must be in [0, 1]."
        )

    audit_records = []
    grouping_by_bucket = {}

    for bucket_id in sorted(
        df["_kc_bucket"].unique()
    ):

        bucket = df[
            df["_kc_bucket"]
            ==
            bucket_id
        ].copy()

        lookup = {}

        for qid_a, qid_b in combinations(
            qid_columns,
            2,
        ):

            value, measure = association(
                bucket[qid_a],
                bucket[qid_b],
                qid_a,
                qid_b,
                numeric_qids,
            )

            record = {
                "bucket_id":
                    int(bucket_id),

                "bucket":
                    int(bucket_id) + 1,

                "qid_a":
                    qid_a,

                "qid_b":
                    qid_b,

                "measure":
                    measure,

                "association":
                    float(value),

                "threshold":
                    float(threshold),

                "correlated":
                    bool(
                        value >= threshold
                    ),
            }

            audit_records.append(
                record
            )

            lookup[
                frozenset(
                    [
                        qid_a,
                        qid_b,
                    ]
                )
            ] = float(value)

        # ----------------------------------------------------
        # Locked deterministic greedy grouping.
        # ----------------------------------------------------

        unused = list(
            qid_columns
        )

        groups = []

        while unused:

            anchor = unused.pop(0)

            group = [
                anchor
            ]

            remaining = []

            for candidate in unused:

                pair = frozenset(
                    [
                        anchor,
                        candidate,
                    ]
                )

                value = lookup.get(
                    pair,
                    0.0,
                )

                if value >= threshold:

                    group.append(
                        candidate
                    )

                else:

                    remaining.append(
                        candidate
                    )

            unused = remaining

            groups.append(
                group
            )

        flat = [
            qid
            for group in groups
            for qid in group
        ]

        assert (
            flat == list(
                dict.fromkeys(flat)
            )
        )

        assert (
            set(flat)
            ==
            set(qid_columns)
        )

        grouping_by_bucket[
            int(bucket_id)
        ] = groups

    correlation_audit = pd.DataFrame(
        audit_records
    )

    return (
        correlation_audit,
        grouping_by_bucket,
    )


# ============================================================
# SID GENERATION
# ============================================================

def generate_sid(
    bucket_number: int,
    sa_number: int,
    bucket_count: int,
    sa_count: int,
) -> str:

    bucket_width = max(
        2,
        len(
            str(bucket_count)
        ),
    )

    sa_width = max(
        1,
        len(
            str(sa_count)
        ),
    )

    return (
        f"{bucket_number:0{bucket_width}d}"
        f"{sa_number:0{sa_width}d}"
    )


# ============================================================
# SENSITIVE CELL TABLES
# ============================================================

def build_sensitive_cells(
    df: pd.DataFrame,
    sensitive_attributes: Sequence[str],
    bucket_count: int,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    Dict[Tuple[int, str], str],
]:

    sensitive_records = []
    row_sid_records = []

    sid_lookup = {}

    for bucket_id in sorted(
        df["_kc_bucket"].unique()
    ):

        bucket_number = (
            int(bucket_id)
            +
            1
        )

        bucket = df[
            df["_kc_bucket"]
            ==
            bucket_id
        ].copy()

        for sa_number, sa in enumerate(
            sensitive_attributes,
            start=1,
        ):

            sid = generate_sid(
                bucket_number,
                sa_number,
                bucket_count,
                len(
                    sensitive_attributes
                ),
            )

            sid_lookup[
                (
                    int(bucket_id),
                    sa,
                )
            ] = sid

            values = (
                bucket[sa]
                .astype(str)
            )

            counts = (
                values
                .value_counts(
                    sort=False
                )
            )

            ordered_counts = OrderedDict(
                (
                    str(value),
                    int(count)
                )
                for value, count
                in counts.items()
            )

            cell_text = ", ".join(
                f"{value}({count})"
                for value, count
                in ordered_counts.items()
            )

            sensitive_records.append({
                "bucket":
                    bucket_number,

                "bucket_id":
                    int(bucket_id),

                "SID":
                    sid,

                "SA":
                    sa,

                "cell":
                    cell_text,

                "cell_record_count":
                    int(
                        sum(
                            ordered_counts.values()
                        )
                    ),
            })

            for row_position in bucket.index:

                source_row = (
                    df.loc[
                        row_position,
                        "__kc_source_row",
                    ]
                )

                row_sid_records.append({
                    "bucket":
                        bucket_number,

                    "bucket_id":
                        int(bucket_id),

                    "row_index":
                        source_row,

                    "SA":
                        sa,

                    "SID":
                        sid,
                })

    sensitive_cells = pd.DataFrame(
        sensitive_records
    )

    row_sid_links = pd.DataFrame(
        row_sid_records
    )

    return (
        sensitive_cells,
        row_sid_links,
        sid_lookup,
    )


# ============================================================
# QID SLICE TABLES + PERMUTATION
# ============================================================

def build_qid_slices(
    df: pd.DataFrame,
    qid_columns: Sequence[str],
    grouping_by_bucket: Dict[int, List[List[str]]],
    sid_lookup: Dict[Tuple[int, str], str],
    sensitive_attributes: Sequence[str],
    seed: int,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    all_bucket_tables = []
    permutation_records = []

    for bucket_id in sorted(
        df["_kc_bucket"].unique()
    ):

        bucket = df[
            df["_kc_bucket"]
            ==
            bucket_id
        ].copy()

        groups = grouping_by_bucket[
            int(bucket_id)
        ]

        bucket_number = (
            int(bucket_id)
            +
            1
        )

        # ----------------------------------------------------
        # All SIDs for this bucket.
        #
        # For 1-SA this produces exactly the Milestone-3
        # string: SID=<sid>.
        #
        # For multiple SAs, every sensitive-column SID is
        # attached because Bsa contains one SID per
        # bucket/SA cell.
        # ----------------------------------------------------

        sid_values = [
            sid_lookup[
                (
                    int(bucket_id),
                    sa,
                )
            ]
            for sa in sensitive_attributes
        ]

        bucket_table = pd.DataFrame(
            index=bucket.index
        )

        bucket_table[
            "_source_row"
        ] = bucket[
            "__kc_source_row"
        ].to_numpy()

        bucket_table[
            "bucket"
        ] = bucket_number

        bucket_table[
            "bucket_id"
        ] = int(bucket_id)

        group_names = []

        for group_number, group in enumerate(
            groups,
            start=1,
        ):

            group_name = (
                f"slice_{group_number}"
            )

            group_names.append(
                group_name
            )

            # Vectorized concatenation.
            values = (
                bucket[
                    list(group)
                ]
                .astype(str)
                .agg(
                    " | ".join,
                    axis=1,
                )
            )

            for sid in sid_values:

                values = (
                    values
                    + f" | SID={sid}"
                )

            bucket_table[
                group_name
            ] = values.to_numpy()

        rng = np.random.default_rng(
            int(seed)
            +
            bucket_number
        )

        permutation = rng.permutation(
            len(group_names)
        )

        permuted_group_names = [
            group_names[i]
            for i in permutation
        ]

        for output_position, original_index in enumerate(
            permutation,
            start=1,
        ):

            permutation_records.append({
                "bucket":
                    bucket_number,

                "bucket_id":
                    int(bucket_id),

                "output_position":
                    output_position,

                "original_group":
                    group_names[
                        original_index
                    ],
            })

        final_columns = [
            "_source_row",
            "bucket",
            "bucket_id",
        ] + permuted_group_names

        final_bucket = bucket_table[
            final_columns
        ].copy()

        all_bucket_tables.append(
            final_bucket
        )

    bqa_internal = pd.concat(
        all_bucket_tables,
        ignore_index=True,
    )

    permutation_log = pd.DataFrame(
        permutation_records
    )

    assert (
        len(bqa_internal)
        ==
        len(df)
    )

    source_counts = (
        bqa_internal[
            "_source_row"
        ]
        .value_counts()
    )

    assert (
        source_counts == 1
    ).all()

    return (
        bqa_internal,
        permutation_log,
    )


# ============================================================
# MAIN ENGINE
# ============================================================

def run_kc_slice(
    df: pd.DataFrame,
    sensitive_attributes: Sequence[str],
    qid_columns: Sequence[str],
    *,
    seed: int = 1,
    project_k_reference: int = 10,
    privacy_c: float = 5.0,
    correlation_threshold: float = 0.40,
    numeric_qids: Optional[Sequence[str]] = None,
) -> KCSliceResult:

    """
    Execute one parameterized KC-Slice run.

    Input:
        df
            Training data for the current experiment.

        sensitive_attributes
            Ordered SA columns from the experiment matrix.

        qid_columns
            Ordered QID columns from the experiment matrix.

        seed
            Main-run seed.

        project_k_reference
            Fixed project k reference (=10).

        privacy_c
            Privacy threshold C in percent.

        correlation_threshold
            Operational Correlate() threshold.

        numeric_qids
            QIDs treated as numeric for association selection.

    Returns:
        KCSliceResult containing all auditable intermediate
        and final tables.
    """

    validate_configuration(
        df,
        sensitive_attributes,
        qid_columns,
    )

    if numeric_qids is None:

        numeric_qids = []

    unknown_numeric = (
        set(numeric_qids)
        -
        set(qid_columns)
    )

    if unknown_numeric:

        raise ValueError(
            "numeric_qids contains non-QIDs: "
            f"{sorted(unknown_numeric)}"
        )

    # --------------------------------------------------------
    # Preserve source-row identity.
    # --------------------------------------------------------

    work = df.copy()

    work["__kc_source_row"] = (
        work.index.to_numpy()
    )

    work = work.reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Calculate KC-Slice bucket parameters.
    # --------------------------------------------------------

    spec = calculate_bucket_spec(
        work,
        sensitive_attributes,
    )

    # --------------------------------------------------------
    # HSA selection.
    # --------------------------------------------------------

    hsa_values = select_hsa_values(
        work,
        sensitive_attributes,
    )

    # --------------------------------------------------------
    # Bucket creation.
    # --------------------------------------------------------

    bucket_series = assign_buckets(
        work,
        sensitive_attributes,
        spec.bucket_count,
    )

    work[
        "_kc_bucket"
    ] = bucket_series.to_numpy()

    bucket_sizes = (
        work[
            "_kc_bucket"
        ]
        .value_counts()
        .sort_index()
    )

    assert len(
        bucket_sizes
    ) == spec.bucket_count

    assert (
        max(bucket_sizes)
        -
        min(bucket_sizes)
        <= 1
    )

    # --------------------------------------------------------
    # Phase 2 privacy enforcement.
    # --------------------------------------------------------

    processed, privacy_report = (
        apply_privacy_check(
            work,
            sensitive_attributes,
            hsa_values,
            privacy_c,
        )
    )

    # --------------------------------------------------------
    # Phase 3A sensitive cells + SID.
    # --------------------------------------------------------

    (
        sensitive_cells,
        row_sid_links,
        sid_lookup,
    ) = build_sensitive_cells(
        processed,
        sensitive_attributes,
        spec.bucket_count,
    )

    # --------------------------------------------------------
    # Phase 3B QID correlation + grouping.
    # --------------------------------------------------------

    (
        correlation_audit,
        grouping_by_bucket,
    ) = correlate_qids(
        processed,
        qid_columns,
        numeric_qids,
        correlation_threshold,
    )

    grouping_records = []

    for bucket_id, groups in (
        grouping_by_bucket.items()
    ):

        for group_number, group in enumerate(
            groups,
            start=1,
        ):

            if len(group) == 1:

                grouping_records.append({
                    "bucket_id":
                        bucket_id,

                    "bucket":
                        bucket_id + 1,

                    "group_id":
                        group_number,

                    "group":
                        group[0],

                    "group_size":
                        1,
                })

            else:

                grouping_records.append({
                    "bucket_id":
                        bucket_id,

                    "bucket":
                        bucket_id + 1,

                    "group_id":
                        group_number,

                    "group":
                        " + ".join(group),

                    "group_size":
                        len(group),
                })

    qid_grouping = pd.DataFrame(
        grouping_records
    )

    # --------------------------------------------------------
    # Phase 3B QID concatenation + permutation.
    # --------------------------------------------------------

    (
        bqa_internal,
        permutation_log,
    ) = build_qid_slices(
        processed,
        qid_columns,
        grouping_by_bucket,
        sid_lookup,
        sensitive_attributes,
        seed,
    )

    # --------------------------------------------------------
    # Final assertions.
    # --------------------------------------------------------

    assert len(
        processed
    ) == len(df)

    assert len(
        sensitive_cells
    ) == (
        spec.bucket_count
        *
        len(sensitive_attributes)
    )

    assert len(
        bqa_internal
    ) == len(df)

    assert (
        bqa_internal[
            "_source_row"
        ]
        .nunique()
        ==
        len(df)
    )

    # Every SA receives one SID in every bucket.
    assert (
        sensitive_cells
        .groupby(
            [
                "bucket_id",
                "SA",
            ]
        )["SID"]
        .nunique()
        .eq(1)
        .all()
    )

    # --------------------------------------------------------
    # Parameter manifest.
    # --------------------------------------------------------

    parameters = {

        "project_k_reference":
            int(project_k_reference),

        "seed":
            int(seed),

        "train_rows":
            int(spec.train_rows),

        "sensitive_attributes":
            list(
                sensitive_attributes
            ),

        "qid_columns":
            list(
                qid_columns
            ),

        "numeric_qids":
            list(
                numeric_qids
            ),

        "sa_cardinalities":
            {
                key: int(value)
                for key, value
                in spec.sa_cardinalities.items()
            },

        "bucket_count":
            int(spec.bucket_count),

        "internal_k":
            float(spec.internal_k),

        "base_bucket_size":
            int(
                spec.base_bucket_size
            ),

        "remainder":
            int(spec.remainder),

        "min_bucket_size":
            int(spec.min_bucket_size),

        "max_bucket_size":
            int(spec.max_bucket_size),

        "privacy_c":
            float(privacy_c),

        "hsa_values":
            dict(hsa_values),

        "correlation_threshold":
            float(correlation_threshold),

        "bucket_allocation":
            (
                "one-SA frequency-balanced "
                "value-wise allocation"
                if len(sensitive_attributes) == 1
                else
                "multi-SA composite percentile "
                "stratification with balanced "
                "floor/ceil bucket sizes"
            ),

        "qid_grouping":
            (
                "deterministic greedy "
                "non-overlapping grouping"
            ),

        "qid_permutation":
            (
                "numpy default_rng(seed + "
                "bucket_number) per bucket"
            ),
    }

    # --------------------------------------------------------
    # Remove only internal source marker from the public-ish
    # bucketed table? No. Keep it in the result because it is
    # required for audit/evaluation.
    # --------------------------------------------------------

    return KCSliceResult(
        parameters=parameters,
        bucketed_data=processed,
        privacy_report=privacy_report,
        sensitive_cells=sensitive_cells,
        row_sid_links=row_sid_links,
        correlation_audit=correlation_audit,
        qid_grouping=qid_grouping,
        bqa_internal=bqa_internal,
        permutation_log=permutation_log,
    )


__all__ = [
    "BucketSpecification",
    "KCSliceResult",
    "balanced_bucket_sizes",
    "calculate_bucket_spec",
    "select_hsa_values",
    "assign_buckets",
    "apply_privacy_check",
    "cramers_v",
    "correlation_ratio",
    "association",
    "correlate_qids",
    "generate_sid",
    "build_sensitive_cells",
    "build_qid_slices",
    "run_kc_slice",
]
