"""
KC-SLICE IMPLEMENTATION — MILESTONE 3

This file is an auditable source extraction from the executed
Milestone 3 Colab notebook.

The code below is preserved from the notebook execution history
rather than rewritten. The source cell numbers are included so
the implementation can be traced back to the experimental run.
"""


########################################################################
# SOURCE NOTEBOOK EXECUTION CELL 33
########################################################################

# Cell 33 — KC-SLICE ADULT 1-SA
# PHASE 3A: SENSITIVE CELL GROUPING + SID GENERATION

from pathlib import Path
from collections import OrderedDict
import pandas as pd

ROOT = Path("/content/kc-slice")

INPUT = (
    ROOT /
    "results/kc_slice_adult_1sa/"
    "adult_1sa_bucket_privacy_output.csv"
)

OUT_DIR = (
    ROOT /
    "results/kc_slice_adult_1sa"
)

df = pd.read_csv(INPUT)

SA_COLUMNS = ["income"]
BUCKET_COLUMN = "_kc_bucket"

print("=" * 70)
print("KC-SLICE — ADULT 1-SA")
print("PHASE 3A: SENSITIVE CELL GROUPING + SID")
print("=" * 70)

print(f"Input rows            : {len(df):,}")
print(f"Sensitive attributes  : {SA_COLUMNS}")

bucket_ids = sorted(
    df[BUCKET_COLUMN].unique()
)

print(f"Buckets               : {len(bucket_ids)}")


# ============================================================
# Generate one SID for each SA cell within each bucket.
#
# Paper's Algorithm 2:
#   7. Select SA column from bucket
#   8. Generate SID for the cell
#   9. Cluster SID with the cell
#  12. Merge the SA cells
#
# We use deterministic SIDs:
#
#   Bucket 1, SA 1 -> 011
#   Bucket 2, SA 1 -> 021
#
# The first digit identifies the bucket and the remaining
# digits identify the SA column.
# ============================================================

sensitive_tables = []
row_sid_records = []

for bucket_number, bucket_id in enumerate(
    bucket_ids,
    start=1
):

    bucket = df[
        df[BUCKET_COLUMN] == bucket_id
    ].copy()

    print()
    print("-" * 70)
    print(f"BUCKET {bucket_number}")
    print("-" * 70)

    for sa_index, sa in enumerate(
        SA_COLUMNS,
        start=1
    ):

        # ----------------------------------------------------
        # One SID for the complete SA cell.
        # ----------------------------------------------------

        sid = (
            f"{bucket_number:02d}"
            f"{sa_index:01d}"
        )

        # ----------------------------------------------------
        # Count every resulting sensitive value, including
        # suppressed values.
        # ----------------------------------------------------

        counts = (
            bucket[sa]
            .astype(str)
            .value_counts()
        )

        # Preserve deterministic ordering.
        counts = OrderedDict(
            (
                str(value),
                int(count)
            )
            for value, count
            in counts.items()
        )

        # ----------------------------------------------------
        # Represent the cell exactly like the paper's
        # illustration:
        #
        # value(count), value(count), ...
        # ----------------------------------------------------

        cell_parts = [
            f"{value}({count})"
            for value, count
            in counts.items()
        ]

        cell_text = ", ".join(
            cell_parts
        )

        sensitive_tables.append({
            "bucket": bucket_number,
            "bucket_id": bucket_id,
            "SID": sid,
            "SA": sa,
            "cell": cell_text,
            "cell_record_count": sum(
                counts.values()
            ),
        })

        # ----------------------------------------------------
        # Link every original record in this bucket to the
        # single SID representing this SA cell.
        # ----------------------------------------------------

        for original_index in bucket["_kc_bucket"].index:

            row_sid_records.append({
                "bucket": bucket_number,
                "bucket_id": bucket_id,
                "row_index": original_index,
                "SA": sa,
                "SID": sid,
            })

        print(f"SA                      : {sa}")
        print(f"SID                     : {sid}")

        for value, count in counts.items():

            pct = (
                count /
                len(bucket) *
                100
            )

            print(
                f"  {value:<10} "
                f"{count:>7,} "
                f"({pct:>7.3f}%)"
            )

        print()
        print(f"Cell representation:")
        print(f"  {cell_text}")


# ============================================================
# Build grouped sensitive table.
# ============================================================

sensitive_table = pd.DataFrame(
    sensitive_tables
)

row_sid_table = pd.DataFrame(
    row_sid_records
)


# ============================================================
# Verification
# ============================================================

print()
print("=" * 70)
print("VERIFICATION")
print("=" * 70)

# For each bucket + SA, cell count must equal bucket size.
expected_cell_rows = (
    len(bucket_ids) *
    len(SA_COLUMNS)
)

assert len(sensitive_table) == expected_cell_rows

for _, row in sensitive_table.iterrows():

    bucket_size = len(
        df[
            df[BUCKET_COLUMN]
            ==
            row["bucket_id"]
        ]
    )

    assert (
        row["cell_record_count"]
        ==
        bucket_size
    )

print(
    "Sensitive-cell record accounting : PASS"
)

# Every bucket/SA combination must have one SID.
assert (
    sensitive_table
    .groupby(
        ["bucket_id", "SA"]
    )["SID"]
    .nunique()
    .eq(1)
    .all()
)

print(
    "One SID per bucket/SA cell      : PASS"
)

# Every source row must receive exactly one SID
# because this is a 1-SA baseline.
expected_sid_links = len(df)

assert len(row_sid_table) == expected_sid_links

print(
    "Row-to-SID accounting            : PASS"
)


# ============================================================
# Save results.
# ============================================================

SA_OUTPUT = (
    OUT_DIR /
    "adult_1sa_sensitive_cells.csv"
)

SID_OUTPUT = (
    OUT_DIR /
    "adult_1sa_row_sid_links.csv"
)

sensitive_table.to_csv(
    SA_OUTPUT,
    index=False
)

row_sid_table.to_csv(
    SID_OUTPUT,
    index=False
)


# ============================================================
# Save human-readable specification.
# ============================================================

SPEC_OUTPUT = (
    OUT_DIR /
    "adult_1sa_sensitive_cell_specification.txt"
)

with open(
    SPEC_OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "KC-SLICE ADULT 1-SA — SENSITIVE CELL SPECIFICATION\n"
    )
    f.write("=" * 60 + "\n\n")

    f.write(
        "Each sensitive attribute column within each bucket "
        "is represented as one grouped cell with value counts.\n"
    )

    f.write(
        "One deterministic SID is generated for each "
        "bucket/SA cell.\n\n"
    )

    for _, row in sensitive_table.iterrows():

        f.write(
            f"Bucket {row['bucket']} | "
            f"SA={row['SA']} | "
            f"SID={row['SID']}\n"
        )

        f.write(
            f"  {row['cell']}\n\n"
        )

print()
print("Sensitive-cell table:")
print(
    sensitive_table[
        [
            "bucket",
            "SID",
            "SA",
            "cell",
        ]
    ].to_string(index=False)
)

print()
print(f"Saved SA cells : {SA_OUTPUT}")
print(f"Saved SID links: {SID_OUTPUT}")
print(f"Saved spec     : {SPEC_OUTPUT}")

print()
print("=" * 70)
print("KC-SLICE ADULT 1-SA PHASE 3A COMPLETE")
print("=" * 70)

########################################################################
# SOURCE NOTEBOOK EXECUTION CELL 34
########################################################################

# Cell 34 — KC-SLICE ADULT 1-SA
# QID CORRELATION AUDIT
#
# Purpose:
#   Audit QID associations INSIDE EACH KC-Slice bucket.
#
# Important:
#   The KC-Slice paper specifies a Correlate() operation but
#   does not specify the exact statistical measure or threshold.
#
# Therefore this cell only measures the associations.
# It does NOT lock a correlation threshold yet.
# ============================================================

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path("/content/kc-slice")

INPUT = (
    ROOT /
    "results/kc_slice_adult_1sa/"
    "adult_1sa_bucket_privacy_output.csv"
)

OUT_DIR = (
    ROOT /
    "results/kc_slice_adult_1sa"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

QIDS = [
    "age",
    "workclass",
    "education.num",
    "marital.status",
    "occupation",
    "race",
    "sex",
    "native.country",
]

BUCKET_COLUMN = "_kc_bucket"

NUMERIC_QIDS = {
    "age",
    "education.num",
}

CATEGORICAL_QIDS = [
    q for q in QIDS
    if q not in NUMERIC_QIDS
]


# ============================================================
# Association functions
# ============================================================

def cramers_v(x, y):
    """
    Bias-unadjusted Cramer's V.
    Used for categorical-categorical association.
    """
    table = pd.crosstab(
        x,
        y
    )

    if table.empty:
        return 0.0

    observed = table.to_numpy(
        dtype=float
    )

    n = observed.sum()

    if n == 0:
        return 0.0

    row_sums = observed.sum(axis=1)
    col_sums = observed.sum(axis=0)

    expected = np.outer(
        row_sums,
        col_sums
    ) / n

    mask = expected > 0

    chi2 = (
        (
            observed[mask] -
            expected[mask]
        ) ** 2 /
        expected[mask]
    ).sum()

    phi2 = chi2 / n

    r, k = observed.shape

    return float(
        np.sqrt(
            phi2 /
            max(
                1,
                min(k - 1, r - 1)
            )
        )
    )


def correlation_ratio(categories, values):
    """
    Eta coefficient.
    Measures numeric-categorical association.
    """
    categories = pd.Series(
        categories
    )

    values = pd.to_numeric(
        pd.Series(values),
        errors="coerce"
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

    for category in categories.unique():

        group = values[
            categories == category
        ]

        numerator += (
            len(group) *
            (
                group.mean()
                -
                grand_mean
            ) ** 2
        )

    denominator = (
        (
            values -
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


def association(x, y, x_name, y_name):
    """
    Select an association measure based on QID types.

    numeric-numeric:
        absolute Spearman rho

    categorical-categorical:
        Cramer's V

    numeric-categorical:
        Eta
    """

    x_numeric = x_name in NUMERIC_QIDS
    y_numeric = y_name in NUMERIC_QIDS

    if x_numeric and y_numeric:

        x_num = pd.to_numeric(
            x,
            errors="coerce"
        )

        y_num = pd.to_numeric(
            y,
            errors="coerce"
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
            y_num[valid]
        )

        return abs(float(rho)), "Spearman"

    elif (not x_numeric) and (not y_numeric):

        return (
            cramers_v(x, y),
            "Cramer's V"
        )

    else:

        # Numeric variable must be passed as the
        # second argument to correlation_ratio.
        if x_numeric:
            value = x
            category = y
        else:
            value = y
            category = x

        return (
            correlation_ratio(
                category,
                value
            ),
            "Eta"
        )


# ============================================================
# Compute associations within every bucket
# ============================================================

bucket_ids = sorted(
    df[BUCKET_COLUMN].unique()
)

records = []

for bucket_number, bucket_id in enumerate(
    bucket_ids,
    start=1
):

    bucket = df[
        df[BUCKET_COLUMN] == bucket_id
    ].copy()

    print()
    print("=" * 70)
    print(
        f"BUCKET {bucket_number} "
        f"({len(bucket):,} rows)"
    )
    print("=" * 70)

    bucket_results = []

    for qid_a, qid_b in combinations(
        QIDS,
        2
    ):

        value, measure = association(
            bucket[qid_a],
            bucket[qid_b],
            qid_a,
            qid_b
        )

        record = {
            "bucket": bucket_number,
            "bucket_id": bucket_id,
            "qid_a": qid_a,
            "qid_b": qid_b,
            "measure": measure,
            "association": value,
        }

        records.append(record)
        bucket_results.append(record)

    bucket_results_df = pd.DataFrame(
        bucket_results
    ).sort_values(
        "association",
        ascending=False
    )

    print(
        bucket_results_df[
            [
                "qid_a",
                "qid_b",
                "measure",
                "association",
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# Combined audit table
# ============================================================

corr_df = pd.DataFrame(
    records
)

print()
print("=" * 70)
print("CROSS-BUCKET CORRELATION SUMMARY")
print("=" * 70)

summary = (
    corr_df
    .groupby(
        [
            "qid_a",
            "qid_b",
            "measure",
        ],
        as_index=False
    )
    .agg(
        min_association=(
            "association",
            "min"
        ),
        mean_association=(
            "association",
            "mean"
        ),
        max_association=(
            "association",
            "max"
        ),
    )
    .sort_values(
        "max_association",
        ascending=False
    )
)

print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# Candidate associations that are >= 0.40
#
# This is ONLY an audit view.
# 0.40 is NOT being declared as the KC-Slice threshold here.
# ============================================================

candidate_threshold = 0.40

candidates = corr_df[
    corr_df["association"]
    >= candidate_threshold
].sort_values(
    [
        "bucket",
        "association",
    ],
    ascending=[
        True,
        False,
    ]
)

print()
print("=" * 70)
print(
    "ASSOCIATIONS >= 0.40 "
    "(AUDIT ONLY)"
)
print("=" * 70)

if len(candidates) == 0:

    print("None")

else:

    print(
        candidates[
            [
                "bucket",
                "qid_a",
                "qid_b",
                "measure",
                "association",
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# Save
# ============================================================

RAW_OUTPUT = (
    OUT_DIR /
    "adult_1sa_qid_correlation_audit.csv"
)

SUMMARY_OUTPUT = (
    OUT_DIR /
    "adult_1sa_qid_correlation_summary.csv"
)

SPEC_OUTPUT = (
    OUT_DIR /
    "adult_1sa_qid_correlation_audit.txt"
)

corr_df.to_csv(
    RAW_OUTPUT,
    index=False
)

summary.to_csv(
    SUMMARY_OUTPUT,
    index=False
)

with open(
    SPEC_OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "KC-SLICE ADULT 1-SA — QID CORRELATION AUDIT\n"
    )
    f.write("=" * 60 + "\n\n")

    f.write(
        "The KC-Slice source specifies a Correlate() "
        "operation but does not specify the statistical "
        "measure or threshold.\n\n"
    )

    f.write(
        "This audit therefore measures:\n"
    )

    f.write(
        "  Numeric-numeric      : absolute Spearman rho\n"
    )

    f.write(
        "  Categorical-categorical : Cramer's V\n"
    )

    f.write(
        "  Numeric-categorical  : Eta\n\n"
    )

    f.write(
        "The 0.40 display is an audit threshold only; "
        "it is not claimed to come from the KC-Slice paper.\n"
    )


print()
print(f"Saved raw audit     : {RAW_OUTPUT}")
print(f"Saved summary       : {SUMMARY_OUTPUT}")
print(f"Saved specification : {SPEC_OUTPUT}")

print()
print("=" * 70)
print("KC-SLICE ADULT 1-SA QID CORRELATION AUDIT COMPLETE")
print("=" * 70)

########################################################################
# SOURCE NOTEBOOK EXECUTION CELL 35
########################################################################

# Cell 35 — KC-SLICE ADULT 1-SA
# LOCK CORRELATE() + DETERMINISTIC QID GROUPING
# ============================================================

from pathlib import Path
from itertools import combinations

import pandas as pd

ROOT = Path("/content/kc-slice")

INPUT = (
    ROOT /
    "results/kc_slice_adult_1sa/"
    "adult_1sa_qid_correlation_audit.csv"
)

OUT_DIR = (
    ROOT /
    "results/kc_slice_adult_1sa"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

corr_df = pd.read_csv(INPUT)

QIDS = [
    "age",
    "workclass",
    "education.num",
    "marital.status",
    "occupation",
    "race",
    "sex",
    "native.country",
]

CORRELATION_THRESHOLD = 0.40


print("=" * 70)
print("KC-SLICE — ADULT 1-SA")
print("CORRELATE() + QID GROUPING SPECIFICATION")
print("=" * 70)

print()
print("Operational Correlate() rule:")
print(
    f"  Correlate(A,B) = 1 when association >= "
    f"{CORRELATION_THRESHOLD:.2f}"
)

print()
print("Association measures:")
print("  Numeric-numeric         : absolute Spearman rho")
print("  Categorical-categorical : Cramer's V")
print("  Numeric-categorical     : Eta")

print()
print("Important:")
print(
    "  The threshold and grouping strategy are project-defined "
    "operationalizations because the source does not specify "
    "the exact statistical threshold or overlap resolution."
)


# ============================================================
# Deterministic greedy grouping
#
# Algorithm:
#   1. Start with the first unused QID.
#   2. Compare it with later unused QIDs.
#   3. Concatenate every later QID with association >= threshold.
#   4. Mark those QIDs as consumed.
#   5. Continue with the next unused QID.
#
# This gives non-overlapping groups and is deterministic.
# ============================================================

bucket_ids = sorted(
    corr_df["bucket_id"].unique()
)

all_group_records = []

for bucket_id in bucket_ids:

    bucket_number = int(
        corr_df.loc[
            corr_df["bucket_id"] == bucket_id,
            "bucket"
        ].iloc[0]
    )

    bucket_corr = corr_df[
        corr_df["bucket_id"] == bucket_id
    ].copy()

    # Lookup table for pairwise association.
    assoc_lookup = {}

    for _, row in bucket_corr.iterrows():

        pair = frozenset(
            [
                row["qid_a"],
                row["qid_b"],
            ]
        )

        assoc_lookup[pair] = {
            "association": float(
                row["association"]
            ),
            "measure": row["measure"],
        }

    unused = list(QIDS)
    groups = []

    while unused:

        anchor = unused.pop(0)

        group = [anchor]

        remaining = []

        for candidate in unused:

            pair = frozenset(
                [
                    anchor,
                    candidate,
                ]
            )

            result = assoc_lookup.get(
                pair
            )

            if (
                result is not None
                and
                result["association"]
                >=
                CORRELATION_THRESHOLD
            ):

                group.append(
                    candidate
                )

            else:

                remaining.append(
                    candidate
                )

        unused = remaining

        groups.append(group)

    print()
    print("-" * 70)
    print(
        f"BUCKET {bucket_number}"
    )
    print("-" * 70)

    for group_id, group in enumerate(
        groups,
        start=1
    ):

        label = "__".join(group)

        print(
            f"Group {group_id}: "
            f"{label}"
        )

        # Save pair decisions relevant to this group.
        if len(group) > 1:

            for qid_a, qid_b in combinations(
                group,
                2
            ):

                pair = frozenset(
                    [
                        qid_a,
                        qid_b,
                    ]
                )

                result = assoc_lookup.get(
                    pair
                )

                # Not every pair inside a group is necessarily
                # above threshold; grouping is based on the
                # anchor QID's qualifying associations.
                if result is not None:

                    all_group_records.append({
                        "bucket": bucket_number,
                        "bucket_id": bucket_id,
                        "group_id": group_id,
                        "qid_a": qid_a,
                        "qid_b": qid_b,
                        "measure": result["measure"],
                        "association": result["association"],
                        "threshold": CORRELATION_THRESHOLD,
                    })

        else:

            all_group_records.append({
                "bucket": bucket_number,
                "bucket_id": bucket_id,
                "group_id": group_id,
                "qid_a": group[0],
                "qid_b": None,
                "measure": None,
                "association": None,
                "threshold": CORRELATION_THRESHOLD,
            })


# ============================================================
# Verify that every QID appears exactly once per bucket.
# ============================================================

for bucket_id in bucket_ids:

    bucket_number = int(
        corr_df.loc[
            corr_df["bucket_id"] == bucket_id,
            "bucket"
        ].iloc[0]
    )

    # Reconstruct the same grouping for verification.
    bucket_corr = corr_df[
        corr_df["bucket_id"] == bucket_id
    ]

    assoc_lookup = {}

    for _, row in bucket_corr.iterrows():

        assoc_lookup[
            frozenset(
                [
                    row["qid_a"],
                    row["qid_b"],
                ]
            )
        ] = float(
            row["association"]
        )

    unused = list(QIDS)
    seen = []

    while unused:

        anchor = unused.pop(0)

        seen.append(anchor)

        remaining = []

        for candidate in unused:

            pair = frozenset(
                [
                    anchor,
                    candidate,
                ]
            )

            if (
                pair in assoc_lookup
                and
                assoc_lookup[pair]
                >=
                CORRELATION_THRESHOLD
            ):

                seen.append(candidate)

            else:

                remaining.append(candidate)

        unused = remaining

    assert (
        len(seen) == len(QIDS)
    )

    assert (
        len(set(seen)) == len(QIDS)
    )


print()
print("=" * 70)
print("VERIFICATION")
print("=" * 70)

print(
    "Every QID assigned exactly once per bucket : PASS"
)

print(
    "Correlation threshold locked at            : "
    f"{CORRELATION_THRESHOLD:.2f}"
)

print(
    "Grouping is deterministic                  : PASS"
)


# ============================================================
# Save specification
# ============================================================

spec_path = (
    OUT_DIR /
    "adult_1sa_qid_correlation_specification.txt"
)

with open(
    spec_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "KC-SLICE ADULT 1-SA — QID CORRELATION SPECIFICATION\n"
    )

    f.write("=" * 65 + "\n\n")

    f.write(
        "Correlate(A,B) = 1 iff association >= 0.40\n\n"
    )

    f.write(
        "Measures:\n"
    )
    f.write(
        "  Numeric-numeric         : absolute Spearman rho\n"
    )
    f.write(
        "  Categorical-categorical : Cramer's V\n"
    )
    f.write(
        "  Numeric-categorical     : Eta\n\n"
    )

    f.write(
        "Correlation is evaluated independently within each "
        "KC-Slice bucket.\n\n"
    )

    f.write(
        "QID grouping:\n"
    )

    f.write(
        "  Deterministic greedy, non-overlapping grouping.\n"
    )

    f.write(
        "  QIDs are processed in the predefined Adult QID order.\n"
    )

    f.write(
        "  For each unused anchor QID, later QIDs meeting the "
        "threshold are concatenated with it.\n\n"
    )

    f.write(
        "This threshold and grouping procedure are project-defined "
        "operationalizations because the source does not specify "
        "the exact Correlate() statistic, threshold, or overlap "
        "resolution.\n"
    )


# ============================================================
# Save detailed grouping information
# ============================================================

grouping_path = (
    OUT_DIR /
    "adult_1sa_qid_grouping_decisions.csv"
)

grouping_df = pd.DataFrame(
    all_group_records
)

grouping_df.to_csv(
    grouping_path,
    index=False
)

print()
print(f"Saved specification : {spec_path}")
print(f"Saved grouping log  : {grouping_path}")

print()
print("=" * 70)
print("KC-SLICE ADULT 1-SA QID GROUPING LOCKED")
print("=" * 70)

########################################################################
# SOURCE NOTEBOOK EXECUTION CELL 36
########################################################################

# Cell 36 — KC-SLICE ADULT 1-SA
# PHASE 3B: QID CONCATENATION + SID + PERMUTATION
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/content/kc-slice")

INPUT = (
    ROOT /
    "results/kc_slice_adult_1sa/"
    "adult_1sa_bucket_privacy_output.csv"
)

SID_INPUT = (
    ROOT /
    "results/kc_slice_adult_1sa/"
    "adult_1sa_sensitive_cells.csv"
)

OUT_DIR = (
    ROOT /
    "results/kc_slice_adult_1sa"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOCKED PARAMETERS
# ============================================================

RUN_SEED = 0

PROJECT_K = 10
CORRELATION_THRESHOLD = 0.40

QID_GROUPS = [
    ["age", "marital.status"],
    ["workclass"],
    ["education.num", "occupation"],
    ["race", "native.country"],
    ["sex"],
]

GROUP_NAMES = [
    "slice_1",
    "slice_2",
    "slice_3",
    "slice_4",
    "slice_5",
]

BUCKET_COLUMN = "_kc_bucket"


# ============================================================
# LOAD INPUTS
# ============================================================

df = pd.read_csv(INPUT)

sid_table = pd.read_csv(
    SID_INPUT
)

print("=" * 70)
print("KC-SLICE — ADULT 1-SA")
print("PHASE 3B: QID CONCATENATION + SID + PERMUTATION")
print("=" * 70)

print(f"Input rows             : {len(df):,}")
print(f"QID groups             : {len(QID_GROUPS)}")
print(f"Correlation threshold  : {CORRELATION_THRESHOLD:.2f}")
print(f"Project k              : {PROJECT_K}")
print(f"Run seed               : {RUN_SEED}")


# ============================================================
# VERIFY LOCKED GROUPING
# ============================================================

flat_qids = [
    qid
    for group in QID_GROUPS
    for qid in group
]

EXPECTED_QIDS = [
    "age",
    "workclass",
    "education.num",
    "marital.status",
    "occupation",
    "race",
    "sex",
    "native.country",
]

# IMPORTANT:
# The grouping order is allowed to differ from the original
# dataset QID order. We only require that the same QIDs are
# present exactly once.

assert len(flat_qids) == len(EXPECTED_QIDS)

assert set(flat_qids) == set(EXPECTED_QIDS)

assert len(set(flat_qids)) == len(EXPECTED_QIDS)


print()
print("Locked QID grouping:")

for group_name, group in zip(
    GROUP_NAMES,
    QID_GROUPS
):
    print(
        f"  {group_name}: "
        + " + ".join(group)
    )

print()
print(
    "QID coverage verification : PASS"
)

print(
    "QID uniqueness verification : PASS"
)


# ============================================================
# GET BUCKET -> SID MAPPING
# ============================================================

bucket_ids = sorted(
    df[BUCKET_COLUMN].unique()
)

bucket_sid = {}

for bucket_id in bucket_ids:

    matches = sid_table[
        sid_table["bucket_id"] == bucket_id
    ]

    # One income SID per bucket.
    assert len(matches) == 1

    bucket_sid[bucket_id] = str(
        matches.iloc[0]["SID"]
    )


print()
print("Bucket -> SID mapping:")

for bucket_id in bucket_ids:

    print(
        f"  Bucket {int(bucket_id) + 1} "
        f"-> SID {bucket_sid[bucket_id]}"
    )


# ============================================================
# BUILD SLICED QID TABLES
# ============================================================

all_sliced_rows = []

bucket_permutation_records = []

for bucket_number, bucket_id in enumerate(
    bucket_ids,
    start=1
):

    bucket = df[
        df[BUCKET_COLUMN] == bucket_id
    ].copy()

    sid = bucket_sid[bucket_id]

    print()
    print("=" * 70)
    print(
        f"BUCKET {bucket_number} "
        f"({len(bucket):,} rows)"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Build unpermuted QID slices.
    # --------------------------------------------------------

    qid_output = pd.DataFrame(
        index=bucket.index
    )

    qid_output[
        "_source_row"
    ] = bucket.index

    qid_output[
        "bucket"
    ] = bucket_number

    qid_output[
        "bucket_id"
    ] = bucket_id

    for group_name, group in zip(
        GROUP_NAMES,
        QID_GROUPS
    ):

        values = []

        for row_index in bucket.index:

            components = [
                str(
                    bucket.loc[
                        row_index,
                        qid
                    ]
                )
                for qid in group
            ]

            # Attach the sensitive-cell SID.
            components.append(
                f"SID={sid}"
            )

            values.append(
                " | ".join(
                    components
                )
            )

        qid_output[
            group_name
        ] = values


    # --------------------------------------------------------
    # Record accounting.
    # --------------------------------------------------------

    assert len(qid_output) == len(bucket)


    # --------------------------------------------------------
    # Randomly permute the QID slice columns.
    #
    # We use a fixed seed for reproducibility.
    # Different buckets receive deterministic derived seeds.
    # --------------------------------------------------------

    rng = np.random.default_rng(
        RUN_SEED + bucket_number
    )

    permutation = rng.permutation(
        len(GROUP_NAMES)
    )

    original_group_names = list(
        GROUP_NAMES
    )

    permuted_group_names = [
        original_group_names[i]
        for i in permutation
    ]

    print()
    print(
        "Original QID slice order:"
    )

    print(
        "  "
        +
        " -> ".join(
            original_group_names
        )
    )

    print(
        "Permuted QID slice order:"
    )

    print(
        "  "
        +
        " -> ".join(
            permuted_group_names
        )
    )


    # --------------------------------------------------------
    # Save permutation mapping.
    # --------------------------------------------------------

    for position, original_index in enumerate(
        permutation,
        start=1
    ):

        bucket_permutation_records.append({
            "bucket": bucket_number,
            "bucket_id": bucket_id,
            "output_position": position,
            "original_group": original_group_names[
                original_index
            ],
        })


    # --------------------------------------------------------
    # Construct final bucket QID table.
    # --------------------------------------------------------

    final_columns = [
        "_source_row",
        "bucket",
        "bucket_id",
    ] + permuted_group_names

    final_qid = qid_output[
        final_columns
    ].copy()

    all_sliced_rows.append(
        final_qid
    )


    # --------------------------------------------------------
    # Display sample.
    # --------------------------------------------------------

    print()
    print("Final QID table sample:")

    print(
        final_qid.head(5).to_string(
            index=False
        )
    )


    # --------------------------------------------------------
    # Save bucket-specific QID table.
    # --------------------------------------------------------

    bucket_output = (
        OUT_DIR /
        f"adult_1sa_qid_bucket_{bucket_number}.csv"
    )

    final_qid.to_csv(
        bucket_output,
        index=False
    )

    print()
    print(
        f"Saved: {bucket_output}"
    )


# ============================================================
# COMBINE ALL BUCKETS
# ============================================================

final_qid_table = pd.concat(
    all_sliced_rows,
    ignore_index=True
)

assert len(final_qid_table) == len(df)


# ============================================================
# FINAL VERIFICATION
# ============================================================

print()
print("=" * 70)
print("FINAL QID TABLE VERIFICATION")
print("=" * 70)

print(
    f"Input rows                  : "
    f"{len(df):,}"
)

print(
    f"Final QID rows              : "
    f"{len(final_qid_table):,}"
)

assert (
    len(final_qid_table)
    ==
    len(df)
)

print(
    "Record accounting           : PASS"
)


# ------------------------------------------------------------
# Every source row must occur exactly once.
# ------------------------------------------------------------

source_counts = (
    final_qid_table[
        "_source_row"
    ]
    .value_counts()
)

assert (
    source_counts == 1
).all()

print(
    "One output row per source row: PASS"
)


# ------------------------------------------------------------
# Bucket sizes must remain unchanged.
# ------------------------------------------------------------

bucket_counts = (
    final_qid_table[
        "bucket_id"
    ]
    .value_counts()
    .sort_index()
)

print()
print("Final bucket sizes:")

for bucket_id, count in (
    bucket_counts.items()
):

    print(
        f"  Bucket {int(bucket_id) + 1}: "
        f"{count:,}"
    )

assert all(
    count == 15081
    for count in bucket_counts
)

print(
    "Bucket-size preservation       : PASS"
)


# ------------------------------------------------------------
# Every QID slice must contain the SID.
# ------------------------------------------------------------

for group_name in GROUP_NAMES:

    assert (
        final_qid_table[
            group_name
        ]
        .astype(str)
        .str.contains(
            "SID=",
            regex=False
        )
        .all()
    )

print(
    "SID linkage in every QID slice : PASS"
)


# ============================================================
# SAVE COMBINED QID TABLE
# ============================================================

FINAL_QID_OUTPUT = (
    OUT_DIR /
    "adult_1sa_qid_sliced_permuted.csv"
)

final_qid_table.to_csv(
    FINAL_QID_OUTPUT,
    index=False
)


# ============================================================
# SAVE PERMUTATION LOG
# ============================================================

permutation_df = pd.DataFrame(
    bucket_permutation_records
)

PERM_OUTPUT = (
    OUT_DIR /
    "adult_1sa_qid_column_permutation.csv"
)

permutation_df.to_csv(
    PERM_OUTPUT,
    index=False
)


# ============================================================
# SAVE Bsa TABLE
# ============================================================

BSA_OUTPUT = (
    OUT_DIR /
    "adult_1sa_Bsa_sensitive_table.csv"
)

sid_table.to_csv(
    BSA_OUTPUT,
    index=False
)


# ============================================================
# SAVE SPECIFICATION
# ============================================================

SPEC_OUTPUT = (
    OUT_DIR /
    "adult_1sa_phase3b_specification.txt"
)

with open(
    SPEC_OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "KC-SLICE ADULT 1-SA — PHASE 3B SPECIFICATION\n"
    )

    f.write("=" * 65 + "\n\n")

    f.write(
        "QID groups:\n"
    )

    for name, group in zip(
        GROUP_NAMES,
        QID_GROUPS
    ):

        f.write(
            f"  {name}: "
            + " + ".join(group)
            + "\n"
        )

    f.write("\n")

    f.write(
        "Correlation threshold: 0.40\n"
    )

    f.write(
        "Run seed: 0\n\n"
    )

    f.write(
        "Each correlated QID group is concatenated into one "
        "QID slice and linked with the bucket's sensitive-cell "
        "SID.\n\n"
    )

    f.write(
        "Only QID slice columns are randomly permuted. "
        "Bucket/source-row accounting columns remain fixed "
        "for reproducibility and evaluation.\n\n"
    )

    f.write(
        "The source specifies random permutation but does not "
        "specify a random seed; seed 0 is therefore an explicit "
        "project reproducibility decision.\n"
    )


# ============================================================
# OUTPUT SUMMARY
# ============================================================

print()
print("=" * 70)
print("SAVED OUTPUTS")
print("=" * 70)

print(
    f"Bsa sensitive table : {BSA_OUTPUT}"
)

print(
    f"Combined Bqa table  : {FINAL_QID_OUTPUT}"
)

print(
    f"Permutation log      : {PERM_OUTPUT}"
)

print(
    f"Specification        : {SPEC_OUTPUT}"
)

print()
print("=" * 70)
print("KC-SLICE ADULT 1-SA PHASE 3B COMPLETE")
print("=" * 70)

########################################################################
# SOURCE NOTEBOOK EXECUTION CELL 37
########################################################################

# Cell 37 — KC-SLICE ADULT 1-SA
# SID FORMAT QA + FINAL PACKAGE
# ============================================================

from pathlib import Path
import pandas as pd

ROOT = Path("/content/kc-slice")

OUT_DIR = (
    ROOT /
    "results/kc_slice_adult_1sa"
)

SID_INPUT = (
    OUT_DIR /
    "adult_1sa_sensitive_cells.csv"
)

QID_INPUT = (
    OUT_DIR /
    "adult_1sa_qid_sliced_permuted.csv"
)

PERM_INPUT = (
    OUT_DIR /
    "adult_1sa_qid_column_permutation.csv"
)

# ============================================================
# Read SID table explicitly as string
# ============================================================

sid_table = pd.read_csv(
    SID_INPUT,
    dtype={"SID": str}
)

qid_table = pd.read_csv(
    QID_INPUT,
    dtype={
        "_source_row": int,
        "bucket": int,
        "bucket_id": int,
    }
)

print("=" * 70)
print("KC-SLICE — ADULT 1-SA")
print("FINAL SID FORMAT + PACKAGE QA")
print("=" * 70)


# ============================================================
# Restore fixed-width SID representation
# ============================================================

sid_table["SID"] = (
    sid_table["SID"]
    .astype(str)
    .str.zfill(3)
)

print()
print("Sensitive-cell SIDs:")

print(
    sid_table[
        ["bucket", "SA", "SID"]
    ].to_string(index=False)
)


# ============================================================
# Rebuild QID SID strings with zero-padded SIDs
# ============================================================

for bucket_id in sorted(
    qid_table["bucket_id"].unique()
):

    matches = sid_table[
        sid_table["bucket_id"] == bucket_id
    ]

    assert len(matches) == 1

    sid = matches.iloc[0]["SID"]

    bucket_mask = (
        qid_table["bucket_id"]
        ==
        bucket_id
    )

    # Replace any SID=11 / SID=21 etc. with fixed-width SID.
    for column in [
        "slice_1",
        "slice_2",
        "slice_3",
        "slice_4",
        "slice_5",
    ]:

        qid_table.loc[
            bucket_mask,
            column
        ] = (
            qid_table.loc[
                bucket_mask,
                column
            ]
            .astype(str)
            .str.replace(
                r"SID=\d+",
                f"SID={sid}",
                regex=True
            )
        )


# ============================================================
# Verify exact SID format
# ============================================================

expected_sids = {
    0: "011",
    1: "021",
}

for bucket_id, expected_sid in (
    expected_sids.items()
):

    actual = sid_table.loc[
        sid_table["bucket_id"] == bucket_id,
        "SID"
    ].iloc[0]

    assert actual == expected_sid


print()
print(
    "Fixed-width SID format      : PASS"
)


# ============================================================
# Verify SID linkage
# ============================================================

slice_columns = [
    "slice_1",
    "slice_2",
    "slice_3",
    "slice_4",
    "slice_5",
]

for bucket_id in sorted(
    qid_table["bucket_id"].unique()
):

    expected_sid = sid_table.loc[
        sid_table["bucket_id"] == bucket_id,
        "SID"
    ].iloc[0]

    bucket_rows = qid_table[
        qid_table["bucket_id"] == bucket_id
    ]

    for column in slice_columns:

        assert (
            bucket_rows[column]
            .astype(str)
            .str.contains(
                f"SID={expected_sid}",
                regex=False
            )
            .all()
        )


print(
    "SID linkage verification     : PASS"
)


# ============================================================
# Record accounting
# ============================================================

assert len(qid_table) == 30162

assert (
    qid_table["_source_row"]
    .nunique()
    ==
    30162
)

print(
    "Final row accounting         : PASS"
)


# ============================================================
# Bucket accounting
# ============================================================

bucket_sizes = (
    qid_table
    .groupby("bucket_id")
    .size()
)

assert (
    bucket_sizes.to_dict()
    ==
    {
        0: 15081,
        1: 15081,
    }
)

print(
    "Final bucket accounting      : PASS"
)


# ============================================================
# Save corrected files
# ============================================================

sid_table.to_csv(
    SID_INPUT,
    index=False
)

qid_table.to_csv(
    QID_INPUT,
    index=False
)


# ============================================================
# Create final package manifest
# ============================================================

MANIFEST = (
    OUT_DIR /
    "adult_1sa_final_package.txt"
)

with open(
    MANIFEST,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "KC-SLICE ADULT 1-SA — FINAL PACKAGE\n"
    )
    f.write("=" * 65 + "\n\n")

    f.write(
        "Dataset records: 30162\n"
    )

    f.write(
        "Sensitive attribute: income\n"
    )

    f.write(
        "Project k: 10\n"
    )

    f.write(
        "KC-Slice internal bucket size: 15081\n"
    )

    f.write(
        "Privacy threshold C: 5%\n"
    )

    f.write(
        "HSA: <=50K\n"
    )

    f.write(
        "QID correlation threshold: 0.40\n"
    )

    f.write(
        "QID groups:\n"
    )
    f.write(
        "  slice_1 = age + marital.status\n"
    )
    f.write(
        "  slice_2 = workclass\n"
    )
    f.write(
        "  slice_3 = education.num + occupation\n"
    )
    f.write(
        "  slice_4 = race + native.country\n"
    )
    f.write(
        "  slice_5 = sex\n"
    )

    f.write(
        "\nRun seed: 0\n"
    )

    f.write(
        "\nSID values:\n"
    )
    f.write(
        "  Bucket 1 = 011\n"
    )
    f.write(
        "  Bucket 2 = 021\n"
    )

    f.write(
        "\nStatus:\n"
    )
    f.write(
        "  Phase 1 bucket creation: PASS\n"
    )
    f.write(
        "  Phase 2 privacy enforcement: PASS\n"
    )
    f.write(
        "  Phase 3A sensitive cells + SID: PASS\n"
    )
    f.write(
        "  Phase 3B QID concatenation + permutation: PASS\n"
    )
    f.write(
        "  Final SID format QA: PASS\n"
    )


print()
print("Final 1-SA package files:")
print(
    f"  Bsa: {SID_INPUT}"
)
print(
    f"  Bqa: {QID_INPUT}"
)
print(
    f"  Manifest: {MANIFEST}"
)

print()
print("=" * 70)
print("KC-SLICE ADULT 1-SA FINAL PACKAGE QA COMPLETE")
print("=" * 70)

########################################################################
# SOURCE NOTEBOOK EXECUTION CELL 38
########################################################################

# Cell 38 — KC-SLICE ADULT 1-SA
# PUBLIC RELEASE / EVALUATION PACKAGE SEPARATION
# ============================================================

from pathlib import Path
import pandas as pd

ROOT = Path("/content/kc-slice")

BASELINE_INPUT = (
    ROOT /
    "data/processed/"
    "adult_standard_baseline_k10_1sa.csv"
)

BQA_INPUT = (
    ROOT /
    "results/kc_slice_adult_1sa/"
    "adult_1sa_qid_sliced_permuted.csv"
)

BSA_INPUT = (
    ROOT /
    "results/kc_slice_adult_1sa/"
    "adult_1sa_sensitive_cells.csv"
)

OUT_DIR = (
    ROOT /
    "results/kc_slice_adult_1sa"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Load source/evaluation data
# ============================================================

baseline = pd.read_csv(
    BASELINE_INPUT
)

bqa_internal = pd.read_csv(
    BQA_INPUT
)

bsa = pd.read_csv(
    BSA_INPUT,
    dtype={"SID": str}
)

bsa["SID"] = (
    bsa["SID"]
    .astype(str)
    .str.zfill(3)
)


print("=" * 70)
print("KC-SLICE — ADULT 1-SA")
print("PUBLIC RELEASE / EVALUATION PACKAGE")
print("=" * 70)

print(
    f"Original baseline rows : {len(baseline):,}"
)

print(
    f"Internal Bqa rows      : {len(bqa_internal):,}"
)

print(
    f"Bsa rows               : {len(bsa):,}"
)


# ============================================================
# Identify QID slice columns from the locked implementation.
# ============================================================

SLICE_COLUMNS = [
    "slice_1",
    "slice_2",
    "slice_3",
    "slice_4",
    "slice_5",
]

for column in SLICE_COLUMNS:

    assert column in bqa_internal.columns


# ============================================================
# 1. PUBLIC Bqa
#
# Only the actual QID slices are exposed.
#
# Internal:
#   _source_row
#   bucket
#   bucket_id
#
# are removed.
# ============================================================

bqa_public = bqa_internal[
    SLICE_COLUMNS
].copy()

assert len(bqa_public) == len(
    baseline
)


# ============================================================
# 2. PUBLIC Bsa
#
# Keep only the sensitive-cell publication representation.
# ============================================================

BSA_PUBLIC_COLUMNS = [
    "bucket",
    "SID",
    "SA",
    "cell",
]

for column in BSA_PUBLIC_COLUMNS:

    assert column in bsa.columns

bsa_public = bsa[
    BSA_PUBLIC_COLUMNS
].copy()


# ============================================================
# 3. PRIVATE EVALUATION DATASET
#
# We retain source_row ONLY internally so that the original
# target can be aligned with the anonymized QID representation.
#
# The target is not included in the published Bqa.
# ============================================================

evaluation = bqa_internal[
    [
        "_source_row",
        "bucket",
        "bucket_id",
    ] + SLICE_COLUMNS
].copy()

# Original income target.
#
# baseline columns:
#   QIDs + income
#
assert "income" in baseline.columns

target_lookup = baseline[
    ["income"]
].copy()

target_lookup[
    "_source_row"
] = target_lookup.index

target_lookup = target_lookup[
    [
        "_source_row",
        "income",
    ]
]

evaluation = evaluation.merge(
    target_lookup,
    on="_source_row",
    how="left",
    validate="one_to_one"
)


# ============================================================
# Evaluation integrity
# ============================================================

print()
print("-" * 70)
print("EVALUATION INTEGRITY")
print("-" * 70)

assert len(evaluation) == len(
    baseline
)

print(
    "Evaluation row count       : PASS"
)


assert (
    evaluation["_source_row"]
    .nunique()
    ==
    len(baseline)
)

print(
    "Unique source-row mapping  : PASS"
)


assert (
    evaluation["income"]
    .notna()
    .all()
)

print(
    "Ground-truth target linkage: PASS"
)


# Target distribution should remain exactly equal to the
# original dataset because anonymization does not alter the
# private evaluation labels.

original_target_counts = (
    baseline["income"]
    .value_counts()
    .sort_index()
)

evaluation_target_counts = (
    evaluation["income"]
    .value_counts()
    .sort_index()
)

assert (
    original_target_counts.equals(
        evaluation_target_counts
    )
)

print(
    "Target distribution         : PASS"
)


# ============================================================
# Verify published Bqa contains no internal metadata.
# ============================================================

assert "_source_row" not in bqa_public.columns
assert "bucket" not in bqa_public.columns
assert "bucket_id" not in bqa_public.columns

print(
    "Internal metadata removed from Bqa: PASS"
)


# ============================================================
# Verify Bsa SID format.
# ============================================================

assert (
    bsa_public["SID"]
    .astype(str)
    .str.fullmatch(r"\d{3}")
    .all()
)

print(
    "Public Bsa SID format       : PASS"
)


# ============================================================
# Verify public row accounting.
# ============================================================

assert len(bqa_public) == 30162

print(
    "Public Bqa record count     : PASS"
)


# ============================================================
# Save PUBLIC RELEASE artifacts
# ============================================================

BQA_PUBLIC_OUTPUT = (
    OUT_DIR /
    "adult_1sa_Bqa_public.csv"
)

BSA_PUBLIC_OUTPUT = (
    OUT_DIR /
    "adult_1sa_Bsa_public.csv"
)

bqa_public.to_csv(
    BQA_PUBLIC_OUTPUT,
    index=False
)

bsa_public.to_csv(
    BSA_PUBLIC_OUTPUT,
    index=False
)


# ============================================================
# Save PRIVATE EVALUATION artifact
# ============================================================

EVAL_OUTPUT = (
    OUT_DIR /
    "adult_1sa_evaluation_dataset.csv"
)

evaluation.to_csv(
    EVAL_OUTPUT,
    index=False
)


# ============================================================
# Save package specification
# ============================================================

SPEC_OUTPUT = (
    OUT_DIR /
    "adult_1sa_public_evaluation_specification.txt"
)

with open(
    SPEC_OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "KC-SLICE ADULT 1-SA — PUBLIC / EVALUATION SPECIFICATION\n"
    )
    f.write("=" * 70 + "\n\n")

    f.write(
        "PUBLIC Bqa:\n"
    )
    f.write(
        "  Contains only the five sliced QID columns.\n"
    )
    f.write(
        "  Internal row/bucket bookkeeping is removed.\n\n"
    )

    f.write(
        "PUBLIC Bsa:\n"
    )
    f.write(
        "  Contains bucket, SID, SA, and grouped cell representation.\n\n"
    )

    f.write(
        "PRIVATE EVALUATION DATASET:\n"
    )
    f.write(
        "  Contains anonymized QID slices plus the original income "
        "target for downstream utility evaluation.\n"
    )
    f.write(
        "  Source-row identifiers are retained only as internal "
        "evaluation join keys.\n\n"
    )

    f.write(
        "This prevents internal bookkeeping and private ground-truth "
        "labels from being confused with the KC-Slice published release.\n"
    )


# ============================================================
# Final summary
# ============================================================

print()
print("=" * 70)
print("PACKAGE CREATED")
print("=" * 70)

print(
    f"Public Bqa : {BQA_PUBLIC_OUTPUT}"
)

print(
    f"Public Bsa : {BSA_PUBLIC_OUTPUT}"
)

print(
    f"Evaluation : {EVAL_OUTPUT}"
)

print(
    f"Specification: {SPEC_OUTPUT}"
)

print()
print("=" * 70)
print("KC-SLICE ADULT 1-SA PUBLIC/EVALUATION PACKAGE COMPLETE")
print("=" * 70)
