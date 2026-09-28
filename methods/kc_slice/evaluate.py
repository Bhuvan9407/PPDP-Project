"""
KC-SLICE EVALUATION — MILESTONE 3

This file is an auditable source extraction from the executed
Milestone 3 Colab notebook.

The code below is preserved from the notebook execution history
rather than rewritten. The source cell numbers are included so
the implementation can be traced back to the experimental run.
"""


########################################################################
# SOURCE NOTEBOOK EXECUTION CELL 39
########################################################################

# Cell 39 — KC-SLICE ADULT 1-SA
# QUANTITATIVE BASELINE EVALUATION
# ============================================================

from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)

ROOT = Path("/content/kc-slice")

BASELINE_INPUT = (
    ROOT /
    "data/processed/"
    "adult_standard_baseline_k10_1sa.csv"
)

EVAL_INPUT = (
    ROOT /
    "results/kc_slice_adult_1sa/"
    "adult_1sa_evaluation_dataset.csv"
)

PRIVACY_INPUT = (
    ROOT /
    "results/kc_slice_adult_1sa/"
    "adult_1sa_privacy_report.csv"
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
# Experimental parameters
# ============================================================

TEST_SIZE = 0.20
SPLIT_SEED = 42

TARGET = "income"

ORIGINAL_QIDS = [
    "age",
    "workclass",
    "education.num",
    "marital.status",
    "occupation",
    "race",
    "sex",
    "native.country",
]

KC_FEATURES = [
    "slice_1",
    "slice_2",
    "slice_3",
    "slice_4",
    "slice_5",
]

PRIVACY_C = 5.0


print("=" * 70)
print("KC-SLICE — ADULT 1-SA")
print("QUANTITATIVE BASELINE EVALUATION")
print("=" * 70)


# ============================================================
# Load data
# ============================================================

baseline = pd.read_csv(
    BASELINE_INPUT
)

evaluation = pd.read_csv(
    EVAL_INPUT
)

privacy = pd.read_csv(
    PRIVACY_INPUT
)


assert len(baseline) == 30162
assert len(evaluation) == 30162

print()
print(f"Baseline rows       : {len(baseline):,}")
print(f"KC-Slice eval rows  : {len(evaluation):,}")


# ============================================================
# PART 1 — PRIVACY / DISTORTION
# ============================================================

print()
print("-" * 70)
print("1. PRIVACY / DISTORTION")
print("-" * 70)


# ------------------------------------------------------------
# Suppression statistics
# ------------------------------------------------------------

total_suppressed = int(
    privacy["suppressed_count"].sum()
)

total_records = len(
    baseline
)

suppression_rate = (
    total_suppressed /
    total_records *
    100
)


# ------------------------------------------------------------
# Minimal Distortion
#
# From the KC-Slice paper:
# distortion count increments with each generalization
# or suppression operation.
#
# Our 1-SA implementation performs only HSA suppression,
# one operation per suppressed HSA record.
# ------------------------------------------------------------

minimal_distortion = (
    total_suppressed
)


# ------------------------------------------------------------
# Loss Metric
#
# Paper:
#
# LM = (M - 1) / (|A| - 1)
#
# For binary income:
#
# |A| = 2
#
# A suppressed value "#####" can represent either value,
# therefore M = 2.
#
# Suppressed record loss = 1
# Unsuppressed record loss = 0
#
# Thus average LM = suppression rate.
# ------------------------------------------------------------

income_domain_size = 2
suppressed_ambiguity = 2

suppressed_lm = (
    (suppressed_ambiguity - 1) /
    (income_domain_size - 1)
)

assert suppressed_lm == 1.0

loss_metric = (
    total_suppressed *
    suppressed_lm /
    total_records
)


print(
    f"Total HSA suppressions : "
    f"{total_suppressed:,}"
)

print(
    f"Suppression rate       : "
    f"{suppression_rate:.4f}%"
)

print(
    f"Minimal Distortion     : "
    f"{minimal_distortion:,}"
)

print(
    f"Loss Metric (income)   : "
    f"{loss_metric:.6f}"
)


# ------------------------------------------------------------
# Bucket privacy verification
# ------------------------------------------------------------

max_hsa_percent = (
    privacy["after_percent"]
    .max()
)

min_hsa_percent = (
    privacy["after_percent"]
    .min()
)

assert (
    max_hsa_percent
    <=
    PRIVACY_C + 1e-9
)


print(
    f"Maximum post-check HSA %: "
    f"{max_hsa_percent:.6f}%"
)

print(
    f"Minimum post-check HSA %: "
    f"{min_hsa_percent:.6f}%"
)

print(
    "Privacy threshold verification: PASS"
)


# ============================================================
# PART 2 — FIXED TRAIN/TEST SPLIT
# ============================================================

print()
print("-" * 70)
print("2. FIXED TRAIN/TEST SPLIT")
print("-" * 70)


# Use original baseline row numbers as the permanent split key.
all_indices = np.arange(
    len(baseline)
)

train_indices, test_indices = (
    train_test_split(
        all_indices,
        test_size=TEST_SIZE,
        random_state=SPLIT_SEED,
        stratify=baseline[TARGET],
    )
)

train_indices = np.sort(
    train_indices
)

test_indices = np.sort(
    test_indices
)


print(
    f"Train rows : {len(train_indices):,}"
)

print(
    f"Test rows  : {len(test_indices):,}"
)

print(
    f"Split      : {100*(1-TEST_SIZE):.0f}/"
    f"{100*TEST_SIZE:.0f}"
)

print(
    f"Random seed: {SPLIT_SEED}"
)


# Verify complete partition.

assert len(
    np.intersect1d(
        train_indices,
        test_indices
    )
) == 0

assert (
    len(train_indices)
    +
    len(test_indices)
    ==
    len(baseline)
)

print(
    "Train/test partition: PASS"
)


# Save split manifest so all later methods use exactly this split.

split_manifest = {
    "dataset": "Adult",
    "records": len(baseline),
    "test_size": TEST_SIZE,
    "random_state": SPLIT_SEED,
    "train_rows": len(train_indices),
    "test_rows": len(test_indices),
}

with open(
    OUT_DIR /
    "adult_fixed_train_test_split.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        split_manifest,
        f,
        indent=2
    )

np.save(
    OUT_DIR /
    "adult_train_indices.npy",
    train_indices
)

np.save(
    OUT_DIR /
    "adult_test_indices.npy",
    test_indices
)


# ============================================================
# PART 3 — ORIGINAL-DATA LOGISTIC REGRESSION
# ============================================================

print()
print("-" * 70)
print("3. ORIGINAL-DATA LOGISTIC REGRESSION")
print("-" * 70)


X_original = baseline[
    ORIGINAL_QIDS
].copy()

y_original = baseline[
    TARGET
].copy()


X_train_original = X_original.iloc[
    train_indices
]

X_test_original = X_original.iloc[
    test_indices
]

y_train = y_original.iloc[
    train_indices
]

y_test = y_original.iloc[
    test_indices
]


numeric_features = [
    "age",
    "education.num",
]

categorical_features = [
    "workclass",
    "marital.status",
    "occupation",
    "race",
    "sex",
    "native.country",
]


original_preprocessor = (
    ColumnTransformer(
        transformers=[
            (
                "num",
                "passthrough",
                numeric_features
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            ),
        ]
    )
)


original_model = Pipeline([
    (
        "preprocess",
        original_preprocessor
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000
        )
    ),
])


original_model.fit(
    X_train_original,
    y_train
)

original_predictions = (
    original_model.predict(
        X_test_original
    )
)

original_probabilities = (
    original_model.predict_proba(
        X_test_original
    )[:, 1]
)


original_accuracy = accuracy_score(
    y_test,
    original_predictions
)

original_macro_f1 = f1_score(
    y_test,
    original_predictions,
    average="macro"
)

original_auc = roc_auc_score(
    y_test,
    original_probabilities
)


print(
    f"Accuracy : {original_accuracy:.6f}"
)

print(
    f"Macro-F1 : {original_macro_f1:.6f}"
)

print(
    f"ROC-AUC  : {original_auc:.6f}"
)


# ============================================================
# PART 4 — KC-SLICE LOGISTIC REGRESSION
# ============================================================

print()
print("-" * 70)
print("4. KC-SLICE LOGISTIC REGRESSION")
print("-" * 70)


# ------------------------------------------------------------
# Important:
#
# KC-Slice publishes five concatenated QID slices.
# The public table does not expose the original QID labels
# after permutation.
#
# Therefore each slice is treated as a categorical released
# feature exactly as it appears in Bqa.
# ------------------------------------------------------------

X_kc = evaluation[
    KC_FEATURES
].copy()

y_kc = evaluation[
    TARGET
].copy()


X_train_kc = X_kc.iloc[
    train_indices
]

X_test_kc = X_kc.iloc[
    test_indices
]

y_train_kc = y_kc.iloc[
    train_indices
]

y_test_kc = y_kc.iloc[
    test_indices
]


kc_preprocessor = (
    ColumnTransformer(
        transformers=[
            (
                "slices",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                KC_FEATURES
            ),
        ]
    )
)


kc_model = Pipeline([
    (
        "preprocess",
        kc_preprocessor
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000
        )
    ),
])


kc_model.fit(
    X_train_kc,
    y_train_kc
)

kc_predictions = (
    kc_model.predict(
        X_test_kc
    )
)

kc_probabilities = (
    kc_model.predict_proba(
        X_test_kc
    )[:, 1]
)


kc_accuracy = accuracy_score(
    y_test_kc,
    kc_predictions
)

kc_macro_f1 = f1_score(
    y_test_kc,
    kc_predictions,
    average="macro"
)

kc_auc = roc_auc_score(
    y_test_kc,
    kc_probabilities
)


print(
    f"Accuracy : {kc_accuracy:.6f}"
)

print(
    f"Macro-F1 : {kc_macro_f1:.6f}"
)

print(
    f"ROC-AUC  : {kc_auc:.6f}"
)


# ============================================================
# PART 5 — UTILITY DIFFERENCE
# ============================================================

print()
print("-" * 70)
print("5. UTILITY COMPARISON")
print("-" * 70)


accuracy_delta = (
    kc_accuracy
    -
    original_accuracy
)

macro_f1_delta = (
    kc_macro_f1
    -
    original_macro_f1
)

auc_delta = (
    kc_auc
    -
    original_auc
)


print(
    f"Accuracy delta : "
    f"{accuracy_delta:+.6f}"
)

print(
    f"Macro-F1 delta : "
    f"{macro_f1_delta:+.6f}"
)

print(
    f"ROC-AUC delta  : "
    f"{auc_delta:+.6f}"
)


# ============================================================
# PART 6 — SAVE RESULTS
# ============================================================

results = pd.DataFrame([
    {
        "dataset": "Adult",
        "method": "Original",
        "sa_count": 1,
        "project_k": 10,
        "kc_bucket_size": 15081,
        "C": 5.0,
        "suppression_rate": 0.0,
        "minimal_distortion": 0,
        "loss_metric_income": 0.0,
        "accuracy": original_accuracy,
        "macro_f1": original_macro_f1,
        "roc_auc": original_auc,
        "split_seed": SPLIT_SEED,
    },
    {
        "dataset": "Adult",
        "method": "KC-Slice",
        "sa_count": 1,
        "project_k": 10,
        "kc_bucket_size": 15081,
        "C": 5.0,
        "suppression_rate": suppression_rate,
        "minimal_distortion": minimal_distortion,
        "loss_metric_income": loss_metric,
        "accuracy": kc_accuracy,
        "macro_f1": kc_macro_f1,
        "roc_auc": kc_auc,
        "split_seed": SPLIT_SEED,
    },
])


RESULT_OUTPUT = (
    OUT_DIR /
    "adult_1sa_quantitative_evaluation.csv"
)

results.to_csv(
    RESULT_OUTPUT,
    index=False
)


# Save readable summary.

SUMMARY_OUTPUT = (
    OUT_DIR /
    "adult_1sa_quantitative_evaluation.txt"
)

with open(
    SUMMARY_OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "KC-SLICE ADULT 1-SA — QUANTITATIVE EVALUATION\n"
    )
    f.write("=" * 70 + "\n\n")

    f.write(
        "Privacy / distortion:\n"
    )

    f.write(
        f"  HSA suppressions : {total_suppressed:,}\n"
    )

    f.write(
        f"  Suppression rate : {suppression_rate:.6f}%\n"
    )

    f.write(
        f"  Minimal distortion: {minimal_distortion:,}\n"
    )

    f.write(
        f"  Loss metric      : {loss_metric:.6f}\n\n"
    )

    f.write(
        "Fixed split:\n"
    )

    f.write(
        f"  Test fraction    : {TEST_SIZE:.2f}\n"
    )

    f.write(
        f"  Random state     : {SPLIT_SEED}\n\n"
    )

    f.write(
        "Original logistic regression:\n"
    )

    f.write(
        f"  Accuracy         : {original_accuracy:.6f}\n"
    )

    f.write(
        f"  Macro-F1         : {original_macro_f1:.6f}\n"
    )

    f.write(
        f"  ROC-AUC          : {original_auc:.6f}\n\n"
    )

    f.write(
        "KC-Slice logistic regression:\n"
    )

    f.write(
        f"  Accuracy         : {kc_accuracy:.6f}\n"
    )

    f.write(
        f"  Macro-F1         : {kc_macro_f1:.6f}\n"
    )

    f.write(
        f"  ROC-AUC          : {kc_auc:.6f}\n\n"
    )

    f.write(
        "KC-Slice minus Original:\n"
    )

    f.write(
        f"  Accuracy delta   : {accuracy_delta:+.6f}\n"
    )

    f.write(
        f"  Macro-F1 delta   : {macro_f1_delta:+.6f}\n"
    )

    f.write(
        f"  ROC-AUC delta    : {auc_delta:+.6f}\n"
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("ADULT 1-SA QUANTITATIVE EVALUATION COMPLETE")
print("=" * 70)

print(
    f"Results : {RESULT_OUTPUT}"
)

print(
    f"Summary : {SUMMARY_OUTPUT}"
)

print()
print("=" * 70)

########################################################################
# SOURCE NOTEBOOK EXECUTION CELL 43
########################################################################

# Cell 43 — MILESTONE 3
# ADULT 1-SA — UNIFIED THREE-METHOD EVALUATION
# ============================================================

from pathlib import Path
import re

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path("/content/kc-slice")

BASELINE_PATH = (
    ROOT /
    "data/processed/"
    "adult_standard_baseline_k10_1sa.csv"
)

MONDRIAN_PATH = (
    ROOT /
    "results/"
    "adult_standard_mondrian_k10_1sa.csv"
)

ARX_PATH = (
    ROOT /
    "results/arx_standard_adult_vgh/"
    "adult_standard_arx_vgh_k10.csv"
)

KC_PATH = (
    ROOT /
    "results/kc_slice_adult_1sa/"
    "adult_1sa_evaluation_dataset.csv"
)

KC_EVAL_RESULTS = (
    ROOT /
    "results/kc_slice_adult_1sa/"
    "adult_1sa_quantitative_evaluation.csv"
)

MONDRIAN_SUMMARY = (
    ROOT /
    "results/"
    "milestone3_mondrian_standard_8qid_rowlevel.txt"
)

ARX_SUMMARY = (
    ROOT /
    "results/"
    "milestone3_arx_vgh_baseline.txt"
)

OUT_DIR = (
    ROOT /
    "results/milestone3_adult_1sa_comparison"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# EXPERIMENT PARAMETERS
# ============================================================

TARGET = "income"

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

MONDRIAN_SOURCE_ID = "_source_row"

SPLIT_SEED = 42

RUN_SEED = 0

KC_SLICE_COLUMNS = [
    "slice_1",
    "slice_2",
    "slice_3",
    "slice_4",
    "slice_5",
]


print("=" * 70)
print("MILESTONE 3 — ADULT 1-SA")
print("UNIFIED THREE-METHOD EVALUATION")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

baseline = pd.read_csv(
    BASELINE_PATH
)

mondrian = pd.read_csv(
    MONDRIAN_PATH
)

arx = pd.read_csv(
    ARX_PATH
)

kc = pd.read_csv(
    KC_PATH
)


assert len(baseline) == 30162
assert len(mondrian) == 30162
assert len(arx) == 30162
assert len(kc) == 30162


print()
print("Dataset loading:")
print("  Baseline : PASS")
print("  Mondrian : PASS")
print("  ARX      : PASS")
print("  KC-Slice : PASS")


# ============================================================
# LOAD FIXED SPLIT
# ============================================================

train_indices = np.load(
    ROOT /
    "results/kc_slice_adult_1sa/"
    "adult_train_indices.npy"
)

test_indices = np.load(
    ROOT /
    "results/kc_slice_adult_1sa/"
    "adult_test_indices.npy"
)


assert len(train_indices) == 24129
assert len(test_indices) == 6033

assert (
    len(
        np.intersect1d(
            train_indices,
            test_indices
        )
    )
    ==
    0
)

assert (
    len(train_indices)
    +
    len(test_indices)
    ==
    30162
)


print()
print("-" * 70)
print("FIXED TRAIN/TEST SPLIT")
print("-" * 70)

print(
    f"Train rows : {len(train_indices):,}"
)

print(
    f"Test rows  : {len(test_indices):,}"
)

print(
    f"Seed       : {SPLIT_SEED}"
)

print(
    "Split verification : PASS"
)


# ============================================================
# ARX TARGET ALIGNMENT AUDIT
#
# ARX output did not previously contain a source-row ID.
# Therefore we must verify that its row order still matches
# the original baseline before using positional alignment.
# ============================================================

print()
print("-" * 70)
print("ARX TARGET ALIGNMENT AUDIT")
print("-" * 70)

assert TARGET in arx.columns

arx_income_matches = (
    arx[TARGET]
    .astype(str)
    .reset_index(drop=True)
    ==
    baseline[TARGET]
    .astype(str)
    .reset_index(drop=True)
)

print(
    "ARX income column present : PASS"
)

print(
    f"Matching target rows      : "
    f"{arx_income_matches.sum():,} / {len(arx):,}"
)

assert arx_income_matches.all()

print(
    "ARX row-order target alignment: PASS"
)


# ============================================================
# MONDRIAN TARGET ALIGNMENT AUDIT
# ============================================================

print()
print("-" * 70)
print("MONDRIAN TARGET ALIGNMENT AUDIT")
print("-" * 70)

assert TARGET in mondrian.columns
assert MONDRIAN_SOURCE_ID in mondrian.columns

mondrian_sorted = (
    mondrian
    .sort_values(MONDRIAN_SOURCE_ID)
    .reset_index(drop=True)
)

mondrian_income_matches = (
    mondrian_sorted[TARGET]
    .astype(str)
    ==
    baseline[TARGET]
    .astype(str)
)

assert mondrian_sorted[
    MONDRIAN_SOURCE_ID
].equals(
    pd.Series(
        np.arange(30162),
        name=MONDRIAN_SOURCE_ID
    )
)

assert mondrian_income_matches.all()

print(
    "Mondrian source-row mapping : PASS"
)

print(
    "Mondrian target alignment    : PASS"
)


# ============================================================
# Prepare sorted Mondrian output
# ============================================================

mondrian = mondrian_sorted


# ============================================================
# KC-SLICE TARGET ALIGNMENT AUDIT
# ============================================================

print()
print("-" * 70)
print("KC-SLICE TARGET ALIGNMENT AUDIT")
print("-" * 70)

assert "_source_row" in kc.columns
assert TARGET in kc.columns

kc_sorted = (
    kc
    .sort_values("_source_row")
    .reset_index(drop=True)
)

assert kc_sorted[
    "_source_row"
].equals(
    pd.Series(
        np.arange(30162),
        name="_source_row"
    )
)

kc_income_matches = (
    kc_sorted[TARGET]
    .astype(str)
    ==
    baseline[TARGET]
    .astype(str)
)

assert kc_income_matches.all()

print(
    "KC-Slice source-row mapping : PASS"
)

print(
    "KC-Slice target alignment   : PASS"
)


# ============================================================
# HELPER — ANONYMIZED QID FEATURES
#
# All released QID values are treated as categorical strings.
#
# This matters because:
#   - Mondrian may emit generalized ranges.
#   - ARX may emit generalized hierarchy values.
#   - KC-Slice emits concatenated categorical slices.
#
# We therefore do not assign numerical meaning to anonymized
# representations.
# ============================================================

def build_categorical_logistic_regression(
    feature_columns
):

    preprocessor = (
        ColumnTransformer(
            transformers=[
                (
                    "features",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    ),
                    feature_columns
                )
            ]
        )
    )

    model = Pipeline([
        (
            "preprocess",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000
            )
        ),
    ])

    return model


def evaluate_model(
    X,
    y,
    feature_columns,
    label
):

    X = X[
        feature_columns
    ].copy()

    # All anonymized feature representations are categorical.
    for column in feature_columns:

        X[column] = (
            X[column]
            .astype(str)
        )

    X_train = X.iloc[
        train_indices
    ]

    X_test = X.iloc[
        test_indices
    ]

    y_train = y.iloc[
        train_indices
    ]

    y_test = y.iloc[
        test_indices
    ]

    model = (
        build_categorical_logistic_regression(
            feature_columns
        )
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro"
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    return {
        "method": label,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "roc_auc": roc_auc,
    }


# ============================================================
# COMMON TARGET
# ============================================================

y = (
    baseline[TARGET]
    .astype(str)
)


# ============================================================
# ORIGINAL REFERENCE
#
# For the original dataset we keep:
#   numeric -> numeric
#   categorical -> one-hot categorical
#
# This is the same model configuration used in Cell 39.
# ============================================================

print()
print("-" * 70)
print("1. ORIGINAL DATA REFERENCE")
print("-" * 70)

numeric_features = [
    "age",
    "education.num",
]

categorical_features = [
    "workclass",
    "marital.status",
    "occupation",
    "race",
    "sex",
    "native.country",
]

original_preprocessor = (
    ColumnTransformer(
        transformers=[
            (
                "num",
                "passthrough",
                numeric_features
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            ),
        ]
    )
)

original_model = Pipeline([
    (
        "preprocess",
        original_preprocessor
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000
        )
    ),
])


X_original = baseline[
    QIDS
].copy()

X_train_original = X_original.iloc[
    train_indices
]

X_test_original = X_original.iloc[
    test_indices
]

y_train = y.iloc[
    train_indices
]

y_test = y.iloc[
    test_indices
]

original_model.fit(
    X_train_original,
    y_train
)

original_predictions = (
    original_model.predict(
        X_test_original
    )
)

original_probabilities = (
    original_model.predict_proba(
        X_test_original
    )[:, 1]
)

original_result = {
    "method": "Original",
    "accuracy": accuracy_score(
        y_test,
        original_predictions
    ),
    "macro_f1": f1_score(
        y_test,
        original_predictions,
        average="macro"
    ),
    "roc_auc": roc_auc_score(
        y_test,
        original_probabilities
    ),
}

print(
    f"Accuracy : "
    f"{original_result['accuracy']:.6f}"
)

print(
    f"Macro-F1 : "
    f"{original_result['macro_f1']:.6f}"
)

print(
    f"ROC-AUC  : "
    f"{original_result['roc_auc']:.6f}"
)


# ============================================================
# 2. MONDRIAN
# ============================================================

print()
print("-" * 70)
print("2. MONDRIAN")
print("-" * 70)

mondrian_result = evaluate_model(
    mondrian,
    y,
    QIDS,
    "Mondrian"
)

print(
    f"Accuracy : "
    f"{mondrian_result['accuracy']:.6f}"
)

print(
    f"Macro-F1 : "
    f"{mondrian_result['macro_f1']:.6f}"
)

print(
    f"ROC-AUC  : "
    f"{mondrian_result['roc_auc']:.6f}"
)


# ============================================================
# 3. ARX
# ============================================================

print()
print("-" * 70)
print("3. ARX")
print("-" * 70)

arx_result = evaluate_model(
    arx,
    y,
    QIDS,
    "ARX"
)

print(
    f"Accuracy : "
    f"{arx_result['accuracy']:.6f}"
)

print(
    f"Macro-F1 : "
    f"{arx_result['macro_f1']:.6f}"
)

print(
    f"ROC-AUC  : "
    f"{arx_result['roc_auc']:.6f}"
)


# ============================================================
# 4. KC-SLICE
# ============================================================

print()
print("-" * 70)
print("4. KC-SLICE")
print("-" * 70)

kc_result = evaluate_model(
    kc_sorted,
    y,
    KC_SLICE_COLUMNS,
    "KC-Slice"
)

print(
    f"Accuracy : "
    f"{kc_result['accuracy']:.6f}"
)

print(
    f"Macro-F1 : "
    f"{kc_result['macro_f1']:.6f}"
)

print(
    f"ROC-AUC  : "
    f"{kc_result['roc_auc']:.6f}"
)


# ============================================================
# UTILITY TABLE
# ============================================================

utility_results = pd.DataFrame([
    original_result,
    mondrian_result,
    arx_result,
    kc_result,
])


# ------------------------------------------------------------
# Add deltas relative to original.
# ------------------------------------------------------------

utility_results[
    "accuracy_delta_vs_original"
] = (
    utility_results["accuracy"]
    -
    original_result["accuracy"]
)

utility_results[
    "macro_f1_delta_vs_original"
] = (
    utility_results["macro_f1"]
    -
    original_result["macro_f1"]
)

utility_results[
    "roc_auc_delta_vs_original"
] = (
    utility_results["roc_auc"]
    -
    original_result["roc_auc"]
)


# ============================================================
# PRIVACY / DISTORTION TABLE
# ============================================================

privacy_results = pd.DataFrame([
    {
        "method": "Mondrian",
        "privacy_model": "k-anonymity",
        "k": 10,
        "ncp_percent": 16.7391,
        "equivalence_classes": 1789,
        "minimum_class_size": 10,
        "maximum_class_size": 109,
        "discernibility_metric": 640056,
        "suppression_rate": 0.0,
    },
    {
        "method": "ARX",
        "privacy_model": "k-anonymity",
        "k": 10,
        "ncp_percent": np.nan,
        "equivalence_classes": np.nan,
        "minimum_class_size": np.nan,
        "maximum_class_size": np.nan,
        "discernibility_metric": np.nan,
        "suppression_rate": 0.0,
    },
    {
        "method": "KC-Slice",
        "privacy_model": "KC-Slice C/HSA bucket constraint",
        "k": 10,
        "ncp_percent": np.nan,
        "equivalence_classes": np.nan,
        "minimum_class_size": 15081,
        "maximum_class_size": 15081,
        "discernibility_metric": np.nan,
        "suppression_rate": 70.1081,
    },
])


# ============================================================
# COMPUTE ARX EQUIVALENCE-CLASS STATISTICS
# ============================================================

arx_qid_counts = (
    arx
    .groupby(QIDS, dropna=False)
    .size()
)

privacy_results.loc[
    privacy_results["method"] == "ARX",
    "equivalence_classes"
] = len(arx_qid_counts)

privacy_results.loc[
    privacy_results["method"] == "ARX",
    "minimum_class_size"
] = arx_qid_counts.min()

privacy_results.loc[
    privacy_results["method"] == "ARX",
    "maximum_class_size"
] = arx_qid_counts.max()

privacy_results.loc[
    privacy_results["method"] == "ARX",
    "discernibility_metric"
] = (
    arx_qid_counts ** 2
).sum()


# ============================================================
# READ VERIFIED ARX NCP FROM SUMMARY
# ============================================================

if ARX_SUMMARY.exists():

    arx_text = ARX_SUMMARY.read_text(
        encoding="utf-8",
        errors="replace"
    )

    match = re.search(
        r"NCP.*?([0-9]+\.[0-9]+)",
        arx_text,
        re.IGNORECASE
    )

    if match:

        arx_ncp = float(
            match.group(1)
        )

        privacy_results.loc[
            privacy_results["method"] == "ARX",
            "ncp_percent"
        ] = arx_ncp


# ============================================================
# DISPLAY COMPARISON
# ============================================================

print()
print("=" * 70)
print("ADULT 1-SA — UTILITY COMPARISON")
print("=" * 70)

print(
    utility_results.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


print()
print("=" * 70)
print("ADULT 1-SA — PRIVACY / DISTORTION COMPARISON")
print("=" * 70)

print(
    privacy_results.to_string(
        index=False
    )
)


# ============================================================
# SAVE OUTPUTS
# ============================================================

UTILITY_OUTPUT = (
    OUT_DIR /
    "adult_1sa_unified_utility_results.csv"
)

PRIVACY_OUTPUT = (
    OUT_DIR /
    "adult_1sa_unified_privacy_results.csv"
)

utility_results.to_csv(
    UTILITY_OUTPUT,
    index=False
)

privacy_results.to_csv(
    PRIVACY_OUTPUT,
    index=False
)


# ============================================================
# SAVE FINAL MILESTONE 3 BASELINE SUMMARY
# ============================================================

SUMMARY_OUTPUT = (
    OUT_DIR /
    "adult_1sa_three_method_baseline.txt"
)

with open(
    SUMMARY_OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "MILESTONE 3 — ADULT 1-SA THREE-METHOD BASELINE\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        "Dataset: Adult Census Income\n"
    )

    f.write(
        "Records: 30,162\n"
    )

    f.write(
        "Sensitive attribute: income\n"
    )

    f.write(
        "Project k: 10\n"
    )

    f.write(
        "Utility split: 80/20 stratified\n"
    )

    f.write(
        "Split seed: 42\n\n"
    )

    f.write(
        "UTILITY RESULTS\n"
    )

    f.write(
        utility_results.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    f.write(
        "\n\nPRIVACY / DISTORTION RESULTS\n"
    )

    f.write(
        privacy_results.to_string(
            index=False
        )
    )

    f.write(
        "\n\nAlignment checks:\n"
    )

    f.write(
        "  ARX target alignment: PASS\n"
    )

    f.write(
        "  Mondrian source-row alignment: PASS\n"
    )

    f.write(
        "  KC-Slice source-row alignment: PASS\n"
    )

    f.write(
        "\nStatus: Adult 1-SA three-method baseline comparison complete.\n"
    )


# ============================================================
# FINAL VERIFICATION
# ============================================================

assert set(
    utility_results["method"]
) == {
    "Original",
    "Mondrian",
    "ARX",
    "KC-Slice",
}

assert len(
    utility_results
) == 4

assert len(
    privacy_results
) == 3


print()
print("=" * 70)
print("ADULT 1-SA THREE-METHOD COMPARISON COMPLETE")
print("=" * 70)

print(
    f"Utility results : {UTILITY_OUTPUT}"
)

print(
    f"Privacy results : {PRIVACY_OUTPUT}"
)

print(
    f"Summary         : {SUMMARY_OUTPUT}"
)

print()
print("=" * 70)
