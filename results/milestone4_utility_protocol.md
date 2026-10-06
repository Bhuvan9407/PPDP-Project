# Milestone 4 Utility / Accuracy Evaluation Protocol

## Locked evaluation

The experiment matrix contains exactly 54 main runs: 3 methods × 2 datasets × 3 sensitive-attribute-count levels × 3 seeds.

For every run, the downstream classifier is the same Logistic Regression configuration used by the project's utility evaluation work:

- `LogisticRegression(max_iter=1000)`
- categorical anonymized features use `OneHotEncoder(handle_unknown="ignore")`
- the locked train/test split from the dataset-specific split manifest is used
- Adult target: `income`
- Diabetes target: `readmitted`

The final 54-row table records:

- discernibility metric
- realized equivalence-class size distribution
- minimum / maximum / mean class size
- downstream accuracy
- macro-F1
- ROC-AUC
- original-data accuracy for the same seed
- accuracy delta against the original-data reference

## Holdout-transform rule

The main-run artifacts contain the anonymized training partition only. They do not contain a separately anonymized held-out test table. The utility evaluation therefore uses an explicit train-fitted holdout transform rather than silently re-anonymizing the test partition independently.

### Mondrian

Each held-out row is first checked for exact containment in one persisted training equivalence class using the class's persisted generalized QID representation. Because those persisted classes cover TRAINING rows only, a held-out row can legitimately fall outside every class cell (for example because a categorical combination is unseen in training). Such rows are assigned to the nearest persisted class using the deterministic token-projection distance defined in Cell 208; ties are resolved by the smallest persisted class index. The held-out feature row then receives that persisted class's generalized QID values. No independent test-partition anonymization is performed.

### ARX

Each held-out row is first checked for exact containment in one persisted ARX QID equivalence class. When no persisted class contains the row, the row is projected to the nearest persisted training class using the same deterministic token-projection distance and class-index tie-break. This preserves the train-fitted evaluation protocol and avoids independently anonymizing the held-out partition.

### KC-Slice

The persisted training bucket allocation in the committed main-run artifact is authoritative; utility evaluation does not reconstruct or alter the already-committed training assignment. Held-out rows are then assigned using a frozen inductive extension that does not change any training assignment. Persisted per-bucket QID grouping and seed-derived permutation rules are applied to construct the held-out `slice_*` features.

This is an explicit project operationalization because the persisted main-run artifacts contain training rows only and do not define a separate held-out test-row transformation.

## Integrity gates

Before evaluation:

- all 54 required main-run packages must contain exactly 10 artifacts
- each run uses its locked matrix row
- each seed uses its locked train/test manifest
- each method's realized class sizes are recomputed from its persisted artifacts
- persisted discernibility metrics are cross-checked against realized class sizes where present

After evaluation:

- Adult checkpoint = 27 rows
- Diabetes checkpoint = 27 rows
- final results table = 54 rows
- no duplicate run IDs
- every matrix run ID is represented exactly once
