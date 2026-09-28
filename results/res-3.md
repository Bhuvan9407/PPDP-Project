# res-3 — Milestone 3 Baseline Reproduction

## Status

Complete

## Dataset

Adult Census Income

## Configuration

- k = 10
- Sensitive attribute = `income`
- QIDs:
  - `age`
  - `workclass`
  - `education.num`
  - `marital.status`
  - `occupation`
  - `race`
  - `sex`
  - `native.country`
- Preprocessed records = 30,162

## Baseline Results

| Method | DM | Equivalence Classes | Min Class | Max Class | Mean Class |
|---|---:|---:|---:|---:|---:|
| Mondrian | 1,248,010 | 1,388 | 10 | 319 | 21.73 |
| ARX | 181,474,786 | 30 | 10 | 10,921 | 1005.40 |
| KC-Slice | 84,338* | 20,910 | 1 | 24 | 1.44 |

* KC-Slice DM is an adapted diagnostic over the complete released
sliced-QID tuple and is not ordinary k-anonymity DM.

## KC-Slice Privacy Result

- HSA = `<=50K`
- C = 5%
- Internal bucket size = 15,081
- Suppressed rows = 21,146
- Suppression rate = 70.1081%

## Required Milestone 3 Artifacts

- [Baseline results](milestone3_baseline.csv)
- [Reproduction note](milestone3_reproduction_note.txt)

## Implementation and Reproduction Source

- [Adult hierarchies](../hierarchies/adult_hierarchies.yaml)
- [KC-Slice implementation](../methods/kc_slice/algorithm.py)
- [KC-Slice evaluation](../methods/kc_slice/evaluate.py)
- [Baseline runner](../scripts/run_milestone3_baselines.py)
- [Source manifest](../scripts/milestone3_source_manifest.txt)
- [Milestone 3 notebook](../notebooks/KC_Slicing_Project_Milestone3.ipynb)

## Verification

Python syntax audit passed for:

- `methods/kc_slice/algorithm.py`
- `methods/kc_slice/evaluate.py`
- `scripts/run_milestone3_baselines.py`

The implementation, hierarchy, baseline results, reproduction note,
runner, and notebook are committed to the repository.
