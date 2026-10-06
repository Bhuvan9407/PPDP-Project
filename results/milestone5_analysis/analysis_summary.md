# Milestone 5 — Analysis Summary

## Scope

54 committed Milestone 4 runs:

- 2 datasets: Adult Census Income and Diabetes-130
- 3 methods: Mondrian, ARX, KC-Slice
- 3 sensitive-attribute counts: 1, 2, 3
- 3 seeds per configuration

No anonymization or classifier evaluation is rerun by this milestone.

## Primary metrics

- Accuracy: higher is better.
- Discernibility Metric (DM): lower is better.
- DM is the common information-loss/discernibility quantity available for all 54 runs.
- NCP is retained only where the committed results persist it; it is not used for cross-method ranking.

## Key result

Adult accuracy ranking holds at all three SA levels:
**ARX > KC-Slice > Mondrian**.

Diabetes accuracy ranking also holds:
**Mondrian ≈ ARX > KC-Slice**.

The main cross-dataset difference is therefore not a ranking reversal with increasing SA count, but a change in which methods separate in utility: Adult shows a strong ARX advantage, while Diabetes collapses Mondrian and ARX to the same baseline accuracy.

## Interpretation

The Adult result suggests that the way a method generalizes the QIDs matters more to predictive utility than the scalar DM alone. Mondrian's lower DM does not translate into better accuracy.

The Diabetes result is strongly constrained by the weak locked baseline: the original Logistic Regression already achieves only 53.38% accuracy with approximately 0.50 ROC-AUC. This leaves little measured utility for anonymization to preserve, so Mondrian/ARX matching baseline is best interpreted as utility-neutral under the chosen evaluation, not as evidence that anonymization has no effect on the underlying data.
