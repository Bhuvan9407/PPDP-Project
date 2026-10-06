# Milestone 5 — Expected Findings Verdicts

This document is generated directly from the committed Milestone 4 54-run results table.

**Ranking convention:** higher mean accuracy is better; lower mean DM is better. DM is the common information-loss metric used for cross-method comparison. NCP is reported only where the committed run persisted it.

## 1. Accuracy ranking — Adult Census Income

- **SA=1:** ARX > KC-Slice > Mondrian
- **SA=2:** ARX > KC-Slice > Mondrian
- **SA=3:** ARX > KC-Slice > Mondrian

### Verdict

ARX is the accuracy leader at all three attribute counts, with mean accuracy of 76.94%, 76.93%, and 76.93% for SA=1, 2, and 3 respectively. KC-Slice remains around 75.10%–75.09%, while Mondrian remains lowest at 66.97%–67.45%.

The ranking therefore **holds** as the number of protected attributes increases: ARX > KC-Slice > Mondrian.

## 2. Accuracy ranking — Diabetes-130

- **SA=1:** ARX > Mondrian > KC-Slice
- **SA=2:** ARX > Mondrian > KC-Slice
- **SA=3:** ARX > Mondrian > KC-Slice

### Verdict

Mondrian and ARX are tied at the original-data baseline accuracy of 53.38% for all three SA levels. KC-Slice is lower at approximately 52.02%, 52.03%, and 51.67%.

Thus the Diabetes utility ranking is effectively **Mondrian ≈ ARX > KC-Slice** at every attribute count; there is no ranking reversal as SA count increases.

## 3. Information loss — Discernibility Metric (DM)

DM is interpreted as the discernibility penalty DM = Σ|E|². Lower is better. The absolute values should be interpreted within the dataset/experiment because the three methods do not use identical privacy mechanisms; in particular, the KC-Slice DM is an adapted diagnostic over released sliced-QID tuples.

### Adult

- **SA=1:** lowest-DM order Mondrian > ARX > KC-Slice; Mondrian=460,399, ARX=125,456,315, KC-Slice=291,104,321
- **SA=2:** lowest-DM order Mondrian > ARX > KC-Slice; Mondrian=497,859, ARX=80,750,558, KC-Slice=83,172,663
- **SA=3:** lowest-DM order Mondrian > KC-Slice > ARX; Mondrian=532,210, KC-Slice=12,129,357, ARX=80,750,558

### Diabetes

- **SA=1:** lowest-DM order KC-Slice > Mondrian > ARX; KC-Slice=8,913,467, Mondrian=508,090,563, ARX=1,140,883,741
- **SA=2:** lowest-DM order KC-Slice > Mondrian > ARX; KC-Slice=4,398,459, Mondrian=508,090,563, ARX=953,382,869
- **SA=3:** lowest-DM order KC-Slice > Mondrian > ARX; KC-Slice=2,856,694, Mondrian=508,090,563, ARX=953,382,869

## 4. Why did Mondrian underperform on Adult?

The results show a persistent utility gap rather than a single-seed anomaly. Mondrian's mean Adult accuracy is 66.97%, 67.06%, and 67.45% for SA=1, 2, and 3, whereas ARX remains near 76.9% and KC-Slice near 75.1%.

A defensible interpretation is that the particular multidimensional Mondrian partitioning used here removed or coarsened predictive structure in the released QIDs more harmfully than the ARX optimization did. The experiment does not justify saying that Mondrian is inherently worse than ARX; the conclusion is specific to this Adult configuration, hierarchy, k=10 setting, and locked utility protocol.

An important supporting observation is that DM alone does not predict utility: ARX has much larger DM than Mondrian in these Adult runs, yet ARX has substantially higher classification accuracy. Therefore the structure and location of generalization in feature space matter more for predictive utility than the scalar DM value alone.

## 5. Why was the slicing/generalization advantage nullified on Diabetes-130?

The strongest clue is the locked original-data baseline: Diabetes accuracy is only about 53.38%, with macro-F1 around 0.232 and ROC-AUC around 0.50. This indicates that the locked Logistic Regression evaluation is already operating with very limited discriminative utility on the selected feature representation; accuracy is dominated by the majority-class behavior.

Consequently, Mondrian and ARX matching the 53.38% baseline does not demonstrate a large preservation advantage. There is little measured predictive utility available for anonymization to preserve. Their generalization therefore appears utility-neutral under this classifier/metric.

KC-Slice does not obtain an accuracy advantage either. Its accuracy is below baseline at every SA count, while its ROC-AUC remains approximately 0.50. The slicing mechanism therefore does not recover useful target discrimination under this locked evaluation; instead, it introduces a small additional utility loss.

## 6. Final expected-findings verdict table

| Guide requirement | Verdict | Evidence |
|---|---|---|
| Accuracy vs sensitive-attribute count | ARX > KC-Slice > Mondrian on Adult; Mondrian ≈ ARX > KC-Slice on Diabetes | Rankings remain stable across SA=1,2,3 |
| Information loss vs sensitive-attribute count | DM changes strongly with method and SA count; lower DM is better, but DM is not a direct utility proxy | Common DM is available for all 54 runs; NCP is not persisted comparably for all methods |
| Adult Mondrian underperformance | Persistent and configuration-specific | Lowest Adult accuracy at all three SA levels |
| Diabetes slicing/generalization advantage | Nullified under the locked utility protocol | Original, Mondrian, and ARX all sit at 53.38%; KC-Slice is lower |

### Methodological limitation

NCP is not a complete cross-method metric in the committed Milestone 4 table: it is persisted for Mondrian but not for ARX/KC-Slice. The primary Milestone 5 information-loss plot therefore uses the common DM field. Any NCP-only discussion is explicitly limited to runs where NCP was persisted.
