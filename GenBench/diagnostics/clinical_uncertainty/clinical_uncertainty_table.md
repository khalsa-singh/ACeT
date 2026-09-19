# Clinical uncertainty table

Source: manuscript-reported confusion counts were used because no existing probability file was found in the shipped outputs searched. No model training was performed. Wilson intervals are shown for sensitivity and specificity; nonparametric bootstrap percentile intervals are shown for balanced accuracy and MCC (50,000 resamples, seed 20260630).

| Cohort | TP | FN | FP | TN | Sensitivity (Wilson 95% CI) | Specificity (Wilson 95% CI) | Balanced accuracy (bootstrap 95% CI) | MCC (bootstrap 95% CI) | Log-loss | Brier score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| Internal test | 10 | 3 | 2 | 8 | 0.769 (0.497-0.918) | 0.800 (0.490-0.943) | 0.785 (0.602-0.944) | 0.565 (0.195-0.906) | not computed: probability file not found | not computed: probability file not found |
| External cohort | 2 | 0 | 4 | 8 | 1.000 (0.342-1) | 0.667 (0.391-0.862) | 0.833 (0.692-0.958) | 0.471 (0.240-0.849) | not computed: probability file not found | not computed: probability file not found |
