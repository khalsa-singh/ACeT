# Robustness analyses

`fixed_holdout_stability/results/locked` retains the five-training-seed predictions on unchanged regression holdouts. `split_sensitivity/results/locked` retains the six composite-partition inputs and outputs. The alternate-partition prediction and metric filenames explicitly name that evaluation scope.

The alternate-partition prediction rows are required by Supplementary Figures S1–S4. They are prediction instances across different test cohorts, not independent new antibodies. The final fixed/alternate summaries are in `TABLE_S13_COMBINED_REGRESSION_ROWS.csv`; do not substitute pooled changing-cohort R² for the fixed-holdout results.

```bash
python acet.py fixed-holdout-preflight --output-dir reruns/fixed_inputs
python acet.py fixed-holdout-train --output-dir reruns/fixed_training
python acet.py split-sensitivity-preflight --output-dir reruns/split_inputs
python acet.py split-sensitivity-train --output-dir reruns/split_training
python acet.py supplementary-figures --output-dir reruns/supplementary_figures
```

Run from the root. Training needs the neural stack; the final plotting command needs MATLAB. S5 external-clinical figure assembly uses the three included panel images and the published aggregate summary; its full per-split neural checkpoints are not supplied. See the root guide for the coverage of each supplementary item.
