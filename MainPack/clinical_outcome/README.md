# Clinical-outcome classification

The five assay inputs are AS, PSR, ACSINS, ELISA and BVP. The 112-antibody internal cohort is split into 89 training and 23 test rows; the temporally external cohort has 14 resolved outcomes. The main head is MLP, with two-class softmax argmax for Approved vs Terminated.

```bash
python acet.py train clinical_outcome --check-inputs
python acet.py train clinical_outcome --output-dir reruns/clinical_training
python acet.py clinical-results --output-dir reruns/clinical_tables
python acet.py figure4 --output-dir reruns/figure4
```

Run from the root. The training command fits classifiers; `clinical-results` recalculates stored predictions, thresholds, uncertainty and illustrative portfolio summaries; `figure4` uses MATLAB for the publication layout.

Reference predictions, fold assignments, training histories and operating thresholds are under `results/targeted_repro`; the final S14 rule comparison is under `results/finalization`. The economic calculation is an assumption-based illustration of classifier behavior, not a forecast of realized revenue or approval probability. Full assumptions and values are described in Table 1 and the main guide.
