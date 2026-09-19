# Figure 4: clinical classification and portfolio illustration

The package-relative MATLAB function `Figure4_current_authoritative(packageRoot, outputDir)` reads the stored clinical predictions and final assay-threshold sources, then assembles the panels. From the root:

```bash
python acet.py clinical-results --output-dir reruns/clinical_tables
python acet.py figure4 --output-dir reruns/figure4
```

The Python command recalculates numerical tables; the MATLAB command uses the preserved figure source and exports to a new directory. `data/` contains the MAT and machine-readable source values, while `rendered/` contains the complete figure exports. `MainPack/clinical_outcome/results/targeted_repro` holds the underlying prediction/threshold records, and `results/finalization` holds the final Table S14 rule comparison.

The retrospective economic illustration uses the explicit Table 1 assumptions. It should be interpreted separately from the classifier's empirical prediction accuracy.
