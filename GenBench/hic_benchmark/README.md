# HIC retention-time benchmark

The 152-antibody dataset is evaluated by ten repeats of fivefold OOF cross-validation. Predictions for each antibody are averaged across repeats. Assays-only, patch-only and combined configurations are supplied, together with the fold-matched Bailly patch-linear reference and the published fixed equation comparison.

```bash
python acet.py train hic --hic-features assays_only --check-inputs
python acet.py train hic --hic-features assays_only --output-dir reruns/hic_assays
python acet.py train hic --hic-features patch_only --output-dir reruns/hic_patch
python acet.py train hic --hic-features all --output-dir reruns/hic_combined
python acet.py hic-results --output-dir reruns/hic_tables
```

The training commands fit 50 fold models per configuration. The final command recalculates performance, comparator tables and plots from the archived OOF records without neural training. The data identify 39 observations at the 50-minute ceiling; metrics use that recorded scale, and good/poor triage uses 30 minutes.

The article distinguishes coefficient R² from squared Pearson correlation r². The assays-only reference has coefficient R² about 0.7955 and Pearson r² about 0.8250. Tables S2/S3/S12 and Figures S6–S8 are mapped in the root guide.
