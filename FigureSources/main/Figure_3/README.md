# Figure 3: mouse IV exposure

Panel a is the primary parity/error-CDF result. Panel b is the separate paired augmentation comparison. Panel c is the matched fixed-comparator evaluation. Panel d is the KernelSHAP interpretation. Keep these source records separate because they correspond to different analysis runs.

`python acet.py figure3-plots --output-dir reruns/figure3` calls the package-relative MATLAB assembly. `python acet.py figure3b-data` extracts the stored paired plotting arrays without retraining. `python acet.py demo` produces a lightweight parity/CDF example from the primary predictions.

The correct primary CDF data are in `panel_a/data`. The main guide documents the ordered-error markers and the distinction between the main, matched-comparator, augmentation and development-only results.
