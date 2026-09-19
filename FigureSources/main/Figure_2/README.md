# Figure 2: viscosity

- Panel a: main held-out parity, using `MainPack/viscosity/results/primary`.
- Panel b: bootstrap-bin summaries and prediction resamples; `python acet.py figure2b-data` regenerates the numeric sources.
- Panel c: fixed matched-model comparisons; `python acet.py figure2c-tables` reconstructs the numeric CSV/MAT sources.
- Panels d/e: permutation importance, paired comparisons, nested learning curves and assay ablations; `python acet.py figure2de-plots` renders the archived sources with MATLAB.

For full d/e model fitting use `python acet.py figure2de-train`. The root guide distinguishes plotting from training. All new outputs go to a new run directory. `Publication_EPS` contains the final main-panel EPS set, including the transparency-consistent a/b exports.
