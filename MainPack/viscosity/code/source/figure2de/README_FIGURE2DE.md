# Viscosity importance, ablation and learning curves

`viscosity_figure2de_current.py` trains the models used for the Figure 2d/e analyses. It locates the repository through `Data/curated`, reads the main numeric viscosity train/test inputs, and loads the shared viscosity builder.

Use these root commands:

```bash
python acet.py figure2de-preflight --output-dir reruns/figure2de_inputs
python acet.py figure2de-train --output-dir reruns/figure2de_training
```

The preflight validates input schemas and partition membership. Full execution uses the TensorFlow/tfkan/SDV/IBLR training environment. The implementation measures held-out permutation importance, compares the full/Top2/HT-assay/No-kD panels, and evaluates nested development sizes while retaining the fixed test cohort. Preserved outputs are under `MainPack/viscosity/results/figure2de`.

For publication-style plots from the archived numeric sources, use `python acet.py figure2de-plots`; this calls the MATLAB plotter in `FigureSources/main/Figure_2/panels_d_e/code`.
