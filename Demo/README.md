# Mouse IV exposure quick start

This example turns the archived 11-antibody mouse-exposure predictions into a metric summary and a parity/error-CDF graphic. It is a fast way to inspect the data flow and expected result without installing TensorFlow.

## Run from the repository root

```bash
python acet.py demo
```

Outputs are written to `reruns/demo/`: a JSON metric summary and a self-contained SVG. The calculation and SVG generation use Python's standard library.

For PNG and JPG as well:

```bash
python -m pip install -r requirements-demo.txt
python acet.py demo --formats svg,png,jpg --output-dir reruns/demo_images
```

Pillow provides the portable raster renderer. When CairoSVG and the Cairo system library are installed, the default `auto` setting uses CairoSVG; otherwise it uses the Pillow renderer for this demonstration's simple SVG elements. To choose explicitly:

```bash
python acet.py demo --formats svg,png,jpg --raster-engine pillow --output-dir reruns/demo_pillow
```

The PNG/JPG exports are 2940 × 1500 pixels. SVG remains the vector master. Example outputs are in `clearance_quickstart/expected_output/`.

## Inputs, units and expected results

The input is `FigureSources/main/Figure_3/panel_a/data/figure3a_current_source_predictions.csv`. AUCt is stored in units of 10⁴ ng·h/mL; divide by 100 for the 10⁶ ng·h/mL display scale in the article. Expected R² is about 0.810019 and nRMSE about 0.143226. The graphic uses logarithmic parity axes for this demonstration; it is not a pixel-for-pixel replacement of the linear-axis publication panel.

CDF markers reproduce the article's ordered-error convention: the sixth, eighth and tenth sorted errors among eleven observations are about 6.01%, 16.58% and 40.42%. These markers are not substituted with interpolated NumPy percentiles.

## From the demonstration to full training

The demonstration reads existing predictions. To obtain a new model fit, activate the Python 3.11 training environment described in the [main guide](../README.md#installation), then run:

```bash
python acet.py train mouse_exposure --check-inputs
python acet.py train mouse_exposure --output-dir reruns/mouse_training
```

Other main endpoint fits are `train viscosity`, `train clinical_outcome`, and `train hic`. Development-only selected-panel fits use `--analysis feature-selection`. Training logs and newly generated outputs are kept in the specified run directory; the public reference results remain unchanged.

For all source/result relationships, Supplementary Figures S1–S11, Tables S1–S16, matched baselines, diagnostics and sensitivity workflows, see the [complete repository guide](../README.md).
