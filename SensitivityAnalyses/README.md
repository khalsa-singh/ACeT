# Sensitivity analyses and comparator benchmarking

These analyses evaluate the effects of assay-selection scope and conventional-model parameter tuning. They support Supplementary Tables S15 and S16 and complement the main endpoint workflows.

## Development-only feature selection

`DevelopmentOnlyFeatureSelection/viscosity/` contains the defined candidate-set measurements for the 52 development antibodies, candidate definitions, row mapping, absolute-Pearson ranking producer and archived ranking. Within this candidate set, the first four are SEC FWHM, kD, SEC plates and AC-SINS spectral shift. The model input order is retained separately in the numeric training files.

`DevelopmentOnlyFeatureSelection/mouse_exposure/` contains the original candidate inputs, antibody mapping, two-stage development-set imputation/selection producer, selected four-feature train/test tables, MLP analysis settings and prediction records. The 42/11 membership matches the analysis described in the article; the raw held-out table is not imputed in the preprocessing producer.

From the repository root:

```bash
python acet.py viscosity-screen --output-dir reruns/viscosity_screen
python acet.py mouse-preprocessing --output-dir reruns/mouse_preprocessing
python acet.py train all --analysis feature-selection --check-inputs
python acet.py train viscosity --analysis feature-selection --output-dir reruns/selected_panel_viscosity
python acet.py train mouse_exposure --analysis feature-selection --output-dir reruns/selected_panel_mouse
```

The first two commands perform actual ranking/preprocessing. Training requires the separate neural environment and generates new fitted results. Model settings differ where documented; the mouse-exposure sensitivity uses 32-dimensional embeddings, a 48-unit feed-forward layer and 63 synthetic observations.

## Conventional-model tuning

`TunedComparatorStressTest/` contains endpoint data, fixed-control records, all 3,380 trial rows, fold caches, the selected hyperparameters and CV/held-out prediction records. Ridge, RBF-SVR and random forest were tuned independently per endpoint. The preserved settings are selected using development-fold scores.

```bash
python acet.py comparator-cv --output-dir reruns/comparator_cv
python acet.py comparator-search --output-dir reruns/comparator_search
python acet.py comparator-heldout --output-dir reruns/comparator_heldout
```

`comparator-cv` refits the 30 model/fold combinations from the stored caches. A new search or held-out generation uses the study-version dependencies; see [ENVIRONMENT.md](ENVIRONMENT.md) and the [root guide](../README.md#feature-selection-and-comparator-tuning). The fixed-control preflight has its own evaluation. The tuning search uses development data, and tuned-model evaluation follows parameter locking.

## Tables and verification

`TableSources/` contains the machine-readable S15/S16 rows. All archived data and model outcomes keep their own analysis labels. Run `python acet.py verify` from the root for integrity and independent metric/ranking checks, or `python SensitivityAnalyses/verify_sensitivity.py` for the sensitivity-specific check.

The methods are in [METHODS.md](METHODS.md); the complete public entry point is [README.md](../README.md).
