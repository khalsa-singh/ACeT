# Reproducing the analysis

The repository provides three complementary paths: recalculating archived numerical results, refitting a recorded model configuration, and running a new experiment. The [root guide](../README.md) lists the commands and maps every main figure, supplementary figure and supplementary table to its source.

## Contents

- `environment/`: the recorded software stack.
- `Table_S4/`: single-descriptor Ridge inputs/producer and reference outputs.
- `table_sources/`: the matched main-regression comparison summaries.
- The root `MANIFEST.csv` and `CHECKSUMS.sha256`: one coordinated integrity index for all included files.

Run `python acet.py verify` for numerical recalculation, or `python acet.py verify --checksums-only` for integrity without scientific dependencies. Run `python acet.py export-tables` to collect the sources for Table 1 and Tables S1–S16 into a new output directory.

## Matching the evaluated configurations

The main regression workflows use native-row upweighting and Gaussian-Copula augmentation before their internal CV folds. The predictor QuantileTransformer is fitted to that development matrix; imbalance handling and target transformation are applied within fold training. Consequently the internal CV scores are selection/ensemble scores rather than fully nested independent estimates. The outer held-out cohort is kept separate from model fitting. The development-only candidate-screen analysis is separately documented under `SensitivityAnalyses/`.

The inherited regression ensemble averages fold predictions in transformed-target space and inverse-transforms that average with the final fold's target transformer. This convention is preserved in the main and sensitivity implementations so that the archived numerical references remain associated with the same computation. The figure/table guide distinguishes the primary exposure result, its matched-comparator result and the separate sensitivity runs.

HIC preprocessing is fold-local in the repeated OOF workflow. Clinical output labels follow the stored outcome encoding and the fixed two-class softmax argmax rule. The clinical application is retrospective developability-linked classification, not a comprehensive approval-probability estimator.

## Inputs and new outputs

Canonical datasets are in `Data/curated`; numerical model inputs and exact column order are documented in the root guide. The viscosity candidate-set definitions and mouse-exposure preprocessing sources remain available for reproducing selection. Source-study attribution is in `Data/SOURCES.md`.

New work is written to `reruns/` or another explicitly chosen empty output directory. The source files and archived predictions are not overwritten by the public launchers. Checkpoints are created only where an endpoint runner supports saving them; pretrained checkpoints are not included in this archive.

## Verification and fitting environments

The archived-results verifier and cached conventional-model CV fits can be run with the lightweight requirements. Full ACeT training uses the separate Python 3.11/TensorFlow 2.15 study stack. Device kernels, software versions and random-state behavior can affect a newly trained model. Compare the correct analysis configuration and preserve the new environment record with each run.

The root checksum updater is for deliberate releases after reviewing changes, not for bypassing an unexpected integrity failure. The independent hyperparameter-lock checksum remains an analytical record and is not rewritten by the root updater.
