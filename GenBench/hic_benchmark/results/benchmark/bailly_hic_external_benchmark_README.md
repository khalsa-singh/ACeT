# HIC reference outputs

`oof_predictions/` contains the repeated OOF predictions for assays-only, patch-only and assays-plus-patch inputs. `metrics/` contains bagged and repeat-level summaries. The reference rows and censoring context are collected under `../summary/`.

Recalculate the tables and plots with `python acet.py hic-results` from the repository root. Run the full OOF model with `python acet.py train hic`; see the root guide for each feature configuration.
