# Software environments

`reported_requirements.txt` records the study's dependency versions. The same training package versions appear in root `requirements-training.txt`. Use Python 3.11 for this TensorFlow/Keras 2.15 workflow.

Root `requirements.txt` is a separate lightweight environment for archived-result recalculation, preprocessing, figure-data generation and cached conventional-model fitting. Root `requirements-demo.txt` installs only the raster-demo dependency. The SVG-only demonstration needs no external Python packages.

Create separate environments; do not install the old training pins over an existing modern TensorFlow stack. `python acet.py environment` lists the packages present in the active environment. The original estimators used scikit-learn 1.4.1.post1; cached-model fits under other releases may differ numerically, especially random forests. New fits remain separate from archived results.

MATLAB R2022b was used for the publication plotting scripts. MATLAB is not needed to browse data, inspect existing exports, run the quick demonstration or recompute stored metrics.
