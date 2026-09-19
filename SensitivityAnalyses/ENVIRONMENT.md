# Environments for the sensitivity analyses

Use root `requirements-training.txt` with Python 3.11 for the ACeT selected-panel fits and original comparator preprocessing/search. The recorded estimator version is scikit-learn 1.4.1.post1.

Candidate ranking, iterative-imputation reconstruction, stored-result verification and cached conventional-model CV fitting can be run in the lightweight analysis environment. The public commands preserve the input tables and write new results separately. Cached random-forest refits under a different scikit-learn version can produce small differences; compare the output's version record with the archived lock.

The viscosity candidate-screen producer requires NumPy, pandas and SciPy. The mouse-preprocessing producer also requires scikit-learn. The cached comparator command requires NumPy, pandas, SciPy and scikit-learn but does not require TensorFlow. The full ACeT runs require TensorFlow, tfkan, SDV and ImbalancedLearningRegression, in addition to their imported plotting dependencies.

See the root README for installation and endpoint-specific commands. No pretrained neural checkpoints are distributed with this package.
