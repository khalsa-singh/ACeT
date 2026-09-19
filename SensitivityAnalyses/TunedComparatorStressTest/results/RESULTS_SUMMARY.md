# Development-CV-tuned comparator stress-test summary

This finite stress test independently tuned Ridge regression, RBF-SVR and random forest within each endpoint's development folds. Held-out outcomes were not used in the hyperparameter search. Tuned-model evaluation followed parameter locking; the preceding fixed-control check also evaluated the holdout.

## Held-out interpretation

- **Viscosity:** ACeT retained the strongest held-out performance (R² = 0.7409). The highest held-out result among the independently development-CV-tuned conventional model classes was Ridge (R² = 0.5288).
- **Mouse exposure:** ACeT retained the strongest held-out performance (R² = 0.7911). The highest held-out result among the independently development-CV-tuned conventional model classes was RBF-SVR (R² = 0.7311).

The phrase “highest held-out result among the independently tuned model classes” is used because each conventional model class was tuned separately by development CV; the held-out set was not used to choose hyperparameters.

These are finite development-CV-tuned stress tests, not claims of universally optimal conventional models.
