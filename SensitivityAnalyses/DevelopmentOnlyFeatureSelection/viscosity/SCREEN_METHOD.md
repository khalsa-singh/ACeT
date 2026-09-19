# Viscosity development-only univariate association

## Candidate set and source

The candidate set is defined in `reference/candidate_definition.csv`, which maps
each candidate assay or descriptor to its source-workbook column. Sample pH is
not an assay candidate. The reference ranking is supplied in
`reference/MLHealth_Viscosity_Development_Only_Univariate_Association.csv`.

Measurements derive from Mock et al. (2023),
`Supporting Data Tables_Mock_etal_revised.xlsx`, sheet `Table S1 Data`.
The source export contains the candidate measurements, high-concentration
viscosity target, and identifiers for the same 52 development antibodies.
The row map retains their source locations and manuscript IDs. Neither a test
table nor held-out observations are used by the screening script.

## Calculation

Each member of the candidate set is ranked by absolute Pearson correlation with
development viscosity. Univariate R2 is Pearson r squared. All candidate
measurements are complete for all 52 development antibodies; no imputation,
target transformation or pairwise deletion is used. Exact correlation ties
retain candidate-definition order. Nominal p-values are unadjusted descriptive
association summaries and are not selection criteria.

The first four in this candidate set are SE-UHPLC main-peak FWHM, DLS kD,
SE-UHPLC main-peak plates, and AC-SINS spectral shift. These match the panel used
by the archived viscosity model. Candidate ranking does not retrain that model.
The model's legacy `AC-SINS λmax (nm)` input header identifies the source's
`AC-SINS Δλmax` values; this header alias does not change any measurement.

## Files and reproduction

- `00_source/development52_candidates.csv`: source measurements in development order.
- `00_source/development_row_mapping.csv`: antibody and source-row mapping.
- `reference/candidate_definition.csv`: candidate definitions and source-column mapping.
- `reference/MLHealth_Viscosity_Development_Only_Univariate_Association.csv`: unchanged reference ranking.
- `data/development_candidate_assays.csv`: numeric candidate matrix with identifiers and target, for screening only.
- `results/viscosity_development_only_candidate_screen.csv`: ranked candidate set.
- `results/viscosity_development_only_ranking.csv`: the same ranking in a separate output file.

From the addendum root:

```text
python DevelopmentOnlyFeatureSelection/viscosity/code/reproduce_candidate_screen.py --verify
python DevelopmentOnlyFeatureSelection/viscosity/code/reproduce_candidate_screen.py --output-dir reruns/viscosity_screen
```

The verification command recalculates correlations from the candidate data and
compares the ranks, coefficients, p-values, panel membership, and Table S15a
source with their archived records. It does not read a held-out file or fit ACeT.
