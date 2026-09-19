# Curated datasets (DataS1–DataS5)

These are the five analysis-ready datasets for **Monoclonal Antibody Developability from Early Assay Panels: Machine Learning for Formulation and Pharmacokinetic Risk**, by Sabitoj Singh Virk and Akashdeep Singh Virk.

| Dataset | Antibodies | Target / purpose | Source study |
|---|---:|---|---|
| [DataS1](DataS1_viscosity_seed0_full.csv) | 75 | High-concentration viscosity; 52 development / 23 held-out | Mock et al. (2023) |
| [DataS2](DataS2_clearance_seed0_full.csv) | 53 | Mouse IV exposure (AUCt); 42 development / 11 held-out | Liu et al. (2023) |
| [DataS3](DataS3_clinical_internal_112_seed0_full_wname.csv) | 112 | Internal clinical-outcome cohort; 89 development / 23 held-out | Jain et al. (2017, 2023) |
| [DataS4](DataS4_clinical_external_14_full_wname.csv) | 14 | Resolved temporally external clinical outcomes | Jain et al. (2017, 2023), outcome annotations described in the article |
| [DataS5](DataS5_bailly2020_hicrt_no_leakage_all_features.csv) | 152 | HIC retention time; repeated out-of-fold evaluation | Bailly et al. (2020) |

## Read the data

The CSVs are directly downloadable and do not require neural-network software.
Use the `mAb_id` / `mab_id` columns as identifiers, not numeric predictors. `split_seed0`
records the frozen partition where applicable. Exact training order is retained
in [fixed split files](../fixed_splits).

The [data dictionary](../DATA_DICTIONARY.csv) is transcribed from Supplementary
Table S1. Header aliases are documented below; original file bytes are preserved.

- DataS2 `AUCt` stores the exposure AUC0–672h divided by 10^4, not clearance.
  Divide a stored value by 100 to express it in the manuscript's 10^6 ng·h/mL units.
- DataS1 legacy `AC-SINS λmax (nm)` denotes the manuscript's AC-SINS Δλmax shift.
- DataS4 has an original trailing space in the `BVP ` header; it corresponds to BVP.
- HIC values recorded at 50 min include non-eluting antibodies; interpretation is
  right-censored, while the archived regressions used the recorded values.

These are curated/derived inputs from the cited publications, not newly collected
experimental data or a redistribution of every original publisher workbook.
Additional candidate-screen inputs and the distinct mouse-exposure sensitivity
inputs are under [R2](../../SensitivityAnalyses/README.md); they do not replace DataS1 or DataS2.

For code, archived predictions, figure/table sources and reproduction commands,
see the [repository README](../../README.md). Source attribution and reuse scope
are documented in [RIGHTS_AND_ATTRIBUTION.md](../../RIGHTS_AND_ATTRIBUTION.md).
