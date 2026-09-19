# Dataset sources and attribution

ACeT uses previously published antibody measurements and author-curated analysis tables. The dataset identifiers below match the article and Supplementary Information.

| Data | Original study | Contents here |
|---|---|---|
| DataS1; viscosity candidate-screen inputs | Mock M. et al. *Development of in silico models to predict viscosity and mouse clearance using a comprehensive analytical data set collected on 83 scaffold-consistent monoclonal antibodies.* mAbs 15, 2256745 (2023). DOI: 10.1080/19420862.2023.2256745 | The 75-row model table, 52/23 fixed splits, and the explicitly defined development candidate set |
| DataS2; mouse-exposure candidate inputs | Liu S. et al. *Utility of physiologically based pharmacokinetic modeling to predict inter-antibody variability in monoclonal antibody pharmacokinetics in mice.* mAbs 15, 2263926 (2023). DOI: 10.1080/19420862.2023.2263926; related assay measurements in Mock et al. | The curated 53-row main model table, fixed 42/11 splits and separate source/processed inputs for the development-only analysis |
| DataS3 and DataS4 | Jain T. et al. *Biophysical properties of the clinical-stage antibody landscape.* PNAS 114, 944–949 (2017); Jain T., Boland T. and Vásquez M. *Identifying developability risks for clinical progression of antibodies using high-throughput in vitro and in silico approaches.* mAbs 15, 2200540 (2023). DOI for the latter: 10.1080/19420862.2023.2200540 | Internal outcome-locked and temporally external cohorts, with the status conventions described in the article |
| DataS5 | Bailly M. et al. *Predicting antibody developability profiles through early stage discovery screening.* mAbs 12, 1743053 (2020). DOI: 10.1080/19420862.2020.1743053 | Assay and patch descriptors, HIC retention times and identifiers for 152 antibodies |

The canonical files are in `curated/`. Their column definitions and units are in `DATA_DICTIONARY.csv`. Numeric model inputs are also supplied next to the endpoint workflows to preserve the required feature order. Identifiers and split labels are metadata, not predictors.

The viscosity development screen uses the candidate definitions in `SensitivityAnalyses/DevelopmentOnlyFeatureSelection/viscosity/reference/candidate_definition.csv`. Mouse-exposure primary and development-only imputation tables are distinct analysis inputs and should not be substituted for one another.

Original measurements remain attributable to their source studies. Reuse is subject to the applicable source terms; this repository does not grant additional rights over third-party material. See `../RIGHTS_AND_ATTRIBUTION.md`.
