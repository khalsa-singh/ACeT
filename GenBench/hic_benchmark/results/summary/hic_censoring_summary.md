# HIC assay ceiling and triage

The DataS5 panel contains 152 antibodies; 39 (25.7%) are recorded at the 50-minute assay ceiling. Those observations are interpreted as right-censored rather than exact retention times. Continuous metrics use the recorded values.

The good/poor triage comparison uses a 30-minute threshold. Every ceiling-censored observation remains above this threshold, so its class does not depend on the unknown exact retention time beyond 50 minutes.

Source: `Data/curated/DataS5_bailly2020_hicrt_no_leakage_all_features.csv`; source study: Bailly et al. (2020), cited in `Data/SOURCES.md`.
