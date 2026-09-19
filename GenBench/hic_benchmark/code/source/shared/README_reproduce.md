# HIC execution

Use the repository-root launchers in `README.md`: `python acet.py train hic` for actual OOF training, and `python acet.py hic-results` for archived-prediction postprocessing. The full source runner is `panel_A_D_parity_featureimp_EDITED_v2.py`, with `analysis_mode=oof_cv`, `id_col=mab_id` and `target_col=hic_rt_min`.

`reproduce_bailly_from_oof.py` reads the included benchmark OOF tables and writes new results to the requested output directory. It preserves the distinction between bagged OOF metrics, replicate variability, the fold-matched patch-linear reference and the published fixed equation.
