#!/usr/bin/env python3
"""Collect the exact machine-readable inputs and producers for Table 1 and Tables S1–S16.

This is a table-source collection operation, not neural training or a replacement
for the typeset Supplement. Each copied file is hashed and linked to its source.
"""
from __future__ import annotations
import argparse,csv,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCES={
'Table_1': [('MainPack/clinical_outcome/results/targeted_repro/FIGURE4_ECONOMICS_RECALC.csv','Economic calculation'),('MainPack/clinical_outcome/code/source/figure4_targeted/postprocess.py','Assumptions and postprocessing producer')],
'Table_S01':[('Data/DATA_DICTIONARY.csv','Assay and target dictionary')],
'Table_S02':[('GenBench/hic_benchmark/results/summary/hic_existing_benchmark_summary.csv','Bagged OOF summaries'),('GenBench/hic_benchmark/results/benchmark/metrics/*/*_oof_metrics.json','Full HIC metric records')],
'Table_S03':[('GenBench/hic_benchmark/results/benchmark/metrics/*/*_repeat_metrics.csv','Per-repeat HIC metrics')],
'Table_S04':[('Reproducibility/Table_S4/results/TABLE_S4_RIDGE_RESULTS.csv','Single-descriptor Ridge metrics'),('Reproducibility/Table_S4/results/TABLE_S4_RIDGE_PREDICTIONS.csv','Single-descriptor predictions'),('MainPack/viscosity/results/primary/viscosity_predictions.csv','Main ACeT predictions')],
'Table_S05':[('Reproducibility/table_sources/*_matched_baseline_metrics.csv','Matched comparison summaries'),('FigureSources/main/Figure_2/panel_c/data/figure2c_corrected_baselines.mat','Viscosity fold and held-out metrics'),('FigureSources/main/Figure_3/panel_c/data/figure3c_current_baselines.mat','Mouse-exposure fold and held-out metrics')],
'Table_S06':[('MainPack/shared/minimal_baseline_patch/*_panel_C_baselines_minimal_patch.py','Fixed estimator configuration in source')],
'Table_S07':[('GenBench/diagnostics/prediction_head_comparison/prediction_head_variability_metrics.csv','Five-head comparison summaries')],
'Table_S08':[('MainPack/viscosity/code/source/shared/panel_A_D_parity_featureimp.py','Viscosity preprocessing and training settings'),('MainPack/mouse_exposure/code/source/shared/panel_A_B_parity.py','Mouse-exposure preprocessing and training settings'),('MainPack/clinical_outcome/code/source/figure4_targeted/panels.py','Clinical preprocessing and training settings'),('GenBench/hic_benchmark/code/source/shared/panel_A_D_parity_featureimp_EDITED_v2.py','HIC OOF settings'),('SensitivityAnalyses/DevelopmentOnlyFeatureSelection/mouse_exposure/analysis_settings.json','Additional selected-panel configuration')],
'Table_S09':[('GenBench/diagnostics/pooling_sensitivity/results/*_full_metrics.csv','Aggregation-layer metrics'),('GenBench/diagnostics/pooling_sensitivity/results/*_full_predictions.csv','Aggregation-layer prediction records')],
'Table_S10':[('GenBench/diagnostics/gaussian_copula/results/synthetic_diagnostics_summary.csv','Copula diagnostics summary'),('GenBench/diagnostics/gaussian_copula/results/*_statistics.csv','Feature-wise and neighbor statistics')],
'Table_S11':[('GenBench/diagnostics/clinical_uncertainty/clinical_uncertainty_table.csv','Clinical uncertainty table')],
'Table_S12':[('GenBench/hic_benchmark/results/summary/hic_existing_benchmark_summary.csv','HIC comparisons and reference equation'),('GenBench/hic_benchmark/results/summary/hic_censoring_summary.md','Ceiling-censoring context')],
'Table_S13':[('GenBench/robustness/fixed_holdout_stability/results/locked/TABLE_S13_COMBINED_REGRESSION_ROWS.csv','Fixed/alternate regression summary'),('GenBench/robustness/fixed_holdout_stability/results/locked/fixed_holdout_stability_summary.csv','Fixed-holdout summary'),('GenBench/robustness/split_sensitivity/results/locked/*_alternate_partition_metrics.csv','Partition-specific metrics'),('GenBench/robustness/clinical_external/clinical_external_seed0_10_summary_aggregate.csv','External clinical aggregate'),('GenBench/hic_benchmark/results/benchmark/metrics/assays_only/*_repeat_metrics.csv','HIC assays-only repeated metrics')],
'Table_S14':[('MainPack/clinical_outcome/results/finalization/TABLE_S14_CLINICAL_RULE_COMPARISON.csv','Clinical assay-rule comparison')],
'Table_S15':[('SensitivityAnalyses/TableSources/Table_S15*.csv','Candidate rankings and selected-panel performance'),('SensitivityAnalyses/DevelopmentOnlyFeatureSelection/viscosity/reference/candidate_definition.csv','Candidate definitions'),('SensitivityAnalyses/DevelopmentOnlyFeatureSelection/mouse_exposure/analysis_settings.json','Selected-panel analysis settings')],
'Table_S16':[('SensitivityAnalyses/TableSources/Table_S16*.csv','Tuned comparator table sources'),('SensitivityAnalyses/TunedComparatorStressTest/settings/LOCKED_HYPERPARAMETERS.json','Selected hyperparameters'),('SensitivityAnalyses/TunedComparatorStressTest/results/OPTIMIZED_COMPARATOR_CV_METRICS.csv','Fold and aggregate CV metrics'),('SensitivityAnalyses/TunedComparatorStressTest/results/OPTIMIZED_COMPARATOR_HELDOUT_METRICS.csv','Held-out metric records')],
}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();out=a.output_dir.resolve()
 if out==ROOT or (ROOT in out.parents and ROOT/'reruns' not in out.parents):p.error('Use reruns/ or an output directory outside the repository.')
 if out.exists() and any(out.iterdir()):raise FileExistsError('Choose a new empty output directory.')
 entries=[]
 for table,patterns in SOURCES.items():
  for pattern,role in patterns:
   files=sorted(ROOT.glob(pattern))
   if not files:raise FileNotFoundError(f'{table}: required source not found: {pattern}')
   for source in files:
    dest=out/table/source.name
    if dest.exists() and dest.read_bytes()!=source.read_bytes():raise ValueError(f'Conflicting source name {dest.name}')
    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
    entries.append({'table':table,'role':role,'source':source.relative_to(ROOT).as_posix(),'exported_file':dest.relative_to(out).as_posix(),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
 with (out/'TABLE_SOURCE_INDEX.csv').open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(entries[0]));w.writeheader();w.writerows(entries)
 (out/'README.md').write_text('# Table-source collection\n\nThese files are copied, unchanged, from the public ACeT repository. The source index records the role and hash of each file. S6/S8 include the exact configuration-bearing source code rather than a newly inferred setting table. Run the root analysis commands for recalculation. This folder does not reproduce the typeset table layout.\n')
 print(f'Collected {len(entries)} source records for Table 1 and Tables S1–S16 in {out}')
if __name__=='__main__':main()
