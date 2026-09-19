#!/usr/bin/env python3
"""ACeT research workflows: demonstration, training, analysis, figures and verification."""
from __future__ import annotations
import argparse,os,subprocess,sys,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
S='SensitivityAnalyses'
TASKS={
 'table-s4':('Reproducibility/Table_S4/code/reproduce_table_s4_ridge.py',['--package-root',str(ROOT)],'--output-dir','Fit the two single-descriptor Ridge baselines'),
 'viscosity-screen':(f'{S}/DevelopmentOnlyFeatureSelection/viscosity/code/reproduce_candidate_screen.py',[],'--output-dir','Recalculate development-only candidate-set Pearson rankings'),
 'mouse-preprocessing':(f'{S}/DevelopmentOnlyFeatureSelection/mouse_exposure/code/reproduce_training_only_preprocessing_and_selection.py',[],'--output-dir','Regenerate development-only mouse-exposure model inputs'),
 'comparator-cv':(f'{S}/TunedComparatorStressTest/run_optimized_comparator_stress_test.py',['cv'],'--output-dir','Fit 30 locked conventional-model/fold combinations'),
 'comparator-preflight':(f'{S}/TunedComparatorStressTest/run_optimized_comparator_stress_test.py',['preflight'],'--output-dir','Regenerate fixed-control comparator caches in the study environment'),
 'comparator-search':(f'{S}/TunedComparatorStressTest/run_optimized_comparator_stress_test.py',['search'],'--output-dir','Repeat the development-fold parameter search'),
 'comparator-heldout':(f'{S}/TunedComparatorStressTest/run_optimized_comparator_stress_test.py',['heldout'],'--output-dir','Evaluate locked tuned comparators in the study environment'),
 'hic-results':('GenBench/hic_benchmark/code/source/shared/reproduce_bailly_from_oof.py',['--bench_root',str(ROOT/'GenBench')],'--out_dir','Recreate HIC OOF tables and plots from archived predictions'),
 'clinical-results':('MainPack/clinical_outcome/code/source/figure4_targeted/postprocess.py',[],'--output-dir','Recalculate clinical counts, threshold summaries and portfolio values'),
 'figure2b-data':('FigureSources/main/Figure_2/panel_b/code/rebuild_figure2b_current.py',[],'--output-dir','Regenerate viscosity bootstrap-bin source data'),
 'figure2c-tables':('FigureSources/main/Figure_2/panel_c/code/prepare_figure2c_corrected_mat.py',[],'--output-dir','Rebuild fixed-comparator viscosity table/MAT sources'),
 'figure2de-preflight':('MainPack/viscosity/code/source/figure2de/viscosity_figure2de_current.py',['--mode','preflight','--repo-root',str(ROOT)],'--output-dir','Validate importance, ablation and learning-curve inputs'),
 'figure2de-train':('MainPack/viscosity/code/source/figure2de/viscosity_figure2de_current.py',['--mode','all','--repo-root',str(ROOT)],'--output-dir','Train importance, feature-ablation and learning-curve models'),
 'fixed-holdout-preflight':('GenBench/robustness/fixed_holdout_stability/code/source/run_fixed_holdout_seed_stability.py',['--mode','preflight','--package-root',str(ROOT)],'--output-dir','Validate fixed-holdout seed-stability inputs'),
 'fixed-holdout-train':('GenBench/robustness/fixed_holdout_stability/code/source/run_fixed_holdout_seed_stability.py',['--mode','run','--package-root',str(ROOT)],'--output-dir','Run fixed-holdout training-seed stability'),
 'split-sensitivity-preflight':('GenBench/robustness/split_sensitivity/code/source/run_current_regression_robustness.py',['--mode','preflight','--package-root',str(ROOT)],'--output-dir','Validate alternate-partition robustness inputs'),
 'split-sensitivity-train':('GenBench/robustness/split_sensitivity/code/source/run_current_regression_robustness.py',['--mode','run','--package-root',str(ROOT)],'--output-dir','Regenerate alternate splits and train fixed-head models'),
 'gaussian-copula':('GenBench/diagnostics/gaussian_copula/code/source/gaussian_copula_diagnostics.py',[],'--output-dir','Refit synthesizers and generate distribution diagnostics'),
 'pooling-viscosity':('MainPack/shared/pooling_ablation/viscosity_pooling_ablation.py',['--full'],'--output_dir','Train the four viscosity representation-aggregation variants'),
 'pooling-mouse-exposure':('MainPack/shared/pooling_ablation/clearance_pooling_ablation.py',['--full'],'--output_dir','Train the four mouse-exposure representation-aggregation variants'),
 'export-tables':('tools/export_table_sources.py',[],'--output-dir','Collect machine-readable sources for Table 1 and Tables S1–S16'),
}
MATLAB={
 'supplementary-figures':('FigureSources/supplementary/code','generate_all_supplementary_figures','Render Supplementary Figures S1–S11'),
 'figure4':('FigureSources/main/Figure_4/code','Figure4_current_authoritative','Assemble Figure 4 from archived clinical results'),
 'figure2de-plots':('FigureSources/main/Figure_2/panels_d_e/code','plot_figure2de_current','Render Figure 2d/e from archived importance and ablation sources'),
 'figure3-plots':('FigureSources/main/Figure_3/panels_b_d/code','plot_mouse_exposure_panels','Render Figure 3a–d from archived predictions and SHAP sources'),
}
def safe_output(task,value):
    out=(value or ROOT/'reruns'/task).resolve()
    if out==ROOT or (ROOT in out.parents and ROOT/'reruns' not in out.parents):raise ValueError('Within the repository, generated outputs must be under reruns/.')
    if out.exists() and any(out.iterdir()):raise FileExistsError('Choose a new empty --output-dir; archived files are not overwritten.')
    return out

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='task',required=True)
    sub.add_parser('list',help='List workflows')
    sub.add_parser('environment',help='Show analysis and training dependency availability')
    ver=sub.add_parser('verify',help='Check root checksums and recalculate archived results')
    ver.add_argument('--report',type=Path);ver.add_argument('--checksums-only',action='store_true')
    demo=sub.add_parser('demo',help='Mouse-exposure quick start from archived predictions')
    demo.add_argument('--formats',default='svg',help='Comma-separated svg,png,jpg; raster formats need requirements-demo.txt')
    demo.add_argument('--output-dir',type=Path)
    demo.add_argument('--raster-engine',choices=['auto','cairo','pillow'],default='auto')
    train=sub.add_parser('train',help='Run endpoint-specific neural training; see train --help')
    train.add_argument('endpoint',choices=['viscosity','mouse_exposure','clinical_outcome','hic','all'])
    train.add_argument('--analysis',choices=['main','feature-selection','matched-comparison','shap'],default='main');train.add_argument('--hic-features',choices=['assays_only','patch_only','all'],default='assays_only')
    train.add_argument('--head',choices=['mlp','rbf','spline','kan','interaction','all']);train.add_argument('--check-inputs',action='store_true');train.add_argument('--output-dir',type=Path);train.add_argument('--epochs',type=int);train.add_argument('--seed',type=int,default=0)
    for name,rec in TASKS.items():
        q=sub.add_parser(name,help=rec[3]);q.add_argument('--output-dir',type=Path)
        if name.startswith('comparator-'):
            q.add_argument('--cache-dir',type=Path);q.add_argument('--lock-dir',type=Path)
    q=sub.add_parser('figure3b-data',help='Extract the paired augmentation-comparison plotting arrays');q.add_argument('--output-dir',type=Path)
    for name,rec in MATLAB.items():q=sub.add_parser(name,help=rec[2]);q.add_argument('--output-dir',type=Path)
    a=parser.parse_args();env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg')
    if a.task=='list':
        print('demo: archived-prediction quick start\nverify: integrity and numerical recalculation\ntrain: endpoint-specific neural fitting\nfigure3b-data: extract paired Figure 3b arrays')
        for name,rec in TASKS.items():print(name+': '+rec[3])
        for name,rec in MATLAB.items():print(name+': '+rec[2]+' (MATLAB)')
        return
    if a.task=='environment':
        import importlib.metadata
        print('Python:',sys.version.split()[0])
        for name in ['numpy','pandas','scipy','scikit-learn','matplotlib','CairoSVG','Pillow','tensorflow','keras','tfkan','sdv','ImbalancedLearningRegression','imbalanced-learn','shap']:
            try:value=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:value='not installed'
            print(f'{name}: {value}')
        print('MATLAB:',shutil.which('matlab') or 'not on PATH');return
    if a.task=='train':
        cmd=[sys.executable,str(ROOT/'tools/run_training.py'),a.endpoint,'--analysis',a.analysis,'--hic-features',a.hic_features,'--seed',str(a.seed)]
        for field in ['head','output_dir','epochs']:
            value=getattr(a,field,None)
            if value is not None:cmd+=['--'+field.replace('_','-'),str(value)]
        if a.check_inputs:cmd+=['--check-inputs']
    elif a.task=='verify':
        cmd=[sys.executable,str(ROOT/'verify_release.py')]
        if a.report:cmd+=['--report',str(a.report.resolve())]
        if a.checksums_only:cmd+=['--checksums-only']
    elif a.task=='demo':
        cmd=[sys.executable,str(ROOT/'Demo/clearance_quickstart/run_demo.py'),'--package-root',str(ROOT),'--formats',a.formats,'--raster-engine',a.raster_engine,'--output-dir',str(safe_output(a.task,a.output_dir))]
    elif a.task=='figure3b-data':
        out=safe_output(a.task,a.output_dir)
        cmd=[sys.executable,str(ROOT/'MainPack/mouse_exposure/code/source/extract_figure3b_locked_arrays.py'),'--output',str(out/'figure3b_plotting_arrays.csv')]
    elif a.task in MATLAB:
        folder,fn,_=MATLAB[a.task];out=safe_output(a.task,a.output_dir)
        exe=shutil.which('matlab')
        if not exe:raise SystemExit('This plotting workflow requires MATLAB. The existing exports are in FigureSources/.')
        quote=lambda p:str(p).replace("'","''")
        expr=f"addpath('{quote(ROOT/folder)}'); {fn}('{quote(ROOT)}','{quote(out)}');"
        cmd=[exe,'-batch',expr]
    else:
        script,args,flag,_=TASKS[a.task];out=safe_output(a.task,a.output_dir)
        cmd=[sys.executable,str(ROOT/script),*args,flag,str(out)]
        for field in ['cache_dir','lock_dir']:
            if getattr(a,field,None):cmd+=['--'+field.replace('_','-'),str(getattr(a,field).resolve())]
    raise SystemExit(subprocess.call(cmd,cwd=ROOT,env=env))
if __name__=='__main__':
    try:main()
    except (ValueError,FileExistsError,FileNotFoundError) as exc:raise SystemExit(str(exc))
