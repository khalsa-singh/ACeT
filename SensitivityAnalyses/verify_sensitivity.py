"""Verify hashes, numeric input schemas, archived predictions and search selection.

This checks archived evidence; it does not retrain the neural networks or repeat
all hyperparameter fits. For those operations see README.md and ENVIRONMENT.md.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def checksums():
    import importlib.util, sys
    public_root = ROOT.parent
    spec = importlib.util.spec_from_file_location("acet_integrity", public_root / "verify_release.py")
    module = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True
    spec.loader.exec_module(module)
    return module.integrity()


def numeric_checks():
    import numpy as np
    import pandas as pd
    from scipy.io import loadmat
    from scipy.stats import spearmanr, pearsonr
    def metrics(y,p):
        y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float)
        if len(y)!=len(p) or not np.isfinite(y).all() or not np.isfinite(p).all():raise ValueError('Bad prediction array')
        err=p-y;rmse=float(np.sqrt(np.mean(err**2)));mae=float(np.mean(np.abs(err)))
        return {'r2':float(1-np.sum(err**2)/np.sum((y-y.mean())**2)), 'rmse':rmse,'mae':mae,
                'nrmse':rmse/float(y.mean()),'nmae':mae/float(y.mean()),'spearman_rho':float(spearmanr(y,p).statistic)}
    def close(a,b,tol=1e-9):
        if not math.isclose(float(a),float(b),abs_tol=tol,rel_tol=0):raise ValueError(f'Numeric mismatch: {a} vs {b}')
    t=ROOT/'TunedComparatorStressTest'
    lock_path=t/'settings/LOCKED_HYPERPARAMETERS.json'
    if digest(lock_path)!=(t/'settings/LOCKED_HYPERPARAMETERS.sha256').read_text().split()[0]:raise ValueError('Lock hash mismatch')
    lock=json.loads(lock_path.read_text())
    if lock.get('heldout_read_during_search') is not False:raise ValueError('Unexpected search-scope declaration')
    cm=pd.read_csv(t/'search/FOLD_CACHE_MANIFEST.csv')
    if len(cm)!=30:raise ValueError('Expected 30 cached folds')
    for r in cm.to_dict('records'):
        p=t/r['cache_file']
        if digest(p)!=r['sha256']:raise ValueError('Cached fold changed')
        with np.load(p,allow_pickle=False) as z:
            for k in ['X_train','y_train','X_validation','y_validation','validation_indices']:
                if not np.isfinite(z[k]).all():raise ValueError('Nonfinite cache')
    predictions=pd.read_csv(t/'results/OPTIMIZED_COMPARATOR_HELDOUT_PREDICTIONS.csv')
    stored=pd.read_csv(t/'results/OPTIMIZED_COMPARATOR_HELDOUT_METRICS.csv')
    derived=[]
    for (e,m),group in predictions.groupby(['endpoint','model']):
        mm=metrics(group.observed,group.predicted)
        n=23 if e=='viscosity' else 11
        if len(group)!=n or group.antibody_id.duplicated().any():raise ValueError('Invalid held-out membership')
        row=stored[(stored.endpoint==e)&(stored.model==m)].iloc[0]
        for k in ['r2','rmse','mae','spearman_rho']+(['nrmse','nmae'] if e=='mouse_exposure' else []):close(mm[k],row[k])
        derived.append({'endpoint':e,'model':m,**mm})
    if len(derived)!=6:raise ValueError('Expected six comparator results')
    trials=pd.read_csv(t/'search/ALL_HYPERPARAMETER_TRIALS.csv')
    selected=[]
    for e,models in lock['locked'].items():
        for m,info in models.items():
            df=trials[(trials.endpoint==e)&(trials.model==m)&trials.valid.astype(str).str.lower().eq('true')].copy()
            if len(df)!=info['valid_candidates']:raise ValueError('Trial count mismatch')
            # Same recorded lexicographic selection criteria, including tie-breaks.
            ranked=sorted(df.to_dict('records'),key=lambda r:(-float(r['mean_r2']),float(r['mean_rmse']),tuple(json.loads(r['complexity_json']))))
            best=ranked[0]
            if int(best['trial_id'])!=info['selection']['trial_id'] or json.loads(best['parameters_json'])!=info['parameters']:raise ValueError('Locked setting is not recorded search winner')
            close(best['mean_r2'],info['selection']['mean_cv_r2'])
            selected.append({'endpoint':e,'model':m,'trial_id':int(best['trial_id'])})
    # Validate schema compatibility for the two neural-network entry points.
    dev=ROOT/'DevelopmentOnlyFeatureSelection'
    for ep,a,b,n in [('viscosity','antibodies_train.csv','antibodies_test.csv',52),('mouse_exposure','train42_model_ready_selected4.csv','test11_raw_untouched_selected4.csv',42)]:
        tr=pd.read_csv(dev/ep/'data'/a);te=pd.read_csv(dev/ep/'data'/b)
        if tr.shape!=(n,5) or te.shape!=((23 if ep=='viscosity' else 11),5) or list(tr)!=list(te):raise ValueError('Neural input shape mismatch')
        if not np.isfinite(tr.to_numpy(float)).all() or not np.isfinite(te.to_numpy(float)).all():raise ValueError('Nonfinite input')
    # Mouse-exposure sensitivity MAT holds the full-precision prediction record.
    m=dev/'mouse_exposure'
    mat=loadmat(m/'results/clearance_training_only_selected_mlp_panel3A_data.mat')
    mm=metrics(mat['trues'].ravel(),mat['preds'].ravel())
    summary=pd.read_csv(m/'results/clearance_training_only_selected_mlp_result_summary.csv').iloc[0]
    for k,field in [('r2','Heldout_R2'),('rmse','Heldout_RMSE'),('mae','Heldout_MAE'),('nrmse','Heldout_nRMSE'),('nmae','Heldout_nMAE'),('spearman_rho','Heldout_Spearman')]:close(mm[k],summary[field],1e-8)
    # Recompute the viscosity candidate set screen.
    import importlib.util
    v=dev/'viscosity'
    source=v/'code/reproduce_candidate_screen.py'
    spec=importlib.util.spec_from_file_location('viscosity_screen',source)
    screen_module=importlib.util.module_from_spec(spec)
    # Avoid leaving compiled files inside a checked scientific archive.
    import sys
    previous_flag=sys.dont_write_bytecode
    sys.dont_write_bytecode=True
    try:spec.loader.exec_module(screen_module)
    finally:sys.dont_write_bytecode=previous_flag
    screen_summary=screen_module.verify_archived_screen(v)
    return {'archived_comparator_metrics_recomputed':derived,'locked_search_winners':selected,'mouse_sensitivity_metrics':mm,'viscosity_candidate_screen':screen_summary,'full_neural_retraining_performed':False}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checksums-only',action='store_true');p.add_argument('--report',type=Path);a=p.parse_args()
    count=checksums();result={'manifest_files_verified':count}
    if not a.checksums_only:result.update(numeric_checks())
    if a.report:
        dest=a.report.resolve()
        if dest.exists():raise FileExistsError('Report already exists; choose a new filename')
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(result,indent=2)+'\n')
    print(f'{count} archive files verified.')
    if not a.checksums_only:print('Six comparator metric rows recalculated; six locked search winners checked; sensitivity predictions and the viscosity candidate set screen verified.')
    print('Verification completed. This command does not retrain ACeT.')
if __name__=='__main__':main()
