#!/usr/bin/env python3
"""Verify the integrated release and recalculate archived results, without neural training.

Run: python verify_release.py --report reruns/release_verification.json
Requires NumPy, pandas, SciPy and scikit-learn for numerical checks.
Use --checksums-only for a Python-standard-library integrity check.
CSV prediction checks allow their recorded rounding/float precision (up to
1e-6 in R-squared and 5e-5 in stored error units); checks against full-precision
MAT/JSON records use the tighter tolerance shown in each report row.
"""
from __future__ import annotations
import argparse,csv,hashlib,importlib.util,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True

def file_hash(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def integrity():
    with (ROOT/'MANIFEST.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    names=set()
    for r in rows:
        p=(ROOT/r['relative_path']).resolve()
        if ROOT not in p.parents or r['relative_path'] in names:raise ValueError('Invalid or duplicate manifest path')
        names.add(r['relative_path'])
        if not p.is_file() or file_hash(p)!=r['sha256']:raise ValueError('Missing or changed file: '+r['relative_path'])
    parsed={n:h for h,n in (line.split('  ',1) for line in (ROOT/'CHECKSUMS.sha256').read_text().splitlines() if line)}
    if parsed!={r['relative_path']:r['sha256'] for r in rows}:raise ValueError('Manifest/checksum disagreement')
    return len(rows)

def numerical_checks():
    import numpy as np
    import pandas as pd
    from scipy.io import loadmat
    from scipy.stats import spearmanr,pearsonr,ks_2samp,wasserstein_distance,ttest_rel
    from sklearn.metrics import confusion_matrix,balanced_accuracy_score,matthews_corrcoef,accuracy_score
    checks=[];summaries=[]
    def check(label,actual,reference,tol=1e-8):
        delta=float(abs(float(actual)-float(reference)))
        checks.append({'check':label,'actual':float(actual),'reference':float(reference),'tolerance':tol,'absolute_difference':delta,'matches':bool(delta<=tol)})
    def metrics(y,p):
        y=np.asarray(y,dtype=float).ravel();p=np.asarray(p,dtype=float).ravel()
        if y.shape!=p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():raise ValueError('Invalid predictions')
        e=p-y;rmse=float(np.sqrt(np.mean(e**2)));mae=float(np.mean(np.abs(e)))
        return {'n':len(y),'r2':float(1-np.sum(e**2)/np.sum((y-y.mean())**2)),'rmse':rmse,'mae':mae,'spearman':float(spearmanr(y,p).statistic),'pearson_r2':float(pearsonr(y,p).statistic**2),'nrmse':rmse/float(y.mean()),'nmae':mae/float(y.mean())}
    # Canonical dataset sizes and exact fixed memberships / values.
    for num,n in [(1,75),(2,53),(3,112),(4,14),(5,152)]:
        paths=list((ROOT/'Data/curated').glob(f'DataS{num}_*.csv'))
        if len(paths)!=1:raise ValueError('Canonical dataset missing/duplicated')
        d=pd.read_csv(paths[0]);check(f'DataS{num} rows',len(d),n,0)
        idcol='mab_id' if num==5 else 'mAb_id';check(f'DataS{num} unique identifiers',d[idcol].nunique(),n,0)
        if num<3:
            ep='viscosity' if num==1 else 'mouse_exposure'
            for split,rows in [('train',52 if num==1 else 42),('test',23 if num==1 else 11)]:
                s=pd.read_csv(next((ROOT/f'Data/fixed_splits/{ep}').glob(f'DataS{num}_*_{split}.csv')))
                check(f'DataS{num} {split} rows',len(s),rows,0)
                if set(s.mAb_id)!=set(d.loc[d.split_seed0==split,'mAb_id']):raise ValueError('Changed split membership')
                a=d.set_index('mAb_id').loc[s.mAb_id,s.columns.difference(['mAb_id','split_seed0'])]
                b=s[s.columns.difference(['mAb_id','split_seed0'])]
                check(f'DataS{num} {split} values',np.max(np.abs(a.to_numpy(float)-b.to_numpy(float))),0,1e-10)
    # Primary endpoint records.
    for ep,path,keys in [('viscosity','MainPack/viscosity/results/primary/viscosity_results.mat',('y_test','y_pred')),('mouse_exposure','MainPack/mouse_exposure/results/primary/clearance_panel3A_data.mat',('trues','preds'))]:
        m=loadmat(ROOT/path);mm=metrics(m[keys[0]],m[keys[1]]);summaries.append({'analysis':ep,'source':path,**mm})
        for metric,key in [('r2','r2_transformer' if ep=='viscosity' else 'r2'),('rmse','rmse_transformer' if ep=='viscosity' else 'rmse'),('mae','mae_transformer')]:
            if key in m:check(ep+' primary '+metric,mm[metric],np.asarray(m[key]).item(),2e-6)
    # Clinical counts and archived metrics, including in-sample training separately.
    C=ROOT/'MainPack/clinical_outcome/results/targeted_repro';stored=pd.read_csv(C/'METRICS_AND_INTERVALS.csv');fs=loadmat(ROOT/'FigureSources/main/Figure_4/data/Figure4_FINAL_source.mat')
    for file,cohort,key in [('TRAIN_PREDICTIONS.csv','training_in_sample_ensemble','cm_train_in_sample_ensemble'),('INTERNAL_TEST_PREDICTIONS.csv','internal_test','cm_internal_test'),('EXTERNAL_PREDICTIONS.csv','external_resolved','cm_external_realized')]:
        d=pd.read_csv(C/file);y=d.true_label.eq('Approved').to_numpy();p=d.predicted_label.eq('Approved').to_numpy();cm=confusion_matrix(y,p,labels=[True,False])
        check(cohort+' softmax decisions',np.count_nonzero(p!=(d.positive_class_probability.to_numpy()>=.5)),0,0)
        check(cohort+' Figure 4 counts',np.max(np.abs(cm-fs[key])),0,0)
        mm={'n':len(y),'balanced_accuracy':float(balanced_accuracy_score(y,p)),'MCC':float(matthews_corrcoef(y,p)),'accuracy':float(accuracy_score(y,p))}
        row=stored.loc[stored.cohort==cohort].iloc[0]
        for k in ['balanced_accuracy','MCC']:check(cohort+' '+k,mm[k],row[k])
        summaries.append({'analysis':'clinical_'+cohort,'source':str((C/file).relative_to(ROOT)),**mm})
    # HIC OOF continuous and triage results, compared to stored aggregate records.
    H=ROOT/'GenBench/hic_benchmark/results';href=pd.read_csv(H/'summary/hic_existing_benchmark_summary.csv')
    for folder,stem,tag in [('assays_only','assays_only','assays_only'),('assays_plus_patch','all','all'),('patch_only','patch_only','patch_only')]:
        p=H/f'benchmark/oof_predictions/{folder}/hic_rt_min_OOF_{stem}_oof_predictions.csv';d=pd.read_csv(p)
        check('HIC '+folder+' rows',len(d),152,0)
        for col,label in [('OOF_ACeT','ACeT'),('OOF_BaillyRefit','BaillyRefit')]:
            mm=metrics(d.True_,d[col]) if 'True_' in d else metrics(d['True'],d[col])
            yy=d['True'].to_numpy()>30;pp=d[col].to_numpy()>30
            mm.update(balanced_accuracy=float(balanced_accuracy_score(yy,pp)),MCC=float(matthews_corrcoef(yy,pp)))
            row=href.loc[ (href.feature_set==tag) if label=='ACeT' else href.summary_item.eq('Bailly fold-matched refit baseline')].iloc[0]
            for k,colname in [('r2','sklearn_R2'),('pearson_r2','pearson_r2'),('spearman','spearman_rho'),('rmse','rmse'),('mae','mae'),('balanced_accuracy','triage_balanced_accuracy'),('MCC','triage_MCC')]:check('HIC '+folder+' '+label+' '+k,mm[k],row[colname],1e-7)
            summaries.append({'analysis':'hic_'+folder+'_'+label,'source':str(p.relative_to(ROOT)),**mm})
    # Learning curves and feature ablations: recalculate each condition from predictions.
    V=ROOT/'MainPack/viscosity/results/figure2de'
    for prefix in ['figure2e_learning_curve','figure2e_ablation']:
        d=pd.read_csv(V/(prefix+'_predictions.csv'));refs=pd.read_csv(V/(prefix+'_runs.csv'))
        grouping=['condition','seed']+(['requested_fraction_of_development_train'] if 'learning_curve' in prefix else [])
        for group_key,g in d.groupby(grouping):
            condition,seed=group_key[:2]
            select=refs.condition.eq(condition)&refs.seed.eq(seed)
            if len(group_key)==3:select &= np.isclose(refs.requested_fraction_of_development_train,float(group_key[2]))
            row=refs.loc[select].iloc[0];mm=metrics(g.true_viscosity,g.predicted_viscosity)
            if len(g)!=23:raise ValueError('Incorrect learning/ablation group size')
            for k in ['r2','rmse','mae','spearman']:check(f'{prefix} {group_key} {k}',mm[k],row[k],1e-6)
    # Permutation importance values, then their seed-level and aggregate summaries.
    d=pd.read_csv(V/'figure2d_permutation_predictions.csv');rep=pd.read_csv(V/'figure2d_permutation_repeats.csv')
    for (seed,feature,repeat),g in d.groupby(['training_seed','feature','repeat']):
        r2=metrics(g.true_viscosity,g.permuted_prediction)['r2'];delta=float(g.baseline_r2.iloc[0])-r2
        row=rep.loc[rep.training_seed.eq(seed)&rep.feature.eq(feature)&rep['repeat'].eq(repeat)].iloc[0]
        check(f'permutation {seed} {feature} {repeat}',delta,row.importance_delta_r2,1e-6)
    s=pd.read_csv(V/'figure2d_seed_importance.csv');su=pd.read_csv(V/'figure2d_importance_summary.csv')
    for (seed,feature),g in rep.groupby(['training_seed','feature']):check(f'importance seed mean {seed} {feature}',g.importance_delta_r2.mean(),s.loc[s.training_seed.eq(seed)&s.feature.eq(feature),'importance_mean'].iloc[0])
    for feature,g in s.groupby('feature'):
        rr=su.loc[su.feature.eq(feature)].iloc[0]
        check('importance mean '+feature,g.importance_mean.mean(),rr.mean_importance);check('importance SD '+feature,g.importance_mean.std(ddof=1),rr.sd_across_seeds)
    # Pooling controls: both CV folds and held-out ensembles; S9, S11.
    G=ROOT/'GenBench/diagnostics/pooling_sensitivity/results'
    for ep in ['viscosity','clearance']:
        p=pd.read_csv(G/f'{ep}_full_predictions.csv');ref=pd.read_csv(G/f'{ep}_full_metrics.csv')
        for (pool,fold,split),g in p.groupby(['pooling','fold','split']):
            mm=metrics(g['true'],g.predicted);row=ref.loc[ref.pooling.eq(pool)&ref.fold.astype(str).eq(str(fold))&ref.split.eq(split)].iloc[0]
            for k in ['r2','rmse','mae','spearman']:check(f'pooling {ep} {pool} {fold} {split} {k}',mm[k],row[k],5e-5 if k in ('rmse','mae') else 1e-6)
        history=pd.read_csv(G/f'{ep}_full_training_history.csv');check(ep+' training history nonfinite',np.count_nonzero(~np.isfinite(history[['loss','val_loss']].to_numpy())),0,0)
    # Fixed holdouts and changing membership are evaluated separately.
    for kind in ['fixed','alternate']:
        F=ROOT/('GenBench/robustness/fixed_holdout_stability/results/locked' if kind=='fixed' else 'GenBench/robustness/split_sensitivity/results/locked')
        for ep in ['viscosity','clearance']:
            pn=f'{ep}_fixed_holdout_predictions.csv' if kind=='fixed' else ('viscosity_alternate_partition_predictions.csv' if ep=='viscosity' else 'mouse_exposure_alternate_partition_predictions.csv')
            rn=f'{ep}_fixed_holdout_metrics.csv' if kind=='fixed' else ('viscosity_alternate_partition_metrics.csv' if ep=='viscosity' else 'mouse_exposure_alternate_partition_metrics.csv')
            d=pd.read_csv(F/pn);ref=pd.read_csv(F/rn);seedcol='training_seed' if kind=='fixed' else 'split_seed'
            for seed,g in d.groupby(seedcol):
                yc='true_value' if 'true_value' in g else 'true';pc='predicted_value' if 'predicted_value' in g else 'predicted'
                mm=metrics(g[yc],g[pc]);row=ref.loc[ref[seedcol].eq(seed)].iloc[0]
                for k in ['r2','rmse','mae','spearman']:check(f'{kind} {ep} {seed} {k}',mm[k],row[k],5e-5 if k in ('rmse','mae') else 1e-6)
    # Gaussian-copula diagnostics from the retained real/synthetic samples (S10).
    F=ROOT/'GenBench/diagnostics/gaussian_copula/results'
    for ep in ['viscosity','clearance']:
        d=pd.read_csv(F/f'{ep}_real_vs_synthetic_samples.csv');ref=pd.read_csv(F/f'{ep}_marginal_statistics.csv')
        real=d.loc[d.source.eq('real_training')];syn=d.loc[d.source.eq('synthetic')]
        if len(real)==0 or len(syn)==0:raise ValueError('Diagnostic source labels changed')
        for r in ref.to_dict('records'):
            f=r['variable'];check(f'KS {ep} {f}',ks_2samp(real[f],syn[f]).statistic,r['ks_statistic']);check(f'Wasserstein {ep} {f}',wasserstein_distance(real[f],syn[f]),r['wasserstein_distance'])
        for label,frame in [('real',real),('synthetic',syn)]:
            columns=ref.variable.tolist();cal=frame[columns].corr(method='spearman');old=pd.read_csv(F/f'{ep}_correlation_{label}.csv',index_col=0).loc[columns,columns]
            check(f'Spearman matrix {ep} {label}',np.max(np.abs(cal.to_numpy()-old.to_numpy())),0,1e-8)
    # sensitivity verifier includes the full defined candidate screen and all six HPO winners.
    p=ROOT/'SensitivityAnalyses/verify_sensitivity.py';sp=importlib.util.spec_from_file_location('sensitivity_verification',p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
    sensitivity=m.numeric_checks()
    failed=[x for x in checks if not x['matches']]
    return {'numeric_checks':checks,'failed_checks':failed,'endpoint_summaries':summaries,'sensitivity_checks':sensitivity,'full_neural_training_performed':False}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--checksums-only',action='store_true');ap.add_argument('--report',type=Path);a=ap.parse_args()
    result={'manifest_entries_verified':integrity()}
    if not a.checksums_only:result.update(numerical_checks())
    if a.report:
        p=a.report.resolve()
        if p.exists():raise FileExistsError('Use a new report filename')
        p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(result,indent=2)+'\n')
    if result.get('failed_checks'):raise SystemExit(f"{len(result['failed_checks'])} numerical checks differ; see report.")
    print(f"Integrity verified for {result['manifest_entries_verified']} files.")
    if not a.checksums_only:print(f"{len(result['numeric_checks'])} archived numerical checks match; sensitivity metrics, candidate ranking and HPO winners also verified. No neural training performed.")
if __name__=='__main__':main()
