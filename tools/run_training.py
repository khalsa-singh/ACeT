#!/usr/bin/env python3
"""Validate inputs and launch the endpoint-specific ACeT training implementations."""
from __future__ import annotations
import argparse,csv,json,math,os,subprocess,sys,importlib.util,importlib.metadata,ast,shlex
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
MAIN={
 'viscosity': {'script':'MainPack/viscosity/code/source/shared/panel_A_D_parity_featureimp.py','train':'MainPack/viscosity/code/source/shared/antibodies_train.csv','test':'MainPack/viscosity/code/source/shared/antibodies_test.csv','head':'kan','task':'regression','rows':(52,23),'feature_count':4},
 'mouse_exposure': {'script':'MainPack/mouse_exposure/code/source/shared/panel_A_B_parity.py','train':'MainPack/mouse_exposure/code/source/shared/clearance_train.csv','test':'MainPack/mouse_exposure/code/source/shared/clearance_test.csv','head':'spline','task':'regression','rows':(42,11),'feature_count':4},
 'clinical_outcome': {'script':'MainPack/clinical_outcome/code/source/figure4_targeted/panels.py','train':'MainPack/clinical_outcome/code/source/shared/InternalCohort_112mAbs_train.csv','test':'MainPack/clinical_outcome/code/source/shared/InternalCohort_112mAbs_test.csv','external':'MainPack/clinical_outcome/code/source/shared/ExternalCohort_14mAbs.csv','head':'mlp','task':'classification','rows':(89,23),'feature_count':5},
 'hic':{'script':'GenBench/hic_benchmark/code/source/shared/panel_A_D_parity_featureimp_EDITED_v2.py','data':'Data/curated/DataS5_bailly2020_hicrt_no_leakage_all_features.csv','head':'mlp','task':'regression'},
}
SENS={
 'viscosity':dict(MAIN['viscosity'],script='SensitivityAnalyses/DevelopmentOnlyFeatureSelection/viscosity/code/panel_A_D_parity_featureimp.py',train='SensitivityAnalyses/DevelopmentOnlyFeatureSelection/viscosity/data/antibodies_train.csv',test='SensitivityAnalyses/DevelopmentOnlyFeatureSelection/viscosity/data/antibodies_test.csv'),
 'mouse_exposure':dict(MAIN['mouse_exposure'],script='SensitivityAnalyses/DevelopmentOnlyFeatureSelection/mouse_exposure/code/panel_A_B_parity_training_only_sensitivity.py',train='SensitivityAnalyses/DevelopmentOnlyFeatureSelection/mouse_exposure/data/train42_model_ready_selected4.csv',test='SensitivityAnalyses/DevelopmentOnlyFeatureSelection/mouse_exposure/data/test11_raw_untouched_selected4.csv',head='mlp'),
}

MATCHED={
 'viscosity':dict(MAIN['viscosity'],script='MainPack/shared/minimal_baseline_patch/viscosity_panel_C_baselines_minimal_patch.py'),
 'mouse_exposure':dict(MAIN['mouse_exposure'],script='MainPack/shared/minimal_baseline_patch/clearance_panel_C_baselines_minimal_patch.py'),
}
SHAP_MODELS={'mouse_exposure':dict(MAIN['mouse_exposure'],script='MainPack/mouse_exposure/code/source/shared/panel_D_shap.py')}
def collection(analysis):
    return {'main':MAIN,'feature-selection':SENS,'matched-comparison':MATCHED,'shap':SHAP_MODELS}[analysis]

def read_numeric(path,n,features,classification=False,target=None):
    with path.open(encoding='utf-8-sig',newline='') as f:r=csv.DictReader(f);rows=list(r);cols=r.fieldnames
    if len(rows)!=n or len(cols)!=features+1:raise ValueError(f'{path.name}: expected {n} rows and {features+1} columns.')
    for row in rows:
        for col in cols[:-1]:
            if not math.isfinite(float(row[col])):raise ValueError(f'Nonfinite {col} in {path.name}')
        if classification:
            if row[cols[-1]] not in {'Approved','Terminated'}:raise ValueError('Unrecognized clinical outcome')
        elif not math.isfinite(float(row[cols[-1]])):raise ValueError('Nonfinite target')
    return cols

def specification(endpoint,analysis,hic_features,head=None):
    cfg=collection(analysis)[endpoint].copy();script=ROOT/cfg['script']
    if not script.is_file():raise FileNotFoundError(script)
    cmd=[sys.executable,str(script),'--task',cfg['task'],'--head_type',head or cfg['head']]
    if endpoint=='hic':
        p=ROOT/cfg['data']
        with p.open(encoding='utf-8-sig',newline='') as f:r=csv.DictReader(f);data=list(r);cols=r.fieldnames
        if len(data)!=152 or len({x['mab_id'] for x in data})!=152:raise ValueError('HIC identifiers/rows differ')
        for row in data:
            for c in cols:
                if c!='mab_id' and not math.isfinite(float(row[c])):raise ValueError('Nonfinite HIC input')
        cmd+=['--analysis_mode','oof_cv','--data_file',str(p),'--id_col','mab_id','--target_col','hic_rt_min','--feature_set',hic_features,'--cv_repeats','10']
        summary={'rows':152,'repeats':10,'folds_per_repeat':5,'feature_set':hic_features}
    else:
        tr=ROOT/cfg['train'];te=ROOT/cfg['test']
        c1=read_numeric(tr,cfg['rows'][0],cfg['feature_count'],cfg['task']=='classification')
        c2=read_numeric(te,cfg['rows'][1],cfg['feature_count'],cfg['task']=='classification')
        if c1!=c2:raise ValueError('Different train/test column order')
        cmd+=['--train_file',str(tr),'--test_file',str(te)]
        if 'external' in cfg:
            ext=ROOT/cfg['external'];ce=read_numeric(ext,14,5,True)
            if ce[:-1]!=c1[:-1]:raise ValueError('External feature order differs')
            cmd+=['--external_file',str(ext),'--status_col','Updated.Status']
        summary={'development_rows':cfg['rows'][0],'heldout_rows':cfg['rows'][1],'columns':c1}
    # Validate options against the actual source without importing TensorFlow.
    t=ast.parse(script.read_text());flags=set()
    for n in ast.walk(t):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='add_argument':
            flags.update(a.value for a in n.args if isinstance(a,ast.Constant) and isinstance(a.value,str))
    for flag in cmd[2:]:
        if flag.startswith('--') and flag not in flags:raise ValueError(f'Unrecognized source flag {flag}')
    return cmd,{'endpoint':endpoint,'analysis':analysis,'head':head or cfg['head'],'source':cfg['script'],**summary}

def dependencies():
    names=['tensorflow','tfkan','sdv','ImbalancedLearningRegression','shap','imblearn','seaborn']
    missing=[n for n in names if importlib.util.find_spec(n) is None]
    if missing:raise RuntimeError('Activate the Python 3.11 training environment described in README.md. Missing: '+', '.join(missing))
    if importlib.metadata.version('scikit-learn')!='1.4.1.post1':raise RuntimeError('Recorded training scripts use scikit-learn 1.4.1.post1. Use requirements-training.txt in a separate environment.')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('endpoint',choices=[*MAIN,'all']);ap.add_argument('--analysis',choices=['main','feature-selection','matched-comparison','shap'],default='main')
    ap.add_argument('--hic-features',choices=['assays_only','patch_only','all'],default='assays_only')
    ap.add_argument('--head',choices=['mlp','rbf','spline','kan','interaction','all'])
    ap.add_argument('--check-inputs',action='store_true');ap.add_argument('--output-dir',type=Path)
    ap.add_argument('--epochs',type=int);ap.add_argument('--seed',type=int,default=0)
    a=ap.parse_args();endpoints=list(collection(a.analysis)) if a.endpoint=='all' else [a.endpoint]
    if any(ep not in collection(a.analysis) for ep in endpoints):ap.error('Endpoint is not available for this analysis. See --help and README.md.')
    plans=[]
    for ep in endpoints:
        heads=['mlp','rbf','spline','kan','interaction'] if a.head=='all' else [a.head]
        for chosen in heads:
            cmd,meta=specification(ep,a.analysis,a.hic_features,chosen)
            cmd+=['--seed',str(a.seed)]
            if a.epochs:
                if a.epochs<1:ap.error('--epochs must be positive')
                cmd+=['--epochs',str(a.epochs)]
            label=ep+'_'+str(chosen) if a.head=='all' else ep
            plans.append((label,cmd,meta));print(json.dumps(meta,indent=2));print('Command:',shlex.join(cmd))
    if a.check_inputs:return
    dependencies()
    base=(a.output_dir or ROOT/'reruns'/f'train_{a.analysis}_{a.endpoint}').resolve()
    if base==ROOT or (ROOT in base.parents and ROOT/'reruns' not in base.parents):ap.error('Choose a location under reruns/ or outside the repository.')
    if base.exists() and any(base.iterdir()):raise FileExistsError('Choose a new empty output directory.')
    for ep,cmd,meta in plans:
        out=base/ep if len(plans)>1 else base;out.mkdir(parents=True,exist_ok=True)
        (out/'training_request.json').write_text(json.dumps({**meta,'command':cmd,'seed':a.seed},indent=2))
        print(f'Training {ep}; output: {out}',flush=True)
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED=str(a.seed),MPLBACKEND='Agg')
        with (out/'training.log').open('w',encoding='utf-8') as log:
            res=subprocess.run(cmd,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT)
        if res.returncode:raise SystemExit(f'{ep} exited {res.returncode}. See {out / "training.log"}')
        print(f'{ep} completed; outputs: {out}')
if __name__=='__main__':main()
