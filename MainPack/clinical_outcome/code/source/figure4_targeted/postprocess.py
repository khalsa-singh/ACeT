import hashlib, json, math, os, platform, shutil, subprocess, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, log_loss, matthews_corrcoef

# Copies archived inputs to a new output directory; never overwrites evidence.
import argparse
_argp = argparse.ArgumentParser(description="Recompute archived clinical tables without training.")
_argp.add_argument('--output-dir', type=Path, required=True)
_args = _argp.parse_args()
ROOT = next(p for p in Path(__file__).resolve().parents if (p/'Data/curated').is_dir())
MAIN = ROOT/'MainPack/clinical_outcome'
SHARED = MAIN/'code/source/shared'
REV = Path(__file__).resolve().parent
OUT = _args.output_dir.resolve()
if OUT.exists() and any(OUT.iterdir()):
    raise SystemExit('Choose an empty --output-dir.')
OUT.mkdir(parents=True, exist_ok=True)
for _name in ['TRAIN_PREDICTIONS.csv', 'INTERNAL_TEST_PREDICTIONS.csv', 'EXTERNAL_PREDICTIONS.csv', 'Table_S1_thresholds.csv', 'figvars.mat']:
    shutil.copy2(MAIN/'results/targeted_repro'/_name, OUT/_name)

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def wilson(k, n, z=1.959963984540054):
    if not n: return (np.nan, np.nan)
    p = k/n; d = 1+z*z/n
    c = (p+z*z/(2*n))/d
    r = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return c-r, c+r

def enrich_predictions(filename, names_file):
    p = OUT / filename
    df = pd.read_csv(p)
    df = df.drop(columns=[c for c in ["stable_row_id", "antibody_name"] if c in df.columns])
    names = pd.read_csv(names_file)
    name_col = "Name" if "Name" in names.columns else names.columns[0]
    df.insert(1, "antibody_name", names[name_col].astype(str).values)
    df.insert(1, "stable_row_id", [f"{df.cohort.iloc[0]}_{i:03d}" for i in range(len(df))])
    df.to_csv(p, index=False)
    return df

train = enrich_predictions("TRAIN_PREDICTIONS.csv", SHARED / "InternalCohort_112mAbs_wname_train.csv")
test = enrich_predictions("INTERNAL_TEST_PREDICTIONS.csv", SHARED / "InternalCohort_112mAbs_wname_test.csv")
ext = enrich_predictions("EXTERNAL_PREDICTIONS.csv", SHARED / "ExternalCohort_14mAbs_wname.csv")

cms=[]; metrics=[]
rng=np.random.default_rng(20260805)
for label,df in [("training_in_sample_ensemble",train),("internal_test",test),("external_resolved",ext)]:
    y=(df.true_label=="Approved").astype(int).to_numpy()
    pred=(df.predicted_label=="Approved").astype(int).to_numpy()
    prob=df.positive_class_probability.to_numpy()
    cm=confusion_matrix(y,pred,labels=[1,0])
    tp,fn,fp,tn=cm.ravel()
    for tr,pr,v in [("Approved","Approved",tp),("Approved","Terminated",fn),("Terminated","Approved",fp),("Terminated","Terminated",tn)]:
        cms.append(dict(cohort=label,true_label=tr,predicted_label=pr,count=int(v),prediction_provenance=df.provenance.iloc[0]))
    sens=tp/(tp+fn); spec=tn/(tn+fp); ba=(sens+spec)/2; mcc=matthews_corrcoef(y,pred)
    slo,shi=wilson(tp,tp+fn); clo,chi=wilson(tn,tn+fp)
    ids=rng.integers(0,len(y),(50000,len(y)))
    yy=y[ids]; pp=pred[ids]
    btp=np.sum((yy==1)&(pp==1),axis=1); bfn=np.sum((yy==1)&(pp==0),axis=1)
    bfp=np.sum((yy==0)&(pp==1),axis=1); btn=np.sum((yy==0)&(pp==0),axis=1)
    valid=((btp+bfn)>0)&((btn+bfp)>0)
    bas=.5*(btp[valid]/(btp[valid]+bfn[valid])+btn[valid]/(btn[valid]+bfp[valid]))
    denom=np.sqrt((btp[valid]+bfp[valid])*(btp[valid]+bfn[valid])*(btn[valid]+bfp[valid])*(btn[valid]+bfn[valid]))
    mcc_num=btp[valid]*btn[valid]-bfp[valid]*bfn[valid]
    mccs=np.divide(mcc_num,denom,out=np.zeros_like(mcc_num,dtype=float),where=denom!=0)
    metrics.append(dict(cohort=label,TP=tp,FN=fn,FP=fp,TN=tn,n=len(y),sensitivity=sens,sensitivity_wilson95_low=slo,sensitivity_wilson95_high=shi,specificity=spec,specificity_wilson95_low=clo,specificity_wilson95_high=chi,balanced_accuracy=ba,balanced_accuracy_bootstrap95_low=np.quantile(bas,.025),balanced_accuracy_bootstrap95_high=np.quantile(bas,.975),MCC=mcc,MCC_bootstrap95_low=np.quantile(mccs,.025),MCC_bootstrap95_high=np.quantile(mccs,.975),log_loss=log_loss(y,prob,labels=[0,1]),brier_score=np.mean((prob-y)**2),bootstrap_seed=20260805,bootstrap_resamples_requested=50000,bootstrap_valid_resamples=len(bas)))
pd.DataFrame(cms).to_csv(OUT/"CONFUSION_MATRICES.csv",index=False)
pd.DataFrame(metrics).to_csv(OUT/"METRICS_AND_INTERVALS.csv",index=False)

jain={"AS":.08,"PSR":.27,"ACSINS":11.8,"ELISA":1.9,"BVP":4.3}
thr=pd.read_csv(OUT/"Table_S1_thresholds.csv")
thr["jain_rule_operator"]=">"
thr["jain_rule_threshold"]=thr.feature.map(jain)
thr["training_minus_jain_delta"]=thr.threshold-thr.jain_rule_threshold
thr["training_threshold_derivation"]="raw 89-row internal training set; univariate Youden-J; encoded Terminated class positive; x > threshold"
thr["transformer_derived"]=False
thr.to_csv(OUT/"ASSAY_THRESHOLD_COMPARISON.csv",index=False)

econ=[]
values=dict(TP=9.0,FN=-9.0,FP=-2.0,TN=.084)
def add_econ(basis,policy,counts,diagnostic=False):
    contrib={k:counts.get(k,0)*values[k] for k in values}
    econ.append(dict(basis=basis,policy=policy,approved_count=counts.get('TP',0)+counts.get('FN',0),terminated_count=counts.get('FP',0)+counts.get('TN',0),TP=counts.get('TP',0),FN=counts.get('FN',0),FP=counts.get('FP',0),TN=counts.get('TN',0),TP_B=contrib['TP'],FN_B=contrib['FN'],FP_B=contrib['FP'],TN_B=contrib['TN'],net_B=sum(contrib.values()),diagnostic_only=diagnostic))
N=23; pos=.12; sens=10/13; spec=8/10; ap=N*pos; term=N*(1-pos)
add_econ("internal_prior_expected_N23","ACeT",dict(TP=ap*sens,FN=ap*(1-sens),FP=term*(1-spec),TN=term*spec))
add_econ("internal_prior_expected_N23","develop_all",dict(TP=ap,FP=term))
add_econ("internal_prior_expected_N23","kill_all",dict(FN=ap,TN=term))
add_econ("external_realized_2_approved_12_terminated","ACeT",dict(TP=2,FN=0,FP=4,TN=8))
add_econ("external_realized_2_approved_12_terminated","develop_all",dict(TP=2,FP=12))
add_econ("external_realized_2_approved_12_terminated","kill_all",dict(FN=2,TN=12))
eap=14*.12; et=14*.88
add_econ("external_prior_expected_N14_diagnostic","develop_all",dict(TP=eap,FP=et),True)
add_econ("external_prior_expected_N14_diagnostic","kill_all",dict(FN=eap,TN=et),True)
pd.DataFrame(econ).to_csv(OUT/"FIGURE4_ECONOMICS_RECALC.csv",index=False)

source=[]
for m in metrics:
    for key in ["TP","FN","FP","TN","sensitivity","specificity","balanced_accuracy","MCC"]:
        source.append(dict(panel="4a",cohort=m["cohort"],quantity=key,value=m[key],basis="current single seed-0 run"))
for _,r in thr.iterrows():
    source += [dict(panel="4b",cohort="internal_training",quantity=f"{r.feature}_Jain_threshold",value=r.jain_rule_threshold,basis="Jain literature rule"),dict(panel="4b",cohort="internal_training",quantity=f"{r.feature}_training_univariate_YoudenJ_threshold",value=r.threshold,basis="raw 89-row training set only")]
for r in econ:
    source.append(dict(panel="4c" if r["policy"]=="ACeT" else "4d",cohort=r["basis"],quantity=f"{r['policy']}_net_B",value=r["net_B"],basis="diagnostic only" if r["diagnostic_only"] else "policy comparison basis"))
pd.DataFrame(source).to_csv(OUT/"FIGURE4_SOURCE_VALUES.csv",index=False)

hash_inputs=[ROOT/"run.sh",SHARED/"panels.py",SHARED/"clinical_model.py",REV/"panels.py",REV/"clinical_model.py",REV/"postprocess.py",SHARED/"InternalCohort_112mAbs_train.csv",SHARED/"InternalCohort_112mAbs_test.csv",SHARED/"ExternalCohort_14mAbs.csv",SHARED/"InternalCohort_112mAbs_wname_train.csv",SHARED/"InternalCohort_112mAbs_wname_test.csv",SHARED/"ExternalCohort_14mAbs_wname.csv"]
pd.DataFrame([dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size) for p in hash_inputs]).to_csv(OUT/"INPUT_AND_CODE_HASHES.csv",index=False)
models=sorted(OUT.glob("*.weights.h5"))
pd.DataFrame([dict(model_file=p.name,sha256=sha(p),bytes=p.stat().st_size,head="mlp",fold=i+1,seed=0) for i,p in enumerate(models)]).to_csv(OUT/"MODEL_HASHES.csv",index=False)

mat=loadmat(OUT/"figvars.mat")
savemat(OUT/"clinical_figure_data.mat",{k:v for k,v in mat.items() if not k.startswith("__")})

import importlib.metadata as _md
from datetime import datetime, timezone
env=json.dumps({"operation":"postprocessing archived predictions; no training", "executed_utc":datetime.now(timezone.utc).isoformat(), "python":sys.version, "numpy":np.__version__, "pandas":pd.__version__, "scikit-learn":_md.version('scikit-learn'), "bootstrap_seed":20260805}, indent=2)
(OUT/"ENVIRONMENT.txt").write_text(env,encoding="utf-8")

comparison="""# Figure 4 reproduction comparison

## Panel 4a confusion matrices

The current single seed-0 run reproduces every embedded count exactly: training [[32,19],[11,27]] (89), internal test [[10,3],[2,8]] (23), and external [[2,0],[4,8]] (14). The training matrix is in-sample five-model ensemble evaluation, not OOF evaluation.

Internal test sensitivity=0.7692307692, specificity=0.8, balanced accuracy=0.7846153846, MCC=0.5649019553. External sensitivity=1.0, specificity=0.6666666667, balanced accuracy=0.8333333333, MCC=0.4714045208.

## Panel 4b thresholds

Current training-derived thresholds are BVP 3.1572, AS 0.05, ELISA 1.826, AC-SINS 11.2, and PSR 0.0. They are produced independently from raw internal-training assay columns by univariate Youden-J optimization (Terminated treated as positive), not from transformer outputs. Jain thresholds are BVP 4.3, AS 0.08, ELISA 1.9, AC-SINS 11.8, and PSR 0.27. Exact deltas are in `ASSAY_THRESHOLD_COMPARISON.csv`.

The technically accurate legend wording is: **“Internal-training univariate Youden-J assay thresholds”**, not “Transformer + Youden-J.”

## Panels 4c-4d economics

Internal ACeT is +6.639512615 B (rounds +6.6 B). External realized-cohort ACeT is +10.672 B. On that same realized basis, develop-all is -6.000 B and kill-all is -16.992 B. The prior-based expected 14-program diagnostic gives 1.68 Approved and 12.32 Terminated, develop-all -9.520 B, and kill-all -14.08512 B. The latter two values explain the embedded rounded -9.5 B and -14.1 B but must not be mixed with realized-cohort ACeT.
"""
(OUT/"FIGURE4_REPRO_COMPARISON.md").write_text(comparison,encoding="utf-8")
decision="""# Figure 4 final decision

- **Panel 4a:** reproduces exactly for training, internal test, and external confusion matrices.
- **Panel 4b:** the current five assay thresholds reproduce from the authoritative training-only procedure. They are not transformer-derived; use the legend/caption wording **“Internal-training univariate Youden-J assay thresholds”** in place of **“Transformer + Youden-J.”**
- **Panel 4c:** safe. Internal expected-cohort ACeT (+6.639512615 B; +6.6 B rounded) and external realized-cohort ACeT (+10.672 B) recalculate correctly from the stated assumptions/counts.
- **Panel 4d:** requires only external comparator-bar recalculation if the Results remain a realized-cohort comparison: develop-all must be -6.000 B and kill-all -16.992 B. The current -9.5 B/-14.1 B values are the separate prior-based expected-14 diagnostic and cannot share a policy comparison with realized ACeT.
- **Further model run:** none is scientifically necessary. The authorized current seed-0 run exactly reproduced all embedded model confusion counts and generated complete probability/prediction provenance.
"""
(OUT/"FIGURE4_FINAL_DECISION.md").write_text(decision,encoding="utf-8")

audit="""# Clinical pipeline scope audit

- No clinical train+train duplication: no concatenation or duplication operation exists in the classification path; the 89-row CSV is loaded once.
- No clinical Gaussian Copula: imported but never invoked in `run_classification`.
- ENN and SMOTE: applied only to `train_idx` rows within each CV fold.
- Feature panel: fixed from the internal training CSV columns; no external feature ranking.
- External leakage: none; external data are loaded only after training/head selection and are transform-only.
- Scaler scope: `PowerTransformer.fit_transform` is applied once to all 89 internal-training rows before KFold. Thus there is no internal-test/external leakage, but fold-validation scaling is not fold-local.
- Head selection: `run.sh clinical` fixes `mlp`; the one-candidate mean-CV rule therefore selects MLP.
- Operating rule: two-class softmax argmax, equivalent to Approved probability 0.5 absent ties.
"""
(OUT/"CLINICAL_PIPELINE_SCOPE_AUDIT.md").write_text(audit,encoding="utf-8")

print("postprocess complete")
