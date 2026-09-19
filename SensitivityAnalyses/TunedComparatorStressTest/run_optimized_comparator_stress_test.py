"""Package-relative reproduction of the development-CV-tuned comparator analysis.

verify: validate supplied caches, inputs and lock; no fitting.
cv: refit the 30 locked fold models and report validation metrics.
preflight: rebuild the original fixed-control caches (study SDV/IBLR required).
search: repeat the finite development-cache search into a new output directory.
heldout: evaluate locked models after reconstructing the development-fitted
         predictor transformation (study SDV required).

Archived results and parameter locks are never overwritten. See README.md.
"""
from __future__ import annotations
import argparse, hashlib, json, os, platform, random, shutil, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, spearmanr, uniform
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, ParameterSampler
from sklearn.preprocessing import QuantileTransformer
from sklearn.svm import SVR
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'
CACHE=ROOT/'fold_cache'
OUT=ROOT/'rerun_outputs'
SEED=0
SEARCH_SEED=20260901
MODELS=('Ridge','SVR','RandomForest')
ENDPOINTS={
 'viscosity':{'train':DATA/'viscosity_train.csv','test':DATA/'viscosity_test.csv',
  'annotated_train':DATA/'DataS1_viscosity_seed0_train.csv','annotated_test':DATA/'DataS1_viscosity_seed0_test.csv',
  'synthetic_n':26,'resampling':'enn_then_smote'},
 'mouse_exposure':{'train':DATA/'clearance_train.csv','test':DATA/'clearance_test.csv',
  'annotated_train':DATA/'DataS2_clearance_seed0_train.csv','annotated_test':DATA/'DataS2_clearance_seed0_test.csv',
  'synthetic_n':84,'resampling':'smote_then_enn'}
}
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def metrics(y, p):
    return {
        "r2": r2_score(y, p),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "mae": mean_absolute_error(y, p),
        "spearman_rho": spearmanr(y, p).statistic,
    }

def model(name, params=None):
    params = dict(params or {})
    if name == "Ridge":
        return Ridge(**params)
    if name == "SVR":
        return SVR(kernel="rbf", **params)
    return RandomForestRegressor(random_state=SEED, n_jobs=1, **params)

def resample(df, target, order):
    import ImbalancedLearningRegression as iblr
    if order == "enn_then_smote":
        clean = iblr.enn(data=df, y=target, rel_coef=.5)
        if clean.isnull().values.any():
            clean = clean.dropna()
        clean = clean.loc[:, clean.nunique() > 1]
        try:
            balanced = iblr.smote(data=clean, y=target, rel_coef=.5)
        except ValueError:
            pass
        try:
            balanced = iblr.smote(data=clean, y=target, rel_coef=.25)
        except ValueError:
            balanced = clean.copy()
        return balanced
    clean = iblr.smote(data=df, y=target, rel_coef=.5)
    if clean.isnull().values.any():
        clean = clean.dropna()
    clean = clean.loc[:, clean.nunique() > 1]
    try:
        balanced = iblr.enn(data=clean, y=target, rel_coef=.5)
    except ValueError:
        pass
    try:
        balanced = iblr.enn(data=clean, y=target, rel_coef=.25)
    except ValueError:
        balanced = clean.copy()
    return balanced

def prepare_endpoint(endpoint, read_test):
    """Recreate the authoritative mixture, transform, folds and RNG sequence."""
    cfg = ENDPOINTS[endpoint]
    random.seed(SEED); np.random.seed(SEED)
    train = pd.read_csv(cfg["train"])
    test = pd.read_csv(cfg["test"]) if read_test else None
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import GaussianCopulaSynthesizer
    metadata = SingleTableMetadata(); metadata.detect_from_dataframe(train)
    synthesizer = GaussianCopulaSynthesizer(metadata, default_distribution="norm")
    synthesizer.fit(train)
    try:
        synthetic = synthesizer.sample(num_rows=cfg["synthetic_n"], random_state=SEED)
    except TypeError:
        synthetic = synthesizer.sample(num_rows=cfg["synthetic_n"])
    mix = pd.concat([train, train, synthetic], ignore_index=True)
    scaler = QuantileTransformer()
    X = scaler.fit_transform(mix.iloc[:, :-1].values)
    y = mix.iloc[:, -1].values
    Xtest = scaler.transform(test.iloc[:, :-1].values) if test is not None else None
    ytest = test.iloc[:, -1].values if test is not None else None
    folds = list(KFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y))
    return train, test, X, y, Xtest, ytest, folds

def cache_fixed_and_reproduce(endpoint):
    cfg = ENDPOINTS[endpoint]
    train, test, X, y, Xtest, ytest, folds = prepare_endpoint(endpoint, True)
    feature_cols, target = list(train.columns[:-1]), train.columns[-1]
    # Advance IBLR's global RNG exactly through the five preceding ACeT fold
    # resampling calls. TensorFlow has its own RNG and does not alter np.random.
    for tr, _ in folds:
        d = pd.DataFrame(X[tr], columns=feature_cols); d[target] = y[tr]
        resample(d, target, cfg["resampling"])
    fixed_rows, manifest_rows = [], []
    CACHE.mkdir(parents=True, exist_ok=True)
    for name in MODELS:
        fold_models = []
        for fold, (tr, vl) in enumerate(folds, 1):
            d = pd.DataFrame(X[tr], columns=feature_cols); d[target] = y[tr]
            b = resample(d, target, cfg["resampling"])
            Xtr, ytr = b.iloc[:, :-1].values, b[target].values
            cache_path = CACHE / f"{endpoint}__{name}__fold{fold}.npz"
            np.savez_compressed(cache_path, X_train=Xtr, y_train=ytr,
                                X_validation=X[vl], y_validation=y[vl],
                                validation_indices=vl)
            manifest_rows.append({
                "endpoint": endpoint, "model": name, "fold": fold,
                "n_train": len(ytr), "n_validation": len(vl),
                "n_features": Xtr.shape[1], "cache_file": cache_path.relative_to(OUT).as_posix(),
                "sha256": sha256(cache_path),
            })
            m = model(name); m.fit(Xtr, ytr); fold_models.append(m)
            mm = metrics(y[vl], m.predict(X[vl]))
            fixed_rows.append({"endpoint": endpoint, "model": name, "scope": "cv_fold", "fold": fold, **mm})
        pred = np.mean([m.predict(Xtest) for m in fold_models], axis=0)
        fixed_rows.append({"endpoint": endpoint, "model": name, "scope": "held_out", "fold": "", **metrics(ytest, pred)})
    return fixed_rows, manifest_rows

def candidate_sets(endpoint, name, target_sd):
    if name == "Ridge":
        alphas = np.logspace(-8, 8, 33)
        return [{"alpha":float(a),"fit_intercept":i,"solver":s,"tol":float(t)}
                for a in alphas for i in (True,False) for s in ("auto","svd","cholesky","lsqr")
                for t in (1e-6,1e-5,1e-4,1e-3,1e-2)]
    rng_seed = SEARCH_SEED + (0 if endpoint == "viscosity" else 1000) + (10 if name == "SVR" else 20)
    if name == "SVR":
        # 220 random candidates plus scale/auto coverage; all deterministic.
        numeric = list(ParameterSampler({"C":loguniform(1e-3,1e5), "epsilon":loguniform(1e-4*target_sd,.5*target_sd),
            "gamma":loguniform(1e-5,1e2), "shrinking":[True,False], "tol":loguniform(1e-5,1e-2)}, n_iter=180, random_state=rng_seed))
        symbolic = list(ParameterSampler({"C":loguniform(1e-3,1e5), "epsilon":loguniform(1e-4*target_sd,.5*target_sd),
            "gamma":["scale","auto"], "shrinking":[True,False], "tol":loguniform(1e-5,1e-2)}, n_iter=40, random_state=rng_seed+1))
        return numeric + symbolic
    raw = ParameterSampler({"n_estimators":[200,300,500,750,1000,1500], "max_depth":[None,*range(2,21)],
        "min_samples_split":randint(2,17), "min_samples_leaf":randint(1,9),
        "max_features":["sqrt","log2",.25,.4,.55,.7,.85,1.0], "bootstrap":[True,False],
        "max_samples":[.5,.6,.7,.8,.9,1.0,None], "criterion":["squared_error","absolute_error","friedman_mse"]},
        n_iter=400, random_state=rng_seed)
    valid=[]
    for p in raw:
        if not p["bootstrap"] and p["max_samples"] is not None: continue
        valid.append(p)
        if len(valid)==150: break
    return valid

def complexity(name, p):
    if name == "Ridge": return (not p["fit_intercept"], -p["alpha"], p["solver"], p["tol"])
    if name == "SVR": return (p["C"], -p["epsilon"], p["gamma"] if isinstance(p["gamma"],str) else p["gamma"], not p["shrinking"])
    return (p["n_estimators"], 999 if p["max_depth"] is None else p["max_depth"], -p["min_samples_leaf"], -p["min_samples_split"])

def search():
    if not (OUT / "FIXED_CONTROL_REPRODUCTION.csv").exists(): raise RuntimeError("Run preflight first")
    if not pd.read_csv(OUT / "FIXED_CONTROL_REPRODUCTION.csv")["pass"].all(): raise RuntimeError("Fixed control did not pass")
    trials=[]; winners={}
    for endpoint in ENDPOINTS:
        # Development targets only, loaded from caches; no held-out CSV is opened.
        target_sd=float(np.std(np.concatenate([np.load(CACHE/f"{endpoint}__Ridge__fold{i}.npz")["y_validation"] for i in range(1,6)]),ddof=1))
        winners[endpoint]={}
        for name in MODELS:
            candidates=candidate_sets(endpoint,name,target_sd)
            valid=[]
            for trial_id,p in enumerate(candidates,1):
                foldm=[]; error=""
                try:
                    for fold in range(1,6):
                        z=np.load(CACHE/f"{endpoint}__{name}__fold{fold}.npz")
                        m=model(name,p); m.fit(z["X_train"],z["y_train"])
                        foldm.append(metrics(z["y_validation"],m.predict(z["X_validation"])))
                except Exception as e:
                    error=f"{type(e).__name__}: {e}"
                row={"endpoint":endpoint,"model":name,"trial_id":trial_id,"valid":not bool(error),"error":error,
                     "parameters_json":json.dumps(p,sort_keys=True,separators=(",",":"),default=float)}
                if not error:
                    for metric_name in ("r2","rmse","mae","spearman_rho"):
                        vals=np.array([x[metric_name] for x in foldm]); row[f"mean_{metric_name}"]=vals.mean(); row[f"sd_{metric_name}"]=vals.std(ddof=1)
                    row["complexity_json"]=json.dumps(complexity(name,p),default=str)
                    valid.append((row,p,foldm))
                trials.append(row)
            if len(valid) < ({"Ridge":1,"SVR":200,"RandomForest":150}[name]): raise RuntimeError(f"Insufficient valid candidates: {endpoint} {name} {len(valid)}")
            best=sorted(valid,key=lambda x:(-x[0]["mean_r2"],x[0]["mean_rmse"],complexity(name,x[1])))[0]
            winners[endpoint][name]={"parameters":best[1],"selection":{"trial_id":best[0]["trial_id"],"mean_cv_r2":best[0]["mean_r2"],"mean_cv_rmse":best[0]["mean_rmse"],"mean_cv_mae":best[0]["mean_mae"]},"valid_candidates":len(valid)}
    pd.DataFrame(trials).to_csv(OUT/"ALL_HYPERPARAMETER_TRIALS.csv",index=False)
    payload={"lock_version":1,"created_utc":pd.Timestamp.utcnow().isoformat(),"search_seed":SEARCH_SEED,
             "selection_rule":"highest mean 5-fold CV R2; lowest mean CV RMSE; lowest complexity","heldout_read_during_search":False,
             "sklearn_version":sklearn_version,"locked":winners}
    lock=OUT/"LOCKED_HYPERPARAMETERS.json"; lock.write_text(json.dumps(payload,indent=2,sort_keys=True,default=float)+"\n",encoding="utf-8")
    digest=sha256(lock); (OUT/"LOCKED_HYPERPARAMETERS.sha256").write_text(f"{digest}  LOCKED_HYPERPARAMETERS.json\n",encoding="ascii")
    protocol=f"""# Search protocol\n\nDevelopment-only deterministic scikit-learn search; search seed `{SEARCH_SEED}`. No Optuna dependency was used. Hyperparameters were selected by highest mean five-fold CV R2, then lowest mean CV RMSE, then lowest model complexity. Cached fold matrices reproduce the corrected minimal baseline pipeline. Ridge used an exhaustive 33-point log alpha grid crossed with intercept, four solvers, and five tolerances (1,320 candidates per endpoint). SVR used 220 deterministic ParameterSampler candidates per endpoint (180 numeric-gamma and 40 scale/auto). Random Forest used 150 valid deterministic candidates per endpoint after excluding `bootstrap=False` with non-null `max_samples`. Random Forest estimator seed was `{SEED}` and `n_jobs=1` was used for deterministic, single-process execution.\n\nThe `search` phase reads only cached development-fold matrices. It does not open either held-out CSV. The lock file is written and SHA-256 hashed before the separate `heldout` phase may open held-out outcomes. This is a finite development-CV-tuned comparator stress test, not a claim of the best possible model.\n"""
    (OUT/"SEARCH_PROTOCOL.md").write_text(protocol,encoding="utf-8")
    print("SEARCH PASS: all six hyperparameters locked and hashed; held-out CSVs were not read")

def validate_inputs():
    manifest=pd.read_csv(ROOT/'search/FOLD_CACHE_MANIFEST.csv')
    if len(manifest)!=30:raise ValueError('Expected 30 cached model-fold matrices.')
    for row in manifest.to_dict('records'):
        p=ROOT/row['cache_file']
        if not p.is_file() or sha256(p)!=row['sha256']:raise ValueError('Cache checksum mismatch: '+str(p.name))
        with np.load(p,allow_pickle=False) as z:
            for k in ['X_train','y_train','X_validation','y_validation','validation_indices']:
                if k not in z:raise ValueError('Cache is missing '+k)
                if not np.isfinite(z[k]).all():raise ValueError('Nonfinite cache values')
            if z['X_train'].shape[1]!=4:raise ValueError('Unexpected assay dimension')
    lock=ROOT/'settings/LOCKED_HYPERPARAMETERS.json'
    if sha256(lock)!=(ROOT/'settings/LOCKED_HYPERPARAMETERS.sha256').read_text().split()[0]:
        raise ValueError('Lock checksum mismatch')
    for cfg in ENDPOINTS.values():
        for k in ('train','test','annotated_train','annotated_test'):
            if not cfg[k].is_file():raise FileNotFoundError(cfg[k])
    return json.loads(lock.read_text())

def locked_cv(lock_payload):
    archived=pd.read_csv(ROOT/'results/OPTIMIZED_COMPARATOR_CV_METRICS.csv')
    archived=archived[pd.to_numeric(archived['fold'],errors='coerce').notna()].copy()
    records=[]
    for endpoint in ENDPOINTS:
        for name in MODELS:
            params=lock_payload['locked'][endpoint][name]['parameters']
            for fold in range(1,6):
                with np.load(CACHE/f'{endpoint}__{name}__fold{fold}.npz',allow_pickle=False) as z:
                    m=model(name,params);m.fit(z['X_train'],z['y_train'])
                    mm=metrics(z['y_validation'],m.predict(z['X_validation']))
                prior=archived[(archived.endpoint==endpoint)&(archived.model==name)&(archived.fold.astype(str)==str(fold))]
                expected=float(prior.iloc[0]['r2'])
                records.append({'endpoint':endpoint,'model':name,'fold':fold,**mm,
                                'archived_r2':expected,'r2_difference':mm['r2']-expected,
                                'sklearn_version':sklearn_version})
    pd.DataFrame(records).to_csv(OUT/'LOCKED_CV_REFIT.csv',index=False)
    print('30 locked comparator fold fits completed. Maximum |R2 difference|:',
          max(abs(x['r2_difference']) for x in records))
    print('Refit environment:',sklearn_version,'; archived environment:',lock_payload['sklearn_version'])

def preflight():
    global CACHE
    CACHE=OUT/'fold_cache'
    expected=pd.read_csv(ROOT/'results/FIXED_CONTROL_REPRODUCTION.csv')
    records=[]; manifests=[]
    for endpoint in ENDPOINTS:
        x,m=cache_fixed_and_reproduce(endpoint);records+=x;manifests+=m
    observed=pd.DataFrame(records);checks=[]
    for r in expected.to_dict('records'):
        g=observed[(observed.endpoint==r['endpoint'])&(observed.model==r['model'])]
        cv=float(g[g.scope=='cv_fold'].r2.mean());test=float(g[g.scope=='held_out'].r2.iloc[0])
        dc=abs(cv-float(r['expected_cv_r2_mean']));dt=abs(test-float(r['expected_heldout_r2']))
        checks.append({**r,'reproduced_cv_r2_mean':cv,'reproduced_heldout_r2':test,
                       'cv_r2_mean_abs_diff':dc,'heldout_r2_abs_diff':dt,'pass':dc<=5e-4 and dt<=5e-4})
    pd.DataFrame(checks).to_csv(OUT/'FIXED_CONTROL_REPRODUCTION.csv',index=False)
    pd.DataFrame(manifests).to_csv(OUT/'FOLD_CACHE_MANIFEST.csv',index=False)
    if not all(r['pass'] for r in checks):raise RuntimeError('Fixed-control drift; see output CSV. Archived results remain unchanged.')
    print('Fixed control and newly generated caches reproduced.')

def heldout(lock_dir):
    lock_path=lock_dir/'LOCKED_HYPERPARAMETERS.json'
    if not lock_path.is_file():raise FileNotFoundError(lock_path)
    if sha256(lock_path)!=(lock_dir/'LOCKED_HYPERPARAMETERS.sha256').read_text().split()[0]:
        raise ValueError('Lock checksum mismatch')
    locked=json.loads(lock_path.read_text())['locked'];cv=[];held=[];predrows=[]
    for endpoint,cfg in ENDPOINTS.items():
        _,test,_,_,Xtest,ytest,_=prepare_endpoint(endpoint,True)
        ann=pd.read_csv(cfg['annotated_test'])
        target=test.columns[-1]
        np.testing.assert_allclose(ann[target].to_numpy(float),ytest,rtol=0,atol=1e-10)
        ids=ann['mAb_id'].astype(str).tolist()
        for name in MODELS:
            fit=[]
            for fold in range(1,6):
                with np.load(CACHE/f'{endpoint}__{name}__fold{fold}.npz',allow_pickle=False) as z:
                    m=model(name,locked[endpoint][name]['parameters']);m.fit(z['X_train'],z['y_train']);fit.append(m)
                    cv.append({'endpoint':endpoint,'model':name,'fold':fold,**metrics(z['y_validation'],m.predict(z['X_validation']))})
            pred=np.mean([m.predict(Xtest) for m in fit],axis=0);mm=metrics(ytest,pred)
            mm.update(nrmse=mm['rmse']/np.mean(ytest) if endpoint=='mouse_exposure' else np.nan,
                      nmae=mm['mae']/np.mean(ytest) if endpoint=='mouse_exposure' else np.nan)
            held.append({'endpoint':endpoint,'model':name,'n':len(ytest),**mm})
            for i,(ident,y,p) in enumerate(zip(ids,ytest,pred),1):
                predrows.append({'endpoint':endpoint,'model':name,'row':i,'antibody_id':ident,'observed':y,'predicted':p,'residual':y-p})
    pd.DataFrame(cv).to_csv(OUT/'OPTIMIZED_COMPARATOR_CV_METRICS.csv',index=False)
    pd.DataFrame(held).to_csv(OUT/'OPTIMIZED_COMPARATOR_HELDOUT_METRICS.csv',index=False)
    pd.DataFrame(predrows).to_csv(OUT/'OPTIMIZED_COMPARATOR_HELDOUT_PREDICTIONS.csv',index=False)
    print('Locked held-out evaluation completed in',OUT)

def main():
    global OUT,CACHE
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('phase',choices=['verify','cv','preflight','search','heldout'])
    ap.add_argument('--output-dir',type=Path)
    ap.add_argument('--cache-dir',type=Path,help='Optional regenerated cache directory; default is the packaged cache.')
    ap.add_argument('--lock-dir',type=Path,help='Optional new search lock directory; default is the archived settings.')
    args=ap.parse_args();lock_payload=validate_inputs()
    if args.phase=='verify':
        print('All 30 cached folds, endpoint file paths and archived lock verified. No models fitted.');return
    if not args.output_dir:raise SystemExit('Supply --output-dir; archived results are read-only.')
    OUT=args.output_dir.resolve()
    for protected in [ROOT/'data',ROOT/'settings',ROOT/'results',ROOT/'search',ROOT/'fold_cache']:
        if OUT==protected or protected in OUT.parents:raise SystemExit('Output directory overlaps archived evidence.')
    if OUT.exists() and any(OUT.iterdir()):raise SystemExit('Choose an empty --output-dir for this command.')
    OUT.mkdir(parents=True,exist_ok=True)
    if args.cache_dir:CACHE=args.cache_dir.resolve()
    if args.phase in ('search','preflight','heldout') and sklearn_version!='1.4.1.post1':
        raise SystemExit('Use study scikit-learn 1.4.1.post1 for a new search/preflight/heldout rerun. The cv and verify phases support the current environment.')
    if args.phase in ('preflight','heldout'):
        import importlib.util
        needed=['sdv']+(['ImbalancedLearningRegression'] if args.phase=='preflight' else [])
        absent=[x for x in needed if importlib.util.find_spec(x) is None]
        if absent:raise SystemExit('Study environment required; missing: '+', '.join(absent))
    if args.phase=='cv':locked_cv(lock_payload)
    elif args.phase=='preflight':preflight()
    elif args.phase=='search':
        shutil.copy2(ROOT/'results/FIXED_CONTROL_REPRODUCTION.csv',OUT/'FIXED_CONTROL_REPRODUCTION.csv')
        search()
    else:heldout(args.lock_dir.resolve() if args.lock_dir else ROOT/'settings')
    (OUT/'execution_environment.json').write_text(json.dumps({'python':sys.version,'sklearn':sklearn_version,'numpy':np.__version__,'phase':args.phase},indent=2)+'\n')
if __name__=='__main__':main()
