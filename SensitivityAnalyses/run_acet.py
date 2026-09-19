"""Run the recorded endpoint-specific ACeT sensitivity configuration from this package."""
from pathlib import Path
import argparse, importlib.util, importlib.metadata, os, subprocess, sys, csv
ROOT=Path(__file__).resolve().parent
CONFIG={
 'viscosity':('viscosity','panel_A_D_parity_featureimp.py','antibodies_train.csv','antibodies_test.csv','kan',52,23),
 'mouse_exposure':('mouse_exposure','panel_A_B_parity_training_only_sensitivity.py','train42_model_ready_selected4.csv','test11_raw_untouched_selected4.csv','mlp',42,11),
}
def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('endpoint',choices=CONFIG)
 ap.add_argument('--output-dir',type=Path)
 ap.add_argument('--check-inputs',action='store_true')
 a=ap.parse_args(); endpoint,script,train_name,test_name,head,ntr,nte=CONFIG[a.endpoint]
 base=ROOT/'DevelopmentOnlyFeatureSelection'/endpoint
 tr=base/'data'/train_name;te=base/'data'/test_name;src=base/'code'/script
 headers=[]
 for path,n in [(tr,ntr),(te,nte)]:
  with path.open(encoding='utf-8-sig',newline='') as f:
   r=csv.DictReader(f);rows=list(r);headers.append(r.fieldnames)
  if len(rows)!=n:raise ValueError(f'Expected {n} rows in {path.name}')
  for row in rows:
   for val in row.values():
    import math
    if not math.isfinite(float(val)):raise ValueError('Nonfinite model input')
 if headers[0]!=headers[1] or len(headers[0])!=5:raise ValueError('Train/test column mismatch')
 print(f'{a.endpoint}: numeric {ntr}/{nte} inputs verified; recorded head={head}.')
 if a.check_inputs:return
 deps=['tensorflow','tfkan','sdv','ImbalancedLearningRegression']+(['shap'] if a.endpoint=='viscosity' else [])
 missing=[m for m in deps if importlib.util.find_spec(m) is None]
 if missing:raise SystemExit('Training requires the study environment. Missing: '+', '.join(missing)+'. See ENVIRONMENT.md.')
 if importlib.metadata.version('scikit-learn')!='1.4.1.post1':
  raise SystemExit('The unchanged training sources require study scikit-learn 1.4.1.post1; use that environment for retraining.')
 if a.output_dir is None:raise SystemExit('Supply --output-dir to keep new model outputs separate from archived results.')
 out=a.output_dir.resolve()
 if out.exists() and any(out.iterdir()):raise SystemExit('Choose an empty --output-dir.')
 out.mkdir(parents=True,exist_ok=True)
 cmd=[sys.executable,str(src),'--task','regression','--train_file',str(tr),'--test_file',str(te),'--head_type',head]
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='0',MPLBACKEND='Agg')
 with (out/'run.log').open('w',encoding='utf-8') as f:
  result=subprocess.run(cmd,cwd=out,env=env,stdout=f,stderr=subprocess.STDOUT)
 if result.returncode:raise SystemExit(f'Training exited with code {result.returncode}; details: {out / "run.log"}')
 print('Training completed; outputs:',out)
if __name__=='__main__':main()
