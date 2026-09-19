#!/usr/bin/env python3
"""Refresh release integrity indexes after an intentional source/data/documentation change."""
from __future__ import annotations
import argparse, csv, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
IGNORED_PARTS={'.git','__pycache__','.venv','.venv-analysis','.venv-training','reruns','ReproducedOutputs','demo_output','rerun_outputs','REPRODUCED_OUTPUTS','.pytest_cache'}
IGNORED_NAMES={'MANIFEST.csv','CHECKSUMS.sha256'}
def inventory(root=ROOT):
    result=[]
    for p in sorted(root.rglob('*')):
        rel=p.relative_to(root)
        if not p.is_file() or any(part in IGNORED_PARTS or part.startswith('.venv') for part in rel.parts) or p.name in IGNORED_NAMES or p.suffix in {'.pyc','.pyo'}: continue
        result.append({'relative_path':rel.as_posix(),'size_bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--check',action='store_true',help='Check that indexes already match, without writing.');a=ap.parse_args()
    rows=inventory()
    if a.check:
        with (ROOT/'MANIFEST.csv').open(encoding='utf-8-sig',newline='') as f:old=list(csv.DictReader(f))
        for row in old: row['size_bytes']=int(row['size_bytes'])
        if old!=rows:raise SystemExit('Release contents differ from MANIFEST.csv. Review changes before refreshing.')
        expected=''.join(f"{r['sha256']}  {r['relative_path']}\n" for r in rows)
        if (ROOT/'CHECKSUMS.sha256').read_text()!=expected:raise SystemExit('Checksum index differs.')
        print(f'Integrity indexes match {len(rows)} files.');return
    with (ROOT/'MANIFEST.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['relative_path','size_bytes','sha256']);w.writeheader();w.writerows(rows)
    (ROOT/'CHECKSUMS.sha256').write_text(''.join(f"{r['sha256']}  {r['relative_path']}\n" for r in rows),encoding='ascii')
    print(f'Updated MANIFEST.csv and CHECKSUMS.sha256 for {len(rows)} files.')
if __name__=='__main__':main()
