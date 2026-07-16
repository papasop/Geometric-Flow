#!/usr/bin/env python3
from __future__ import annotations
import argparse, importlib.util, json, platform, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXPERIMENT=ROOT/'experiments/h1314f_fix_gpt2_full12_long_horizon.py'
PROFILES={
 'smoke':dict(trials=1,steps=20,train_samples=64,val_samples=32,batch_size=1,eval_batch_size=1,probe_steps='0,5,10,19',output_dir='external_validation/h1314_smoke'),
 'formal':dict(trials=6,steps=200,train_samples=512,val_samples=128,batch_size=4,eval_batch_size=4,probe_steps='0,10,25,50,100,150,199',output_dir='external_validation/h1314_formal_6seed'),
 'long':dict(trials=3,steps=1000,train_samples=512,val_samples=128,batch_size=4,eval_batch_size=4,probe_steps='0,25,50,100,200,400,600,800,999',output_dir='external_validation/h1314_long_1000step')}
REQ=('torch','transformers','datasets','huggingface_hub','numpy')
def missing(): return [x for x in REQ if importlib.util.find_spec(x) is None]
def install(): subprocess.run([sys.executable,'-m','pip','install','-U','numpy>=1.23','torch>=2.0','transformers>=4.40','datasets>=2.18','huggingface_hub>=0.23','matplotlib>=3.7'],check=True)
def cli(d):
 out=[]
 for k,v in d.items(): out += ['--'+k.replace('_','-'),str(v)]
 return out
def main():
 p=argparse.ArgumentParser(description='One-command external H13.14 GPT-2 validator')
 p.add_argument('--mode',choices=sorted(PROFILES),default='smoke'); p.add_argument('--install-deps',action='store_true'); p.add_argument('--device'); p.add_argument('--output-dir'); p.add_argument('--no-plots',action='store_true')
 a,extra=p.parse_known_args()
 if missing() and a.install_deps: install()
 if missing(): raise SystemExit('missing packages: '+', '.join(missing())+'; rerun with --install-deps')
 cfg=dict(PROFILES[a.mode])
 if a.device: cfg['device']=a.device
 if a.output_dir: cfg['output_dir']=a.output_dir
 cmd=[sys.executable,str(EXPERIMENT),*cli(cfg)]
 if a.no_plots: cmd.append('--no-plots')
 cmd += extra
 out=ROOT/cfg['output_dir']; out.mkdir(parents=True,exist_ok=True)
 (out/'external_run_manifest.json').write_text(json.dumps({'mode':a.mode,'command':cmd,'python':platform.python_version()},indent=2),encoding='utf-8')
 print('[external-validator]', ' '.join(cmd)); return subprocess.run(cmd,cwd=ROOT).returncode
if __name__=='__main__': raise SystemExit(main())
