#!/usr/bin/env python3
from pathlib import Path
import ast
r=Path(__file__).resolve().parents[1]
files=[r/'experiments/h1314f_fix_gpt2_full12_long_horizon.py',r/'scripts/run_external_gpt2_validation.py',r/'docs/h1314_gpt2_full_depth_results.md',r/'README.md']
for p in files:
 if not p.exists(): raise SystemExit(f'MISSING: {p.relative_to(r)}')
for p in files[:2]: ast.parse(p.read_text(encoding='utf-8'))
rd=(r/'README.md').read_text(encoding='utf-8')
for x in ('GPT-2 Full-Depth','4.124508','3.892090','production-scale'):
 if x not in rd: raise SystemExit('README missing '+x)
e=files[0].read_text(encoding='utf-8')
for x in ('sigma_condition_mean_mean','per_trial_checkpoint.csv','PASS_COUPLED_GAUGE_EVERY_SEED_STRICT','target_last_n_layers: int = 12'):
 if x not in e: raise SystemExit('experiment missing '+x)
print('H13.14 integration static verification: PASS')
