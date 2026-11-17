#!/usr/bin/env python3
"""Summarise one experimental arm under the fixed-epoch protocol: per-fold mean over seeds, 5-fold mean, seed spread.
Usage: python scripts/evaluation/aggregate_arm.py --name arm1_baseline --runs results/controls results/baseline_fixed --out results/arms"""
import argparse, json, glob
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument('--name', required=True); ap.add_argument('--runs', nargs='+', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--model', default='yolov8s.pt')
a = ap.parse_args()
R = []
for d in a.runs:
    R += [json.loads(Path(p).read_text()) for p in glob.glob(str(Path(d) / '*/metrics.json'))]
R = [r for r in R if r['mode'] == 'clean' and r.get('protocol') == 'fixed_last' and r['model'] == a.model]
folds = sorted({r['fold'] for r in R}); seeds = sorted({r['seed'] for r in R})
get = {'map5095': lambda r: r['test']['map5095'], 'macro': lambda r: r['test']['macro_eligible_ap5095'], 'map50': lambda r: r['test']['map50']}
res = {'arm': a.name, 'folds': folds, 'seeds': seeds, 'n_runs': len(R), 'per_fold': {}, 'complete': True}
for f in folds:
    g = [r for r in R if r['fold'] == f]
    res['per_fold'][f] = {m: {'mean': float(np.mean([fn(r) for r in g])), 'n_seeds': len(g)} for m, fn in get.items()}
for m in get:
    pf = [res['per_fold'][f][m]['mean'] for f in folds]
    res[m] = {'mean_over_folds': float(np.mean(pf)), 'sd_over_folds': float(np.std(pf, ddof=1)) if len(pf) > 1 else None}
# per-seed 5-fold means (only seeds present in every fold)
full = [s for s in seeds if all(any(r['fold'] == f and r['seed'] == s for r in R) for f in folds)]
res['seeds_complete_across_folds'] = full
if len(full) > 1:
    sm = [float(np.mean([get['map5095'](r) for r in R if r['seed'] == s])) for s in full]
    res['seed_means_map5095'] = sm; res['seed_sd_of_fold_mean'] = float(np.std(sm, ddof=1))
Path(a.out).mkdir(parents=True, exist_ok=True)
Path(a.out, f'{a.name}.json').write_text(json.dumps(res, indent=1))
md = [f'# {a.name}: {len(R)} runs, folds {folds}, seeds {seeds} (fixed protocol, last.pt)', '', '| fold | seeds | mAP50-95 | macro(>=10 groups) | mAP50 |', '|---|---|---|---|---|']
for f in folds:
    p = res['per_fold'][f]; md.append(f"| {f} | {p['map5095']['n_seeds']} | {p['map5095']['mean']:.4f} | {p['macro']['mean']:.4f} | {p['map50']['mean']:.4f} |")
md.append(f"| **mean** | | {res['map5095']['mean_over_folds']:.4f} | {res['macro']['mean_over_folds']:.4f} | {res['map50']['mean_over_folds']:.4f} |")
if 'seed_sd_of_fold_mean' in res: md.append(f"\nseed sd of the 5-fold mean (mAP50-95): {res['seed_sd_of_fold_mean']:.4f} over seeds {full}")
Path(a.out, f'{a.name}.md').write_text('\n'.join(md) + '\n'); print('\n'.join(md))
