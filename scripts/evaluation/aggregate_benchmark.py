#!/usr/bin/env python3
"""Benchmark table under the fixed-epoch protocol: per detector, official split vs HBFMID-Clean.

Official: mean and sd over seeds of the single 64-image test set (mAP50-95, macro over the 8 eligible classes, mAP50).
Clean   : per fold mean over seeds, then mean and sd over the 5 folds (sd = spread between folds; seed sd of the fold mean in brackets when >1 seed).
Only plain runs (arm 'baseline', protocol fixed_last) are used.
Usage: python scripts/evaluation/aggregate_benchmark.py --runs results/controls results/baseline_fixed results/bench_yolo11s --out results/benchmark/benchmark"""
import argparse, glob, json
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--runs', nargs='+', required=True)
ap.add_argument('--out', required=True)
a = ap.parse_args()
R = []
for d in a.runs:
    for p in glob.glob(str(Path(d) / '*/metrics.json')):
        r = json.loads(Path(p).read_text())
        if r.get('protocol') == 'fixed_last' and r.get('arm', 'baseline') == 'baseline' and r['mode'] in ('official', 'clean'):
            R.append(r)
M = {'map5095': lambda r: r['test']['map5095'], 'macro': lambda r: r['test']['macro_eligible_ap5095'], 'map50': lambda r: r['test']['map50']}
sd = lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else None
out = {}
for model in sorted({r['model'] for r in R}):
    o = {}
    off = [r for r in R if r['model'] == model and r['mode'] == 'official']
    if off:
        o['official'] = {'seeds': sorted(r['seed'] for r in off), **{m: {'mean': float(np.mean([fn(r) for r in off])), 'sd': sd([fn(r) for r in off])} for m, fn in M.items()}}
    cl = [r for r in R if r['model'] == model and r['mode'] == 'clean']
    if cl:
        folds = sorted({r['fold'] for r in cl})
        c = {'folds': folds, 'seeds': sorted({r['seed'] for r in cl}), 'n_runs': len(cl)}
        for m, fn in M.items():
            pf = [float(np.mean([fn(r) for r in cl if r['fold'] == f])) for f in folds]
            c[m] = {'mean': float(np.mean(pf)), 'sd_over_folds': sd(pf)}
        full = [s for s in c['seeds'] if all(any(r['fold'] == f and r['seed'] == s for r in cl) for f in folds)]
        if len(full) > 1:
            c['seed_sd_of_fold_mean'] = sd([float(np.mean([M['map5095'](r) for r in cl if r['seed'] == s])) for s in full])
        o['clean'] = c
    out[model] = o
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(str(a.out) + '.json').write_text(json.dumps(out, indent=1))
f = lambda v, k='mean': 'n/a' if v is None else f'{v[k]:.4f}'
md = ['# Benchmark (fixed 100 epochs, last.pt)', '', '| detector | official: mAP50-95 (seeds) | official macro | clean: mAP50-95 | clean macro | clean mAP50 | clean runs |', '|---|---|---|---|---|---|---|']
for model, o in out.items():
    of, cl = o.get('official'), o.get('clean')
    md.append(f"| {model} | {f(of['map5095']) + ' ' + str(of['seeds']) if of else 'n/a'} | {f(of['macro']) if of else 'n/a'} | "
              f"{f(cl['map5095']) + ' +/- ' + f(cl['map5095'], 'sd_over_folds') if cl else 'n/a'} | {f(cl['macro']) if cl else 'n/a'} | {f(cl['map50']) if cl else 'n/a'} | {cl['n_runs'] if cl else 0} |")
Path(str(a.out) + '.md').write_text('\n'.join(md) + '\n'); print('\n'.join(md))
