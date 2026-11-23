#!/usr/bin/env python3
"""3-seed benchmark on the v2 clean folds for YOLO11s and YOLOv8s (seeds 42 from results/v2_A, v2_B; seeds 43/44 from results/v2s_A, v2s_B).
Per detector: per-fold mean over seeds, mean and sd over folds, seed sd of the 5-fold mean. Gap YOLO11s - YOLOv8s paired by fold (seed-averaged), t=2.776 (4 df).
Also prints the noise floor: sd of the 5-fold mean across seeds. Usage: python scripts/evaluation/compare_v2_seeds.py --out results/analysis/v2_three_seeds"""
import argparse, glob, json
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('--out', default='results/analysis/v2_three_seeds'); a = ap.parse_args()
dirs = {'yolo11s.pt': ['results/v2_A', 'results/v2s_A'], 'yolov8s.pt': ['results/v2_B', 'results/v2s_B']}
M = {'map5095': 'map5095', 'macro': 'macro_eligible_ap5095', 'map50': 'map50'}
R = {}
for model, ds in dirs.items():
    for d in ds:
        for p in glob.glob(f'{d}/*/metrics.json'):
            r = json.load(open(p))
            if r.get('mode') == 'clean' and r.get('protocol') == 'fixed_last' and r.get('arm', 'baseline') == 'baseline' and r['model'] == model:
                R.setdefault(model, {})[(r['fold'], r['seed'])] = r['test']
out = {}; lines = ['| model | seeds | runs | mAP50-95 mean (sd over folds) | seed sd of 5-fold mean | macro | mAP50 |', '|---|---|---|---|---|---|---|']
pf = {}
for model, X in R.items():
    seeds = sorted({s for _, s in X}); full = [s for s in seeds if all((f, s) in X for f in range(5))]
    if not full: continue
    per_fold = [np.mean([X[(f, s)]['map5095'] for s in full]) for f in range(5)]; pf[model] = np.array(per_fold)
    seed_means = [np.mean([X[(f, s)]['map5095'] for f in range(5)]) for s in full]
    o = {'seeds': full, 'n_runs': len(full) * 5, 'per_fold': per_fold, 'mean': float(np.mean(per_fold)), 'sd_over_folds': float(np.std(per_fold, ddof=1)),
         'seed_means': seed_means, 'seed_sd': float(np.std(seed_means, ddof=1)) if len(full) > 1 else None}
    for m, k in M.items():
        o[m] = float(np.mean([np.mean([X[(f, s)][k] for s in full]) for f in range(5)]))
    out[model] = o
    lines.append(f"| {model} | {full} | {len(full) * 5} | {o['mean']:.4f} ({o['sd_over_folds']:.4f}) | {o['seed_sd'] if o['seed_sd'] is None else round(o['seed_sd'], 4)} | {o['macro']:.4f} | {o['map50']:.4f} |")
if len(pf) == 2:
    d = pf['yolo11s.pt'] - pf['yolov8s.pt']; half = 2.776 * d.std(ddof=1) / np.sqrt(5)
    out['gap_yolo11s_minus_yolov8s'] = {'mean': float(d.mean()), 'half_ci': float(half), 'folds_positive': int((d > 0).sum()), 'per_fold': d.tolist()}
    lines.append(f"\nGap YOLO11s - YOLOv8s (seed-averaged, paired by fold): {d.mean():+.4f} +/- {half:.4f}; folds positive {(d > 0).sum()}/5")
Path(a.out + '.json').write_text(json.dumps(out, indent=1)); Path(a.out + '.md').write_text('\n'.join(lines) + '\n'); print('\n'.join(lines))
