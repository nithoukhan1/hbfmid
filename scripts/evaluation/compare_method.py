#!/usr/bin/env python3
"""Gate evaluation for a method arm on the v2 clean folds, paired with the plain baseline (same fold, same seed).
Baseline dirs: YOLO11s results/v2_A + results/v2s_A; YOLOv8s results/v2_B + results/v2s_B. Method dirs: results/<arm>_A (YOLOv8s), results/<arm>_B (YOLO11s) for notebook 12-style layout.
Gate (docs/METHOD_DESIGN.md): mean paired gain >= +0.9 pt mAP50-95 over folds (seed-averaged), >= 4/5 folds positive; reported with the t-interval (4 df).
Usage: python scripts/evaluation/compare_method.py --arm gb --out results/analysis/method_gb"""
import argparse, glob, json
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('--arm', required=True); ap.add_argument('--out', required=True); ap.add_argument('--gate', type=float, default=0.009); a = ap.parse_args()
cfg = {'yolov8s.pt': (['results/v2_B', 'results/v2s_B'], [f'results/{a.arm}_A']), 'yolo11s.pt': (['results/v2_A', 'results/v2s_A'], [f'results/{a.arm}_B'])}
def load(dirs, model, arm):
    R = {}
    for d in dirs:
        for p in glob.glob(f'{d}/*/metrics.json'):
            r = json.load(open(p))
            if r.get('mode') == 'clean' and r.get('protocol') == 'fixed_last' and r['model'] == model and r.get('arm', 'baseline') == arm:
                R[(r['fold'], r['seed'])] = r['test']['map5095']
    return R
out = {}; lines = ['| model | seeds | base mean | method mean | gain (pts) +/- t95 | folds positive | gate |', '|---|---|---|---|---|---|---|']
for model, (bd, md) in cfg.items():
    B = load(bd, model, 'baseline'); Mth = load(md, model, a.arm)
    seeds = sorted({s for f, s in Mth if all((ff, s) in Mth and (ff, s) in B for ff in range(5))})
    if not seeds: lines.append(f'| {model} | none complete | | | | | |'); continue
    b = np.array([np.mean([B[(f, s)] for s in seeds]) for f in range(5)]); m = np.array([np.mean([Mth[(f, s)] for s in seeds]) for f in range(5)])
    d = m - b; half = 2.776 * d.std(ddof=1) / np.sqrt(5); passed = bool(d.mean() >= a.gate and (d > 0).sum() >= 4)
    per_seed = {int(s): float(np.mean([Mth[(f, s)] - B[(f, s)] for f in range(5)])) for s in seeds}
    out[model] = {'seeds': seeds, 'base': float(b.mean()), 'method': float(m.mean()), 'gain': float(d.mean()), 'half_ci': float(half), 'folds_positive': int((d > 0).sum()),
                  'per_fold_gain': d.tolist(), 'per_seed_gain': per_seed, 'gate_passed': passed}
    lines.append(f"| {model} | {seeds} | {b.mean():.4f} | {m.mean():.4f} | {100 * d.mean():+.2f} +/- {100 * half:.2f} | {(d > 0).sum()}/5 | {'PASS' if passed else 'FAIL'} |")
    lines.append(f"|  | per-seed gains (pts) | {', '.join(f'{s}: {100 * g:+.2f}' for s, g in per_seed.items())} | | | | |")
Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out + '.json').write_text(json.dumps(out, indent=1)); Path(a.out + '.md').write_text('\n'.join(lines) + '\n'); print('\n'.join(lines))
