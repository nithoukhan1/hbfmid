#!/usr/bin/env python3
"""Gate evaluation for a method arm on the v2 clean folds, paired with the plain baseline (same fold, same seed).
Baseline dirs: YOLO11s results/v2_A + results/v2s_A; YOLOv8s results/v2_B + results/v2s_B.
Method dirs: results/<arm>_A and results/<arm>_B; each run is matched to its detector by metrics.json 'model', so the A/B letter does not matter.
Gate (docs/METHOD_DESIGN.md section 3b, fixed 2026-11-23 from the v2 noise floor before any method run): mean paired gain >= +1.8 pt mAP50-95
(seed-averaged, paired by fold) AND >= 4/5 folds positive, in the YOLOv8s part (primary); YOLO11s = replication (mean gain > 0 to claim generality).
The gate is read only when seeds 42, 43, 44 are complete. Also reported: t-interval (4 df), per-seed gains, descriptive pooled gain over both detectors.
Usage: python scripts/evaluation/compare_method.py --arm gb --out results/analysis/method_gb"""
import argparse, glob, json
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('--arm', required=True); ap.add_argument('--out', required=True); ap.add_argument('--gate', type=float, default=0.018); a = ap.parse_args()
SEEDS = [42, 43, 44]
mdirs = [f'results/{a.arm}_A', f'results/{a.arm}_B']
cfg = {'yolov8s.pt': ('primary', ['results/v2_B', 'results/v2s_B']), 'yolo11s.pt': ('replication', ['results/v2_A', 'results/v2s_A'])}
def load(dirs, model, arm):
    R = {}
    for d in dirs:
        for p in glob.glob(f'{d}/*/metrics.json'):
            r = json.load(open(p))
            if r.get('mode') == 'clean' and r.get('protocol') == 'fixed_last' and r['model'] == model and r.get('arm', 'baseline') == arm:
                R[(r['fold'], r['seed'])] = r['test']['map5095']
    return R
out = {'gate_pts': 100 * a.gate, 'arm': a.arm}; gains = {}
lines = [f'Gate: mean paired gain >= +{100 * a.gate:.1f} pt mAP50-95 AND >= 4/5 folds positive, primary = YOLOv8s; YOLO11s = replication (gain > 0)', '',
         '| model | role | seeds | base mean | method mean | gain (pts) +/- t95 | folds positive | result |', '|---|---|---|---|---|---|---|---|']
for model, (role, bd) in cfg.items():
    B = load(bd, model, 'baseline'); Mth = load(mdirs, model, a.arm)
    seeds = sorted({s for f, s in Mth if all((ff, s) in Mth and (ff, s) in B for ff in range(5))})
    if not seeds: lines.append(f'| {model} | {role} | none complete | | | | | |'); continue
    b = np.array([np.mean([B[(f, s)] for s in seeds]) for f in range(5)]); m = np.array([np.mean([Mth[(f, s)] for s in seeds]) for f in range(5)])
    d = m - b; half = 2.776 * d.std(ddof=1) / np.sqrt(5); complete = seeds == SEEDS
    if not complete: result = f'INCOMPLETE (seeds {seeds}; gate needs {SEEDS})'
    elif role == 'primary': result = 'PASS' if (d.mean() >= a.gate and (d > 0).sum() >= 4) else 'FAIL'
    else: result = 'same direction' if d.mean() > 0 else 'not same direction'
    per_seed = {int(s): float(np.mean([Mth[(f, s)] - B[(f, s)] for f in range(5)])) for s in seeds}
    out[model] = {'role': role, 'seeds': seeds, 'complete': complete, 'base': float(b.mean()), 'method': float(m.mean()), 'gain': float(d.mean()), 'half_ci': float(half),
                  'folds_positive': int((d > 0).sum()), 'per_fold_gain': d.tolist(), 'per_seed_gain': per_seed, 'result': result}
    gains[model] = d
    lines.append(f"| {model} | {role} | {seeds} | {b.mean():.4f} | {m.mean():.4f} | {100 * d.mean():+.2f} +/- {100 * half:.2f} | {(d > 0).sum()}/5 | {result} |")
    lines.append(f"|  |  | per-seed gains (pts) | {', '.join(f'{s}: {100 * g:+.2f}' for s, g in per_seed.items())} | | | | |")
if len(gains) == 2:
    pooled = np.concatenate(list(gains.values())); out['pooled_descriptive'] = {'gain': float(pooled.mean()), 'detector_folds_positive': int((pooled > 0).sum())}
    lines.append(f"\nPooled over both detectors (descriptive, no decision weight): {100 * pooled.mean():+.2f} pt, {(pooled > 0).sum()}/10 detector-folds positive")
Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out + '.json').write_text(json.dumps(out, indent=1)); Path(a.out + '.md').write_text('\n'.join(lines) + '\n'); print('\n'.join(lines))
