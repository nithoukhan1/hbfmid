#!/usr/bin/env python3
"""Compare the clean benchmark on grouping v1 folds vs v2 folds (seed 42 only) for YOLO11s and YOLOv8s.
v1 runs: results/bench_A (YOLO11s), results/baseline_fixed (YOLOv8s). v2 runs: results/v2_A, results/v2_B.
Reports per-fold and mean mAP50-95 / macro / mAP50, the v1->v2 shift of the 5-fold mean, and the YOLO11s-minus-YOLOv8s gap per protocol
(the v2 gap is paired by fold because both detectors share the same v2 test sets; the v1->v2 shift is NOT paired: test sets differ).
Usage: python scripts/evaluation/compare_v1_v2.py --out results/analysis/clean_v1_vs_v2"""
import argparse, glob, json
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('--out', default='results/analysis/clean_v1_vs_v2'); a = ap.parse_args()
def load(dirs, model, seed=42):
    R = {}
    for d in dirs:
        for p in glob.glob(f'{d}/*/metrics.json'):
            r = json.load(open(p))
            if r.get('mode') == 'clean' and r.get('protocol') == 'fixed_last' and r.get('arm', 'baseline') == 'baseline' and r['model'] == model and r['seed'] == seed:
                R[r['fold']] = r['test']
    return R
src = {('v1', 'yolo11s.pt'): ['results/bench_A'], ('v1', 'yolov8s.pt'): ['results/baseline_fixed', 'results/controls'],
       ('v2', 'yolo11s.pt'): ['results/v2_A'], ('v2', 'yolov8s.pt'): ['results/v2_B']}
D = {k: load(v, k[1]) for k, v in src.items()}
M = {'map5095': 'map5095', 'macro': 'macro_eligible_ap5095', 'map50': 'map50'}
tcrit = 2.776
out = {}; lines = ['| protocol | model | folds | mAP50-95 mean (sd over folds) | macro (8 classes) | mAP50 |', '|---|---|---|---|---|---|']
for (proto, model), R in D.items():
    if len(R) < 5: lines.append(f'| {proto} | {model} | {len(R)}/5 | incomplete | | |'); continue
    row = {}
    for m, k in M.items():
        x = [R[f][k] for f in range(5)]; row[m] = {'per_fold': x, 'mean': float(np.mean(x)), 'sd': float(np.std(x, ddof=1))}
    out[f'{proto}:{model}'] = row
    lines.append(f"| {proto} | {model} | 5/5 | {row['map5095']['mean']:.4f} ({row['map5095']['sd']:.4f}) | {row['macro']['mean']:.4f} | {row['map50']['mean']:.4f} |")
lines.append('')
for model in ('yolo11s.pt', 'yolov8s.pt'):
    a1, a2 = out.get(f'v1:{model}'), out.get(f'v2:{model}')
    if a1 and a2: lines.append(f"Shift v1 -> v2, {model}: mAP50-95 {a2['map5095']['mean'] - a1['map5095']['mean']:+.4f} (unpaired; v2 test sets differ; single seed)")
for proto in ('v1', 'v2'):
    p, q = out.get(f'{proto}:yolo11s.pt'), out.get(f'{proto}:yolov8s.pt')
    if p and q:
        d = np.array(p['map5095']['per_fold']) - np.array(q['map5095']['per_fold']); half = tcrit * d.std(ddof=1) / np.sqrt(5)
        lines.append(f"Gap YOLO11s - YOLOv8s, {proto}: {d.mean():+.4f} +/- {half:.4f} (paired by fold, t=2.776, 4 df); folds positive {(d > 0).sum()}/5")
Path(a.out + '.json').parent.mkdir(parents=True, exist_ok=True)
Path(a.out + '.json').write_text(json.dumps(out, indent=1)); Path(a.out + '.md').write_text('\n'.join(lines) + '\n'); print('\n'.join(lines))
