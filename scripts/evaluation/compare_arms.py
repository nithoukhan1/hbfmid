#!/usr/bin/env python3
"""Paired comparison of an arm against the baseline, fold by fold (fixed-epoch protocol, last.pt).

Per fold the metric is averaged over the seeds both arms share; the paired difference is arm - baseline.
Reports mean difference over folds, sd, a t-based 95% interval (4 df: rough), number of folds with a positive difference,
and the localisation recall (share of test GT boxes with any prediction at IoU>=0.5 and conf>=0.25, class-agnostic).
Usage: python scripts/evaluation/compare_arms.py --data-root "<HBFMID root>" --baseline-runs results/controls results/baseline_fixed
           --arm-runs results/arms_cb --arm-name cb_group --arm-tag cbgroup --out results/arms/cb_group_vs_baseline
(--arm-tag is the `arm` field in metrics.json; baseline = arm 'baseline'.)"""
import argparse, glob, json, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from evaluation.confusion import load_gt, iou  # noqa: E402

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}

ap = argparse.ArgumentParser()
ap.add_argument('--data-root', required=True)
ap.add_argument('--baseline-runs', nargs='+', required=True)
ap.add_argument('--arm-runs', nargs='+', required=True)
ap.add_argument('--arm-name', required=True)
ap.add_argument('--arm-tag', required=True)
ap.add_argument('--model', default='yolov8s.pt')
ap.add_argument('--out', required=True)
ap.add_argument('--thr', type=float, default=0.25)
a = ap.parse_args()


def load(dirs, arm):
    R = []
    for d in dirs:
        for p in glob.glob(str(Path(d) / '*/metrics.json')):
            r = json.loads(Path(p).read_text())
            if (r['mode'] == 'clean' and r.get('protocol') == 'fixed_last' and r['model'] == a.model
                    and r.get('arm', 'baseline') == arm):
                r['_dir'] = str(Path(p).parent); R.append(r)
    return R


def loc_recall(r):
    pj = Path(r['_dir']) / 'preds_test.json'
    pj_gen = Path(r['_dir']) / 'preds_test_generic.json'
    if not pj.exists():
        return None
    preds = json.loads(pj.read_text())
    if pj_gen.exists():                       # joint arm: the generic-fracture head also localises
        for k, v in json.loads(pj_gen.read_text()).items():
            preds.setdefault(k, []).extend(v)
    n = l = 0
    for k, P in preds.items():
        gt = load_gt(os.path.join(a.data_root, k.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'))
        P = [p for p in P if p[1] >= a.thr]
        for g in gt:
            n += 1
            l += any(iou(g[1:], p[2:6]) >= 0.5 for p in P)
    return l / n if n else None


B, A = load(a.baseline_runs, 'baseline'), load(a.arm_runs, a.arm_tag)
metrics = {'map5095': lambda r: r['test']['map5095'], 'macro': lambda r: r['test']['macro_eligible_ap5095'],
           'map50': lambda r: r['test']['map50'], 'loc_recall': loc_recall}
folds = sorted({r['fold'] for r in A} & {r['fold'] for r in B})
out = {'arm': a.arm_name, 'folds': folds, 'metrics': {}}
for m, fn in metrics.items():
    rows = []
    for f in folds:
        seeds = sorted({r['seed'] for r in A if r['fold'] == f} & {r['seed'] for r in B if r['fold'] == f})
        if not seeds:
            continue
        va = [fn(r) for r in A if r['fold'] == f and r['seed'] in seeds]
        vb = [fn(r) for r in B if r['fold'] == f and r['seed'] in seeds]
        if any(v is None for v in va + vb):
            continue
        rows.append((f, len(seeds), float(np.mean(vb)), float(np.mean(va))))
    if not rows:
        continue
    d = np.array([r[3] - r[2] for r in rows])
    n = len(d)
    sd = float(np.std(d, ddof=1)) if n > 1 else None
    ci = float(T975.get(n - 1, 1.96) * sd / np.sqrt(n)) if sd is not None else None
    out['metrics'][m] = {'per_fold': [{'fold': f, 'n_seeds': ns, 'baseline': b, 'arm': x, 'diff': x - b} for f, ns, b, x in rows],
                         'mean_diff': float(d.mean()), 'sd_diff': sd, 'ci95_halfwidth': ci, 'folds_positive': int((d > 0).sum()), 'n_folds': n}
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(str(a.out) + '.json').write_text(json.dumps(out, indent=1))
md = [f'# {a.arm_name} vs baseline (paired by fold; fixed protocol, last.pt)', '']
for m, v in out['metrics'].items():
    ci = '' if v['ci95_halfwidth'] is None else f" +/- {v['ci95_halfwidth']:.4f} (95%, t, {v['n_folds']-1} df)"
    md.append(f"**{m}**: mean diff {v['mean_diff']:+.4f}{ci}; positive in {v['folds_positive']}/{v['n_folds']} folds")
    md.append('| fold | seeds | baseline | arm | diff |'); md.append('|---|---|---|---|---|')
    md += [f"| {r['fold']} | {r['n_seeds']} | {r['baseline']:.4f} | {r['arm']:.4f} | {r['diff']:+.4f} |" for r in v['per_fold']]
    md.append('')
Path(str(a.out) + '.md').write_text('\n'.join(md)); print('\n'.join(md))
