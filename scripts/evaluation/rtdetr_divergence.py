#!/usr/bin/env python3
"""RT-DETR divergence diagnostics and three views of the clean benchmark.

A run counts as 'diverged' when its final fp32 test mAP50-95 of last.pt is < 0.01 (train loss stays finite; the evaluated EMA weights are damaged).
Separately, 'fp16_val_nan_at_end' marks runs whose in-training (fp16, AMP) validation loss was NaN at the last epoch although the fp32 test evaluation is valid.
Views of RT-DETR clean mAP50-95 (per fold: mean over seeds, then mean over the 5 folds):
  V1 as measured   : last.pt, every run, diverged runs score 0 (the pre-declared protocol)
  V2 healthy only  : last.pt, diverged runs removed (fold mean over the remaining seeds)  -> optimistic, selection on the outcome
  V3 best.pt       : validation-selected checkpoint, every run (pre-declared secondary metric)
Paired differences vs a YOLO reference (per-fold mean over its seeds, last.pt), t-interval with 4 df.
Usage: python scripts/evaluation/rtdetr_divergence.py --rtdetr results/rtdetr_A results/rtdetr_B --yolo results/bench_A results/baseline_fixed --out results/analysis/rtdetr_divergence"""
import argparse, csv, glob, json, math
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--rtdetr', nargs='+', required=True)
ap.add_argument('--yolo', nargs='+', required=True)
ap.add_argument('--out', required=True)
a = ap.parse_args()
T4 = 2.776  # t(0.975, 4 df)


def load(dirs):
    out = []
    for d in dirs:
        for p in glob.glob(str(Path(d) / '*/metrics.json')):
            r = json.loads(Path(p).read_text())
            if r.get('protocol') == 'fixed_last' and r.get('arm', 'baseline') == 'baseline':
                r['_dir'] = str(Path(p).parent)
                out.append(r)
    return out


def num(x):
    try:
        return float(x)
    except ValueError:
        return float('nan')


def diag(r):
    rows = [{k.strip(): v for k, v in q.items()} for q in csv.DictReader(open(Path(r['_dir']) / 'train' / 'results.csv'))]
    nan = [math.isnan(num(q['val/giou_loss'])) for q in rows]
    tr_nan = sum(math.isnan(num(q['train/giou_loss'])) or math.isnan(num(q['train/cls_loss'])) for q in rows)
    perm = None
    for i in range(len(nan)):
        if all(nan[i:]):
            perm = i + 1
            break
    m = [num(q['metrics/mAP50-95(B)']) for q in rows]
    return {'tag': r['tag'], 'fold': r['fold'], 'seed': r['seed'], 'test_last': r['test']['map5095'],
            'test_best': r.get('test_best', {}).get('map5095'), 'val_nan_epochs': int(sum(nan)), 'train_nan_epochs': int(tr_nan),
            'diverged': bool(r['test']['map5095'] < 0.01), 'fp16_val_nan_at_end': bool(nan[-1]), 'permanent_nan_from_epoch': perm,
            'peak_val_map5095_during_training': float(np.nanmax(m)), 'peak_epoch': int(np.nanargmax(m)) + 1}


R = [r for r in load(a.rtdetr) if r['model'] == 'rtdetr-l.pt']
clean = [r for r in R if r['mode'] == 'clean']
D = [diag(r) for r in clean]
folds = sorted({d['fold'] for d in D})
seeds = sorted({d['seed'] for d in D})
by = {(d['fold'], d['seed']): d for d in D}


def fold_means(get, healthy_only=False):
    pf = []
    for f in folds:
        v = [get(by[(f, s)]) for s in seeds if (f, s) in by and not (healthy_only and by[(f, s)]['diverged'])]
        pf.append(float(np.mean(v)) if v else float('nan'))
    return pf


views = {'V1_last_all': fold_means(lambda d: d['test_last']), 'V2_last_healthy_only': fold_means(lambda d: d['test_last'], True),
         'V3_best_all': fold_means(lambda d: d['test_best'])}
Y = load(a.yolo)
ref = {}
for m in ('yolo11s.pt', 'yolov8s.pt'):
    rr = [r for r in Y if r['model'] == m and r['mode'] == 'clean']
    ref[m] = [float(np.mean([r['test']['map5095'] for r in rr if r['fold'] == f])) for f in folds]


def paired(x, y):
    d = np.array(x) - np.array(y)
    return {'mean_pts': float(100 * d.mean()), 'half_width_pts': float(100 * T4 * d.std(ddof=1) / math.sqrt(len(d))),
            'folds_positive': int((d > 0).sum()), 'per_fold_pts': [round(float(100 * v), 2) for v in d]}


off = {}
for r in R:
    if r['mode'] == 'official':
        off.setdefault('seeds', []).append(r['seed'])
        for k, g in (('map5095', r['test']['map5095']), ('map50', r['test']['map50'])):
            off.setdefault(k, []).append(g)
res = {'n_clean_runs': len(clean), 'n_diverged': sum(d['diverged'] for d in D), 'n_fp16_val_nan_at_end': sum(d['fp16_val_nan_at_end'] for d in D), 'n_runs_with_any_val_nan': sum(d['val_nan_epochs'] > 0 for d in D), 'runs': D,
       'views_mean_over_folds': {k: float(np.nanmean(v)) for k, v in views.items()},
       'views_per_fold': views, 'yolo_reference_per_fold': ref,
       'yolo_reference_mean': {m: float(np.mean(v)) for m, v in ref.items()},
       'paired_vs_yolo11s': {k: paired(v, ref['yolo11s.pt']) for k, v in views.items() if not any(math.isnan(x) for x in v)},
       'paired_vs_yolov8s': {k: paired(v, ref['yolov8s.pt']) for k, v in views.items() if not any(math.isnan(x) for x in v)},
       'official': {'seeds': off.get('seeds'), 'map5095_mean': float(np.mean(off['map5095'])), 'map5095_sd': float(np.std(off['map5095'], ddof=1)),
                    'map50_mean': float(np.mean(off['map50']))}}
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out + '.json').write_text(json.dumps(res, indent=1))
L = ['| run | last.pt | best.pt | val-NaN epochs | train-NaN epochs | fp16 val NaN at end | permanent NaN from | peak val mAP50-95 (epoch) |', '|---|---|---|---|---|---|---|---|']
for d in sorted(D, key=lambda d: (d['fold'], d['seed'])):
    L.append(f"| fold {d['fold']} seed {d['seed']} | {d['test_last']:.4f}{' (DIVERGED)' if d['diverged'] else ''} | {d['test_best']:.4f} | {d['val_nan_epochs']} | {d['train_nan_epochs']} | {'yes' if d['fp16_val_nan_at_end'] else 'no'} | {d['permanent_nan_from_epoch'] or '-'} | {d['peak_val_map5095_during_training']:.3f} ({d['peak_epoch']}) |")
print('\n'.join(L))
print('\nclean mAP50-95 (mean over folds): ' + ', '.join(f'{k} {v:.4f}' for k, v in res['views_mean_over_folds'].items()))
print('YOLO references: ' + ', '.join(f'{k} {v:.4f}' for k, v in res['yolo_reference_mean'].items()))
for ref_name in ('paired_vs_yolo11s', 'paired_vs_yolov8s'):
    for k, v in res[ref_name].items():
        print(f"{ref_name} {k}: {v['mean_pts']:+.2f} +/- {v['half_width_pts']:.2f} pts, folds positive {v['folds_positive']}/5, per fold {v['per_fold_pts']}")
print('diverged:', res['n_diverged'], '| fp16-val NaN at end:', res['n_fp16_val_nan_at_end'], '| runs with any val NaN epoch:', res['n_runs_with_any_val_nan'], 'of', res['n_clean_runs'])
print('official:', res['official'])
Path(a.out + '.md').write_text('\n'.join(L) + '\n')
