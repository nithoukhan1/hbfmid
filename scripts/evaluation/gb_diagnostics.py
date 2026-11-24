#!/usr/bin/env python3
"""EXPLORATORY (not pre-registered) diagnostics for the C1 group-balanced arm vs the plain v2 baselines (3 seeds x 5 folds, both detectors).
1) Validation-fold check: paired change of val mAP50-95 (second held-out set). 2) Operating point: change of test precision / recall.
3) Final-epoch losses (train losses are on different training distributions: indicative only; val losses are comparable).
4) Per-class: change of the class's share of training images caused by the balancing (exposure ratio, image level) vs change of test AP50-95.
Usage: python scripts/evaluation/gb_diagnostics.py --out results/analysis/method_gb_diagnostics"""
import argparse, csv, glob, json
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('--out', default='results/analysis/method_gb_diagnostics'); a = ap.parse_args()
DIRS = {('yolov8s.pt', 'baseline'): ['results/v2_B', 'results/v2s_B'], ('yolo11s.pt', 'baseline'): ['results/v2_A', 'results/v2s_A'],
        ('yolov8s.pt', 'gb'): ['results/gb_A', 'results/gb_B'], ('yolo11s.pt', 'gb'): ['results/gb_A', 'results/gb_B']}
R = {}
for (model, arm), ds in DIRS.items():
    for d in ds:
        for p in glob.glob(f'{d}/*/metrics.json'):
            r = json.load(open(p))
            if r.get('mode') == 'clean' and r.get('protocol') == 'fixed_last' and r['model'] == model and r.get('arm', 'baseline') == arm:
                rows = list(csv.DictReader(open(Path(p).parent / 'train' / 'results.csv'))); last = {k.strip(): float(v) for k, v in rows[-1].items()}
                R[(model, arm, r['fold'], r['seed'])] = {'r': r, 'last': last, 'dir': Path(p).parent}
models, folds, seeds = ['yolov8s.pt', 'yolo11s.pt'], range(5), [42, 43, 44]
assert all((m, arm, f, s) in R for m in models for arm in ['baseline', 'gb'] for f in folds for s in seeds), 'missing runs'
out = {'note': 'exploratory, not pre-registered'}; L = ['# C1 group-balanced arm: exploratory diagnostics (not pre-registered)', '']
def paired(getter):
    res = {}
    for m in models:
        d = np.array([np.mean([getter(R[(m, 'gb', f, s)]) - getter(R[(m, 'baseline', f, s)]) for s in seeds]) for f in folds])
        res[m] = {'mean': float(d.mean()), 'per_fold': d.tolist(), 'folds_positive': int((d > 0).sum())}
    return res
checks = {'test_map5095': lambda x: x['r']['test']['map5095'], 'val_map5095': lambda x: x['r']['val']['map5095'],
          'test_precision': lambda x: x['r']['test']['precision'], 'test_recall': lambda x: x['r']['test']['recall'],
          'train_box_loss_last': lambda x: x['last']['train/box_loss'], 'train_cls_loss_last': lambda x: x['last']['train/cls_loss'],
          'val_box_loss_last': lambda x: x['last']['val/box_loss'], 'val_cls_loss_last': lambda x: x['last']['val/cls_loss']}
L += ['| quantity (gb - baseline, seed-averaged, paired by fold) | YOLOv8s mean (folds > 0) | YOLO11s mean (folds > 0) |', '|---|---|---|']
for k, g in checks.items():
    o = paired(g); out[k] = o; sc = 100 if 'map' in k or 'precision' in k or 'recall' in k else 1
    unit = ' pts' if sc == 100 else ''
    L.append(f"| {k} | {sc * o['yolov8s.pt']['mean']:+.3f}{unit} ({o['yolov8s.pt']['folds_positive']}/5) | {sc * o['yolo11s.pt']['mean']:+.3f}{unit} ({o['yolo11s.pt']['folds_positive']}/5) |")
# per-class exposure (image level) and AP change
names = list(R[('yolov8s.pt', 'baseline', 0, 42)]['r']['test']['per_class'].keys())
cls_of = {}
for row in csv.DictReader(open('data_splits_v2/clean5/assignments.csv')):
    cls_of[row['path']] = {int(c) for c in row['classes'].replace(',', ' ').replace(';', ' ').split() if c.strip() != ''}
def rel(pth):
    i = pth.find('Bone Fractures Detection/'); return pth[i + len('Bone Fractures Detection/'):].strip()
expo = {c: [] for c in range(len(names))}
for f in folds:
    base = [rel(l) for l in (R[('yolov8s.pt', 'baseline', f, 42)]['dir'] / 'lists' / 'train.txt').read_text().splitlines() if l.strip()]
    gbl = [rel(l) for l in (R[('yolov8s.pt', 'gb', f, 42)]['dir'] / 'lists' / 'train.txt').read_text().splitlines() if l.strip()]
    for c in range(len(names)):
        sb = sum(c in cls_of[p] for p in base) / len(base); sg = sum(c in cls_of[p] for p in gbl) / len(gbl)
        if sb > 0: expo[c].append(sg / sb)
per_class = {}
for c, nm in enumerate(names):
    ds = []
    for m in models:
        for f in folds:
            pres = nm in R[(m, 'baseline', f, 42)]['r']['test'].get('classes_present', [])
            if not pres: continue
            ds.append(np.mean([R[(m, 'gb', f, s)]['r']['test']['per_class'][nm]['ap5095'] - R[(m, 'baseline', f, s)]['r']['test']['per_class'][nm]['ap5095'] for s in seeds]))
    per_class[nm] = {'exposure_ratio': float(np.mean(expo[c])) if expo[c] else None, 'delta_ap5095': float(np.mean(ds)) if ds else None, 'n_detector_folds': len(ds)}
out['per_class'] = per_class
L += ['', 'Per class: exposure ratio = class share of training images after / before balancing (mean over folds); AP change pooled over both detectors and the folds where the class is in the test set.', '',
      '| class | exposure ratio | delta AP50-95 (pts) | detector-folds |', '|---|---|---|---|']
for nm, o in sorted(per_class.items(), key=lambda kv: -(kv[1]['exposure_ratio'] or 0)):
    L.append(f"| {nm} | {o['exposure_ratio']:.2f} | {100 * o['delta_ap5095']:+.2f} | {o['n_detector_folds']} |" if o['delta_ap5095'] is not None else f"| {nm} | {o['exposure_ratio']} | n/a | 0 |")
xs = [o['exposure_ratio'] for o in per_class.values() if o['delta_ap5095'] is not None]; ys = [o['delta_ap5095'] for o in per_class.values() if o['delta_ap5095'] is not None]
rho = float(np.corrcoef(np.argsort(np.argsort(xs)), np.argsort(np.argsort(ys)))[0, 1]); out['spearman_exposure_vs_delta_ap'] = rho
L += ['', f'Spearman rank correlation, exposure ratio vs AP change, over {len(xs)} classes: {rho:+.2f} (exploratory; per-class AP on about 57 test images per fold is very noisy).']
Path(a.out + '.json').write_text(json.dumps(out, indent=1)); Path(a.out + '.md').write_text('\n'.join(L) + '\n'); print('\n'.join(L))
