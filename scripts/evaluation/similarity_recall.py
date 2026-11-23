#!/usr/bin/env python3
"""Leakage test by behaviour (saved predictions, no GPU): does box recall rise with a test image's similarity to the TRAIN set?
For each test image: max thumbnail NCC (32x32, 8 dihedral modes) to any image in that run's train list; boxes found = conf>=thr, IoU>=0.5, any class.
A leak-free split shows flat recall across similarity bins; a leaky split shows high recall in the top bins.
Usage: python scripts/evaluation/similarity_recall.py --data-root "<.../Bone Fractures Detection>" --out results/analysis/similarity_recall"""
import argparse, glob, json, os, sys
import cv2, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from evaluation.confusion import load_gt, iou
ap = argparse.ArgumentParser(); ap.add_argument('--data-root', required=True); ap.add_argument('--out', required=True); ap.add_argument('--thr', type=float, default=0.25)
a = ap.parse_args(); root = a.data_root
G = pd.read_csv('results/eda/image_groups.csv'); keys = [f'{s}/images/{i}' for s, i in zip(G.split, G.image)]; ix = {k: i for i, k in enumerate(keys)}
T = np.stack([cv2.resize(cv2.imread(os.path.join(root, k), cv2.IMREAD_GRAYSCALE), (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32) for k in keys])
def nm(X): X = X.reshape(len(X), -1); X = X - X.mean(1, keepdims=True); return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-6)
B = nm(T); S = np.full((len(keys),) * 2, -1.0, np.float32)
for k in range(4):
    for fl in (False, True):
        M = np.rot90(T, k, axes=(1, 2)); M = M[:, :, ::-1] if fl else M; S = np.maximum(S, B @ nm(np.ascontiguousarray(M)).T)
SETS = {'official': ('data_splits', ['results/bench_A', 'results/bench_B', 'results/baseline_fixed', 'results/controls'], 'official'),
        'clean_v1': ('data_splits', ['results/bench_A', 'results/baseline_fixed', 'results/controls'], 'clean'),
        'clean_v2': ('data_splits_v2', ['results/v2_A', 'results/v2_B'], 'clean')}
bins = [-1, 0.8, 0.9, 0.95, 1.01]; labs = ['<0.80', '0.80-0.90', '0.90-0.95', '>=0.95']
rows = []
for name, (sp, dirs, mode) in SETS.items():
    for d in dirs:
        for mp in glob.glob(os.path.join(d, '*/metrics.json')):
            m = json.load(open(mp))
            if m.get('mode') != mode or m.get('protocol') != 'fixed_last' or m.get('arm', 'baseline') != 'baseline' or m['seed'] != 42 or m['model'] not in ('yolo11s.pt', 'yolov8s.pt'): continue
            pp = os.path.join(os.path.dirname(mp), 'preds_test.json')
            if not os.path.exists(pp): continue
            tr = open(f"{sp}/official/train.txt" if mode == 'official' else f"{sp}/clean5/fold{m['fold']}_train.txt").read().split()
            tri = np.array([ix[k] for k in tr if k in ix])
            for k, P in json.load(open(pp)).items():
                mx = float(S[ix[k], tri].max()); P = [p for p in P if p[1] >= a.thr]
                gt = load_gt(os.path.join(root, k.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'))
                for g in gt: rows.append((name, m['model'], m['fold'], k, mx, any(iou(g[1:], p[2:6]) >= 0.5 for p in P)))
D = pd.DataFrame(rows, columns=['protocol', 'model', 'fold', 'img', 'maxsim', 'found']); D['bin'] = pd.cut(D.maxsim, bins=bins, labels=labs, right=False)
lines = ['| protocol | model | similarity bin (max NCC to train) | test images | boxes | box recall |', '|---|---|---|---|---|---|']
out = {}
for (pr, mo, b), x in D.groupby(['protocol', 'model', 'bin'], observed=True):
    out[f'{pr}|{mo}|{b}'] = {'imgs': int(x.drop_duplicates(['fold', 'img']).shape[0]), 'boxes': int(len(x)), 'recall': float(x.found.mean())}
    lines.append(f"| {pr} | {mo} | {b} | {out[f'{pr}|{mo}|{b}']['imgs']} | {len(x)} | {x.found.mean():.3f} |")
json.dump(out, open(a.out + '.json', 'w'), indent=1); open(a.out + '.md', 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines))
