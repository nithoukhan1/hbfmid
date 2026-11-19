#!/usr/bin/env python3
"""Which ground-truth boxes do the detectors miss, and why? Uses saved predictions only (no GPU).

For every ground-truth box of every HBFMID-Clean test run: found = any prediction with conf >= thr and IoU >= 0.5 (any class).
Reports the miss rate by box size (fraction of image area), by modality, by class, per detector, plus how many boxes are
missed by EVERY run ("never found") and how many are found by every run.
Usage: python scripts/evaluation/miss_analysis.py --runs results/bench_A results/bench_B results/baseline_fixed results/controls \
         --data-root "<.../Bone Fractures Detection>" --out results/analysis/miss_analysis [--thr 0.25]"""
import argparse, glob, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from evaluation.confusion import load_gt, iou
from common.taxonomy import CLASS_NAMES

ap = argparse.ArgumentParser()
ap.add_argument('--runs', nargs='+', required=True); ap.add_argument('--data-root', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--modality', default='results/eda/modality_tags.csv'); ap.add_argument('--thr', type=float, default=0.25)
a = ap.parse_args()
mod = pd.read_csv(a.modality).set_index('image').modality.to_dict()
rows = []
for d in a.runs:
    for mp in glob.glob(os.path.join(d, 'clean_f*_*', 'metrics.json')):
        m = json.load(open(mp))
        if m.get('protocol') != 'fixed_last' or m.get('arm', 'baseline') != 'baseline':
            continue
        pp = os.path.join(os.path.dirname(mp), 'preds_test.json')
        if not os.path.exists(pp):
            continue
        for k, P in json.load(open(pp)).items():
            gt = load_gt(os.path.join(a.data_root, k.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'))
            P = [p for p in P if p[1] >= a.thr]
            for gi, g in enumerate(gt):
                found = any(iou(g[1:], p[2:6]) >= 0.5 for p in P)
                top = max([(p[1], int(p[0])) for p in P if iou(g[1:], p[2:6]) >= 0.5], default=(0, -1))[1]
                rows.append((m['model'], m['fold'], m['seed'], os.path.basename(k), gi, g[0], (g[3] - g[1]) * (g[4] - g[2]), found, top == g[0] if found else False))
D = pd.DataFrame(rows, columns=['model', 'fold', 'seed', 'img', 'gi', 'cls', 'area', 'found', 'cls_ok'])
D['cls_name'] = D.cls.map(lambda c: CLASS_NAMES[c]); D['mod'] = D.img.map(mod)
bins = [0, 0.01, 0.02, 0.05, 0.15, 1.01]; labs = ['<1%', '1-2%', '2-5%', '5-15%', '>=15%']
D['size'] = pd.cut(D.area, bins=bins, labels=labs, right=False)
out = {'thr': a.thr, 'n_rows': int(len(D)), 'models': sorted(D.model.unique())}
box = D.groupby(['fold', 'img', 'gi', 'cls']).size().index  # unique GT boxes
for key in ['size', 'mod', 'cls_name']:
    t = D.groupby(['model', key], observed=True).agg(n=('found', 'size'), miss=('found', lambda x: 1 - x.mean()), cls_acc=('cls_ok', 'mean')).round(3)
    out[f'by_{key}'] = {f'{m}|{k}': v for (m, k), v in t.to_dict('index').items()}
# per unique box, pooled over every run of every detector
U = D.groupby(['fold', 'img', 'gi']).agg(cls=('cls_name', 'first'), area=('area', 'first'), mod=('mod', 'first'), size=('size', 'first'), runs=('found', 'size'), found_frac=('found', 'mean')).reset_index()
out['unique_boxes'] = int(len(U)); out['never_found'] = int((U.found_frac == 0).sum()); out['always_found'] = int((U.found_frac == 1).sum())
out['mixed'] = int(len(U) - out['never_found'] - out['always_found'])
out['never_found_by_size'] = U[U.found_frac == 0]['size'].value_counts().reindex(labs).fillna(0).astype(int).to_dict()
out['unique_by_size'] = U['size'].value_counts().reindex(labs).fillna(0).astype(int).to_dict()
out['never_found_by_class'] = U[U.found_frac == 0].cls.value_counts().to_dict()
out['never_found_by_mod'] = U[U.found_frac == 0]['mod'].value_counts().to_dict()
out['unique_by_mod'] = U['mod'].value_counts().to_dict()
out['median_area_found_vs_never'] = {'always_found': float(U[U.found_frac == 1].area.median()), 'never_found': float(U[U.found_frac == 0].area.median())}
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(out, open(a.out + '.json', 'w'), indent=1, default=str)
D.to_csv(a.out + '_boxes.csv', index=False)
print(json.dumps({k: v for k, v in out.items() if not k.startswith('by_')}, indent=1, default=str))
for key in ['size', 'mod']:
    print(f'\nmiss rate by {key} (conf>={a.thr}):')
    print(D.groupby(['model', key], observed=True).agg(n=('found', 'size'), miss=('found', lambda x: round(1 - x.mean(), 3))).unstack(0).to_string())
