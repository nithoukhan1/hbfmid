#!/usr/bin/env python3
"""Does residual leakage in the v1 clean folds explain the v1 -> v2 drop? Uses saved predictions only (no GPU).
For each v1 clean test image, find its grouping-v2 group; the image is 'v2-leaky' if another member of that group is in the v1 TRAIN list of that fold.
Reports ground-truth box recall (conf >= thr, IoU >= 0.5, any class) for leaky vs non-leaky test images, per model, plus counts.
Usage: python scripts/evaluation/v1_leak_stratified.py --runs results/bench_A results/baseline_fixed results/controls --data-root "<.../Bone Fractures Detection>" --out results/analysis/v1_leak_stratified"""
import argparse, glob, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from evaluation.confusion import load_gt, iou
ap = argparse.ArgumentParser()
ap.add_argument('--runs', nargs='+', required=True); ap.add_argument('--data-root', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--groups-v2', default='results/eda/grouping_v2/image_groups_v2_ncc0.95.csv'); ap.add_argument('--splits', default='data_splits/clean5')
ap.add_argument('--groups-v2-low', default='results/eda/grouping_v2/image_groups_v2_ncc0.90.csv', help='looser grouping; adds a band stratum (linked at 0.90 but not at 0.95)')
ap.add_argument('--thr', type=float, default=0.25); ap.add_argument('--seed', type=int, default=42)
a = ap.parse_args()
G = pd.read_csv(a.groups_v2); g2 = {f'{s}/images/{i}': g for s, i, g in zip(G.split, G.image, G.group_id)}
members = {}
for k, g in g2.items(): members.setdefault(g, set()).add(k)
Gl = pd.read_csv(a.groups_v2_low); g3 = {f'{s}/images/{i}': g for s, i, g in zip(Gl.split, Gl.image, Gl.group_id)}
mem3 = {}
for k, g in g3.items(): mem3.setdefault(g, set()).add(k)
train = {f: set(open(f'{a.splits}/fold{f}_train.txt').read().split()) for f in range(5)}
rows = []
for d in a.runs:
    for mp in glob.glob(os.path.join(d, 'clean_f*_*', 'metrics.json')):
        m = json.load(open(mp))
        if m.get('protocol') != 'fixed_last' or m.get('arm', 'baseline') != 'baseline' or m['seed'] != a.seed: continue
        pp = os.path.join(os.path.dirname(mp), 'preds_test.json')
        if not os.path.exists(pp): continue
        for k, P in json.load(open(pp)).items():
            leaky = any(o != k and o in train[m['fold']] for o in members[g2[k]])
            band = (not leaky) and any(o != k and o in train[m['fold']] for o in mem3[g3[k]])
            gt = load_gt(os.path.join(a.data_root, k.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'))
            P = [p for p in P if p[1] >= a.thr]
            for g in gt: rows.append((m['model'], m['fold'], k, leaky, band, any(iou(g[1:], p[2:6]) >= 0.5 for p in P)))
D = pd.DataFrame(rows, columns=['model', 'fold', 'img', 'leaky', 'band', 'found'])
out = {}
lines = ['| model | test images leaky(0.95) / band(0.90-0.95) / total | boxes leaky / band / total | recall clean | recall band | recall leaky |', '|---|---|---|---|---|---|']
for model, S in D.groupby('model'):
    imgs = S.drop_duplicates(['fold', 'img']); nl, l, bd = S[~S.leaky & ~S.band], S[S.leaky], S[S.band]
    out[model] = {'imgs_leaky': int(imgs.leaky.sum()), 'imgs': int(len(imgs)), 'boxes_leaky': int(len(l)), 'boxes': int(len(S)),
                  'recall_nonleaky': float(nl.found.mean()), 'imgs_band': int(imgs.band.sum()), 'boxes_band': int(len(bd)), 'recall_band': float(bd.found.mean()) if len(bd) else None, 'recall_leaky': float(l.found.mean()) if len(l) else None,
                  'per_fold_leaky_imgs': {int(f): int(x.drop_duplicates('img').leaky.sum()) for f, x in S.groupby('fold')}}
    lines.append(f"| {model} | {out[model]['imgs_leaky']} / {out[model]['imgs_band']} / {out[model]['imgs']} | {len(l)} / {len(bd)} / {len(S)} | {nl.found.mean():.3f} | {bd.found.mean():.3f} | {l.found.mean():.3f} |")
json.dump(out, open(a.out + '.json', 'w'), indent=1); open(a.out + '.md', 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines)); print(json.dumps({k: v['per_fold_leaky_imgs'] for k, v in out.items()}))
