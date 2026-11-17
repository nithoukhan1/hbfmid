#!/usr/bin/env python3
"""Where do the errors come from? For every ground-truth box: missed (no prediction with IoU>=0.5 of ANY class),
correct class, sibling class, other class - using the saved predictions (conf >= thr). Also by modality and by class.
Usage: python scripts/evaluation/error_decomposition.py --runs results/baseline/baseline_results --data-root "<.../Bone Fractures Detection>" --out results/baseline/baseline_results"""
import argparse, glob, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from evaluation.confusion import load_gt, iou
from common.taxonomy import CLASS_NAMES, is_sibling

ap = argparse.ArgumentParser()
ap.add_argument('--runs', required=True); ap.add_argument('--data-root', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--modality', default='results/eda/modality_tags.csv'); ap.add_argument('--seed', default='42')
a = ap.parse_args()
mod = pd.read_csv(a.modality).set_index('image').modality.to_dict()


def analyse(preds, thr):
    out = []
    for k, P in preds.items():
        gt = load_gt(os.path.join(a.data_root, k.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'))
        P = [p for p in P if p[1] >= thr]
        for g in gt:
            ov = [(p[1], int(p[0])) for p in P if iou(g[1:], p[2:6]) >= 0.5]
            if not ov:
                out.append((g[0], 'missed', os.path.basename(k))); continue
            top = max(ov)[1]
            out.append((g[0], 'correct' if top == g[0] else ('sibling' if is_sibling(g[0], top) else 'other_class'), os.path.basename(k)))
    return out


res = {}
for name, pat in (('E2_clean', f'{a.runs}/clean_f?_yolov8s_s{a.seed}/preds_test.json'), ('E1_official', f'{a.runs}/official_yolov8s_s{a.seed}/preds_test.json')):
    for thr in (0.05, 0.10, 0.25):
        rows = []
        for p in sorted(glob.glob(pat)):
            rows += analyse(json.load(open(p)), thr)
        d = pd.DataFrame(rows, columns=['gt', 'outcome', 'img']); n = len(d); vc = d.outcome.value_counts(); loc = n - vc.get('missed', 0)
        res[f'{name}_conf{thr}'] = {'gt_boxes': n, 'localized': round(loc / n, 3), 'correct': round(vc.get('correct', 0) / n, 3),
                                    'sibling': round(vc.get('sibling', 0) / n, 3), 'other_class': round(vc.get('other_class', 0) / n, 3),
                                    'missed': round(vc.get('missed', 0) / n, 3), 'class_correct_among_localized': round(vc.get('correct', 0) / max(loc, 1), 3)}
        if name == 'E2_clean' and thr == 0.05:
            d['mod'] = d.img.map(mod)
            res['E2_by_modality_conf0.05'] = d.groupby('mod').outcome.value_counts(normalize=True).unstack(fill_value=0).round(2).assign(n=d.groupby('mod').size()).to_dict('index')
            t = d.groupby('gt').outcome.value_counts().unstack(fill_value=0); t.index = [CLASS_NAMES[i] for i in t.index]
            res['E2_by_class_conf0.05'] = t.to_dict('index')
json.dump(res, open(os.path.join(a.out, 'error_decomposition.json'), 'w'), indent=1)
for k, v in res.items():
    if 'conf' in k and 'by' not in k: print(k, v)
