#!/usr/bin/env python3
"""Leak probe on the official test set using box recall (more stable than AP on 16 images):
share of ground-truth boxes that get a prediction with IoU>=0.5 (any class / correct class), for the 48 test images with a
verified train copy vs the 16 without, in E1 (leaky training) and E1b (official split de-duplicated).
Usage: python scripts/evaluation/leak_probe_recall.py --data-root "<...>" --e1 <preds.json> --e1b <preds.json> --out results/controls/leak_probe_recall.json"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from evaluation.confusion import load_gt, iou

ap = argparse.ArgumentParser()
for k in ('data_root', 'e1', 'e1b', 'out'): ap.add_argument('--' + k.replace('_', '-'), required=True)
ap.add_argument('--splits-dir', default='data_splits'); ap.add_argument('--thr', type=float, default=0.25)
a = ap.parse_args()
rd = lambda p: {l.strip() for l in open(p) if l.strip()}
subsets = {'with_train_copy': rd(f'{a.splits_dir}/official/test_with_train_copy.txt'), 'no_train_copy': rd(f'{a.splits_dir}/official/test_no_train_copy.txt')}


def score(preds, keys):
    n = l = c = 0
    for k in keys:
        gt = load_gt(os.path.join(a.data_root, k.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'))
        P = [p for p in preds.get(k, []) if p[1] >= a.thr]
        for g in gt:
            n += 1; ov = [(p[1], int(p[0])) for p in P if iou(g[1:], p[2:6]) >= 0.5]
            if ov:
                l += 1; c += max(ov)[1] == g[0]
    return {'gt_boxes': n, 'localized': round(l / n, 3), 'correct_class': round(c / n, 3)}


out = {}
for name, p in (('E1_leaky', a.e1), ('E1b_dedup', a.e1b)):
    pr = json.load(open(p)); out[name] = {s: score(pr, keys) for s, keys in subsets.items()}
json.dump(out, open(a.out, 'w'), indent=1); print(json.dumps(out, indent=1))
