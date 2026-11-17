#!/usr/bin/env python3
"""Read-only inspection of FracAtlas annotations + box-scale comparison with HBFMID fracture boxes.
Usage: python scripts/eda/fracatlas_inspect.py --fracatlas "<.../FracAtlas>" --hbfmid "<.../Bone Fractures Detection>" --out results/eda"""
import argparse, os, json
import numpy as np, pandas as pd
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument('--fracatlas', required=True); ap.add_argument('--hbfmid', required=True); ap.add_argument('--out', default='results/eda')
a = ap.parse_args()
d = pd.read_csv(os.path.join(a.fracatlas, 'dataset.csv'))
size = {}
for sub in ('Fractured', 'Non_fractured'):
    for f in os.listdir(os.path.join(a.fracatlas, 'images', sub)):
        size[f] = Image.open(os.path.join(a.fracatlas, 'images', sub, f)).size  # (W,H), header only


def yolo(path):
    return [[float(x) for x in l.split()] for l in open(path).read().splitlines() if l.strip()] if os.path.exists(path) else None


fa_side, fa_area, mism, nf_boxes, fr_nobox = [], [], 0, 0, 0
for r in d.itertuples():
    b = yolo(os.path.join(a.fracatlas, 'Annotations', 'YOLO', r.image_id.replace('.jpg', '.txt')))
    if b is None: continue
    if len(b) != r.fracture_count: mism += 1
    if r.fractured == 0 and b: nf_boxes += 1
    if r.fractured == 1 and not b: fr_nobox += 1
    W, H = size[r.image_id]; sc = 640 / max(W, H)             # letterbox to imgsz 640
    for x in b:
        fa_area.append(x[3] * x[4]); fa_side.append(np.sqrt(x[3] * W * x[4] * H) * sc)
hb_side, hb_area = [], []
for sp in ('train', 'valid', 'test'):
    ld = os.path.join(a.hbfmid, sp, 'labels')
    for f in os.listdir(ld):
        for l in open(os.path.join(ld, f)):
            p = l.split()
            if p and int(p[0]) != 2:                          # class 2 = Healthy
                w, h = float(p[3]), float(p[4]); hb_area.append(w * h); hb_side.append(np.sqrt(w * 640 * h * 640))
q = lambda v: {k: round(float(np.percentile(v, p)), 4) for k, p in (('p10', 10), ('median', 50), ('p90', 90))}
sizes = pd.Series([f'{w}x{h}' for w, h in size.values()]).value_counts()
out = {'images': len(d), 'fractured_images': int(d.fractured.sum()), 'boxes': len(fa_area),
       'count_mismatch_images': mism, 'nonfractured_with_boxes': nf_boxes, 'fractured_without_boxes': fr_nobox,
       'fracture_count_hist': d.fracture_count.value_counts().sort_index().to_dict(),
       'fractured_by_region': d[d.fractured == 1][['hand', 'leg', 'hip', 'shoulder', 'mixed']].sum().to_dict(),
       'hardware_among_fractured': int(d[d.fractured == 1].hardware.sum()),
       'top_image_sizes': sizes.head(5).to_dict(),
       'fracatlas_box_area_frac': q(fa_area), 'hbfmid_fracture_box_area_frac': q(hb_area),
       'fracatlas_box_side_px_at_640': q(fa_side), 'hbfmid_fracture_box_side_px_at_640': q(hb_side),
       'hbfmid_nonhealthy_boxes': len(hb_area)}
json.dump(out, open(os.path.join(a.out, 'fracatlas_summary.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
