#!/usr/bin/env python3
"""Build a manifest of FracAtlas (READ-ONLY; nothing is written inside the FracAtlas folder).
One row per image: relative path, size, fractured flag, normalised YOLO boxes (all class 0 = 'fracture'), region/hardware flags,
and 'decode_ok' (full decode succeeds). Resumable: use --budget seconds, exit code 3 = not finished.
Usage: python scripts/data/fracatlas_manifest.py --fracatlas "<.../FracAtlas>" --out data_splits/fracatlas --cache <local dir> --budget 150"""
import argparse, os, sys, json, time, pickle
import pandas as pd
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument('--fracatlas', required=True); ap.add_argument('--out', default='data_splits/fracatlas')
ap.add_argument('--cache', required=True); ap.add_argument('--budget', type=float, default=0)
a = ap.parse_args()
assert not os.path.abspath(a.out).startswith(os.path.abspath(a.fracatlas)), 'output inside dataset folder'
os.makedirs(a.out, exist_ok=True); os.makedirs(a.cache, exist_ok=True)
d = pd.read_csv(os.path.join(a.fracatlas, 'dataset.csv'))
cp = os.path.join(a.cache, 'fa_manifest.pkl'); st = pickle.load(open(cp, 'rb')) if os.path.exists(cp) else {}
t0 = time.time()
for r in d.itertuples():
    if r.image_id in st: continue
    sub = 'Fractured' if r.fractured == 1 else 'Non_fractured'
    p = os.path.join(a.fracatlas, 'images', sub, r.image_id)
    try:
        im = Image.open(p); W, H = im.size; im.load(); ok = True
    except Exception as e:
        W, H, ok = (im.size if 'im' in dir() else (0, 0)) + (False,) if False else (0, 0, False)
        try: W, H = Image.open(p).size
        except Exception: pass
    lp = os.path.join(a.fracatlas, 'Annotations', 'YOLO', r.image_id.replace('.jpg', '.txt'))
    boxes = [[float(x) for x in l.split()[1:5]] for l in open(lp).read().splitlines() if l.strip()] if os.path.exists(lp) else []
    st[r.image_id] = dict(image_id=r.image_id, rel_path=f'images/{sub}/{r.image_id}', width=W, height=H, fractured=int(r.fractured),
                          n_boxes=len(boxes), boxes_cxcywh=json.dumps(boxes), hand=r.hand, leg=r.leg, hip=r.hip, shoulder=r.shoulder,
                          mixed=r.mixed, hardware=r.hardware, decode_ok=ok)
    if a.budget and time.time() - t0 > a.budget:
        pickle.dump(st, open(cp, 'wb')); print('PAUSED at', len(st), '/', len(d)); sys.exit(3)
pickle.dump(st, open(cp, 'wb'))
m = pd.DataFrame(list(st.values())); m.to_csv(os.path.join(a.out, 'manifest.csv'), index=False)
s = {'images': len(m), 'fractured': int(m.fractured.sum()), 'boxes': int(m.n_boxes.sum()), 'decode_failures': m[~m.decode_ok].image_id.tolist(),
     'zero_size': int((m.width == 0).sum())}
json.dump(s, open(os.path.join(a.out, 'manifest_summary.json'), 'w'), indent=1); print(json.dumps(s, indent=1))
