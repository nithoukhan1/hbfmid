#!/usr/bin/env python3
"""Build a YOLO-format working copy of FracAtlas (single 'fracture' objective) outside the dataset folders.

Why: the FracAtlas folder layout (images/Fractured, images/Non_fractured) has no YOLO labels that Ultralytics can find, and the
dataset folders must stay read-only. Labels come from data_splits/fracatlas/manifest.csv (boxes_cxcywh, normalized).

Variants
  generic : original images (symlinks), one class id for every box (--cls-id: 0 for pre-training, 10 = extra class for joint training)
  zoom    : scale-matched crops. FracAtlas boxes are ~2.4x smaller than HBFMID boxes at imgsz 640 (median side 36.6 vs 88.3 px).
            Each fractured image yields --crops-per-image crops whose zoom z = clip(target/box_side, 1, max_zoom) brings the anchor
            box to ~target px; crops are saved at native resolution (Ultralytics resizes them to imgsz later).
Broken JPEGs (decode_ok == False, all non-fractured) are skipped. Negatives (non-fractured images, empty label files) are added
at --neg-ratio x (number of fractured sources) for the pre-training lists; the joint-training list (`fractured_all.txt`) has none.

Output (--out):  images/{train,val}/*.jpg   labels/{train,val}/*.txt   lists/{train,val,fractured_all}.txt (absolute paths)   summary.json
The val split (--val-frac of source images, image-level, seeded) exists only to monitor pre-training; all fractured items are in fractured_all.txt.
"""
import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def side640(bw, bh, W, H):
    return math.sqrt(bw * bh) * 640.0 / max(W, H)


def make_crop(W, H, boxes, anchor, target, max_zoom, rng):
    """boxes: list of (cx,cy,w,h) normalized. Returns (x0,y0,cw,ch, z, kept_boxes_in_crop_norm)."""
    cx, cy, w, h = boxes[anchor]
    bw, bh = w * W, h * H
    bx1, by1 = cx * W - bw / 2, cy * H - bh / 2
    z = min(max(target / max(side640(bw, bh, W, H), 1e-6), 1.0), max_zoom)
    z = min(z, W / (1.15 * bw), H / (1.15 * bh))          # crop must hold the anchor box with a margin
    if z < 1.05:
        return 0, 0, W, H, 1.0, [(0,) + tuple(b) for b in boxes]
    cw, ch = W / z, H / z
    x0 = rng.uniform(max(0.0, bx1 + bw - cw), min(bx1, W - cw))
    y0 = rng.uniform(max(0.0, by1 + bh - ch), min(by1, H - ch))
    kept = []
    for (c2x, c2y, w2, h2) in boxes:
        x1, y1, x2, y2 = (c2x - w2 / 2) * W, (c2y - h2 / 2) * H, (c2x + w2 / 2) * W, (c2y + h2 / 2) * H
        ix1, iy1, ix2, iy2 = max(x1, x0), max(y1, y0), min(x2, x0 + cw), min(y2, y0 + ch)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        if (ix2 - ix1) * (iy2 - iy1) < 0.5 * (x2 - x1) * (y2 - y1):
            continue                                       # mostly outside: drop (documented label noise, see summary)
        kept.append((0, ((ix1 + ix2) / 2 - x0) / cw, ((iy1 + iy2) / 2 - y0) / ch, (ix2 - ix1) / cw, (iy2 - iy1) / ch))
    return x0, y0, cw, ch, z, kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fa-root', required=True, help='FracAtlas folder containing images/ (read-only)')
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--variant', choices=['generic', 'zoom'], default='generic')
    ap.add_argument('--cls-id', type=int, default=0)
    ap.add_argument('--neg-ratio', type=float, default=0.5)
    ap.add_argument('--val-frac', type=float, default=0.05)
    ap.add_argument('--target-side', type=float, default=88.3, help='HBFMID median fracture box side at imgsz 640 (results/eda/fracatlas_summary.json)')
    ap.add_argument('--max-zoom', type=float, default=4.0)
    ap.add_argument('--crops-per-image', type=int, default=2)
    ap.add_argument('--jpeg-quality', type=int, default=95)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--limit', type=int, default=0, help='debug: use only this many fractured images')
    a = ap.parse_args()

    rng = random.Random(a.seed)
    root, out = Path(a.fa_root), Path(a.out).resolve()
    assert not str(out).startswith(str(root.resolve())), 'output must not be inside the FracAtlas folder'
    m = pd.read_csv(a.manifest)
    n_broken = int((~m.decode_ok).sum())
    m = m[m.decode_ok].reset_index(drop=True)
    pos = m[m.fractured == 1].sample(frac=1.0, random_state=a.seed).reset_index(drop=True)
    neg = m[m.fractured == 0].sample(frac=1.0, random_state=a.seed).reset_index(drop=True)
    if a.limit:
        pos = pos.iloc[:a.limit]
    neg = neg.iloc[:int(round(a.neg_ratio * len(pos)))]
    for sub in ('images/train', 'images/val', 'labels/train', 'labels/val', 'lists'):
        (out / sub).mkdir(parents=True, exist_ok=True)

    items = {'train': [], 'val': []}            # (abs image path, is_pos)
    split_of = {}
    for df in (pos, neg):
        ids = list(df.image_id)
        rng.shuffle(ids)
        nv = int(round(a.val_frac * len(ids)))
        for i, iid in enumerate(ids):
            split_of[iid] = 'val' if i < nv else 'train'

    sides_before, sides_after, zooms, dropped, kept_n = [], [], [], 0, 0
    for _, r in pos.iterrows():
        W, H = int(r.width), int(r.height)
        boxes = [tuple(b) for b in json.loads(r.boxes_cxcywh)]
        sp = split_of[r.image_id]
        stem = Path(r.image_id).stem
        src = root / r.rel_path
        sides_before += [side640(b[2] * W, b[3] * H, W, H) for b in boxes]
        if a.variant == 'generic':
            dst = out / 'images' / sp / f'{stem}.jpg'
            if not dst.exists():
                os.symlink(src, dst)
            (out / 'labels' / sp / f'{stem}.txt').write_text(
                '\n'.join(f'{a.cls_id} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}' for b in boxes) + '\n')
            items[sp].append((str(dst), True)); sides_after += sides_before[-len(boxes):]; zooms.append(1.0); kept_n += len(boxes)
        else:
            im = Image.open(src).convert('RGB')
            anchors = list(range(len(boxes)))
            rng.shuffle(anchors)
            for k in range(a.crops_per_image):
                an = anchors[k % len(anchors)]
                x0, y0, cw, ch, z, kept = make_crop(W, H, boxes, an, a.target_side, a.max_zoom, rng)
                if not kept:
                    continue
                dropped += len(boxes) - len(kept)
                name = f'{stem}_z{k}'
                crop = im.crop((int(round(x0)), int(round(y0)), int(round(x0 + cw)), int(round(y0 + ch))))
                crop.save(out / 'images' / sp / f'{name}.jpg', quality=a.jpeg_quality)
                (out / 'labels' / sp / f'{name}.txt').write_text(
                    '\n'.join(f'{a.cls_id} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}' for b in kept) + '\n')
                items[sp].append((str(out / 'images' / sp / f'{name}.jpg'), True))
                zooms.append(z); kept_n += len(kept)
                sides_after += [side640(b[3] * crop.width, b[4] * crop.height, crop.width, crop.height) for b in kept]
    for _, r in neg.iterrows():
        sp = split_of[r.image_id]
        stem = Path(r.image_id).stem
        dst = out / 'images' / sp / f'{stem}.jpg'
        if not dst.exists():
            os.symlink(root / r.rel_path, dst)
        (out / 'labels' / sp / f'{stem}.txt').write_text('')
        items[sp].append((str(dst), False))

    for sp in ('train', 'val'):
        (out / 'lists' / f'{sp}.txt').write_text('\n'.join(p for p, _ in items[sp]) + '\n')
    allpos = [p for sp in ('train', 'val') for p, ispos in items[sp] if ispos]
    (out / 'lists' / 'fractured_all.txt').write_text('\n'.join(allpos) + '\n')
    q = lambda x: {k: round(float(np.percentile(x, v)), 2) for k, v in (('p10', 10), ('median', 50), ('p90', 90))} if len(x) else None
    summ = {'variant': a.variant, 'cls_id': a.cls_id, 'seed': a.seed, 'n_broken_skipped': n_broken, 'fractured_sources': int(len(pos)),
            'negatives': int(len(neg)), 'items_train': len(items['train']), 'items_val': len(items['val']),
            'fractured_items': len(allpos), 'boxes_kept': kept_n, 'boxes_dropped_in_crops': dropped,
            'box_side_px640_before': q(sides_before), 'box_side_px640_after': q(sides_after),
            'zoom': q(zooms), 'target_side': a.target_side, 'max_zoom': a.max_zoom, 'crops_per_image': a.crops_per_image}
    (out / 'summary.json').write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
