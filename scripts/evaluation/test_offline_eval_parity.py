#!/usr/bin/env python3
"""Parity tests for scripts/evaluation/offline_eval.py against the real Ultralytics 8.4.215 code (needs ultralytics + torch).
unit : 20 random scenarios (near-misses, wrong classes, duplicates, empty images, tied scores) scored by Ultralytics'
       box_iou + BaseValidator.match_predictions + DetMetrics and by offline_eval.evaluate -> must agree exactly.
e2e  : synthetic 10-class dataset, YOLOv8n fine-tuned 25 epochs on CPU, model.val(save_json=True), conversion by
       run_baseline.save_val_preds, re-scored offline -> must reproduce Ultralytics mAP50-95 / mAP50.
Result 2026-11-24 (cloud CPU): unit max |diff| 0.0 (mAP50-95, mAP50, per-class AP), 1e-8 (P, R);
e2e Ultralytics 0.709163 / 0.732417 vs offline 0.709163 / 0.732417 (3600 predictions, diff 0.0).
Usage: python scripts/evaluation/test_offline_eval_parity.py [--e2e]"""
import argparse, json, os, sys, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.offline_eval import evaluate  # noqa: E402
from common.taxonomy import CLASS_NAMES  # noqa: E402


def unit(trials=20, S=640.0):
    import torch
    from ultralytics.utils.metrics import box_iou, DetMetrics
    from ultralytics.engine.validator import BaseValidator
    v = type('V', (), {})(); v.iouv = torch.linspace(0.5, 0.95, 10)
    worst = {}
    for trial in range(trials):
        rng = np.random.default_rng(trial); gt, preds = {}, {}
        for i in range(60):
            name = f'test/images/im{i}.jpg'; g = []
            for _ in range(int(rng.integers(0, 4))):
                x1, y1 = rng.uniform(0, 0.7, 2); w, h = rng.uniform(0.03, 0.3, 2)
                g.append([int(rng.integers(0, 10)), x1, y1, min(1, x1 + w), min(1, y1 + h)])
            gt[name] = g; P = []
            for c, x1, y1, x2, y2 in g:
                for _ in range(int(rng.integers(0, 4))):
                    j = rng.normal(0, 0.02, 4); cc = c if rng.random() < 0.7 else int(rng.integers(0, 10))
                    P.append([cc, float(np.round(rng.uniform(0.001, 1), 3)), *np.clip([x1 + j[0], y1 + j[1], x2 + j[2], y2 + j[3]], 0, 1)])
            for _ in range(int(rng.integers(0, 5))):
                x1, y1 = rng.uniform(0, 0.8, 2); w, h = rng.uniform(0.02, 0.2, 2)
                P.append([int(rng.integers(0, 10)), float(np.round(rng.uniform(0.001, 1), 3)), x1, y1, min(1, x1 + w), min(1, y1 + h)])
            P.sort(key=lambda q: -q[1]); preds[name] = P
        dm = DetMetrics(names=dict(enumerate(CLASS_NAMES)))
        for name in preds:
            g = np.array(gt[name], dtype=np.float32).reshape(-1, 5); P = np.array(preds[name], dtype=np.float32).reshape(-1, 6)
            if len(g) and len(P):
                tp = BaseValidator.match_predictions(v, torch.tensor(P[:, 0]), torch.tensor(g[:, 0]),
                                                     box_iou(torch.tensor(g[:, 1:] * S), torch.tensor(P[:, 2:] * S))).numpy()
            else:
                tp = np.zeros((len(P), 10), dtype=bool)
            dm.update_stats({'tp': tp, 'conf': P[:, 1] if len(P) else np.zeros(0), 'pred_cls': P[:, 0] if len(P) else np.zeros(0),
                             'target_cls': g[:, 0], 'target_img': np.unique(g[:, 0]), 'im_name': name})
        dm.process(); O = evaluate(preds, gt, scale=S, keep_order=True)
        U = {'map5095': float(dm.box.map), 'map50': float(dm.box.map50), 'precision': float(dm.box.mp), 'recall': float(dm.box.mr)}
        for k in U: worst[k] = max(worst.get(k, 0.0), abs(U[k] - O[k]))
        Uc = {CLASS_NAMES[int(c)]: float(dm.box.ap[i]) for i, c in enumerate(dm.box.ap_class_index)}
        worst['per_class'] = max(worst.get('per_class', 0.0), max(abs(Uc[k] - O['per_class'][k]['ap5095']) for k in Uc))
    print('unit parity, max |ultralytics - offline| over', trials, 'trials:', {k: f'{x:.1e}' for k, x in worst.items()})
    assert worst['map5095'] < 1e-9 and worst['map50'] < 1e-9 and worst['per_class'] < 1e-9


def e2e():
    from PIL import Image, ImageDraw
    from ultralytics import YOLO
    from training.run_baseline import save_val_preds, extract
    W = Path(tempfile.mkdtemp()); rng = np.random.default_rng(1)
    cols = [tuple(int(x) for x in rng.integers(40, 255, 3)) for _ in range(10)]
    for split, n in [('train', 40), ('test', 12)]:
        (W / split / 'images').mkdir(parents=True); (W / split / 'labels').mkdir(parents=True)
        for i in range(n):
            im = Image.new('RGB', (640, 640), (20, 20, 20)); d = ImageDraw.Draw(im); L = []
            for _ in range(int(rng.integers(1, 4))):
                c = int(rng.integers(0, 10)); w, h = rng.integers(60, 220, 2); x, y = rng.integers(0, 640 - w), rng.integers(0, 640 - h)
                d.rectangle([x, y, x + w, y + h], fill=cols[c]); L.append(f'{c} {(x + w / 2) / 640:.6f} {(y + h / 2) / 640:.6f} {w / 640:.6f} {h / 640:.6f}')
            im.save(W / split / 'images' / f'{split}_{i:03d}.jpg'); (W / split / 'labels' / f'{split}_{i:03d}.txt').write_text('\n'.join(L) + '\n')
    (W / 'data.yaml').write_text(f"path: {W}\ntrain: train/images\nval: test/images\ntest: test/images\nnames: {CLASS_NAMES}\n")
    m = YOLO('yolov8n.pt')
    m.train(data=str(W / 'data.yaml'), epochs=25, imgsz=320, batch=8, device='cpu', workers=0, plots=False, verbose=False, project=str(W / 'runs'), name='tr', exist_ok=True, amp=False, seed=0)
    m = YOLO(str(W / 'runs/tr/weights/last.pt'))
    r = m.val(data=str(W / 'data.yaml'), split='test', imgsz=320, batch=4, conf=0.001, iou=0.7, device='cpu', workers=0, project=str(W / 'runs'), name='val_test', exist_ok=True, plots=False, verbose=False, save_json=True)
    ultra = extract(r); test_rel = sorted(f'test/images/{f}' for f in os.listdir(W / 'test/images'))
    par = save_val_preds(W / 'runs/val_test/predictions.json', test_rel, str(W), ultra, W / 'preds_test_val.json')
    print('e2e parity:', json.dumps(par)); assert par.get('abs_diff_map5095', 1) < 1e-6


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--e2e', action='store_true'); a = ap.parse_args()
    unit()
    if a.e2e:
        e2e()
