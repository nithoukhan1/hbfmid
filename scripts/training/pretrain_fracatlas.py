#!/usr/bin/env python3
"""Pre-train a detector on FracAtlas (single 'fracture' class) for arm 6 (pre-train -> fine-tune on HBFMID).

Input: a prepared folder from scripts/data/prepare_fracatlas_yolo.py run with --cls-id 0 (lists/train.txt, lists/val.txt).
Fixed epochs (no early stopping), last.pt (EMA) is the checkpoint handed to run_baseline.py --init-weights.
Output: <out>/<tag>/weights_last.pt (copy, ~22 MB for yolov8s, not for git) and <out>/<tag>/metrics.json (FracAtlas val mAP, monitoring only).
FracAtlas and HBFMID share no image (cross-dataset audit), so this cannot leak into the HBFMID test folds.
"""
import argparse
import json
import shutil
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--model', default='yolov8s.pt')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tag', required=True)
    a = ap.parse_args()

    d, run = Path(a.data_dir).resolve(), Path(a.out).resolve() / a.tag
    if (run / 'metrics.json').exists():
        print(f'[skip] {a.tag}'); return
    run.mkdir(parents=True, exist_ok=True)
    yaml = run / 'data.yaml'
    yaml.write_text(f"train: {d / 'lists/train.txt'}\nval: {d / 'lists/val.txt'}\nnc: 1\nnames:\n  0: fracture\n")

    from ultralytics import YOLO
    t0 = time.time()
    m = YOLO(a.model)
    m.train(data=str(yaml), epochs=a.epochs, patience=a.epochs, imgsz=a.imgsz, batch=a.batch, seed=a.seed, deterministic=True,
            workers=a.workers, device=a.device, project=str(run), name='train', exist_ok=True, plots=False)
    last = run / 'train' / 'weights' / 'last.pt'
    shutil.copy(last, run / 'weights_last.pt')
    mm = YOLO(str(last)).val(data=str(yaml), split='val', imgsz=a.imgsz, batch=a.batch, conf=0.001, iou=0.7, device=a.device,
                              workers=a.workers, project=str(run), name='val', exist_ok=True, plots=False, verbose=False)
    res = {'tag': a.tag, 'model': a.model, 'epochs': a.epochs, 'seed': a.seed, 'train_seconds': round(time.time() - t0, 1),
           'fa_val_map50': float(mm.box.map50), 'fa_val_map5095': float(mm.box.map), 'data_dir': str(d)}
    (run / 'metrics.json').write_text(json.dumps(res, indent=1))
    shutil.rmtree(run / 'train' / 'weights', ignore_errors=True)
    print(res)


if __name__ == '__main__':
    main()
