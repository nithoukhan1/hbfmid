#!/usr/bin/env python3
"""E1 / E2 baseline: train one YOLO model on one split and write metrics.json.

  --mode official      : official Roboflow split (E1), also runs the leak probe on the official test set
  --mode clean --fold k: HBFMID-Clean fold k (E2)

Everything is driven by the txt lists in data_splits/ (paths relative to the dataset root).
The dataset folder is NEVER written to: lists/yaml/outputs go to --out. Because Ultralytics would
write a labels.cache next to the labels, the script refuses to train when the dataset folder is
writable (Kaggle /kaggle/input is read-only) unless --allow-writable-data is given.
"""
import argparse
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.taxonomy import CLASS_NAMES, NC, ELIGIBLE_IDS  # noqa: E402
from evaluation.confusion import load_gt, build_confusion, error_stats, matrix_to_csv  # noqa: E402


def read_list(p):
    return [l.strip() for l in Path(p).read_text().splitlines() if l.strip()]


def label_of(root, rel):
    return Path(root) / (rel.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt')


def split_files(splits_dir, mode, fold):
    if mode == 'official':
        d = Path(splits_dir) / 'official'
        f = {'train': d / 'train.txt', 'val': d / 'val.txt', 'test': d / 'test.txt'}
        probe = {'test_with_train_copy': d / 'test_with_train_copy.txt',
                 'test_no_train_copy': d / 'test_no_train_copy.txt'}
    elif mode == 'official_dedup':
        d = Path(splits_dir) / 'official_dedup'
        f = {r: d / f'{r}.txt' for r in ['train', 'val', 'test']}
        probe = {}
    else:
        d = Path(splits_dir) / 'clean5'
        f = {r: d / f'fold{fold}_{r}.txt' for r in ['train', 'val', 'test']}
        probe = {}
    return f, probe


def materialize(lists, data_root, dest, name):
    """Write absolute-path list; return its path."""
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / f'{name}.txt'
    out.write_text('\n'.join(str(Path(data_root) / r) for r in read_list(lists)) + '\n')
    return out


def write_yaml(path, train, val, test):
    names = '\n'.join(f'  {i}: {n}' for i, n in enumerate(CLASS_NAMES))
    path.write_text(f"train: {train}\nval: {val}\ntest: {test}\nnc: {NC}\nnames:\n{names}\n")


def extract(m):
    box = m.box
    idx = [int(i) for i in box.ap_class_index]
    ap, ap50 = box.ap, box.ap50
    per = {CLASS_NAMES[c]: {'ap50': float(ap50[i]), 'ap5095': float(ap[i])} for i, c in enumerate(idx)}
    elig = [float(ap[i]) for i, c in enumerate(idx) if c in ELIGIBLE_IDS]
    return {'map50': float(box.map50), 'map5095': float(box.map),
            'precision': float(box.mp), 'recall': float(box.mr), 'per_class': per,
            'classes_present': [CLASS_NAMES[c] for c in idx],
            'macro_eligible_ap5095': (sum(elig) / len(elig)) if elig else None,
            'n_eligible_present': len(elig)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['official', 'official_dedup', 'clean'], required=True)
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--model', default='yolov8s.pt')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--patience', type=int, default=30)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--device', default=None)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--splits-dir', default=str(Path(__file__).resolve().parents[2] / 'data_splits'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--tag', default=None)
    ap.add_argument('--conf-thr', type=float, default=0.25)
    ap.add_argument('--fixed-epochs', action='store_true',
                    help='no early stopping (patience=epochs); primary metrics from last.pt (EMA, no val-based selection), best.pt kept as *_best')
    ap.add_argument('--keep-weights', action='store_true')
    ap.add_argument('--allow-writable-data', action='store_true')
    ap.add_argument('--dry-run', action='store_true', help='write lists/yaml and exit (no ultralytics)')
    a = ap.parse_args()

    stem = Path(a.model).stem
    base = {'official': f'official_{stem}_s{a.seed}', 'official_dedup': f'officialdedup_{stem}_s{a.seed}',
            'clean': f'clean_f{a.fold}_{stem}_s{a.seed}'}[a.mode]
    tag = a.tag or (base + ('_fx' if a.fixed_epochs else ''))
    run = Path(a.out).resolve() / tag
    root = Path(a.data_root).resolve()
    assert root not in run.parents and run != root, 'output must not be inside the dataset folder'
    if (run / 'metrics.json').exists():
        print(f'[skip] {tag}: metrics.json exists')
        return
    run.mkdir(parents=True, exist_ok=True)

    files, probes = split_files(a.splits_dir, a.mode, a.fold)
    lists = {r: materialize(p, root, run / 'lists', r) for r, p in files.items()}
    yaml_main = run / 'data.yaml'
    write_yaml(yaml_main, lists['train'], lists['val'], lists['test'])
    probe_yaml = {}
    for name, p in probes.items():
        lp = materialize(p, root, run / 'lists', name)
        y = run / f'data_{name}.yaml'
        write_yaml(y, lists['train'], lists['val'], lp)
        probe_yaml[name] = y
    n_img = {r: len(read_list(p)) for r, p in files.items()}
    print(f'[{tag}] images: {n_img}  probes: {list(probe_yaml)}')
    if a.dry_run:
        print('[dry-run] lists and yaml written to', run)
        return

    if os.access(root, os.W_OK) and not a.allow_writable_data:
        sys.exit('REFUSING: dataset folder is writable; Ultralytics would write labels.cache into it. '
                 'Run on Kaggle (/kaggle/input is read-only) or pass --allow-writable-data knowingly.')

    import torch
    import ultralytics
    from ultralytics import YOLO

    t0 = time.time()
    model = YOLO(a.model)
    model.train(data=str(yaml_main), epochs=a.epochs, patience=(a.epochs if a.fixed_epochs else a.patience), imgsz=a.imgsz, batch=a.batch,
                seed=a.seed, deterministic=True, workers=a.workers, device=a.device,
                project=str(run), name='train', exist_ok=True, plots=False)
    train_s = time.time() - t0
    wdir = run / 'train' / 'weights'
    model = YOLO(str(wdir / ('last.pt' if a.fixed_epochs else 'best.pt')))

    def run_val(yaml_path, split, name, m=None):
        return extract((m or model).val(data=str(yaml_path), split=split, imgsz=a.imgsz, batch=a.batch, conf=0.001,
                                 iou=0.7, device=a.device, workers=a.workers, project=str(run), name=name,
                                 exist_ok=True, plots=False, verbose=False))

    res = {'tag': tag, 'mode': a.mode, 'fold': a.fold if a.mode == 'clean' else None, 'model': a.model,
           'seed': a.seed, 'epochs_cfg': a.epochs, 'patience': (a.epochs if a.fixed_epochs else a.patience),
           'protocol': 'fixed_last' if a.fixed_epochs else 'earlystop_best', 'imgsz': a.imgsz, 'batch': a.batch,
           'n_images': n_img, 'train_seconds': round(train_s, 1),
           'ultralytics': ultralytics.__version__, 'torch': torch.__version__, 'python': platform.python_version(),
           'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}
    try:
        import pandas as pd
        df = pd.read_csv(run / 'train' / 'results.csv')
        df.columns = [c.strip() for c in df.columns]
        fit = 0.1 * df['metrics/mAP50(B)'] + 0.9 * df['metrics/mAP50-95(B)']
        res['epochs_run'] = int(len(df))
        res['best_epoch'] = int(fit.idxmax()) + 1
    except Exception as e:  # noqa: BLE001
        res['results_csv_error'] = str(e)
    res['val'] = run_val(yaml_main, 'val', 'val_val')
    res['test'] = run_val(yaml_main, 'test', 'val_test')
    for name, y in probe_yaml.items():
        res['probe_' + name] = run_val(y, 'test', 'probe_' + name)
    if a.fixed_epochs:   # secondary: validation-selected checkpoint, to compare noise
        mb = YOLO(str(wdir / 'best.pt'))
        res['val_best'] = run_val(yaml_main, 'val', 'valbest_val', mb)
        res['test_best'] = run_val(yaml_main, 'test', 'valbest_test', mb)

    # predictions on test + confusion (Gate A2)
    test_rel = read_list(files['test'])
    abs2rel = {str(root / r): r for r in test_rel}
    preds = {r: [] for r in test_rel}
    for r in model.predict(source=list(abs2rel), conf=0.05, iou=0.7, imgsz=a.imgsz, batch=a.batch,
                           stream=True, verbose=False, device=a.device, save=False):
        rel = abs2rel.get(str(Path(r.path)))
        if rel is None:
            continue
        b = r.boxes
        xy, cf, cl = b.xyxyn.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int)
        preds[rel] = [[int(c), float(s), *map(float, x)] for c, s, x in zip(cl, cf, xy)]
    (run / 'preds_test.json').write_text(json.dumps(preds))
    gt = {r: load_gt(label_of(root, r)) for r in test_rel}
    M = build_confusion(preds, gt, a.conf_thr, 0.5)
    matrix_to_csv(M, run / f'confusion_test_conf{a.conf_thr}.csv')
    res['confusion_conf_thr'] = a.conf_thr
    res['confusion'] = M.tolist()
    res['error_stats'] = error_stats(M)
    res['test_gt_boxes_per_class'] = {CLASS_NAMES[c]: sum(1 for g in gt.values() for b in g if b[0] == c)
                                      for c in range(NC)}
    (run / 'metrics.json').write_text(json.dumps(res, indent=2))
    if not a.keep_weights:
        shutil.rmtree(run / 'train' / 'weights', ignore_errors=True)
    print(f"[{tag}] test mAP50-95={res['test']['map5095']:.4f} macro_eligible={res['test']['macro_eligible_ap5095']}")


if __name__ == '__main__':
    main()
