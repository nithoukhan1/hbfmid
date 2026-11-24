#!/usr/bin/env python3
"""RF-DETR (DINOv2 backbone) run under the same protocol and with the same outputs as run_baseline.py.

1. COCO-format copy of the requested split built from the SAME list files as the YOLO runs (images symlinked, never written
   into the dataset folder; categories 1..10 = CLASS_NAMES, mapped by RF-DETR to label indices 0..9).
2. Training: fixed epochs, no early stopping, seed, EMA on, package defaults otherwise (rfdetr 1.11.2, see docs/BENCHMARK_V2_PLAN.md).
3. Inference with the last-epoch EMA weights (last_ema.pth), threshold 0.001, top-300 (RF-DETR num_select), in chunks.
4. Scoring with the shared evaluator scripts/evaluation/offline_eval.py (exact replica of the Ultralytics 8.4.215 metric,
   verified in scripts/evaluation/test_offline_eval_parity.py), so the numbers sit on the same scale as YOLO / RT-DETR.
5. metrics.json with the run_baseline.py schema, preds_test.json (conf >= 0.05, like the YOLO runs), preds_test_c001.json (all
   predictions), confusion at conf 0.25. Checkpoints and the COCO copy are deleted unless --keep-weights.

Usage: python scripts/rfdetr_bench/train_rfdetr.py --mode clean --fold 0 --seed 42 --data-root <...> --splits-dir data_splits_v2 --out <dir> [--dry-run]
"""
import argparse, json, os, platform, shutil, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.taxonomy import CLASS_NAMES, NC  # noqa: E402
from evaluation.confusion import load_gt, build_confusion, error_stats, matrix_to_csv  # noqa: E402
from evaluation.offline_eval import evaluate  # noqa: E402


def read_list(p):
    return [l.strip() for l in Path(p).read_text().splitlines() if l.strip()]


def label_of(root, rel):
    return Path(root) / (rel.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt')


def split_files(splits_dir, mode, fold):
    if mode == 'official':
        d = Path(splits_dir) / 'official'
        return ({'train': d / 'train.txt', 'val': d / 'val.txt', 'test': d / 'test.txt'},
                {'test_with_train_copy': d / 'test_with_train_copy.txt', 'test_no_train_copy': d / 'test_no_train_copy.txt'})
    d = Path(splits_dir) / 'clean5'
    return {r: d / f'fold{fold}_{r}.txt' for r in ['train', 'val', 'test']}, {}


def build_coco(rels, root, split_dir):
    """COCO json + symlinked images for one split. Category ids 1..10 (COCO style) = CLASS_NAMES order."""
    from PIL import Image
    split_dir.mkdir(parents=True, exist_ok=True)
    images, anns, aid = [], [], 1
    for i, rel in enumerate(rels, start=1):
        src = Path(root) / rel; fname = rel.replace('/', '__'); dst = split_dir / fname
        if not dst.exists():
            os.symlink(src, dst)
        w, h = Image.open(src).size
        images.append({'id': i, 'file_name': fname, 'width': w, 'height': h})
        for c, x1, y1, x2, y2 in load_gt(label_of(root, rel)):
            bw, bh = (x2 - x1) * w, (y2 - y1) * h
            anns.append({'id': aid, 'image_id': i, 'category_id': int(c) + 1, 'bbox': [x1 * w, y1 * h, bw, bh], 'area': bw * bh, 'iscrowd': 0})
            aid += 1
    cats = [{'id': k + 1, 'name': n, 'supercategory': 'fracture'} for k, n in enumerate(CLASS_NAMES)]
    (split_dir / '_annotations.coco.json').write_text(json.dumps({'images': images, 'annotations': anns, 'categories': cats}))
    return {'images': len(images), 'boxes': len(anns)}


def predict_all(model, rels, root, chunk=8, threshold=0.001):
    """{rel: [[cls, conf, x1, y1, x2, y2] normalised, sorted by conf]} plus the number of no-object (background-slot) entries dropped."""
    from PIL import Image
    raw = {}
    for k in range(0, len(rels), chunk):
        part = rels[k:k + chunk]
        dets = model.predict([str(Path(root) / r) for r in part], threshold=threshold)
        dets = dets if isinstance(dets, list) else [dets]
        for r, d in zip(part, dets):
            w, h = Image.open(Path(root) / r).size
            rows = [[int(c), float(s), float(b[0]) / w, float(b[1]) / h, float(b[2]) / w, float(b[3]) / h]
                    for b, s, c in zip(d.xyxy, d.confidence, d.class_id)]
            raw[r] = sorted(rows, key=lambda q: -q[1])
    # Fine-tuned RF-DETR returns 0-based class ids. Its head has one extra logit, the no-object slot at index NC
    # (predict() names it "__background__"); at threshold 0.001 some of the top-300 entries land there. They are not
    # class predictions: drop them, as RF-DETR's own COCO evaluator does (label without a category -> ignored).
    bad = {q[0] for v in raw.values() for q in v if not 0 <= q[0] <= NC}
    assert not bad, f'unexpected class ids {sorted(bad)[:10]} (expected 0..{NC - 1} plus the no-object slot {NC})'
    n_bg = sum(1 for v in raw.values() for q in v if q[0] == NC)
    return {r: [q for q in v if q[0] < NC] for r, v in raw.items()}, n_bg


def log_train_seconds(log_file):
    """Recovered runs: training time from the timestamps of the rf-detr log lines (first line .. last line before a traceback)."""
    import re
    from datetime import datetime
    if not log_file or not Path(log_file).exists():
        return None
    ts = []
    for line in Path(log_file).read_text(errors='ignore').splitlines():
        if line.startswith('Traceback'):
            break
        mt = re.match(r'^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\]', line)
        if mt:
            ts.append(datetime.strptime(mt.group(1), '%Y-%m-%d %H:%M:%S'))
    return (ts[-1] - ts[0]).total_seconds() if len(ts) > 1 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['official', 'clean'], required=True)
    ap.add_argument('--fold', type=int, default=0); ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--size', default='medium', choices=['nano', 'small', 'medium', 'large'])
    ap.add_argument('--epochs', type=int, default=100); ap.add_argument('--resolution', type=int, default=640)
    ap.add_argument('--batch', type=int, default=4); ap.add_argument('--accum', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--lr-encoder', type=float, default=1.5e-4)
    ap.add_argument('--workers', type=int, default=2); ap.add_argument('--eval-interval', type=int, default=10)
    ap.add_argument('--data-root', required=True); ap.add_argument('--splits-dir', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--conf-thr', type=float, default=0.25); ap.add_argument('--keep-weights', action='store_true')
    ap.add_argument('--dry-run', action='store_true', help='build the COCO copy and exit (no rfdetr needed)')
    ap.add_argument('--pretrain-weights', default=None, help='testing only: "none" = random init (default: the package checkpoint)')
    ap.add_argument('--device-check', action='store_true', help='testing only: allow CPU')
    ap.add_argument('--predict-only', action='store_true', help='skip training: score an already trained run from its last_ema.pth')
    ap.add_argument('--ckpt-root', default=None, help='predict-only: folder holding <tag>/train/last_ema.pth (default: --out)')
    ap.add_argument('--log-file', default=None, help='predict-only: notebook log of the training run (training time)')
    a = ap.parse_args()

    model_name = f'rfdetr-{a.size}'
    tag = (f'official_{model_name}_s{a.seed}' if a.mode == 'official' else f'clean_f{a.fold}_{model_name}_s{a.seed}') + '_fx'
    root = Path(a.data_root).resolve(); run = Path(a.out).resolve() / tag
    assert root not in run.parents and run != root, 'output must not be inside the dataset folder'
    if (run / 'metrics.json').exists():
        print(f'[skip] {tag}: metrics.json exists'); return
    run.mkdir(parents=True, exist_ok=True)
    files, probes = split_files(a.splits_dir, a.mode, a.fold)
    lists = {r: read_list(p) for r, p in files.items()}
    (run / 'lists').mkdir(exist_ok=True)
    for r, L in lists.items():
        (run / 'lists' / f'{r}.txt').write_text('\n'.join(L) + '\n')
    ds = run / 'coco'
    counts = None if a.predict_only else {('valid' if r == 'val' else r): build_coco(L, root, ds / ('valid' if r == 'val' else r)) for r, L in lists.items()}
    print(f'[{tag}] COCO copy: {counts}')
    if a.dry_run:
        print('[dry-run] COCO copy written to', ds); return

    import torch
    from importlib.metadata import version as _version
    from rfdetr import RFDETR, RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge
    cls = {'nano': RFDETRNano, 'small': RFDETRSmall, 'medium': RFDETRMedium, 'large': RFDETRLarge}[a.size]
    if not a.device_check:
        assert torch.cuda.is_available(), 'no GPU'
    if a.predict_only:
        src = (Path(a.ckpt_root).resolve() / tag) if a.ckpt_root else run
        ck = src / 'train' / 'last_ema.pth'
        (run / 'train').mkdir(exist_ok=True)
        for f in ('metrics.csv', 'training_config.json'):
            if src != run and (src / 'train' / f).exists():
                shutil.copy(src / 'train' / f, run / 'train' / f)
        train_s, mem_gb, train_src = log_train_seconds(a.log_file), None, 'recovered: rf-detr log timestamps'
    else:
        t0 = time.time()
        kw = {} if a.pretrain_weights is None else {'pretrain_weights': None if a.pretrain_weights.lower() == 'none' else a.pretrain_weights}
        model = cls(resolution=a.resolution, **kw)
        model.train(dataset_dir=str(ds), output_dir=str(run / 'train'), epochs=a.epochs, batch_size=a.batch, grad_accum_steps=a.accum,
                    lr=a.lr, lr_encoder=a.lr_encoder, seed=a.seed, use_ema=True, early_stopping=False, num_workers=a.workers,
                    eval_interval=a.eval_interval, checkpoint_interval=a.epochs, tensorboard=False, progress_bar=None,
                    class_names=list(CLASS_NAMES), run_test=False)
        train_s, train_src = time.time() - t0, 'measured'
        mem_gb = round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None
        ck = run / 'train' / 'last_ema.pth'
    assert ck.exists(), f'missing {ck}: {sorted(p.name for p in (run / "train").iterdir())}'
    try:
        m = RFDETR.from_checkpoint(str(ck))
    except Exception:  # noqa: BLE001  (our own checkpoint: trusted)
        m = RFDETR.from_checkpoint(str(ck), trust_checkpoint=True)

    res = {'tag': tag, 'arm': 'baseline', 'mode': a.mode, 'fold': a.fold if a.mode == 'clean' else None, 'model': model_name, 'seed': a.seed,
           'epochs_cfg': a.epochs, 'patience': a.epochs, 'protocol': 'fixed_last', 'checkpoint': 'last_ema.pth', 'imgsz': a.resolution,
           'batch': a.batch * a.accum, 'batch_physical': a.batch, 'grad_accum': a.accum, 'lr': a.lr, 'lr_encoder': a.lr_encoder,
           'amp': 'auto (fp16 on T4)', 'n_images': {r: len(L) for r, L in lists.items()}, 'coco_copy': counts, 'train_seconds': None if train_s is None else round(train_s, 1), 'train_seconds_source': train_src, 'gpu_mem_peak_gb': mem_gb,
           'recovered_from': str(ck) if a.predict_only else None,
           'rfdetr': _version('rfdetr'), 'pretrain_weights': a.pretrain_weights or 'package default', 'torch': torch.__version__, 'python': platform.python_version(),
           'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
           'evaluator': 'scripts/evaluation/offline_eval.py (replica of the Ultralytics 8.4.215 box metric)'}
    gt = {r: load_gt(label_of(root, r)) for r in set(lists['val']) | set(lists['test'])}
    for split in ['val', 'test']:
        P, n_bg = predict_all(m, lists[split], root)
        res[f'no_object_preds_dropped_{split}'] = n_bg
        res[split] = evaluate(P, {r: gt[r] for r in lists[split]})
        if split == 'test':
            (run / 'preds_test_c001.json').write_text(json.dumps({r: [[q[0], round(q[1], 5), *[round(x, 6) for x in q[2:]]] for q in v] for r, v in P.items()}))
            p05 = {r: [q for q in v if q[1] >= 0.05] for r, v in P.items()}
            (run / 'preds_test.json').write_text(json.dumps(p05))
            res['test_conf005'] = evaluate(p05, {r: gt[r] for r in lists[split]})
            for name, pf in probes.items():
                sub = [r for r in read_list(pf) if r in P]
                res['probe_' + name] = evaluate({r: P[r] for r in sub}, {r: gt[r] for r in sub})
            M = build_confusion(p05, {r: gt[r] for r in lists['test']}, a.conf_thr, 0.5)
            matrix_to_csv(M, run / f'confusion_test_conf{a.conf_thr}.csv')
            res.update({'confusion_conf_thr': a.conf_thr, 'confusion': M.tolist(), 'error_stats': error_stats(M),
                        'test_gt_boxes_per_class': {CLASS_NAMES[c]: sum(1 for r in lists['test'] for b in gt[r] if b[0] == c) for c in range(NC)}})
    (run / 'metrics.json').write_text(json.dumps(res, indent=2))
    shutil.rmtree(ds, ignore_errors=True)
    if not a.keep_weights and not a.predict_only:
        for p in (run / 'train').glob('*'):
            if p.suffix in ('.pth', '.ckpt', '.pt'):
                p.unlink()
    print(f"[{tag}] test mAP50-95={res['test']['map5095']:.4f} mAP50={res['test']['map50']:.4f} macro_eligible={res['test']['macro_eligible_ap5095']} train_min={'n/a' if train_s is None else round(train_s / 60, 1)}")


if __name__ == '__main__':
    main()
