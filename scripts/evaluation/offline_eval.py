#!/usr/bin/env python3
"""Shared offline box evaluator: numpy re-implementation of the Ultralytics 8.4.215 detection metrics
(DetectionValidator._process_batch, BaseValidator.match_predictions with the default greedy matching,
metrics.ap_per_class / compute_ap / smooth), so detectors trained outside Ultralytics (RF-DETR) are scored
exactly like YOLO and RT-DETR.

Predictions: {rel_path: [[cls, conf, x1, y1, x2, y2], ...]} with normalised xyxy. Ground truth: YOLO label files.
Parity: run_baseline.py --save-val-preds keeps the predictions Ultralytics' own validator scored (preds_test_val.json)
and stores the offline result next to the Ultralytics one in metrics.json['test_offline_parity'].

Usage:
  python scripts/evaluation/offline_eval.py --selftest
  python scripts/evaluation/offline_eval.py --runs results/v2_A results/v2_B --preds preds_test.json --data-root "<.../Bone Fractures Detection>"
"""
import argparse, glob, json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common.taxonomy import CLASS_NAMES, ELIGIBLE_IDS  # noqa: E402
from evaluation.confusion import load_gt  # noqa: E402

IOUV = np.linspace(0.5, 0.95, 10).astype(np.float32)
AP_X = np.linspace(0, 1, 101)
TRAPZ = getattr(np, 'trapezoid', None) or np.trapz


def box_iou(a, b, eps=1e-7):
    """IoU of every box in a (L, 4) with every box in b (D, 4), xyxy, float32 like Ultralytics."""
    a = np.asarray(a, dtype=np.float32); b = np.asarray(b, dtype=np.float32)
    a1, a2 = a[:, None, :2], a[:, None, 2:]; b1, b2 = b[None, :, :2], b[None, :, 2:]
    inter = np.clip(np.minimum(a2, b2) - np.maximum(a1, b1), 0, None).prod(2)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def match_predictions(pred_cls, true_cls, iou):
    """Greedy matching of Ultralytics 8.4.215 (detections must be in descending confidence order)."""
    correct = np.zeros((pred_cls.shape[0], len(IOUV)), dtype=bool)
    iou = iou * (true_cls[:, None] == pred_cls[None, :])
    matched = np.zeros((iou.shape[0], len(IOUV)), dtype=bool)
    cols = np.arange(len(IOUV))
    for j in np.flatnonzero((iou >= IOUV.min()).any(0)):
        available = np.where(matched, 0, iou[:, j, None])
        k = available.argmax(0)
        correct[j] = available[k, cols] >= IOUV
        matched[k, cols] |= correct[j]
    return correct


def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 1.0], [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    return TRAPZ(np.interp(AP_X, mrec, mpre), AP_X)


def smooth(y, f=0.05):
    nf = round(len(y) * f * 2) // 2 + 1
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')


def ap_per_class(tp, conf, pred_cls, target_cls, eps=1e-16):
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]
    x = np.linspace(0, 1, 1000)
    ap, p_curve, r_curve = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l, n_p = nt[ci], i.sum()
        if n_p == 0 or n_l == 0:
            continue
        fpc = (1 - tp[i]).cumsum(0); tpc = tp[i].cumsum(0)
        recall = tpc / (n_l + eps)
        r_curve[ci] = np.interp(-x, -conf[i], recall[:, 0], left=0)
        precision = tpc / (tpc + fpc)
        p_curve[ci] = np.interp(-x, -conf[i], precision[:, 0], left=1)
        for j in range(tp.shape[1]):
            ap[ci, j] = compute_ap(recall[:, j], precision[:, j])
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
    k = smooth(f1_curve.mean(0), 0.1).argmax() if nc else 0
    return p_curve[:, k], r_curve[:, k], ap, unique_classes.astype(int)


def evaluate(preds, gt, conf_min=None, max_det=None, keep_order=False, scale=640.0, names=CLASS_NAMES, eligible=ELIGIBLE_IDS):
    """Return a dict with the same keys as run_baseline.extract(): map50, map5095, precision, recall, per_class,
    classes_present, macro_eligible_ap5095, n_eligible_present. Images are processed in the order of `preds`
    (then images that only have ground truth). keep_order=True keeps each image's prediction order as given
    (use for predictions exported by the Ultralytics validator); otherwise predictions are sorted by confidence."""
    tps, confs, pcls, tcls = [], [], [], []
    for k in list(preds.keys()) + [k for k in gt if k not in preds]:
        g = np.array(gt.get(k, []), dtype=np.float64).reshape(-1, 5)
        P = np.array(preds.get(k, []), dtype=np.float64).reshape(-1, 6)
        if conf_min is not None:
            P = P[P[:, 1] >= conf_min]
        if not keep_order and len(P):
            P = P[np.argsort(-P[:, 1], kind='stable')]
        if max_det is not None:
            P = P[:max_det]
        t = g[:, 0].astype(int)
        if len(g) and len(P):
            tp = match_predictions(P[:, 0].astype(int), t, box_iou(g[:, 1:] * scale, P[:, 2:] * scale))
        else:
            tp = np.zeros((len(P), len(IOUV)), dtype=bool)
        tps.append(tp); confs.append(P[:, 1]); pcls.append(P[:, 0]); tcls.append(t)
    tp, conf, pc, tc = (np.concatenate(v, 0) for v in (tps, confs, pcls, tcls))
    if not len(tc):
        return {'map50': 0.0, 'map5095': 0.0, 'precision': 0.0, 'recall': 0.0, 'per_class': {}, 'classes_present': [],
                'macro_eligible_ap5095': None, 'n_eligible_present': 0}
    p, r, ap, idx = ap_per_class(tp, conf, pc, tc)
    per = {names[c]: {'ap50': float(ap[i, 0]), 'ap5095': float(ap[i].mean())} for i, c in enumerate(idx)}
    elig = [float(ap[i].mean()) for i, c in enumerate(idx) if c in eligible]
    return {'map50': float(ap[:, 0].mean()), 'map5095': float(ap.mean()), 'precision': float(p.mean()), 'recall': float(r.mean()),
            'per_class': per, 'classes_present': [names[c] for c in idx],
            'macro_eligible_ap5095': (sum(elig) / len(elig)) if elig else None, 'n_eligible_present': len(elig)}


def gt_for(rels, data_root):
    return {r: load_gt(Path(data_root) / (r.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt')) for r in rels}


def selftest():
    rng = np.random.default_rng(0)
    gt = {f'img{i}.jpg': [(int(c), *b) for c, b in zip(rng.integers(0, 10, 3), [(0.1, 0.1, 0.3, 0.3), (0.5, 0.5, 0.7, 0.8), (0.2, 0.6, 0.4, 0.9)])] for i in range(20)}
    perfect = {k: [[c, 0.9, *b] for c, *b in v] for k, v in gt.items()}
    # Ultralytics caps AP at 0.995: compute_ap's end sentinel puts precision 0 at recall exactly 1 (101-point trapezoid),
    # the familiar 'mAP50 0.995' of Ultralytics logs; the replica must show the same.
    assert abs(evaluate(perfect, gt)['map5095'] - 0.995) < 1e-6, 'perfect predictions must give 0.995 (Ultralytics cap)'
    wrong = {k: [[(c + 1) % 10, 0.9, *b] for c, *b in v] for k, v in gt.items()}
    assert evaluate(wrong, gt)['map5095'] == 0.0, 'wrong classes must give 0'
    assert evaluate({}, gt)['map5095'] == 0.0, 'no predictions must give 0'
    shifted = {k: [[c, 0.9, b[0] + 0.02, b[1], b[2] + 0.02, b[3]] for c, *b in v] for k, v in gt.items()}
    s = evaluate(shifted, gt); assert 0.0 < s['map5095'] < s['map50'] <= 1.0, 'shifted boxes: AP50 high, AP50-95 lower'
    extra_fp = {k: v + [[(v[0][0] + 3) % 10, 0.95, 0.6, 0.1, 0.8, 0.3]] for k, v in perfect.items()}
    assert evaluate(extra_fp, gt)['map5095'] < 1.0, 'high-confidence false positives must lower AP'
    print('selftest OK: perfect=0.995 (Ultralytics cap), wrong-class=0, empty=0, shifted AP50 %.3f / AP50-95 %.3f' % (s['map50'], s['map5095']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true'); ap.add_argument('--runs', nargs='*', default=[])
    ap.add_argument('--preds', default='preds_test.json'); ap.add_argument('--data-root')
    ap.add_argument('--keep-order', action='store_true'); ap.add_argument('--conf-min', type=float); ap.add_argument('--out')
    a = ap.parse_args()
    if a.selftest:
        selftest()
    rows = []
    for d in a.runs:
        for mp in sorted(glob.glob(os.path.join(d, '*', 'metrics.json'))):
            pp = Path(mp).parent / a.preds
            if not pp.exists():
                continue
            m = json.load(open(mp)); P = json.load(open(pp))
            o = evaluate(P, gt_for(list(P.keys()), a.data_root), conf_min=a.conf_min, keep_order=a.keep_order)
            rows.append({'run': Path(mp).parent.name, 'model': m['model'], 'ultra': m['test']['map5095'], 'offline': o['map5095'],
                         'ultra_macro': m['test'].get('macro_eligible_ap5095'), 'offline_macro': o['macro_eligible_ap5095']})
    if rows:
        d = np.array([r['offline'] - r['ultra'] for r in rows])
        print(f'{len(rows)} runs, preds={a.preds}: offline - ultralytics mAP50-95 mean {100 * d.mean():+.3f} pt, '
              f'min {100 * d.min():+.3f}, max {100 * d.max():+.3f}, mean |diff| {100 * np.abs(d).mean():.3f} pt')
        if a.out:
            Path(a.out).write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
