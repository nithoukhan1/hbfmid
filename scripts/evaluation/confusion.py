"""Detection confusion matrix + sibling-confusion statistics (Gate A2).

Matrix orientation: M[true, pred]; index NC = background.
  M[i, j]   (i, j < NC, i != j) : localized correctly (IoU >= iou_thr) but wrong class
  M[i, NC]                      : missed ground truth (FN)
  M[NC, j]                      : false positive
Matching is greedy by descending confidence and is class-agnostic (best-IoU unmatched GT).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.taxonomy import NC, CLASS_NAMES, is_sibling, is_same_base  # noqa: E402


def load_gt(label_path):
    """YOLO txt -> list of (cls, x1, y1, x2, y2) normalized."""
    out = []
    p = Path(label_path)
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        t = line.split()
        if len(t) < 5:
            continue
        c, x, y, w, h = int(float(t[0])), *map(float, t[1:5])
        out.append((c, x - w / 2, y - h / 2, x + w / 2, y + h / 2))
    return out


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def confusion_for_image(gt, preds, conf_thr, iou_thr, M):
    """gt: [(c,x1,y1,x2,y2)], preds: [[c,conf,x1,y1,x2,y2]] ; updates M in place."""
    used = [False] * len(gt)
    for p in sorted([p for p in preds if p[1] >= conf_thr], key=lambda r: -r[1]):
        pc, pb = int(p[0]), p[2:6]
        best, bi = 0.0, -1
        for gi, g in enumerate(gt):
            if used[gi]:
                continue
            v = iou(pb, g[1:5])
            if v > best:
                best, bi = v, gi
        if bi >= 0 and best >= iou_thr:
            used[bi] = True
            M[gt[bi][0], pc] += 1
        else:
            M[NC, pc] += 1
    for gi, g in enumerate(gt):
        if not used[gi]:
            M[g[0], NC] += 1


def build_confusion(preds_by_image, gt_by_image, conf_thr=0.25, iou_thr=0.5):
    M = np.zeros((NC + 1, NC + 1), dtype=int)
    for img, gt in gt_by_image.items():
        confusion_for_image(gt, preds_by_image.get(img, []), conf_thr, iou_thr, M)
    return M


def error_stats(M):
    """Gate A2 numbers from a (pooled) confusion matrix."""
    M = np.asarray(M)
    correct = int(np.trace(M[:NC, :NC]))
    wrong = int(M[:NC, :NC].sum() - correct)
    fn = int(M[:NC, NC].sum())
    fp = int(M[NC, :NC].sum())
    sib = base = 0
    for i in range(NC):
        for j in range(NC):
            if i != j and M[i, j]:
                if is_sibling(i, j):
                    sib += int(M[i, j])
                if is_same_base(i, j):
                    base += int(M[i, j])
    total_err = wrong + fn + fp
    return {
        'correct': correct, 'wrong_class': wrong, 'missed_fn': fn, 'false_pos_fp': fp,
        'sibling_confusions': sib, 'same_base_confusions': base,
        'sibling_share_of_wrong': (sib / wrong) if wrong else None,
        'sibling_share_of_all_errors': (sib / total_err) if total_err else None,
    }


def matrix_to_csv(M, path):
    names = CLASS_NAMES + ['background']
    lines = ['true\\pred,' + ','.join(names)]
    for i, n in enumerate(names):
        lines.append(n + ',' + ','.join(str(int(v)) for v in M[i]))
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    import argparse
    ap = argparse.ArgumentParser(description='confusion from saved predictions')
    ap.add_argument('--preds', required=True)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--iou', type=float, default=0.5)
    a = ap.parse_args()
    preds = json.loads(Path(a.preds).read_text())
    gt = {k: load_gt(Path(a.data_root) / (k.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt')) for k in preds}
    M = build_confusion(preds, gt, a.conf, a.iou)
    print(json.dumps(error_stats(M), indent=2))


if __name__ == '__main__':
    main()
