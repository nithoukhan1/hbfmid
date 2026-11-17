"""Class-frequency weights for the cls (BCE) loss, from raw box counts or independent-group counts.

Formula = the one Ultralytics >= 8.4 uses for its own `cls_pw` option:  w_c = (1 / n_c) ** p, rescaled to mean 1,
classes with no training samples get the mean weight. The weights multiply the per-class column of the BCE term
(positives AND negatives of that class), exactly like `model.class_weights` inside v8DetectionLoss.

What differs between the arms is only n_c:
  raw   : number of training boxes of class c in the training list (what cls_pw itself would count)
  group : number of independent image groups (leakage-audit groups) in the training fold that contain class c
          (data_splits/clean5/fold_class_groups.csv). Near-duplicate copies of one radiograph then count once.

install_callback() sets model.class_weights at `on_train_start`, i.e. before the criterion is built (it is created lazily
at the first training step), so no Ultralytics code is patched.
"""
import csv
from pathlib import Path

import numpy as np


def weights_from_counts(counts, p=0.5):
    c = np.asarray(counts, dtype=float)
    present = c > 0
    if not present.any():
        return np.ones_like(c)
    w = (1.0 / np.where(present, c, 1.0)) ** p
    w = np.where(present, w, w[present].mean())
    return w / w.mean()


def counts_raw(train_list, data_root, nc):
    """Boxes per class over the training list (paths relative to data_root, as in data_splits/)."""
    n = np.zeros(nc)
    for rel in Path(train_list).read_text().split():
        lab = Path(data_root) / (rel.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt')
        if not lab.exists():
            continue
        for line in lab.read_text().splitlines():
            t = line.split()
            if len(t) >= 5:
                n[int(float(t[0]))] += 1
    return n


def counts_group(fold, splits_dir, nc):
    """Independent groups per class in the training role of `fold` (from fold_class_groups.csv)."""
    n = np.zeros(nc)
    with open(Path(splits_dir) / 'clean5' / 'fold_class_groups.csv') as f:
        for r in csv.DictReader(f):
            if int(r['fold']) == fold and r['role'] == 'train':
                n[int(r['class_id'])] = int(r['n_groups'])
    return n


def install_callback(model, weights, nc_model=None):
    """Register an on_train_start callback that sets model.class_weights (length = number of model classes).
    If the model has more classes than `weights` (extra generic class), the extra entries get weight 1."""
    import torch
    w = np.asarray(weights, dtype=float)

    def cb(trainer):
        from ultralytics.utils.torch_utils import unwrap_model
        m = unwrap_model(trainer.model)
        nc = m.model[-1].nc
        full = np.ones(nc)
        full[:len(w)] = w
        m.class_weights = torch.from_numpy(full).float().to(trainer.device)
        print('[class_weights]', np.round(full, 3).tolist(), flush=True)

    model.add_callback('on_train_start', cb)
