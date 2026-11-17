#!/usr/bin/env python3
"""Independent check of data_splits/ using only the txt lists + image_groups.csv (no generator code)."""
import sys
from pathlib import Path
import pandas as pd

splits = Path(sys.argv[1] if len(sys.argv) > 1 else 'data_splits')
groups = pd.read_csv(sys.argv[2] if len(sys.argv) > 2 else 'results/eda/image_groups.csv')
groups['path'] = groups.split + '/images/' + groups.image
g_of = dict(zip(groups.path, groups.group_id))
K, ok = 5, True


def rd(p):
    return [l for l in Path(p).read_text().splitlines() if l]


def check(cond, msg):
    global ok
    print(('OK   ' if cond else 'FAIL ') + msg)
    ok &= bool(cond)


test_all, val_all = [], []
for k in range(K):
    L = {r: rd(splits / 'clean5' / f'fold{k}_{r}.txt') for r in ['train', 'val', 'test']}
    for r, v in L.items():
        check(all(p in g_of for p in v) and len(v) == len(set(v)), f'fold{k} {r}: {len(v)} paths, known and unique')
    G = {r: {g_of[p] for p in v} for r, v in L.items()}
    check(not (G['train'] & G['val']) and not (G['train'] & G['test']) and not (G['val'] & G['test']),
          f'fold{k}: group sets disjoint (train {len(G["train"])}, val {len(G["val"])}, test {len(G["test"])})')
    # one rep per (group,label-set): at most a few groups have >1 eval image
    multi = pd.Series([g_of[p] for p in L['test']]).value_counts()
    check(multi.max() <= 2, f'fold{k}: test images per group max={multi.max()} (>1 only for label-inconsistent groups: {int((multi>1).sum())})')
    # all members of an eval group not used as eval are not in train
    evalg = G['val'] | G['test']
    tr_groups_of_eval = [p for p in L['train'] if g_of[p] in evalg]
    check(not tr_groups_of_eval, f'fold{k}: no member of a val/test group appears in train')
    test_all += L['test']
    val_all += L['val']
check(len(test_all) == len(set(test_all)), f'pooled OOF test: {len(test_all)} images, each exactly once')
check(len(val_all) == len(set(val_all)), f'pooled val: {len(val_all)} images, each exactly once')
check(len({g_of[p] for p in test_all}) == groups.group_id.nunique(), 'every group is tested in exactly one fold')
off = {r: rd(splits / 'official' / f'{r}.txt') for r in ['train', 'val', 'test']}
check([len(off[r]) for r in ['train', 'val', 'test']] == [1347, 128, 64], 'official split sizes 1347/128/64')
tw, tn = rd(splits / 'official' / 'test_with_train_copy.txt'), rd(splits / 'official' / 'test_no_train_copy.txt')
check(len(tw) == 48 and len(tn) == 16 and not set(tw) & set(tn), 'leak probe: 48 with copy / 16 without')
print('\nALL CHECKS PASSED' if ok else '\nSOME CHECKS FAILED')
sys.exit(0 if ok else 1)
