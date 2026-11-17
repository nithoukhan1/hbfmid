#!/usr/bin/env python3
"""Control experiment list: the OFFICIAL split with its leakage removed, test set unchanged.
  test      = official test (all 64 images, unchanged)
  val       = official valid images whose group has NO member in test
  train     = official train images whose group has NO member in test or in the kept val
Isolates how much of the official-vs-clean gap is caused by leakage on the SAME test images.
Usage: python scripts/eda/make_official_dedup.py --groups results/eda/image_groups.csv --out data_splits/official_dedup"""
import argparse, json, os
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--groups', default='results/eda/image_groups.csv'); ap.add_argument('--out', default='data_splits/official_dedup')
a = ap.parse_args()
g = pd.read_csv(a.groups); g['path'] = g.split + '/images/' + g.image
test = g[g.split == 'test']; tg = set(test.group_id)
val = g[(g.split == 'valid') & ~g.group_id.isin(tg)]; vg = set(val.group_id)
train = g[(g.split == 'train') & ~g.group_id.isin(tg | vg)]
assert not (set(train.group_id) & (tg | vg)) and not (vg & tg)
os.makedirs(a.out, exist_ok=True)
for n, d in (('train', train), ('val', val), ('test', test)):
    open(f'{a.out}/{n}.txt', 'w').write('\n'.join(d.path) + '\n')
orig = g.split.value_counts().to_dict()
s = {'train': len(train), 'val': len(val), 'test': len(test), 'official_sizes': orig,
     'train_removed': orig['train'] - len(train), 'val_removed': orig['valid'] - len(val),
     'train_groups': int(train.group_id.nunique()), 'val_groups': int(val.group_id.nunique()), 'test_groups': int(test.group_id.nunique())}
json.dump(s, open(f'{a.out}/summary.json', 'w'), indent=1); print(json.dumps(s, indent=1))
