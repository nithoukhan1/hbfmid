#!/usr/bin/env python3
"""Tag every image as MRI or X-ray from its file name (read-only; uses no image pixels).
Rule: base name (before '.rf.') contains 'mri' -> MRI, else X-ray.
Checked 2026-11-17: 0/48 randomly sampled untagged group representatives were MRI (visual review).
Usage: python scripts/eda/tag_modality.py --groups results/eda/image_groups.csv --assign data_splits/clean5/assignments.csv --out results/eda
"""
import argparse, json
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--groups', default='results/eda/image_groups.csv')
ap.add_argument('--assign', default='data_splits/clean5/assignments.csv')
ap.add_argument('--out', default='results/eda')
a = ap.parse_args()

g = pd.read_csv(a.groups)
g['base'] = g.image.str.split('.rf.').str[0]
g['modality'] = g.base.str.contains('mri', case=False).map({True: 'MRI', False: 'XR'})
mixed = g.groupby('group_id').modality.nunique().gt(1).sum()
assert mixed == 0, f'{mixed} groups mix modalities'
g[['split', 'image', 'group_id', 'modality']].to_csv(f'{a.out}/modality_tags.csv', index=False)

grp = g.groupby('group_id').agg(modality=('modality', 'first'), classes=('classes', 'first'))
s = {'images': g.modality.value_counts().to_dict(),
     'groups': grp.modality.value_counts().to_dict(),
     'images_by_split': g.groupby(['split', 'modality']).size().unstack(fill_value=0).to_dict('index')}
asg = pd.read_csv(a.assign)
if 'group_fold' in asg.columns and 'is_rep' in asg.columns:
    m = asg.merge(g[['split', 'image', 'modality']], on=['split', 'image'], how='left')
    s['test_reps_by_fold'] = m[m.is_rep].groupby(['group_fold', 'modality']).size().unstack(fill_value=0).to_dict('index')
json.dump(s, open(f'{a.out}/modality_summary.json', 'w'), indent=1, default=int)
print(json.dumps(s, indent=1, default=int))
