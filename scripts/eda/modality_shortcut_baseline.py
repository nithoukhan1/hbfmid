#!/usr/bin/env python3
"""How much does modality alone (MRI vs X-ray, from file name) explain the labels? Pooled over the 5 clean folds.
No images are read. Rule is fitted on each fold's TRAIN images and scored on that fold's TEST representatives."""
import json, pandas as pd, numpy as np
NAMES = ['Comminuted', 'Greenstick', 'Healthy', 'Linear', 'Oblique Displaced', 'Oblique', 'Segmental', 'Spiral', 'Transverse Displaced', 'Transverse']
A = pd.read_csv('data_splits/clean5/assignments.csv'); M = pd.read_csv('results/eda/modality_tags.csv')
A = A.merge(M[['split', 'image', 'modality']], on=['split', 'image'], how='left'); assert A.modality.notna().all()
A['cs'] = A.classes.astype(str).str.replace(r'[;,\[\]]', ' ', regex=True).str.split().apply(lambda l: {int(x) for x in l if x.isdigit()})
rows, hit_rule, hit_major, n = [], 0, 0, 0
per_mod = {}
for k in range(5):
    tr = A[A[f'role_f{k}'] == 'train']; te = A[A[f'role_f{k}'] == 'test']
    top = {}
    for mod, g in tr.groupby('modality'):
        cnt = pd.Series([c for s in g.cs for c in s]).value_counts(); top[mod] = int(cnt.index[0])
    major = int(pd.Series([c for s in tr.cs for c in s]).value_counts().index[0])
    for r in te.itertuples():
        n += 1; hit_rule += top[r.modality] in r.cs; hit_major += major in r.cs
        rows.append((r.group_id, r.modality, r.cs, top.get(r.modality)))
    per_mod[k] = {m: NAMES[c] for m, c in top.items()}
T = pd.DataFrame(rows, columns=['g', 'mod', 'cs', 'rule'])
comm = T.cs.apply(lambda s: 0 in s); mri = T['mod'] == 'MRI'
out = {'test_reps_pooled': n, 'rule_per_fold(train-fitted)': per_mod,
       'rule_hit_rate(any true class == predicted)': round(hit_rule / n, 3), 'always_majority_hit_rate': round(hit_major / n, 3),
       'comminuted_test_reps': int(comm.sum()), 'mri_test_reps': int(mri.sum()),
       'comminuted_that_are_MRI(recall_of_MRI_flag)': round(float((comm & mri).sum() / max(comm.sum(), 1)), 3),
       'MRI_that_are_comminuted(precision_of_MRI_flag)': round(float((comm & mri).sum() / max(mri.sum(), 1)), 3),
       'MRI_test_reps_by_class': {NAMES[c]: int(sum(c in s for s in T[mri].cs)) for c in range(10) if any(c in s for s in T[mri].cs)}}
json.dump(out, open('results/eda/modality_shortcut_baseline.json', 'w'), indent=1); print(json.dumps(out, indent=1))
