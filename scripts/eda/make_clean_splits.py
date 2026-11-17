#!/usr/bin/env python3
"""Build the HBFMID-Clean protocol: group-stratified rotating K-fold splits.

Inputs : results/eda/image_groups.csv  (from leakage_audit.py: split,image,group_id,classes)
Outputs: data_splits/clean5/*   (txt lists with paths RELATIVE to the dataset root)
         data_splits/official/* (official split lists + leak-probe lists)

Design rules (see docs/IDEA_V3.md):
  * Unit of splitting = independent group (near-duplicate cluster), never the image.
  * Fold k: test = group-fold k, val = group-fold (k+1) % K, train = the other folds.
  * Train uses EVERY member image of its groups (copies + augmentations stay together).
  * Val/test use ONE representative image per (group, label-set); the remaining members of
    an evaluation group are EXCLUDED from that fold (they must never enter train).
  * Representative preference: official test > official valid > train (un-augmented first),
    then file name (deterministic).
  * The dataset folder is only READ. Output directory must not be inside it.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

CLASS_NAMES = ['Comminuted', 'Greenstick', 'Healthy', 'Linear', 'Oblique Displaced',
               'Oblique', 'Segmental', 'Spiral', 'Transverse Displaced', 'Transverse']
SPLIT_RANK = {'test': 0, 'valid': 1, 'train': 2}


def rel_path(split, image):
    return f"{split}/images/{image}"


def parse_classes(x):
    if not isinstance(x, str) or not x.strip():
        return ()
    return tuple(sorted(int(t) for t in x.split()))


def choose_reps(df):
    """One representative per (group, label-set). Returns set of image names."""
    d = df.assign(rank=df.split.map(SPLIT_RANK))
    reps = set()
    for _, sub in d.groupby(['group_id', 'cls_key']):
        reps.add(sub.sort_values(['rank', 'image']).iloc[0].image)
    return reps


def assign_folds(P, K, seed, restarts):
    """Greedy multi-label stratified assignment of groups to K folds (rarest first)."""
    G, C = P.shape
    total = P.sum(0)
    desired = total / K
    has = P.sum(1) > 0
    rarity = np.where(has, np.min(np.where(P > 0, total[None, :], np.inf), axis=1), np.inf)
    best = None
    for r in range(restarts):
        rng = np.random.default_rng(seed + r)
        order = np.lexsort((rng.random(G), rarity))
        fold = -np.ones(G, int)
        cur = np.zeros((K, C))
        cnt = np.zeros(K)
        for g in order:
            cls = np.where(P[g] > 0)[0]
            if len(cls):
                score = ((desired[cls][None, :] - cur[:, cls]) / desired[cls][None, :]).sum(1)
            else:
                score = np.zeros(K)
            score = score - 0.02 * cnt + rng.random(K) * 1e-6
            f = int(np.argmax(score))
            fold[g] = f
            cnt[f] += 1
            cur[f, cls] += 1
        dev = np.abs(cur - desired[None, :]).max(0) / np.maximum(desired, 1)
        obj = float(dev.mean() + 0.5 * (cnt.max() - cnt.min()) / (G / K))
        if best is None or obj < best[0]:
            best = (obj, fold.copy(), cur.copy(), cnt.copy())
    return best


def write_list(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(items) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups', required=True, help='results/eda/image_groups.csv')
    ap.add_argument('--out', required=True, help='repo data_splits directory')
    ap.add_argument('--data-root', default=None, help='dataset root (READ-ONLY, used only to verify files exist)')
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--restarts', type=int, default=2000)
    a = ap.parse_args()

    out = Path(a.out).resolve()
    if a.data_root:
        root = Path(a.data_root).resolve()
        assert root not in out.parents and out != root, "output dir must not be inside the dataset folder"

    raw = Path(a.groups).read_bytes()
    df = pd.read_csv(a.groups, dtype={'classes': str})
    assert df.image.is_unique, "image names are not unique across splits"
    df['cls_tuple'] = df.classes.map(parse_classes)
    df['cls_key'] = df.cls_tuple.map(lambda t: ' '.join(map(str, t)))
    df['path'] = [rel_path(s, i) for s, i in zip(df.split, df.image)]
    K = a.k

    # ---- groups -------------------------------------------------------------
    gids = sorted(df.group_id.unique())
    gidx = {g: i for i, g in enumerate(gids)}
    P = np.zeros((len(gids), len(CLASS_NAMES)))
    for g, sub in df.groupby('group_id'):
        for t in sub.cls_tuple:
            for c in t:
                P[gidx[g], c] = 1
    obj, gfold, cur, cnt = assign_folds(P, K, a.seed, a.restarts)
    df['group_fold'] = df.group_id.map(lambda g: int(gfold[gidx[g]]))
    reps = choose_reps(df)
    df['is_rep'] = df.image.isin(reps)

    # ---- per-fold roles -------------------------------------------------------
    summary = {'k': K, 'seed': a.seed, 'restarts': a.restarts, 'objective': obj,
               'groups_csv_sha256': hashlib.sha256(raw).hexdigest(),
               'n_images': int(len(df)), 'n_groups': int(len(gids)),
               'n_representatives': int(df.is_rep.sum()), 'folds': {}}
    rows = []
    warnings = []
    for k in range(K):
        tf, vf = k, (k + 1) % K
        role = np.where(df.group_fold == tf, 'test', np.where(df.group_fold == vf, 'val', 'train')).astype(object)
        eval_nonrep = (role != 'train') & (~df.is_rep.values)
        role[eval_nonrep] = 'excluded'
        df[f'role_f{k}'] = role
        lists = {r: df.loc[df[f'role_f{k}'] == r, 'path'].sort_values().tolist() for r in ['train', 'val', 'test']}
        for r, items in lists.items():
            write_list(out / 'clean5' / f'fold{k}_{r}.txt', items)
        # disjointness checks (images AND groups)
        sets = {r: set(df.loc[df[f'role_f{k}'] == r, 'image']) for r in ['train', 'val', 'test', 'excluded']}
        gsets = {r: set(df.loc[df[f'role_f{k}'] == r, 'group_id']) for r in ['train', 'val', 'test']}
        for a1 in sets:
            for b1 in sets:
                if a1 < b1:
                    assert not (sets[a1] & sets[b1]), f"fold {k}: image overlap {a1}/{b1}"
        for a1, b1 in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
            assert not (gsets[a1] & gsets[b1]), f"fold {k}: GROUP overlap {a1}/{b1}"
        fs = {'test_fold': tf, 'val_fold': vf,
              'n_groups': {r: len(gsets[r]) for r in gsets},
              'n_images': {r: len(sets[r]) for r in sets}, 'class_groups': {}}
        for r in ['train', 'val', 'test']:
            sub = df[df[f'role_f{k}'] == r]
            for c, name in enumerate(CLASS_NAMES):
                ng = int(sub[sub.cls_tuple.map(lambda t: c in t)].group_id.nunique())
                ni = int(sub.cls_tuple.map(lambda t: c in t).sum())
                rows.append({'fold': k, 'role': r, 'class_id': c, 'class': name,
                             'n_groups': ng, 'n_images': ni})
                fs['class_groups'].setdefault(name, {})[r] = ng
        for name, d in fs['class_groups'].items():
            if d['train'] < 1:
                warnings.append(f"fold {k}: class {name} has no train group")
        summary['folds'][k] = fs

    # every group is test in exactly one fold and val in exactly one fold
    for g in gids:
        sub = df[df.group_id == g]
        n_test = sum(int((sub[f'role_f{k}'] == 'test').any()) for k in range(K))
        n_val = sum(int((sub[f'role_f{k}'] == 'val').any()) for k in range(K))
        assert n_test == 1 and n_val == 1, (g, n_test, n_val)
        assert sub.group_fold.nunique() == 1

    df.drop(columns=['cls_tuple']).to_csv(out / 'clean5' / 'assignments.csv', index=False)
    pd.DataFrame(rows).to_csv(out / 'clean5' / 'fold_class_groups.csv', index=False)

    # ---- official split lists + leak-probe lists ---------------------------------
    for s, name in [('train', 'train'), ('valid', 'val'), ('test', 'test')]:
        write_list(out / 'official' / f'{name}.txt', df[df.split == s].path.sort_values().tolist())
    train_groups = set(df[df.split == 'train'].group_id)
    for s, name in [('test', 'test'), ('valid', 'val')]:
        sub = df[df.split == s]
        has = sub.group_id.isin(train_groups)
        write_list(out / 'official' / f'{name}_with_train_copy.txt', sub[has].path.sort_values().tolist())
        write_list(out / 'official' / f'{name}_no_train_copy.txt', sub[~has].path.sort_values().tolist())
        summary[f'official_{name}_with_train_copy'] = int(has.sum())
        summary[f'official_{name}_no_train_copy'] = int((~has).sum())

    # ---- optional: verify files exist (READ-ONLY) ----------------------------------
    if a.data_root:
        missing = []
        for p in df.path:
            ip = root / p
            lp = root / (p.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt')
            if not ip.exists() or not lp.exists():
                missing.append(p)
        summary['missing_files'] = len(missing)
        assert not missing, f"{len(missing)} listed files missing, e.g. {missing[:3]}"

    # pooled OOF test set = every representative exactly once
    oof = sum(len(df[df[f'role_f{k}'] == 'test']) for k in range(K))
    assert oof == int(df.is_rep.sum()), (oof, int(df.is_rep.sum()))
    summary['oof_test_images'] = oof
    summary['warnings'] = warnings
    (out / 'clean5' / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    # ---- console report ---------------------------------------------------------------
    print(f"groups={len(gids)} images={len(df)} reps={int(df.is_rep.sum())} objective={obj:.4f}")
    print("groups per fold:", cnt.astype(int).tolist())
    t = pd.DataFrame(rows)
    piv = t[t.role == 'test'].pivot(index='class', columns='fold', values='n_groups')
    piv['total'] = piv.sum(1)
    print("\nTEST groups per class per fold:\n", piv.to_string())
    pv = t[t.role == 'train'].pivot(index='class', columns='fold', values='n_groups')
    print("\nTRAIN groups per class per fold:\n", pv.to_string())
    for k in range(K):
        print(f"fold {k}: images {summary['folds'][k]['n_images']}")
    print("warnings:", warnings if warnings else "none")


if __name__ == '__main__':
    main()
