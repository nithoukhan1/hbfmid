"""Grouping v2: v1 SIFT groups + near-duplicate links from 32x32 thumbnail NCC over the 8 dihedral modes.
Motivation: audit validation (docs/RESULTS_AUDIT_VALIDATION.md) showed v1 misses rotated/cropped duplicates (N_hard 8/8 real, R_below 9/16 real).
Read-only on the dataset. Writes <out>/image_groups_v2_ncc<thr>.csv (same columns as image_groups.csv) and a summary json.
Also reports, for the CURRENT clean5 folds (v1), how many v2 duplicate pairs fall in different test-folds (= residual leakage in v1 folds)."""
import argparse, json
from pathlib import Path
import cv2, numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--data-root', required=True)
ap.add_argument('--groups', default='results/eda/image_groups.csv')
ap.add_argument('--assign', default='data_splits/clean5/assignments.csv')
ap.add_argument('--out', default='results/eda/grouping_v2')
ap.add_argument('--thr', type=float, nargs='+', default=[0.90, 0.95, 0.97])
a = ap.parse_args(); root = Path(a.data_root); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
G = pd.read_csv(a.groups); A = pd.read_csv(a.assign)[['split', 'image', 'group_fold']]
G = G.merge(A, on=['split', 'image'], how='left'); n = len(G)
def thumb(s, f):
    im = cv2.imread(str(root / s / 'images' / f), cv2.IMREAD_GRAYSCALE)
    v = cv2.resize(im, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32); return v
T = np.stack([thumb(s, f) for s, f in zip(G.split, G.image)])
def norm(X):
    X = X.reshape(len(X), -1); X = X - X.mean(1, keepdims=True); return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-6)
base = norm(T); S = np.full((n, n), -1.0, np.float32)
for k in range(4):
    for fl in (False, True):
        M = np.rot90(T, k, axes=(1, 2)); M = M[:, :, ::-1] if fl else M
        S = np.maximum(S, base @ norm(np.ascontiguousarray(M)).T)
np.fill_diagonal(S, -1)
g1 = G.group_id.values; fold = G.group_fold.values
def uf_groups(extra):
    parent = {g: g for g in set(g1)}
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i, j in extra:
        a_, b_ = find(g1[i]), find(g1[j])
        if a_ != b_: parent[b_] = a_
    return np.array([find(g) for g in g1])
summary = {'n_images': n, 'v1_groups': int(len(set(g1)))}
iu = np.triu_indices(n, 1)
for thr in a.thr:
    mask = S[iu] >= thr; pairs = list(zip(iu[0][mask], iu[1][mask]))
    unl = [(i, j) for i, j in pairs if g1[i] != g1[j]]
    g2 = uf_groups(unl); ids = {g: k for k, g in enumerate(sorted(set(g2)))}
    df = G[['split', 'image', 'classes']].copy(); df['group_id'] = [ids[g] for g in g2]
    df[['split', 'image', 'group_id', 'classes']].to_csv(out / f'image_groups_v2_ncc{thr:.2f}.csv', index=False)
    cross = sum(1 for i, j in unl if fold[i] != fold[j])
    # images whose v1 group is merged with a group of a different v1 fold
    gf = pd.Series(fold).groupby(g2).nunique(); split_groups = int((gf > 1).sum())
    tst = G.split.values == 'test'
    test_with_train = int(sum(1 for t in np.where(tst)[0] if any(S[t, j] >= thr and G.split.values[j] == 'train' and g1[t] != g1[j] for j in range(n))))
    summary[f'ncc>={thr:.2f}'] = {'unlinked_pairs': len(unl), 'v2_groups': len(ids), 'pairs_in_different_v1_test_folds': int(cross),
        'v2_groups_spanning_multiple_v1_folds': split_groups, 'official_test_images_with_NEW_train_link': test_with_train}
(out / 'summary.json').write_text(json.dumps(summary, indent=1)); print(json.dumps(summary, indent=1))
