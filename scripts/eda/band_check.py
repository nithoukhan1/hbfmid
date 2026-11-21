"""Sample unlinked pairs (different v1 groups) with thumbnail NCC in [lo,hi) and render review composites (A | B | B mirrored). Read-only on the dataset.
usage: python3 scripts/eda/band_check.py --data-root <root> --lo 0.90 --hi 0.95 --n 18 --seed 20261123 --out tmp_cache/band_comp"""
import argparse, csv
from pathlib import Path
import cv2, numpy as np, pandas as pd
ap = argparse.ArgumentParser()
ap.add_argument('--data-root', required=True); ap.add_argument('--groups', default='results/eda/image_groups.csv')
ap.add_argument('--lo', type=float, default=0.90); ap.add_argument('--hi', type=float, default=0.95)
ap.add_argument('--n', type=int, default=18); ap.add_argument('--seed', type=int, default=20261123); ap.add_argument('--out', default='tmp_cache/band_comp')
a = ap.parse_args(); root = Path(a.data_root); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
G = pd.read_csv(a.groups); n = len(G)
rd = lambda s, f: cv2.imread(str(root / s / 'images' / f))
def nm(X): X = X.reshape(len(X), -1); X = X - X.mean(1, keepdims=True); return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-6)
T = np.stack([cv2.resize(cv2.cvtColor(rd(s, f), cv2.COLOR_BGR2GRAY), (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32) for s, f in zip(G.split, G.image)])
B = nm(T); S = np.full((n, n), -1.0, np.float32)
for k in range(4):
    for fl in (False, True):
        M = np.rot90(T, k, axes=(1, 2)); M = M[:, :, ::-1] if fl else M
        S = np.maximum(S, B @ nm(np.ascontiguousarray(M)).T)
g = G.group_id.values; iu = np.triu_indices(n, 1)
m = (S[iu] >= a.lo) & (S[iu] < a.hi) & (g[iu[0]] != g[iu[1]])
cand = list(zip(iu[0][m], iu[1][m])); print('pairs in band:', len(cand))
rng = np.random.default_rng(a.seed); pick = [cand[i] for i in rng.permutation(len(cand))[:a.n]]
def sq(im, s=320):
    c = np.zeros((s, s, 3), np.uint8); h, w = im.shape[:2]; im = cv2.resize(im, (s, s)) if (h > s or w > s) else im; c[:im.shape[0], :im.shape[1]] = im; return c
rows = []; tiles = []
for q, (i, j) in enumerate(pick, 1):
    A_, B_ = sq(rd(G.split[i], G.image[i])), sq(rd(G.split[j], G.image[j])); r = np.hstack([A_, B_, B_[:, ::-1]])
    cv2.putText(r, f'Q{q:02d}', (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2); tiles.append(r)
    rows.append({'qid': f'Q{q:02d}', 'a': f'{G.split[i]}/{G.image[i]}', 'b': f'{G.split[j]}/{G.image[j]}', 'ncc': f'{S[i, j]:.3f}', 'split_pair': f'{G.split[i]}-{G.split[j]}'})
for k in range(0, len(tiles), 3):
    cv2.imwrite(str(out / f'band_{k // 3 + 1:02d}.jpg'), np.vstack([np.vstack([t, np.full((6, 960, 3), 255, np.uint8)]) for t in tiles[k:k + 3]]), [cv2.IMWRITE_JPEG_QUALITY, 88])
with open(out / 'band_key.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print('composites:', (len(tiles) + 2) // 3)
