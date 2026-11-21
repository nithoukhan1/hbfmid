"""Quantify near-duplicate pairs the SIFT audit did NOT link (different groups), via 32x32 thumbnail NCC over 4 flip modes.
Read-only on the dataset. Output: results/eda/thumbnail_dup_scan.json
usage: python3 scripts/eda/thumbnail_dup_scan.py --data-root <Bone Fractures Detection> --groups results/eda/image_groups.csv --out results/eda/thumbnail_dup_scan.json"""
import argparse, csv, json
from pathlib import Path
import cv2, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--data-root', required=True); ap.add_argument('--groups', default='results/eda/image_groups.csv')
ap.add_argument('--out', default='results/eda/thumbnail_dup_scan.json')
a = ap.parse_args(); root = Path(a.data_root)
G = list(csv.DictReader(open(a.groups)))
keys = [(g['split'], g['image']) for g in G]; gi = np.array([int(g['group_id']) for g in G]); sp = np.array([g['split'] for g in G])
def thumb(k):
    im = cv2.imread(str(root / k[0] / 'images' / k[1]), cv2.IMREAD_GRAYSCALE)
    v = cv2.resize(im, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32).ravel(); v -= v.mean()
    return v / (np.linalg.norm(v) + 1e-6)
T = np.stack([thumb(k) for k in keys]); T4 = T.reshape(-1, 32, 32)
mods = [T4, T4[:, ::-1, :], T4[:, :, ::-1], T4[:, ::-1, ::-1]]
S = np.max([T @ m.reshape(len(keys), -1).T for m in mods], axis=0)
diff = gi[:, None] != gi[None, :]
out = {'n_images': len(keys)}
for thr in (0.90, 0.95, 0.97):
    M = (S >= thr) & diff; np.fill_diagonal(M, False)
    iu = np.triu_indices(len(keys), 1); pairs = int(M[iu].sum())
    test = np.where(sp == 'test')[0]; valid = np.where(sp == 'valid')[0]
    tr = sp == 'train'
    out[f'ncc>={thr}'] = {'unlinked_pairs': pairs,
        'test_images_with_unlinked_train_partner': int(sum(M[i][tr].any() for i in test)),
        'valid_images_with_unlinked_train_partner': int(sum(M[i][tr].any() for i in valid)),
        'n_test': int(len(test)), 'n_valid': int(len(valid))}
Path(a.out).write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
