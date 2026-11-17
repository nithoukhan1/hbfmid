#!/usr/bin/env python3
"""Cross-dataset leakage check: does any FracAtlas image appear (as a copy) inside HBFMID?
READ-ONLY on both datasets. Outputs go to --out (never inside a dataset folder).
Method = same as leakage_audit.py: 32x32 NCC candidates (4 flip variants) -> SIFT + RANSAC-affine inlier count.
HBFMID side: one representative image per independent group (results/eda/image_groups.csv); FracAtlas side: all images.
Usage: python scripts/eda/cross_dataset_audit.py --hbfmid "<.../Bone Fractures Detection>" --fracatlas "<.../FracAtlas (FigShare)/FracAtlas>" --out results/eda --cache tmp_cache/cross
"""
import argparse, os, sys, json, csv, pickle, time
import numpy as np, cv2, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from leakage_audit import norm_vec, verify, SIFT_SIZE

ap = argparse.ArgumentParser()
ap.add_argument('--hbfmid', required=True); ap.add_argument('--fracatlas', required=True)
ap.add_argument('--groups', default='results/eda/image_groups.csv')
ap.add_argument('--out', default='results/eda'); ap.add_argument('--cache', default='tmp_cache/cross')
ap.add_argument('--topk', type=int, default=20); ap.add_argument('--thr', type=int, default=25)
ap.add_argument('--budget', type=float, default=0, help='seconds per invocation (0 = unlimited); resumable, exit code 3 = not finished')
a = ap.parse_args()
for d in (a.hbfmid, a.fracatlas):
    assert not os.path.abspath(a.out).startswith(os.path.abspath(d)), 'output inside a dataset folder'
os.makedirs(a.cache, exist_ok=True)
T_START = time.time()
def out_of_time():
    return a.budget > 0 and time.time() - T_START > a.budget

g = pd.read_csv(a.groups)
reps = g.sort_values(['group_id', 'split']).groupby('group_id').head(1).reset_index(drop=True)
H = [os.path.join(a.hbfmid, r.split, 'images', r.image) for r in reps.itertuples()]
F = []
for sub in ('Fractured', 'Non_fractured'):
    d = os.path.join(a.fracatlas, 'images', sub)
    F += [os.path.join(d, f) for f in sorted(os.listdir(d))]
print('HBFMID reps', len(H), '| FracAtlas', len(F), flush=True)


def feats(paths, tag, flip):
    pth = os.path.join(a.cache, tag + '.pkl')
    st = pickle.load(open(pth, 'rb')) if os.path.exists(pth) else {'T': [], 'D': []}
    if len(st['T']) == len(paths):
        return np.array(st['T']), st['D']
    sift = cv2.SIFT_create(nfeatures=600); t0 = time.time()
    for k in range(len(st['T']), len(paths)):
        im = cv2.imread(paths[k], 0)
        if im is None:
            st['T'].append(np.zeros((32, 32), np.uint8)); st['D'].append([(np.zeros((0, 2), np.float32), np.zeros((0, 128), np.uint8))] * (2 if flip else 1))
        else:
            st['T'].append(cv2.resize(im, (32, 32), interpolation=cv2.INTER_AREA))
            gg = cv2.resize(im, (SIFT_SIZE, SIFT_SIZE), interpolation=cv2.INTER_AREA); rec = []
            for x in ((gg, gg[:, ::-1].copy()) if flip else (gg,)):
                kp, d = sift.detectAndCompute(x, None)
                rec.append((np.array([q.pt for q in kp], np.float32) if kp else np.zeros((0, 2), np.float32),
                            np.rint(d).astype(np.uint8) if d is not None else np.zeros((0, 128), np.uint8)))
            st['D'].append(rec)
        if k % 250 == 0:
            print(tag, k, '/', len(paths), '%.0fs' % (time.time() - t0), flush=True)
        if out_of_time():
            pickle.dump(st, open(pth, 'wb')); print('PAUSED in', tag, 'at', len(st['T']), flush=True); sys.exit(3)
    pickle.dump(st, open(pth, 'wb')); return np.array(st['T']), st['D']

TH, DH = feats(H, 'hbfmid', True); TF, DF = feats(F, 'fracatlas', False)
bh = np.array([norm_vec(t) for t in TH]); S = np.full((len(H), len(F)), -1.0, np.float32)
for v in (TF, TF[:, :, ::-1], TF[:, ::-1, :], TF[:, ::-1, ::-1]):
    S = np.maximum(S, bh @ np.array([norm_vec(t) for t in v]).T)
cand = set()
for i in range(len(H)):
    for j in np.argsort(-S[i])[:a.topk]: cand.add((i, int(j)))
for i, j in zip(*np.where(S >= 0.90)): cand.add((int(i), int(j)))
print('candidate pairs', len(cand), flush=True)
bf = cv2.BFMatcher(cv2.NORM_L2); t0 = time.time()
vp = os.path.join(a.cache, 'verified.pkl'); done = pickle.load(open(vp, 'rb')) if os.path.exists(vp) else {}
for c, (i, j) in enumerate(sorted(cand)):
    if (i, j) in done: continue
    q = [(DF[j][0][0], DF[j][0][1].astype(np.float32))]; t0_ = (DH[i][0][0], DH[i][0][1].astype(np.float32)); t1_ = (DH[i][1][0], DH[i][1][1].astype(np.float32))
    n, r, m = verify(q, t0_, t1_, bf); done[(i, j)] = (i, j, n, r, m, float(S[i, j]))
    if c % 500 == 0: print('verify', c, '/', len(cand), flush=True)
    if out_of_time():
        pickle.dump(done, open(vp, 'wb')); print('PAUSED in verify at', len(done), '/', len(cand), flush=True); sys.exit(3)
pickle.dump(done, open(vp, 'wb')); res = list(done.values())
print('verified', len(res), 'pairs', flush=True)
res.sort(key=lambda x: -x[2])
with open(os.path.join(a.out, 'cross_pairs_top.csv'), 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['hbfmid_group', 'hbfmid_image', 'fracatlas_image', 'inliers', 'ratio', 'mode', 'thumb_ncc'])
    for i, j, n, r, m, s in res[:200]:
        w.writerow([int(reps.group_id[i]), os.path.basename(H[i]), os.path.basename(F[j]), n, round(r, 3), m, round(s, 3)])
inl = np.array([x[2] for x in res])
bins = [0, 5, 8, 10, 12, 15, 20, 25, 30, 40, 1000]
summary = {'n_hbfmid_reps': len(H), 'n_fracatlas': len(F), 'candidate_pairs': len(cand), 'threshold': a.thr,
           'pairs_ge_thr': int((inl >= a.thr).sum()), 'hbfmid_groups_with_fracatlas_match': int(len({x[0] for x in res if x[2] >= a.thr})),
           'inlier_hist': {f'{bins[k]}-{bins[k+1]-1}': int(((inl >= bins[k]) & (inl < bins[k+1])).sum()) for k in range(len(bins) - 1)},
           'top10': [(int(reps.group_id[x[0]]), os.path.basename(F[x[1]]), x[2]) for x in res[:10]]}
json.dump(summary, open(os.path.join(a.out, 'cross_audit_summary.json'), 'w'), indent=1)
print(json.dumps(summary, indent=1))
# montage of top-8 pairs (left HBFMID, right FracAtlas)
rows = []
for i, j, n, r, m, s in res[:8]:
    A = cv2.resize(cv2.imread(H[i]), (200, 200)); B = cv2.resize(cv2.imread(F[j]), (200, 200))
    cv2.putText(A, f'inl={n}', (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1); rows.append(np.hstack([A, B]))
os.makedirs(os.path.join(a.out, 'figures'), exist_ok=True)
cv2.imwrite(os.path.join(a.out, 'figures', 'cross_top_pairs.jpg'), np.vstack([np.hstack(rows[k:k + 2]) for k in range(0, len(rows), 2)]))
