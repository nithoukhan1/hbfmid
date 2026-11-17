#!/usr/bin/env python3
"""
Strict duplicate / leakage audit for HBFMID (YOLO/Roboflow export).

READ-ONLY on the dataset. All outputs go to --out (must NOT be inside --data-root).

Method
  1. Candidate generation: 32x32 grayscale thumbnails, normalised cross-correlation
     under the 4 flip variants -> top-K neighbours per image (+ all pairs >= 0.90).
  2. Verification: SIFT keypoints (384px) matched with Lowe ratio test against the other
     image and its horizontal flip (vertical flip == hflip + 180deg, which SIFT handles),
     then RANSAC full-affine fit. Inlier count = evidence that both files are the same
     underlying radiograph (covers flips, small rotation/shear, brightness changes).
  3. Union-find on verified pairs -> groups (= one underlying original image).
  4. Reports leakage (val/test images that have a verified copy in another split),
     effective number of independent groups per class, and verification montages.

Usage:
  python leakage_audit.py --data-root "<.../Bone Fractures Detection>" --out <repo>/results/eda \
         --cache <repo>/tmp_cache [--inlier-thr 25] [--topk 25]
"""
import argparse, os, json, csv, pickle, time, random
import numpy as np, cv2

SPLITS = ['train', 'valid', 'test']
SIFT_SIZE = 384
NAMES = ['Comminuted', 'Greenstick', 'Healthy', 'Linear', 'Oblique Displaced',
         'Oblique', 'Segmental', 'Spiral', 'Transverse Displaced', 'Transverse']


def list_images(root):
    items = []
    for sp in SPLITS:
        d = os.path.join(root, sp, 'images')
        for f in sorted(os.listdir(d)):
            items.append((sp, f))
    return items


def read_classes(root, sp, f):
    lf = os.path.join(root, sp, 'labels', f.rsplit('.', 1)[0] + '.txt')
    if not os.path.exists(lf):
        return []
    out = []
    for line in open(lf):
        p = line.split()
        if p:
            out.append(int(p[0]))
    return out


def norm_vec(a):
    a = a.astype(np.float32).ravel()
    a -= a.mean()
    return a / (np.linalg.norm(a) + 1e-6)


def thumbs(items, root, cache):
    pth = os.path.join(cache, 'thumbs.npy')
    if os.path.exists(pth):
        return np.load(pth)
    T = []
    for sp, f in items:
        g = cv2.imread(os.path.join(root, sp, 'images', f), 0)
        T.append(cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA))
    T = np.array(T)
    np.save(pth, T)
    return T


def sift_all(items, root, cache):
    pth = os.path.join(cache, 'sift384.pkl')
    if os.path.exists(pth):
        return pickle.load(open(pth, 'rb'))
    sift = cv2.SIFT_create(nfeatures=600)
    D = []
    t = time.time()
    for k, (sp, f) in enumerate(items):
        g = cv2.imread(os.path.join(root, sp, 'images', f), 0)
        g = cv2.resize(g, (SIFT_SIZE, SIFT_SIZE), interpolation=cv2.INTER_AREA)
        rec = []
        for im in (g, g[:, ::-1].copy()):
            kp, d = sift.detectAndCompute(im, None)
            xy = np.array([p.pt for p in kp], dtype=np.float32) if kp else np.zeros((0, 2), np.float32)
            rec.append((xy, d if d is not None else np.zeros((0, 128), np.float32)))
        D.append(rec)
    print('sift done %.0fs' % (time.time() - t))
    pickle.dump(D, open(pth, 'wb'))
    return D


def verify(Di, Dj_orig, Dj_flip, bf):
    best = (0, 0.0, 'none')
    di = Di[0][1]
    xi = Di[0][0]
    if len(di) < 8:
        return best
    for mode, (xj, dj) in (('same', Dj_orig), ('hflip', Dj_flip)):
        if len(dj) < 8:
            continue
        m = bf.knnMatch(di, dj, k=2)
        good = [a for a, b in (x for x in m if len(x) == 2) if a.distance < 0.75 * b.distance]
        if len(good) < 8:
            continue
        src = xi[[g.queryIdx for g in good]]
        dst = xj[[g.trainIdx for g in good]]
        M, inl = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0)
        if M is None:
            continue
        n = int(inl.sum())
        if n > best[0]:
            best = (n, n / len(good), mode)
    return best


class UF:
    def __init__(s, n): s.p = list(range(n))
    def f(s, x):
        while s.p[x] != x:
            s.p[x] = s.p[s.p[x]]
            x = s.p[x]
        return x
    def u(s, a, b): s.p[s.f(a)] = s.f(b)


def montage(pairs, items, root, path, title):
    tiles = []
    for (i, j, n, ratio, mode) in pairs:
        ims = []
        for k in (i, j):
            sp, f = items[k]
            im = cv2.imread(os.path.join(root, sp, 'images', f))
            ims.append(cv2.resize(im, (256, 256)))
        t = np.full((276, 512, 3), 255, np.uint8)
        t[20:, :256] = ims[0]
        t[20:, 256:] = ims[1]
        cv2.putText(t, '%s | %s inl=%d %s' % (items[i][0], items[j][0], n, mode), (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(t)
    if not tiles:
        return
    while len(tiles) % 2:
        tiles.append(np.full((276, 512, 3), 255, np.uint8))
    rows = [np.hstack(tiles[r:r + 2]) for r in range(0, len(tiles), 2)]
    cv2.imwrite(path, np.vstack(rows))
    print('saved', path, title)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--cache', required=True)
    ap.add_argument('--inlier-thr', type=int, default=25)
    ap.add_argument('--topk', type=int, default=25)
    ap.add_argument('--stage', default='all', choices=['all', 'features'])
    a = ap.parse_args()
    root = os.path.abspath(a.data_root)
    out = os.path.abspath(a.out)
    assert not out.startswith(root), 'output dir must not be inside the dataset folder'
    os.makedirs(out, exist_ok=True); os.makedirs(os.path.join(out, 'figures'), exist_ok=True)
    os.makedirs(a.cache, exist_ok=True)

    items = list_images(root)
    n = len(items)
    print('images', n)
    T = thumbs(items, root, a.cache)
    D = sift_all(items, root, a.cache)
    if a.stage == 'features':
        return

    # ---- candidate generation
    base = np.array([norm_vec(t) for t in T])
    S = np.full((n, n), -1.0, np.float32)
    for v in (T, T[:, :, ::-1], T[:, ::-1, :], T[:, ::-1, ::-1]):
        V = np.array([norm_vec(t) for t in v])
        S = np.maximum(S, base @ V.T)
    np.fill_diagonal(S, -1)
    cand = set()
    for i in range(n):
        for j in np.argsort(-S[i])[:a.topk]:
            cand.add((min(i, int(j)), max(i, int(j))))
    ii, jj = np.where(np.triu(S >= 0.90, 1))
    for i, j in zip(ii, jj):
        cand.add((int(i), int(j)))
    cand = sorted(cand)
    print('candidate pairs', len(cand))

    # ---- verification
    bf = cv2.BFMatcher(cv2.NORM_L2)
    res = []
    t0 = time.time()
    for k, (i, j) in enumerate(cand):
        nin, ratio, mode = verify(D[i], D[j][0], D[j][1], bf)
        res.append((i, j, nin, ratio, mode, float(S[i, j])))
    print('verified in %.0fs' % (time.time() - t0))

    # ---- inlier histogram / threshold sensitivity
    inl = np.array([r[2] for r in res])
    bins = [0, 5, 8, 10, 12, 15, 20, 25, 30, 40, 60, 100, 1000]
    hist = {'%d-%d' % (bins[b], bins[b + 1] - 1): int(((inl >= bins[b]) & (inl < bins[b + 1])).sum())
            for b in range(len(bins) - 1)}
    print('inlier histogram', hist)

    # ---- groups for several thresholds
    def groups_at(thr):
        uf = UF(n)
        for (i, j, nin, *_ ) in res:
            if nin >= thr:
                uf.u(i, j)
        return [uf.f(x) for x in range(n)]

    sp_arr = np.array([s for s, _ in items])
    classes = [read_classes(root, s, f) for s, f in items]

    def leakage_stats(g):
        g = np.array(g)
        grp_splits = {}
        for x in range(n):
            grp_splits.setdefault(g[x], set()).add(sp_arr[x])
        multi = sum(1 for s in grp_splits.values() if len(s) > 1)
        d = {'n_groups': len(grp_splits), 'groups_spanning_multiple_splits': multi}
        for s in ('valid', 'test'):
            idx = [x for x in range(n) if sp_arr[x] == s]
            d['%s_imgs_with_train_copy' % s] = int(sum('train' in grp_splits[g[x]] for x in idx))
            d['%s_imgs_total' % s] = len(idx)
        idx = [x for x in range(n) if sp_arr[x] == 'test']
        d['test_imgs_with_valid_copy'] = int(sum('valid' in grp_splits[g[x]] for x in idx))
        return d

    sens = {str(t): leakage_stats(groups_at(t)) for t in (12, 15, 20, 25, 30, 40)}
    thr = a.inlier_thr
    g = groups_at(thr)
    main = leakage_stats(g)

    # per-class effective independent groups
    cls_groups = {NAMES[c]: len({g[x] for x in range(n) if c in classes[x]}) for c in range(10)}
    cls_imgs = {NAMES[c]: sum(1 for x in range(n) if c in classes[x]) for c in range(10)}
    # test leakage per class
    test_leak = {}
    grp_has_train = {}
    for x in range(n):
        if sp_arr[x] == 'train':
            grp_has_train[g[x]] = True
    for x in range(n):
        if sp_arr[x] == 'test':
            for c in set(classes[x]):
                tl = test_leak.setdefault(NAMES[c], [0, 0])
                tl[1] += 1
                tl[0] += int(g[x] in grp_has_train)
    sizes = {}
    for x in range(n):
        sizes[g[x]] = sizes.get(g[x], 0) + 1
    size_dist = {}
    for v in sizes.values():
        size_dist[v] = size_dist.get(v, 0) + 1

    summary = {'inlier_threshold': thr, 'n_images': n, 'candidate_pairs': len(cand),
               'inlier_histogram_candidate_pairs': hist, 'main': main,
               'threshold_sensitivity': sens,
               'group_size_distribution': dict(sorted(size_dist.items())),
               'per_class_images_all_splits': cls_imgs,
               'per_class_independent_groups': cls_groups,
               'test_images_with_train_copy_by_class [leaked,total]': test_leak}
    json.dump(summary, open(os.path.join(out, 'leakage_audit_summary.json'), 'w'), indent=2)

    with open(os.path.join(out, 'image_groups.csv'), 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['split', 'image', 'group_id', 'classes'])
        for x, (sp, f) in enumerate(items):
            w.writerow([sp, f, int(g[x]), ' '.join(map(str, sorted(set(classes[x]))))])
    with open(os.path.join(out, 'pairs_verified.csv'), 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['split_i', 'img_i', 'split_j', 'img_j', 'inliers', 'inlier_ratio', 'mode', 'thumb_ncc'])
        for (i, j, nin, ratio, mode, s) in res:
            if nin >= 12:
                w.writerow([items[i][0], items[i][1], items[j][0], items[j][1], nin, round(ratio, 3), mode, round(s, 3)])

    # ---- figures for manual verification
    rnd = random.Random(0)
    cross = [r for r in res if r[2] >= thr and {sp_arr[r[0]], sp_arr[r[1]]} == {'train', 'test'}]
    rnd.shuffle(cross)
    montage([(r[0], r[1], r[2], r[3], r[4]) for r in cross[:10]], items, root,
            os.path.join(out, 'figures', 'verified_test_train_pairs.png'), 'verified test-train pairs')
    border = [r for r in res if 12 <= r[2] < thr]
    rnd.shuffle(border)
    montage([(r[0], r[1], r[2], r[3], r[4]) for r in border[:8]], items, root,
            os.path.join(out, 'figures', 'borderline_pairs_12_to_thr.png'), 'borderline pairs')
    print(json.dumps(summary['main'], indent=1))
    print('sensitivity', json.dumps(sens))


if __name__ == '__main__':
    main()
