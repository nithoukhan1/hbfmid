#!/usr/bin/env python3
"""Blinded manual validation of the leakage audit (no GPU, read-only on the dataset).

  build : automatic checks + stratified sample of image pairs -> out/review.html (self-contained, images embedded),
          out/review_key.csv (strata, inlier counts: do NOT look before you finish reviewing), out/auto_checks.json
  score : read the reviewer's CSV (pair_id,label,note) + the key -> precision / recall tables with Wilson 95% intervals

Strata (66 pairs; file-name number = the 'NNN' of 'NNN_jpg.rf.hash'; pairs with different numbers are not trivially augmentation copies):
  T_weakest : the 12 test images with the WEAKEST direct link (>=25 inliers) to a train image
  T_random  : 8 more test images, random, with their strongest direct train link
  C_chain   : the test image(s) linked to train only through a chain: every edge of the shortest chain
  B_border  : 16 pairs of 25-39 inliers (8 of 25-29, 8 of 30-39), different file-name numbers  -> precision at the threshold
  R_below   : 16 pairs of 12-24 inliers (8 of 12-17, 8 of 18-24), different file-name numbers  -> missed duplicates (recall)
  N_hard    : 8 hard negatives: most similar thumbnails among pairs the audit did NOT link (different groups, <12 inliers)
  P_control : 4 positive controls (>=100 inliers, different file-name numbers)
Labels: same_image (same radiograph: identical, flipped, rotated, cropped, brightness/contrast change) | same_patient (same limb,
other exposure/view) | different | unsure.
Usage: python scripts/eda/audit_validation.py build --data-root "<.../Bone Fractures Detection>" --out results/eda/audit_validation
       python scripts/eda/audit_validation.py score --out results/eda/audit_validation --review <downloaded review_results.csv>"""
import argparse, base64, collections, csv, itertools, json, math, random, re
from pathlib import Path
import numpy as np

THR = 25


def pref(f):
    m = re.match(r'(.+?)_(?:jpg|jpeg|png)\.rf\.', f, re.I)
    return m.group(1) if m else f


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


HTML = """<!doctype html><html><head><meta charset="utf-8"><title>HBFMID audit review</title><style>
body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:0;background:#f4f4f2;color:#222}
#bar{position:sticky;top:0;background:#fff;border-bottom:1px solid #ccc;padding:10px 16px;z-index:5;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
#bar button{padding:6px 12px;font-size:14px;cursor:pointer}
.card{background:#fff;margin:14px 16px;padding:12px 14px;border:1px solid #ddd;border-radius:8px}
.card.done{border-color:#2e8b57}
.row{display:flex;gap:10px;flex-wrap:wrap}.p{text-align:center;font-size:12px;color:#666}
.p img{width:300px;height:300px;object-fit:contain;background:#000;cursor:zoom-in;display:block}
.p img.big{width:640px;height:640px;cursor:zoom-out}.m img{transform:scaleX(-1)}
label{margin-right:14px;font-size:15px;cursor:pointer}.id{font-weight:600;margin-bottom:6px}
input[type=text]{width:320px;padding:3px}#help{margin:12px 16px;font-size:14px;line-height:1.5}
</style></head><body><div id="bar"><b>HBFMID audit review</b><span id="prog"></span>
<button onclick="dl()">Download results CSV</button><span style="color:#666;font-size:13px">progress is kept in this browser; click an image to enlarge</span></div>
<div id="help"><b>For each pair decide whether the two images show the same underlying radiograph.</b> Ignore flips, rotation, crop, zoom, brightness or contrast changes (the third picture is B mirrored).<br>
<b>Same radiograph</b> = one exposure, possibly edited. <b>Same patient, other exposure</b> = same limb but a different view or time. <b>Different</b> = different patients.
<b>Unsure</b> if you cannot tell. Judge the bones only, not the labels. Do not open review_key.csv before you finish.</div><div id="list"></div>
<script>
const PAIRS=__DATA__;const KEY='hbfmid_audit_review_v1';let S={};try{S=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}
const L=[['same_image','Same radiograph'],['same_patient','Same patient, other exposure'],['different','Different patient'],['unsure','Unsure']];
function save(){try{localStorage.setItem(KEY,JSON.stringify(S))}catch(e){}}
function upd(){const n=PAIRS.filter(p=>S[p.id]&&S[p.id].l).length;document.getElementById('prog').textContent='Done '+n+' / '+PAIRS.length;}
function pan(src,t,m){return '<div class="p'+(m?' m':'')+'"><img src="data:image/jpeg;base64,'+src+'" onclick="this.classList.toggle(\\'big\\')">'+t+'</div>'}
const root=document.getElementById('list');
PAIRS.forEach((p,i)=>{const d=document.createElement('div');d.className='card';d.id='c'+p.id;
let h='<div class="id">'+(i+1)+' / '+PAIRS.length+' &nbsp; '+p.id+'</div><div class="row">'+pan(p.a,'A')+pan(p.b,'B')+pan(p.b,'B mirrored',1)+'</div><div style="margin-top:8px">';
L.forEach(x=>{h+='<label><input type="radio" name="r'+p.id+'" value="'+x[0]+'"> '+x[1]+'</label>'});
h+='<input type="text" placeholder="note (optional)" id="n'+p.id+'"></div>';d.innerHTML=h;root.appendChild(d);
const s=S[p.id]||{};if(s.l){d.querySelector('input[value="'+s.l+'"]').checked=true;d.classList.add('done')}
if(s.n)document.getElementById('n'+p.id).value=s.n;
d.querySelectorAll('input[type=radio]').forEach(r=>r.onchange=()=>{S[p.id]=S[p.id]||{};S[p.id].l=r.value;d.classList.add('done');save();upd()});
document.getElementById('n'+p.id).oninput=e=>{S[p.id]=S[p.id]||{};S[p.id].n=e.target.value;save()}});
upd();
function dl(){let c='pair_id,label,note\\n';PAIRS.forEach(p=>{const s=S[p.id]||{};c+=p.id+','+(s.l||'')+',"'+(s.n||'').replace(/"/g,'""')+'"\\n'});
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([c],{type:'text/csv'}));a.download='review_results.csv';a.click()}
</script></body></html>"""


def build(a):
    import cv2
    out = Path(a.out).resolve(); root = Path(a.data_root).resolve()
    assert root not in out.parents and out != root, 'output must not be inside the dataset folder'
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    G = list(csv.DictReader(open(a.groups))); P = list(csv.DictReader(open(a.pairs)))
    grp = {(g['split'], g['image']): g['group_id'] for g in G}
    cls = {(g['split'], g['image']): g['classes'] for g in G}
    E = []  # edges
    for r in P:
        E.append({'a': (r['split_i'], r['img_i']), 'b': (r['split_j'], r['img_j']), 'inl': int(float(r['inliers'])),
                  'mode': r['mode'], 'ncc': r['thumb_ncc']})
    dp = {frozenset([e['a'], e['b']]): e for e in E}
    # ---- automatic checks -------------------------------------------------------------------------------------------
    sets = collections.defaultdict(list)
    for k, c in cls.items():
        if k[0] == 'train':
            sets[(pref(k[1]), c)].append(k)
    s3 = [v for v in sets.values() if len(v) == 3]       # 1 original x 3 Roboflow augmentations = known same radiograph
    pr = [p for v in s3 for p in itertools.combinations(v, 2)]
    auto = {'aug_copy_sets_of_3': len(s3), 'sets_fully_in_one_group': sum(len({grp[x] for x in v}) == 1 for v in s3),
            'aug_pairs': len(pr), 'aug_pairs_direct_ge25': sum(dp.get(frozenset(p), {'inl': 0})['inl'] >= THR for p in pr)}
    best = {}
    adj = collections.defaultdict(list)
    for e in E:
        if e['inl'] >= THR:
            adj[e['a']].append((e['b'], e)); adj[e['b']].append((e['a'], e))
        for x, y in ((e['a'], e['b']), (e['b'], e['a'])):
            if x[0] == 'test' and y[0] == 'train' and e['inl'] >= THR and e['inl'] > best.get(x, (0,))[0]:
                best[x] = (e['inl'], e)
    test = [k for k in grp if k[0] == 'test']
    train_groups = {grp[k] for k in grp if k[0] == 'train'}
    leaked = [t for t in test if grp[t] in train_groups]
    direct = [t for t in leaked if t in best]
    chain_only = [t for t in leaked if t not in best]
    auto.update({'test_images': len(test), 'test_in_group_with_train': len(leaked), 'test_direct_link_ge25': len(direct),
                 'test_chain_only': len(chain_only), 'weakest_direct_inliers': sorted(best[t][0] for t in direct)[:12]})
    # ---- sampling ---------------------------------------------------------------------------------------------------
    used = set(); rows = []

    def add(st, e, note=''):
        k = frozenset([e['a'], e['b']])
        if k in used:
            return False
        used.add(k); rows.append({'stratum': st, 'a': e['a'], 'b': e['b'], 'inl': e['inl'], 'mode': e['mode'], 'ncc': e['ncc'], 'note': note})
        return True
    order = sorted(direct, key=lambda t: (best[t][0], t))
    for t in order[:a.n_weak]:
        add('T_weakest', best[t][1])
    rest = order[a.n_weak:]; rng.shuffle(rest)
    for t in rest[:a.n_rand]:
        add('T_random', best[t][1])
    for t in chain_only:                                  # shortest chain test -> ... -> train over edges >= THR
        prev = {t: None}; dq = collections.deque([t]); goal = None
        while dq and goal is None:
            u = dq.popleft()
            for v, e in adj[u]:
                if v not in prev:
                    prev[v] = (u, e); dq.append(v)
                    if v[0] == 'train':
                        goal = v; break
        while goal is not None and prev[goal] is not None:
            u, e = prev[goal]; add('C_chain', e); goal = u

    def band(lo, hi, st, n):
        c = [e for e in E if lo <= e['inl'] <= hi and pref(e['a'][1]) != pref(e['b'][1]) and frozenset([e['a'], e['b']]) not in used]
        rng.shuffle(c)
        for e in c[:n]:
            add(st, e)
    band(25, 29, 'B_border', a.n_band); band(30, 39, 'B_border', a.n_band)
    band(12, 17, 'R_below', a.n_band); band(18, 24, 'R_below', a.n_band)
    band(100, 10 ** 6, 'P_control', a.n_pos)
    # hard negatives from thumbnails (not linked by the audit, different groups, most similar)
    keys = [(g['split'], g['image']) for g in G]
    def thumb(k):
        im = cv2.imread(str(root / k[0] / 'images' / k[1]), cv2.IMREAD_GRAYSCALE)
        v = cv2.resize(im, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32).ravel(); v -= v.mean()
        return v / (np.linalg.norm(v) + 1e-6)
    T = np.stack([thumb(k) for k in keys]); T4 = T.reshape(-1, 32, 32)
    S = np.max([T @ np.stack([x.ravel() for x in (T4 if f == 0 else T4[:, ::-1, :] if f == 1 else T4[:, :, ::-1] if f == 2 else T4[:, ::-1, ::-1])]).T for f in range(4)], axis=0)
    gi = np.array([int(grp[k]) for k in keys])
    iu = np.triu_indices(len(keys), 1); sc = S[iu]; idx = np.argsort(-sc)
    cand, seen = [], set()
    for m in idx:
        i, j = iu[0][m], iu[1][m]
        if gi[i] == gi[j] or frozenset([keys[i], keys[j]]) in dp:
            continue
        gp = frozenset([gi[i], gi[j]])
        if gp in seen:
            continue
        seen.add(gp); cand.append((keys[i], keys[j], float(sc[m])))
        if len(cand) >= 60:
            break
    rng.shuffle(cand)
    for ka, kb, s in cand[:a.n_neg]:
        add('N_hard', {'a': ka, 'b': kb, 'inl': -1, 'mode': '', 'ncc': f'{s:.3f}'})
    # ---- output -----------------------------------------------------------------------------------------------------
    rng.shuffle(rows)
    data, keyrows = [], []
    def enc(k):
        im = cv2.imread(str(root / k[0] / 'images' / k[1]))
        h, w = im.shape[:2]; sc_ = 320 / max(h, w)
        im = cv2.resize(im, (int(w * sc_), int(h * sc_)), interpolation=cv2.INTER_AREA)
        return base64.b64encode(cv2.imencode('.jpg', im, [cv2.IMWRITE_JPEG_QUALITY, 82])[1].tobytes()).decode()
    for n, r in enumerate(rows, 1):
        pid = f'P{n:03d}'
        sw = rng.random() < 0.5
        x, y = (r['b'], r['a']) if sw else (r['a'], r['b'])
        data.append({'id': pid, 'a': enc(x), 'b': enc(y)})
        keyrows.append({'pair_id': pid, 'stratum': r['stratum'], 'split_a': x[0], 'img_a': x[1], 'split_b': y[0], 'img_b': y[1],
                        'inliers': r['inl'], 'mode': r['mode'], 'thumb_ncc': r['ncc'], 'group_a': grp[x], 'group_b': grp[y],
                        'classes_a': cls[x], 'classes_b': cls[y]})
    (out / 'review.html').write_text(HTML.replace('__DATA__', json.dumps(data)), encoding='utf-8')
    with open(out / 'review_key.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(keyrows[0])); w.writeheader(); w.writerows(keyrows)
    cnt = collections.Counter(r['stratum'] for r in rows)
    auto['sample'] = dict(cnt); auto['n_pairs'] = len(rows); auto['seed'] = a.seed
    (out / 'auto_checks.json').write_text(json.dumps(auto, indent=1))
    print(json.dumps(auto, indent=1)); print('html MB:', round((out / 'review.html').stat().st_size / 1e6, 2))


def score(a):
    out = Path(a.out)
    key = {r['pair_id']: r for r in csv.DictReader(open(out / 'review_key.csv'))}
    rev = {r['pair_id']: r['label'].strip() for r in csv.DictReader(open(a.review))}
    miss = [p for p in key if not rev.get(p)]
    assert not miss, f'{len(miss)} pairs unlabeled, e.g. {miss[:5]}'
    by = collections.defaultdict(collections.Counter)
    for p, r in key.items():
        by[r['stratum']][rev[p]] += 1
    L = ['| stratum | n | same_image | same_patient | different | unsure | share same_image (Wilson 95%) | share same radiograph or patient (Wilson 95%) |', '|---|---|---|---|---|---|---|---|']
    res = {}
    for st in ['T_weakest', 'T_random', 'C_chain', 'B_border', 'R_below', 'N_hard', 'P_control']:
        c = by.get(st)
        if not c:
            continue
        n = sum(c.values()); k1 = c['same_image']; k2 = c['same_image'] + c['same_patient']
        res[st] = {'n': n, **{x: c[x] for x in ['same_image', 'same_patient', 'different', 'unsure']}, 'ci_same_image': wilson(k1, n), 'ci_same_any': wilson(k2, n)}
        L.append(f"| {st} | {n} | {k1} | {c['same_patient']} | {c['different']} | {c['unsure']} | {k1 / n:.2f} {wilson(k1, n)} | {k2 / n:.2f} {wilson(k2, n)} |")
    (out / 'validation_results.json').write_text(json.dumps(res, indent=1)); (out / 'validation_results.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest='cmd', required=True)
    b = sp.add_parser('build'); b.add_argument('--data-root', required=True); b.add_argument('--out', required=True)
    b.add_argument('--groups', default='results/eda/image_groups.csv'); b.add_argument('--pairs', default='results/eda/pairs_verified.csv')
    b.add_argument('--seed', type=int, default=20261121); b.add_argument('--n-weak', type=int, default=12); b.add_argument('--n-rand', type=int, default=8)
    b.add_argument('--n-band', type=int, default=8); b.add_argument('--n-pos', type=int, default=4); b.add_argument('--n-neg', type=int, default=8)
    s = sp.add_parser('score'); s.add_argument('--out', required=True); s.add_argument('--review', required=True)
    a = ap.parse_args(); build(a) if a.cmd == 'build' else score(a)
