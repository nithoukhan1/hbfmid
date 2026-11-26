#!/usr/bin/env python3
"""HBFMID-Clean v2 benchmark: the 5-detector table under the rules fixed in docs/BENCHMARK_V2_PLAN.md (2026-11-24).

Run sets are explicit (directories per detector), never a glob over all results.
  v2 rows : data_splits_v2/clean5, 3 seeds x 5 folds = 15 runs per detector
  official: data_splits/official, 3 seeds (the official split is the same for v1 and v2 grouping)
Rules (pre-declared):
  per detector  mean over folds of the seed-averaged fold means (= mean of the 15 runs when complete), sd over folds,
                seed sd of the 5-fold mean, official 3-seed mean, inflation = official / v2
  vs YOLOv8s    paired by fold (seed-averaged): mean, t95 interval (4 df), folds positive; 'better' / 'worse' only if
                |diff| >= gate (results/analysis/noise_floor_v2.json, 1.8 pt) AND the same sign in >= 4/5 folds
  failed run    test mAP50-95 < 0.01 or NaN final validation mAP50-95. Primary = as measured (failed runs included);
                secondary = failed runs removed (fold mean over the remaining seeds), labelled post-hoc. No re-runs.
  incomplete    fewer than 15 v2 runs: runs listed, no mean, no comparison, no verdict
Exploratory (labelled): Kendall tau of the official vs v2 ranking (only when every row is complete); official leak
  probe = test images with vs without a near-duplicate in train (48 vs 16 images, so noisy).
Checks: no duplicate (detector, split, fold, seed); within a fold every run has the same image counts and test GT boxes;
  Ultralytics runs that exported predictions: |offline evaluator - Ultralytics| on the run's own export (< 1e-3).
Usage: python scripts/evaluation/benchmark_v2.py [--out results/analysis/benchmark_v2]
"""
import argparse, csv, json, math
from datetime import date
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / 'results'
T4 = 2.776  # t(0.975, 4 df)
FOLDS, SEEDS = [0, 1, 2, 3, 4], [42, 43, 44]
RF_DIRS = ['rfd_A', 'rfd_B', 'rfd_C', 'rfd_D']
DETECTORS = [  # name, model id in metrics.json, v2 dirs, official dirs
    ('YOLOv8s', 'yolov8s.pt', ['v2_B', 'v2s_B'], ['bench_B']),
    ('YOLO11s', 'yolo11s.pt', ['v2_A', 'v2s_A'], ['bench_A']),
    ('YOLOv8m', 'yolov8m.pt', ['v2m'], ['bench_B', 'v2m']),
    ('RT-DETR-L', 'rtdetr-l.pt', ['v2r_A', 'v2r_B'], ['rtdetr_A', 'rtdetr_B']),
    ('RF-DETR-M', 'rfdetr-medium', RF_DIRS, RF_DIRS),
]
REF = 'YOLOv8s'


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float('nan')


def rel(p):
    return str(Path(p).relative_to(ROOT))


def load(model, dirs, mode):
    runs, dup, seen_dirs = {}, [], []
    for d in dirs:
        if not (RES / d).is_dir():
            continue
        seen_dirs.append(d)
        for p in sorted((RES / d).glob('*/metrics.json')):
            m = json.loads(p.read_text())
            if (m.get('model') != model or m.get('mode') != mode or m.get('protocol') != 'fixed_last'
                    or m.get('arm', 'baseline') != 'baseline'):
                continue
            key = (m.get('fold'), m['seed'])
            m['_dir'] = p.parent
            if key in runs:
                dup.append([rel(runs[key]['_dir']), rel(p.parent)])
            runs[key] = m
    return runs, dup, seen_dirs


def failed(m):
    return m['test']['map5095'] < 0.01 or math.isnan(num(m['val']['map5095']))


def health(m):
    """In-training log of an Ultralytics run (fp16 validation each epoch). None for RF-DETR (own log format)."""
    f = m['_dir'] / 'train' / 'results.csv'
    if 'ultralytics' not in m or not f.exists():
        return None
    rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(open(f))]
    vcols = [k for k in rows[0] if k.startswith('val/') and 'loss' in k]
    tcols = [k for k in rows[0] if k.startswith('train/') and 'loss' in k]
    vnan = [any(math.isnan(num(r[k])) for k in vcols) for r in rows]
    tnan = [any(math.isnan(num(r[k])) for k in tcols) for r in rows]
    perm = next((i + 1 for i in range(len(vnan)) if all(vnan[i:])), None)
    return {'epochs_logged': len(rows), 'val_nan_epochs': int(sum(vnan)), 'train_nan_epochs': int(sum(tnan)),
            'val_nan_permanent_from_epoch': perm, 'final_train_loss_sum': round(sum(num(rows[-1][k]) for k in tcols), 4)}


def fold_means(runs, get=lambda m: m['test']['map5095'], drop_failed=False):
    out = []
    for f in FOLDS:
        v = [get(runs[(f, s)]) for s in SEEDS if (f, s) in runs and not (drop_failed and failed(runs[(f, s)]))]
        out.append(float(np.mean(v)) if v else float('nan'))
    return out


def paired(x, y, gate):
    d = 100 * (np.array(x) - np.array(y))
    mean, half = float(d.mean()), float(T4 * d.std(ddof=1) / math.sqrt(len(d)))
    pos, neg, g = int((d > 0).sum()), int((d < 0).sum()), 100 * gate
    verdict = 'better' if (mean >= g and pos >= 4) else 'worse' if (mean <= -g and neg >= 4) else 'not distinguishable'
    return {'mean_pts': round(mean, 2), 'ci95_pts': [round(mean - half, 2), round(mean + half, 2)],
            'folds_positive': pos, 'folds_negative': neg, 'per_fold_pts': [round(float(v), 2) for v in d], 'verdict': verdict}


def kendall_tau(a, b):
    n, c, dc = len(a), 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (a[i] - a[j]) * (b[i] - b[j])
            c, dc = c + (s > 0), dc + (s < 0)
    return (c - dc) / (n * (n - 1) / 2)


def P(x):
    return f'{100 * x:.2f}'


ap = argparse.ArgumentParser()
ap.add_argument('--out', default=str(RES / 'analysis' / 'benchmark_v2'))
ap.add_argument('--gate', type=float, default=None, help='default: gate_recomputed in results/analysis/noise_floor_v2.json')
a = ap.parse_args()
gate = a.gate if a.gate is not None else json.loads((RES / 'analysis' / 'noise_floor_v2.json').read_text())['gate_recomputed']

rows, checks = {}, {'duplicates': [], 'fold_consistency': [], 'parity': {}}
for name, model, v2dirs, offdirs in DETECTORS:
    v2, d1, seen = load(model, v2dirs, 'clean')
    off, d2, _ = load(model, offdirs, 'official')
    checks['duplicates'] += [[name] + d for d in d1 + d2]
    missing = [[f, s] for f in FOLDS for s in SEEDS if (f, s) not in v2]
    r = {'model': model, 'v2_dirs': seen, 'n_v2_runs': len(v2), 'complete': not missing, 'missing_v2_runs': missing,
         'failed_runs': [], 'runs': {}, 'official': {}}
    for (f, s), m in sorted(v2.items()):
        h = health(m)
        r['runs'][f'f{f}_s{s}'] = {'tag': m['tag'], 'dir': rel(m['_dir']), 'fold': f, 'seed': s,
                                   'test_map5095': m['test']['map5095'], 'val_map5095': num(m['val']['map5095']),
                                   'failed': failed(m), 'health': h}
        if failed(m):
            r['failed_runs'].append({'tag': m['tag'], 'fold': f, 'seed': s, 'test_map5095': m['test']['map5095'],
                                     'val_map5095': num(m['val']['map5095']), 'health': h,
                                     'best_pt_test_map5095_diagnostic_only': m.get('test_best', {}).get('map5095'),
                                     'best_epoch': m.get('best_epoch')})
    hs = [x['health'] for x in r['runs'].values() if x['health']]
    r['health_summary'] = None if not hs else {
        'n': len(hs), 'runs_with_any_val_nan_epoch': sum(h['val_nan_epochs'] > 0 for h in hs),
        'runs_with_permanent_val_nan': sum(h['val_nan_permanent_from_epoch'] is not None for h in hs),
        'runs_with_train_nan_epoch': sum(h['train_nan_epochs'] > 0 for h in hs)}
    if r['complete']:
        fm = fold_means(v2)
        per_seed = [float(np.mean([v2[(f, s)]['test']['map5095'] for f in FOLDS])) for s in SEEDS]
        r.update({'v2_map5095': float(np.mean(fm)), 'v2_fold_means': fm, 'v2_sd_over_folds': float(np.std(fm, ddof=1)),
                  'v2_seed_means': per_seed, 'v2_seed_sd': float(np.std(per_seed, ddof=1)),
                  'v2_macro_eligible_ap5095': float(np.mean(fold_means(v2, lambda m: m['test']['macro_eligible_ap5095']))),
                  'v2_map50': float(np.mean(fold_means(v2, lambda m: m['test']['map50'])))})
        if r['failed_runs']:
            fm2 = fold_means(v2, drop_failed=True)
            r.update({'v2_map5095_failed_removed_posthoc': float(np.mean(fm2)), 'v2_fold_means_failed_removed_posthoc': fm2})
    if off:
        ks = sorted(off, key=lambda k: k[1])
        v = [off[k]['test']['map5095'] for k in ks]
        gaps = [100 * (off[k]['probe_test_with_train_copy']['map5095'] - off[k]['probe_test_no_train_copy']['map5095'])
                for k in ks if 'probe_test_with_train_copy' in off[k]]
        r['official'] = {'seeds': [k[1] for k in ks], 'dirs': sorted({rel(off[k]['_dir'].parent) for k in ks}),
                         'map5095_per_seed': v, 'map5095_mean': float(np.mean(v)),
                         'map5095_sd': float(np.std(v, ddof=1)) if len(v) > 1 else None,
                         'map50_mean': float(np.mean([off[k]['test']['map50'] for k in ks])),
                         'leak_probe_gap_pts_per_seed': [round(g, 2) for g in gaps],
                         'leak_probe_gap_pts_mean': round(float(np.mean(gaps)), 2) if gaps else None}
        if r['complete'] and len(v) == 3:
            r['inflation_official_over_v2'] = r['official']['map5095_mean'] / r['v2_map5095']
    allruns = list(v2.values()) + list(off.values())
    par = [m.get('test_offline_parity') for m in allruns if 'ultralytics' in m]
    ok = [p['abs_diff_map5095'] for p in par if isinstance(p, dict) and 'abs_diff_map5095' in p]
    err = [m['tag'] for m in allruns if isinstance(m.get('test_offline_parity'), dict) and 'error' in m['test_offline_parity']]
    if par:
        checks['parity'][name] = {'n_checked': len(ok), 'max_abs_diff_map5095': max(ok) if ok else None, 'no_export': err,
                                  'n_older_than_parity_check': sum(p is None for p in par)}
    rows[name] = (r, v2)

sig = {}
for name, (r, v2) in rows.items():
    for (f, s), m in sorted(v2.items()):
        key = json.dumps([m['n_images'], m.get('test_gt_boxes_per_class')], sort_keys=True)
        if f not in sig:
            sig[f] = (key, f'{name} seed {s}')
        elif sig[f][0] != key:
            checks['fold_consistency'].append(f'fold {f}: {name} seed {s} differs from {sig[f][1]}')

ref_r, ref_v2 = rows[REF]
comparisons = {}
for name, (r, v2) in rows.items():
    if name == REF or not (r['complete'] and ref_r['complete']):
        continue
    c = {'as_measured_primary': paired(r['v2_fold_means'], ref_r['v2_fold_means'], gate)}
    if r['failed_runs'] or ref_r['failed_runs']:
        c['failed_removed_posthoc'] = paired(fold_means(v2, drop_failed=True), fold_means(ref_v2, drop_failed=True), gate)
    comparisons[name] = c

names = list(rows)
all_complete = all(rows[n][0]['complete'] and len(rows[n][0]['official'].get('seeds', [])) == 3 for n in names)
rank = None
if all_complete:
    o = [rows[n][0]['official']['map5095_mean'] for n in names]
    v = [rows[n][0]['v2_map5095'] for n in names]
    rank = {'official_order': [names[i] for i in np.argsort(o)[::-1]], 'v2_order': [names[i] for i in np.argsort(v)[::-1]],
            'kendall_tau': kendall_tau(o, v), 'label': 'exploratory'}

out = {'generated': date.today().isoformat(), 'gate': gate, 'reference': REF, 't95_4df': T4,
       'detectors': {n: rows[n][0] for n in names}, 'comparisons_vs_reference': comparisons,
       'rank_agreement_exploratory': rank, 'checks': checks}
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out + '.json').write_text(json.dumps(out, indent=1, default=str))

L = [f'# HBFMID-Clean v2 benchmark ({out["generated"]}, scripts/evaluation/benchmark_v2.py)', '',
     f'Rules fixed before the runs (docs/BENCHMARK_V2_PLAN.md). mAP50-95 on the test fold, in points (x100). '
     f'Gate {100 * gate:.1f} pt: a verdict needs |diff| >= gate AND the same sign in >= 4/5 folds. '
     f'"Failed" = test mAP50-95 < 1 pt or NaN validation; primary numbers include failed runs as measured.', '',
     '## Main table', '',
     '| detector | v2 runs | failed | v2 mean | sd over folds | seed sd | v2 macro AP (8 cl.) | v2 mAP50 | official (3 seeds) | official / v2 |',
     '|---|---|---|---|---|---|---|---|---|---|']
for n in names:
    r = rows[n][0]
    o = r['official']
    if o and o.get('map5095_sd') is not None:
        offs = f"{P(o['map5095_mean'])} (sd {P(o['map5095_sd'])})"
    else:
        offs = P(o['map5095_mean']) if o else '-'
    if r['complete']:
        L.append(f"| {n} | 15/15 | {len(r['failed_runs'])} | {P(r['v2_map5095'])} | {P(r['v2_sd_over_folds'])} | "
                 f"{P(r['v2_seed_sd'])} | {P(r['v2_macro_eligible_ap5095'])} | {P(r['v2_map50'])} | {offs} | "
                 f"{r.get('inflation_official_over_v2', float('nan')):.2f}x |")
    else:
        L.append(f"| {n} | {r['n_v2_runs']}/15 INCOMPLETE | {len(r['failed_runs'])} | pending | | | | | {offs} | |")
for n in names:
    if 'v2_map5095_failed_removed_posthoc' in rows[n][0]:
        L += ['', f"Post-hoc (failed runs removed): {n} v2 mean {P(rows[n][0]['v2_map5095_failed_removed_posthoc'])}."]
L += ['', f'## Against {REF} (paired by fold, seed-averaged; t95 with 4 df)', '',
      '| detector | view | diff (pt) | 95% CI (pt) | folds +/- | per fold (pt) | verdict |', '|---|---|---|---|---|---|---|']
for n, c in comparisons.items():
    for view, v in c.items():
        L.append(f"| {n} | {view.replace('_', ' ')} | {v['mean_pts']:+.2f} | [{v['ci95_pts'][0]:+.2f}, {v['ci95_pts'][1]:+.2f}] | "
                 f"{v['folds_positive']}/{v['folds_negative']} | {', '.join(f'{x:+.1f}' for x in v['per_fold_pts'])} | {v['verdict']} |")
for n in names:
    if not rows[n][0]['complete']:
        L.append(f"| {n} | - | pending ({rows[n][0]['n_v2_runs']}/15 runs) | | | | |")
L += ['', '## Per fold (seed-averaged mAP50-95, pt)', '',
      '| detector | ' + ' | '.join(f'fold {f}' for f in FOLDS) + ' |', '|---|' + '---|' * len(FOLDS)]
for n in names:
    r = rows[n][0]
    if r['complete']:
        L.append(f'| {n} | ' + ' | '.join(P(x) for x in r['v2_fold_means']) + ' |')
    else:
        cells = []
        for f in FOLDS:
            got = [f"s{x['seed']} {P(x['test_map5095'])}" for x in r['runs'].values() if x['fold'] == f]
            cells.append(', '.join(got) or '-')
        L.append(f'| {n} (runs so far) | ' + ' | '.join(cells) + ' |')
L += ['', '## Failed runs and training health (Ultralytics in-training fp16 validation log)', '']
for n in names:
    r = rows[n][0]
    hs = r['health_summary']
    line = f"- {n}: failed {len(r['failed_runs'])}/{r['n_v2_runs']}"
    if hs:
        line += (f"; runs with any NaN validation epoch {hs['runs_with_any_val_nan_epoch']}/{hs['n']}, "
                 f"NaN validation to the end {hs['runs_with_permanent_val_nan']}/{hs['n']}, "
                 f"NaN train-loss epoch {hs['runs_with_train_nan_epoch']}/{hs['n']}")
    else:
        line += '; no Ultralytics log (RF-DETR)'
    L.append(line)
    for fr in r['failed_runs']:
        h = fr['health'] or {}
        best = fr['best_pt_test_map5095_diagnostic_only']
        L.append(f"  - {fr['tag']}: test {P(fr['test_map5095'])}, val {fr['val_map5095']}; NaN validation from epoch "
                 f"{h.get('val_nan_permanent_from_epoch')} to the end, NaN train-loss epochs {h.get('train_nan_epochs')}, "
                 f"final train loss {h.get('final_train_loss_sum')} (finite); best.pt (epoch {fr['best_epoch']}) test "
                 f"{P(best) if best is not None else '-'}: diagnostic only, not part of the protocol")
L += ['', '## Exploratory (labelled as such)', '']
for n in names:
    o = rows[n][0]['official']
    if o and o.get('leak_probe_gap_pts_mean') is not None:
        L.append(f"- Official leak probe {n}: with-copy minus without-copy test images {o['leak_probe_gap_pts_mean']:+.2f} pt "
                 f"(per seed {o['leak_probe_gap_pts_per_seed']})")
L.append('- Rank agreement official vs v2: ' + (f"Kendall tau {rank['kendall_tau']:+.2f}; official {rank['official_order']}; "
                                                f"v2 {rank['v2_order']}" if rank else 'pending (all rows must be complete)'))
L += ['', '## Checks', '', f"- duplicates: {checks['duplicates'] or 'none'}",
      f"- fold consistency (image counts and test GT boxes): {checks['fold_consistency'] or 'OK'}"]
for n, p in checks['parity'].items():
    L.append(f"- evaluator parity {n}: {p['n_checked']} runs, max |diff| {p['max_abs_diff_map5095']}"
             + (f"; no export (failed run): {p['no_export']}" if p['no_export'] else '')
             + (f"; {p['n_older_than_parity_check']} runs older than the parity check" if p['n_older_than_parity_check'] else ''))
Path(a.out + '.md').write_text('\n'.join(L) + '\n')
print('\n'.join(L))
