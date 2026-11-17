#!/usr/bin/env python3
"""Aggregate E1/E2 runs: Gate A (inflation), leak probe, Gate A2 (sibling confusions), seed noise floor.

Usage: python scripts/evaluation/aggregate_baseline.py --runs results/baseline
Reads  <runs>/*/metrics.json  (written by scripts/training/run_baseline.py)
Writes <runs>/baseline_summary.json and baseline_summary.md
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.confusion import error_stats  # noqa: E402


def ms(x):
    x = [v for v in x if v is not None]
    if not x:
        return None, None, 0
    return float(np.mean(x)), (float(np.std(x, ddof=1)) if len(x) > 1 else None), len(x)


def fmt(v, d=4):
    return 'n/a' if v is None else f'{v:.{d}f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    a = ap.parse_args()
    runs = [json.loads(p.read_text()) for p in sorted(Path(a.runs).glob('*/metrics.json'))]
    if not runs:
        sys.exit('no metrics.json found')
    E1 = [r for r in runs if r['mode'] == 'official']
    E2 = [r for r in runs if r['mode'] == 'clean']
    S = {'n_runs': len(runs), 'E1_runs': len(E1), 'E2_runs': len(E2)}
    md = ['# Baseline summary (E1 official vs E2 clean)', '']

    # ---- E1 -------------------------------------------------------------------------
    md += ['## E1 official split', '', '| tag | val mAP50-95 | test mAP50-95 | test macro(>=10 groups) | with-copy mAP50-95 | no-copy mAP50-95 |', '|---|---|---|---|---|---|']
    for r in E1:
        pw, pn = r.get('probe_test_with_train_copy'), r.get('probe_test_no_train_copy')
        md.append(f"| {r['tag']} | {fmt(r['val']['map5095'])} | {fmt(r['test']['map5095'])} | {fmt(r['test']['macro_eligible_ap5095'])} | "
                  f"{fmt(pw['map5095'] if pw else None)} | {fmt(pn['map5095'] if pn else None)} |")
    # ---- E2 -------------------------------------------------------------------------
    folds = sorted({r['fold'] for r in E2})
    per_fold = {}
    for f in folds:
        rs = [r for r in E2 if r['fold'] == f]
        per_fold[f] = {'map5095': np.mean([r['test']['map5095'] for r in rs]),
                       'macro': np.mean([r['test']['macro_eligible_ap5095'] for r in rs
                                         if r['test']['macro_eligible_ap5095'] is not None] or [np.nan]),
                       'n_seeds': len(rs)}
    md += ['', '## E2 clean folds (test, mean over seeds)', '', '| fold | seeds | test mAP50-95 | macro(>=10 groups) |', '|---|---|---|---|']
    for f in folds:
        md.append(f"| {f} | {per_fold[f]['n_seeds']} | {fmt(per_fold[f]['map5095'])} | {fmt(per_fold[f]['macro'])} |")
    m2, s2, n2 = ms([per_fold[f]['map5095'] for f in folds])
    mm2, sm2, _ = ms([per_fold[f]['macro'] for f in folds if not np.isnan(per_fold[f]['macro'])])
    md.append(f"| **mean (n={n2})** | | {fmt(m2)} +/- {fmt(s2)} | {fmt(mm2)} +/- {fmt(sm2)} |")
    S['E2_test_map5095_mean'], S['E2_test_map5095_sd'] = m2, s2
    S['E2_macro_mean'] = mm2

    # ---- Gate A -------------------------------------------------------------------------
    md += ['', '## Gate A (does the official split inflate results?)']
    if E1 and E2:
        e1 = float(np.mean([r['test']['map5095'] for r in E1]))
        e1m = [r['test']['macro_eligible_ap5095'] for r in E1 if r['test']['macro_eligible_ap5095'] is not None]
        e1m = float(np.mean(e1m)) if e1m else None
        d = e1 - m2
        S['gateA'] = {'E1_test_map5095': e1, 'E2_mean_map5095': m2, 'delta': d,
                      'E1_macro': e1m, 'E2_macro': mm2, 'delta_macro': (e1m - mm2) if (e1m is not None and mm2 is not None) else None}
        md.append(f"- E1 test mAP50-95 = {fmt(e1)} ; E2 clean mean = {fmt(m2)} ; delta = {fmt(d)}")
        if S['gateA']['delta_macro'] is not None:
            md.append(f"- macro(>=10 groups): E1 {fmt(e1m)} vs E2 {fmt(mm2)} ; delta = {fmt(S['gateA']['delta_macro'])}")
        md.append(f"- Gate A (E1 > E2): **{'PASS' if d > 0 else 'FAIL'}**")
    else:
        md.append('- need both E1 and E2 runs')
    # leak probe
    pw = [r['probe_test_with_train_copy']['map5095'] for r in E1 if 'probe_test_with_train_copy' in r]
    pn = [r['probe_test_no_train_copy']['map5095'] for r in E1 if 'probe_test_no_train_copy' in r]
    if pw and pn:
        S['leak_probe'] = {'with_copy_map5095': float(np.mean(pw)), 'no_copy_map5095': float(np.mean(pn)),
                           'gap': float(np.mean(pw) - np.mean(pn))}
        md.append(f"- Leak probe (official test, 48 with train copy vs 16 without): {fmt(np.mean(pw))} vs {fmt(np.mean(pn))} ; gap = {fmt(np.mean(pw) - np.mean(pn))} (16 images: interpret with caution)")

    # ---- Gate A2 --------------------------------------------------------------------------
    md += ['', '## Gate A2 (are errors sibling-class confusions?)']
    primary = []
    for f in folds:
        rs = sorted([r for r in E2 if r['fold'] == f], key=lambda r: r['seed'])
        primary.append(rs[0])
    if primary:
        M = np.sum([np.array(r['confusion']) for r in primary], axis=0)
        st = error_stats(M)
        S['gateA2_pooled_E2'] = st
        md.append(f"- Pooled E2 (primary seed per fold, conf>={primary[0]['confusion_conf_thr']}, IoU>=0.5): {st}")
        share = st['sibling_share_of_all_errors']
        md.append(f"- sibling confusions / all errors = {fmt(share)} -> Gate A2 (>= 0.20): **{'PASS' if (share is not None and share >= 0.20) else 'FAIL'}**")
        md.append(f"- sibling confusions / wrong-class matches = {fmt(st['sibling_share_of_wrong'])}")
    for r in E1:
        S.setdefault('E1_error_stats', {})[r['tag']] = r['error_stats']

    # ---- noise floor -------------------------------------------------------------------------
    md += ['', '## Seed noise floor (same split, different seeds)']
    groups = {}
    for r in runs:
        groups.setdefault((r['mode'], r['fold']), []).append(r)
    sds, sdm = [], []
    for k, rs in groups.items():
        if len({r['seed'] for r in rs}) > 1:
            _, s, _ = ms([r['test']['map5095'] for r in rs])
            _, sm, _ = ms([r['test']['macro_eligible_ap5095'] for r in rs])
            md.append(f"- {k}: seeds={sorted(r['seed'] for r in rs)} sd(mAP50-95)={fmt(s)} sd(macro)={fmt(sm)}")
            sds.append(s)
            sdm.append(sm)
    if sds:
        S['noise_floor'] = {'sd_map5095': float(np.mean([x for x in sds if x is not None])),
                            'sd_macro': float(np.mean([x for x in sdm if x is not None])) if any(x is not None for x in sdm) else None}
        md.append(f"- mean sd: mAP50-95 {fmt(S['noise_floor']['sd_map5095'])}, macro {fmt(S['noise_floor']['sd_macro'])} (Gate B threshold of +0.010 is only meaningful if this is well below 0.010)")
    else:
        md.append('- no multi-seed groups yet (run the noise-floor seeds)')

    S['environment'] = {k: runs[0].get(k) for k in ['ultralytics', 'torch', 'python', 'gpu', 'model', 'epochs_cfg', 'imgsz', 'batch']}
    md += ['', f"Environment: {S['environment']}"]
    Path(a.runs, 'baseline_summary.json').write_text(json.dumps(S, indent=2, default=float))
    Path(a.runs, 'baseline_summary.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
