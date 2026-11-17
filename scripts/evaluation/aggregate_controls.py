#!/usr/bin/env python3
"""Controls after the first baseline: (1) leakage attribution E1 vs E1b (official split de-duplicated, same test images),
(2) seed noise under the fixed-epoch protocol (last.pt) vs validation-selected best.pt.
Usage: python scripts/evaluation/aggregate_controls.py --runs results/controls --e1 results/baseline/baseline_results/official_yolov8s_s42/metrics.json"""
import argparse, json
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument('--runs', required=True); ap.add_argument('--e1', required=True)
a = ap.parse_args()
R = [json.loads(p.read_text()) for p in sorted(Path(a.runs).glob('*/metrics.json'))]
e1 = json.loads(Path(a.e1).read_text())
sd = lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else None
f = lambda v, d=4: 'n/a' if v is None else f'{v:.{d}f}'
md = ['# Controls summary', '']
dd = [r for r in R if r['mode'] == 'official_dedup']
md += ['## 1. Leakage attribution on the SAME 64 official test images', '',
       '| run | train imgs | test mAP50 | test mAP50-95 | macro(>=10 groups) |', '|---|---|---|---|---|',
       f"| E1 official (leaky train+val) | {e1['n_images']['train']} | {f(e1['test']['map50'])} | {f(e1['test']['map5095'])} | {f(e1['test']['macro_eligible_ap5095'])} |"]
for r in dd:
    md.append(f"| E1b official de-duplicated, seed {r['seed']} ({r['protocol']}) | {r['n_images']['train']} | {f(r['test']['map50'])} | {f(r['test']['map5095'])} | {f(r['test']['macro_eligible_ap5095'])} |")
S = {'E1b_runs': len(dd)}
if dd:
    drop = e1['test']['map5095'] - float(np.mean([r['test']['map5095'] for r in dd]))
    md += ['', f'- drop E1 -> E1b (same test images, leakage removed from train/val) = {drop:.4f} mAP50-95', '']
    S['leak_drop_map5095'] = drop
clean = [r for r in R if r['mode'] == 'clean']
md += ['## 2. Seed noise (clean folds, several seeds, same split)', '', '| fold | seeds | last.pt test mAP50-95 (sd) | best.pt test mAP50-95 (sd) | best_epochs |', '|---|---|---|---|---|']
sl, sb = [], []
for fold in sorted({r['fold'] for r in clean}):
    g = [r for r in clean if r['fold'] == fold and 'test_best' in r]
    if len(g) < 2: continue
    L = [r['test']['map5095'] for r in g]; B = [r['test_best']['map5095'] for r in g]
    sl.append(sd(L)); sb.append(sd(B))
    md.append(f"| {fold} | {[r['seed'] for r in g]} | {f(np.mean(L))} ({f(sd(L))}) | {f(np.mean(B))} ({f(sd(B))}) | {[r.get('best_epoch') for r in g]} |")
if sl:
    ml, mb = float(np.mean(sl)), float(np.mean(sb))
    md += ['', f'- mean seed sd: last.pt {ml:.4f} vs best.pt {mb:.4f} (early-stopping baseline run earlier: 0.0211)']
    for n in (1, 3):
        d = np.sqrt(2) * ml / np.sqrt(5 * n)
        md.append(f'- paired difference of two arms over 5 folds x {n} seed(s): sd about {d:.4f}; a ~2-sigma effect size is about {2*d:.4f} mAP50-95')
    S.update(sd_last=ml, sd_best=mb)
Path(a.runs, 'controls_summary.md').write_text('\n'.join(md) + '\n'); Path(a.runs, 'controls_summary.json').write_text(json.dumps(S, indent=1))
print('\n'.join(md))
