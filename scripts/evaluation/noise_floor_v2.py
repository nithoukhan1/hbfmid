#!/usr/bin/env python3
"""Noise floor on the v2 clean folds from the plain baselines (seeds 42/43/44, 5 folds, YOLO11s + YOLOv8s).
Same convention as docs/RESULTS_CONTROLS.md: detectable effect = 2 * sigma_run * sqrt(2 / n_pairs), where sigma_run is the
per-run seed sd at a fixed fold (pooled over folds). Second estimate from the seed sd of the 5-fold mean (captures any
same-seed correlation across folds): 2 * sd_5fold * sqrt(2 / n_seeds). Also A/A differences (same model, two seeds).
Usage: python scripts/evaluation/noise_floor_v2.py --out results/analysis/noise_floor_v2"""
import argparse, glob, itertools, json
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('--out', default='results/analysis/noise_floor_v2'); a = ap.parse_args()
dirs = {'yolo11s.pt': ['results/v2_A', 'results/v2s_A'], 'yolov8s.pt': ['results/v2_B', 'results/v2s_B']}
X = {}
for model, ds in dirs.items():
    for d in ds:
        for p in glob.glob(f'{d}/*/metrics.json'):
            r = json.load(open(p))
            if r.get('mode') == 'clean' and r.get('protocol') == 'fixed_last' and r.get('arm', 'baseline') == 'baseline' and r['model'] == model:
                X.setdefault(model, {})[(r['fold'], r['seed'])] = r['test']['map5095']
seeds = [42, 43, 44]; folds = range(5); out = {'models': {}}; L = ['# Noise floor on v2 clean folds (plain baselines, 3 seeds x 5 folds)', '']
L += ['| model | fold | seed 42 | seed 43 | seed 44 | seed sd |', '|---|---|---|---|---|---|']
pooled_var, pooled_df = [], 0
for model, R in X.items():
    M = np.array([[R[(f, s)] for s in seeds] for f in folds])  # 5 x 3
    fold_sd = M.std(axis=1, ddof=1); sig_run = float(np.sqrt((fold_sd ** 2).mean())); pooled_var.append((fold_sd ** 2).sum() * 2); pooled_df += 10
    seed_means = M.mean(axis=0); sd5 = float(seed_means.std(ddof=1))
    aa = {f'{s1}-{s2}': float(seed_means[i] - seed_means[j]) for (i, s1), (j, s2) in itertools.combinations(enumerate(seeds), 2)}
    pair_diffs = np.concatenate([M[:, i] - M[:, j] for i, j in itertools.combinations(range(3), 2)])
    out['models'][model] = {'runs': M.tolist(), 'fold_seed_sd': fold_sd.tolist(), 'sigma_run_pooled': sig_run, 'df': 10,
                            'seed_means': seed_means.tolist(), 'sd_of_5fold_mean': sd5, 'sigma_run_implied_by_5fold': sd5 * np.sqrt(5),
                            'aa_5fold_mean_diffs': aa, 'aa_pair_sd': float(pair_diffs.std(ddof=1))}
    for f in folds: L.append(f'| {model} | {f} | ' + ' | '.join(f'{v:.4f}' for v in M[f]) + f' | {fold_sd[f]:.4f} |')
sig = float(np.sqrt(sum(pooled_var) / 2 / pooled_df / 1)) if False else None
# pooled within-fold variance across both models: sum of squares / df
ss = 0.0
for model in X:
    M = np.array(out['models'][model]['runs']); ss += ((M - M.mean(axis=1, keepdims=True)) ** 2).sum()
sig = float(np.sqrt(ss / pooled_df)); sd5_pooled = float(np.sqrt(np.mean([out['models'][m]['sd_of_5fold_mean'] ** 2 for m in X])))
de = lambda s, n: 2 * s * np.sqrt(2 / n)
out['pooled'] = {'sigma_run': sig, 'df': pooled_df, 'sd_of_5fold_mean': sd5_pooled,
                 'detectable_2sigma': {'from_sigma_run': {'1_seed_5_pairs': de(sig, 5), '3_seeds_15_pairs': de(sig, 15)},
                                       'from_5fold_mean_sd': {'1_seed': de(sd5_pooled, 1), '3_seeds': de(sd5_pooled, 3)}}}
dd = out['pooled']['detectable_2sigma']; gate = max(0.009, dd['from_sigma_run']['3_seeds_15_pairs'], dd['from_5fold_mean_sd']['3_seeds'])
out['gate_recomputed'] = float(np.ceil(gate * 1000) / 1000)
L += ['', '| model | sigma_run (pooled over folds, 10 df) | seed means (42/43/44) | sd of 5-fold mean | sigma implied (x sqrt 5) | A/A 5-fold-mean diffs | A/A pair sd |', '|---|---|---|---|---|---|---|']
for m, o in out['models'].items():
    L.append(f"| {m} | {o['sigma_run_pooled']:.4f} | {' / '.join(f'{v:.4f}' for v in o['seed_means'])} | {o['sd_of_5fold_mean']:.4f} | {o['sigma_run_implied_by_5fold']:.4f} | "
             + ', '.join(f'{k}: {v:+.4f}' for k, v in o['aa_5fold_mean_diffs'].items()) + f" | {o['aa_pair_sd']:.4f} |")
p = out['pooled']
L += ['', f"Pooled sigma_run = {p['sigma_run']:.4f} ({p['df']} df); pooled sd of the 5-fold mean = {p['sd_of_5fold_mean']:.4f}.",
      f"Detectable effect (2 sigma, paired): from sigma_run {dd['from_sigma_run']['1_seed_5_pairs'] * 100:.2f} pt (1 seed) / {dd['from_sigma_run']['3_seeds_15_pairs'] * 100:.2f} pt (3 seeds); "
      f"from the 5-fold-mean sd {dd['from_5fold_mean_sd']['1_seed'] * 100:.2f} pt (1 seed) / {dd['from_5fold_mean_sd']['3_seeds'] * 100:.2f} pt (3 seeds).",
      f"Recomputed gate (max of 0.9 pt and both 3-seed estimates, rounded up to 0.1 pt): {out['gate_recomputed'] * 100:.1f} pt."]
Path(a.out + '.json').write_text(json.dumps(out, indent=1)); Path(a.out + '.md').write_text('\n'.join(L) + '\n'); print('\n'.join(L))
