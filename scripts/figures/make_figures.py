#!/usr/bin/env python3
"""Paper figures from saved result files only (no dataset images). Usage: python3 scripts/figures/make_figures.py --out results/figures"""
import argparse, json
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument('--out', default='results/figures'); a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 150})
C = {'official': '#b2182b', 'clean_v1': '#ef8a62', 'clean_v2': '#2166ac'}

# Fig 1: score chain official -> clean v1 -> clean v2 (all 3 seeds; clean = 5-fold mean; error = sd over seeds)
B = json.load(open('results/benchmark_all.json')); T3 = json.load(open('results/analysis/v2_three_seeds.json'))  # clean v1 = benchmark_all 'clean' (3 seeds); v2 = notebooks 10+11
models = [('yolov8s.pt', 'YOLOv8s'), ('yolo11s.pt', 'YOLO11s')]; w = 0.35
fig, ax = plt.subplots(figsize=(4.2, 3.2))
for i, (m, nm) in enumerate(models):
    vals = [B[m]['official']['map5095']['mean'], B[m]['clean']['map5095']['mean'], T3[m]['mean']]
    errs = [B[m]['official']['map5095']['sd'], B[m]['clean']['seed_sd_of_fold_mean'], T3[m]['seed_sd']]
    ax.bar(np.arange(3) + (i - 0.5) * w, vals, w, yerr=errs, capsize=2, label=nm, color=['#999999', '#777777'][i] if False else None, alpha=0.9)
    for x, v, e in zip(np.arange(3) + (i - 0.5) * w, vals, errs): ax.text(x, v + e + 0.012, f'{v:.2f}', ha='center', fontsize=7)
ax.set_xticks(range(3)); ax.set_xticklabels(['Official\nsplit', 'Clean v1\n(SIFT groups)', 'Clean v2\n(+ thumbnail links)']); ax.set_ylabel('mAP50-95'); ax.set_ylim(0, 0.62); ax.legend(frameon=False)
ax.set_title('Same detectors, stricter duplicate removal', fontsize=9); ax.text(0.99, -0.34, 'all bars: 3 seeds (42-44); clean = mean over 5 folds; error bars = sd over seeds', transform=ax.transAxes, ha='right', fontsize=5.5, color='#555555'); fig.tight_layout(); fig.savefig(out / 'fig1_score_chain.png'); fig.savefig(out / 'fig1_score_chain.pdf'); plt.close(fig)

# Fig 2: box recall vs similarity of the test image to the training set
S = json.load(open('results/analysis/similarity_recall.json')); bins = ['<0.80', '0.80-0.90', '0.90-0.95', '>=0.95']
fig, ax = plt.subplots(figsize=(4.2, 3))
for proto, lab in [('official', 'Official'), ('clean_v1', 'Clean v1'), ('clean_v2', 'Clean v2')]:
    xs, ys, ns = [], [], []
    for i, b in enumerate(bins):
        r = [S[f'{proto}|{m}|{b}'] for m in ('yolo11s.pt', 'yolov8s.pt') if f'{proto}|{m}|{b}' in S]
        if r: xs.append(i); ys.append(np.mean([x['recall'] for x in r])); ns.append(r[0]['imgs'])
    ax.plot(xs, ys, marker='o', color=C[proto], label=lab)
    dy = {'official': 7, 'clean_v1': -12, 'clean_v2': -22}[proto]
    for x, y, n in zip(xs, ys, ns): ax.annotate(f'n={n}', (x, y), textcoords='offset points', xytext=(0, dy), ha='center', fontsize=6, color=C[proto])
ax.set_xticks(range(4)); ax.set_xticklabels(bins); ax.set_xlabel('max similarity to training set (thumbnail NCC)'); ax.set_ylabel('box recall (conf 0.25, IoU 0.5)'); ax.set_ylim(0, 1.15); ax.legend(frameon=False, loc='lower right'); ax.text(0.02, 0.02, 'n = test images per bin', transform=ax.transAxes, fontsize=6, color='#555555')
fig.tight_layout(); fig.savefig(out / 'fig2_similarity_recall.png'); fig.savefig(out / 'fig2_similarity_recall.pdf'); plt.close(fig)

# Fig 3: group sizes (v1 vs v2) and image share held by large groups
G1 = pd.read_csv('results/eda/image_groups.csv').groupby('group_id').size(); G2 = pd.read_csv('results/eda/grouping_v2/image_groups_v2_ncc0.95.csv').groupby('group_id').size()
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.8))
mx = 31; axs[0].hist(G1.clip(upper=mx), bins=np.arange(0.5, mx + 1.5), alpha=0.6, color=C['clean_v1'], label=f'v1: {len(G1)} groups'); axs[0].hist(G2.clip(upper=mx), bins=np.arange(0.5, mx + 1.5), alpha=0.6, color=C['clean_v2'], label=f'v2: {len(G2)} groups')
axs[0].set_xlabel('images per group'); axs[0].set_ylabel('groups'); axs[0].legend(frameon=False)
s = np.sort(G2.values)[::-1]; axs[1].plot(np.arange(1, len(s) + 1) / len(s) * 100, np.cumsum(s) / s.sum() * 100, color=C['clean_v2'])
axs[1].plot([0, 100], [0, 100], color='#999999', ls='--', lw=0.8); k = int((G2 >= 8).sum()); sh = G2[G2 >= 8].sum() / G2.sum() * 100
axs[1].scatter([k / len(G2) * 100], [sh], color='k', zorder=3, s=12); axs[1].annotate(f'{k} groups (>=8 images)\nhold {sh:.0f}% of images', (k / len(G2) * 100, sh), textcoords='offset points', xytext=(12, -22), fontsize=7)
axs[1].set_xlabel('% of groups (largest first)'); axs[1].set_ylabel('% of images'); fig.tight_layout(); fig.savefig(out / 'fig3_group_sizes.png'); fig.savefig(out / 'fig3_group_sizes.pdf'); plt.close(fig)
print('written', sorted(p.name for p in out.glob('fig*')))
