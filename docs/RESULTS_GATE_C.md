# Gate C result: FracAtlas transfer and class weighting (2026-11-18)

Protocol as in docs/ARMS_DESIGN.md: YOLOv8s, 100 epochs, last.pt, 5 HBFMID-Clean folds, differences paired by fold (and by seed where several seeds exist) against the plain baseline (arm 1).
Gate C = improvement >= the detectable effect (0.9 pt with 3 seeds), positive on >= 4/5 folds, better localisation recall.
Raw comparisons: `results/arms/*_vs_baseline.{md,json}`.

| Arm | seeds | mAP50-95 diff | folds + | macro diff | mAP50 diff | localisation recall diff |
|---|---|---|---|---|---|---|
| 2 class weights, box counts | 1 | +0.0025 +/- 0.022 | 3/5 | +0.003 | -0.015 | -0.004 |
| 3 class weights, group counts | 1 | -0.0096 +/- 0.014 | 1/5 | -0.011 | -0.023 | -0.028 |
| 6 FracAtlas pre-train, generic | 1 | -0.0211 +/- 0.035 | 1/5 | -0.026 | -0.030 | -0.003 |
| 6 FracAtlas pre-train, zoom | 1 | +0.0027 +/- 0.060 | 2/5 | +0.002 | +0.001 | +0.000 |
| 7' joint (extra class), generic | **3** | **+0.0034 +/- 0.014** | **3/5** | +0.004 | +0.010 (4/5) | +0.023 +/- 0.027 (4/5) |
| 7' joint, zoom | 1 | +0.0021 +/- 0.025 | 2/5 | +0.003 | +0.002 | +0.033 +/- 0.025 (5/5) |
| control: baseline 177 epochs (same gradient steps as 7') | 1 | -0.0019 +/- 0.021 | 2/5 | -0.003 | -0.006 | -0.013 |

(+/- = 95% t interval over the 5 fold differences, 4 df, rough.)

## What happened to the pilot signal
Arm 7' generic by seed (5-fold mean mAP50-95, arm vs baseline of the same seed): seed 42 0.2308 vs 0.2151 (+1.57 pts); seed 43 0.2128 vs 0.2233 (-1.05); seed 44 0.2177 vs 0.2127 (+0.50). Across all 15 (fold, seed) pairs: mean +0.34 pt (pair sd 2.36 pts), positive in 7/15. The pilot's +1.6 pts was mostly seed noise, which is exactly what the 3-seed requirement was for.
The compute control agrees: training the plain baseline for 1.77x the epochs gives -0.2 pt, so extra gradient steps alone do not help.

## Verdict
- **Gate C fails** on the primary criterion (mAP50-95): +0.34 pt with 3 seeds is below the 0.9-pt detectable effect and positive in 3/5 folds, not 4/5. Class re-weighting (arms 2, 3) and FracAtlas pre-training (arm 6, both variants) give nothing or hurt. Scale-matched crops (zoom) do not change this.
- **Exploratory observation (not a claim):** the two joint-training variants raise class-agnostic localisation recall at conf 0.25 in 13 of 15 fold comparisons across three experiments (the generic seed-42 pilot is also part of the 3-seed run, so they are not independent) (generic seed 42 +5.2 pts, generic 3 seeds +2.3 pts, zoom +3.3 pts), without a matching gain in mAP50-95 (mAP50 +1.0 pt for the 3-seed run). Intervals include zero for the 3-seed run. Worth one sentence in a paper, as a hypothesis.
- Limits: one detector (YOLOv8s), one external dataset (single class), one pre-training seed, zoom run with one seed.

## Consequence for the paper (plan v3.1, branch "Gate C fails")
Anchor = leakage-free benchmark: audit (388 groups in 1,539 images), dual reporting (official vs HBFMID-Clean), error decomposition (recall-limited, not class-confusion-limited), modality analysis, noise-floor protocol (fixed 100 epochs + last.pt), several detectors, and an honest negative-results section (class weights, FracAtlas transfer: nothing beats seed noise). Expected venue level Q3-Q4.
Next: notebook 08 (more detectors on both protocols).
