# Results: controls (2026-11-17, notebook 02) - leakage attribution and fixed-protocol noise floor

Environment as before (ultralytics 8.4.215, torch 2.11.0, 2x T4, repo commit 74e3f8f). 7 runs, all OK, 23-29 min each (about 2 h wall).

## 1. Leakage attribution (E1 vs E1b, the SAME 64 official test images)
E1b = official split with train/val de-duplicated (train 806 / val 109 images; every train image whose group touches the test or kept-val groups removed).
| run | train imgs | test mAP50 | test mAP50-95 |
|---|---|---|---|
| E1 official (leaky) | 1347 | 0.940 | 0.490 |
| E1b de-duplicated, fixed protocol, seed 42 | 806 | 0.350 | 0.166 |
Removing the leakage from training drops the same test images from 0.490 to 0.166 mAP50-95 (single seed; seed sd about 0.013). The leakage-free official-test number is in the range of the clean folds (0.20-0.23).

Box recall at conf >= 0.25 (leak_probe_recall.py; ground-truth boxes with any prediction IoU >= 0.5 of the same class):
| official test subset | E1 (leaky) | E1b (de-duplicated) |
|---|---|---|
| 48 images WITH a verified train copy (49 boxes) | 0.98 | 0.37 |
| 16 images WITHOUT a train copy (17 boxes) | 0.71 | 0.59 |
Reading: the damage is concentrated exactly where the audit found copies (0.98 -> 0.37), while images without a copy change little (0.71 -> 0.59; 17 boxes, so +/- 0.1 and partly the 40% smaller training set). A training-size confound cannot explain a 0.61 drop on the copy images. The earlier AP-based leak probe (0.512 vs 0.466) was uninformative because AP on 16 images is too noisy; use box recall for the probe.
Caveats: one seed for E1b; 64 test images, 66 boxes; E1b uses 806 training images against 893-970 in the clean folds.

## 2. Seed noise under the fixed-epoch protocol (folds 0 and 1, seeds 42/43/44, 100 epochs, no early stopping)
| fold | last.pt mAP50-95 (sd) | best.pt mAP50-95 (sd) | best.pt epochs |
|---|---|---|---|
| 0 | 0.1993 (0.0116) | 0.1935 (0.0205) | 26, 91, 85 |
| 1 | 0.2339 (0.0137) | 0.2283 (0.0272) | 86, 50, 51 |
Mean seed sd: last.pt 0.0126 vs best.pt 0.0238 (early-stopping runs earlier: 0.0211). The last-epoch (EMA) checkpoint halves the noise because it removes validation-based selection on a 78-image validation set. Fold 0 means: 0.199 (fixed) vs 0.175 (early-stopped run), i.e. early stopping with patience 30 cost about 2 points. Seed 42 reproduced the earlier run exactly (best.pt epoch 26, test 0.1711), so training is deterministic.
Detectable effect (2 sigma, paired over the same folds): about 1.6 pts for 5 folds x 1 seed, about 0.9 pt for 5 folds x 3 seeds. These come from two folds x three seeds (4 degrees of freedom): treat as +/- a factor of about 2; re-estimate when arm 1 has all 5 folds x 3 seeds.

## Decisions
- Primary protocol for all arms: fixed 100 epochs, last.pt, 5 folds; 3 seeds for any claim.
- E1c (random subsample of the leaky train) is not needed: the subset recall analysis already isolates leakage.
- Next: finish arm 1 (folds 2-4 x 3 seeds, notebook 03), then FracAtlas arms.
