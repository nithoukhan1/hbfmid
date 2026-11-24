# Method C1: group-balanced (duplicate-aware) training — result (2026-11-24)

Runs: notebook 12, part A = YOLOv8s, part B = YOLO11s, seeds 42/43/44 x 5 v2 folds, 30 runs, no failures. Repo commit d5ecff6, which already contained the final gate (docs/METHOD_DESIGN.md section 3b), so the gate was fixed before the runs. Folders `results/gb_A`, `results/gb_B`.
Tools: `scripts/evaluation/compare_method.py --arm gb` -> `results/analysis/method_gb.{json,md}`; `scripts/evaluation/gb_diagnostics.py` -> `results/analysis/method_gb_diagnostics.{json,md}` (exploratory).

Setting (pre-declared, the only one tried): per group keep at most 12 images, repeat each kept image round(7 / kept) times clipped to [1, 4]; epochs scaled so optimisation steps match the baseline (78-82 instead of 100).
Implementation verified: metrics.json `group_balance` shows 897-957 -> 1133-1160 training entries, 168-170 groups, epochs 78-82; the Ultralytics logs report the balanced counts (train: Scanning ... 1133/1136/1146/1149/1160 images; fold-0 baseline 897).

## Gate result (pre-registered)
| detector | role | baseline (3 seeds) | C1 (3 seeds) | gain, 95% CI (t, 4 df) | folds positive | result |
|---|---|---|---|---|---|---|
| YOLOv8s | primary | 0.1513 | 0.1466 | -0.47 pt [-1.77, +0.83] | 3/5 | FAIL |
| YOLO11s | replication | 0.1539 | 0.1479 | -0.59 pt [-2.09, +0.90] | 2/5 | not same direction |

Per-fold gains (pts): YOLOv8s +0.32, +0.13, -2.14, +0.21, -0.87; YOLO11s -1.08, +0.62, -0.11, -2.45, +0.04. Per-seed 5-fold gains: YOLOv8s -0.81 / -0.30 / -0.31; YOLO11s +0.11 / -1.69 / -0.21. Pooled over both detectors (descriptive): -0.53 pt, 5/10 detector-folds positive.

Conclusion: C1 fails the gate. Both point estimates are slightly negative and the upper 95% bounds (+0.83 and +0.90 pt) exclude the +1.8 pt gate and also gains of about 1 pt or more. Stop rule applied: no tuning, no further settings; reported as a negative result.

## Exploratory diagnostics (not pre-registered; for the discussion only)
- Second held-out set: validation-fold mAP50-95 changes by +0.07 pt (YOLOv8s, 2/5 folds) and +0.37 pt (YOLO11s, 4/5): no gain there either.
- Fit vs transfer: the balanced models end with lower training losses in 5/5 folds for both detectors (box -0.06 / -0.07, cls -0.03; measured on their own, different training lists), while validation losses do not move (box -0.00 / -0.03, cls -0.00 / +0.03). Repeated copies are fitted more tightly; this does not carry over to new images.
- Size of the intervention: in fold 0, 710 of 897 training images (79%) were used once as before, 143 twice, 35 four times and 9 were dropped by the cap. Class exposure (share of training images) changed by at most +/-18% (Linear 1.18 ... Transverse 0.87); per-class AP changes do not track exposure (Spearman +0.30 over 10 classes, very noisy per-class AP).
- Operating point (Ultralytics P/R): YOLOv8s recall +1.4 pt; YOLO11s recall -3.6 pt (0/5 folds) and precision +1.2 pt. No consistent pattern across detectors.

## Interpretation
- Re-weighting the copies of about 170 source groups per training fold adds no new information: the baseline already sees every copy, and exact repeats mainly add memorisation.
- With Gate C on the v1 folds (class weights, FracAtlas pre-training, joint training: no gain), this supports the paper's thesis: on duplicate-aware folds the limit is the number of distinct source images, not the training recipe.

## Miss analysis refreshed with all baseline seeds (results/analysis/miss_analysis_v2_3seeds.*)
6 runs per fold (2 detectors x 3 seeds), 328 unique test boxes, conf >= 0.25, IoU >= 0.5 (any class): never found by any run 127 (39%; it was 48% with the 2 seed-42 runs), found by every run 93 (28%), run-dependent 108 (33%). MRI is still not the weak modality: per-run miss 45-47% vs X-ray 57-59%; never found 34% (21/62) vs 40% (106/266). Size does not explain misses (median box area 2.1% always-found vs 1.8% never-found).

## Consequence for the plan
- C1 closed as a pre-registered negative result (paper: negative-results section).
- Next per docs/METHOD_DESIGN.md: C3 (RF-DETR, DINOv2 backbone) as a benchmark row, or stop the method track. Decision pending (user).
