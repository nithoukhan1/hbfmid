# Clean benchmark on grouping v2 folds (2026-11-23)

Runs: notebook 10, YOLO11s (part A) and YOLOv8s (part B), 5 folds, seed 42, fixed protocol (100 epochs, last.pt, ultralytics 8.4.215, T4 x2), no failed runs. Folders `results/v2_A`, `results/v2_B`. Tools: `scripts/evaluation/compare_v1_v2.py` -> `results/analysis/clean_v1_vs_v2.{json,md}`; `scripts/evaluation/v1_leak_stratified.py` -> `results/analysis/v1_leak_stratified.{json,md}`.

## Result (mAP50-95, mean over 5 folds, sd over folds; seed 42 only)
| model | official (3 seeds) | clean v1 | clean v2 | v1 -> v2 | official / v2 |
|---|---|---|---|---|---|
| YOLO11s | 0.485 | 0.2260 (0.0345) | 0.1427 (0.0328) | -8.3 pts | 3.4x |
| YOLOv8s | 0.494 | 0.2151 (0.0406) | 0.1594 (0.0252) | -5.6 pts | 3.1x |
Macro (8 classes) v2: 0.1743 / 0.1948; mAP50 v2: 0.2877 / 0.3230.
Detector gap YOLO11s - YOLOv8s, paired by fold: v1 +1.09 +/- 2.68 pts (4/5 folds positive); v2 -1.66 +/- 1.88 pts (0/5 folds positive). Both intervals include 0 at one seed (single-seed noise floor about 1.6 pts); the sign flipped between protocols again.

## Why the drop: leakage in the v1 folds (model-based check, saved predictions, no GPU)
For each v1 clean test image: 'leaky' = another member of its grouping-v2 group is in that fold's v1 train list. Box recall at conf 0.25, IoU 0.5:
| model | test images leaky / band(0.90-0.95) / total | recall not leaky | recall band | recall leaky |
|---|---|---|---|---|
| YOLO11s | 121 / 35 / 391 | 0.450 | 0.625 | 0.939 |
| YOLOv8s | 121 / 35 / 391 | 0.413 | 0.600 | 0.966 |
- 31% of the v1 clean test images (121/391) have a v2 duplicate in train, and the detectors find 94-97% of their boxes against 41-45% for the rest. That also validates the NCC>=0.95 links by behaviour: unrelated pairs would not be found at 95%.
- The band 0.90-0.95 (9% of test images) sits in between (0.60-0.63), consistent with a minority of real duplicates, as the visual check (4/18) suggested.
- Train sizes are similar across protocols (v1 893-970 vs v2 888-957 images per fold), so a smaller training set does not explain the drop. The test sets differ (v1 about 78, v2 about 57 images per fold), so the v1-v2 shift is unpaired; the recall split above is the direct evidence.

## Corrections to earlier statements (mine)
- docs/RESULTS_GROUPING_V2.md said the v1 clean folds had 'limited' residual leakage (29 groups) and a 'small' upward bias. That was wrong in size: those 29 merged groups cover 31% of the test images, and the drop is 5.6-8.3 pts (26-37% relative).
- Every result measured on v1 folds (Gate C arms, benchmark, RT-DETR, miss analysis) carries this contamination. Differences between arms may be diluted rather than reversed; not yet re-tested on v2.

## What it means
- Inflation chain for one detector (YOLOv8s): official 0.494 -> clean v1 0.215 -> clean v2 0.159. 'Clean' depends on how hard duplicates are searched for; a residual tail remains (band images, v3).
- Strengthens the audit paper: a leakage-detection sensitivity curve (official, SIFT-only, SIFT + thumbnail) with model-based validation is a concrete, checkable contribution.
- Open: seeds 43/44 and RT-DETR on v2 (the detector ranking is still within noise); re-evaluate the best Gate C arms on v2 if the method track continues.
