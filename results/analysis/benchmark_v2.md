# HBFMID-Clean v2 benchmark (2026-11-26, scripts/evaluation/benchmark_v2.py)

Rules fixed before the runs (docs/BENCHMARK_V2_PLAN.md). mAP50-95 on the test fold, in points (x100). Gate 1.8 pt: a verdict needs |diff| >= gate AND the same sign in >= 4/5 folds. "Failed" = test mAP50-95 < 1 pt or NaN validation; primary numbers include failed runs as measured.

## Main table

| detector | v2 runs | failed | v2 mean | sd over folds | seed sd | v2 macro AP (8 cl.) | v2 mAP50 | official (3 seeds) | official / v2 |
|---|---|---|---|---|---|---|---|---|---|
| YOLOv8s | 15/15 | 0 | 15.13 | 3.94 | 0.94 | 18.04 | 30.04 | 49.42 (sd 0.56) | 3.27x |
| YOLO11s | 15/15 | 0 | 15.39 | 2.98 | 1.17 | 18.75 | 30.93 | 48.46 (sd 1.79) | 3.15x |
| YOLOv8m | 15/15 | 0 | 14.45 | 3.34 | 1.11 | 17.60 | 30.34 | 47.78 (sd 0.76) | 3.31x |
| RT-DETR-L | 15/15 | 1 | 14.75 | 2.99 | 2.21 | 18.06 | 29.72 | 49.13 (sd 1.26) | 3.33x |
| RF-DETR-M | 1/15 INCOMPLETE | 0 | pending | | | | | 48.71 (sd 0.54) | |

Post-hoc (failed runs removed): RT-DETR-L v2 mean 16.13.

## Against YOLOv8s (paired by fold, seed-averaged; t95 with 4 df)

| detector | view | diff (pt) | 95% CI (pt) | folds +/- | per fold (pt) | verdict |
|---|---|---|---|---|---|---|
| YOLO11s | as measured primary | +0.26 | [-2.10, +2.62] | 3/2 | +0.7, +1.3, -2.2, +2.6, -1.1 | not distinguishable |
| YOLOv8m | as measured primary | -0.68 | [-2.44, +1.09] | 2/3 | -0.3, +0.6, -2.9, +0.3, -1.1 | not distinguishable |
| RT-DETR-L | as measured primary | -0.38 | [-3.45, +2.69] | 3/2 | +1.8, -4.2, -1.3, +0.2, +1.6 | not distinguishable |
| RT-DETR-L | failed removed posthoc | +1.01 | [-0.93, +2.95] | 4/1 | +1.8, +2.7, -1.3, +0.2, +1.6 | not distinguishable |
| RF-DETR-M | - | pending (1/15 runs) | | | | |

## Per fold (seed-averaged mAP50-95, pt)

| detector | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
|---|---|---|---|---|---|
| YOLOv8s | 12.04 | 18.06 | 19.96 | 10.59 | 14.98 |
| YOLO11s | 12.75 | 19.36 | 17.76 | 13.15 | 13.92 |
| YOLOv8m | 11.73 | 18.67 | 17.04 | 10.93 | 13.88 |
| RT-DETR-L | 13.83 | 13.85 | 18.67 | 10.82 | 16.57 |
| RF-DETR-M (runs so far) | s42 17.80 | - | - | - | - |

## Failed runs and training health (Ultralytics in-training fp16 validation log)

- YOLOv8s: failed 0/15; runs with any NaN validation epoch 0/15, NaN validation to the end 0/15, NaN train-loss epoch 0/15
- YOLO11s: failed 0/15; runs with any NaN validation epoch 0/15, NaN validation to the end 0/15, NaN train-loss epoch 0/15
- YOLOv8m: failed 0/15; runs with any NaN validation epoch 1/15, NaN validation to the end 0/15, NaN train-loss epoch 0/15
- RT-DETR-L: failed 1/15; runs with any NaN validation epoch 9/15, NaN validation to the end 1/15, NaN train-loss epoch 1/15
  - clean_f1_rtdetr-l_s44_fx: test 0.00, val 0.0; NaN validation from epoch 7 to the end, NaN train-loss epochs 1, final train loss 0.5309 (finite); best.pt (epoch 5) test 9.06: diagnostic only, not part of the protocol
- RF-DETR-M: failed 0/1; no Ultralytics log (RF-DETR)

## Exploratory (labelled as such)

- Official leak probe YOLOv8s: with-copy minus without-copy test images +4.45 pt (per seed [4.27, 1.39, 7.7])
- Official leak probe YOLO11s: with-copy minus without-copy test images +4.88 pt (per seed [0.81, 8.41, 5.41])
- Official leak probe YOLOv8m: with-copy minus without-copy test images +5.51 pt (per seed [5.34, 6.55, 4.63])
- Official leak probe RT-DETR-L: with-copy minus without-copy test images +4.42 pt (per seed [1.3, 4.57, 7.39])
- Official leak probe RF-DETR-M: with-copy minus without-copy test images +0.53 pt (per seed [1.93, -0.89, 0.54])
- Rank agreement official vs v2: pending (all rows must be complete)

## Checks

- duplicates: none
- fold consistency (image counts and test GT boxes): OK
- evaluator parity YOLOv8s: 0 runs, max |diff| None; 18 runs older than the parity check
- evaluator parity YOLO11s: 0 runs, max |diff| None; 18 runs older than the parity check
- evaluator parity YOLOv8m: 17 runs, max |diff| 0.0009750000000000036; 1 runs older than the parity check
- evaluator parity RT-DETR-L: 14 runs, max |diff| 1.6493226914521264e-05; no export (failed run): ['clean_f1_rtdetr-l_s44_fx']; 3 runs older than the parity check
