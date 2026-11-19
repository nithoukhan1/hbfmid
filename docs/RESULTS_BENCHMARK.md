# Detector benchmark: official split vs HBFMID-Clean (2026-11-19)

Notebook 08 (parts A and B), commit e84c924. Fixed protocol: 100 epochs, no early stopping, last.pt, imgsz 640, batch 16, Ultralytics 8.4.215, Tesla T4.
Raw runs: `results/bench_A`, `results/bench_B`, `results/baseline_fixed`, `results/controls`. Merged table: `results/benchmark_all.{md,json}`.
Official = the 64-image test set of the shipped split (seeds in brackets). Clean = HBFMID-Clean, 5 folds, pooled test 392 representatives; the "+/-" after a clean score is the sd between folds (not a confidence interval).

## 1. Table

| detector | official mAP50-95 | official mAP50 | clean mAP50-95 (sd over folds) | clean macro (8 classes) | clean mAP50 | official / clean | seeds |
|---|---|---|---|---|---|---|---|
| YOLOv8s | 0.4942 [42,43,44] | 0.9052 | 0.2170 (0.0458) | 0.2663 | 0.4160 | 2.28 | official 3, clean 3 |
| YOLO11s | 0.4846 [42,43,44] | 0.9264 | 0.2225 (0.0358) | 0.2731 | 0.4427 | 2.18 | official 3, clean 3 |
| YOLOv8m | 0.4724 [42] | 0.9163 | 0.2271 (0.0383) | 0.2790 | 0.4380 | 2.08 | official 1, clean 1 |

Seed spread on the official test set (mAP50-95 per seed): YOLOv8s 0.4987 / 0.4879 / 0.4961; YOLO11s 0.4711 / 0.5049 / 0.4778 (range 3.4 pts); YOLOv8m 0.4724 (one seed).
Clean, 5-fold mean per seed: YOLOv8s 0.2151 / 0.2233 / 0.2127; YOLO11s 0.2260 / 0.2236 / 0.2177; YOLOv8m 0.2271 (seed 42).

## 2. Paired comparisons on HBFMID-Clean (difference in points, 95% t-interval over 5 fold differences, rough, 4 df)

| comparison | mAP50-95 | folds + | macro | mAP50 | recall (conf 0.001 eval) |
|---|---|---|---|---|---|
| YOLO11s - YOLOv8s (3 seeds) | +0.54 +/- 1.83 | 2/5 (10/15 pairs) | +0.68 +/- 2.29 | +2.67 +/- 4.56 (3/5) | +2.04 +/- 3.42 (4/5) |
| YOLOv8m - YOLOv8s (seed 42) | +1.20 +/- 2.21 | 3/5 | +1.50 +/- 2.76 | +1.89 +/- 3.86 (4/5) | -0.05 +/- 4.79 |
| YOLOv8m - YOLO11s (seed 42) | +0.11 +/- 3.54 | 2/5 | +0.15 +/- 4.42 | -0.69 +/- 6.49 | -4.66 +/- 5.12 (0/5) |

All mAP50-95 intervals include zero and are below the 0.9-pt detectable effect (3 seeds) or the 1.6-pt one (1 seed). The three detectors cannot be separated on HBFMID-Clean.
(The "recall" column is the Ultralytics box recall in `test.recall`, not the class-agnostic localisation recall of the arms analysis.)

## 3. What the table says

1. **Leaky scores are about 2.1-2.3 times the clean ones for all three detectors** (0.47-0.49 vs 0.22-0.23 mAP50-95; mAP50 0.91-0.93 vs 0.42-0.44). The gap is not detector specific.
2. **Rankings are unstable.** On the official split the nominal order is YOLOv8s > YOLO11s > YOLOv8m; on HBFMID-Clean it is the reverse (YOLOv8m >= YOLO11s > YOLOv8s). Neither order is significant. On the official test set the seed-to-seed range of a single detector (3.4 pts for YOLO11s) is larger than the gaps between detectors (0.96 and 2.2 pts), so a 1-2 pt "improvement" measured there is not distinguishable from noise.
3. **Larger or newer does not help much here.** YOLOv8m (about 2.5x the compute of YOLOv8s: 45 min vs 27 min per fold run) gains +1.2 pts, not significant; YOLO11s gains +0.5 pts. This is consistent with a data-limited, localisation-limited problem rather than a capacity-limited one.
4. **Error mix at conf 0.25, pooled over runs** (share of ground-truth boxes):

| split | detector | correct | wrong class | missed |
|---|---|---|---|---|
| clean | YOLOv8s | 56.1% | 5.0% | 38.9% |
| clean | YOLO11s | 56.3% | 5.4% | 38.3% |
| clean | YOLOv8m | 56.9% | 4.6% | 38.5% |
| official | YOLOv8s | 88.4% | 1.5% | 10.1% |
| official | YOLO11s | 88.9% | 0.0% | 11.1% |
| official | YOLOv8m | 86.4% | 0.0% | 13.6% |

Misses dominate the errors of every detector on the clean protocol (about 38-39% of boxes, wrong class only 5%), while on the leaky official test set about 89% of boxes are found. This repeats the earlier error decomposition (31% never localised at conf 0.05) for two more detectors, so the finding is not specific to YOLOv8s.

## 4. Limits
- YOLOv8m has one seed per protocol, so its comparisons use 5 paired fold differences only.
- YOLO family only (all Ultralytics, one training recipe). A transformer row (RT-DETR or DEIMv2) is still open.
- Official test set: 64 images, 0 Linear, 0 Segmental; its classes are not the same population as the clean test.
- Clean test = one representative per near-duplicate group; groups are not patients.

## 5. Consequence
The benchmark claim for the paper: under leakage-free evaluation the three YOLO detectors are statistically indistinguishable, and the official split overstates every score by roughly a factor of two while also ordering the detectors differently. Next: RT-DETR/DEIMv2 row, validate the audit (manual pair review, threshold sensitivity), per-class and modality tables.
