# Noise floor on v2 clean folds (plain baselines, 3 seeds x 5 folds)

| model | fold | seed 42 | seed 43 | seed 44 | seed sd |
|---|---|---|---|---|---|
| yolo11s.pt | 0 | 0.1184 | 0.1337 | 0.1303 | 0.0080 |
| yolo11s.pt | 1 | 0.1661 | 0.2340 | 0.1806 | 0.0357 |
| yolo11s.pt | 2 | 0.1853 | 0.1702 | 0.1771 | 0.0076 |
| yolo11s.pt | 3 | 0.1064 | 0.1531 | 0.1350 | 0.0236 |
| yolo11s.pt | 4 | 0.1374 | 0.1396 | 0.1404 | 0.0016 |
| yolov8s.pt | 0 | 0.1260 | 0.1139 | 0.1212 | 0.0061 |
| yolov8s.pt | 1 | 0.1756 | 0.1904 | 0.1756 | 0.0085 |
| yolov8s.pt | 2 | 0.1858 | 0.1774 | 0.2357 | 0.0315 |
| yolov8s.pt | 3 | 0.1402 | 0.0930 | 0.0846 | 0.0299 |
| yolov8s.pt | 4 | 0.1693 | 0.1299 | 0.1500 | 0.0197 |

| model | sigma_run (pooled over folds, 10 df) | seed means (42/43/44) | sd of 5-fold mean | sigma implied (x sqrt 5) | A/A 5-fold-mean diffs | A/A pair sd |
|---|---|---|---|---|---|---|
| yolo11s.pt | 0.0198 | 0.1427 / 0.1661 / 0.1527 | 0.0117 | 0.0262 | 42-43: -0.0234, 42-44: -0.0100, 43-44: +0.0134 | 0.0281 |
| yolov8s.pt | 0.0218 | 0.1594 / 0.1409 / 0.1534 | 0.0094 | 0.0210 | 42-43: +0.0184, 42-44: +0.0059, 43-44: -0.0125 | 0.0317 |

Pooled sigma_run = 0.0208 (20 df); pooled sd of the 5-fold mean = 0.0106.
Detectable effect (2 sigma, paired): from sigma_run 2.64 pt (1 seed) / 1.52 pt (3 seeds); from the 5-fold-mean sd 3.01 pt (1 seed) / 1.74 pt (3 seeds).
Recomputed gate (max of 0.9 pt and both 3-seed estimates, rounded up to 0.1 pt): 1.8 pt.
