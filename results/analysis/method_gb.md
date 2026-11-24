Gate: mean paired gain >= +1.8 pt mAP50-95 AND >= 4/5 folds positive, primary = YOLOv8s; YOLO11s = replication (gain > 0)

| model | role | seeds | base mean | method mean | gain (pts) +/- t95 | folds positive | result |
|---|---|---|---|---|---|---|---|
| yolov8s.pt | primary | [42, 43, 44] | 0.1513 | 0.1466 | -0.47 +/- 1.30 | 3/5 | FAIL |
|  |  | per-seed gains (pts) | 42: -0.81, 43: -0.30, 44: -0.31 | | | | |
| yolo11s.pt | replication | [42, 43, 44] | 0.1539 | 0.1479 | -0.59 +/- 1.50 | 2/5 | not same direction |
|  |  | per-seed gains (pts) | 42: +0.11, 43: -1.69, 44: -0.21 | | | | |

Pooled over both detectors (descriptive, no decision weight): -0.53 pt, 5/10 detector-folds positive
