| model | seeds | runs | mAP50-95 mean (sd over folds) | seed sd of 5-fold mean | macro | mAP50 |
|---|---|---|---|---|---|---|
| yolo11s.pt | [42, 43, 44] | 15 | 0.1539 (0.0298) | 0.0117 | 0.1875 | 0.3093 |
| yolov8s.pt | [42, 43, 44] | 15 | 0.1513 (0.0394) | 0.0094 | 0.1804 | 0.3004 |

Gap YOLO11s - YOLOv8s (seed-averaged, paired by fold): +0.0026 +/- 0.0236; folds positive 3/5
