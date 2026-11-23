| protocol | model | folds | mAP50-95 mean (sd over folds) | macro (8 classes) | mAP50 |
|---|---|---|---|---|---|
| v1 | yolo11s.pt | 5/5 | 0.2260 (0.0345) | 0.2775 | 0.4449 |
| v1 | yolov8s.pt | 5/5 | 0.2151 (0.0406) | 0.2640 | 0.4191 |
| v2 | yolo11s.pt | 5/5 | 0.1427 (0.0328) | 0.1743 | 0.2877 |
| v2 | yolov8s.pt | 5/5 | 0.1594 (0.0252) | 0.1948 | 0.3230 |

Shift v1 -> v2, yolo11s.pt: mAP50-95 -0.0833 (unpaired; v2 test sets differ; single seed)
Shift v1 -> v2, yolov8s.pt: mAP50-95 -0.0558 (unpaired; v2 test sets differ; single seed)
Gap YOLO11s - YOLOv8s, v1: +0.0109 +/- 0.0268 (paired by fold, t=2.776, 4 df); folds positive 4/5
Gap YOLO11s - YOLOv8s, v2: -0.0166 +/- 0.0188 (paired by fold, t=2.776, 4 df); folds positive 0/5
