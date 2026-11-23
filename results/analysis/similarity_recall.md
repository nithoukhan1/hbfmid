| protocol | model | similarity bin (max NCC to train) | test images | boxes | box recall |
|---|---|---|---|---|---|
| clean_v1 | yolo11s.pt | <0.80 | 134 | 154 | 0.416 |
| clean_v1 | yolo11s.pt | 0.80-0.90 | 116 | 132 | 0.485 |
| clean_v1 | yolo11s.pt | 0.90-0.95 | 33 | 37 | 0.784 |
| clean_v1 | yolo11s.pt | >=0.95 | 108 | 134 | 0.955 |
| clean_v1 | yolov8s.pt | <0.80 | 134 | 154 | 0.416 |
| clean_v1 | yolov8s.pt | 0.80-0.90 | 116 | 132 | 0.417 |
| clean_v1 | yolov8s.pt | 0.90-0.95 | 33 | 37 | 0.784 |
| clean_v1 | yolov8s.pt | >=0.95 | 108 | 134 | 0.970 |
| clean_v2 | yolo11s.pt | <0.80 | 133 | 153 | 0.379 |
| clean_v2 | yolo11s.pt | 0.80-0.90 | 128 | 145 | 0.448 |
| clean_v2 | yolo11s.pt | 0.90-0.95 | 24 | 30 | 0.567 |
| clean_v2 | yolov8s.pt | <0.80 | 133 | 153 | 0.425 |
| clean_v2 | yolov8s.pt | 0.80-0.90 | 128 | 145 | 0.469 |
| clean_v2 | yolov8s.pt | 0.90-0.95 | 24 | 30 | 0.600 |
| official | yolo11s.pt | <0.80 | 15 | 15 | 0.667 |
| official | yolo11s.pt | 0.80-0.90 | 9 | 9 | 0.778 |
| official | yolo11s.pt | 0.90-0.95 | 6 | 6 | 1.000 |
| official | yolo11s.pt | >=0.95 | 34 | 36 | 0.972 |
| official | yolov8s.pt | <0.80 | 15 | 15 | 0.800 |
| official | yolov8s.pt | 0.80-0.90 | 9 | 9 | 0.778 |
| official | yolov8s.pt | 0.90-0.95 | 6 | 6 | 1.000 |
| official | yolov8s.pt | >=0.95 | 34 | 36 | 0.972 |
