# C1 group-balanced arm: exploratory diagnostics (not pre-registered)

| quantity (gb - baseline, seed-averaged, paired by fold) | YOLOv8s mean (folds > 0) | YOLO11s mean (folds > 0) |
|---|---|---|
| test_map5095 | -0.471 pts (3/5) | -0.595 pts (2/5) |
| val_map5095 | +0.070 pts (2/5) | +0.370 pts (4/5) |
| test_precision | -0.042 pts (2/5) | +1.192 pts (3/5) |
| test_recall | +1.401 pts (4/5) | -3.563 pts (0/5) |
| train_box_loss_last | -0.059 (0/5) | -0.074 (0/5) |
| train_cls_loss_last | -0.031 (0/5) | -0.027 (0/5) |
| val_box_loss_last | -0.002 (2/5) | -0.030 (1/5) |
| val_cls_loss_last | -0.004 (2/5) | +0.028 (3/5) |

Per class: exposure ratio = class share of training images after / before balancing (mean over folds); AP change pooled over both detectors and the folds where the class is in the test set.

| class | exposure ratio | delta AP50-95 (pts) | detector-folds |
|---|---|---|---|
| Linear | 1.18 | -1.66 | 10 |
| Greenstick | 1.15 | +1.01 | 10 |
| Spiral | 1.10 | +0.52 | 10 |
| Comminuted | 1.06 | -0.92 | 10 |
| Transverse Displaced | 1.00 | -0.24 | 10 |
| Oblique | 0.99 | +0.60 | 10 |
| Oblique Displaced | 0.95 | +1.07 | 10 |
| Segmental | 0.94 | +0.00 | 8 |
| Healthy | 0.92 | -4.01 | 10 |
| Transverse | 0.87 | -1.78 | 10 |

Spearman rank correlation, exposure ratio vs AP change, over 10 classes: +0.30 (exploratory; per-class AP on about 57 test images per fold is very noisy).
