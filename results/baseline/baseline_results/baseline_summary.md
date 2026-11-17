# Baseline summary (E1 official vs E2 clean)

## E1 official split

| tag | val mAP50-95 | test mAP50-95 | test macro(>=10 groups) | with-copy mAP50-95 | no-copy mAP50-95 |
|---|---|---|---|---|---|
| official_yolov8s_s42 | 0.5001 | 0.4898 | 0.4898 | 0.5118 | 0.4660 |

## E2 clean folds (test, mean over seeds)

| fold | seeds | test mAP50-95 | macro(>=10 groups) |
|---|---|---|---|
| 0 | 3 | 0.1750 | 0.1969 |
| 1 | 1 | 0.2202 | 0.2753 |
| 2 | 1 | 0.2110 | 0.2638 |
| 3 | 1 | 0.1766 | 0.2208 |
| 4 | 1 | 0.2590 | 0.3238 |
| **mean (n=5)** | | 0.2084 +/- 0.0348 | 0.2561 +/- 0.0494 |

## Gate A (does the official split inflate results?)
- E1 test mAP50-95 = 0.4898 ; E2 clean mean = 0.2084 ; delta = 0.2814
- macro(>=10 groups): E1 0.4898 vs E2 0.2561 ; delta = 0.2337
- Gate A (E1 > E2): **PASS**
- Leak probe (official test, 48 with train copy vs 16 without): 0.5118 vs 0.4660 ; gap = 0.0457 (16 images: interpret with caution)

## Gate A2 (are errors sibling-class confusions?)
- Pooled E2 (primary seed per fold, conf>=0.25, IoU>=0.5): {'correct': 248, 'wrong_class': 23, 'missed_fn': 186, 'false_pos_fp': 70, 'sibling_confusions': 16, 'same_base_confusions': 8, 'sibling_share_of_wrong': 0.6956521739130435, 'sibling_share_of_all_errors': 0.05734767025089606}
- sibling confusions / all errors = 0.0573 -> Gate A2 (>= 0.20): **FAIL**
- sibling confusions / wrong-class matches = 0.6957

## Seed noise floor (same split, different seeds)
- ('clean', 0): seeds=[42, 43, 44] sd(mAP50-95)=0.0211 sd(macro)=0.0238
- mean sd: mAP50-95 0.0211, macro 0.0238 (Gate B threshold of +0.010 is only meaningful if this is well below 0.010)

Environment: {'ultralytics': '8.4.215', 'torch': '2.11.0+cu130', 'python': '3.13.16', 'gpu': 'Tesla T4', 'model': 'yolov8s.pt', 'epochs_cfg': 100, 'imgsz': 640, 'batch': 16}
