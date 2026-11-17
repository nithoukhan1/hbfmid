# Controls summary

## 1. Leakage attribution on the SAME 64 official test images

| run | train imgs | test mAP50 | test mAP50-95 | macro(>=10 groups) |
|---|---|---|---|---|
| E1 official (leaky train+val) | 1347 | 0.9398 | 0.4898 | 0.4898 |
| E1b official de-duplicated, seed 42 (fixed_last) | 806 | 0.3500 | 0.1660 | 0.1660 |

- drop E1 -> E1b (same test images, leakage removed from train/val) = 0.3238 mAP50-95

## 2. Seed noise (clean folds, several seeds, same split)

| fold | seeds | last.pt test mAP50-95 (sd) | best.pt test mAP50-95 (sd) | best_epochs |
|---|---|---|---|---|
| 0 | [42, 43, 44] | 0.1993 (0.0116) | 0.1935 (0.0205) | [26, 91, 85] |
| 1 | [42, 43, 44] | 0.2339 (0.0137) | 0.2283 (0.0272) | [86, 50, 51] |

- mean seed sd: last.pt 0.0126 vs best.pt 0.0238 (early-stopping baseline run earlier: 0.0211)
- paired difference of two arms over 5 folds x 1 seed(s): sd about 0.0080; a ~2-sigma effect size is about 0.0160 mAP50-95
- paired difference of two arms over 5 folds x 3 seed(s): sd about 0.0046; a ~2-sigma effect size is about 0.0092 mAP50-95
