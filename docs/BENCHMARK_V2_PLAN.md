# Benchmark completion on the v2 folds (option C): protocol fixed before the runs (2026-11-24)

Why: (1) answer the reviewer question "is this only small YOLOs?" with a complete 5-detector table under the duplicate-aware protocol; (2) test the one lever left after C1 failed: a stronger pretrained representation (foundation backbone). This is the C3 step of docs/METHOD_DESIGN.md, run as a benchmark row, not as "our method".

## Detectors and runs
| detector | family / backbone, pretraining | params | v2 folds, 3 seeds x 5 | official split, 3 seeds | notebook |
|---|---|---|---|---|---|
| YOLOv8s | CNN, COCO | 11.2 M | done (10, 11) | done | - |
| YOLO11s | CNN, COCO | 9.4 M | done (10, 11) | done | - |
| YOLOv8m | CNN, COCO | 25.9 M | NEW, 15 runs | seed 42 done; NEW seeds 43, 44 | 13 |
| RT-DETR-L | hybrid CNN encoder + transformer decoder, COCO | 32 M | NEW, 15 runs (parts A, B) | done | 14 |
| RF-DETR-Medium | DINOv2 ViT (windowed) backbone, Objects365 + COCO | 33.7 M | NEW, 15 runs | NEW, 3 runs | 15 |

## Common protocol
- Same list files as all earlier rows (`data_splits_v2/clean5` fold k; `data_splits/official`), seeds 42/43/44, 100 epochs, no early stopping, last-epoch EMA weights, input 640 x 640 (native image size), effective batch 16, each detector's default augmentation and optimiser, no tuning, Kaggle T4.
- Ultralytics rows: ultralytics 8.4.215, identical settings to the earlier runs (RT-DETR-L as in notebook 09: batch 16, AMP on).
- RF-DETR row: rfdetr 1.11.2 (pinned), RFDETRMedium, published pretrained checkpoint, resolution 640 (package default 576; 640 keeps the input identical across detectors), batch 4 x gradient accumulation 4, lr 1e-4, encoder lr 1.5e-4, EMA (decay 0.993), fp16 (auto on T4), validation every 10 epochs (never used for checkpoint selection), class names fixed to the 10 HBFMID classes; inference with last_ema.pth, threshold 0.001, top 300 (RF-DETR num_select).
- Decided now, on memory and time only: if batch 4 runs out of GPU memory at 640, use batch 2 x accumulation 8 (same effective batch). Runs that would not fit a 12 h Kaggle session move to a later part; the number of epochs is never changed.

## Scoring
- Primary: box mAP50-95 over the classes present in the test fold (Ultralytics definition). Secondary: macro AP50-95 over the 8 eligible classes, mAP50.
- YOLO and RT-DETR: Ultralytics 8.4.215 validation (conf 0.001, NMS IoU 0.7, max 300 detections). RF-DETR: its own decoding scored with `scripts/evaluation/offline_eval.py`, an exact replica of the Ultralytics metric:
  - unit parity against the real Ultralytics functions: max difference 0 (mAP50-95, mAP50, per-class AP) over 20 random scenarios;
  - end-to-end parity on a real Ultralytics validation export (YOLOv8n, synthetic data, 3600 predictions): 0.709163 vs 0.709163;
  - every new Ultralytics run (notebooks 13, 14) re-checks parity on its own exported predictions (`metrics.json: test_offline_parity`, must be below 1e-3).
  Test: `python scripts/evaluation/test_offline_eval_parity.py --e2e`.
- Do not compare mAP from the old `preds_test.json` files (single-label predict, conf >= 0.05): re-scoring them gives on average -1.8 pt against Ultralytics (range -6.0 to +0.9).

## Comparisons (decided now)
- Per detector: mean over 3 seeds x 5 folds, sd over folds, seed sd of the 5-fold mean; official 3-seed mean; inflation ratio official / v2.
- Each detector against YOLOv8s (reference): difference paired by fold (seed-averaged), mean with t95 interval (4 df), folds positive. A detector is called better or worse only if |difference| >= 1.8 pt (2-sigma noise floor, docs/METHOD_DESIGN.md 3b) AND the sign holds in >= 4/5 folds; otherwise "not distinguishable".
- RF-DETR hypothesis: gain >= +1.8 pt over YOLOv8s on v2 under that rule. Either outcome is reported.
- Failure rule: a run with test mAP50-95 < 0.01 or NaN validation is a failed run. Primary means include it as measured; a secondary mean without failed runs is labelled post-hoc. No re-runs, no replacement seeds.
- Exploratory (labelled as such): rank agreement official vs v2 over the 5 detectors (Kendall tau); behavioural leak test (similarity-recall) and miss analysis per detector.

## Budget and order (T4 x2 sessions, one per account at a time)
- Notebook 13 (YOLOv8m): 17 runs, about 6.5-7 h, one session.
- Notebook 14 (RT-DETR-L): part A 8 runs, part B 7 runs, about 6.5 h each.
- Notebook 15 (RF-DETR): SMOKE (2 epochs, about 15-20 min) first; then parts A (3 official + 1 v2), B (7 v2), C (7 v2); part sizes confirmed from the smoke timing; a time guard skips any run that would end after 11.5 h and prints it for part D.
- Order: push; account 1 -> notebook 13; account 2 -> notebook 15 SMOKE, then notebook 14 part A; then notebook 14 part B and notebook 15 parts as accounts free up.
