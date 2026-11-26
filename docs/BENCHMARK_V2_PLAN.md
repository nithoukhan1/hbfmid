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

## Deviation and status log
- 2026-11-24, notebook 13 (YOLOv8m): 17/17 runs OK. v2 0.1445 (seed means 0.1431 / 0.1342 / 0.1562, seed sd 0.011); vs YOLOv8s -0.68 pt [-2.44, +1.09], 2/5 folds: not distinguishable. Official 3 seeds 0.4778 (sd 0.0076); official / v2 = 3.3x.
- 2026-11-24, notebook 14 part A (RT-DETR-L): 8/8 runs OK, none collapsed (0.096-0.220). Part B pending; no comparison before all 15 runs.
- Evaluator parity on the 25 new Ultralytics runs: 14 exactly 0, median 0, max 9.75e-4 (YOLOv8m); RT-DETR-L <= 1.7e-5. The YOLO residuals come from the Ultralytics export, not from the evaluator: Ultralytics validates 640x640 images in 672x672 padded rect batches and matches unclipped boxes, but clips the exported boxes to the image. Traced example: one Comminuted prediction at the image border, clipped IoU 0.753 vs threshold 0.75 (runs without such border boxes match exactly). RT-DETR boxes lie inside the image, hence its tighter parity. RF-DETR is scored on its raw decoded boxes by the same evaluator, so this does not affect it.
- 2026-11-24, notebook 15 part A (RF-DETR): all 4 runs trained the full 100 epochs (RF-DETR's own validation printout at epoch 100: official 0.476-0.499, clean fold 0 0.283) and then stopped in the prediction step. Cause (my code): RF-DETR's head has one extra no-object output (class id 10, named "__background__" by predict()); a guard against 1-based ids treated id 10 as a sign of 1-based numbering and shifted all ids by one (class 0 -> -1), which the safety check rejected. Fix: no-object entries are dropped (RF-DETR's own COCO evaluator also ignores labels without a category), ids are never shifted. Training, data, epochs, weights, threshold, top-300 and evaluator are unchanged: no protocol change.
- Recovery without retraining: checkpoints are deleted only after a successful prediction, so every stopped run kept train/last_ema.pth. train_rfdetr.py --predict-only re-runs only the prediction and scoring from that file (training time recovered from the run log). Verified on a toy run: recovered and directly scored metrics identical (test, val, probes); log-based training time 147.0 s vs 146.3 s measured. Notebook 16 does this on Kaggle from the attached notebook-15 output (folders or the results zip).
- Parts B and C were already running with the old code: each run will train fully and stop at the same step; recover them with notebook 16. If a part's output was not saved (the old zip step also packed checkpoints and could exceed Kaggle's 20 GB output limit), rerun only that part with the fixed notebook 15 (zips now exclude checkpoints and COCO copies).
- Measured RF-DETR-M time at 640, batch 4 x 4, two runs in parallel on T4 x2: about 160 min per v2 run, 230-240 min per official run.
- .gitignore now also excludes *.ckpt and coco/ (RF-DETR checkpoints and per-run COCO copies).
- 2026-11-26, notebook 14 part B (RT-DETR-L): 7/7 runs completed; 1 failed run under the failure rule: clean fold 1 seed 44, test mAP50-95 0.0000. In-training (fp16) validation NaN from epoch 7 to the end, one NaN train-loss epoch (8), training loss finite and normal at epoch 100 (0.531; healthy runs 0.48-0.63). Same signature as the 2 failed v1 runs (docs/RESULTS_RTDETR.md): consistent with the evaluated EMA weights being contaminated once; cause not isolated (weights not kept). The Ultralytics RT-DETR trainer documentation itself notes that AMP training can lead to NaN outputs. Counted as measured; no re-run. 9/15 v2 RT-DETR runs had at least one NaN validation epoch (v1: 10/15); YOLO 1/45 (YOLOv8m fold 3 seed 43, one epoch, recovered). Over both protocols: RT-DETR-L 3/30 clean runs failed, YOLO 0/80.
- RT-DETR-L row complete (scripts/evaluation/benchmark_v2.py): v2 14.75 as measured (post-hoc without the failed run: 16.13). vs YOLOv8s: -0.38 pt [-3.45, +2.69], 3/5 folds -> not distinguishable; post-hoc +1.01 pt [-0.93, +2.95], 4/5 folds -> not distinguishable (below the 1.8 pt gate). Official 49.13 (sd 1.26); official / v2 3.33x.
- 2026-11-26, notebook 16 part A (RF-DETR recovery): 4/4 runs recovered from last_ema.pth. Official seeds 42/43/44: 48.10 / 48.89 / 49.14 (mean 48.71, sd 0.54; YOLOv8s 49.42); clean fold 0 seed 42: 17.80. Checks: class ids 0-9 only; 53-162 no-object entries dropped per test set; 113-300 predictions per image (top 300); our val mAP50-95 vs RF-DETR's own COCO-style validation printout at epoch 100: 49.34/49.52, 49.72/49.89, 47.51/47.57, 27.68/28.25 (ours 0.06-0.57 pt lower; different metric definitions, so a plausibility check, not parity). No RF-DETR comparison before its 15 v2 runs.
- 2026-11-26, notebook 15 parts B and C: all 14 runs trained 100 epochs and stopped at the old prediction bug, as expected; outputs saved (about 9.7 GB each, checkpoints included). Recovery with notebook 16, one part per session.
- Table script: scripts/evaluation/benchmark_v2.py -> results/analysis/benchmark_v2.{json,md}. Explicit run directories per detector; checks: no duplicate runs, identical image counts and test GT boxes per fold across detectors, evaluator parity. Reproduces the YOLOv8m numbers of 2026-11-24 exactly. YOLO11s vs YOLOv8s on v2 under the same rule: +0.26 pt [-2.10, +2.62], 3/5 -> not distinguishable.
- Exploratory (labelled, not a pre-declared test): official leak probe = test images with a train copy (48) minus without (16): YOLOv8s +4.45, YOLO11s +4.88, YOLOv8m +5.51, RT-DETR-L +4.42, RF-DETR-M +0.53 pt (mean of 3 seeds; the 16-image subset makes this noisy).
