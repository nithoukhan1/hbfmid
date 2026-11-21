# RT-DETR-L row (notebook 09, parts A and B) - 2026-11-20

Setup: `rtdetr-l.pt` (COCO pre-trained, Ultralytics 8.4.215), default recipe (AdamW auto, AMP on, batch 16), 100 epochs, no early stopping, last.pt evaluated in fp32. Seeds 42/43/44 on the official split and on the 5 clean folds = 18 runs (3 official + 15 clean), none crashed. Runtime about 91-108 min per clean run and 137-149 min per official run on one T4.

## Result (mAP50-95, test)
| detector | official (seeds) | clean, as measured | clean, diverged runs removed | clean, best.pt |
|---|---|---|---|---|
| RT-DETR-L | 0.4913 (sd 0.0126) [42,43,44] | 0.1814 (V1) | 0.2120 (V2) | 0.2023 (V3) |
| YOLO11s | 0.4846 [42,43,44] | 0.2225 | | |
| YOLOv8s | 0.4942 [42,43,44] | 0.2170 | | |
| YOLOv8m | 0.4724 [42] | 0.2271 (1 seed) | | |

Official mAP50: RT-DETR 0.911, YOLO11s 0.926, YOLOv8s 0.905.

## Findings
1. On the official (leaky) split RT-DETR-L is indistinguishable from YOLO (0.491 vs 0.485 / 0.494; one detector's seed sd is 0.013). Nothing there separates the architectures.
2. On HBFMID-Clean it is not better. Even with the two broken runs removed (a post-hoc exclusion, so optimistic) it is 1.05 +/- 0.69 pts below YOLO11s (0/5 folds positive) and 0.51 +/- 1.65 pts below YOLOv8s (1/5). With all runs counted as measured it is 4.1 +/- 5.2 pts below YOLO11s. With the validation-selected best.pt it is 2.0 +/- 3.9 pts below. No view shows a gain.
3. Reliability problem: 2 of 15 clean runs (fold 4 seed 42, fold 0 seed 44) ended with an unusable last.pt (fp32 test mAP 0.0000) although the training loss stayed finite and normal to epoch 100. Validation went NaN permanently from epoch 16 and epoch 68 respectively, which fits the EMA weights being contaminated once and never recovering. A third run (fold 3 seed 42) had NaN in the fp16 in-training validation from epoch 98 but a valid fp32 test score (0.1757), so fp16 evaluation alone can also produce NaN. 10 of 15 clean runs had at least one NaN validation epoch. We did not isolate the cause (weights are not kept). The YOLO runs had none of these failures.
4. The ratio official/clean is 2.7 as measured (2.3 without the broken runs), in line with the 2.1-2.3 seen for YOLO.

## Reading
* A COCO-pretrained transformer detector does not rescue the clean protocol: the "hard core" of the task is not an architecture-family issue.
* This does NOT test the foundation-backbone idea (RT-DETR-L uses a CNN backbone, not DINOv2). RF-DETR would.
* Reporting rule used: the pre-declared protocol (last.pt, all runs) is the headline; the diverged runs are reported, not dropped; the healthy-only and best.pt views are secondary and labelled as such.
* Limits: default recipe, untuned (a lower learning rate or fp32 might remove the failures); cause of the failures not isolated; one backbone size.

Files: results/rtdetr_A, results/rtdetr_B, results/analysis/rtdetr_divergence.{json,md}, results/benchmark_all.{json,md} (now includes RT-DETR), scripts/evaluation/rtdetr_divergence.py.
