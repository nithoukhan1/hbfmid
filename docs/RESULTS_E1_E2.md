# Results: E1 (official) vs E2 (HBFMID-Clean), YOLOv8s, 100 epochs, early stopping patience 30, 2026-11-17

Environment: ultralytics 8.4.215, torch 2.11.0+cu130, Python 3.13, Tesla T4 (two runs in parallel). All 8 runs OK. Run time 16-32 min each (about 1.6 h wall for all 8).
Source files: results/baseline/baseline_results/{baseline_summary.md,*/metrics.json,error_decomposition.json}

## Headline
| | test mAP50 | test mAP50-95 | macro (>=10 groups) |
|---|---|---|---|
| E1 official split (64 test images, 66 boxes) | 0.940 | 0.490 | 0.490 |
| E2 clean, mean of 5 folds (392 test reps, 457 boxes) | 0.43 (folds 0.32-0.50) | 0.208 +/- 0.035 | 0.256 +/- 0.049 |
Gate A PASS: the official protocol reports about 2.3x the clean mAP50-95 (0.49 vs 0.21) and 2.2x mAP50 (0.94 vs 0.43).
CAUTION on attribution: the leak probe (official test images WITH a detected train copy 0.512 vs WITHOUT 0.466, 16 images) does not isolate leakage; E1 vs E2 also differ in test composition (E2 includes Linear/Segmental), train size (about 900-970 vs 1,347) and val protocol. Control E1b (official split de-duplicated, same test images) is the planned isolating experiment.

## Gate A2 FAIL (and robust to the confidence threshold)
At conf >= 0.25 (aggregator): correct 248, wrong-class 23, missed 186, false positive 70; sibling confusions / all errors = 0.057 (threshold 0.20).
error_decomposition.py (class-agnostic IoU >= 0.5 matching per ground-truth box, E2 pooled, 457 boxes):
| conf >= | localized | correct class | sibling | other class | missed | class-correct among localized |
|---|---|---|---|---|---|---|
| 0.05 | 0.69 | 0.61 | 0.05 | 0.03 | 0.31 | 0.88 |
| 0.10 | 0.65 | 0.58 | 0.05 | 0.02 | 0.35 | 0.90 |
| 0.25 | 0.59 | 0.54 | 0.04 | 0.02 | 0.41 | 0.92 |
Reading: the clean-protocol failure is DETECTION (31-41% of ground-truth boxes get no IoU>=0.5 prediction of any class), not classification. A hierarchy head can at most address the roughly 8% of boxes that are localized with a wrong class. Sibling confusions are concentrated in the displaced / non-displaced pairs (Transverse vs Transverse Displaced, Oblique vs Oblique Displaced).
By modality (conf 0.05): MRI boxes missed 42% vs X-ray 29% (62 vs 395 boxes).
Per class (pooled E2, conf 0.05; localized / correct): Transverse Displaced 0.76/0.72 (181 boxes), Oblique Displaced 0.68/0.63 (107), Greenstick 0.81/0.67 (43), Comminuted 0.53/0.45 (38), Transverse 0.74/0.49 (35), Spiral 0.55/0.55 (22), Healthy 0.42/0.42 (12), Oblique 0.60/0.20 (10), Segmental 0.50/0.00 (4), Linear 0.00/0.00 (5).

## Per-class AP50-95 (E2 mean over folds vs E1)
Comminuted 0.229 vs 0.328; Greenstick 0.291 vs 0.597; Healthy 0.227 vs 0.807; Oblique Displaced 0.269 vs 0.410; Oblique 0.127 vs 0.472; Spiral 0.234 vs 0.532; Transverse Displaced 0.297 vs 0.399; Transverse 0.368 vs 0.374; Linear 0.000 and Segmental 0.000 (not in the official test set). The official numbers for Greenstick (1 test box), Oblique (2), Healthy (3), Spiral (3) rest on a handful of boxes.

## Seed noise (fold 0, seeds 42/43/44)
Test mAP50-95 0.171 / 0.198 / 0.156 (sd 0.021; macro sd 0.024). Early stopping picked best epochs 26, 91, 39 and stopped runs at 56-100 epochs: the 78-image validation set makes checkpoint selection noisy. The +1.0 pt Gate B threshold is NOT detectable with 1 seed (paired sd over 5 folds about 1.3 pt, so about 2.7 pt for 2 sigma). Remedy: fixed-epoch protocol (last.pt) and 3 seeds (about 1.5 pt for 2 sigma); measured in controls notebook 02.
