# Which boxes are missed? (2026-11-19)

Script: `scripts/evaluation/miss_analysis.py`; outputs `results/analysis/miss_analysis*.json` and `*_boxes.csv`. Saved predictions only, no GPU.
Data: HBFMID-Clean test boxes (457 unique boxes, 392 images), 7 runs per box (YOLOv8s x3 seeds, YOLO11s x3 seeds, YOLOv8m x1). Found = any prediction with IoU >= 0.5 (any class) at conf >= 0.25 (also reported at 0.05).

## 1. Box size does not explain the misses
Miss rate at conf 0.25 by box area (fraction of image):

| area | n (per run) | YOLOv8s | YOLO11s | YOLOv8m |
|---|---|---|---|---|
| < 1% | 71 | 0.418 | 0.465 | 0.451 |
| 1-2% | 164 | 0.415 | 0.398 | 0.378 |
| 2-5% | 161 | 0.333 | 0.319 | 0.335 |
| 5-15% | 40 | 0.417 | 0.417 | 0.425 |
| >= 15% | 21 | 0.476 | 0.413 | 0.524 |

The curve is flat to U-shaped: very large boxes (mostly Healthy / whole-bone boxes) are missed as often as the smallest lesions. Median box area is 2.2% for boxes found by every run and 1.9% for boxes found by none. Consequence: a resolution or P2-head experiment aimed at "tiny lesions" has weak support from our own data and is not prioritised.

## 2. Misses are partly noise, partly hard boxes
Over the 7 runs per box (conf 0.25): found by all 7 = 215 (47%); found by some = 131 (29%); found by none = 111 (24%). At conf 0.05: 239 / 126 / 92 (52% / 28% / 20%).
The 29% "sometimes found" boxes are the seed/detector noise that limits what a single run can claim. The 24% never found are the hard core (label ambiguity, subtle fractures, low contrast, or annotation issues; not yet inspected).

Never found by class (unique boxes): Transverse Displaced 32/181 (18%), Oblique Displaced 28/107 (26%), Greenstick 6/43 (14%), Comminuted 12/38 (32%), Transverse 6/35 (17%), Spiral 9/22 (41%), Healthy 6/12 (50%), Oblique 5/10 (50%), Linear 4/5, Segmental 3/4.

## 3. Modality
Miss rate (conf 0.25): MRI 0.457 / 0.484 / 0.473 vs X-ray 0.371 / 0.370 / 0.376 (YOLO11s / YOLOv8m / YOLOv8s). MRI is 62 of 457 boxes (14%); never-found: MRI 17 of 62 (27%), X-ray 94 of 395 (24%).

## 4. Classification is not the problem
When a box is localised, the class is right in 91.3% (YOLO11s), 92.5% (YOLOv8m), 91.9% (YOLOv8s) of cases.

## 5. Next
Inspect a sample of the 111 never-found boxes (do they have a consistent label? are they duplicates with conflicting annotations?). This also feeds the open item on the 4 label-inconsistent groups.
