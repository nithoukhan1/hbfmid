# Method track under option C: design and gate (2026-11-23)

## 1. What changed since the literature note (docs/LITERATURE_METHOD_TRACK.md)
Miss analysis on the v2 folds (results/analysis/miss_analysis_v2.*, YOLO11s + YOLOv8s seed 42, conf 0.25, IoU 0.5, 328 unique test boxes, 6 runs pooled):
- Box miss rate by modality: MRI 42-48%, X-ray 57-59%. On v1 folds MRI looked worse (46-48% vs 37%); that was leakage (X-ray copies leaked more). With duplicates removed MRI is NOT the weak modality.
- Within classes present in both modalities: Comminuted MRI 0.33 vs X-ray 0.83 miss; Greenstick 0.50 vs 0.60; Oblique Displaced 0.61 vs 0.63; Transverse Displaced 0.52 vs 0.46; Linear 0.50 vs 1.00. No consistent MRI deficit.
- Weakest classes (X-ray only): Linear 0.90, Segmental 0.88, Oblique 0.70, Spiral 0.70, Transverse 0.71, Greenstick 0.60, Oblique Displaced 0.63. 158/328 boxes (48%) are never found by any run; miss is not driven by box size (median area 2.1% found vs 1.8% never found).
Consequence: the premise of M1 (modality-specific normalisation), M3 (modality-conditioned detector) and M4 (cross-modal restyling), "MRI underperforms, so condition on modality", is not supported. Modality is also a class shortcut (it alone flags 69% of Comminuted test images), so a modality-conditioned gain could be shortcut learning. Modality candidates are demoted to analysis (shuffled-modality control), not a headline method.

## 2. Candidates (ranked by motivation x cost)
| id | idea | motivation from our data | cost to test | notes |
|---|---|---|---|---|
| C1 | duplicate-aware (group-balanced) training: cap large groups at 12 images, repeat small groups (up to x4) to a target of 7 effective images per group; epochs scaled to equal optimisation steps | v2: 282 groups, median 5 images, max 30; 56 groups with >= 8 images hold 36% of images. Training weights images, the test counts groups (one representative per group): a train/test weighting mismatch | notebook 12: 15 runs per detector, about 3.5 h per part | implemented in run_baseline.py (--group-balance, --gb-match-steps); dry run verified (fold 0: 169 groups, 897 -> 1136 images, epochs 100 -> 79) |
| C2 | higher input resolution (960) | boxes median 1.5-4.4% of the image, 48% never found | similar runs, about 2x slower | not built; images are 640x640 so this is upsampling; weak prior |
| C3 | foundation-backbone detector (RF-DETR, DINOv2 backbone) | small data (about 170-230 groups per training fold) favours pretrained backbones | new COCO-format pipeline + evaluator parity | a benchmark row rather than a method; closest to the 'foundation backbone' idea; build only if C1 fails |
| C4 | modality conditioning | see section 1 | | analysis only |

Order: C1 now (cheap, motivated by the audit itself); C3 second; C2 only if time.

## 3. Pre-registered gate for any method (written before the runs)
- Base detectors: YOLOv8s (higher v2 seed-42 mean, 0.159) and YOLO11s (0.143). [3 seeds, 2026-11-23: 0.151 vs 0.154, tied; YOLOv8s stays primary as pre-declared] Method runs use the same seeds (42, 43, 44) and folds as the plain baseline (notebooks 10 + 11), same fixed protocol, same optimisation steps (--gb-match-steps).
- Primary metric: mAP50-95 on the v2 test folds, paired by fold, seed-averaged. Pass = mean gain >= +0.9 pt [SUPERSEDED 2026-11-23 by section 3b: +1.8 pt] AND >= 4/5 folds positive, in the YOLOv8s part; the YOLO11s part is a replication (same direction required to claim generality).
- Secondary: macro (8 classes), mAP50, box recall; official split (3 seeds) must not be worse beyond noise (if the method gains on official but not on v2, report it as leak exploitation).
- Noise floor: from notebook 11, seed sd of the 5-fold mean on v2; the +0.9 pt gate is only meaningful if it exceeds about 2 seed sd; recompute the gate from the measured sd before looking at method results.
- Stop rule: if C1 fails the gate in the YOLOv8s part, do not tune it (no extra hyper-parameter search on the test folds); report it as a negative result and go to C3 or stop the method track. One pre-declared setting only (target 7, cap 12, max repeat 4).
- Tooling: scripts/evaluation/compare_method.py --arm gb --out results/analysis/method_gb.

## 3b. Gate fixed from the measured v2 noise (2026-11-23, after notebook 11, before any method run)
Source: results/analysis/noise_floor_v2.{json,md} (scripts/evaluation/noise_floor_v2.py); plain baselines, seeds 42/43/44 x 5 folds, YOLO11s and YOLOv8s.
- Per-run seed sd at a fixed fold, pooled over folds and both detectors: 2.08 pt (20 df). The v1 estimate behind the old gate was 1.26 pt (4 df).
- Seed sd of the 5-fold mean: YOLO11s 1.17 pt, YOLOv8s 0.94 pt (pooled 1.06 pt). One-seed A/A differences of the 5-fold mean reach 2.34 pt (YOLO11s seed 42 vs 43).
- Detectable effect, same convention as docs/RESULTS_CONTROLS.md (2 sigma of the paired difference, 3 seeds x 5 folds): 1.52 pt from the per-run sd; 1.74 pt from the seed sd of the 5-fold mean (the statistic named in section 3).
- Decision: +0.9 pt is below the v2 noise and is replaced. Final gate: mean paired gain >= +1.8 pt mAP50-95 (larger estimate, rounded up) AND >= 4/5 folds positive, in the YOLOv8s part (primary). YOLO11s is the replication: mean gain > 0 is needed to claim generality. All three seeds must be complete before the gate is read.
- Power: with 3 seeds the experiment confirms a true gain of about 2.5 pt with about 80% probability; a true 1 pt gain would pass at most about 15-20% of the time (less with the 4/5-folds rule). A FAIL means 'no gain large enough to detect', not 'no effect'; the gain and its interval are reported either way. No seed extension after seeing the result (stop rule of section 3 unchanged).
- Secondary, descriptive only: gain pooled over both detectors (30 pairs).
- Implementation checks done before launch: group-balanced lists built for all 5 folds with the notebook 12 arguments (groups 169/170/170/169/168; images 897->1136, 940->1146, 957->1160, 935->1149, 888->1133; epochs 79/82/82/81/78). Ultralytics 8.4.215 keeps repeated lines of a list file (get_img_files does not de-duplicate; the label cache iterates over every entry), so repeated images are sampled as intended. After the run: check metrics.json 'group_balance' and the 'Scanning ... N images' line in the logs.
- scripts/evaluation/compare_method.py: default --gate 0.018; method runs matched to the detector by metrics.json 'model' (A/B letters do not matter); refuses to read the gate unless seeds 42-44 are complete.

## 4. Risks
- The gain may be zero: the baseline already trains on all copies as light augmentation; balancing could hurt by overfitting repeated small groups (up to x4).
- v2 test folds have about 57 images: a +0.9 pt gate may be below the fold-level noise; the paired design and 3 seeds are the mitigation. [Confirmed by notebook 11: it was below the noise; gate raised in 3b]
- Group-balancing relies on grouping v2; residual duplicate tail (band 0.90-0.95) is ignored.
- Novelty: group-balanced training is simple; it supports the paper's thesis (duplicates distort training and evaluation) but would be a modest method contribution. If it passes, frame it as a duplicate-aware training protocol, not a new architecture.

## 5. Execution plan
1. Now: user starts notebook 11 (baselines seeds 43/44 on v2). 2. When it finishes: compute the noise floor (compare_v2_seeds.py), confirm the gate. [Done 2026-11-23: gate +1.8 pt, section 3b] 3. Run notebook 12 (parts A and B). 4. compare_method.py -> gate decision -> document. 5. Official-split check for the passing method (3 seeds).
