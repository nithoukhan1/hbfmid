# HBFMID (Project 3) — Tracker (updated 2026-11-16)

Dataset (read-only, never modified): ../Bone Fractures Detection
Workflow: VS Code -> GitHub (repo not created yet) -> Kaggle clone -> "Save Version" training

## OVERALL
| # | Phase | Status |
|---|-------|--------|
| 0 | Read base paper | DONE |
| 1 | Literature landscape | DONE (first pass) |
| 2 | Data audit / EDA | DONE for structure + leakage; modality split pending |
| 3 | Idea selection + novelty check | IN PROGRESS (provisional pick below) |
| 4 | Framework design | TODO |
| 5 | Implementation | TODO |
| 6 | Baselines + experiments (Kaggle) | TODO |
| 7 | Ablations, statistics, explainability | TODO |
| 8 | Manuscript | TODO |
| 9 | Journal choice + submission | TODO |

## PROGRESS (current: phase 3 -> 4)
- [x] Repo skeleton created (docs, scripts, notebooks, configs, data_splits, results)
- [x] scripts/eda/leakage_audit.py written and run (SIFT+RANSAC, flips/rotation aware)
- [x] Visual verification: 10/10 random verified test-train pairs are true copies
- [ ] scripts/eda/make_clean_splits.py (group-stratified folds)   <- NEXT
- [ ] Baseline notebook: official split vs clean folds (YOLOv8/11 + others)
- [ ] Create GitHub repo, connect, push
- [ ] Final idea decision (see docs/literature_evidence.md)

## LEAKAGE AUDIT RESULT (threshold 25 inliers; sensitivity 12-40 in results/eda/leakage_audit_summary.json)
- test: 48/64 images (75%) have a verified copy in train (range 44-54 across thresholds)
- valid: 96/128 (75%) have a verified copy in train
- test images with a copy in valid: 18/64
- 105 groups span more than one split
- Independent groups ~388 (upper-bound estimate; threshold 25 is conservative, borderline 12-24 contains real duplicates) vs 1,539 images
- Independent groups per class: Transverse Displaced 163, Oblique Displaced 102, Greenstick 43, Comminuted 36, Transverse 31, Spiral 22, Healthy 12, Oblique 10, Linear 5, Segmental 4


## Update 2026-11-16 (night)
- GitHub repo created by user: https://github.com/nithoukhan1/hbfmid (public). Local repo not yet connected.
- Idea v2 proposed (docs/IDEA_V2_METHOD.md): FracTree head + audit-calibrated imbalance + HBFMID-Clean protocol. Pending Gate A2/B.
- Added scripts/eda/class_group_stats.py + results/eda/class_group_stats.csv.

## Update 2026-11-17
- User approved the plan in docs/IDEA_V3.md (A + B with gates). Pilot detector: YOLOv8s; confirm on YOLO11s / YOLOv8m.
- DONE: scripts/eda/make_clean_splits.py (run: 388 groups, 5 folds, pooled OOF test 392 images), scripts/eda/verify_splits.py (all checks pass), data_splits/{official,clean5}/, scripts/common/taxonomy.py, scripts/evaluation/{confusion,aggregate_baseline}.py, scripts/training/run_baseline.py (dry-run tested; GPU path untested), notebooks/kaggle/01_baseline_E1_E2.ipynb.
- NEXT: user git init/commit/push of hbfmid-research -> run the Kaggle notebook (E1 + E2 folds, then noise-floor seeds) -> aggregate -> Gates A and A2.
- Pending: pin Ultralytics version after the first run; modality tags; FracAtlas inspection.

## Update 2026-11-17 (evening)
- Kaggle smoke test passed (ultralytics 8.4.215, torch 2.11.0+cu130, python 3.13, 2x T4; 2 runs in parallel, 2.2-2.3 min each for 2 epochs). Real run E1 + E2 folds 0-4 + noise seeds 43/44 started by user (Save Version "E1E2 v1").
- Modality tags: scripts/eda/tag_modality.py -> results/eda/modality_tags.csv, modality_summary.json. Rule = 'mri' in file name (no group mixes modalities). 202 MRI images / 56 groups (13%), 1,337 X-ray; name-tag gives 86 raw MRI, the dataset card says 131 (unexplained; 0/48 sampled untagged groups were MRI). Pooled out-of-fold test: 57 MRI of 392. Plan: report X-ray and MRI pooled, not per fold. Simple intensity statistics do NOT separate modality (tested, failed).
- requirements.txt pin ultralytics==8.4.215 (local; commit pending from user's repo folder).

## Update 2026-11-17 (FracAtlas inspection)
- FracAtlas (local: HBFMID/FracAtlas (FigShare); Kaggle: /kaggle/input/datasets/nithoukhan/fracatlasfigshare/FracAtlas). scripts/eda/fracatlas_inspect.py -> results/eda/fracatlas_summary.json.
- Annotations are internally consistent: 4,083 images, 717 fractured, 922 boxes (YOLO class 0 only); box count == fracture_count for every image; no box on any non-fractured image; no fractured image without a box. Three JPEGs are truncated ("Premature end of JPEG file" warnings): handle with PIL LOAD_TRUNCATED_IMAGES / skip.
- SCALE GAP (design risk for Gate C): median fracture box side at imgsz 640 = 37 px (p10 23, p90 69) in FracAtlas vs 88 px (p10 58, p90 146) in HBFMID non-Healthy boxes. FracAtlas boxes are about 2.4x smaller and mostly smaller than nearly every HBFMID box. Mitigation to test: scale-matched zoom augmentation on FracAtlas (about 2-3x), plus an arm without it.
- FracAtlas image sizes: 373x454 / 454x373 (2,704 + 1,023), 2304x2880 (303), others. Region of fractured images: hand 438, leg 263, hip 63, shoulder 63, mixed 106.
- scripts/eda/cross_dataset_audit.py: 388 HBFMID group representatives vs 4,083 FracAtlas images (thumbnail NCC candidates, 7,760 pairs, SIFT + RANSAC): best pair 8 inliers (threshold 25; true duplicates inside HBFMID gave >= 25) -> NO evidence that FracAtlas images are inside HBFMID. Limit: method catches copies with flips, small rotation and brightness change, not heavy crops.

## Update 2026-11-17 (modality shortcut + FracAtlas manifest)
- scripts/eda/modality_shortcut_baseline.py -> results/eda/modality_shortcut_baseline.json (rule fitted on train folds, scored on pooled 392 test reps). Modality alone barely beats always-majority (hit rate 0.429 vs 0.418; rule = MRI->Comminuted, X-ray->Transverse Displaced). But 25/36 (69%) of Comminuted test reps are MRI and 25/57 (44%) of MRI test reps are Comminuted: a partial shortcut for Comminuted only. Plan: report Comminuted separately for MRI and X-ray (X-ray Comminuted has only 11 groups: exploratory).
- scripts/data/fracatlas_manifest.py -> data_splits/fracatlas/manifest.csv (+ manifest_summary.json): 4,083 rows, boxes as normalised cx,cy,w,h (class 0), region flags, decode_ok. 59 images fail full decode (truncated JPEG), ALL non-fractured -> training loaders must skip rows with decode_ok == False. Dataset folders untouched.
- DECISION: do not write the FracTree head code until Gates A and A2 are read (A2 failing would weaken the motivation for the tree).

## Update 2026-11-17 (E1/E2 results) - see docs/RESULTS_E1_E2.md and docs/IDEA_V3_1_DECISION.md
- E1/E2 done (8 runs, all OK, about 1.6 h wall). Gate A PASS (0.490 vs 0.208 mAP50-95). Gate A2 FAIL (sibling share 5.7%; 31-41% of GT boxes unlocalized). Seed sd 2.1 pts.
- DECISION: FracTree dropped as core; paper A (benchmark + audit + error analysis) is the anchor; improvement study = FracAtlas transfer for detection recall (Gate C); fixed-epoch protocol (last.pt) + 3 seeds from now on.
- New code (to commit): scripts/eda/make_official_dedup.py + data_splits/official_dedup/, run_baseline.py (--mode official_dedup, --fixed-epochs), scripts/evaluation/{aggregate_controls,error_decomposition}.py, notebooks/kaggle/02_controls_E1b_noise.ipynb. run_baseline.py changes are dry-run tested only.
- NEXT: Kaggle notebook 02 (smoke first), then pilot arms.
