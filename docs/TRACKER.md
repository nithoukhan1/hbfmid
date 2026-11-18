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

## Update 2026-11-17 (controls) - see docs/RESULTS_CONTROLS.md
- E1b (official split de-duplicated, same 64 test images): 0.166 mAP50-95 vs 0.490 leaky. Box recall on the 48 test images with a train copy: 0.98 -> 0.37; on the 16 without: 0.71 -> 0.59. Leakage attribution confirmed (AP-based probe was too noisy; use box recall).
- Fixed-epoch protocol (last.pt): mean seed sd 0.0126 vs 0.0238 for validation-selected best.pt. Detectable effect about 1.6 pts (1 seed) / 0.9 pt (3 seeds), rough (4 df).
- New: scripts/evaluation/{leak_probe_recall,aggregate_arm}.py, notebooks/kaggle/03_arm1_baseline_fixed.ipynb (folds 2-4 x seeds 42/43/44).
- NEXT: run notebook 03 (about 2.4 h wall); meanwhile build FracAtlas loader + class-balanced loss arms.

## Update 2026-11-17 (arms built, not yet run) - see docs/ARMS_DESIGN.md
- Controls committed (93d34af); the zip was extracted twice, so results/controls/controls_results/ duplicates results/controls/ (remove with git rm -r; compare/aggregate scripts only read one level, so results are unaffected). Notebook 03 (arm 1, folds 2-4 x 3 seeds) started by the user.
- New code (CPU dry-tested on a synthetic dataset, not run on GPU): scripts/training/class_weights.py, pretrain_fracatlas.py; scripts/data/prepare_fracatlas_yolo.py (generic + scale-matched zoom variants, tested on 30 real FracAtlas images: zoom brings the anchor boxes to a median 88.3 px, crops checked visually); run_baseline.py gains --cls-weight {raw,group} --cls-pw --init-weights --extra-train --extra-names --arm; aggregate_arm.py --arm-tag; scripts/evaluation/compare_arms.py (paired by fold, localisation recall); notebooks 04 (arms 2, 3) and 05 (arms 6, 7'; VARIANT generic|zoom).
- Verified: our class weights equal Ultralytics' native cls_pw=0.5 weights (callback reaches the criterion). FINDING: Ultralytics 8.4.215 already ships cls_pw, so arm 2 is a built-in option and arm 3 (group counts) is a thin extension; raw and group weights differ <= 33% per class, so arms 2 vs 3 will likely not be separable.
- Settings fixed before any arm result: p = 0.5; FracAtlas pre-training 50 epochs seed 42; crops: target 88.3 px, max zoom 4, 2 per image; pilot = seed 42 x 5 folds; advance rule in ARMS_DESIGN.md.
- NEXT: user commits + pushes; notebook 03 result must be pushed before 04/05 (they compare against it); approval to spend GPU on 04 then 05.

## Update 2026-11-18 (notebooks 03, 04, 05 done) - see docs/RESULTS_ARMS_PILOT.md
- Arm 1 complete: 15 runs, 5 folds x 3 seeds, mAP50-95 0.2170 (macro 0.2663), seed sd of the fold mean 0.0056. Results in results/baseline_fixed/.
- Pilot (seed 42, 5 folds, paired by fold): cbraw +0.0025, cbgroup -0.0096, fapre_generic -0.0211, fajoint_generic +0.0157 (4/5 folds, localisation recall +0.052; class-specific heads only +0.039). Only arm 7' meets the advance rule; its interval still includes 0 (t about 1.4).
- Results folders renamed to results/baseline_fixed, results/arms_cb, results/arms_fa_generic; compare_arms re-run locally on 5 folds -> results/arms/*_vs_baseline.*.
- Open confounds for arm 7': 1.77x training steps (control = notebook 06, baseline 177 epochs), single seed (notebook 07 = seeds 43/44), scale mismatch (notebook 05 VARIANT zoom, RUN_PRE False).
- NEXT: user pushes results + notebooks 05-07; runs 06 and 07 (and 05 zoom).
