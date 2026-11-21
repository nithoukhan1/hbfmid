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

## Update 2026-11-18 (Gate C read) - see docs/RESULTS_GATE_C.md
- Notebooks 06 (baseline 177 epochs), 07 (arm 7' seeds 43/44), 05-zoom (arms 6 and 7', seed 42) done. Result folders renamed to results/ctrl_ep177, results/arm7_seeds, results/arms_fa_zoom; comparisons copied to results/arms/.
- Arm 7' generic by seed: +1.57 (s42), -1.05 (s43), +0.50 (s44) pts; 3-seed mean +0.34 pt (3/5 folds). The pilot gain was seed noise. Compute control (177 epochs) -0.2 pt. Zoom: no effect. Class weights and pre-training: no effect or negative.
- GATE C FAILS (mAP50-95). Joint training raises localisation recall by 2-3 pts in most folds (exploratory, CI includes 0 for 3 seeds). Branch taken: benchmark + audit + analysis paper (Q3-Q4), FracAtlas/class-weight results reported as negative findings.
- New: scripts/evaluation/aggregate_benchmark.py (official vs clean table per detector), notebooks/kaggle/08_benchmark_detectors.ipynb (PART A: YOLO11s 3 seeds, 18 runs; PART B: YOLOv8m seed 42 + YOLOv8s official seeds 42-44).
- NEXT: user pushes results + notebook 08, runs PART A and B; then RT-DETR/DEIMv2 row, per-class/modality analysis, label-inconsistent group inspection, manuscript outline.

## Update 2026-11-19 (notebook 08 done) - see docs/RESULTS_BENCHMARK.md
- Part A (YOLO11s: 15 clean + 3 official runs) and Part B (YOLOv8m seed 42 on both protocols; YOLOv8s official seeds 42-44) finished, no failed runs. Folders results/bench_A, results/bench_B; merged table results/benchmark_all.{md,json}.
- Official mAP50-95: YOLOv8s 0.494, YOLO11s 0.485, YOLOv8m 0.472. Clean: 0.217, 0.2225, 0.227. Official/clean = 2.1-2.3 for all three. Orders reverse between protocols; no clean difference is significant (11s-8s +0.54 +/-1.83 pts, 8m-8s +1.20 +/-2.21 pts, seed 42 only).
- Official seed range for one detector (YOLO11s 3.4 pts) is larger than the gaps between detectors. Misses = 38-39% of boxes on clean vs 10-14% on official for all three; wrong class about 5% on clean.
- NEXT: git add/commit/push results/bench_A results/bench_B results/benchmark_all.* docs; audit validation (manual pair review + threshold sensitivity, no GPU); RT-DETR/DEIMv2 row; per-class and modality tables; optional audit on FracAtlas.

## Update 2026-11-19 (miss analysis; push 0e24b1e confirmed) - see docs/RESULTS_MISS_ANALYSIS.md
- Pushed benchmark (0e24b1e). New: scripts/evaluation/miss_analysis.py, results/analysis/*.
- Misses are not explained by box size (flat to U-shaped; median area found 2.2% vs never found 1.9%), so a resolution/P2 experiment is deprioritised. 47% of boxes are found by all 7 runs, 29% by some (noise), 24% by none (hard core). Class is right 91-93% when localised.
- Correction: the audit already contains a threshold sensitivity (thr 12..40: test images with a train copy 54/64 .. 44/64; groups 244 .. 518). Group count depends on the threshold; the leakage conclusion does not. Missing: manual review of a sample of matched pairs.

## Update 2026-11-19 (method-track literature check; notebook 09) - see docs/LITERATURE_METHOD_TRACK.md
- Literature (web only): modality-conditioned mixed-modality detection exists (Align Your Query, arXiv 2510.02789, DETR + modality tokens); modality-specific normalisation exists in unpaired multimodal segmentation. Application to HBFMID not found, but it would be application novelty only. Candidates M1-M4 recorded; no method chosen yet.
- Decision on the leakage question: report the official split for comparability, state the leakage, make HBFMID-Clean the headline protocol; any method claim needs the pre-registered gate (paired >= +0.9 pt mAP50-95, >= 4/5 folds, not worse on official) on Clean.
- New: scripts/training/run_baseline.py can train RT-DETR (load_model(), --no-amp; YOLO behaviour unchanged, amp default stays True); notebooks/kaggle/09_benchmark_rtdetr.ipynb (PART A rtdetr-l seed 42 official + 5 clean folds; PART B seeds 43/44; SMOKE switch). Not yet run; RF-DETR/DEIMv2 deferred (needs a COCO-format pipeline and a common mAP evaluator).
- NEXT: user pushes; run 09 with SMOKE=True, then PART A; conference-paper steps 1-3 (audit pair review, MRI count 86 vs 131, never-found boxes) in parallel.

## Update 2026-11-20 (notebook 09 done: RT-DETR-L, 18 runs) - see docs/RESULTS_RTDETR.md
- Parts A (seed 42) and B (seeds 43, 44) finished, no crashed runs. Official 0.4913 (sd 0.0126) = YOLO. Clean: 0.1814 as measured, 0.2120 without the 2 broken runs, 0.2023 best.pt; vs YOLO11s 0.2225 -> no view shows a gain (healthy-only -1.05 +/- 0.69 pts, 0/5 folds).
- 2 of 15 clean runs gave an unusable last.pt (fp32 test mAP 0.0000, train loss normal, validation NaN from epoch 16 / 68; probable EMA contamination); a third had fp16-only NaN validation. Reported as failures, not dropped. Cause not isolated.
- Folders renamed to results/rtdetr_A, results/rtdetr_B (smoke zip kept, git-ignored). Merged table results/benchmark_all.* regenerated with the RT-DETR row.
- Not tested: foundation-backbone detector (RF-DETR/DINOv2). Method-track gate (docs/LITERATURE_METHOD_TRACK.md) unchanged: build M3 only if such a detector gains >= 0.9 pt on Clean.
- NEXT: user pushes; decide RF-DETR (time-boxed) vs closing the method track; conference steps 1-3 (audit pair review, MRI 86 vs 131, never-found boxes).

## Update 2026-11-21 (audit validation prepared) - see docs/AUDIT_VALIDATION_PLAN.md
- Option C chosen provisionally: finish audit validation first, then decide evaluation paper vs evaluation + one gated method experiment (RF-DETR, group-aware training).
- New: scripts/eda/audit_validation.py (build/score), results/eda/audit_validation/{review.html (git-ignored, embeds images), review_key.csv, auto_checks.json}. Decision rules written before labels.
- Automatic checks: 109/113 clean 3-copy augmentation sets fully grouped (95.9% of pairs direct >= 25), so errors lean toward missed duplicates; 47/48 leaked test images have a direct link >= 25, 1 via a strong chain (232 / 289 inliers).
- NEXT: user reviews 66 pairs in review.html (about 25 min), saves review_results.csv into results/eda/audit_validation/, then Claude runs score and applies the decision rules.

## 2026-11-21 Audit validation results (AI-assisted review, 66 pairs)
- Headline 48/64 test images with a train copy: validated (T_weakest 12/12, C_chain 2/2, controls 4/4).
- Rules 3, 4, 5 triggered: false links via overlay markers (train-train only); below-threshold and rotated/cropped duplicates missed (R_below 9/16, N_hard 8/8) -> leakage is a lower bound and HBFMID-Clean still contains cross-fold duplicates.
- Thumbnail scan: 206 unlinked pairs with NCC>=0.95 (test 4/64, valid 7/128 with an unlinked train partner).
- NEXT: grouping v2 (dihedral thumbnail/embedding + marker-masked SIFT), recompute folds, human spot-check, then decide benchmark re-run and paper option (A/B/C). See docs/RESULTS_AUDIT_VALIDATION.md.
