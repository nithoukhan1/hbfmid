# Manuscript outline (option C; draft 2026-11-23, before notebook 11 results)

Status: outline only. 3-seed v2 numbers added 2026-11-23 (notebook 11). Numbers still marked (s42) come from seed-42 runs only. Method section stays empty until the C1 gate is decided (docs/METHOD_DESIGN.md).

## Working titles
1. How leaky is HBFMID? Duplicate-aware evaluation of bone-fracture detectors
2. Near-duplicate leakage inflates bone-fracture detection scores threefold: an audit and duplicate-aware benchmark of HBFMID
(Claim wording rule: 'duplicate-aware' or 'leakage-reduced', never 'leak-free'.)

## Contributions (each one has evidence in the repo)
C1 Audit of the official HBFMID split: 48/64 test and 96/128 validation images have a verified train copy (SIFT + RANSAC); validation by blinded pair review (AI-assisted, 66 pairs) supports the headline and shows the matcher misses rotated/cropped copies. [docs/RESULTS_AUDIT_VALIDATION.md]
C2 Duplicate-aware protocol: group-aware 5-fold splits in two versions (v1 SIFT groups, 388; v2 + thumbnail links, 282 groups) with released split files. v1 had 157 isolated images, v2 only 30. [docs/RESULTS_GROUPING_V2.md, data_splits*/]
C3 Sensitivity of reported accuracy to duplicate detection: official 0.494 -> clean v1 0.217 -> clean v2 0.151 (YOLOv8s), 0.485 -> 0.222 -> 0.154 (YOLO11s), all 3 seeds; official / v2 = 3.1-3.3x. Model-based validation: box recall rises to 0.96-0.97 for test images with a near-identical training image (official, v1) and that bin is empty in v2. [docs/RESULTS_V2_BENCHMARK.md, docs/EVIDENCE_LEAKAGE.md]
C4 Fair benchmark under a fixed-epoch, noise-floor protocol: 5 detectors (YOLOv8s, YOLO11s, YOLOv8m, RT-DETR-L, RF-DETR-M with a DINOv2 backbone; the last three on v2 in progress, docs/BENCHMARK_V2_PLAN.md) on official and clean folds, all scored with one verified evaluator (exact Ultralytics replica); with 3 seeds no detector difference exceeds the noise under any protocol (v2: YOLO11s - YOLOv8s +0.26 +/- 2.36 pts); single-seed comparisons flip sign (v2 seed 42: YOLOv8s ahead by 1.7 pts; seed 43: YOLO11s ahead by 2.5 pts); v2 per-run seed sd 2.1 pts, about twice v1. Message: single-seed leaderboards on HBFMID are not reliable. RT-DETR training failures reported, not dropped. [docs/RESULTS_RTDETR.md]
C5 Error analysis on duplicate-aware folds: 39% of test boxes are never found by any of 6 runs per fold (2 detectors x 3 seeds), 28% by every run, 33% depend on the run; misses not explained by box size; MRI is not the weak modality (miss 42-48% vs X-ray 57-59%); the earlier MRI deficit was leakage. [results/analysis/miss_analysis_v2.*]
C6 Controlled negative results: class weights, FracAtlas pre-training, joint training, compute control gave no reliable gain (v1 folds); group-balanced (duplicate-aware) training on v2 with a pre-registered gate: -0.47 pt [-1.77, +0.83] (YOLOv8s), -0.59 pt [-2.09, +0.90] (YOLO11s), FAIL. [docs/RESULTS_METHOD_GB.md]
C7 dropped (2026-11-24): group-balanced training failed its gate; it is reported under C6.

## Section plan (EMBC-style 4-6 pages; journal version expands each section)
1 Introduction: augmented-copy datasets, why detection benchmarks on HBFMID look high; research questions RQ1 how much leakage, RQ2 how much does it inflate scores, RQ3 do detector rankings survive.
2 Related work: HBFMID detection papers; data leakage in medical imaging (Sci Data 2022 OCT, Sci Data 2025 dermatology, Sci Rep 2021 wrist fracture evaluation, arXiv 2508.17416); near-duplicate detection.
3 Data and audit: dataset structure (1,539 images, Roboflow 3x train augmentation); matcher (SIFT + RANSAC, inlier threshold 25); thumbnail linking (NCC >= 0.95, 8 dihedral modes); validation protocol with pre-registered rules.
4 Protocols: official, clean v1, clean v2 (group fold construction, one representative per group and label set, eligible classes), fixed-epoch training, noise floor (seed sd), paired statistics.
5 Results: Fig 1 score chain; Table benchmark (4 detectors x 3 protocols); Fig 2 similarity-recall; error decomposition (miss by size, class, modality; Fig 3 group-size skew); RT-DETR reliability.
6 Discussion: what 'clean' means (upper bound, residual tail in 0.90-0.95 band), implications for HBFMID papers, recommendations (report group-aware folds, release split files, report seed noise).
7 Limitations: single dataset; AI-assisted visual review (not radiologists), human spot-check pending; thumbnail threshold judgement; test folds of about 57 images; groups are not patients (no patient IDs); detectors limited to YOLO/RT-DETR.
8 Conclusion + code/split release.

## Figures and tables (source files)
- Fig 1 score chain: results/figures/fig1_score_chain.{png,pdf} (script scripts/figures/make_figures.py) [done, 3 seeds]
- Fig 2 similarity vs recall: results/figures/fig2_similarity_recall.* [done]
- Fig 3 group-size skew v1 vs v2: results/figures/fig3_group_sizes.* [done]
- Fig 4 audit examples (verified duplicate pairs, rotated/cropped): needs dataset images; HBFMID is CC BY 4.0 (confirm on the Mendeley page) so figures need attribution; build from the validated pairs (not yet made)
- Fig 5 per-class / per-modality miss rates on v2 (not yet made)
- Fig 6 per-seed 5-fold means on v2, both detectors: the single-seed ranking flip (not yet made; data in results/analysis/v2_three_seeds.json)
- Table 1 dataset and group statistics (class x groups, v1 vs v2) [results/eda/class_group_stats.csv]
- Table 2 audit validation by stratum (66 pairs, Wilson intervals) [results/eda/audit_validation/validation_results.md]
- Table 3 benchmark: 5 detectors x {official, clean v1 (4 detectors), clean v2}, with inflation ratio and paired differences vs YOLOv8s [notebooks 10-15]
- Table 4 RT-DETR failure table [results/analysis/rtdetr_divergence.md]

## Still missing before a draft can be written
1. [done 2026-11-23] 3-seed v2 baselines and the measured noise floor (results/analysis/v2_three_seeds.*, noise_floor_v2.*).
2. [in progress 2026-11-24] YOLOv8m, RT-DETR-L and RF-DETR-M on v2 (notebooks 13-15), RF-DETR and YOLOv8m official seeds.
3. Human spot-check of about 25 AI review labels (user, 30 min with a contact sheet), or an explicit limitation.
4. [done 2026-11-24] C1 failed the gate. Open decision: C3 (RF-DETR row on v2) or stop the method track.
5. Novelty re-check in WoS/Scopus; JCR quartiles; venue choice (EMBC 2027 full paper listed for 24 Jan 2027 on third-party pages: confirm on embc.embs.org).
6. Housekeeping: 4 label-inconsistent groups and 1 unlabeled group, MRI count 86 tagged vs 131 on the dataset card (unexplained).
7. Statement about HBFMID licence/attribution and release of split files and scripts.
