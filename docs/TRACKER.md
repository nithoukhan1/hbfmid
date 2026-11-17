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
