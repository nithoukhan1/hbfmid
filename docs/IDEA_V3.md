# IDEA V3 - plan agreed 2026-11-17 (A + B with gates)

Status: APPROVED by user to proceed with Phase 1 (clean splits + E1/E2). The paper choice (A or B) is decided by gates, not in advance.

## 1. What we are building
- **A. HBFMID-Clean benchmark and audit.** Leakage-audited, group-aware protocol; re-benchmark detectors on the official and clean splits; report the gap and whether detector rankings change. Safe paper, about Q3-Q4 on its own.
- **B. FracTree-PL.** Hierarchy-aware detection head for fracture morphology, trained with partial (coarse) labels from FracAtlas and audit-calibrated class balancing, evaluated on the clean protocol. Target Q2-Q3 if gates pass.
- Extra rows only: DEIMv2 (DINOv3 backbone) as one more detector; tree-structured conformal sets only if time allows.
- Dropped: open-vocabulary text heads, SAM auxiliary masks, architecture-only YOLO modules.

Honest novelty statement: hierarchy-aware heads with mixed-granularity labels exist in remote sensing (LEVIRDet), general detection (11K-class detection from coarse boxes + fine image labels) and classification (free-grained recognition, CVPR 2026); partial-label multi-dataset detection exists for CT lesions (LENS, MELD). We found no fracture-morphology detector that combines a clinical tree head, coarse-label cross-dataset training, group-count-calibrated imbalance handling and a leakage-audited protocol. The claim is therefore an applied, combined contribution, not a new generic algorithm. Re-run the novelty check in Web of Science / Scopus before submission.

## 2. HBFMID-Clean protocol (implemented: scripts/eda/make_clean_splits.py)
- Unit = independent group (388 groups from 1,539 images; SIFT+RANSAC audit, threshold 25 = conservative).
- Rotating 5-fold at group level: fold k test, fold (k+1) mod 5 validation, rest train. Greedy multi-label stratification (rarest class first), seed 42, 2,000 restarts; objective 0.1609.
- Train: every member image of its groups (originals + Roboflow augmentations stay together).
- Val/test: ONE representative per (group, label-set); remaining members are `excluded` for that fold and never used for training. Representative preference: official test > official valid > train (un-augmented first), then file name.
- 245 groups exist only as augmented train copies, so their representative is an augmented copy (flip/small rotation/brightness); 143 groups have an un-augmented official valid/test member. Document this in the paper.
- 4 groups have inconsistent class sets across members (e.g. Comminuted vs Greenstick): kept together (conservative for leakage), one representative per label-set; flagged as possible label noise. 1 group (3 images) has no boxes.
- Pooled out-of-fold test = 392 images, every group exactly once. Per-fold groups 77-78; each class has >= 2 train groups in every fold (Segmental 2-3, Linear 3).
- Linear (5 groups) and Segmental (4 groups) cannot support per-class claims: report as exploratory. Gate B metric uses the 8 classes with >= 10 groups (Comminuted, Greenstick, Healthy, Oblique Displaced, Oblique, Spiral, Transverse Displaced, Transverse).
- Verified by scripts/eda/verify_splits.py (independent check): group sets disjoint in every fold, no eval-group member in train, each group tested exactly once.

## 3. Phases, arms and gates
Phase 1 (no regret): E1 = YOLOv8s on official split (+ leak probe: official test 48 images with a train copy vs 16 without); E2 = YOLOv8s on 5 clean folds; noise floor = 2-3 extra seeds on fold 0.

Pilot arms (YOLOv8s, 1 seed, 5 clean folds, identical hyper-parameters):
1 flat baseline | 2 + class-balanced loss (raw counts) | 3 + class-balanced loss (group counts) | 4 FracTree head | 5 FracTree + group-balanced loss | 6 flat, pretrained on FracAtlas then fine-tuned | 7 FracTree-PL (joint coarse labels) | 8 = 7 + group-balanced loss.

Stages: 2a arms 1-3; 2b arms 4-5 (Gate B); 2c arms 6-8 (Gate C).

Gates:
- A: E1 official > E2 clean mean (Phase 1). Leak probe gap reported.
- A2: sibling confusions (same family, i != j) / all errors (wrong-class + missed + false positives) >= 0.20 at conf 0.25, IoU 0.5, pooled E2.
- B: >= +1.0 pt macro mAP50-95 on classes with >= 10 groups vs arm 1, positive on >= 3/5 folds. Threshold is only meaningful if the measured seed sd is well below 1.0 pt (noise-floor check).
- C: joint FracAtlas training improves fracture-vs-background AP on clean folds, and arm 7 beats arm 6 (plain pretraining) and arm 4.

Outcome table: A fails -> B stands alone; A passes + A2 fails -> imbalance + A; B and C pass -> full FracTree-PL paper; B passes + C fails -> tree + group loss without FracAtlas; B fails + arm 3 passes -> imbalance paper (Q3-Q4); all fail -> paper A.

## 4. Detector choice
- Pilot on YOLOv8s (simple blocks, same family as the base paper's YOLOv8m, keeps Project 3 distinct from the YOLO11-based project).
- Confirmation stage: best arms on YOLO11s and YOLOv8m (base-paper comparability), 3 seeds; DEIMv2 as one extra row.
- YOLOv8s 11.2M params / 28.6 GFLOPs / COCO mAP50-95 44.9; YOLO11s 9.4M / 21.5 / 47.0 (Ultralytics docs).
- Custom head/loss via the documented hooks: subclass v8DetectionLoss, return it from DetectionModel.init_criterion, build the model in CustomTrainer.get_model; keep custom classes in an importable module. Pin the Ultralytics version after the first Kaggle run (API changes across releases).

## 5. FracTree-PL design notes (Phase 3, not yet implemented)
- Tree: Healthy | Fracture -> Simple {Transverse, Oblique, Spiral, Linear} / Multifragmentary {Comminuted, Segmental} / Incomplete {Greenstick}; shared "displaced" node for Transverse Displaced and Oblique Displaced. 17 node logits, fixed 10x17 0/1 matrix to 10 class logits; sigmoid BCE per class unchanged.
- Partial labels: FracAtlas fracture boxes supervise only the "Fracture" node (leaf losses masked). Non-fractured FracAtlas images are background for fracture classes only; they do NOT supervise HBFMID "Healthy" (whose boxes cover about half the image).
- To verify before use: FracAtlas fractured images are fully annotated (unannotated fractures would be false negatives); HBFMID MRI images (131 of 641 raw) get no benefit from X-ray-only FracAtlas, so report X-ray and MRI separately (modality tags still to be created).
- Per-epoch FracAtlas sampling: all fractured images + a sampled subset of non-fractured ones, so FracAtlas does not swamp HBFMID.
- Group-calibrated weights: effective number from measured independent groups in the TRAIN folds (data_splits/clean5/fold_class_groups.csv).

## 6. Compute (ESTIMATES, replace with measured epoch times after E1)
- About 40-50 min per YOLOv8s run (100 epochs, T4); pilot = 8 arms x 5 folds = 40 runs, about 30 GPU-hours plus extra for FracAtlas arms.
- Kaggle limits free GPU hours per week (about 30 as last known; check your account) and session length, so expect 2-3 weeks wall-clock for the pilot.
- Never train locally on the dataset folder: Ultralytics would write labels.cache into it. run_baseline.py refuses to train if the dataset folder is writable.

## 7. Open items
- Create modality (X-ray/MRI) tags; inspect the 4 label-inconsistent groups and the 1 unlabeled group.
- Inspect FracAtlas annotation completeness; write FracAtlas -> YOLO converter (coarse class).
- Decide whether figures containing dataset images (results/eda/figures) stay in the public repo (dataset licence CC BY 4.0 per listing; keep attribution).
- Confirm journal quartiles in Clarivate JCR (MDPI quartile statements conflict across sources).

## 8. Prior art / sources
LEVIRDet hierarchy-aware head arXiv 2606.25312; Detecting 11K classes arXiv 1908.05217; Free-grained hierarchical recognition arXiv 2510.14737; LENS arXiv 2009.02577; MELD arXiv 2005.13753; Zhao et al. ECCV 2020 unified label space; Zhou et al. arXiv 2102.13086; fracture morphology from AO codes arXiv 2512.14196; DEIMv2 arXiv 2509.20787; data leakage in visual datasets arXiv 2508.17416; SeqCRC arXiv 2505.24038; LiteFracNet (Applied Sciences 2026), ASC-YOLO (Appl. Sci. 2025), Fracture-YOLO arXiv 2509.23408.
