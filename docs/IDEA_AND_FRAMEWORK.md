# Idea, framework, implementation and experiment plan (2026-11-16)

## Working title
A leakage-aware benchmark and transfer recipe for fracture-type detection on HBFMID

## Why (evidence)
- Strict audit: ~75% of official test images (48/64) and valid images (96/128) have a verified copy in train; ~388 independent groups vs 1,539 images (docs/TRACKER.md, results/eda/).
- Published HBFMID scores (YOLOv8 92%, YOLOv11 99.5%, YOLO12n 94%, LiteFracNet 93.5% mAP50) all use the official split.
- Precedent: leakage-inflation papers are publishable (Tampu et al., Sci Data 2022; pre-split augmentation leakage in YOLO detection, up to 66 mAP50 points).

## Contributions (target)
1. Reproducible audit + released group IDs / split lists (no images) for HBFMID.
2. HBFMID-Clean protocol: group-stratified, rotating train/val/test folds, multi-seed, bootstrap CIs over groups, 3 reporting granularities (10-class, merged morphology, fracture vs healthy).
3. Re-benchmark of detector families: official split vs clean folds -> quantified inflation.
4. Transfer recipe study under the clean protocol (scratch vs COCO vs fracture-domain pretraining).
5. (Optional) external check on FracAtlas, shortcut probe for burned-in markers, same audit on other public fracture sets.

## Modules
M1 audit (DONE) -> M2 clean splits -> M3 benchmark + M4 transfer (parallel) -> M5 validity checks (optional)

## Implementation (files)
- scripts/eda/leakage_audit.py                 DONE
- scripts/eda/make_clean_splits.py             NEXT: groups -> 5 rotating folds (test=k, val=k+1, train=rest); writes data_splits/fold{k}_{train,val,test}.txt
- scripts/training/train_detector.py           args: --model --fold --seed --init --epochs; Ultralytics reads txt image lists (labels found via /images/ -> /labels/)
- scripts/evaluation/aggregate_results.py      out-of-fold per-class metrics, bootstrap CIs over groups, paired tests
- notebooks/kaggle/*.ipynb                     thin wrappers: clone repo, run scripts, Save Version
- results/*                                    small CSV/JSON/figures only

## Experiments
E1 reproduce paper on official split (YOLOv8m)            -> sanity check
E2 same detector on clean folds                           -> inflation = E1 - E2   (GATE 1)
E3 detector families on clean folds (YOLOv8/11, RT-DETR, Faster R-CNN, 1 lightweight)
E4 initialisation ablation: scratch / COCO / GRAZPEDWRI-DX / FracAtlas (/ crack)
E5 duplicates & pre-augmented copies in training: keep vs dedupe
E6 granularity: 10-class vs merged morphology vs binary
E7 (optional) zero-shot fracture-vs-healthy on FracAtlas
E8 (optional) marker/shortcut probe + EigenCAM

Statistics: 5 folds x 3 seeds; bootstrap over test groups; Wilcoxon / paired bootstrap; Friedman+Nemenyi across models.
Gate 1: if clean-vs-official drop is small, re-evaluate story before spending GPU on E3-E8.

## Risks
- Reviewers may see "benchmark only": mitigate with E4 (method-side finding), released splits, multi-dataset audit.
- Single-dataset scope; rare classes (Linear 5 groups, Segmental 4) cannot support per-class claims -> merge/report pooled.
- Possible scooping: re-run a literature check (WoS/Scopus) before submission.
- Group chaining at low thresholds (one group of 30): review large components manually.
- Licence: Roboflow export README says "Private"; Mendeley says CC BY 4.0 -> release IDs/lists only.
