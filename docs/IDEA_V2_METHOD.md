# IDEA v2 - method + benchmark (supersedes evaluation-only plan; keeps it as safety net)
Date: 2026-11-16. Status: PROPOSED, pending Gate B. Separate from Projects 1/2 (no shared code; YOLOv8/11 base, NOT YOLO26).

## Why change
- Architecture-only papers on HBFMID are already numerous (YOLOv11/YOLO12 comparisons, LiteFracNet, graph-augmented, etc.) -> low novelty, high scooping risk.
- Loss-only (focal/CB/LDAM) is crowded in detection and overlaps Project 2 (LDAM-DRW, tau-norm on GRAZPEDWRI-DX).
- Evaluation-only is new for HBFMID but not algorithmic.

## Data evidence (results/eda/class_group_stats.csv)
- Independent groups: TD 163, OD 102, Greenstick 43, Comminuted 36, Transverse 31, Spiral 22, Healthy 12, Oblique 10, Linear 5, Segmental 4.
- Images per group (duplication) is HIGHEST in rare classes: Oblique 5.7, Healthy 5.3, Segmental 5.25, Comminuted 5.0 vs Greenstick 2.1 -> raw counts overstate rare-class information.
- Official test has 0 groups of Linear/Segmental -> per-class claims need CV.
- Healthy median box = 55% of image; fracture boxes median 1.5-4.4% -> two scale regimes in one detector.
- 113/1536 images contain >1 class (OD+TD in 51 images, T+TD in 17): displacement is a per-instance attribute.

## Proposed contributions
C1 FracTree head (architecture): 10 leaf logits = sum of node logits along a clinically-motivated tree
   root -> {Healthy, Fracture}; Fracture -> {Simple, Multifragmentary, Incomplete};
   Simple -> {Transverse, Oblique, Spiral, Linear}; Multifragmentary -> {Comminuted, Segmental}; Incomplete -> {Greenstick};
   shared 'displaced' node adds to Transverse and Oblique -> TD, OD.
   Output stays nc=10 so Ultralytics loss/NMS/metrics are unchanged. ~17 node logits (10 leaf residuals + 7 shared).
   Rare leaves (Oblique 10, Segmental 4, Linear 5 groups) share nodes with abundant relatives.
   Tree is AO/OTA-INSPIRED (simple/wedge/complex; spiral/oblique/transverse) - not AO-compliant; say so in the paper.
C2 Audit-calibrated imbalance handling (loss/sampling): effective number from MEASURED independent groups
   (n_eff = (1-b^g)/(1-b), g from our SIFT audit, computed on TRAIN folds only) + group-aware sampler
   (draw group, then a member) + optional EFL-style per-class focusing. Compare vs raw-count CB, EFL, LDAM-DRW (copy of P2 code as baseline only).
C3 HBFMID-Clean protocol (kept): group-stratified rotating 5-fold, one test rep per group, 3 seeds, bootstrap CI over groups,
   10-class / merged / fracture-vs-healthy reporting.

## Prior art to cite and differentiate
YOLO9000 WordTree; hYOLO 2025 (grocery, no imbalance, no medical; flat model had higher finest-level F1);
Forest R-CNN (parent classes for LVIS tail, two-stage); Deep-RTC; HCA (CVPR 2024, regression);
Tanzi 2020 (hierarchical AO femur, classification); hyperbolic rib fracture (CT classification);
arXiv 2512.14196 (pediatric morphology classification; 6/11 classes too few samples);
CB loss (Cui 2019), EQL/EFL/Seesaw (long-tail detection); Truong 2023 (medical near-duplicate detection).

## Risks (be honest)
R1 hierarchy may give no gain (hYOLO: flat won at finest level). R2 raw-count vs group-count weights differ little
(imbalance ratio ~34:1 vs ~41:1) -> C2 may be a null result; sampler may matter more. R3 4-5 groups cannot support per-class claims.
R4 within-image TD+OD / T+TD co-occurrence may reflect label noise -> inspect before trusting 'displaced' attribute.
R5 Ultralytics version drift -> pin version.

## Gates
Gate A: E1 (official) vs E2 (clean folds) shows drop (audit already predicts it).
Gate A2 (free, from baseline confusion matrix): >=20% of errors are within-parent sibling confusions (T<->TD, O<->OD, Comm<->Seg) else tree has little to fix.
Gate B (pilot, 5 folds x 1 seed, YOLOv8s): FracTree vs flat; continue if macro mAP50-95 gain >=1.0 pt on classes with >=10 groups, positive on >=3/5 folds.
Fail Gate B -> publish as benchmark + analysis (negative-result framing), Plan B journals.

## Experiments (revised)
E1 official split reproduce | E2 clean folds | E3 detector family | E4 loss arms (BCE, w-BCE, CB-raw, CB-group, EFL, LDAM-DRW, ours)
E5 head arms (flat; flat+merged-label; tree w/o displaced; tree full; +aux node loss) | E6 sampler arms (instance, class-aware, group-aware)
E7 combined best | E8 granularity reporting | E9 init (scratch/COCO/GRAZPEDWRI-DX/FracAtlas) | E10 stats | E11 EigenCAM | E12 optional FracAtlas zero-shot
Ablations on YOLOv8s 1 seed; final arms 3 seeds. Estimate 30-45 GPU-h total (check Kaggle quota).
