# Literature evidence for idea selection (first pass, 2026-11-16)
Verify every item before citing in the manuscript.

## Leakage / evaluation rigor
- Tampu et al., arXiv 2202.12267: improper splits inflated OCT classification accuracy by several to ~30 points.
- D-LeDe (Babu et al., VISAPP 2025; SN Comput Sci 2026 7:704): leakage detection for object-detection benchmarks (automotive: KITTI). No fracture-domain equivalent found.
- Leakage-aware oversampling in YOLOv8 hand fracture detection (JAIC): single-seed, no statistical testing.

## HBFMID results (all on the official split; none audits leakage in my searches)
- YOLOv11 mAP50 99.49% (jurnal.itscience.org CNAPC 6051); YOLO12n 94.09% mAP50 (GitHub project); LiteFracNet 93.53% mAP50, 2.41M params (Appl Sci 2026, 16(15):7484)

## Method-side evidence
- Crack-pretrained YOLOv9 beats COCO pretraining on small fracture data (IEEE 10838409).
- GRAZPEDWRI-DX pretraining improves a small wrist set (Bone Joint Orthop Proc 2024, 10.1302/1358-992X.2024.18.057).
- Directional/elongated-structure modules already in fracture YOLO: WCAY (DSConv, Sci Rep 2024), FDEA-Net (rotation-aware) -> low novelty on its own.
- Attribute-factorized fracture classification exists for ribs (arXiv 2504.10889) and AO classes (SCFAST-YOLO) but not on HBFMID.

## Added 2026-11-16 (idea v2 search)
- HBFMID architecture papers: YOLOv11 vs v5/v8/v10 comparison; YOLO12n; LiteFracNet (Appl. Sci. 2026, 93.53 mAP50, 2.41M params); graph-augmented multimodal framework; 'real-time YOLO models' comparison. None audits leakage (first-pass).
- Long-tail detection: EQL (CVPR20), EFL (CVPR22, one-stage), Equalization Losses (TPAMI), Seesaw, CB loss (CVPR19).
- Hierarchy: YOLO9000; hYOLO arXiv 2510.23278; Forest R-CNN arXiv 2008.05676; Deep-RTC arXiv 2007.09898; HCA CVPR24.
- Fracture hierarchy: Tanzi 2020 EJR; arXiv 2504.10889 (hyperbolic, rib CT); arXiv 2512.14196 (morphology, pediatric wrist).
- Ultralytics supports subclassing v8DetectionLoss / DetectionModel.init_criterion (official custom-trainer guide).
- Searches were web-only; re-run in WoS/Scopus before submission.
