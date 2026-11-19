# Method-track literature check and decision record (2026-11-19)

Context: user asked to reconsider the project, use the authors' official split, and look for a modality-guided method (X-ray + MRI) at Q2/Q3. Web search only (not WoS/Scopus); verify before submission.

## 1. Policy on the official split
Report the official split for comparability with prior HBFMID papers AND HBFMID-Clean, and state the leakage in one paragraph. Reason: on the official test set the seed range of one detector is 3.4 pts (YOLO11s), so a method gain below that is untestable there, and 75% of its images (48/64) have a train copy. A method that wins on both protocols is more convincing; a method that wins only on the leaky split is not.

## 2. Prior art found
| Topic | Finding |
|---|---|
| HBFMID papers | Base paper (YOLOv8, Applied Intelligence 2024); YOLOv11 vs older YOLO comparison; LiteFracNet (Applied Sciences 2026, mAP50 93.53% on official split); FDEA-Net (Mathematics 14(3):567, 2026, uses HBFMID as external dataset). None found that conditions on modality or audits leakage. |
| Mixed-modality medical detection | Align Your Query (KAIST, arXiv 2510.02789, Oct 2025): DETR-style detectors (DINO, Deformable DETR), modality tokens (MoCA) + QueryREPA; 7 public sets, about 20.8k images, 27 classes; DINO AP 37.7 -> 41.1; most of the gain from MoCA; QueryREPA sometimes slightly lowers AP. Not fracture, not YOLO. |
| Unpaired multimodal learning | Shared convolutions + modality-specific normalisation (Dou et al., TMI 2020, CT/MRI segmentation); MulModSeg; FedNorm. Segmentation, not detection. |
| Foundation backbones | RF-DETR (DINOv2 backbone) strong on RF100-VL small datasets; DINOv2 -> lightweight student distillation exists for classification/segmentation (MedAlmighty, Polyp-DiFoM). |
| Cross-modal augmentation | Discriminative cross-modal augmentation via unpaired translation (Yang and Xie, 2020). |
| Shortcut mitigation | Feature disentanglement benchmarks in medical imaging (Mueller and Berens 2026; MIMM-X). |

## 3. What our own data says about modality
MRI = 202 images / 56 groups (13%), unpaired with X-ray (different patients), class-confounded (Comminuted 24 MRI vs 11 X-ray groups; most other classes X-ray only). Miss rate at conf 0.25: MRI 0.46-0.48 vs X-ray 0.37. A modality rule alone flags 69% of Comminuted test images. So "multimodal" here means mixed-modality training, not fusion, and a modality-conditioned model can exploit modality as a class prior.

## 4. Candidate methods
| id | idea | novelty verdict | main risk |
|---|---|---|---|
| M1 | Modality-conditioned YOLO: modality-specific normalisation / FiLM, modality-balanced sampling; modality predicted from the image, not the file name | application novelty only (technique exists in segmentation and DETR detection) | only 13% MRI; effect small; shortcut |
| M2 | Foundation-model-guided detector (RF-DETR DINOv2 backbone, or distil DINOv2/MedSAM features into YOLO) | RF-DETR exists; distillation into YOLO overlaps the KD work of the separate pediatric-wrist project (keep separate) | overlap; low novelty |
| M3 | Modality-conditioned DINOv2-DETR (M1 on an RF-DETR backbone) | combination; closest prior art is Align Your Query | compute, implementation effort |
| M4 | Cross-modal translation augmentation to break the class-modality confound | more novel | GAN on about 200 MRI images; lesion fidelity |

## 5. Decision record
Recommended order: (a) RT-DETR and RF-DETR rows under both protocols (also the Q3 requirement), (b) if a foundation-backbone detector gains at least 0.9 pt mAP50-95 on 3 seeds x 5 folds, build M3 on it; else test M1 on YOLO. Pre-registered gate for any method: paired mean >= +0.9 pt mAP50-95, >= 4/5 folds positive, on HBFMID-Clean, and not worse on the official split. My (unmeasured) estimate that a modality-conditioning method clears this gate: well under 50%.

## 6. Conference track (parallel)
Core: audit validation (manual pair review) -> figures and tables -> 4-page draft -> number check against JSON -> code/split release -> submit. Optional: never-found box inspection, RT-DETR row. ISBI 2027 deadline (26 Oct 2026) has passed. EMBC 2027 (Singapore, 11-15 July 2027) full-paper deadline listed as 24 Jan 2027 on third-party pages; confirm on embc.embs.org. Journal version must add substantial new content (method and extended experiments).
