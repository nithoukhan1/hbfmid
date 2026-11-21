# Audit validation: results (2026-11-21)

Reviewer: Claude (AI-assisted visual identity review, 66 pairs, seed 20261122, labels written before the key was read). Human spot-check still pending. Plan and rules: `docs/AUDIT_VALIDATION_PLAN.md`. Raw: `results/eda/audit_validation/{review_results_claude.csv, review_key.csv, validation_results.md}`; extra scan: `results/eda/thumbnail_dup_scan.json`.

## Per-stratum result (label = same_image unless stated; Wilson 95%)
| stratum | n | same_image | other | share same_image |
|---|---|---|---|---|
| T_weakest (test-train links, weakest 12) | 12 | 12 | 0 | 1.00 (0.76-1.00) |
| T_random | 8 | 8 | 0 | 1.00 (0.68-1.00) |
| C_chain | 2 | 2 | 0 | 1.00 (0.34-1.00) |
| P_control (>=100 inliers) | 4 | 4 | 0 | 1.00 (0.51-1.00) |
| B_border (25-39 inliers) | 16 | 13 | 2 different, 1 unsure | 0.81 (0.57-0.93) |
| R_below (12-24, NOT linked) | 16 | 9 | 1 same_patient, 5 different, 1 unsure | 0.56 (0.33-0.77) |
| N_hard (thumbnail NCC 0.96-0.99, NOT linked) | 8 | 8 | 0 | 1.00 (0.68-1.00) |

## Pre-registered rules
1. P_control 4/4: PASS (review valid).
2. T_weakest 12/12 and C_chain 2/2: PASS. The headline "48/64 test images have a verified train copy" is supported.
3. B_border same_image >=14/16: FAIL (13/16). The 3 non-confirmed are really 2 distinct image pairs: images 67 and 58 (listed twice, P014 and P018, 25-27 inliers, matched through the shared "BM" sticker) and P040 (29 inliers, shared "*"/R marker). All three are train-train; none involves a test image. Action per the plan: raise the threshold or mask overlay markers before matching. Reporting: headline range 44/64 (threshold 40) to 48/64 (threshold 25).
4. R_below same_image >=4/16: TRIGGERED (9/16, 8 of them in different groups). Pairs below the threshold are often real duplicates, so the 388-group count is an overestimate and leakage is a LOWER bound.
5. N_hard same_image >=4/8: TRIGGERED (8/8, all in different groups; 3 cross a split: train-valid P023, train-test P058 and P062 with the R_below band). The matcher misses duplicates (rotated, cropped, low-feature).

## Extra scan: unlinked near-duplicates over the whole dataset (thumbnail NCC, 4 flip modes, different SIFT groups)
| NCC threshold | unlinked pairs | test images with an unlinked train partner | valid images with one |
|---|---|---|---|
| >=0.90 | 574 | 9 / 64 | 16 / 128 |
| >=0.95 | 206 | 4 / 64 | 7 / 128 |
| >=0.97 | 129 | 4 / 64 | 4 / 128 |
(These counts can overlap with the 48 test images already linked; not de-duplicated.)

## Consequences
- The official-split leakage claim is strengthened (lower bound), not weakened.
- HBFMID-Clean (388 groups) is NOT fully clean: duplicates remain across folds, so the clean mAP values (0.217 / 0.2225 / 0.227 / RT-DETR 0.18-0.21) are probably still optimistic and the 2.1-2.7x inflation is understated. A grouping v2 is required: dihedral-invariant thumbnail/embedding links plus marker-masked SIFT, then recompute the folds and check which images move.
- Limits: AI reviewer (not independent of the AI that built the tool); about 15 labels on heavily augmented MRI/X-ray copies are moderate-confidence (P004, P009, P015, P017, P022, P026, P028, P042, P049, P050, P051, P057, P062, P065); human spot-check recommended on these, the 7 'different' and the 2 'unsure' pairs. Report as "AI-assisted visual review", never as radiologist review.
