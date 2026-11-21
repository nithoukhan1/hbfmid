# Audit validation plan (written 2026-11-21, BEFORE any manual label exists)

Goal: show that the leakage audit (SIFT + RANSAC, inlier threshold 25; 48/64 test and 96/128 valid images have a train copy; 388 groups) is right, and in which direction it can be wrong.
Tool: `scripts/eda/audit_validation.py build` -> `results/eda/audit_validation/review.html` (blinded: no file names, splits, inlier counts; A/B order randomised; seed 20261121) and `review_key.csv`. Score with `... score --review review_results.csv`.

## A. Automatic checks (done, no human; results/eda/audit_validation/auto_checks.json)
* Known-positive recall: 113 clean sets of "1 original x 3 Roboflow augmentations" (same file-name number and label set, train): 109 are grouped completely (96.5%); 325 of 339 pairs (95.9%) have a direct verified pair >= 25. The failures are heavy augmentations (low inliers), so errors make the audit MISS duplicates, not invent them. Limit: this tests augmentation copies only, not duplicates across original files.
* Test images: 48 are in a group with a train image; 47 have a DIRECT verified pair >= 25 to a train image (weakest direct links: 25, 33, 33, 35, 49, 50, ...); 1 reaches train through a chain (test-test pair 232 inliers, then test-train pair 289 inliers).

## B. Manual review (66 pairs, one reviewer = the author; limitation: single reviewer, not blinded to the project)
| stratum | n | question |
|---|---|---|
| T_weakest | 12 | are the weakest direct test-train links real duplicates? |
| T_random | 8 | same, random test images |
| C_chain | 2 | is the chain link real? |
| B_border | 16 | precision of pairs just above the threshold (25-39 inliers, different file-name numbers) |
| R_below | 16 | missed duplicates just below the threshold (12-24 inliers) |
| N_hard | 8 | most similar thumbnails the audit did NOT link |
| P_control | 4 | obvious positives (reviewer/page check) |
Labels: same_image / same_patient / different / unsure.

## C. Decision rules (fixed now)
1. P_control must be 4/4 same_image, otherwise the review is invalid (page or reviewer problem).
2. Headline "48/64 test images have a train copy" is VALIDATED if >= 11 of the 12 T_weakest pairs are same_image (and C_chain 2/2). If 8-10: report the claim as a lower bound computed from the confirmed share. If < 8: raise the threshold and recompute everything.
3. Threshold 25 is acceptable if B_border same_image >= 14/16. If lower: use the higher threshold of the sensitivity table (results/eda/leakage_audit_summary.json) and say so.
4. If R_below same_image >= 4/16 (25%): the group count 388 is an overestimate and leakage is a lower bound; report both.
5. If N_hard same_image >= 4/8: the matcher misses rotated/cropped duplicates; report leakage as a lower bound. same_patient labels are reported separately (patient-level leakage the audit does not claim to detect).
6. Unsure counts as not confirmed (rules 2, 3) and as not missed (rules 4, 5): both are shown.
Intervals: Wilson 95%; n is small, so the review supports a statement of direction, not a precise rate.

## D. Reviewer change (2026-11-21, before any label exists)
The author found the manual review confusing and asked Claude (AI) to do it. While testing the page, Claude had printed some rows of the first key (it knew which pair IDs were hard negatives and chain pairs), so the first sample was discarded unlabeled and rebuilt with seed 20261122 (same strata and rules). Claude labels the new pairs from the images alone and does not read review_key.csv until all labels are written. Reviewer = AI (Claude), visual identity judgement only (no clinical reading). Limitation: an AI reviewer is not independent of the AI that wrote the tool; the author spot-checks a subset of Claude's labels (the unsure/different calls on the test-image strata plus 6 random pairs). Report this as "AI-assisted visual review with human spot-check", never as a radiologist review.
