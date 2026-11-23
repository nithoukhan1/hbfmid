# Evidence for the leakage claims (2026-11-23)

Question: what proves that the official split leaks and that v2 is (much) cleaner? Four independent lines; none alone proves 'zero leakage'.

1. Direct matching (official split): SIFT+RANSAC finds a verified train copy for 48/64 test and 96/128 valid images; the weakest 12 test links were all the same image on visual review (12/12, AI-assisted, human spot-check pending); strong controls 4/4 (docs/RESULTS_AUDIT_VALIDATION.md).
2. Behavioural test (the model gives it away): box recall rises with the test image's similarity to the train set (results/analysis/similarity_recall.md, seed 42, conf 0.25, IoU 0.5).
| protocol (YOLO11s / YOLOv8s) | max NCC to train <0.80 | 0.80-0.90 | 0.90-0.95 | >=0.95 |
|---|---|---|---|---|
| official (64 test images) | 0.67 / 0.80 (n=15 img) | 0.78 / 0.78 (9) | 1.00 / 1.00 (6) | 0.97 / 0.97 (34) |
| clean v1 (391 test images) | 0.42 / 0.42 (134) | 0.49 / 0.42 (116) | 0.78 / 0.78 (33) | 0.96 / 0.97 (108) |
| clean v2 (about 286 test images) | 0.38 / 0.43 (133) | 0.45 / 0.47 (128) | 0.57 / 0.60 (24) | none (removed by construction) |
Official: 34/64 test images have a near-identical train image by thumbnail alone (48/64 by SIFT). v1: 108/391 have one (the gradient to 0.96 is the leak). v2: the >=0.95 bin is empty and recall at the top remaining bin is much lower, but the 0.90-0.95 bin (24 images, 30 boxes, wide interval) is still above the lowest bin: a residual tail, or simply typical-pose images being easier; not separable with this data.
3. Score attribution: de-duplicated official split 0.166 vs 0.490 leaky (E1b); official -> v1 -> v2 chain 0.494 -> 0.215 -> 0.159 (YOLOv8s), 0.485 -> 0.226 -> 0.143 (YOLO11s).
4. Visual validation of the grouping rules: band 0.90-0.95 4/18 real duplicates; below-threshold SIFT pairs 9/16 real; most similar unlinked thumbnails 8/8 real (AI-assisted).

Claim to make in the paper: the official split contains extensive near-duplicate leakage (validated); v2 removes the detectable duplicates (thumbnail >=0.95 plus SIFT) and shows no similarity-recall gradient above 0.95; a residual tail below the threshold cannot be excluded, so v2 numbers are upper bounds. Never write 'leak-free'; write 'leakage-reduced' or 'duplicate-aware'.
Possible further evidence: human spot-check of AI labels; grouping v3 (marker-masked SIFT in the 0.90-0.95 band) and a repeat of the similarity-recall curve; planted-duplicate control (insert known copies into test and check the recall jump).
