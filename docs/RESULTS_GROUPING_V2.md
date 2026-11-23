# Grouping v2 (2026-11-21): analysis only, no model re-run yet

Method (`scripts/eda/grouping_v2.py`): v1 SIFT groups (388) plus near-duplicate links from 32x32 thumbnail NCC over the 8 dihedral modes, union-find. Dataset is only read. Motivation: audit validation found v1 under-merged (docs/RESULTS_AUDIT_VALIDATION.md). Outputs: `results/eda/grouping_v2/` (groups per threshold + summary.json), candidate folds `data_splits_v2/clean5` (threshold 0.95, seed 42, same fold builder as v1). The current v1 folds in `data_splits/` are untouched.

| NCC threshold | v2 groups (v1: 388) | unlinked pairs added | of these, in different v1 test folds | v2 groups spanning several v1 folds | official test images with a NEW train link |
|---|---|---|---|---|---|
| >= 0.90 | 254 | 574 | 443 | 34 | 9 |
| >= 0.95 (chosen) | 282 | 206 | 158 | 29 | 4 |
| >= 0.97 | 307 | 129 | 102 | 27 | 4 |

Reading:
- [CORRECTED 2026-11-23, see docs/RESULTS_V2_BENCHMARK.md: the leakage in v1 folds covers 31% of test images and v2 mAP is 5.6-8.3 pts lower] Residual leakage in the v1 clean folds is real but limited: about 29 duplicate groups (of 282) straddle folds. The official split is far worse (48/64 test with a train copy at v1, 52/64 at v2; valid 102/128).
- [WRONG, corrected] Expected effect on the clean mAP values (0.217 / 0.2225 / 0.227): small upward bias; size unmeasured until the benchmark is re-run on v2 folds.
- v2 candidate folds: 282 groups, 286 test representatives (v1: 392), about 57 test images per fold, so each fold is noisier; every gate class still has >= 10 groups (Healthy 12, Oblique 10). Linear and Segmental stay exploratory.
- Threshold 0.95 is a judgement: validated on only 8 hard negatives (0.96-0.99); the 0.90-0.95 band is unvalidated. Thumbnail NCC can link unrelated images with large black backgrounds, which would over-merge (conservative for leakage, costs test size).
- Dihedral rot90 modes added no pairs over the earlier 4-flip scan.

Open: validate a sample of the 0.90-0.95 band by eye; decide whether to re-run the benchmark on v2 folds (about 20 runs for 4 detectors x 5 folds x 1 seed, roughly 30-35 T4-hours at 91-108 min per run).

## Band check 0.90 <= NCC < 0.95 (added 2026-11-21, AI-assisted visual review, 18 random pairs, seed 20261123)
`scripts/eda/band_check.py`; labels `results/eda/grouping_v2/band_check_0.90_0.95.csv`. The band holds 368 unlinked pairs.
Result: 4/18 same image (Q11, Q12, Q15, Q17), 14/18 different; Wilson 95% for the duplicate share 0.09-0.45. Most false matches are standardised radiographs of other patients in the same pose (ankle AP, forearm lateral, tibia AP).
Reading: linking the whole band (threshold 0.90, 254 groups) would over-merge about 78% of its pairs; threshold 0.95 stays the primary choice. A tail of roughly 80 real duplicate pairs (0.22 x 368, wide interval) is likely still unlinked, so residual leakage in v2 folds is not zero. A tiered rule for the band (e.g. SIFT with a lower inlier threshold on marker-masked images) would be grouping v3; not built.
