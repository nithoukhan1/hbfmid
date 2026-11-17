# Decision after E1/E2 (2026-11-17): the plan changes

## What the data said
- Gate A PASS, strongly: official protocol 0.490 mAP50-95 / 0.940 mAP50 vs clean 0.208 / 0.43. Attribution to leakage is NOT yet isolated (see control E1b).
- Gate A2 FAIL: only 5.7% of errors are sibling-class confusions; 31-41% of ground-truth boxes are never localized (any class, IoU 0.5); when localized the class is right 88-92% of the time. The errors are a detection/recall problem in a small-data regime (about 230 independent radiographs per training fold), not a classification-hierarchy problem.
- Seed noise 2.1 pts (single fold), partly from validation-based early stopping on 78 images (best epoch 26 / 91 / 39 across seeds).

## Consequences
1. FracTree (tree head, arms 4, 5, 8) is dropped as the core contribution: its ceiling is the about 8% of boxes that are localized with a wrong class. The sibling errors that exist are mostly displaced vs non-displaced pairs; keep as an analysis, not a method.
2. Paper A (leakage-free benchmark + audit + error analysis) becomes the main paper, strengthened by: multiple detectors, the error decomposition, modality analysis (MRI 13% of images; Comminuted 69% MRI), label-noise groups, and the dual (official + clean) reporting.
3. The improvement study is about DETECTION RECALL with external fracture data: FracAtlas transfer (Gate C). Arms: 1 baseline; 2 CB raw counts; 3 CB group counts; 6 FracAtlas pretrain then fine-tune; 7' joint training with a generic "fracture" objective (FracAtlas boxes supervise only that objective). FracAtlas boxes are 2.4x smaller than HBFMID boxes (37 vs 88 px at 640): add a scale-matched zoom arm.
4. Protocol change for all further arms: fixed 100 epochs, no early stopping, primary metric from last.pt (EMA); 3 seeds on the final comparisons. Detectable effect: about 2.7 pts (1 seed x 5 folds) or about 1.5 pts (3 seeds x 5 folds). The +1.0 pt Gate B rule is withdrawn.

## Revised gates
- A (done, PASS) with control E1b to attribute the gap.
- C: FracAtlas arm improves macro mAP50-95 (>= 8 classes with >= 10 groups) by at least the detectable effect, positive on >= 4/5 folds, and improves localization recall (fraction of GT boxes with any-class IoU>=0.5 prediction).
- Imbalance arms 2/3: reported as secondary; rare classes Linear (5 groups) and Segmental (4 groups) are not learnable and are exploratory.
- Outcome table: C passes -> paper = benchmark + FracAtlas transfer (Q2-Q3 attempt); C fails -> paper = benchmark/audit + analysis (Q3-Q4); in both cases the audit is the anchor.

## Next experiments (Kaggle)
1. Notebook 02: E1b (official de-duplicated: train 806 / val 109 / test 64 images) + noise floor under the fixed protocol (folds 0, 1 x seeds 42/43/44). 7 runs, about 1.5 h wall.
2. Then the pilot arms with the fixed protocol; about 65 min wall per arm for 5 folds x 1 seed (estimate from measured run times).
