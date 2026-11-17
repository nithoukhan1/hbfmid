# Improvement arms: design, settings fixed in advance, decision rules (2026-11-17)

All arms use the fixed-epoch protocol (YOLOv8s, 100 epochs, imgsz 640, batch 16, last.pt) on HBFMID-Clean (5 group folds). Arm 1 (plain baseline) is the reference.
Settings below are fixed **before** seeing any arm result; none is tuned on test folds.

| Arm | What changes | Code |
|---|---|---|
| 2 `cbraw` | cls-loss class weights w_c = (1/n_c)^0.5, mean-normalised, n_c = boxes per class in the training list | `class_weights.py`, `run_baseline.py --cls-weight raw` |
| 3 `cbgroup` | same formula, n_c = independent groups containing class c in the training fold | `--cls-weight group` |
| 6 `fapre_<variant>` | pre-train on FracAtlas (1 class, 50 epochs, seed 42), fine-tune on the fold | `pretrain_fracatlas.py`, `--init-weights` |
| 7' `fajoint_<variant>` | HBFMID train + all FracAtlas fractured images labelled as an extra 11th class ("generic fracture"); val/test HBFMID only | `--extra-train --extra-names` |
| variant `generic` / `zoom` | original FracAtlas images / scale-matched crops (anchor box ~88 px at imgsz 640, zoom <= 4, 2 crops per image) | `prepare_fracatlas_yolo.py` |

## Facts that shape the interpretation
- **Ultralytics 8.4.215 already has `cls_pw`** (inverse-frequency class weights, raw counts). Arm 2 reproduces that formula; the only thing we add is counting independent groups (arm 3). The class-weight arms are therefore a controlled comparison, not a new method.
- Group-based and raw weights differ by at most ~33% per class (fold-wise, p = 0.5): groups/images is about 0.17-0.31 for nine of ten classes and 0.43 for Greenstick. Arms 2 and 3 will probably not be separable with 5 folds; they are run as a pair because it costs ~1 h.
- FracAtlas boxes are 2.4x smaller than HBFMID boxes (median 36.6 vs 88.3 px at 640) and FracAtlas has one class: transfer can help localisation (recall) but cannot teach fracture types.
- Zoom crops upsample low-resolution sources (many FracAtlas images are 373x454), and crops drop boxes that are less than half inside (logged in `fa_prep_summary_*.json`).

## Pilot protocol and decisions
- Pilot = seed 42, 5 folds per arm (the detectable effect with one seed is about 1.6 points, from the controls). The pilot screens; it does not support claims.
- An arm advances to 3 seeds (Gate C) when its paired mean difference in mAP50-95 is >= +1.0 point AND positive in >= 4/5 folds AND localisation recall does not fall. Arms that do not advance are reported as negative results.
- Reported per comparison (`compare_arms.py`): mean paired difference, sd, t-interval (4 df, rough), folds positive, for mAP50-95, macro mAP over the 8 eligible classes, mAP50 and localisation recall (class-agnostic, IoU >= 0.5, conf >= 0.25; the generic head counts as a localiser in arm 7').
- Known limits stated up front: pre-training variance is not sampled (one FracAtlas checkpoint); the extra class in arm 7' lets the model hedge between generic and specific classes, which can only affect the specific-class scores through shared features.

## Cost (T4 x2, estimates from the control runs)
Notebook 04 (arms 2, 3): 10 runs, ~2 h. Notebook 05 `generic`: 1 pre-training + 10 runs, ~3-4 h. `zoom` repeats notebook 05 with `VARIANT = "zoom"`.
