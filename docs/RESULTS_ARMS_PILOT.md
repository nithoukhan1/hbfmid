# Arms pilot results (seed 42, 5 folds) - 2026-11-18

Protocol: YOLOv8s, 100 epochs, last.pt, HBFMID-Clean folds 0-4. Differences are paired by fold against the baseline seed-42 run of the same fold
(`scripts/evaluation/compare_arms.py`, run locally on all 5 folds; the in-notebook comparisons only saw folds 0-1 because `baseline_fixed` was not on GitHub yet).
Raw files: `results/arms/<arm>_vs_baseline.{md,json}`, `results/arms_cb/`, `results/arms_fa_generic/`, `results/baseline_fixed/`.

## Reference (arm 1, 3 seeds x 5 folds)
mAP50-95 0.2170, macro(8 eligible classes) 0.2663, mAP50 0.4160. Per fold: 0.1993 / 0.2339 / 0.2090 / 0.1594 / 0.2836. Seed sd of the 5-fold mean: 0.0056.
Seed spread inside one fold is larger (fold 3: 0.175 / 0.158 / 0.146 for seeds 42 / 43 / 44).

## Pilot arms (mean paired difference over 5 folds; 95% interval is t with 4 df, rough)
| Arm | mAP50-95 | diff vs baseline s42 | folds positive | macro diff | mAP50 diff | localisation recall diff |
|---|---|---|---|---|---|---|
| 2 `cbraw` (class weights, box counts) | 0.2177 | +0.0025 +/- 0.022 | 3/5 | +0.0028 | -0.0147 | -0.004 (1/5) |
| 3 `cbgroup` (class weights, group counts) | 0.2055 | -0.0096 +/- 0.014 | 1/5 | -0.0114 | -0.0225 | -0.028 (0/5) |
| 6 `fapre_generic` (FracAtlas pre-train, fine-tune) | 0.1940 | -0.0211 +/- 0.035 | 1/5 | -0.0257 | -0.0297 | -0.003 (1/5) |
| 7' `fajoint_generic` (joint, extra generic class) | 0.2308 | **+0.0157 +/- 0.031** | **4/5** | +0.0193 | +0.0200 (5/5) | **+0.052 +/- 0.041** (4/5) |

Arm 7' per fold (diff in mAP50-95): +0.0094, +0.0567, +0.0048, -0.0109, +0.0184. Against the 3-seed baseline mean the mean difference is +0.0138 (sd 0.019).
The mean difference is about 1.4 times its standard error (t, 4 df), i.e. NOT significant on its own: this is a screen, not evidence of an effect.

## Reading against the rules fixed in advance (docs/ARMS_DESIGN.md)
Advance rule: mean diff >= +1.0 pt AND >= 4/5 folds positive AND localisation recall not lower.
- Arm 7' meets all three -> advances to 3 seeds. Arms 2, 3 and 6 do not advance and are reported as negative results.
- Class weights do not help (cbgroup slightly hurts recall): consistent with Gate A2 (the baseline's errors are missed boxes, not class confusions, so re-weighting classes has little to act on).
- FracAtlas pre-training then fine-tuning hurts (-2.1 pts). Possible reasons (untested hypotheses): 2.4x box-scale mismatch, single-class pre-training, X-ray-only source vs a mixed X-ray/MRI target.

## Checks on arm 7' (done locally, 2026-11-18)
- Localisation recall (class-agnostic, IoU >= 0.5, conf >= 0.25) with the class-specific heads only: +0.039 (base 0.61 -> 0.65); with the generic head added: +0.052. False-positive boxes per fold are unchanged (8.6 base vs 8.2 / 9.0). The recall gain is not an artefact of the extra head.
- Per-class AP50-95 change (mean over folds): Oblique +0.049, Healthy +0.044, Greenstick +0.026, Spiral +0.016, Oblique Displaced +0.008, Transverse +0.006, Comminuted +0.004, Transverse Displaced 0.000, Linear 0, Segmental 0. Gains are not concentrated in the two largest classes.

## Confounds still open (must be closed before any claim)
1. **Training compute.** Arm 7' trains on 933 + 717 images per epoch (about 1.77x the gradient steps; 54 vs 25 min). Control: baseline for 177 epochs (notebook 06, arm tag `ep177`).
2. **Seeds.** One seed per arm; baseline seed spread inside a fold reaches 3 pts. Control: seeds 43 and 44 for arm 7' (notebook 07).
3. **Scale mismatch.** Test the zoom variant of arm 7' (notebook 05 with VARIANT = "zoom", RUN_PRE = False).
4. Pre-training variance for arm 6 was not sampled (one FracAtlas checkpoint) - moot while arm 6 does not advance.
