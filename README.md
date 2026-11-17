# HBFMID fracture research (Project 3)

Dataset (kept OUTSIDE this repo, never modified):
  Local : E:\PhD\Admitted\Research\Project 3\HBFMID\Bone Fractures Detection
  Kaggle: /kaggle/input/datasets/nithoukhan2/hbfmid/Human Bone Fractures Multi-modal Image Dataset (HBFMID)/Bone Fractures Detection

Workflow: VS Code (local) -> GitHub -> clone in Kaggle -> train via "Save Version" (background).

## Layout
- docs/            tracker, decisions, literature notes
- scripts/eda/     data audit scripts (read-only on dataset)
- scripts/training/ training scripts (run_baseline.py)
- scripts/evaluation/ confusion + aggregation scripts
- scripts/common/  shared definitions (taxonomy)
- notebooks/kaggle/ Kaggle notebooks
- configs/         YAML configs
- data_splits/     text lists of clean splits (no images)
- results/eda|training|evaluation/  outputs (small files only)
- tmp_cache/       local caches (git-ignored)

All scripts take --data-root and --out arguments so they run identically locally and on Kaggle.

## Run order (Phase 1)
1. `python scripts/eda/make_clean_splits.py --groups results/eda/image_groups.csv --out data_splits --data-root "<dataset root>" --k 5 --seed 42`
2. `python scripts/eda/verify_splits.py data_splits results/eda/image_groups.csv`
3. Kaggle: notebooks/kaggle/01_baseline_E1_E2.ipynb (E1 official + E2 clean folds), then `scripts/evaluation/aggregate_baseline.py --runs results/baseline`
4. Plan and gates: docs/IDEA_V3.md; progress: docs/TRACKER.md
