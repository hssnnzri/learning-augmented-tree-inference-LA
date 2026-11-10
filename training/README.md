# Training (not needed to use LA)

The trained models are already included in `models/`, so you do not need anything in this folder to run LA. It is
here only for anyone who wants to rebuild the models from scratch.

Run every command from the repository's main folder, with the environment switched on
(`source .venv/bin/activate`).

## Rebuilding a model

The training data is simulated to resemble the 23-cell dataset in `data/`.

1. Train. This takes about 35 minutes per model.

   ```bash
   python training/train.py combined          # or: strict
   ```

   The new model is saved as `models/combined_retrained.ubj` (or `models/strict_retrained.ubj`). The included models
   are never overwritten. While training runs, temporary files are kept in `train_blocks_combined/` so a stopped run
   can pick up where it left off; they are deleted at the end. `--workers 6` sets how many processes simulate data
   in parallel.

2. Use the new model:

   ```bash
   python code/run.py --input data/sublines24_input.tsv --fp 0.0001 --fn 0.075 --oracle models/combined_retrained.ubj
   ```

## Files

| File | What it does |
|---|---|
| `train.py` | trains either oracle (writes `models/<oracle>_retrained.ubj`) |
| `train_accuracy.py` | the main oracle's triplet accuracy on training vs held-out data, to `results/train-accuracy.csv` |

## Training data

15,000 simulated datasets in blocks of 100, each block with its own NumPy seed (`code/simulate.py` has the generators):

| Part | Datasets | Generator |
|---|---|---|
| general | 3,000 | general |
| wide | 1,500 | wide |
| R | 4,500 | ranges |
| general extra | 3,000 | general |
| few | 1,500 | 5-50 mutations per cell, 20-100 cells (wide and ranges, alternating) |
| large | 1,500 | 100-200 cells, 500-15,000 mutations (wide and ranges, alternating) |

From each dataset, up to 1,500 triplets of cells are labelled with the cell that splits away first in the simulated
tree. The combined oracle trains on data with mutation losses; the strict oracle uses the same simulators and seeds
with losses switched off. Full details are at the top of `train.py`.
