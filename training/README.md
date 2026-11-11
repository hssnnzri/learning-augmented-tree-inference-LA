# Training (not needed to use LA)

The trained models are included in `models/`. This folder holds the scripts that rebuild them from scratch. Commands
run from the repository's main folder.

## Rebuilding a model

Part of the training data is simulated with settings taken from the 23-cell dataset in `data/`: tree-shape and
noise settings fitted to it, its missing-read rate per cell, and its mutation positions on the genome.

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

| Datasets | Size | Settings |
|---|---|---|
| 6,000 | 16-40 cells, 800-15,000 mutations | dropout 1-60%, false positives 0.0001-0.003, missing reads in 70% of the datasets |
| 1,500 | 16-100 cells, 20-1,500 mutations | dropout 1-60%, false positives 0.0001-0.01; half with no missing reads or losses |
| 4,500 | 20-100 cells, 100-12,000 mutations | dropout 5-45%, false positives 0.00001-0.003, missing reads 0-50%, doublets in a quarter of the datasets |
| 1,500 | 20-100 cells, 5-50 mutations per cell | half as row 2, half as row 3 |
| 1,500 | 100-200 cells, 500-15,000 mutations | half as row 2, half as row 3 |

From each dataset, up to 1,500 triplets of cells are labelled with the cell that splits away first in the simulated
tree. The combined oracle trains on data with mutation losses; the strict oracle uses the same simulators and seeds
with losses switched off. Full details are at the top of `train.py`.
