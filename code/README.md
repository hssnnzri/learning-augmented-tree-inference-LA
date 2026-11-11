# Code

This folder holds LA and the scripts behind the paper's results. To run LA on a table, only `run.py` is needed; see
the [main README](../README.md). Run every command below from the repository's main folder, with the environment
switched on (`source .venv/bin/activate`).

## Files

| File | What it does |
|---|---|
| `run.py` | LA on one matrix: tree, corrected matrix, summary |
| `la.py` | LA: top-down tree, small groups by whole-tree likelihood, agreement splitting, global search; mutation placement (plain, with losses in groups of cells, with one-cell losses); ScisTree2 alone |
| `features.py` | the oracle's 58 inputs and its predictions (probabilities averaged over the 6 cell orders) |
| `trees.py` | tree helpers |
| `simulate.py` | the three training-data generators (general, wide, ranges) |
| `tests.py` | the simulated test sets and their scoring |
| `evaluate.py` | both oracles and ScisTree2 on a test set or the 24 sublines dataset, to `results/` (clades, triplets, genotype, ancestor-descendant and different-lineage accuracy, seconds) |
| `timing.py` | run times one dataset at a time with all threads, to `results/timing_<test>.csv` |
| `noisy_oracle.py` | LA with the 95% synthetic oracle on the clean ISA test set, to `results/noisy-oracle-0.05.csv` |
| `figures_clean_isa.py` | figures on the clean ISA test set: Robinson-Foulds accuracy, genotype / AD / DL accuracy |
| `fig_runtime_oracle.py` | run time, and oracle accuracy on training vs held-out data |
| `fig_mixed_sets.py` | Robinson-Foulds accuracy by cells and by mutations per cell, one row per test set |

Elsewhere in the repository: `../training/` (training the oracles), `../PhISCS/exact_ml_tree.py` (exact
maximum-likelihood tree: the PhISCS integer program (no losses) solved with HiGHS), `../data/` (the 24-subline and
23-cell matrices, simulator settings, and `clean_isa/`).

## Two oracles

| Oracle | File | Training data |
|---|---|---|
| combined | `models/combined.ubj` | 15,000 simulated datasets with mutation losses (random losses and region deletions) |
| strict | `models/strict.ubj` | 15,000 datasets from the same simulators, mixture, and seeds as combined, with mutation losses disabled (infinite-sites assumption), for robustness tests. The trees can differ because simulating losses uses additional random numbers. |

Both are XGBoost classifiers (3 classes, depth 8, 600 rounds) on 58 inputs per ordered triplet (`features.py`), trained
on 21.5 million labelled triplets (`training/train.py`). The labels come from the simulated trees: the tree of every
training dataset is generated first, so the answer to every triplet is known.

## Data and test sets

data: `sublines24` (24 sublines x 1,689 mutations, mutation matrix; reference = proven optimal tree from PhISCS,
fp 0.0001, fn 0.075) and `cells23` (23 cells x 10,862 mutations, 6.6% missing reads, fp 0.001, fn 0.15; the
calibration matrix the general and wide simulators were fitted to). The clean ISA matrices and trees are in `data/clean_isa/`
(3,920 files, about 114 MB).

Simulated test sets (`tests.py`): general (200, broad ranges, independent of the training simulators), recoverable (200),
realistic (100), loss grid (1,680; random losses 0-40%), factor grid (2,160; deletions, parallel mutations, doublets,
dropout, missing reads, false positives), parallel control (540; the factor grid's parallel datasets with both copies in
one column, as there, or the second copy in its own column with the same reads: the cost of the violation at equal
information), clean ISA (3,360; 20-100 cells, in `data/clean_isa/`: ISA, mutations placed uniformly on internal edges,
one error rate, no missing reads).

Ancestor-descendant and different-lineage accuracy leave out the mutations changed by ISA violations (losses, deletions,
parallel copies; the simulators record them). Seconds in `evaluate.py` results come from parallel workers; run times
for reporting come from `timing.py`.

## Reproducing the results

```bash
python code/run.py --input data/sublines24_input.tsv --fp 0.0001 --fn 0.075 --reference data/sublines24_reference.tsv
python code/run.py --input data/cells23_input.tsv --fp 0.001 --fn 0.15 --reference data/cells23_reference.tsv --oracle strict
python code/evaluate.py real
python code/evaluate.py general            # also: recoverable, realistic, loss-grid, factor-grid, parallel-control, clean-isa
python code/timing.py loss-grid            # also: clean-isa
python code/noisy_oracle.py                # 95% synthetic oracle on the clean ISA test set
python training/train_accuracy.py          # oracle accuracy on training vs held-out data
python code/figures_clean_isa.py; python code/fig_runtime_oracle.py; python code/fig_mixed_sets.py
python PhISCS/exact_ml_tree.py --input data/sublines24_input.tsv --best data/sublines24_reference.tsv --fp 0.0001 --fn 0.075 --mip-rel-gap 0
```
