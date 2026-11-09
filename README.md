# Learning-augmented tree inference (LA)

Learning-augmented tree inference (LA) for single-cell mutation data: a learned split-away oracle answers, for every
triplet of cells, which cell split away first; LA builds the tree from those answers and places the mutations on it.
The repository contains the current pipeline, trained models, simulated clean ISA test data, and final results.
The original 23-cell data is hosted separately; download it as described below before retraining or running tests
that use it. Historical experiments remain in the local research archive and are not included here.

## Data source and attribution

The calibration matrix, comprising data from 23 sublines, was obtained from a previously published study:
[Liu et al., Nature Communications 17, 8740 (2026)](https://doi.org/10.1038/s41467-026-75172-9).
Synthetic training datasets were generated using fitted tree and noise parameters, per-cell missing-data rates,
and genomic coordinates derived from this matrix.
The BibTeX entry is provided as `longreadpaper2026` in [references.bib](references.bib).

The 24-subline input and reference matrices are included in `data/`. The 23-cell input and reference matrices
and saved inference outputs are available in the
[23-cell data repository](https://github.com/hssnnzri/learning-augmented-tree-inference-LA-cells23-data).

## Setup and 23-cell data download

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python fetch_cells23.py
```

The downloader checks SHA-256 hashes and saves the two matrices as `data/cells23_input.tsv` and
`data/cells23_reference.tsv`. These downloaded files are ignored by Git. To use a local clone of the data
repository instead:

```bash
.venv/bin/python fetch_cells23.py --source ../learning-augmented-tree-inference-LA-cells23-data
```

The 23-cell download is needed for retraining, the realistic test, the clean ISA evaluation (which samples
chromosome labels from the 23-cell input), and evaluation of the 23-cell real data. Running inference on the
included 24-subline matrix does not need this download. The clean ISA matrices and trees themselves are included
in `data/clean_isa/` (3,920 files, about 114 MB).

## Two oracles

| Oracle | File | Training data |
|---|---|---|
| combined | `models/combined.ubj` | 15,000 simulated datasets with mutation losses (random losses and region deletions) |
| strict | `models/strict.ubj` | 15,000 datasets from the same simulators, mixture and seeds with every loss switched off (infinite-sites assumption), for the robustness tests; the random stream shifts at the first dataset with losses, so the trees differ from combined's |

Both are XGBoost classifiers (3 classes, depth 8, 600 rounds) on 58 inputs per ordered triplet (`features.py`), trained
on 21.5 million labelled triplets (`train.py`). The labels come from the simulated trees: the tree of every training
dataset is generated first, so the answer to every triplet is known.

## Files

| File | What it does |
|---|---|
| `simulate.py` | the three training-data generators (general, wide, ranges) |
| `features.py` | the oracle's 58 inputs and its predictions (probabilities averaged over the 6 cell orders) |
| `la.py` | LA: top-down tree, small groups by whole-tree likelihood, agreement splitting, global search; mutation placement (plain, with losses in groups of cells, with one-cell losses); ScisTree2 alone |
| `trees.py` | tree helpers |
| `train.py` | trains either oracle (writes `models/<oracle>_retrained.ubj`, never the shipped files) |
| `run.py` | LA on one matrix: tree, corrected matrix, summary |
| `tests.py` | the simulated test sets and their scoring |
| `evaluate.py` | both oracles and ScisTree2 on a test set or the real data, to `results/` (clades, triplets, genotype, ancestor-descendant and different-lineage accuracy, seconds) |
| `timing.py` | run times one dataset at a time with all threads (old-paper and loss-grid subsets), to `results/timing_<test>.csv` |
| `noisy_oracle.py` | LA with the 95% synthetic oracle on the clean ISA test set, to `results/noisy-oracle-0.05.csv` |
| `train_accuracy.py` | the main oracle's triplet accuracy on training vs held-out data, to `results/train-accuracy.csv` |
| `figures_clean_isa.py` | paper figures on the clean ISA test set: Robinson-Foulds accuracy, genotype / AD / DL accuracy |
| `fig_runtime_oracle.py` | paper figure: run time, and oracle accuracy on training vs held-out data |
| `fig_mixed_sets.py` | Robinson-Foulds accuracy by cells and by mutations per cell, one row per test set |
| `PhISCS/exact_ml_tree.py` | exact maximum-likelihood tree: the PhISCS integer program (no losses) solved with HiGHS |
| `verify.py` | checks against the local archived experiments; requires the separately retained archive |
| `fetch_cells23.py` | downloads or copies the separately hosted 23-cell matrices and verifies their checksums |
| `data/` | the 24-subline matrices, simulator settings, download manifest, and `clean_isa/`; downloaded 23-cell matrices are ignored by Git |
| `requirements.txt` | Python packages |

## Data and test sets

Real data: `sublines24` (24 sublines x 1,689 mutations, curated matrix; reference = the released ScisTree tree,
fp 0.0001, fn 0.075) and `cells23` (23 cells x 10,862 mutations, a separate calling of the same tumor; reference =
its best tree, fp 0.001, fn 0.15).

Simulated test sets (`tests.py`): general (200, broad ranges, independent of the training simulators), recoverable (200),
realistic (100), loss grid (1,680; random losses 0-40%), factor grid (2,160; deletions, parallel mutations, doublets,
dropout, missing reads, false positives), parallel control (540; the factor grid's parallel datasets with both copies in
one column, as there, or the second copy in its own column with the same reads: the cost of the violation at equal
information), old paper = clean ISA (3,360; the paper's previous simulations, 20-100 cells, in `data/clean_isa/`: ISA,
mutations placed uniformly on internal edges, one error rate, no missing reads).

Ancestor-descendant and different-lineage accuracy leave out the mutations changed by ISA violations (losses, deletions,
parallel copies; the simulators record them). Seconds in `evaluate.py` results come from parallel workers; run times
for reporting come from `timing.py`.

## Run

Activate the environment with `source .venv/bin/activate`, then run these commands from the repository root.
Commands using the 23-cell data require the download above.

```
python run.py --input data/sublines24_input.tsv --fp 0.0001 --fn 0.075 --reference data/sublines24_reference.tsv
python run.py --input data/cells23_input.tsv --fp 0.001 --fn 0.15 --reference data/cells23_reference.tsv --oracle strict
python evaluate.py real
python evaluate.py general            # also: recoverable, realistic, loss-grid, factor-grid, parallel-control, old-paper
python timing.py old-paper            # also: loss-grid
python noisy_oracle.py               # 95% synthetic oracle on the clean ISA test set
python train_accuracy.py              # oracle accuracy on training vs held-out data
python figures_clean_isa.py; python fig_runtime_oracle.py; python fig_mixed_sets.py
python PhISCS/exact_ml_tree.py --input data/sublines24_input.tsv --best data/sublines24_reference.tsv --fp 0.0001 --fn 0.075 --mip-rel-gap 0
python verify.py                      # local historical checks only (needs ../archive; not shipped)
python train.py combined              # only to rebuild a model (about 35 minutes)
```

Requirements: `requirements.txt` (numpy, pandas, scipy, xgboost 3.0.5, scistree2, ete3).
