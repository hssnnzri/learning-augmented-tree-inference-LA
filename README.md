# Learning-augmented tree inference (LA)

Learning-augmented tree inference (LA) for single-cell mutation data: a learned split-away oracle answers, for every
triplet of cells, which cell split away first; LA builds the tree from those answers and places the mutations on it.
The trained oracles are included, so no training is needed.

## Setup

```bash
git clone https://github.com/hssnnzri/learning-augmented-tree-inference-LA.git
cd learning-augmented-tree-inference-LA
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requirements: Python 3 (tested with Python 3.13 on macOS) and the packages in `requirements.txt`, which the last
command installs: numpy 2.2.6, pandas 2.3.2, scipy 1.18.1, xgboost 3.0.5, ScisTree2, ete3 3.1.3 and matplotlib 3.11.2.
On macOS, XGBoost also needs the OpenMP library: `brew install libomp`.

## Input Description

The input is a cells x mutations matrix in a tab-separated file. The first row holds the mutation names and the first
column the cell names. Each entry is `1` (mutation present), `0` (absent) or `3` (missing).

Mutation names must be `chrom-pos-ref-alt` (for example `1-5602235-T-C`); LA uses the chromosome and position.
The start of `data/sublines24_input.tsv`:

```
cell_id_x_mut_id  1-5602235-T-C  1-6248992-T-C  1-10151994-A-G
C20               1              0              0
C21               0              0              1
C23               0              0              0
```

## Parameters Description

| Parameter | Description |
|---|---|
| `--input` | the input matrix file (required) |
| `--fp` | false positive rate of the data (required) |
| `--fn` | false negative rate of the data (required) |
| `--reference` | a known tree to compare with, as a 0/1 matrix of the same cells and mutations (optional) |
| `--out` | where to write the output files, as a path and file name without extension (optional; default `results/<input name>_<oracle>`) |
| `--oracle` | the trained oracle to use: `combined` (default), `strict`, or a model file (optional) |
| `--small-groups` | how small groups of cells are solved: `exhaustive` (default) or `vote-dp` (optional) |
| `--no-loss-aware` | place every mutation on one edge, without mutation losses (optional) |
| `--no-cell-losses` | do not place losses in single cells (optional) |

## Output Description

LA writes three files, named `results/<input name>_combined` by default:

| File | Description |
|---|---|
| `.nwk` | the cell tree, in Newick format (opens in tree viewers such as [iTOL](https://itol.embl.de)) |
| `.corrected.tsv` | the corrected matrix: same cells and mutations, 0/1, with errors corrected and missing entries filled in from the tree |
| `.summary.json` | the printed numbers plus details: corrections, mutation losses placed, run time, ScisTree2 comparison |

## Test Run

```bash
python code/run.py --input data/sublines24_input.tsv --fp 0.0001 --fn 0.075 --reference data/sublines24_reference.tsv
```

It prints:

```
24 cells x 1689 mutations, 0.1% missing, oracle combined
  LA             : FP 8, FN 144, log10 likelihood -549, 22/22 reference clades, triplet agreement 94.0%
  ScisTree2 alone: FP 8, FN 144, log10 likelihood -549, 22/22 reference clades
  LA plain placement: log10 likelihood -1,301.5; same tree as ScisTree2: True; oracle 0.09 s, tree 0.02 s
wrote .../results/sublines24_input_combined.corrected.tsv, .nwk, .summary.json
```

FP and FN are the entries corrected from 1 to 0 and from 0 to 1. `22/22 reference clades` means LA recovered all 22
clades of the reference tree. ScisTree2 is shown for comparison.

The 24-subline reference (`data/sublines24_reference.tsv`) is the proven maximum-likelihood tree from PhISCS. The
PhISCS code is in `PhISCS/`, and its saved output (tree, corrected matrix, clade list) is in
`results/sublines24_phiscs.*`.

## Folders

| Folder | Contents |
|---|---|
| `code/` | LA (`run.py`) and the scripts that reproduce the paper's results; see [code/README.md](code/README.md) |
| `models/` | the two trained oracles |
| `data/` | the 24-subline and 23-cell datasets and the simulated test data |
| `results/` | saved results |
| `training/` | only needed to retrain the oracles; see [training/README.md](training/README.md) |
| `PhISCS/` | exact maximum-likelihood tree, used for the 24-subline reference |

## Data source and attribution

These two datasets are included in `data/`:

- **23-cell data**: from [Liu et al., Nature Communications 17, 8740 (2026)](https://doi.org/10.1038/s41467-026-75172-9).
  The simulated training data was generated with settings derived from this matrix.
- **24-subline data**: sublines from [Gruen et al., bioRxiv (2023)](https://doi.org/10.1101/2023.04.03.535074); the
  matrix is the one released in the [Partition-Function repository](https://github.com/algo-cancer/Partition-Function).

BibTeX entries are in [references.bib](references.bib).

## License

MIT; see [LICENSE](LICENSE).
