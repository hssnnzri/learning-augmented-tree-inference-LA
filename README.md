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

Tested with Python 3.13 on macOS.

## Test run

```bash
python code/run.py --input data/sublines24_input.tsv --fp 0.0001 --fn 0.075 --reference data/sublines24_reference.tsv
```

This takes about a second (24 sublines, 1,689 mutations) and prints:

```
24 cells x 1689 mutations, 0.1% missing, oracle combined
  LA             : FP 8, FN 144, log10 likelihood -549, 22/22 reference clades, triplet agreement 94.0%
  ScisTree2 alone: FP 8, FN 144, log10 likelihood -549, 22/22 reference clades
  LA plain placement: log10 likelihood -1,301.5; same tree as ScisTree2: True; oracle 0.09 s, tree 0.02 s
wrote .../results/sublines24_input_combined.corrected.tsv, .nwk, .summary.json
```

FP and FN are the entries corrected from 1 to 0 and from 0 to 1. `22/22 reference clades` means LA recovered all 22
clades of the reference tree. ScisTree2 is shown for comparison.

## Input

| Option | Description |
|---|---|
| `--input` | cells x mutations matrix, tab-separated. First row: mutation names; first column: cell names; entries `1` (mutation present), `0` (absent), `3` (missing). |
| `--fp`, `--fn` | false positive and false negative rates of the data |
| `--reference` | optional: a known tree, as a corrected 0/1 matrix of the same cells and mutations, for comparison |

Mutation names must be `chrom-pos-ref-alt` (for example `1-5602235-T-C`); LA uses the chromosome and position.
The start of `data/sublines24_input.tsv`:

```
cell_id_x_mut_id  1-5602235-T-C  1-6248992-T-C  1-10151994-A-G
C20               1              0              0
C21               0              0              1
C23               0              0              0
```

## Output

Three files in `results/`, named `<input name>_combined` (change with `--out`):

| File | Description |
|---|---|
| `.nwk` | the cell tree, in Newick format (opens in tree viewers such as [iTOL](https://itol.embl.de)) |
| `.corrected.tsv` | the corrected matrix: same cells and mutations, 0/1, with errors corrected and missing entries filled in from the tree |
| `.summary.json` | the printed numbers plus details: corrections, mutation losses placed, run time, ScisTree2 comparison |

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

The calibration matrix, comprising data from 23 sublines, was obtained from a previously published study:
[Liu et al., Nature Communications 17, 8740 (2026)](https://doi.org/10.1038/s41467-026-75172-9).
Synthetic training datasets were generated using fitted tree and noise parameters, per-cell missing-data rates,
and genomic coordinates derived from this matrix.
The BibTeX entry is provided as `longreadpaper2026` in [references.bib](references.bib).

## License

MIT; see [LICENSE](LICENSE).
