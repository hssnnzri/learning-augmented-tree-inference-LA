# Learning-augmented tree inference (LA)

LA works out the family tree of a group of cells from their mutations.

As a tumour grows, its cells divide and pick up new mutations (small changes in their DNA). A cell passes its
mutations on to all of its descendants, so cells that share mutations are related. Sequencing single cells shows
which mutations each cell carries, but the readout is noisy: some mutations are missed, a few are reported that are
not really there, and some positions are not read at all.

LA takes this noisy table and gives back:

- the **tree** of the cells: which cells are most closely related, and in what order the groups split apart;
- a **corrected table** that agrees with the tree.

To build the tree, LA uses a trained model (the *oracle*) that looks at three cells at a time and guesses which one
split off first. LA then puts these guesses together into one tree. The trained models are included, so you do not
need to train anything.

## 1. Install (once)

You need Python 3 and git. LA was tested with Python 3.13 on macOS. Open a terminal and run:

```bash
git clone https://github.com/hssnnzri/learning-augmented-tree-inference-LA.git
cd learning-augmented-tree-inference-LA
python3 -m venv .venv                  # make a separate Python environment for LA
source .venv/bin/activate              # switch it on
pip install -r requirements.txt        # install the packages LA needs (a few minutes)
```

Each time you open a new terminal, go back into the folder and switch the environment on again:

```bash
cd learning-augmented-tree-inference-LA
source .venv/bin/activate
```

## 2. Test run

The folder includes a real example: 24 cell lines (sublines) from one tumour, with 1,689 mutations. To run LA on it:

```bash
python code/run.py --input data/sublines24_input.tsv --fp 0.0001 --fn 0.075
```

It takes about a second and prints:

```
24 cells x 1689 mutations, 0.1% missing, oracle combined
  LA             : FP 8, FN 144, log10 likelihood -549
  ScisTree2 alone: FP 8, FN 144, log10 likelihood -549
  LA plain placement: log10 likelihood -1,301.5; same tree as ScisTree2: True; oracle 0.09 s, tree 0.02 s
wrote .../results/sublines24_input_combined.corrected.tsv, .nwk, .summary.json
```

What the numbers mean:

- `FP 8`: LA judged 8 entries that were read as 1 to be errors, and changed them to 0.
- `FN 144`: LA judged 144 entries that were read as 0 to be missed mutations, and changed them to 1.
- `log10 likelihood`: how well the tree explains the table. Closer to 0 is better.
- `ScisTree2 alone`: the same numbers from ScisTree2, an existing tree-building program, for comparison. On this
  example both programs find the same tree.
- The third line gives extra detail and the run time. The last line lists the files LA wrote (see [Output](#4-output)).

To compare the result with a known tree, add `--reference`:

```bash
python code/run.py --input data/sublines24_input.tsv --fp 0.0001 --fn 0.075 --reference data/sublines24_reference.tsv
```

The LA line then also shows `22/22 reference clades`: the known tree has 22 groups of related cells (*clades*), and
LA found all 22. It also shows `triplet agreement 94.0%`: how often the oracle's guesses agree with the known tree.

## 3. Input

LA reads one table, saved as a tab-separated text file. In Excel, use *File > Save As > Text (Tab delimited)*.

- First row: the mutation names.
- First column: the cell names.
- Every other entry: `1` if the mutation was seen in that cell, `0` if it was not seen, `3` if there is no reading.

The first rows and columns of the example file `data/sublines24_input.tsv`:

| cell_id_x_mut_id | 1-5602235-T-C | 1-6248992-T-C | 1-10151994-A-G | 1-11289146-G-T |
|---|---|---|---|---|
| C20 | 1 | 0 | 0 | 1 |
| C21 | 0 | 0 | 1 | 1 |
| C23 | 0 | 0 | 0 | 1 |

Write each mutation name as `chromosome-position-from-to`. For example, `1-5602235-T-C` means chromosome 1,
position 5,602,235, T changed to C. LA uses the chromosome and the position. The top-left corner can hold any text.

You also give two error rates:

- `--fp` (false positive rate): how often a mutation that is not there is reported as present (a true 0 read as 1).
- `--fn` (false negative rate): how often a mutation that is there is missed (a true 1 read as 0).

Use the rates estimated for your sequencing method. The example uses `--fp 0.0001 --fn 0.075`.

## 4. Output

LA writes three files to `results/`, named after the input file plus `_combined` (the name of the model used):

| File | What it holds |
|---|---|
| `<name>_combined.nwk` | the tree, in Newick format (a standard text format for trees). Open it in a tree viewer such as [iTOL](https://itol.embl.de) or [FigTree](https://github.com/rambaut/figtree) to see it drawn. |
| `<name>_combined.corrected.tsv` | the corrected table: the same cells and mutations as the input, every entry 0 or 1. Errors are fixed and missing readings are filled in from the tree. |
| `<name>_combined.summary.json` | the numbers printed on screen plus more detail, as a text file. |

How to read the tree: brackets group related cells. In `((C12,C10),(C14,C3))`, C12 and C10 are each other's
closest relatives, so are C14 and C3, and the two pairs are more closely related to each other than to any cell
outside the brackets. The example's full tree is in `results/sublines24_input_combined.nwk`.

The corrected table also allows for mutations that were lost again in some cells, for example when part of a
chromosome is deleted.

## 5. Run it on your own data

Save your table as described in [Input](#3-input), then run with your file name and error rates:

```bash
python code/run.py --input path/to/my_table.tsv --fp 0.001 --fn 0.1
```

The results appear as `results/my_table_combined.nwk`, `.corrected.tsv` and `.summary.json`.

Optional settings:

| Setting | What it does |
|---|---|
| `--out folder/name` | writes the files somewhere else, as `folder/name.nwk` and so on |
| `--reference known.tsv` | compares with a known tree, given as a 0/1 table of the same cells and mutations |
| `--oracle strict` | uses the second model, trained on data where mutations are never lost. The default model (`combined`) allows losses. |

Typical run times on a laptop: under a second for 20 to 30 cells, about 6 seconds for 100 cells. LA was tested with
up to 200 cells; the time grows quickly with more cells.

## What is in each folder

| Folder | What is inside | Needed to run LA? |
|---|---|---|
| `code/` | the LA program (`run.py` is the one to run) and the scripts that reproduce the paper's tests and figures. Details in [code/README.md](code/README.md). | yes |
| `models/` | the two trained models | yes |
| `data/` | the two real datasets (24 sublines and 23 cells), the simulated test data, and the settings used to simulate training data | only for the examples and tests |
| `results/` | the saved results behind the paper; new runs are written here too | no |
| `training/` | scripts to train the models again. Details in [training/README.md](training/README.md). | no |
| `PhISCS/` | a separate program that finds the exact best tree, used to make the 24-subline reference | no |

## Data source and attribution

The calibration matrix, comprising data from 23 sublines, was obtained from a previously published study:
[Liu et al., Nature Communications 17, 8740 (2026)](https://doi.org/10.1038/s41467-026-75172-9).
Synthetic training datasets were generated using fitted tree and noise parameters, per-cell missing-data rates,
and genomic coordinates derived from this matrix.
The BibTeX entry is provided as `longreadpaper2026` in [references.bib](references.bib).

The 24-subline and 23-cell input and reference matrices are included in `data/`.

## License

MIT; see [LICENSE](LICENSE).
