# Code

To run LA on a matrix, only `run.py` is needed; see the [main README](../README.md). The other scripts reproduce the
paper's results. Run every command from the repository's main folder.

## Files

| File | What it does |
|---|---|
| `run.py` | runs LA on one matrix and writes the tree, corrected matrix and summary |
| `la.py` | the LA algorithm: builds the tree from the oracle's answers and places the mutations |
| `features.py` | the oracle: computes its inputs for each triplet of cells and returns its predictions |
| `trees.py` | helper functions for trees |
| `simulate.py` | simulates datasets with a known true tree, for training and testing |
| `tests.py` | the simulated test sets and how results are scored against the truth |
| `evaluate.py` | runs LA and ScisTree2 on a test set and saves the scores in `results/` |
| `timing.py` | measures the run time of LA and ScisTree2 |
| `noisy_oracle.py` | runs LA with a synthetic oracle that is right 95% of the time |
| `figures_clean_isa.py` | figures for the clean ISA test set |
| `fig_runtime_oracle.py` | figure of run time and oracle accuracy |
| `fig_mixed_sets.py` | figures for the other test sets |

## Reproducing the results

```bash
python code/evaluate.py real               # the 24-subline and 23-cell datasets
python code/evaluate.py general            # also: recoverable, realistic, loss-grid, factor-grid, parallel-control, clean-isa
python code/timing.py loss-grid            # also: clean-isa
python code/noisy_oracle.py
python training/train_accuracy.py
python code/figures_clean_isa.py; python code/fig_runtime_oracle.py; python code/fig_mixed_sets.py
```
