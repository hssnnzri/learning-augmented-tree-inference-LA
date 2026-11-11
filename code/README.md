# Code

To run LA on a matrix, only `run.py` is needed; see the [main README](../README.md). The other scripts reproduce the
paper's results. Run every command from the repository's main folder.

## Files

| File | What it does |
|---|---|
| `run.py` | LA on one matrix: tree, corrected matrix, summary |
| `la.py` | tree building and mutation placement |
| `features.py` | the oracle's inputs and predictions |
| `trees.py` | tree helpers |
| `simulate.py` | the data simulators |
| `tests.py` | the simulated test sets and their scoring |
| `evaluate.py` | LA (both oracles) and ScisTree2 on a test set, to `results/` |
| `timing.py` | run times, to `results/` |
| `noisy_oracle.py` | LA with the 95% synthetic oracle on the clean ISA test set |
| `figures_clean_isa.py`, `fig_runtime_oracle.py`, `fig_mixed_sets.py` | the figures |

## Reproducing the results

```bash
python code/evaluate.py real               # the 24-subline and 23-cell datasets
python code/evaluate.py general            # also: recoverable, realistic, loss-grid, factor-grid, parallel-control, clean-isa
python code/timing.py loss-grid            # also: clean-isa
python code/noisy_oracle.py
python training/train_accuracy.py
python code/figures_clean_isa.py; python code/fig_runtime_oracle.py; python code/fig_mixed_sets.py
```
