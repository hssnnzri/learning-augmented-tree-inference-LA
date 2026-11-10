"""Train the two oracles: combined (training data with mutation losses) and strict (the same simulators, parts and seeds
with every loss switched off: the infinite-sites assumption, for the robustness tests). Strict's datasets are not
combined's with the losses removed: the random stream of a block shifts at its first dataset with losses (simulate.py).

15,000 simulated datasets in blocks of 100, each block with its own NumPy seed (simulate.py has the generators):
  general        3,000   general generator                        seeds 5000-5029
  wide           1,500   wide generator                           seeds 9000-9014
  R              4,500   ranges generator                         seeds 200000-200044
  general extra  3,000   general generator                        seeds 600000-600029
  few            1,500   5-50 mutations per cell, 20-100 cells    seeds 610000-610014
  large          1,500   100-200 cells, 500-15,000 mutations      seeds 620000-620014
  (few and large: even datasets from the wide generator, odd ones from the ranges generator)
From each dataset up to 1,500 triplets (cells in random order; triplets with a doublet cell left out), labelled with
the cell that splits away first in the generating tree. XGBoost, 3 classes, depth 8, 600 rounds, eta 0.1, row and
column subsampling 0.8, seed 42, streamed from disk. Same code, seeds and block order as the experiments
(train_context_oracle_V2_combined.py, train_context_oracle_V2_combined_noloss.py); models/combined.ubj and
models/strict.ubj are the models those runs saved.

Run: python training/train.py combined|strict [--out file.ubj] [--workers 6]      (about 35 minutes each)
"""
import argparse
import itertools
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))     # the LA code
from features import PARAMS, Context, ordered_rows
from simulate import calibration, draw_ranges, logu, simulate_general, simulate_wide
from trees import split_away

HERE = Path(__file__).resolve().parent.parent                      # repository root
PARTS = {"general": 30, "wide": 15, "R": 45, "general extra": 30, "few": 15, "large": 15}     # blocks of 100 datasets
SEEDS = {"general": 5_000, "wide": 9_000, "R": 200_000, "general extra": 600_000, "few": 610_000, "large": 620_000}
PER_TREE, ROUNDS = 1_500, 600


def dataset(part, k, rng, settings, layouts, miss1, losses=True):
    """Dataset k of a block: (observed, clades, chromosome codes, doublet cells)."""
    if part in ("general", "general extra"):
        O, _, clades, chrom = simulate_general(rng, settings, layouts, miss1, losses=losses)[:4]
        return O, clades, chrom, set()
    if part == "wide":
        O, _, clades, chrom = simulate_wide(rng, settings, layouts, miss1, losses=losses)
        return O, clades, chrom, set()
    if part == "R":
        O, _, clades, chrom, _, _, doublets, _ = draw_ranges(rng, losses=losses)
        return O, clades, chrom, set(doublets)
    if part == "few":
        n = int(rng.integers(20, 101)); m = max(20, int(n * logu(rng, 5, 50)))
    else:
        n = int(rng.integers(100, 201)); m = int(logu(rng, 500, 15_000))
    if k % 2 == 0:
        O, _, clades, chrom = simulate_wide(rng, settings, layouts, miss1, size=(n, m), losses=losses)
        return O, clades, chrom, set()
    O, _, clades, chrom, _, _, doublets, _ = draw_ranges(rng, size=(n, m), losses=losses)
    return O, clades, chrom, set(doublets)


def block(part, c, losses=True):
    """Training rows and labels of block c of a part: (X float32, y int8)."""
    settings, layouts, miss1 = calibration()
    rng = np.random.default_rng(SEEDS[part] + c); X, y = [], []
    for k in range(100):
        O, clades, chrom, bad = dataset(part, k, rng, settings, layouts, miss1, losses)
        n = O.shape[0]
        canon = [t for t in itertools.combinations(range(n), 3) if not (bad & set(t))]
        pick = rng.choice(len(canon), min(PER_TREE, len(canon)), replace=False)
        trips = [tuple(rng.permutation(canon[i])) for i in pick]
        cl = sorted(clades, key=len)
        X.append(ordered_rows(Context(O, np.asarray(chrom)), trips)); y += [split_away(t, cl) for t in trips]
    return np.vstack(X).astype(np.float32), np.asarray(y, dtype=np.int8)


def _write_block(args):
    part, c, losses, folder = args
    stem = folder / f"{part.replace(' ', '_')}_{c:03d}"
    if not Path(f"{stem}_y.npy").exists():
        X, y = block(part, c, losses)
        np.save(f"{stem}_X.npy", X); np.save(f"{stem}_y.npy", y)
    return part, len(np.load(f"{stem}_y.npy"))


class _Blocks(xgb.DataIter):
    """Feeds the saved blocks to XGBoost one at a time."""

    def __init__(self, stems):
        self.stems, self.i = stems, 0
        super().__init__()

    def next(self, input_data):
        if self.i == len(self.stems):
            return False
        input_data(data=np.load(f"{self.stems[self.i]}_X.npy"), label=np.load(f"{self.stems[self.i]}_y.npy"))
        self.i += 1
        return True

    def reset(self):
        self.i = 0


def train(which, out, workers=6):
    losses = which == "combined"
    t0 = time.time()
    folder = HERE / f"train_blocks_{which}"                                   # kept until training ends (resumable)
    folder.mkdir(exist_ok=True)
    with Pool(workers) as pool:
        done = list(pool.imap_unordered(_write_block, [(p, c, losses, folder) for p, k in PARTS.items() for c in range(k)]))
    counts = pd.DataFrame(done, columns=["part", "rows"]).groupby("part")["rows"].sum()
    print(f"{which}: training triplets by part " + ", ".join(f"{p} {counts[p]:,}" for p in PARTS)
          + f"; total {counts.sum():,} ({time.time() - t0:.0f}s)", flush=True)
    stems = sorted(str(p)[:-6] for p in folder.glob("*_X.npy"))
    D = xgb.QuantileDMatrix(_Blocks(stems))
    xgb.train({**PARAMS, "max_depth": 8, "nthread": 8}, D, num_boost_round=ROUNDS).save_model(str(out))
    del D
    shutil.rmtree(folder)
    print(f"{which}: saved {out} ({time.time() - t0:.0f}s)", flush=True)


def main():
    a = argparse.ArgumentParser(description="Train the combined or the strict (no-loss) oracle.")
    a.add_argument("which", choices=("combined", "strict"))
    a.add_argument("--out", type=Path, default=None, help="default: models/<which>_retrained.ubj (never overwrites)")
    a.add_argument("--workers", type=int, default=6)
    args = a.parse_args()
    train(args.which, args.out or HERE / "models" / f"{args.which}_retrained.ubj", args.workers)


if __name__ == "__main__":
    main()
