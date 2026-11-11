"""Run time of LA (each oracle) and ScisTree2 alone, one dataset at a time with all threads for every method (as the
paper's runtime figure and the experiments' timing_combined_V2), on a subset of a test set:
  clean-isa  112 datasets: 20 / 30 / 50 / 100 cells x 7 mutation counts x dropout 5% or 20% x
             trees 1-2, FP 1e-3
  loss-grid  every cells x mutations point at 0% and 40% losses, trees 0-1 (84 datasets)
Seconds as tests.score records them: LA = oracle inputs and answers + tree + placement; ScisTree2 = tree + placement.
evaluate.py's seconds come from parallel workers sharing the machine; use these for run times.
Writes results/timing_<test>.csv.

Run: python code/timing.py clean-isa|loss-grid [--limit N]
"""
import argparse
import time
from pathlib import Path

import pandas as pd

from features import load_oracle
from tests import GENERATORS, LOSS_CELLS, LOSS_MUTATIONS, ISA_CELLS, ISA_MUTATIONS, score

HERE = Path(__file__).resolve().parent.parent                      # repository root
SUBSETS = {"clean-isa": [(n, m, 0.001, fn, k) for n in ISA_CELLS for m in ISA_MUTATIONS for fn in (0.05, 0.2)
                         for k in (1, 2)],
           "loss-grid": [(n, m, L, s) for n in LOSS_CELLS for m in LOSS_MUTATIONS for L in (0.0, 0.4) for s in (0, 1)]}


def run(test, a, models):
    O, G, clades, chrom, fp, fn, doublets, info = GENERATORS[test](*a)
    affected = info.pop("_affected")
    return {"job": repr(a), **info, **score(O, G, clades, chrom, fp, fn, doublets, models, affected=affected)}


def main():
    p = argparse.ArgumentParser(description="Sequential run times of LA and ScisTree2.")
    p.add_argument("test", choices=list(SUBSETS)); p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    models = {k: load_oracle(k) for k in ("combined", "strict")}
    todo = SUBSETS[args.test][:args.limit]
    t0 = time.time()
    run(args.test, todo[0], models)                                  # warm-up, not recorded
    rows = []
    for k, a in enumerate(todo, 1):
        rows.append(run(args.test, a, models))
        if k % 10 == 0:
            print(f"  ({time.time() - t0:.0f}s) {k}/{len(todo)}", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(HERE / "results" / f"timing_{args.test}.csv", index=False)
    cols = ["combined s", "strict s", "ScisTree2 s"]
    print(f"({time.time() - t0:.0f}s) {args.test}: median seconds by cells")
    print(d.groupby("cells")[cols].median().round(3).to_string())


if __name__ == "__main__":
    main()
