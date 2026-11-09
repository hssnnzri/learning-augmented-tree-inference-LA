"""Evaluate LA with both oracles (combined, strict) and ScisTree2 alone on the test sets (tests.py) and the real data.

  real         both real matrices: reference clades recovered and triplet agreement with the reference tree
  general      200 broad-range datasets             recoverable   200 recoverable datasets
  realistic    100 realistic datasets               loss-grid     1,680 datasets (random losses 0-40%)
  factor-grid  2,160 datasets (deletions, parallel mutations, doublets, dropout, missing reads, false positives)
  old-paper    3,360 datasets (the paper's previous simulations, 20-100 cells)
  parallel-control  540 datasets (the factor grid's 270 parallel datasets, merged and split)
Simulated tests also record seconds per method and the corrected matrices' genotype, ancestor-descendant and
different-lineage accuracy (the last two without the mutations changed by ISA violations; tests.score). Seconds are
measured inside parallel workers, so they compare methods on the same dataset rather than give standalone run times.
Writes results/<test>.csv; finished datasets are appended to results/<test>_partial.csv, so a stopped run resumes.

--min-cells / --max-cells / --factors run part of a test set; results/<test>.csv always holds every dataset finished so far.

Run: python evaluate.py real|general|recoverable|realistic|loss-grid|factor-grid|parallel-control|old-paper [--oracles combined strict]
     [--workers 6] [--limit N] [--min-cells N] [--max-cells N] [--factors deletions doublets ...]
"""
import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from features import load_oracle, predict
from la import build_tree, scistree2_tree
from simulate import genome_layout
from tests import FACTORS, GENERATORS, cells_of, jobs, score
from trees import matrix_clades, triplet_answers

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
REAL = {"sublines24": ("sublines24_input.tsv", "sublines24_reference.tsv", 0.0001, 0.075),
        "cells23": ("cells23_input.tsv", "cells23_reference.tsv", 0.001, 0.15)}
_M = {}


def init(oracles):
    for k in oracles:
        _M[k] = load_oracle(k, threads=1)


def job(args):
    test, a = args
    O, G, clades, chrom, fp, fn, doublets, info = GENERATORS[test](*a)
    affected, support = info.pop("_affected"), info.pop("_support", None)
    return {"job": repr(a), **info, **score(O, G, clades, chrom, fp, fn, doublets, _M, affected=affected, support=support)}


def real(oracles):
    rows = []
    for name, (inp, ref, fp, fn) in REAL.items():
        raw = pd.read_csv(HERE / "data" / inp, sep="\t", index_col=0); n = len(raw)
        refc = {c for c in matrix_clades(pd.read_csv(HERE / "data" / ref, sep="\t", index_col=0).loc[raw.index, raw.columns])
                if 1 < len(c) < n}
        row = {"data": name, "cells": n, "mutations": raw.shape[1], "reference clades": len(refc)}
        for k in oracles:
            trips, P = predict(load_oracle(k), raw.to_numpy(), genome_layout(raw.columns)[0])
            row[f"{k} clades"] = len(build_tree(raw, P, fp, fn) & refc)
            row[f"{k} triplets"] = round(float((P.argmax(1) == triplet_answers(list(raw.index), refc, trips)).mean()), 4)
        row["ScisTree2 clades"] = len(scistree2_tree(raw, fp, fn) & refc)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    a = argparse.ArgumentParser(description="LA (both oracles) and ScisTree2 on a test set.")
    a.add_argument("test", choices=["real", *GENERATORS])
    a.add_argument("--oracles", nargs="+", default=["combined", "strict"], choices=["combined", "strict"])
    a.add_argument("--workers", type=int, default=6); a.add_argument("--limit", type=int, default=None)
    a.add_argument("--min-cells", type=int, default=0); a.add_argument("--max-cells", type=int, default=10_000)
    a.add_argument("--factors", nargs="+", default=list(FACTORS), choices=list(FACTORS), help="factor-grid only")
    args = a.parse_args()
    RESULTS.mkdir(exist_ok=True)
    t0 = time.time()
    if args.test == "real":
        d = real(args.oracles)
        d.to_csv(RESULTS / "real.csv", index=False)
        print(d.to_string(index=False))
        return
    todo_all = jobs(args.test)[:args.limit]
    partial = RESULTS / f"{args.test}_partial.csv"
    rows = pd.read_csv(partial).to_dict("records") if partial.exists() else []
    done = {r["job"] for r in rows}
    part = [j for j in todo_all if args.min_cells <= cells_of(args.test, j) <= args.max_cells
            and (args.test != "factor-grid" or j[0] in args.factors)]
    todo = [j for j in part if repr(j) not in done]
    print(f"{args.test}: {len(part)} of {len(todo_all)} datasets selected ({len(part) - len(todo)} already done), "
          f"oracles {', '.join(args.oracles)} + ScisTree2", flush=True)
    with Pool(args.workers, initializer=init, initargs=(args.oracles,)) as pool:
        for k, r in enumerate(pool.imap_unordered(job, [(args.test, j) for j in todo]), 1):
            rows.append(r)
            pd.DataFrame([r]).to_csv(partial, mode="a", header=not partial.exists(), index=False)
            if k % 25 == 0:
                print(f"  ({time.time() - t0:.0f}s) {k}/{len(todo)}", flush=True)
    order = {repr(j): i for i, j in enumerate(todo_all)}
    d = pd.DataFrame([r for r in rows if r["job"] in order]).sort_values("job", key=lambda s: s.map(order))
    d.to_csv(RESULTS / f"{args.test}.csv", index=False)
    methods = [*args.oracles, "ScisTree2"]
    print(f"({time.time() - t0:.0f}s) {len(d)} datasets: true clades " + ", ".join(f"{m} {d[f'{m} clades'].mean():.3f}"
                                                                                   for m in methods)
          + "; triplets " + ", ".join(f"{m} {d[f'{m} triplets'].mean():.3f}" for m in args.oracles))
    print("median seconds per dataset: " + ", ".join(f"{m} {d[f'{m} s'].median():.2f}" for m in methods)
          + "; ancestor-descendant " + ", ".join(f"{m} {d[f'{m} ancestor-descendant'].mean():.3f}" for m in methods)
          + "; different-lineage " + ", ".join(f"{m} {d[f'{m} different-lineage'].mean():.3f}" for m in methods))
    for key in ("level", "factor"):
        if key in d:
            cols = [c for c in ("factor", "version") if c in d] + ["level"]
            print(d.groupby(cols)[[f"{m} clades" for m in methods]].mean().round(3).to_string())
            break


if __name__ == "__main__":
    main()
