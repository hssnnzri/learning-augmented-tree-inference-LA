"""The main (combined) oracle's triplet accuracy on its own training data and on held-out data from the same simulators,
for the train vs test panel of fig_runtime_oracle.py. Train: the first block of every training part (100 datasets each,
600 in all), rebuilt exactly as train.block builds it (same seeds, same random stream, same sampled triplets).
Held out: the next, unused block of every part (seed SEEDS[part] + PARTS[part], 600 datasets), built the same way.
Scored as tests.score scores test data (probabilities averaged over the 6 orders, argmax against the generating tree's
split-away answer). Writes results/train-accuracy.csv (one row per dataset: data, part, cells, mutations, triplets,
accuracy).

Run: python train_accuracy.py [--workers 6]
"""
import argparse
import itertools
from multiprocessing import Pool

import numpy as np
import pandas as pd

from features import Context, load_oracle, oracle_probs, triplet_rows
from simulate import calibration
from train import PARTS, PER_TREE, SEEDS, dataset
from trees import split_away
from figures_clean_isa import RESULTS


def part_accuracy(args):
    part, c = args
    settings, layouts, miss1 = calibration()
    model = load_oracle("combined", threads=1)
    rng = np.random.default_rng(SEEDS[part] + c); rows = []
    for k in range(100):
        O, clades, chrom, bad = dataset(part, k, rng, settings, layouts, miss1)
        n = O.shape[0]
        canon = [t for t in itertools.combinations(range(n), 3) if not (bad & set(t))]
        pick = rng.choice(len(canon), min(PER_TREE, len(canon)), replace=False)
        [rng.permutation(canon[i]) for i in pick]                       # keep the random stream as in train.block
        trips = [canon[i] for i in pick]
        cl = sorted(clades, key=len)
        P = oracle_probs(model, triplet_rows(Context(O, np.asarray(chrom)), trips), len(trips))
        truth = np.array([split_away(t, cl) for t in trips])
        rows.append({"data": "train" if c < PARTS[part] else "held out", "part": part, "dataset": k, "cells": n, "mutations": O.shape[1], "triplets": len(trips),
                     "accuracy": round(float((P.argmax(1) == truth).mean()), 4)})
    return rows


def main():
    a = argparse.ArgumentParser(description="Main oracle's triplet accuracy on training and held-out data.")
    a.add_argument("--workers", type=int, default=6)
    args = a.parse_args()
    with Pool(args.workers) as pool:
        jobs = [(p, c) for p in PARTS for c in (0, PARTS[p])]
        rows = [r for part in pool.imap_unordered(part_accuracy, jobs) for r in part]
    d = pd.DataFrame(rows).sort_values(["data", "part", "dataset"])
    d.to_csv(RESULTS / "train-accuracy.csv", index=False)
    print(d.groupby(["part", "data"])["accuracy"].agg(["mean", "median", "count"]).round(3).unstack().to_string())


if __name__ == "__main__":
    main()
