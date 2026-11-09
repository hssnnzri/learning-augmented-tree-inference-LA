"""LA with a synthetic oracle of known error on the clean ISA test set (the old-paper datasets, tests.old_paper): for
every triplet the true answer with probability 1 - error, otherwise one of the two wrong answers uniformly at random,
given to LA as hard answers (probability 1 on the answer). Same datasets, tree building, clade scoring and placement
scoring (LA's placement with losses in groups of cells: genotype, ancestor-descendant and different-lineage accuracy)
as evaluate.py old-paper; the XGBoost oracles and ScisTree2 are read from results/old-paper.csv for the summary.
Each dataset's answers are fixed by its own seed (950000, error, cells, mutations, fp, dropout, tree).
Writes results/noisy-oracle-<error>.csv; finished datasets are appended to results/noisy-oracle-<error>_partial.csv, so
a stopped run resumes.

Run: python noisy_oracle.py [--error 0.05] [--workers 4] [--limit N]
"""
import argparse
import itertools
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from la import build_tree, place_loss_aware
from tests import PAPER_FN, PAPER_FP, jobs, old_paper, pair_accuracies
from trees import matrix_clades, split_away

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def job(args):
    error, a = args
    n, m, fp, fn, k = a
    O, G, clades, chrom, fp, fn, doublets, info = old_paper(*a)
    info.pop("_affected")
    cn = [f"s{i}" for i in range(n)]
    df = pd.DataFrame(O, index=cn, columns=[f"m{j}" for j in range(m)]).astype("int64")
    true = ({frozenset(cn[i] for i in c) for c in clades if 1 < len(c) < n}
            & matrix_clades(pd.DataFrame(G, index=cn, columns=df.columns)))
    cls = sorted({frozenset(c) for c in clades} | {frozenset(range(n))}, key=len)
    truth = np.array([split_away(t, cls) for t in itertools.combinations(range(n), 3)])
    rng = np.random.default_rng([950_000, round(error * 1_000), n, m, PAPER_FP.index(fp), PAPER_FN.index(fn), k])
    wrong = rng.random(len(truth)) < error
    answer = np.where(wrong, (truth + rng.integers(1, 3, len(truth))) % 3, truth)
    t0 = time.perf_counter()
    cl = build_tree(df, np.eye(3)[answer], fp, fn)
    secs = time.perf_counter() - t0
    found = {frozenset(c) for c in cl if 1 < len(c) < n} & true
    I = place_loss_aware(df, cl, fp, fn)[0].loc[cn, df.columns].to_numpy()
    T = np.asarray(G)
    ad, dl = pair_accuracies(T, I)                    # no ISA violations here, so every mutation is scored
    return {"job": repr(a), **info, "noisy triplets": round(float((answer == truth).mean()), 4),
            "noisy clades": round(len(found) / max(len(true), 1), 4), "true clades": len(true), "noisy tree s": round(secs, 4),
            "noisy genotypes": round(float((I == T).mean()), 5), "noisy ancestor-descendant": round(ad, 4),
            "noisy different-lineage": round(dl, 4)}


def main():
    a = argparse.ArgumentParser(description="LA with a synthetic oracle of known error on the clean ISA test set.")
    a.add_argument("--error", type=float, default=0.05)
    a.add_argument("--workers", type=int, default=4); a.add_argument("--limit", type=int, default=None)
    args = a.parse_args()
    RESULTS.mkdir(exist_ok=True)
    t0 = time.time()
    name = f"noisy-oracle-{args.error:g}"
    todo_all = jobs("old-paper")[:args.limit]
    partial = RESULTS / f"{name}_partial.csv"
    rows = pd.read_csv(partial).to_dict("records") if partial.exists() else []
    done = {r["job"] for r in rows}
    todo = [j for j in todo_all if repr(j) not in done]
    print(f"{name}: {len(todo_all)} datasets ({len(todo_all) - len(todo)} already done)", flush=True)
    with Pool(args.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(job, [(args.error, j) for j in todo]), 1):
            rows.append(r)
            pd.DataFrame([r]).to_csv(partial, mode="a", header=not partial.exists(), index=False)
            if k % 100 == 0:
                print(f"  ({time.time() - t0:.0f}s) {k}/{len(todo)}", flush=True)
    order = {repr(j): i for i, j in enumerate(todo_all)}
    d = pd.DataFrame([r for r in rows if r["job"] in order]).sort_values("job", key=lambda s: s.map(order))
    d.to_csv(RESULTS / f"{name}.csv", index=False)
    cols = ["combined clades", "strict clades", "ScisTree2 clades"]
    d = d.merge(pd.read_csv(RESULTS / "old-paper.csv")[["job"] + cols], on="job", how="left")
    print(f"({time.time() - t0:.0f}s) {len(d)} datasets: noisy triplets {d['noisy triplets'].mean():.3f}; true clades "
          + ", ".join(f"{c.split()[0]} {d[c].mean():.3f}" for c in ["noisy clades", *cols]))
    for key in ("cells", "mutations"):
        print(d.groupby(key)[["noisy clades", *cols]].agg(["mean", "median"]).round(3).to_string())


if __name__ == "__main__":
    main()
