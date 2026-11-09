"""Check that this pipeline reproduces the experiments exactly (needs ../archive/part2/experiments and ../archive/original).

  real      both real matrices, both oracles: identical oracle probabilities, trees, ScisTree2 trees and placements
  datasets  sample datasets of every test set: identical observed matrices, true genotypes, trees and rates; the
            recoverable test keeps the same 200 seeds
  scores    sample datasets of every test set: identical scores to the experiments' result files
  training  block 0 of every training part, both oracles: identical training rows and labels

Run: python verify.py [real] [datasets] [scores] [training]      (all four by default; training takes ~10 minutes)
"""
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parent / "archive" / "part2"
EXP = ARCHIVE / "experiments"
sys.path[1:1] = [str(EXP), str(ARCHIVE)]                  # after this folder: the clean modules come first

import la
import tests
import train
from features import load_oracle, oracle_probs, predict
from simulate import genome_layout

OLD_MODELS = {"combined": EXP / "context_oracle_V2_combined_r600.ubj", "strict": EXP / "context_oracle_V2_combined_noloss_r600.ubj"}
REAL = {"sublines24": ("sublines24_input.tsv", 0.0001, 0.075), "cells23": ("cells23_input.tsv", 0.001, 0.15)}
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}", flush=True)


def real():
    import xgboost as xgb
    from check_seeds_global_V2 import both
    from context_oracle_V2 import ContextV2
    from fast_features_V2 import fast_probs, triplet_rows
    from pipeline_loss_aware_V2 import place_loss_aware
    from pipeline_V2 import reconstruct_v2
    from run_context_oracle_real_V2 import plain
    from single_cell_losses_V2 import add_cell_losses
    print("real data")
    for name, (inp, fp, fn) in REAL.items():
        raw = pd.read_csv(HERE / "data" / inp, sep="\t", index_col=0); n = len(raw)
        codes = genome_layout(raw.columns)[0]
        chrom = np.array([c.split("-")[0] for c in raw.columns]); pos = np.array([int(c.split("-")[1]) for c in raw.columns])
        for oracle in ("combined", "strict"):
            trips, P = predict(load_oracle(oracle), raw.to_numpy(), codes)
            old = xgb.Booster(model_file=str(OLD_MODELS[oracle]))
            P0 = fast_probs(old, triplet_rows(ContextV2(raw.to_numpy(), codes), trips), len(trips))
            check(f"{name} {oracle}: oracle probabilities", np.array_equal(P, P0))
            clades, clades0 = la.build_tree(raw, P, fp, fn), both(raw, P0, fp, fn)[1]
            check(f"{name} {oracle}: tree", clades == clades0, f"{len(clades)} clades")
            A, A0 = la.place_loss_aware(raw, clades, fp, fn)[0], place_loss_aware(raw, clades0, fp, fn)[0]
            check(f"{name} {oracle}: placement with losses", A.equals(A0))
            C, C0 = la.add_cell_losses(raw, A, chrom, pos, fp, fn)[0], add_cell_losses(raw, A0, chrom, pos, fp, fn)[0]
            check(f"{name} {oracle}: one-cell losses", C.equals(C0))
            p1, p0 = la.place(raw, clades, fp, fn), plain(raw, clades0, fp, fn)
            differ = (p1.to_numpy() != p0.to_numpy()).any(0)
            L = lambda M: la.corrections(M, raw, fp, fn)["log10_likelihood"]
            check(f"{name} {oracle}: plain placement", L(p1) == L(p0),
                  f"{int(differ.sum())} columns placed differently (equal-likelihood ties), same likelihood")
        check(f"{name}: ScisTree2 tree", la.scistree2_tree(raw, fp, fn) == reconstruct_v2(raw, lambda a, b, c: a, n + 1, fp, fn)[1])


def _old_dataset(test, a):
    """The experiments' generator for one dataset, in the clean generators' output order."""
    if test == "general":
        import general_test_V2
        return general_test_V2.draw(*a)
    if test == "recoverable":
        import realistic_ranges_V2 as R
        return R.draw(np.random.default_rng(400_000 + a[0]))
    if test == "realistic":
        import diagnose_V2 as dg
        if not dg._G:
            dg.init()
        O, G, clades, chrom, fp, fn, _ = dg.load(("realistic", 0, a[0]))
        return O, G, clades, chrom, fp, fn, []
    if test == "loss-grid":
        import loss_grid_V2
        O, G, clades, chrom, fp, fn, _ = loss_grid_V2.load(*a)
        return O, G, clades, chrom, fp, fn, []
    import robustness_grid_V2
    return robustness_grid_V2.draw(*a)


SAMPLES = {"general": [(s,) for s in (0, 1, 7, 42, 199)],
           "recoverable": [(14,), (13,), (500,)],
           "realistic": [(s,) for s in (0, 5, 99)],
           "loss-grid": [(20, 100, 0.0, 0), (50, 1000, 0.2, 3), (100, 10000, 0.4, 19)],
           "factor-grid": [(f, 20, 200, FL[1], 2) for f, FL in tests.FACTORS.items()] + [("doublets", 100, 5000, 0.1, 9)]}


def datasets():
    print("test datasets")
    for test, samples in SAMPLES.items():
        for a in samples:
            new = tests.GENERATORS[test](*a)[:7]
            old = _old_dataset(test, a)[:7]
            same = (np.array_equal(new[0], old[0]) and np.array_equal(new[1], old[1]) and set(map(frozenset, new[2])) == set(map(frozenset, old[2]))
                    and np.array_equal(np.asarray(new[3]), np.asarray(old[3])) and new[4] == old[4] and new[5] == old[5]
                    and list(new[6]) == list(old[6]))
            check(f"{test} {a}", same, f"{new[0].shape[0]} cells x {new[0].shape[1]} mutations")
    import recoverable_test_V2
    check("recoverable: the same 200 seeds", tests.recoverable_seeds() == recoverable_test_V2.pick()[0])


def _score_rows(test):
    """(job args, expected values) for a few small datasets of each test from the experiments' result files."""
    out = []
    if test == "general":
        d = pd.read_csv(EXP / "combined_broad_tests_V2_general.csv")
        for _, r in d[(d.cells <= 40) & (d.mutations <= 2000)].head(3).iterrows():
            out.append(((int(r.dataset),), {"combined triplets": r["combined triplets"], "combined clades": r["combined clades"],
                                            "ScisTree2 clades": r["ScisTree2 clades"], "true clades": r["true clades"]}))
    elif test == "recoverable":
        d = pd.read_csv(EXP / "train_context_oracle_V2_combined_recoverable.csv")
        for _, r in d[(d.cells <= 40) & (d.mutations <= 3000)].head(3).iterrows():
            out.append(((int(r.seed),), {"combined triplets": r["combined triplets"], "combined clades": r["combined clades"],
                                         "ScisTree2 clades": r["ScisTree2 clades"]}))
    elif test == "realistic":
        d = pd.read_csv(EXP / "train_context_oracle_V2_combined_old.csv"); d = d[d["test set"] == "realistic"]
        for _, r in d.head(3).iterrows():
            out.append(((int(eval(r.dataset)[1]),), {"combined triplets": r["combined triplets"],
                                                      "combined clades": r["combined clades"], "ScisTree2 clades": r["ScisTree2 clades"]}))
    else:
        f = "loss_grid_V2.csv" if test == "loss-grid" else "robustness_grid_V2.csv"
        d = pd.read_csv(EXP / f); d = d[(d.cells == 20) & (d.mutations <= 1000)]
        if test == "loss-grid":                                       # one dataset per loss level
            d = d.groupby("level").head(1)
        else:                                                         # one dataset per factor, at its third level
            d = pd.concat([d[(d.factor == f) & np.isclose(d.level, L[2])].head(1) for f, L in tests.FACTORS.items()])
        for _, r in d.iterrows():
            if test == "loss-grid":
                a = (int(r.cells), int(r.mutations), float(r.level), int(r.tree))
            else:
                a = (r.factor, int(r.cells), int(r.mutations), int(r.level) if r.factor == "deletions" else float(r.level),
                     int(r.tree))
            out.append((a, {"combined triplets": r["combined triplets"], "combined clades": r["combined clades"],
                            "strict triplets": r["no losses triplets"], "strict clades": r["no losses clades"],
                            "ScisTree2 clades": r["ScisTree2 clades"], "true clades": r["true clades"]}))
    return out


def scores():
    print("scores against the experiments' result files")
    models = {k: load_oracle(k) for k in ("combined", "strict")}
    for test in SAMPLES:
        for a, want in _score_rows(test):
            O, G, clades, chrom, fp, fn, doublets, _ = tests.GENERATORS[test](*a)
            got = tests.score(O, G, clades, chrom, fp, fn, doublets, models)
            bad = {k: (got[k], v) for k, v in want.items() if abs(got[k] - v) > 1e-9}
            check(f"{test} {a}", not bad, f"{len(want)} values" + (f", differ: {bad}" if bad else ""))


def _block(args):
    which, part, version, folder = args
    losses = which == "combined"
    if version == "new":
        return args, train.block(part, 0, losses)
    import realistic_ranges_V2 as R
    import train_context_oracle_V2_combined as cb
    import train_context_oracle_V2_combined_noloss as nl
    import train_context_oracle_V2_wide as tw
    if not losses:
        nl.CHUNK_DIR = Path(folder)
        nl.chunk((part, 0))
        stem = Path(folder) / f"{part.replace(' ', '_')}_000"
        return args, (np.load(f"{stem}_X.npy"), np.load(f"{stem}_y.npy"))
    if part in ("general", "wide"):
        return args, tw.chunk((part == "wide", 0))
    if part == "R":
        return args, R.chunk(("R", 0))
    return args, cb.new_chunk(part, 0)


def training():
    print("training data (block 0 of every part)")
    with tempfile.TemporaryDirectory() as folder, Pool(6) as pool:
        got = dict(pool.imap_unordered(_block, [(w, p, v, folder) for w in ("combined", "strict") for p in train.PARTS
                                                for v in ("new", "old")]))
    for w in ("combined", "strict"):
        for p in train.PARTS:
            (X1, y1), (X0, y0) = got[(w, p, "new", folder)], got[(w, p, "old", folder)]
            check(f"{w} {p}", np.array_equal(X1, X0) and np.array_equal(y1, y0), f"{len(y1):,} rows")


def main():
    parts = sys.argv[1:] or ["real", "datasets", "scores", "training"]
    for p in parts:
        {"real": real, "datasets": datasets, "scores": scores, "training": training}[p]()
    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
