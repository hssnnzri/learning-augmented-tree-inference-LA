"""The simulated test sets and their scoring. Each dataset is fixed by its seed (same generators and seeds as the
experiments, so the datasets are identical):
  general      200 datasets from broad ranges, independent of both training simulators (seeds 100000 + s):
               20-100 cells, 50-15,000 mutations, dropout 2-55%, FP 1e-4 - 1e-2, missing reads, losses or 0-30
               deletions, doublets in a quarter, random genome layouts
  recoverable  the first 200 datasets from the ranges simulator (seeds 400000 + s) whose true tree is in the data: at
               most 10% of the true clades rest on 1-2 mutations, and at least 5 such clades
  realistic    100 datasets with 20-60 cells, 1,500-15,000 mutations, dropout 5-45% (seeds 95000 + s); tree shapes,
               missing rates and genome layout from the 23-cell calibration
  loss grid    20 / 50 / 100 cells x 100-10,000 mutations x random losses in 0 / 10 / 20 / 40% of clades, 20 trees per
               point, the same tree at every level; broad ranges otherwise
  factor grid  20 / 50 / 100 cells x 200 / 1,000 / 5,000 mutations x one factor at 4 levels (deletions, parallel
               mutations, doublets, dropout, missing reads, false positives), 10 trees per point
  parallel control  the factor grid's parallel datasets at 5 / 10 / 20% in two versions: "merged" (as the factor grid:
               both copies of a parallel mutation in one column) and "split" (the second copy in its own column, the
               same reads for its cells, so the same information without the ISA violation); both scored on the true
               clades the split version's mutations mark
  old paper    the paper's previous simulations (data/clean_isa), the 20-100-cell block: 20 / 30 / 50 / 100 cells x
               20-1,000 mutations x FP 1e-4 / 1e-3 / 1e-2 x dropout 5 / 10 / 15 / 20%, 10 trees per point; ISA, mutations
               on internal edges chosen uniformly, one error rate for every entry, no missing reads
Rates given to every method: the true FP and the mean dropout (realistic: 0.001 and 0.15).
score(): LA with each oracle and ScisTree2 alone; true clades recovered (true clades with mutation support; doublet
cells left out of both trees and the truth), the oracle's triplet accuracy against the true tree, and seconds per method.
With the mask of mutations changed by ISA violations (each generator's description holds it as "_affected"), also the
corrected matrices: LA's (placement with losses in groups of cells), ScisTree2's (each mutation on its most likely
node) and ScisTree2's tree with LA's placement, scored by genotype accuracy (all mutations, and without the changed
ones) and ancestor-descendant / different-lineage accuracy without the changed ones.
"""
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd

from features import Context, oracle_probs, triplet_rows
from la import build_tree, place, place_loss_aware, scistree2_tree
from simulate import (CALIBRATION, calibration, deleted_mask, draw_ranges, genome_layout, logu, random_layout, simulate,
                      simulate_setting, simulate_v2)
from trees import matrix_clades, split_away

# ---------------------------------------------------------------- general

def general(s):
    """(observed, true genotypes, clades, chromosome codes, given fp, given fn, doublet cells, description)."""
    rng = np.random.default_rng(100_000 + s)
    n = int(rng.integers(20, 101)); m = int(logu(rng, 50, 15_000))
    lo = rng.uniform(0.02, 0.35)
    cfg = dict(beta=rng.uniform(-1.5, 2.0), alpha=logu(rng, 0.2, 2.0), gamma=rng.uniform(0, 1), leaf_w=logu(rng, 0.01, 1.0),
               clonal=rng.uniform(0, 0.2), fn_lo=lo, fn_hi=lo + rng.uniform(0.02, 0.2), fp=logu(rng, 1e-4, 1e-2),
               p_bad=rng.uniform(0, 0.1), hard_k=0.0 if rng.random() < 1 / 3 else rng.uniform(0.3, 2.0),
               p_loss=0.0, loss_lo=0.05, loss_hi=0.3)
    if rng.random() < 1 / 3:
        miss = np.zeros(n)
    else:
        mu = rng.uniform(0.02, 0.3); miss = np.clip(rng.beta(2 * mu / (1 - mu), 2, n), 0, 0.9)
    codes, pos = random_layout(rng, slots=20_000)
    kind = int(rng.integers(3)); weak = 0.2; hit, record = [], []
    if kind < 2:
        cfg["p_loss"] = 0.0 if kind == 0 else rng.uniform(0.1, 0.4)
        O, G, clades = simulate(n, m, cfg["beta"], cfg["alpha"], cfg["fn_lo"], cfg["fn_hi"], cfg["fp"], rng, miss, cfg["gamma"],
                                cfg["leaf_w"], cfg["clonal"], cfg["p_bad"], hard_k=cfg["hard_k"], p_loss=cfg["p_loss"],
                                loss_lo=0.05, loss_hi=0.3, weak_deep=weak, affected=hit)
        chrom = codes[rng.choice(len(codes), O.shape[1])]
    else:
        O, G, clades, chrom, _ = simulate_v2(n, m, cfg, rng, miss, (codes, pos), n_del=rng.uniform(0, 30),
                                             min_lost=max(3, int(0.002 * m)), min_group=1 if rng.random() < 0.5 else 2,
                                             record=record, weak_deep=weak)
        hit = [deleted_mask(O.shape[1], record)]
    O = O.copy(); doublets = []
    if rng.random() < 0.25:
        doublets = sorted(rng.choice(n, int(rng.integers(1, max(1, int(0.05 * n)) + 1)), replace=False))
        for i in doublets:
            j = rng.choice([k for k in range(n) if k not in doublets])
            a, b = O[i].copy(), O[j]
            O[i] = np.where((a == 1) | (b == 1), 1, np.where((a == 0) | (b == 0), 0, 3))
    if rng.random() < 0.5:
        keep = (O == 1).sum(0) >= 2
        if keep.sum() >= 5:
            O, G, chrom, hit[0] = O[:, keep], G[:, keep], chrom[keep], hit[0][keep]
    info = {"cells": n, "mutations": O.shape[1], "dropout": round((cfg["fn_lo"] + cfg["fn_hi"]) / 2, 3), "fp": round(cfg["fp"], 5),
            "missing": round(float((O == 3).mean()), 3), "losses": ["none", "random", "deletions"][kind],
            "doublets": len(doublets), "balance": round(cfg["beta"], 2), "_affected": hit[0]}
    return O, G, list(clades), chrom, cfg["fp"], (cfg["fn_lo"] + cfg["fn_hi"]) / 2, doublets, info


# ---------------------------------------------------------------- recoverable

def weak_share(G, clades):
    """Share of the true clades with mutation support that rest on 1-2 mutations, and how many have support."""
    n = G.shape[0]; cols = {}
    for col in G.T:
        k = frozenset(np.flatnonzero(col)); cols[k] = cols.get(k, 0) + 1
    sup = [cols[frozenset(c)] for c in set(clades) if 1 < len(c) < n and frozenset(c) in cols]
    return (np.mean([x <= 2 for x in sup]) if sup else 1.0), len(sup)


def recoverable_seeds(keep=200, max_weak=0.10):
    seeds, s = [], 0
    while len(seeds) < keep:
        O, G, clades, *_ = draw_ranges(np.random.default_rng(400_000 + s))
        w, k = weak_share(G, clades)
        if w <= max_weak and k >= 5:
            seeds.append(s)
        s += 1
    return seeds


def recoverable(s):
    hit = []
    O, G, clades, chrom, fp, fn, doublets, info = draw_ranges(np.random.default_rng(400_000 + s), affected=hit)
    return O, G, clades, chrom, fp, fn, doublets, {**info, "weak share": round(weak_share(G, clades)[0], 3), "_affected": hit[0]}


# ---------------------------------------------------------------- realistic

REALISTIC_FN = [(0.05, 0.15), (0.1, 0.25), (0.15, 0.35), (0.2, 0.45)]
REALISTIC_FP = [0.0001, 0.0003, 0.001, 0.003]
_CAL = {}


def realistic(s):
    if not _CAL:
        _CAL["settings"], _CAL["layouts"], _CAL["miss"] = calibration()
    settings, layouts, miss1 = _CAL["settings"], _CAL["layouts"], _CAL["miss"]
    rng = np.random.default_rng(95_000 + s)
    n = int(rng.integers(20, 61))
    m = int(np.exp(rng.uniform(np.log(1_500), np.log(15_000))))
    cfg = dict(settings[rng.integers(len(settings))])
    cfg["fn_lo"], cfg["fn_hi"] = REALISTIC_FN[rng.integers(len(REALISTIC_FN))]
    cfg["fp"] = float(rng.choice(REALISTIC_FP)); cfg["p_bad"] = float(rng.choice([0.0, 0.025, 0.05]))
    r = rng.random()
    miss = np.zeros(n) if r < 0.3 else rng.choice(miss1, n) if r < 0.7 else rng.beta(0.5, 6, n)
    codes, pos = layouts[rng.integers(len(layouts))]
    kind = rng.integers(3); hit, record = [], []
    if kind < 2:
        cfg.update(p_loss=0.0 if kind == 0 else float(rng.choice([0.2, 0.4])), loss_lo=0.05, loss_hi=0.3)
        O, G, clades = simulate_setting(cfg, rng, n, m, miss, weak_deep=0.2, affected=hit)
        chrom = codes[rng.choice(len(codes), O.shape[1])]
    else:
        O, G, clades, chrom, _ = simulate_v2(n, m, cfg, rng, miss, (codes, pos), n_del=rng.uniform(0, 15),
                                             min_lost=max(5, int(0.005 * m)), min_group=1 if rng.random() < 0.5 else 2,
                                             record=record, weak_deep=0.2)
        hit = [deleted_mask(O.shape[1], record)]
    if rng.random() < 0.5:
        keep = (O == 1).sum(0) >= 2
        O, G, chrom, hit[0] = O[:, keep], G[:, keep], chrom[keep], hit[0][keep]
    info = {"cells": n, "mutations": O.shape[1], "dropout": f"{cfg['fn_lo']}-{cfg['fn_hi']}",
            "losses": ["none", "random", "deletions"][kind], "_affected": hit[0]}
    return O, G, list(clades), chrom, 0.001, 0.15, [], info


# ---------------------------------------------------------------- loss grid

LOSS_CELLS, LOSS_MUTATIONS, LOSS_LEVELS, LOSS_TREES = (20, 50, 100), (100, 200, 500, 1000, 2000, 5000, 10000), (0.0, 0.1, 0.2, 0.4), 20


def loss_grid(n, m, level, s):
    """Random losses at one level; the same tree and mutations at every level. Description includes the share of true
    mutation entries lost."""
    def draw(p, hit=None):
        rng = np.random.default_rng([920_000, n, m, s])
        lo = rng.uniform(0.02, 0.35)
        cfg = dict(beta=rng.uniform(-1.5, 2.0), alpha=logu(rng, 0.2, 2.0), gamma=rng.uniform(0, 1),
                   leaf_w=logu(rng, 0.01, 1.0), clonal=rng.uniform(0, 0.2), fn_lo=lo, fn_hi=lo + rng.uniform(0.02, 0.2),
                   fp=logu(rng, 1e-4, 1e-2), p_bad=rng.uniform(0, 0.1),
                   hard_k=0.0 if rng.random() < 1 / 3 else rng.uniform(0.3, 2.0))
        if rng.random() < 1 / 3:
            miss = np.zeros(n)
        else:
            mu = rng.uniform(0.02, 0.3); miss = np.clip(rng.beta(2 * mu / (1 - mu), 2, n), 0, 0.9)
        codes, _ = random_layout(rng, slots=20_000)
        O, G, clades = simulate(n, m, cfg["beta"], cfg["alpha"], cfg["fn_lo"], cfg["fn_hi"], cfg["fp"], rng, miss,
                                cfg["gamma"], cfg["leaf_w"], cfg["clonal"], cfg["p_bad"], hard_k=cfg["hard_k"], p_loss=p,
                                loss_lo=0.05, loss_hi=0.3, weak_deep=0.2, affected=hit)
        return O, G, clades, codes[rng.choice(len(codes), O.shape[1])], cfg["fp"], (cfg["fn_lo"] + cfg["fn_hi"]) / 2
    hit = []
    O, G, clades, chrom, fp, fn = draw(level, hit)
    lost = 1 - G.sum() / draw(0.0)[1].sum() if level > 0 else 0.0
    info = {"cells": n, "mutations": m, "level": level, "tree": s, "share lost": round(float(lost), 4),
            "dropout": round(fn, 3), "fp": round(fp, 5), "missing": round(float((O == 3).mean()), 3), "_affected": hit[0]}
    return O, G, list(clades), chrom, fp, fn, [], info


# ---------------------------------------------------------------- factor grid

FACTORS = {"deletions": (0, 5, 15, 30), "parallel": (0.0, 0.05, 0.1, 0.2), "doublets": (0.0, 0.025, 0.05, 0.1),
           "dropout": (0.05, 0.15, 0.30, 0.45), "missing": (0.0, 0.1, 0.3, 0.5), "fp": (1e-4, 1e-3, 1e-2, 3e-2)}
FACTOR_CELLS, FACTOR_MUTATIONS, FACTOR_TREES = (20, 50, 100), (200, 1000, 5000), 10


def _factor_base(rng, n):
    lo = rng.uniform(0.02, 0.35)
    cfg = dict(beta=rng.uniform(-1.5, 2.0), alpha=logu(rng, 0.2, 2.0), gamma=rng.uniform(0, 1),
               leaf_w=logu(rng, 0.01, 1.0), clonal=rng.uniform(0, 0.2), fn_lo=lo, fn_hi=lo + rng.uniform(0.02, 0.2),
               fp=logu(rng, 1e-4, 1e-2), p_bad=rng.uniform(0, 0.1),
               hard_k=0.0 if rng.random() < 1 / 3 else rng.uniform(0.3, 2.0), p_loss=0.0, loss_lo=0.05, loss_hi=0.3)
    if rng.random() < 1 / 3:
        miss = np.zeros(n)
    else:
        mu = rng.uniform(0.02, 0.3); miss = np.clip(rng.beta(2 * mu / (1 - mu), 2, n), 0, 0.9)
    return cfg, miss, random_layout(rng, slots=20_000)


def _factor_draw(factor, n, m, level, s, hit=None, copies=None):
    k = list(FACTORS).index(factor)
    rng = np.random.default_rng([930_000, k, n, m, s])
    aux = np.random.default_rng([931_000, k, n, m, s])                # the factor's own draws, same at every level
    cfg, miss, layout = _factor_base(rng, n)
    if factor == "dropout":
        cfg.update(fn_lo=level - 0.04, fn_hi=level + 0.04)
    elif factor == "fp":
        cfg["fp"] = level
    elif factor == "missing":
        miss = np.zeros(n) if level == 0 else np.clip(aux.beta(2 * level / (1 - level), 2, n), 0, 0.9)
    if factor == "deletions":
        record = []
        O, G, clades, chrom, _ = simulate_v2(n, m, cfg, rng, miss, layout, n_del=level, min_lost=max(3, int(0.002 * m)),
                                             min_group=1 if aux.random() < 0.5 else 2, record=record, weak_deep=0.2)
        if hit is not None:
            hit.append(deleted_mask(O.shape[1], record))
    else:
        O, G, clades = simulate(n, m, cfg["beta"], cfg["alpha"], cfg["fn_lo"], cfg["fn_hi"], cfg["fp"], rng, miss,
                                cfg["gamma"], cfg["leaf_w"], cfg["clonal"], cfg["p_bad"], hard_k=cfg["hard_k"],
                                weak_deep=0.2, p_parallel=level if factor == "parallel" else 0.0, affected=hit,
                                parallel_copies=copies)
        chrom = layout[0][rng.choice(len(layout[0]), O.shape[1])]
    doublets = []
    if factor == "doublets" and level > 0:
        order = aux.permutation(n); d = max(1, int(round(level * n)))
        O = O.copy(); doublets = sorted(order[:d].tolist())
        for t, i in enumerate(order[:d]):
            a, b = O[i].copy(), O[order[n - 1 - t]]
            O[i] = np.where((a == 1) | (b == 1), 1, np.where((a == 0) | (b == 0), 0, 3))
    return O, G, list(clades), chrom, cfg["fp"], (cfg["fn_lo"] + cfg["fn_hi"]) / 2, doublets


def factor_grid(factor, n, m, level, s):
    hit = []
    O, G, clades, chrom, fp, fn, doublets = _factor_draw(factor, n, m, level, s, hit)
    info = {"factor": factor, "level": level, "cells": n, "mutations": m, "tree": s, "dropout": round(fn, 3),
            "fp": round(fp, 5), "missing": round(float((O == 3).mean()), 3), "doublets": len(doublets),
            "share changed": float("nan"), "_affected": hit[0]}
    if factor in ("deletions", "parallel"):
        G0 = _factor_draw(factor, n, m, FACTORS[factor][0], s)[1]
        info["share changed"] = round(float(abs(int(G.sum()) - int(G0.sum())) / G0.sum()), 4)
    return O, G, clades, chrom, fp, fn, doublets, info


def parallel_control(version, n, m, level, s):
    """A factor-grid parallel dataset, merged (unchanged) or split. Split: each second copy moves to a new column at
    the same genome position; its cells keep the reads they had, the other cells get false positives at the
    dataset's rate and missing reads at each cell's observed rate, and the second copy's cells in the old column get
    false positives (missing reads stay)."""
    hit, copies = [], []
    O, G, clades, chrom, fp, fn, _ = _factor_draw("parallel", n, m, level, s, hit, copies)
    rng = np.random.default_rng([932_000, n, m, s, round(1_000 * level)])
    miss = (O == 3).mean(1)
    O2, G2, new_O, new_G = O.copy(), G.copy(), [], []
    for j, rows in copies:
        col = np.where(rng.random(n) < miss, 3, (rng.random(n) < fp).astype(O.dtype))
        col[rows] = O[rows, j]
        O2[rows, j] = np.where(O[rows, j] == 3, 3, (rng.random(len(rows)) < fp).astype(O.dtype))
        G2[rows, j] = 0; g = np.zeros(n, dtype=G.dtype); g[rows] = 1
        new_O.append(col); new_G.append(g)
    if copies:
        O2, G2 = np.column_stack([O2, *new_O]), np.column_stack([G2, *new_G])
        chrom2 = np.concatenate([chrom, chrom[[j for j, _ in copies]]])
    else:
        chrom2 = chrom
    info = {"version": version, "level": level, "cells": n, "mutations": m, "tree": s, "copies": len(copies),
            "dropout": round(fn, 3), "fp": round(fp, 5), "_support": G2}
    if version == "merged":
        return O, G, clades, chrom, fp, fn, [], {**info, "_affected": hit[0]}
    return O2, G2, clades, chrom2, fp, fn, [], {**info, "_affected": np.zeros(O2.shape[1], dtype=bool)}


# ---------------------------------------------------------------- old paper

PAPER = Path(__file__).resolve().parent.parent / "data" / "clean_isa"            # copied from the archive's part2/large-data
PAPER_CELLS, PAPER_MUTATIONS = (20, 30, 50, 100), (20, 30, 50, 100, 250, 500, 1000)
PAPER_FP, PAPER_FN, PAPER_TREES = (0.0001, 0.001, 0.01), (0.05, 0.1, 0.15, 0.2), 10
_CODES = []


def old_paper(n, m, fp, fn, k):
    """Tree k of the paper's previous simulations. Chromosome codes drawn from the 23-cell matrix's (no genome there),
    as the experiments did."""
    stem = PAPER / f"simNo_{k}-n_{n}-m_{m}"
    df = pd.read_csv(f"{stem}-fp_{fp}-fn_{fn}-na_0.SC.after_noise", sep="\t", index_col=0)
    G = pd.read_csv(f"{stem}.ground_truth_matrix", sep="\t", index_col=0).loc[df.index, df.columns].to_numpy()
    pos = {c: i for i, c in enumerate(df.index)}
    clades = [frozenset(pos[c] for c in line.split("\t")[0].split(","))
              for line in Path(f"{stem}.BCLT.with_mutations").read_text().splitlines() if line.strip()]
    if not _CODES:
        _CODES.append(genome_layout(pd.read_csv(CALIBRATION, sep="\t", index_col=0, nrows=0).columns)[0])
    chrom = np.random.default_rng(1_000 * n + m + k).choice(_CODES[0], m)
    info = {"cells": n, "mutations": m, "fp": fp, "dropout": fn, "tree": k, "_affected": np.zeros(m, dtype=bool)}
    return df.to_numpy(), G, clades, chrom, fp, fn, [], info


# ---------------------------------------------------------------- the jobs of each test set

def jobs(test):
    """Arguments of every dataset of a test set, in a fixed order."""
    if test == "general":
        return [(s,) for s in range(200)]
    if test == "recoverable":
        return [(s,) for s in recoverable_seeds()]
    if test == "realistic":
        return [(s,) for s in range(100)]
    if test == "loss-grid":
        return [(n, m, L, s) for n in LOSS_CELLS[::-1] for m in LOSS_MUTATIONS[::-1] for L in LOSS_LEVELS
                for s in range(LOSS_TREES)]
    if test == "factor-grid":
        return [(f, n, m, L, s) for n in FACTOR_CELLS[::-1] for m in FACTOR_MUTATIONS[::-1] for f, levels in FACTORS.items()
                for L in levels for s in range(FACTOR_TREES)]
    if test == "parallel-control":
        return [(v, n, m, L, s) for n in FACTOR_CELLS[::-1] for m in FACTOR_MUTATIONS[::-1] for L in FACTORS["parallel"][1:]
                for s in range(FACTOR_TREES) for v in ("merged", "split")]
    if test == "old-paper":                                           # trees outermost: a cut run keeps every setting
        return [(n, m, fp, fn, k) for k in range(1, PAPER_TREES + 1) for n in PAPER_CELLS[::-1]
                for m in PAPER_MUTATIONS[::-1] for fp in PAPER_FP for fn in PAPER_FN]
    raise ValueError(test)


def cells_of(test, a):
    """Number of cells of a test dataset from its job arguments, without simulating it (for general, recoverable and
    realistic it is the generator's first draw)."""
    if test in ("old-paper", "loss-grid"):
        return a[0]
    if test in ("factor-grid", "parallel-control"):
        return a[1]
    seed = {"general": 100_000, "recoverable": 400_000, "realistic": 95_000}[test] + a[0]
    return int(np.random.default_rng(seed).integers(20, 61 if test == "realistic" else 101))


GENERATORS = {"general": general, "recoverable": recoverable, "realistic": realistic, "loss-grid": loss_grid,
              "factor-grid": factor_grid, "parallel-control": parallel_control, "old-paper": old_paper}


# ---------------------------------------------------------------- scoring

def _strict_subsets(cols):
    """For distinct carrier patterns (patterns x cells, 0/1): S[a, b] = pattern b is a strict subset of pattern a."""
    c = cols.astype(np.float32)                             # exact for these counts; float products use BLAS
    with np.errstate(all="ignore"):                          # spurious macOS BLAS warnings
        ov = c @ c.T
    size = c.sum(1)
    return (ov == size[None, :]) & (size[:, None] > size[None, :])


def pair_accuracies(true, inferred, block=2_000):
    """(ancestor-descendant accuracy, different-lineage accuracy) of an inferred cells x mutations 0/1 matrix against
    the true one, exactly as the repo's metrics.ad_accuracy / metrics.dl_accuracy (which take mutations x cells; as there,
    m' leaves m's inferred different-lineage set when m''s inferred descendants equal m's TRUE descendants). Same code
    as the experiments' paper_metrics_V2."""
    T, I = np.asarray(true, dtype=np.int64), np.asarray(inferred, dtype=np.int64)
    tcols, tinv = np.unique(T.T, axis=0, return_inverse=True)
    icols, iinv = np.unique(I.T, axis=0, return_inverse=True)
    tinv, iinv = tinv.ravel(), iinv.ravel()
    St, Si = _strict_subsets(tcols), _strict_subsets(icols)
    ids = {}                                                  # descendant sets (as sets of mutations) -> shared id
    idT = np.array([ids.setdefault(np.packbits(St[p][tinv]).tobytes(), len(ids)) for p in range(len(tcols))])
    idI = np.array([ids.setdefault(np.packbits(Si[p][iinv]).tobytes(), len(ids)) for p in range(len(icols))])
    combos, w = np.unique(np.stack([tinv, iinv], 1), axis=0, return_counts=True)
    a, b, w = combos[:, 0], combos[:, 1], w.astype(float)
    ad = [0.0, 0.0]; dl = [0.0, 0.0]
    for s in range(0, len(a), block):                       # rows of the (combination x combination) pair table
        r = slice(s, s + block)
        dt, di = St[np.ix_(a[r], a)], Si[np.ix_(b[r], b)]     # column is a descendant of the row
        ut, ui = St[np.ix_(a, a[r])].T, Si[np.ix_(b, b[r])].T  # column is an ancestor of the row
        lt = ~dt & ~ut & (idT[a[r]][:, None] != idT[a][None, :])
        li = ~di & ~ui & (idT[a[r]][:, None] != idI[b][None, :])
        with np.errstate(all="ignore"):
            ad[0] += w[r] @ (dt & di) @ w; ad[1] += w[r] @ dt @ w
            dl[0] += w[r] @ (lt & li) @ w; dl[1] += w[r] @ lt @ w
    return (ad[0] / ad[1] if ad[1] else 1.0), (dl[0] / dl[1] if dl[1] else 1.0)


def score(O, G, clades, chrom, fp, fn, doublets, oracles, scistree=True, affected=None, support=None):
    """oracles: {name: model}. Returns {"<name> triplets", "<name> clades", ..., "ScisTree2 clades", "true clades"} and
    seconds per method ("<name> s" = oracle inputs and answers + tree + placement; "ScisTree2 s" = tree + placement).
    affected: mask of the mutations changed by ISA violations; when given, also the corrected matrices' genotype accuracy
    (all mutations; "kept" = without the changed ones) and ancestor-descendant / different-lineage accuracy without the
    changed ones. support: genotypes whose columns decide which true clades count (default G)."""
    clock = time.perf_counter
    n, m = O.shape; cn = [f"s{i}" for i in range(n)]
    df = pd.DataFrame(O, index=cn, columns=[f"m{j}" for j in range(m)]).astype("int64")
    keep = [cn[i] for i in range(n) if i not in doublets]; k = len(keep); drop = {cn[i] for i in doublets}
    restrict = lambda cl: {frozenset(c - drop) for c in cl if 1 < len(c - drop) < k}
    S = G if support is None else support
    true = restrict({frozenset(cn[i] for i in c) for c in clades}) & matrix_clades(pd.DataFrame(S, index=cn).loc[keep])
    trips = list(itertools.combinations(range(n), 3)); cls = sorted({frozenset(c) for c in clades} | {frozenset(range(n))}, key=len)
    truth = np.array([split_away(t, cls) for t in trips]); ok = np.array([not (set(t) & set(doublets)) for t in trips])
    t0 = clock()
    rows = triplet_rows(Context(O, np.asarray(chrom)), trips)
    t_inputs = clock() - t0
    out, secs, mats = {}, {}, {}
    for name, model in oracles.items():
        t0 = clock()
        P = oracle_probs(model, rows, len(trips))
        t1 = clock()
        cl = build_tree(df, P, fp, fn)
        t2 = clock()
        out[f"{name} triplets"] = round(float((P.argmax(1) == truth)[ok].mean()), 4)
        out[f"{name} clades"] = round(len(restrict(cl) & true) / max(len(true), 1), 4)
        if affected is not None:
            mats[name] = place_loss_aware(df, cl, fp, fn)[0]
        t3 = clock()
        secs.update({f"{name} oracle s": t_inputs + t1 - t0, f"{name} tree s": t2 - t1, f"{name} placement s": t3 - t2,
                     f"{name} s": t_inputs + t3 - t0})
    if scistree:
        t0 = clock()
        scis = scistree2_tree(df, fp, fn)
        t1 = clock()
        out["ScisTree2 clades"] = round(len(restrict(scis) & true) / max(len(true), 1), 4)
        if affected is not None:
            mats["ScisTree2"] = place(df, scis, fp, fn)
        t2 = clock()
        secs.update({"ScisTree2 tree s": t1 - t0, "ScisTree2 s": t2 - t0})
        if affected is not None:
            mats["ScisTree2 + A"] = place_loss_aware(df, scis, fp, fn)[0]
    out["true clades"] = len(true)
    out.update({key: round(v, 4) for key, v in secs.items()})
    if affected is not None:
        rows_kept = [i for i in range(n) if i not in doublets]
        T = np.asarray(G)[rows_kept]; good = ~np.asarray(affected, dtype=bool)
        out["affected mutations"] = int((~good).sum())
        for name, M in mats.items():
            I = M.loc[keep, df.columns].to_numpy()
            out[f"{name} genotypes"] = round(float((I == T).mean()), 5)
            kept, ad, dl = float("nan"), float("nan"), float("nan")      # every column present, so rows line up
            if good.any():
                kept = float((I[:, good] == T[:, good]).mean())
                ad, dl = pair_accuracies(T[:, good], I[:, good])
            out.update({f"{name} genotypes kept": round(kept, 5), f"{name} ancestor-descendant": round(ad, 4),
                        f"{name} different-lineage": round(dl, 4)})
    return out
