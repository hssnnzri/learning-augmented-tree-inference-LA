"""Simulated single-cell mutation data with a known tree: the training data of both oracles.

One dataset: a random binary tree (Aldous beta-splitting), mutations placed on its edges (the true genotypes), then
mutation losses (random, or region deletions on a chromosome), noise (per-cell dropout, very poor cells, per-mutation
detectability, false positives, missing reads), doublets in some datasets, and removal of mutations read in fewer than
two cells in some datasets (the observed matrix: 0 / 1 / 3 = missing). Three generators:
  general     16-40 cells, 800-15,000 mutations; tree shape from settings fitted to the raw 23-cell matrix
              (data/simulator_settings.json), its per-cell missing rates and genome layout, or random layouts
  wide        as general, 16-100 cells and 20-1,500 mutations; half of the datasets clean (no missing reads, losses or
              filtering)
  ranges      everything from broad published ranges, independent of any dataset (20-100 cells, 100-12,000 mutations,
              dropout 5-45%, doublets in a quarter)
losses=False switches every loss off; the loss kind is still drawn but the loss settings are not, so the random stream
shifts at the first dataset that would have had losses and the later datasets differ from the loss runs'. The code is
the experiments' (realistic_simulator_V1/V2, train_context_oracle_V2_general/_wide, realistic_ranges_V2) unchanged, so
the same seed gives the same dataset.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln

DATA = Path(__file__).resolve().parent.parent / "data"
SETTINGS_PATH = DATA / "simulator_settings.json"
CALIBRATION = DATA / "cells23_input.tsv"          # the raw matrix the general and wide settings were fitted to

FN_RANGES = [(0.01, 0.05), (0.03, 0.1), (0.05, 0.2), (0.1, 0.4), (0.2, 0.6)]
FP_RATES = [0.0001, 0.0005, 0.001, 0.003]
DELETIONS = dict(n_del=15.0, seg=(0.5, 1.0), q=(0.7, 1.0))
logu = lambda rng, lo, hi: float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def beta_split_tree(n, beta, rng):
    """Clades (frozensets of leaf indices, leaves and root included) of a random binary tree on n leaves."""
    clades = []

    def split(leaves):
        clades.append(frozenset(leaves))
        k = len(leaves)
        if k == 1:
            return
        i = np.arange(1, k)
        logw = gammaln(beta + i + 1) + gammaln(beta + k - i + 1) - gammaln(i + 1) - gammaln(k - i + 1)
        w = np.exp(logw - logw.max()); w /= w.sum()
        left = rng.choice(i, p=w)
        perm = rng.permutation(leaves)
        split(list(perm[:left])); split(list(perm[left:]))

    split(list(range(n)))
    return clades


def simulate(n, m, beta, alpha, fn_lo, fn_hi, fp, rng, missing_rates=None, gamma=0.0, leaf_w=1.0,
             clonal=0.0, p_bad=0.0, bad_lo=0.5, bad_hi=0.9, hard_k=0.0, p_loss=0.0, loss_lo=0.05, loss_hi=0.3,
             weak_deep=0.0, p_parallel=0.0, affected=None, parallel_copies=None):
    """Returns (O: observed cells x mutations with 0/1/3; G: true genotypes; clades of the true tree).
    Mutations go on edges with weights Gamma(alpha) x (clade size / n)^gamma, times leaf_w on leaf edges; a share
    `clonal` is carried by every cell. weak_deep: chance that the edges above clades of n/4+ cells keep only 0.3-30% of
    their weight. p_loss: chance that an internal clade loses a share U(loss_lo, loss_hi) of its inherited mutations.
    p_parallel: share of mutations that also arise in a second, disjoint clade. Per-cell dropout U(fn_lo, fn_hi) (very
    poor cells U(bad_lo, bad_hi) with chance p_bad), times a per-mutation detectability Gamma(hard_k, 1/hard_k) when
    hard_k > 0; false positives at rate fp; per-cell missing rates resampled from missing_rates. affected: a list that,
    when given, receives a mask of the returned mutations whose true genotypes a loss or a parallel copy changed (no
    random draws, so the data are the same either way). parallel_copies: likewise a list that receives (returned
    mutation, cells of its second copy) for every parallel copy."""
    clades = beta_split_tree(n, beta, rng)
    edges = [c for c in clades if len(c) < n]
    sizes = np.array([len(c) for c in edges], dtype=float)
    w = rng.gamma(alpha, 1.0, size=len(edges)) * (sizes / n) ** gamma
    w[sizes == 1] *= leaf_w
    if weak_deep and rng.random() < weak_deep:
        deep = sizes >= n / 4
        w[deep] *= 10 ** rng.uniform(-2.5, -0.5, size=deep.sum())
    w = (1 - clonal) * w / w.sum()
    edges = edges + [frozenset(range(n))]
    w = np.append(w, clonal)
    edge_of = rng.choice(len(edges), size=m, p=w)
    G = np.zeros((n, m), dtype=np.int16)
    for e, c in enumerate(edges):
        cols = np.where(edge_of == e)[0]
        if len(cols):
            G[np.ix_(sorted(c), cols)] = 1
    G0 = G.copy() if affected is not None else None
    if p_loss > 0:
        edge_size = np.array([len(edges[e]) for e in edge_of])
        for c in clades:
            if 1 < len(c) < n and rng.random() < p_loss:
                rows = sorted(c)
                ancestral = np.where(G[rows].all(0) & (edge_size > len(c)))[0]
                lost = ancestral[rng.random(len(ancestral)) < rng.uniform(loss_lo, loss_hi)]
                G[np.ix_(rows, lost)] = 0
    copies = []
    if p_parallel > 0:
        others = [c for c in clades if len(c) < n]
        for j in np.flatnonzero(rng.random(m) < p_parallel):
            cand = [d for d in others if not (d & edges[edge_of[j]])]
            if cand:
                second = sorted(cand[rng.integers(len(cand))])
                G[second, j] = 1; copies.append((j, second))
    fn_cell = rng.uniform(fn_lo, fn_hi, size=n)
    bad = rng.random(n) < p_bad
    fn_cell[bad] = rng.uniform(bad_lo, bad_hi, size=bad.sum())
    fn = fn_cell[:, None]
    if hard_k > 0:
        fn = np.clip(fn * rng.gamma(hard_k, 1.0 / hard_k, size=m)[None, :], 0.0, 0.95)
    u = rng.random((n, m))
    O = np.where(G == 1, (u >= fn).astype(np.int16), (u < fp).astype(np.int16))
    if missing_rates is not None:
        miss = rng.choice(missing_rates, size=n, replace=True)[:, None]
        O[rng.random((n, m)) < miss] = 3
    keep = (G.sum(0) > 0)
    if affected is not None:
        affected.append((G != G0).any(0)[keep])
    if parallel_copies is not None:
        new = np.cumsum(keep) - 1
        parallel_copies.extend((int(new[j]), rows) for j, rows in copies if keep[j])
    return O[:, keep], G[:, keep], clades


def deleted_mask(m, record):
    """simulate_v2's record (cells, lost mutations) as a mask over its m returned mutations."""
    mask = np.zeros(m, dtype=bool)
    for _, lost in record:
        mask[lost] = True
    return mask


def simulate_v2(n, m, cfg, rng, missing_rates, layout, n_del=DELETIONS["n_del"], seg=DELETIONS["seg"], q=DELETIONS["q"],
                min_group=2, record=None, min_lost=0, weak_deep=0.0):
    """simulate with genome positions and region deletions instead of random losses: Poisson(n_del) deletions, each in
    a random clade (min_group to n-2 cells), on a chromosome chosen in proportion to its mutations, over a stretch
    covering a share U(seg) of it; each inherited mutation there is lost with probability U(q). Deletions removing
    fewer than min_lost mutations are dropped. Returns (O, G, clades, chromosome codes, positions)."""
    codes, positions = layout
    take = rng.permutation(len(codes))[:m] if m <= len(codes) else rng.choice(len(codes), m)
    chrom, pos = codes[take], positions[take]
    clades = beta_split_tree(n, cfg["beta"], rng)
    edges = [c for c in clades if len(c) < n]
    sizes = np.array([len(c) for c in edges], dtype=float)
    w = rng.gamma(cfg["alpha"], 1.0, size=len(edges)) * (sizes / n) ** cfg["gamma"]
    w[sizes == 1] *= cfg["leaf_w"]
    if weak_deep and rng.random() < weak_deep:
        deep = sizes >= n / 4
        w[deep] *= 10 ** rng.uniform(-2.5, -0.5, size=deep.sum())
    w = (1 - cfg["clonal"]) * w / w.sum()
    edges = edges + [frozenset(range(n))]
    w = np.append(w, cfg["clonal"])
    edge_of = rng.choice(len(edges), size=m, p=w)
    G = np.zeros((n, m), dtype=np.int16)
    for e, c in enumerate(edges):
        cols = np.where(edge_of == e)[0]
        if len(cols):
            G[np.ix_(sorted(c), cols)] = 1
    edge_size = np.array([len(edges[e]) for e in edge_of])
    groups = [c for c in clades if min_group <= len(c) <= n - 2]
    chrom_share = np.bincount(chrom) / m
    for _ in range(rng.poisson(n_del)):
        c = groups[rng.integers(len(groups))]
        ch = rng.choice(len(chrom_share), p=chrom_share)
        on = np.flatnonzero(chrom == ch)
        lo, hi = pos[on].min(), pos[on].max()
        span = rng.uniform(*seg) * (hi - lo)
        start = rng.uniform(lo, hi - span)
        rows = sorted(c)
        inherited = on[(pos[on] >= start) & (pos[on] <= start + span) & G[rows][:, on].all(0) & (edge_size[on] > len(c))]
        lost = inherited[rng.random(len(inherited)) < rng.uniform(*q)]
        if len(lost) < min_lost:
            continue
        G[np.ix_(rows, lost)] = 0
        if record is not None:
            record.append((tuple(rows), lost))
    fn_cell = rng.uniform(cfg["fn_lo"], cfg["fn_hi"], size=n)
    bad = rng.random(n) < cfg["p_bad"]
    fn_cell[bad] = rng.uniform(0.5, 0.9, size=bad.sum())
    fn = fn_cell[:, None]
    if cfg["hard_k"] > 0:
        fn = np.clip(fn * rng.gamma(cfg["hard_k"], 1.0 / cfg["hard_k"], size=m)[None, :], 0.0, 0.95)
    u = rng.random((n, m))
    O = np.where(G == 1, (u >= fn).astype(np.int16), (u < cfg["fp"]).astype(np.int16))
    if missing_rates is not None:
        miss = rng.choice(missing_rates, size=n, replace=True)[:, None]
        O[rng.random((n, m)) < miss] = 3
    keep = G.sum(0) > 0
    if record is not None:
        new = np.cumsum(keep) - 1
        record[:] = [(rows, new[lost[keep[lost]]]) for rows, lost in record]
    return O[:, keep], G[:, keep], clades, chrom[keep], pos[keep]


def genome_layout(columns):
    """Chromosome code and position of each mutation, from names like '1-3960057-A-T'."""
    chrom = np.array([c.split("-")[0] for c in columns])
    _, codes = np.unique(chrom, return_inverse=True)
    return codes, np.array([int(c.split("-")[1]) for c in columns])


def random_layout(rng, slots=12_000):
    """A made-up genome: 8-23 chromosomes of random sizes; mutation slots placed in proportion to size."""
    k = int(rng.integers(8, 24)); sizes = rng.uniform(50e6, 250e6, k)
    chrom = rng.choice(k, slots, p=sizes / sizes.sum())
    return chrom, rng.uniform(0, sizes[chrom]).astype(np.int64)


def load_settings(path=SETTINGS_PATH, per_fit=3):
    """The fitted tree and noise settings: the best per_fit of each calibration fit (plain and conflict-weighted)."""
    fits = json.load(open(path))
    chosen = fits["plain"][:per_fit] + fits["conflict_weighted"][:per_fit]
    return [dict(beta=c["beta"], alpha=c["alpha"], fn_lo=c["fn"][0], fn_hi=c["fn"][1], fp=c["fp"], gamma=c["gamma"],
                 leaf_w=c["leaf_w"], clonal=c["clonal"], p_bad=c["p_bad"], hard_k=c["hard_k"], p_loss=c["p_loss"],
                 loss_lo=c.get("loss", [0.05, 0.3])[0], loss_hi=c.get("loss", [0.05, 0.3])[1]) for c in chosen]


def simulate_setting(cfg, rng, n, m, missing_rates, **extra):
    return simulate(n, m, cfg["beta"], cfg["alpha"], cfg["fn_lo"], cfg["fn_hi"], cfg["fp"], rng, missing_rates,
                    cfg["gamma"], cfg["leaf_w"], cfg["clonal"], cfg["p_bad"], hard_k=cfg["hard_k"],
                    p_loss=cfg["p_loss"], loss_lo=cfg["loss_lo"], loss_hi=cfg["loss_hi"], **extra)


def calibration():
    """(fitted settings, genome layouts, per-cell missing rates) of the calibration matrix, as the generators use them."""
    raw = pd.read_csv(CALIBRATION, sep="\t", index_col=0)
    rng = np.random.default_rng(7)
    layouts = [genome_layout(raw.columns)] + [random_layout(rng) for _ in range(3)]
    return load_settings(), layouts, (raw.to_numpy() == 3).mean(1)


def simulate_general(rng, settings, layouts, miss1, losses=True):
    """One general dataset: (observed, true genotypes, clades, chromosome codes)."""
    n = int(rng.integers(16, 41))
    m = int(np.exp(rng.uniform(np.log(800), np.log(15_000))))
    cfg = dict(settings[rng.integers(len(settings))])
    cfg["fn_lo"], cfg["fn_hi"] = FN_RANGES[rng.integers(len(FN_RANGES))]
    cfg["fp"] = float(rng.choice(FP_RATES)); cfg["p_bad"] = float(rng.choice([0.0, 0.05, 0.1]))
    r = rng.random()
    miss = np.zeros(n) if r < 0.3 else rng.choice(miss1, n) if r < 0.7 else rng.beta(0.5, 6, n)
    codes, pos = layouts[rng.integers(len(layouts))]
    kind = rng.integers(3)                                   # 0 no losses, 1 random losses, 2 deletions
    if not losses:
        kind = 0
    if kind < 2:
        cfg.update(p_loss=0.0 if kind == 0 else float(rng.choice([0.2, 0.4])), loss_lo=0.05, loss_hi=0.3)
        O, G, clades = simulate_setting(cfg, rng, n, m, miss, weak_deep=0.5)
        chrom = codes[rng.choice(len(codes), O.shape[1])]
    else:
        O, G, clades, chrom, _ = simulate_v2(n, m, cfg, rng, miss, (codes, pos), n_del=rng.uniform(0, 15),
                                             min_lost=max(5, int(0.005 * m)), min_group=1 if rng.random() < 0.5 else 2,
                                             weak_deep=0.5)
    if rng.random() < 0.5:
        keep = (O == 1).sum(0) >= 2
        O, G, chrom = O[:, keep], G[:, keep], chrom[keep]
    return O, G, clades, chrom


def simulate_wide(rng, settings, layouts, miss1, size=None, losses=True):
    """One wide dataset: (observed, true genotypes, clades, chromosome codes). size: (cells, mutations) instead of
    drawing them."""
    if size is None:
        n = int(rng.integers(16, 101))
        m = int(np.exp(rng.uniform(np.log(20), np.log(1_500))))
    else:
        n, m = size
    cfg = dict(settings[rng.integers(len(settings))])
    cfg["fn_lo"], cfg["fn_hi"] = FN_RANGES[rng.integers(len(FN_RANGES))]
    cfg["fp"] = float(rng.choice(FP_RATES + [0.01])); cfg["p_bad"] = float(rng.choice([0.0, 0.05, 0.1]))
    codes, pos = layouts[rng.integers(len(layouts))]
    if rng.random() < 0.5:                                   # clean: no missing reads, losses or filtering
        cfg.update(p_loss=0.0, loss_lo=0.05, loss_hi=0.3)
        O, G, clades = simulate_setting(cfg, rng, n, m, np.zeros(n), weak_deep=0.5)
        return O, G, clades, codes[rng.choice(len(codes), O.shape[1])]
    r = rng.random()
    miss = np.zeros(n) if r < 0.3 else rng.choice(miss1, n) if r < 0.7 else rng.beta(0.5, 6, n)
    kind = rng.integers(3)
    if not losses:
        kind = 0
    if kind < 2:
        cfg.update(p_loss=0.0 if kind == 0 else float(rng.choice([0.2, 0.4])), loss_lo=0.05, loss_hi=0.3)
        O, G, clades = simulate_setting(cfg, rng, n, m, miss, weak_deep=0.5)
        chrom = codes[rng.choice(len(codes), O.shape[1])]
    else:
        O, G, clades, chrom, _ = simulate_v2(n, m, cfg, rng, miss, (codes, pos), n_del=rng.uniform(0, 15),
                                             min_lost=max(5, int(0.005 * m)), min_group=1 if rng.random() < 0.5 else 2,
                                             weak_deep=0.5)
    if rng.random() < 0.5:
        keep = (O == 1).sum(0) >= 2
        if keep.sum() >= 5:
            O, G, chrom = O[:, keep], G[:, keep], chrom[keep]
    return O, G, clades, chrom


def draw_ranges(rng, size=None, losses=True, affected=None):
    """One dataset from the broad published ranges: (observed, true genotypes, clades, chromosome codes, given fp,
    given fn, doublet cells, description). size: (cells, mutations) instead of drawing them. affected: a list that
    receives the mask of mutations changed by losses or deletions (as simulate's)."""
    n, m = (int(rng.integers(20, 101)), int(logu(rng, 100, 12_000))) if size is None else size
    lo = rng.uniform(0.05, 0.40); hi = min(0.45, lo + rng.uniform(0.02, 0.15))
    cfg = dict(beta=rng.uniform(-1.0, 1.5), alpha=logu(rng, 0.3, 2.0), gamma=rng.uniform(0, 1), leaf_w=logu(rng, 0.02, 1.0),
               clonal=rng.uniform(0, 0.15), fn_lo=lo, fn_hi=hi, fp=logu(rng, 1e-5, 3e-3), p_bad=rng.uniform(0, 0.05),
               hard_k=0.0 if rng.random() < 1 / 3 else rng.uniform(0.3, 2.0), p_loss=0.0, loss_lo=0.05, loss_hi=0.3)
    mu = rng.uniform(0, 0.5)
    miss = np.zeros(n) if mu < 0.005 else np.clip(rng.beta(2 * mu / (1 - mu), 2, n), 0, 0.9)
    codes, pos = random_layout(rng, slots=15_000)
    kind = int(rng.integers(3))
    if not losses:
        kind = 0
    hit, record = [], []
    if kind < 2:
        cfg["p_loss"] = 0.0 if kind == 0 else rng.uniform(0.05, 0.2)
        O, G, clades = simulate(n, m, cfg["beta"], cfg["alpha"], lo, hi, cfg["fp"], rng, miss, cfg["gamma"], cfg["leaf_w"],
                                cfg["clonal"], cfg["p_bad"], hard_k=cfg["hard_k"], p_loss=cfg["p_loss"], loss_lo=0.05,
                                loss_hi=0.3, weak_deep=0.2, affected=hit)
        chrom = codes[rng.choice(len(codes), O.shape[1])]
    else:
        O, G, clades, chrom, _ = simulate_v2(n, m, cfg, rng, miss, (codes, pos), n_del=rng.uniform(0, 15),
                                             min_lost=max(3, int(0.002 * m)), min_group=1 if rng.random() < 0.5 else 2,
                                             record=record, weak_deep=0.2)
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
    if affected is not None:
        affected.append(hit[0])
    info = {"cells": n, "mutations": O.shape[1], "dropout": round((lo + hi) / 2, 3), "fp": round(cfg["fp"], 6),
            "missing": round(float((O == 3).mean()), 3), "losses": ["none", "random", "deletions"][kind], "doublets": len(doublets)}
    return O, G, list(clades), chrom, cfg["fp"], (lo + hi) / 2, doublets, info
