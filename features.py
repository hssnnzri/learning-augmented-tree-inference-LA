"""The oracle's 58 inputs per ordered triplet, and its predictions.

For an ordered triplet (a, b, c), on the whole observed matrix (3 = missing), using the mutations read in all three:
  42  7 presence patterns x 6 bins of the share of the OTHER cells carrying the mutation (<0.1, 0.1-0.3, 0.3-0.5,
      0.5-0.7, 0.7-0.9, >=0.9), as fractions of the informative mutations (at least one of the three carries it)
   1  log(1 + informative mutations)
   9  per cell: share of read entries that are 1, missing rate, dropout estimate (0s on mutations 90%+ of the other
      cells carry)
   6  for the three pair patterns (two cells share it, the third lacks it) x rare / common (carried by < 50% / >= 50% of
      the other cells): how concentrated those mutations are on single chromosomes, log(1 + chi-square / count)
Each triplet is counted once (sorted order) and its 6 orders are filled in by relabelling; identical mutation columns on
the same chromosome are counted once with a weight. The oracle's answer is the probability that each cell split away
first, averaged over the 6 orders. Same code as the experiments' context_oracle_V1/V2 and fast_features_V2, so the
rows are identical.
"""
import itertools
from pathlib import Path

import numpy as np
import xgboost as xgb

MODELS = Path(__file__).resolve().parent / "models"
ORACLES = {"combined": MODELS / "combined.ubj", "strict": MODELS / "strict.ubj"}
BINS = np.array([0.1, 0.3, 0.5, 0.7, 0.9]); NB = len(BINS) + 1
PERMS = list(itertools.permutations(range(3)))
PAIR_CODES = (3, 5, 6)                       # 011, 101, 110
PARAMS = {"objective": "multi:softprob", "num_class": 3, "max_depth": 6, "eta": 0.1, "subsample": 0.8,
          "colsample_bytree": 0.8, "eval_metric": "mlogloss", "seed": 42, "nthread": 8}


def cell_features(V):
    obs = V != 3; ones = V == 1; T = ones.sum(0); O = obs.sum(0); drop = []
    for i in range(len(V)):
        f = (T - ones[i]) / np.maximum(O - obs[i], 1); trunk = (f >= 0.9) & obs[i]
        drop.append((trunk & ~ones[i]).sum() / max(trunk.sum(), 1))
    return np.column_stack([ones.sum(1) / np.maximum(obs.sum(1), 1), 1 - obs.mean(1), drop])


class Context:
    """Column sums, per-cell inputs and chromosome background of one observed matrix V (cells x mutations, 0/1/3)."""

    def __init__(self, V, chrom):
        V = np.asarray(V, dtype=np.int16)
        self.obs = V != 3; self.onesI = (V == 1).astype(np.int32); self.obsI = self.obs.astype(np.int32)
        self.T = self.onesI.sum(0); self.O = self.obsI.sum(0); self.cell = cell_features(V)
        self.chrom = np.asarray(chrom)
        self.K = int(self.chrom.max()) + 1
        self.bg = np.bincount(self.chrom, minlength=self.K) / len(self.chrom)


def _relabel():
    """For each order p: inv[p][new code] = code in sorted order (codes 1-7, index 0 unused)."""
    out = []
    for p in PERMS:
        inv = np.zeros(8, dtype=int)
        for code in range(8):
            bits = [(code >> 2) & 1, (code >> 1) & 1, code & 1]
            inv[4 * bits[p[0]] + 2 * bits[p[1]] + bits[p[2]]] = code
        out.append(inv)
    return out


INV = _relabel()


class _Compressed:
    """The context's per-mutation arrays restricted to one column per group of identical columns, with weights w."""

    def __init__(self, ctx, first, w):
        self.onesI, self.obsI, self.obs = ctx.onesI[:, first], ctx.obsI[:, first], ctx.obs[:, first]
        self.T, self.O, self.cell, self.w = ctx.T[first], ctx.O[first], ctx.cell, w
        self.chrom, self.K, self.bg = ctx.chrom[first], ctx.K, ctx.bg


def _three_way(X, J, w):
    """For the columns J: S[a, b, c] = sum_j X[0][a, j] X[1][b, j] X[2][c, j] w[j] for all cells a, b, c."""
    n = X[0].shape[0]
    P = (X[0][:, None, J] * X[1][None, :, J]).reshape(n * n, len(J))
    with np.errstate(all="ignore"):                       # spurious BLAS matmul warnings on macOS
        S = (P @ (X[2][:, J] * w[J]).T).reshape(n, n, n)
    assert np.isfinite(S).all()
    return S


def _counts(ctx, trips):
    """For all sorted triplets: pattern x band histogram (T x 7 x NB), informative mutations (T), and the chromosome
    counts behind the 6 chromosome-concentration inputs (T x 3 pair codes x 2 bands x K)."""
    key = np.vstack([ctx.onesI + 2 * (~ctx.obs), ctx.chrom[None, :]])
    _, first, w = np.unique(key, axis=1, return_index=True, return_counts=True)
    ctx = _Compressed(ctx, first, w)
    ONE = (ctx.onesI == 1).astype(np.float32); ZERO = (ctx.obs & (ctx.onesI == 0)).astype(np.float32)
    wf = ctx.w.astype(np.float32)
    den = np.maximum(ctx.O - 3, 1)
    share = {p: (ctx.T - p) / den for p in (1, 2, 3)}
    t = np.asarray(trips); A, B, C = t[:, 0], t[:, 1], t[:, 2]
    hist = np.zeros((len(t), 7, NB))
    for code in range(1, 8):
        bits = [(code >> 2) & 1, (code >> 1) & 1, code & 1]
        X = [ONE if bit else ZERO for bit in bits]
        band = np.digitize(share[sum(bits)], BINS)
        for k in range(NB):
            J = np.flatnonzero(band == k)
            if len(J):
                hist[:, code - 1, k] = _three_way(X, J, wf)[A, B, C]
    tot = hist.sum((1, 2)).round().astype(np.int64)
    chrom_counts = np.zeros((len(t), 3, 2, ctx.K))
    high = share[2] >= 0.5
    for slot, code in enumerate(PAIR_CODES):
        X = [ONE if (code >> s) & 1 else ZERO for s in (2, 1, 0)]
        for band in (0, 1):
            for ch in range(ctx.K):
                J = np.flatnonzero((high == bool(band)) & (ctx.chrom == ch))
                if len(J):
                    chrom_counts[:, slot, band, ch] = _three_way(X, J, wf)[A, B, C]
    return hist, tot, chrom_counts


def triplet_rows(ctx, trips):
    """Feature rows for every sorted triplet in trips, in all 6 orders (PERMS), triplet-major, as float32."""
    trips = np.asarray([tuple(int(x) for x in t) for t in trips])
    assert (trips[:, 0] < trips[:, 1]).all() and (trips[:, 1] < trips[:, 2]).all(), "triplets must be sorted"
    hist, tot, cc = _counts(ctx, trips)
    k = len(trips)
    N = cc.sum(-1)
    e = N[..., None] * ctx.bg
    x = np.ascontiguousarray((cc - e) ** 2 / np.maximum(e, 1e-9)).sum(-1) / np.maximum(N, 1)
    chi = np.where(N > 0, np.log1p(x), 0.0)                           # (k, 3 pair codes, 2 bands)
    slot = {code: s for s, code in enumerate(PAIR_CODES)}
    rows = np.empty((k * 6, 7 * NB + 1 + 9 + 6), dtype=np.float32)
    totf = tot[:, None]
    for pi, p in enumerate(PERMS):
        inv = INV[pi]
        h = hist[:, inv[1:] - 1, :].reshape(k, 7 * NB)
        h = np.where(totf > 0, h / np.maximum(totf, 1), h)
        parts = [h, np.log1p(tot)[:, None], ctx.cell[trips[:, p[0]]], ctx.cell[trips[:, p[1]]], ctx.cell[trips[:, p[2]]],
                 np.column_stack([chi[:, slot[inv[code]], band] for code in PAIR_CODES for band in (0, 1)])]
        rows[pi::6] = np.hstack(parts)
    return rows


def ordered_rows(ctx, ordered):
    """Rows for triplets given in any order (as in the training data)."""
    srt = [tuple(sorted(int(x) for x in t)) for t in ordered]
    uniq = sorted(set(srt)); index = {t: i for i, t in enumerate(uniq)}
    rows = triplet_rows(ctx, uniq)
    order = {p: i for i, p in enumerate(PERMS)}
    pick = [index[s] * 6 + order[tuple(s.index(int(x)) for x in t)] for t, s in zip(ordered, srt)]
    return rows[pick]


def load_oracle(name_or_path, threads=None):
    """An oracle by name ("combined", "strict") or model file."""
    model = xgb.Booster(model_file=str(ORACLES.get(name_or_path, name_or_path)))
    if threads is not None:
        model.set_param({"nthread": threads})
    return model


def oracle_probs(model, rows, k):
    """Probability that each cell (0, 1, 2) of each of the k triplets split away first, averaged over the 6 orders.
    Identical rows (common with few mutations) are predicted once."""
    R = np.ascontiguousarray(rows.reshape(k, 6, -1).reshape(k * 6, -1))
    sample = R[np.random.default_rng(0).choice(len(R), min(len(R), 5_000), replace=False)]
    void = lambda A: A.view(np.dtype((np.void, A.dtype.itemsize * A.shape[1]))).ravel()
    if len(np.unique(void(np.ascontiguousarray(sample)))) < 0.5 * len(sample):
        _, first, inv = np.unique(void(R), return_index=True, return_inverse=True)
        P = model.inplace_predict(R[first], iteration_range=(0, 0))[inv.ravel()]
    else:
        P = model.inplace_predict(R, iteration_range=(0, 0))
    P = P.reshape(k, 6, 3)
    probs = np.zeros((k, 3))
    for j in range(6):
        for q in range(3):
            probs[:, PERMS[j][q]] += P[:, j, q]
    return probs / 6


def predict(model, O, chrom):
    """(sorted index triplets, probabilities) for every triplet of the observed matrix O (cells x mutations)."""
    trips = list(itertools.combinations(range(len(O)), 3))
    return trips, oracle_probs(model, triplet_rows(Context(O, chrom), trips), len(trips))
