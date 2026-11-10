"""Exact maximum-likelihood tree by integer programming: the PhISCS formulation (Malikic et al. 2019) without mutation
losses (K = 0) and without bulk data, solved with HiGHS (scipy.optimize.milp), to prove whether a tree is optimal.

The likelihood of a tree places every mutation on its best node; maximizing it over all trees is the same as finding the
0/1 matrix Y (cells x mutations) without conflicts (a perfect phylogeny: any two mutation columns are nested or
disjoint) that best explains the reads:
  maximize   sum over reads of log10 P(read | Y)        (read 1: 1-fn if Y=1, fp if Y=0; read 0: fn / 1-fp; missing: 0)
  subject to for every pair of columns p, q: the gametes (1,1), (1,0), (0,1) do not all appear among the cells
             (B11, B10, B01 binary; B11 >= Y_ip + Y_iq - 1, B10 >= Y_ip - Y_iq, B01 >= Y_iq - Y_ip; B11+B10+B01 <= 2).
Identical read columns are merged and each is weighted by how many mutations share it: in an optimal Y they can always
share one column, so the optimum is unchanged (sublines24: 1,689 columns -> 108).
HiGHS reports "Optimal" once the relative gap is within mip_rel_gap (default 1e-4); --mip-rel-gap 0 demands gap 0 (up to
HiGHS's absolute gap tolerance, 1e-6). The best value found, the proven upper bound and the gap are printed.

Input: cells x mutations TSV (first column = cell names), entries 0 / 1 / 3 (3 = missing).
Writes <out>.corrected.tsv (optimal genotypes), <out>.nwk and <out>.clades.txt when --out is given.

Run (from the repository's main folder):
  python PhISCS/exact_ml_tree.py --input data/sublines24_input.tsv --fp 0.0001 --fn 0.075 --mip-rel-gap 0
      --best data/sublines24_reference.tsv --out results/sublines24_phiscs   [--time-limit 3600]
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


def solve(raw, fp, fn, time_limit, mip_rel_gap=None):
    """Optimal conflict-free Y for the distinct read columns of raw (cells x distinct columns) and the solver's report."""
    V = raw.to_numpy()
    U, counts = np.unique(V, axis=1, return_counts=True)
    n, k = U.shape
    on = np.where(U == 1, np.log10(1 - fn), np.where(U == 0, np.log10(fn), 0.0))     # Y = 1
    off = np.where(U == 1, np.log10(fp), np.where(U == 0, np.log10(1 - fp), 0.0))    # Y = 0
    w = (on - off) * counts[None, :]
    const = float((off * counts[None, :]).sum())
    pairs = [(p, q) for p in range(k) for q in range(p + 1, k)]
    nY, nB = n * k, 3 * len(pairs)
    y = lambda i, p: i * k + p
    rows, cols, vals, lo, hi = [], [], [], [], []
    r = 0
    for t, (p, q) in enumerate(pairs):
        b11, b10, b01 = nY + 3 * t, nY + 3 * t + 1, nY + 3 * t + 2
        for i in range(n):
            for coef_p, coef_q, b, bound in ((1, 1, b11, 1), (1, -1, b10, 0), (-1, 1, b01, 0)):
                rows += [r, r, r]; cols += [y(i, p), y(i, q), b]; vals += [coef_p, coef_q, -1]
                lo.append(-np.inf); hi.append(bound); r += 1
        rows += [r, r, r]; cols += [b11, b10, b01]; vals += [1, 1, 1]; lo.append(-np.inf); hi.append(2); r += 1
    A = coo_matrix((vals, (rows, cols)), shape=(r, nY + nB)).tocsr()
    c = np.concatenate([-w.ravel(), np.zeros(nB)])
    t0 = time.time()
    res = milp(c, constraints=LinearConstraint(A, lo, hi), integrality=np.ones(nY + nB), bounds=Bounds(0, 1),
               options={"time_limit": time_limit, "disp": False, **({} if mip_rel_gap is None else {"mip_rel_gap": mip_rel_gap})})
    info = {"columns": k, "pairs": len(pairs), "variables": nY + nB, "constraints": r, "seconds": round(time.time() - t0, 1),
            "status": res.message, "status code": int(res.status), "nodes": getattr(res, "mip_node_count", None)}
    if res.x is None:
        return None, info
    Y = np.round(res.x[:nY]).reshape(n, k).astype(int)
    info["best_found"] = const - float(res.fun)
    info["upper_bound"] = const - float(res.mip_dual_bound) if res.mip_dual_bound is not None else None
    info["solver objective"], info["solver dual bound"] = float(res.fun), res.mip_dual_bound
    info["solver relative gap"] = getattr(res, "mip_gap", None)
    return Y, info


def clades_of(genotypes):
    """Cell sets of a 0/1 genotype matrix's columns, without single cells and the root."""
    n = genotypes.shape[0]
    return {frozenset(genotypes.index[genotypes[m] == 1]) for m in genotypes.columns if 1 < (genotypes[m] == 1).sum() < n}


def newick(cells, clades):
    """Newick string of the tree with these clades (nodes with more than two children allowed)."""
    def node(s):
        if len(s) == 1:
            return next(iter(s))
        inner = [c for c in clades if c < s]
        children = [c for c in inner if not any(c < o for o in inner)]
        children += [frozenset([x]) for x in s if not any(x in c for c in children)]
        return "(" + ",".join(node(c) for c in sorted(children, key=lambda c: (-len(c), sorted(c)))) + ")"
    return node(frozenset(cells)) + ";"


def main():
    a = argparse.ArgumentParser(description="Exact maximum-likelihood tree by integer programming (PhISCS, K = 0).")
    a.add_argument("--input", type=Path, required=True)
    a.add_argument("--fp", type=float, required=True); a.add_argument("--fn", type=float, required=True)
    a.add_argument("--best", type=Path, default=None, help="a reference tree as a corrected matrix (optional)")
    a.add_argument("--time-limit", type=float, default=3600)
    a.add_argument("--mip-rel-gap", type=float, default=None, help="HiGHS relative gap tolerance (default 1e-4)")
    a.add_argument("--out", type=Path, default=None, help="prefix for the optimal tree (.corrected.tsv, .nwk, .clades.txt)")
    args = a.parse_args()
    raw = pd.read_csv(args.input, sep="\t", index_col=0)
    cells = list(raw.index)
    Y, info = solve(raw, args.fp, args.fn, args.time_limit, args.mip_rel_gap)
    print({k: v for k, v in info.items() if k not in ("best_found", "upper_bound")})
    if Y is None:
        return
    print(f"best tree found: log10 likelihood {info['best_found']:,.1f}; proven upper bound {info['upper_bound']:,.1f} "
          f"(gap {info['upper_bound'] - info['best_found']:.2f})")
    print(f"exact: best found {info['best_found']!r}, upper bound {info['upper_bound']!r}, "
          f"gap {info['upper_bound'] - info['best_found']!r} log10 units")
    _, inv = np.unique(raw.to_numpy(), axis=1, return_inverse=True)            # same column order as solve()
    G = pd.DataFrame(Y[:, inv.ravel()], index=raw.index, columns=raw.columns)
    clades = clades_of(G)
    assert all(x <= z or z <= x or not (x & z) for x in clades for z in clades), "solution has conflicts"
    lab = lambda s: "{" + ",".join(sorted(s, key=lambda x: (len(x), x))) + "}"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        G.to_csv(f"{args.out}.corrected.tsv", sep="\t")
        tree = newick(cells, clades)
        Path(f"{args.out}.nwk").write_text(tree + "\n")
        Path(f"{args.out}.clades.txt").write_text("\n".join(lab(c) for c in sorted(clades, key=lambda c: (-len(c), lab(c)))) + "\n")
        print(f"optimal tree ({len(clades)} clades): {tree}")
        print(f"wrote {args.out}.corrected.tsv, {args.out}.nwk, {args.out}.clades.txt")
    if args.best is not None:
        ref = clades_of(pd.read_csv(args.best, sep="\t", index_col=0).loc[raw.index, raw.columns])
        print(f"reference tree: clades with mutations in the optimal solution {len(clades)}, shared with the reference {len(clades & ref)}")
        print("  only in the optimal solution:", [lab(c) for c in clades - ref])
        print("  reference clades without mutations in the optimal solution:", len(ref - clades))


if __name__ == "__main__":
    main()
