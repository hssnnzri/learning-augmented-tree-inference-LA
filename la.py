"""LA: tree building from the oracle's answers, and mutation placement.

build_tree(raw, probs, fp, fn) runs, in order:
  1. top-down: the paper's Algorithm 1 / Proposition 2.1 split (every cell tried as the base cell), down to groups of
     fewer than log2 n cells (or a group no cell can split)
  2. small groups: each group's subtree by exhaustive search over every rooted binary tree on its cells, scored by
     the WHOLE tree's likelihood (each mutation on its best node, missing reads skipped) with the rest fixed; one group
     at a time, repeated until no subtree changes (groups over MAX_EXACT cells go to ScisTree2)
  3. agreement splitting: each split re-chosen top-down by moving single cells or whole clades across it while it
     agrees better with the oracle's probabilities; small groups then re-solved as in 2
  4. global search: subtree prune-and-regraft moves while the whole tree agrees better; small groups re-solved
Placement on the finished tree (never changes the tree):
  place(raw, clades)            each mutation on its maximum-likelihood edge (the paper's post-processing)
  place_loss_aware(raw, clades) also on an edge minus one clade below it (a loss in a group of cells, penalty 3 log10)
  add_cell_losses(...)          then keeps a cell's 0s along a chromosome stretch where it reads 0 far above its own
                                dropout (a loss in one cell)
scistree2_tree(raw, fp, fn) is ScisTree2 alone, for comparison. raw: cells x mutations DataFrame with 0 / 1 / 3.
Same code as the experiments (la_augmented, pipeline_V2, pipeline_loss_aware_V2, agreement_split_V2,
global_search_V2, single_cell_losses_V2, check_seeds_global_V2.both), so the trees are identical.
"""
import itertools
from functools import lru_cache

import numpy as np
import pandas as pd
import scistree2 as s2
from ete3 import Tree

from trees import TreeNode, build_tree_from_clades, tree, tree_clades

LOSS_PENALTY = 3.0              # log10 cost of a loss in a group of cells
CELL_LOSS_THRESHOLD = 10.0      # log10 evidence needed for a loss in one cell
MAX_EXACT = 7                   # largest small group searched exhaustively (10,395 trees); bigger ones go to ScisTree2
MAX_VOTE_DP = 12                # largest small group searched by the vote DP
EPSILON = 0.01                  # the counterpart test's epsilon


# ---------------------------------------------------------------- likelihood pieces

def read_scores(V, fp, fn):
    """Per cell and mutation: log10 gain of 'carries it' over 'does not' (D), and the score when no cell carries it."""
    ones, zeros = (V == 1).astype(float), (V == 0).astype(float)
    on = ones * np.log10(1 - fn) + zeros * np.log10(fn)
    off = ones * np.log10(fp) + zeros * np.log10(1 - fp)
    return on - off, off.sum(0)


@lru_cache(maxsize=None)
def tree_shapes(k):
    """All rooted binary trees on positions 0..k-1, each as a tuple of internal clades (frozensets of positions)."""
    def build(cells):
        if len(cells) == 1:
            return [()]
        first, rest = cells[0], cells[1:]
        out = []
        for r in range(len(rest)):
            for comb in itertools.combinations(rest, r):
                left = (first,) + comb
                right = tuple(c for c in rest if c not in comb)
                for tl in build(left):
                    for tr in build(right):
                        out.append(tl + tr + (frozenset(cells),))
        return out
    return build(tuple(range(k)))


# ---------------------------------------------------------------- ScisTree2 (baseline, and groups too big to search)

def _genotype_probs(values, fp, fn):
    """P(true genotype is 0 | read): read 0, read 1, or 0.5 when missing."""
    return np.where(values == 1, fp / (fp + 1 - fn), np.where(values == 0, (1 - fp) / (1 - fp + fn), 0.5))


def _attach(node, ete3_node):
    if len(node.cells) == 1:
        return
    left, right = ete3_node.children[0], ete3_node.children[1]
    node.left_child = TreeNode(left.get_leaf_names())
    node.right_child = TreeNode(right.get_leaf_names())
    _attach(node.left_child, left); _attach(node.right_child, right)


def run_scistree2(mat, node, rem_muts, fp, fn):
    """ScisTree2 on node's cells (missing reads as probability 0.5), attached under node; overwrites the group's entries
    in mat and removes the mutations it used from rem_muts (as the repo's run_scis2)."""
    if len(node.cells) <= 1:
        return
    cells = sorted(node.cells)
    present = (mat.loc[cells, rem_muts] == 1).sum(axis="index")
    muts = present.index[present > 0].values
    place = True
    if len(muts) == 0:
        muts, place = mat.columns.values, False
    probs = _genotype_probs(mat.loc[cells, muts].to_numpy(), fp, fn)
    out, imputed, _ = s2.ScisTree2(threads=10).infer(s2.probability.from_probs(probs.T, cells, list(muts)))
    if place:
        mat.loc[cells, muts] = np.asarray(imputed).T
        used = set(muts)
        rem_muts[:] = [m for m in rem_muts if m not in used]
    _attach(node, Tree(out.output(), format=9))


def scistree2_tree(raw, fp, fn):
    """ScisTree2 alone on the whole matrix: internal clades (2 to n-1 cells)."""
    root = TreeNode(raw.index)
    run_scistree2(raw.copy(), root, list(raw.columns), fp, fn)
    return {c for c in tree_clades(root, set()) if 1 < len(c) < len(raw)}


# ---------------------------------------------------------------- small groups

def _spr_moves(t, k):
    """Every tree one subtree prune-and-regraft away. t: internal clades as bitmasks of k positions, root included."""
    full = 2 ** k - 1
    leaves = [1 << i for i in range(k)]
    for s in [u for u in t if u != full] + leaves:
        p = min((u for u in t if u & s == s and u != s), key=lambda u: bin(u).count("1"))
        sub = {u for u in t if u & s == u}
        rest = {u for u in t if u & s == 0} | {u & ~s for u in t if u & s == s and u not in (s, p)}
        for target in rest | {l for l in leaves if not l & s}:
            new = frozenset(sub | {(u | s) if (u & target == target and u != target) else u for u in rest} | {target | s})
            if new != t:
                yield new


def vote_dp_search(D, base, floor, counts, k, starts=()):
    """A DP over subsets of a small group's k cells that aims at the exhaustive search's tree (the --small-groups
    vote-dp option): each mutation votes for the cells reading 1, weighted by how far it beats its floor there; the
    tree with the most votes is cleaned up by prune-and-regraft moves (also from `starts`), then by an escape step.
    Returns (internal clades as bitmasks of positions, root included; score above floor)."""
    full = 2 ** k - 1
    pos = D > 0
    top = base + np.where(pos, D, 0.0).sum(0)
    keep = top > floor + 1e-9
    D, base, floor, counts, top = D[:, keep], base[keep], floor[keep], counts[keep], top[keep]
    vote = pos[:, keep].T.astype(np.int64) @ (1 << np.arange(k, dtype=np.int64))
    w = np.bincount(vote, weights=(top - floor) * counts, minlength=2 ** k)
    F, arg = np.zeros(2 ** k), np.zeros(2 ** k, dtype=np.int64)
    for C in sorted(range(1, 2 ** k), key=lambda c: bin(c).count("1")):
        if C & (C - 1) == 0:
            continue
        low = C & -C; rest = C ^ low; val, sub = -np.inf, rest
        while True:
            S = low | sub
            if S != C and F[S] + F[C ^ S] > val:
                val, arg[C] = F[S] + F[C ^ S], S
            if not sub:
                break
            sub = (sub - 1) & rest
        F[C] = val + w[C]
    t, stack = set(), [full]
    while stack:
        C = stack.pop()
        if C & (C - 1):
            t.add(C); stack += [int(arg[C]), C ^ int(arg[C])]
    nodes = {}

    def score(u):
        inner = [v for v in u if v != full]
        for v in inner:
            if v not in nodes:
                nodes[v] = base + D[[i for i in range(k) if v >> i & 1]].sum(0)
        best = np.max([nodes[v] for v in inner], 0) if inner else floor
        return float(((np.maximum(floor, best) - floor) * counts).sum())

    def climb(u):
        cur, cs = u, score(u)
        while True:
            nxt, ns = None, cs
            for m in _spr_moves(cur, k):
                s = score(m)
                if s > ns + 1e-9:
                    nxt, ns = m, s
            if nxt is None:
                return cur, cs
            cur, cs = nxt, ns
    cur, cs = max((climb(frozenset(u)) for u in [t, *starts]), key=lambda x: x[1])
    while True:
        nxt, ns = None, cs
        for m in _spr_moves(cur, k):
            c2, s2_ = climb(m)
            if s2_ > ns + 1e-9:
                nxt, ns = c2, s2_
        if nxt is None:
            return set(cur), cs
        cur, cs = nxt, ns


def _score_above(D, base, floor, t, k):
    inner = [v for v in t if v != 2 ** k - 1]
    if not inner:
        return 0.0
    best = np.max([base + D[[i for i in range(k) if v >> i & 1]].sum(0) for v in inner], 0)
    return float((np.maximum(floor, best) - floor).sum())


def _limit(search):
    return MAX_VOTE_DP if search == "vote-dp" else MAX_EXACT


def solve_group(mat, node, rem_muts, fp, fn, search="exhaustive"):
    """First pass for a small group: the best tree on the group's own reads, attached under node (ScisTree2 above the
    search limit)."""
    if len(node.cells) > _limit(search):
        return run_scistree2(mat, node, rem_muts, fp, fn)
    if len(node.cells) <= 1:
        return
    cells = sorted(node.cells); k = len(cells)
    V = mat.loc[cells].to_numpy()
    V = V[:, (V == 1).any(0)]                              # mutations absent from the group score the same everywhere
    cols, counts = np.unique(V, axis=1, return_counts=True)
    D, base = read_scores(cols, fp, fn)
    if search == "vote-dp":
        floor = np.vstack([base, base + D.sum(0), base + D]).max(0)
        t, _ = vote_dp_search(D, base, floor, counts.astype(float), k)
        clades = {frozenset(cells[i] for i in range(k) if v >> i & 1) for v in t} | {frozenset([c]) for c in cells}
        return build_tree_from_clades(node, clades)
    masks = np.array([[(m >> i) & 1 for i in range(k)] for m in range(2 ** k)], dtype=float)
    S = np.einsum("ki,ij->kj", masks, D) + base[None, :]
    assert np.isfinite(S).all()
    bit = lambda s: sum(1 << i for i in s)
    best_total, best_shape = -np.inf, None
    for shape in tree_shapes(k):
        allowed = {0: 0.0, **{1 << i: 0.0 for i in range(k)}, **{bit(c): 0.0 for c in shape}}
        ms = np.fromiter(allowed, dtype=int)
        total = ((S[ms] - np.fromiter(allowed.values(), dtype=float)[:, None]).max(0) * counts).sum()
        if total > best_total:
            best_total, best_shape = total, shape
    clades = {frozenset(cells[i] for i in c) for c in best_shape} | {frozenset([c]) for c in cells}
    build_tree_from_clades(node, clades)


def refine_small_clades(raw, clades, groups, fp, fn, search="exhaustive", rounds=5):
    """Each small group's subtree re-chosen under the WHOLE tree's likelihood, the rest of the tree fixed; a subtree is
    replaced only by a strictly better one; repeated over the groups until nothing changes.
    Returns (new clade set, number of subtree changes)."""
    cells = list(raw.index); idx = {c: i for i, c in enumerate(cells)}; n = len(cells)
    D, base = read_scores(raw.to_numpy(), fp, fn)
    bit = lambda s: sum(1 << i for i in s)
    clades, changes = set(clades), 0
    for _ in range(rounds):
        changed = 0
        for G in groups:
            k = len(G)
            if k < 3 or k > _limit(search):
                continue
            g = sorted(G, key=idx.get)
            outside = [c for c in clades if not c < G] + [frozenset(cells)] + [frozenset([c]) for c in cells]
            P = np.zeros((len(outside), n))
            for r, c in enumerate(outside):
                P[r, [idx[x] for x in c]] = 1
            fixed = (np.einsum("ki,ij->kj", P, D) + base).max(0)           # best placement outside the group's subtree
            old = {c for c in clades if c < G and len(c) >= 2}
            if search == "vote-dp":
                Dg, full = D[[idx[c] for c in g]], 2 ** k - 1
                now = frozenset(bit(g.index(x) for x in c) for c in old) | {full}
                t, s = vote_dp_search(Dg, base, fixed, np.ones(len(base)), k, [now] if len(now) == k - 1 else [])
                if s > _score_above(Dg, base, fixed, now, k) + 1e-9:
                    clades = (clades - old) | {frozenset(g[i] for i in range(k) if v >> i & 1) for v in t if v != full}
                    changed += 1
                continue
            masks = np.array([[(m >> i) & 1 for i in range(k)] for m in range(2 ** k)], dtype=float)
            inside = np.einsum("ki,ij->kj", masks, D[[idx[c] for c in g]]) + base
            assert np.isfinite(fixed).all() and np.isfinite(inside).all()
            score = lambda inner: float(np.maximum(fixed, inside[inner].max(0)).sum()) if inner else float(fixed.sum())
            best_total = score([bit(g.index(x) for x in c) for c in old])
            best = None
            for shape in tree_shapes(k):
                total = score([bit(c) for c in shape if len(c) < k])
                if total > best_total + 1e-9:
                    best_total, best = total, shape
            if best is not None:
                clades = (clades - old) | {frozenset(g[i] for i in c) for c in best if len(c) < k}
                changed += 1
        changes += changed
        if not changed:
            break
    return clades, changes


# ---------------------------------------------------------------- 1-2. top-down tree and small groups

def table_oracle(probs, cells):
    """oracle(a, b, c) -> the cell (name) the oracle says split away first."""
    trips = list(itertools.combinations(range(len(cells)), 3))
    table = {t: cells[t[x]] for t, x in zip(trips, probs.argmax(1))}
    pos = {c: i for i, c in enumerate(cells)}
    return lambda a, b, c: table[tuple(sorted((pos[a], pos[b], pos[c])))]


def _is_counterpart(cells, u, v, thresh1, thresh2, oracle):
    """v is u's counterpart if v rarely splits away from (u, t) and t rarely splits away from (u, v)."""
    c1 = c2 = 0
    for t in sorted(cells - {u, v}):
        split = oracle(u, v, t)
        if split == v:
            c1 += 1
        if split == t:
            c2 += 1
    return c1 <= thresh1 and c2 <= thresh2


def _top_down(mat, node, cutoff, fp, fn, rem_muts, oracle, small):
    """Proposition 2.1's split, every cell tried as the base cell; groups below the cut-off (or that no cell splits)
    go to small(mat, node, rem_muts, fp, fn)."""
    if len(node.cells) < cutoff:
        small(mat, node, rem_muts, fp, fn)
        return
    best = {}
    for base_cell in sorted(node.cells):
        side = TreeNode([])
        others = node.cells - {base_cell}
        thresh1, thresh2 = (3 / 5 - EPSILON) * len(others), (1 / 6 - EPSILON) * len(others)
        for test_cell in sorted(others):
            if _is_counterpart(node.cells, base_cell, test_cell, thresh1, thresh2, oracle):
                side.cells.add(test_cell)
        best[base_cell] = side
    left = best[max(best, key=lambda b: len(best[b].cells))]
    if len(left.cells) == 0:
        small(mat, node, rem_muts, fp, fn)
        return
    node.left_child, node.right_child = left, TreeNode(node.cells - left.cells)
    _top_down(mat, node.left_child, cutoff, fp, fn, rem_muts, oracle, small)
    _top_down(mat, node.right_child, cutoff, fp, fn, rem_muts, oracle, small)


def top_down_tree(raw, probs, fp, fn, search="exhaustive", whole_tree=True):
    """Steps 1-2. whole_tree=False keeps each small group's first-pass subtree (judged on its own reads only).
    Returns (internal clades, small groups, subtree changes)."""
    n = len(raw); groups = []

    def small(mat, node, rem_muts, fp_, fn_):
        groups.append(frozenset(node.cells))
        solve_group(mat, node, rem_muts, fp_, fn_, search)
    root = TreeNode(raw.index)
    _top_down(raw.copy(), root, float(np.log2(n)), fp, fn, list(raw.columns), table_oracle(probs, list(raw.index)), small)
    clades = {c for c in tree_clades(root, set()) if 1 < len(c) < n}
    changes = 0
    if whole_tree:
        clades, changes = refine_small_clades(raw, clades, groups, fp, fn, search)
        clades = {c for c in clades if 1 < len(c) < n}
    return clades, groups, changes


# ---------------------------------------------------------------- 3-4. agreement with the oracle's probabilities

def split_tensor(probs, n):
    """T[a, b, c]: probability that c split away first in the triplet {a, b, c} (0 when not 3 different cells)."""
    T = np.zeros((n, n, n))
    for (x, y, z), p in zip(itertools.combinations(range(n), 3), probs):
        T[y, z, x] = T[z, y, x] = p[0]; T[x, z, y] = T[z, x, y] = p[1]; T[x, y, z] = T[y, x, z] = p[2]
    return T


def regret_tensor(T):
    """The regret agreement splitting uses (as in every reported result): R[x, y, z] = log P(z split away) minus the
    larger of log P(z) and log P(y) (the intended maximum is over all three answers, regret_tensor_exact). It is 0
    exactly when z is the oracle's top answer, so a perfect oracle moves nothing."""
    L = np.log(np.maximum(T, 1e-3))
    best = np.maximum(np.maximum(L, L.transpose(0, 2, 1)), L.transpose(1, 2, 0))
    n = len(T); idx = np.arange(n)
    R = L - best
    R[idx, idx, :] = 0; R[idx, :, idx] = 0; R[:, idx, idx] = 0
    return R


def regret_tensor_exact(T):
    """R[x, y, z] = log P(z split away | x, y, z) - log of the largest of the three answers (P floored at 0.001)."""
    L = np.log(np.maximum(T, 1e-3))
    best = np.maximum(np.maximum(L, L.transpose(0, 2, 1)), L.transpose(2, 0, 1))
    n = len(T); idx = np.arange(n)
    R = L - best
    R[idx, idx, :] = 0; R[idx, :, idx] = 0; R[:, idx, idx] = 0
    return R


def _improve(T, S, side, groups=(), max_moves=200):
    """Move single cells, or whole groups lying on one side, across the split while its score rises."""
    sub = T[np.ix_(S, S, S)]

    def score(U):
        V = 1 - U
        inA = np.einsum("kx,ky,xyz->kz", U, U, sub); inB = np.einsum("kx,ky,xyz->kz", V, V, sub)
        return 0.5 * ((inA * V).sum(1) + (inB * U).sum(1))
    for _ in range(max_moves):
        cands = []
        for i in range(len(S)):
            c = side.copy(); c[i] = not c[i]; cands.append(c)
        for g in groups:
            if g.sum() < 2:
                continue
            if (side[g]).all():
                c = side.copy(); c[g] = False; cands.append(c)
            elif (~side[g]).all():
                c = side.copy(); c[g] = True; cands.append(c)
        cands = [c for c in cands if 0 < c.sum() < len(S)]
        if not cands:
            break
        U = np.array(cands, dtype=float)
        sc = score(U); cur = float(score(side.astype(float)[None, :])[0])
        i = int(np.argmax(sc))
        if sc[i] <= cur + 1e-9:
            break
        side = cands[i]
    return side


def _induced_split(S, start_nodes):
    """Top split of the starting tree restricted to the cells S."""
    Sset = set(S)
    for node in sorted(start_nodes, key=len):
        if Sset <= node:
            kids = [k for k in start_nodes if k < node and not any(k < j < node for j in start_nodes)]
            parts = [Sset & k for k in kids]
            if sum(1 for p in parts if p) >= 2:
                first = next(p for p in parts if p)
                return np.array([c in first for c in S])
    return np.array([i < len(S) // 2 for i in range(len(S))])


def agreement_tree(raw, probs, start_clades, fp, fn, search="exhaustive"):
    """Step 3. Returns (internal clades, {"moved": cells moved over all splits})."""
    cells = list(raw.index); n = len(cells); pos = {c: i for i, c in enumerate(cells)}
    cutoff = float(np.log2(n))
    T = regret_tensor(split_tensor(probs, n))
    start = {frozenset(pos[c] for c in v) for v in start_clades} | {frozenset([i]) for i in range(n)} | {frozenset(range(n))}
    clades, small, moved = set(), [], 0

    def split(S):
        nonlocal moved
        if len(S) < cutoff:
            small.append(frozenset(S))
            clades.update(g for g in (c & frozenset(S) for c in start) if len(g) > 1)
            return
        side0 = _induced_split(S, start)
        groups, seen = [], set()
        for c in start:
            g = frozenset(c) & frozenset(S)
            if 1 < len(g) < len(S) and g not in seen:
                seen.add(g); groups.append(np.array([x in g for x in S]))
        side = _improve(T, np.array(S), side0, groups)
        moved += int((side != side0).sum())
        A, B = [c for c, s in zip(S, side) if s], [c for c, s in zip(S, side) if not s]
        for part in (A, B):
            if len(part) > 1:
                clades.add(frozenset(part)); split(part)

    split(list(range(n)))
    names = {frozenset(cells[i] for i in c) for c in clades if 1 < len(c) < n}
    groups = [frozenset(cells[i] for i in g) for g in small]
    names, _ = refine_small_clades(raw, names, groups, fp, fn, search)
    return {c for c in names if 1 < len(c) < n}, {"moved": moved}


def _nodes(clades, n):
    """Nodes (largest first), parent index (-1 for the root), membership matrix (nodes x cells)."""
    nodes = sorted(clades, key=lambda c: (-len(c), sorted(c)))
    parent = np.full(len(nodes), -1)
    for i, c in enumerate(nodes):
        for j in range(i - 1, -1, -1):
            if c < nodes[j]:
                parent[i] = j
                break
    M = np.zeros((len(nodes), n), dtype=bool)
    for i, c in enumerate(nodes):
        M[i, list(c)] = True
    return nodes, parent, M


def _best_move(R, clades, n):
    """The best subtree prune-and-regraft move: (gain, new clade set as index sets), or (1e-9, None)."""
    nodes, parent, M = _nodes(clades, n)
    N = len(nodes)
    sib = np.full(N, -1)
    for i in range(N):
        kids = np.flatnonzero(parent == i)
        if len(kids) == 2:
            sib[kids[0]], sib[kids[1]] = kids[1], kids[0]
    best = (1e-9, None)
    for v in range(1, N):
        p, s0 = parent[v], sib[v]
        g = parent[p]
        S = M[v]; Si = np.flatnonzero(S)
        D = R[Si].sum(0) - R[:, :, Si].sum(2)
        inside = ~(M & ~S).any(1)
        keep = ~inside; keep[p] = False
        Mr = (M & ~S).astype(float)
        par = parent.copy(); par[s0] = g
        sb = sib.copy()
        if g >= 0:
            sb[s0] = sib[p]; sb[sib[p]] = s0
        else:
            sb[s0] = -1
        idx = np.flatnonzero(keep)
        gi = idx[sb[idx] >= 0]
        gain = np.zeros(N)
        with np.errstate(all="ignore"):
            gain[gi] = ((Mr[gi] @ D) * Mr[sb[gi]]).sum(1)
        assert np.isfinite(gain).all()
        cum = np.zeros(N)
        for u in idx:
            cum[u] = cum[par[u]] + gain[u] if par[u] >= 0 else 0.0
        t = idx[np.argmax(cum[idx])]
        d = cum[t] - cum[s0]
        if d > best[0] and t != s0:
            Sset, anc, u = nodes[v], set(), t
            while par[u] >= 0:
                u = par[u]; anc.add(u)
            new = {(nodes[u] - Sset) | (Sset if u in anc else frozenset()) for u in idx}
            new.add((nodes[t] - Sset) | Sset)
            new |= {nodes[u] for u in np.flatnonzero(inside)}
            best = (d, new)
    return best


def global_search(raw, probs, start_clades, fp, fn, search="exhaustive", max_moves=500):
    """Step 4. Returns (internal clades, {"moves": n, "gain": total agreement gained})."""
    cells = list(raw.index); n = len(cells); pos = {c: i for i, c in enumerate(cells)}
    R = regret_tensor_exact(split_tensor(probs, n))
    clades = {frozenset(pos[c] for c in v) for v in start_clades} | {frozenset([i]) for i in range(n)} | {frozenset(range(n))}
    moves, total = 0, 0.0
    for _ in range(max_moves):
        d, new = _best_move(R, clades, n)
        if new is None:
            break
        clades, moves, total = new, moves + 1, total + d
    names = {frozenset(cells[i] for i in c) for c in clades if 1 < len(c) < n}
    cutoff = float(np.log2(n))
    groups = [c for c in names | {frozenset(cells)} if len(c) < cutoff and len(c) >= 3
              and not any(c < d and len(d) < cutoff for d in names)]
    names, _ = refine_small_clades(raw, names, groups, fp, fn, search)
    return {c for c in names if 1 < len(c) < n}, {"moves": moves, "gain": round(total, 3)}


def build_tree(raw, probs, fp, fn, search="exhaustive", details=False):
    """LA's tree (steps 1-4) from the oracle's probabilities (features.predict). Returns internal clades (2 to n-1 cells),
    or (clades, per-step information) with details=True."""
    first, groups, changes = top_down_tree(raw, probs, fp, fn, search)
    agreed, a = agreement_tree(raw, probs, first, fp, fn, search)
    final, g = global_search(raw, probs, agreed, fp, fn, search)
    if not details:
        return final
    return final, {"small groups": sorted(len(x) for x in groups), "small-group changes": changes,
                   "agreement moves": a["moved"], "global moves": g["moves"], "top-down clades": first,
                   "agreement clades": agreed}


# ---------------------------------------------------------------- placement on the finished tree

def place(raw, clades, fp, fn):
    """Every mutation on its maximum-likelihood node (root, clade or leaf), missing reads skipped. Exact ties go to the
    first node in the tree's pre-order, which follows Python's set order (hash seed)."""
    cells, nodes = list(raw.index), []

    def walk(node):
        if node is None or not node.cells:
            return
        nodes.append([c in node.cells for c in cells])
        walk(node.left_child); walk(node.right_child)

    walk(tree(cells, clades))
    P = np.array(nodes, dtype=float)
    E = raw.to_numpy()
    ones, zeros = (E == 1).astype(float), (E == 0).astype(float)
    count = lambda A, B: np.einsum("ki,ij->kj", A, B)
    L = (count(P, ones) * np.log10(1 - fn) + count(P, zeros) * np.log10(fn)
         + count(1 - P, ones) * np.log10(fp) + count(1 - P, zeros) * np.log10(1 - fp))
    assert np.isfinite(L).all()
    return pd.DataFrame(P[L.argmax(axis=0)].T.astype(int), index=raw.index, columns=raw.columns)


def _loss_candidates(cells, clades):
    """Placement patterns: every node, plus node-minus-subclade losses. Returns (keys, patterns x cells, penalties)."""
    idx = {c: i for i, c in enumerate(cells)}
    internal = sorted((c for c in clades if len(c) >= 2), key=lambda c: (len(c), sorted(idx[x] for x in c)))
    nodes = list(internal) + [frozenset([c]) for c in cells]
    pats = {frozenset(v): 0.0 for v in nodes}
    for v in internal:
        for u in internal:
            if u < v and len(u) >= 2 and len(v - u) >= 1:
                pats.setdefault(frozenset(v - u), LOSS_PENALTY)
    keys = list(pats)
    P = np.zeros((len(keys), len(cells)))
    for k, s in enumerate(keys):
        P[k, [idx[c] for c in s]] = 1
    return keys, P, np.array([pats[k] for k in keys])


def place_loss_aware(raw, clades, fp, fn):
    """Each mutation on one node or on a node minus one clade below it that lost it (penalty LOSS_PENALTY); fixed
    candidate order, so ties break the same way every run. Returns (corrected matrix, losses used, penalized log10
    score, chosen patterns)."""
    cells = list(raw.index)
    keys, P, pen = _loss_candidates(cells, set(clades) | {frozenset(cells)})
    D, base = read_scores(raw.to_numpy(), fp, fn)
    S = np.einsum("ki,ij->kj", P, D) + base[None, :] - pen[:, None]
    best = S.argmax(0)
    assert np.isfinite(S).all()
    corrected = pd.DataFrame(P[best].T.astype(int), index=cells, columns=raw.columns)
    return corrected, int((pen[best] > 0).sum()), float(S.max(0).sum()), [keys[k] for k in best]


def _cell_dropout(V, carried, chrom):
    """Each cell's dropout: median over chromosomes (10+ reads) of its share of 0s among the mutations it carries."""
    out = np.empty(len(V))
    for c in range(len(V)):
        rates = [(V[c, s] == 0).mean() for ch in np.unique(chrom)
                 for s in [carried[c] & (chrom == ch) & (V[c] != 3)] if s.sum() >= 10]
        out[c] = np.median(rates) if rates else 0.15
    return np.clip(out, 0.005, 0.95)


def _best_stretch(z, fn):
    """z: 0/1 (1 = read 0) in genome order. Stretch [a, b) maximizing the binomial log10 likelihood ratio of a fitted
    loss share p > fn against fn. Returns (evidence, a, b)."""
    Z = np.concatenate([[0], np.cumsum(z)])
    N = np.arange(len(z) + 1)
    k = Z[None, :] - Z[:, None]; n = N[None, :] - N[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        p = k / n
        llr = (np.where(k > 0, k * np.log10(p / fn), 0.0)
               + np.where(n - k > 0, (n - k) * np.log10((1 - p) / (1 - fn)), 0.0))
    llr = np.where((n > 0) & (p > fn), llr, 0.0)
    a, b = np.unravel_index(np.argmax(llr), llr.shape)
    return float(llr[a, b]), a, b


def add_cell_losses(raw, corrected, chrom, pos, fp, fn, threshold=CELL_LOSS_THRESHOLD):
    """One-cell losses on top of place_loss_aware's matrix. chrom, pos: chromosome and position of each mutation.
    Returns (new matrix, score gain, one-cell losses as dicts)."""
    V, M = raw.to_numpy(), corrected.to_numpy().copy()
    chrom, pos, gain, events = np.asarray(chrom), np.asarray(pos), 0.0, []
    carried = M == 1
    fn_cell = _cell_dropout(V, carried, chrom)
    stretches = []
    for c in range(len(V)):
        for ch in np.unique(chrom):
            cols = np.flatnonzero(carried[c] & (chrom == ch) & (V[c] != 3))
            if len(cols) == 0:
                continue
            cols = cols[np.argsort(pos[cols], kind="stable")]
            ev, a, b = _best_stretch((V[c, cols] == 0).astype(int), fn_cell[c])
            if ev > 0:
                seg = cols[a:b]
                stretches.append((ev, c, seg[V[c, seg] == 0], seg[V[c, seg] == 1]))
    for ev, c, zeros, ones in stretches:
        if ev > threshold:
            M[c, zeros] = 0
            gain += len(zeros) * np.log10((1 - fp) / fn) - threshold
            events.append({"cell": str(raw.index[c]), "chrom": str(chrom[zeros[0]]), "zeros_kept": int(len(zeros)),
                           "ones_inside": int(len(ones)), "evidence": round(float(ev), 1)})
    return pd.DataFrame(M, index=corrected.index, columns=corrected.columns), gain, events


def corrections(mat, raw, fp, fn):
    """FP corrections (read 1, corrected 0), FN corrections (read 0, corrected 1), log10 likelihood; missing skipped."""
    d, e = mat.loc[raw.index, raw.columns].to_numpy().astype(int), raw.to_numpy().astype(int)
    seen = e != 3
    k = {name: int(((d == a) & (e == b) & seen).sum()) for name, (a, b) in
         {"kept_0": (0, 0), "fp": (0, 1), "fn": (1, 0), "kept_1": (1, 1)}.items()}
    lik = k["kept_0"] * np.log10(1 - fp) + k["fp"] * np.log10(fp) + k["fn"] * np.log10(fn) + k["kept_1"] * np.log10(1 - fn)
    return {"fp_corrections": k["fp"], "fn_corrections": k["fn"], "log10_likelihood": round(float(lik), 1)}
