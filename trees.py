"""Binary cell lineage trees as nested TreeNodes or as sets of clades (frozensets of cells), and split-away answers."""
import numpy as np


class TreeNode:
    def __init__(self, cells, left_child=None, right_child=None):
        self.cells = set(cells)
        self.left_child = left_child
        self.right_child = right_child


def tree_clades(node, out):
    """Every internal node's cells (root included) as frozensets, added to out."""
    if node.left_child is not None:
        out.add(frozenset(node.cells)); tree_clades(node.left_child, out); tree_clades(node.right_child, out)
    return out


def build_tree_from_clades(node, clades):
    """Build the binary TreeNode hierarchy under node from a laminar family of clades."""
    if len(node.cells) == 1:
        return
    proper = [clade for clade in clades if clade < node.cells]
    children = [clade for clade in proper if not any(clade < other < node.cells for other in proper)]
    unique_children = []
    for child in children:
        if child not in unique_children:
            unique_children.append(child)
    if len(unique_children) != 2:
        raise ValueError(f"Expected two children for clade of size {len(node.cells)}, found {len(unique_children)}")
    node.left_child = TreeNode(unique_children[0])
    node.right_child = TreeNode(unique_children[1])
    build_tree_from_clades(node.left_child, clades)
    build_tree_from_clades(node.right_child, clades)


def tree(cells, clades):
    """TreeNode root of the binary tree with these internal clades (single cells added)."""
    root = TreeNode(cells)
    build_tree_from_clades(root, set(clades) | {frozenset([c]) for c in cells})
    return root


def newick(node):
    if node.left_child is None and node.right_child is None:
        if len(node.cells) != 1:
            raise ValueError(f"Unresolved leaf contains multiple cells: {sorted(node.cells)}")
        return next(iter(node.cells))
    return f"({newick(node.left_child)},{newick(node.right_child)})"


def matrix_clades(genotypes):
    """Clades of a 0/1 genotype matrix (cells x mutations DataFrame): the cells carrying each distinct mutation column."""
    unique = genotypes.loc[:, ~genotypes.T.duplicated(keep="first")]
    return {frozenset(genotypes.index[genotypes[m] == 1]) for m in unique.columns if (genotypes[m] == 1).any()}


def split_away(t, clades_sorted_by_size):
    """Index (0, 1, 2) of the cell in triplet t that split away first: the smallest clade that holds exactly two of the
    three cells decides it. clades_sorted_by_size must include the root."""
    a, b, c = t
    for cl in clades_sorted_by_size:
        inside = (a in cl) + (b in cl) + (c in cl)
        if inside == 2:
            return [a not in cl, b not in cl, c not in cl].index(True)
    raise ValueError("unresolved triplet")


def triplet_answers(cells, clades, trips):
    """Index of the split-away cell for each sorted index triplet, under the tree with these internal clades."""
    pos = {c: i for i, c in enumerate(cells)}
    cl = sorted({frozenset(pos[c] for c in v) for v in clades} | {frozenset(range(len(cells)))}, key=len)
    return np.array([split_away(t, cl) for t in trips])
