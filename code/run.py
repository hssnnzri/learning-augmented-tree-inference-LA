"""Run LA on one observed matrix: oracle answers -> tree -> corrected matrix, with ScisTree2 alone for comparison.

Input: cells x mutations TSV (first column = cell names), entries 0 / 1 / 3 (3 = missing); mutation names
"chrom-pos-ref-alt" (the oracle uses the chromosomes, the one-cell losses the positions).
Placement (the tree is the same either way): default = maximum-likelihood edge, then losses in groups of cells, then
one-cell losses; --no-cell-losses stops after the group losses; --no-loss-aware keeps every mutation on one edge.
Writes <out>.corrected.tsv, <out>.nwk and <out>.summary.json (default <out> = results/<input name>_<oracle>).

Run: python code/run.py --input data/sublines24_input.tsv --fp 0.0001 --fn 0.075 --reference data/sublines24_reference.tsv
     python code/run.py --input data/cells23_input.tsv --fp 0.001 --fn 0.15 --reference data/cells23_reference.tsv
     [--oracle combined|strict|file.ubj] [--small-groups exhaustive|vote-dp] [--no-loss-aware] [--no-cell-losses]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from features import load_oracle, predict
from la import add_cell_losses, build_tree, corrections, place, place_loss_aware, scistree2_tree
from simulate import genome_layout
from trees import matrix_clades, newick, tree, triplet_answers

HERE = Path(__file__).resolve().parent.parent                      # repository root


def placed(raw, clades, fp, fn, loss_aware=True, cell_losses=True):
    """(corrected matrix, extra information) for one tree."""
    if not loss_aware:
        return place(raw, clades, fp, fn), {}
    mat, losses, score, _ = place_loss_aware(raw, clades, fp, fn)
    extra = {"losses_placed": losses, "loss_aware_score": round(score, 1)}
    if cell_losses:
        chrom = np.array([c.split("-")[0] for c in raw.columns]); pos = np.array([int(c.split("-")[1]) for c in raw.columns])
        mat, gain, events = add_cell_losses(raw, mat, chrom, pos, fp, fn)
        extra.update({"one_cell_losses": len(events), "zeros_kept": sum(e["zeros_kept"] for e in events),
                      "loss_aware_score": round(score + gain, 1), "one_cell_loss_calls": events})
    return mat, extra


def main():
    a = argparse.ArgumentParser(description="LA on one observed matrix.")
    a.add_argument("--input", type=Path, required=True)
    a.add_argument("--fp", type=float, required=True); a.add_argument("--fn", type=float, required=True)
    a.add_argument("--oracle", default="combined", help="combined, strict, or a model file")
    a.add_argument("--reference", type=Path, default=None, help="a reference tree as a corrected matrix (optional)")
    a.add_argument("--small-groups", choices=("exhaustive", "vote-dp"), default="exhaustive")
    a.add_argument("--no-loss-aware", action="store_true"); a.add_argument("--no-cell-losses", action="store_true")
    a.add_argument("--out", type=Path, default=None)
    args = a.parse_args()

    raw = pd.read_csv(args.input, sep="\t", index_col=0)
    try:
        codes = genome_layout(raw.columns)[0]
    except (IndexError, ValueError):
        sys.exit("Mutation names must look like 'chrom-pos-ref-alt': the oracle uses the chromosomes.")
    cells, n = list(raw.index), len(raw)
    name = Path(args.oracle).stem if args.oracle not in ("combined", "strict") else args.oracle
    out = args.out or HERE / "results" / f"{args.input.stem}_{name}"
    out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    trips, P = predict(load_oracle(args.oracle), raw.to_numpy(), codes)
    t_oracle = time.perf_counter() - t0
    clades, info = build_tree(raw, P, args.fp, args.fn, args.small_groups, details=True)
    t_tree = time.perf_counter() - t0 - t_oracle
    scis = scistree2_tree(raw, args.fp, args.fn)
    loss_aware, cell_losses = not args.no_loss_aware, not (args.no_loss_aware or args.no_cell_losses)
    mat, extra = placed(raw, clades, args.fp, args.fn, loss_aware, cell_losses)
    scis_mat, scis_extra = placed(raw, scis, args.fp, args.fn, loss_aware, cell_losses)
    plain = corrections(place(raw, clades, args.fp, args.fn), raw, args.fp, args.fn)

    summary = {"input": str(args.input), "oracle": args.oracle, "cells": n, "mutations": raw.shape[1],
               "missing_share": round(float((raw.to_numpy() == 3).mean()), 4), "fp": args.fp, "fn": args.fn,
               "small_groups": args.small_groups, "loss_aware_placement": loss_aware, "one_cell_losses": cell_losses,
               "seconds": {"oracle": round(t_oracle, 3), "tree": round(t_tree, 3)},
               "steps": {k: v for k, v in info.items() if not k.endswith("clades")},
               "LA": {**corrections(mat, raw, args.fp, args.fn), "plain_placement": plain, **extra},
               "ScisTree2 alone": {**corrections(scis_mat, raw, args.fp, args.fn), **scis_extra},
               "same_tree_as_ScisTree2": clades == scis}
    if args.reference is not None:
        ref = pd.read_csv(args.reference, sep="\t", index_col=0).loc[raw.index, raw.columns]
        ref_clades = {c for c in matrix_clades(ref) if 1 < len(c) < n}
        summary["reference"] = str(args.reference)
        summary["LA"]["clades_in_reference"] = f"{len(clades & ref_clades)}/{len(ref_clades)}"
        summary["ScisTree2 alone"]["clades_in_reference"] = f"{len(scis & ref_clades)}/{len(ref_clades)}"
        summary["LA"]["triplet_agreement"] = round(float((P.argmax(1) == triplet_answers(cells, ref_clades, trips)).mean()), 4)

    mat.to_csv(f"{out}.corrected.tsv", sep="\t")
    Path(f"{out}.nwk").write_text(newick(tree(cells, clades)) + ";\n")
    json.dump(summary, open(f"{out}.summary.json", "w"), indent=2)
    print(f"{n} cells x {raw.shape[1]} mutations, {summary['missing_share']:.1%} missing, oracle {args.oracle}")
    for who in ("LA", "ScisTree2 alone"):
        s = summary[who]
        more = f", {s['clades_in_reference']} reference clades" if "clades_in_reference" in s else ""
        more += f", triplet agreement {s['triplet_agreement']:.1%}" if "triplet_agreement" in s else ""
        print(f"  {who:15s}: FP {s['fp_corrections']}, FN {s['fn_corrections']}, log10 likelihood "
              f"{s['log10_likelihood']:,.0f}{more}")
    print(f"  LA plain placement: log10 likelihood {plain['log10_likelihood']:,.1f}; same tree as ScisTree2: "
          f"{summary['same_tree_as_ScisTree2']}; oracle {t_oracle:.2f} s, tree {t_tree:.2f} s")
    print(f"wrote {out}.corrected.tsv, .nwk, .summary.json")


if __name__ == "__main__":
    main()
