"""Paper figure on run time and oracle accuracy.
Left: run time of LA (main oracle: oracle answers + tree + placement) and ScisTree2 by mutation count at 20 cells, on
the clean ISA timing subset (timing.py clean-isa: one dataset at a time with all threads), mean line and range band over
the 4 datasets per point. Right: the main oracle's triplet accuracy on its training data vs held-out data from the same
simulators, by cell count (training/train_accuracy.py), one small violin panel per cell-count bin.
Reads results/timing_clean-isa.csv and results/train-accuracy.csv; writes results/figures/runtime_oracle.png / .pdf.

Run: python code/fig_runtime_oracle.py [--cells 20]
"""
import argparse

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec

from figures_clean_isa import BLUE, ORANGE, OUT, RESULTS, violins

AQUA, VIOLET = "#1baf7a", "#6250d6"
METHODS = [("ScisTree2", ORANGE, "ScisTree2 s"), ("LA (ours)", BLUE, "combined s")]
BINS = ([0, 25, 40, 70, 250], ["16–25 cells", "26–40 cells", "41–70 cells", "71–200 cells"])
DATA = [("Train", VIOLET, "train"), ("Held out", AQUA, "held out")]


def main():
    a = argparse.ArgumentParser(description="Run time and oracle train vs held-out accuracy figure.")
    a.add_argument("--cells", type=int, default=20)
    n = a.parse_args().cells
    t = pd.read_csv(RESULTS / "timing_clean-isa.csv")
    acc = pd.read_csv(RESULTS / "train-accuracy.csv")
    acc["bin"] = pd.cut(acc.cells, BINS[0], labels=BINS[1])
    fig = plt.figure(figsize=(14, 3.8))
    gs = GridSpec(1, 6, figure=fig, width_ratios=[5, 0.35, 1.4, 1.4, 1.4, 1.4], wspace=0.45)
    ax = fig.add_subplot(gs[0])
    g = t[t.cells == n]
    for name, col, c in METHODS:
        s = g.groupby("mutations")[c]
        ax.plot(s.mean().index, s.mean(), ls="--", color=col, lw=1.8, label=name)
        ax.fill_between(s.mean().index, s.min(), s.max(), color=col, alpha=0.25, lw=0)
    ax.set_xlabel("Mutation count"); ax.set_ylabel("Run time (seconds)"); ax.set_ylim(bottom=0)
    ax.set_title(f"Performance comparison varying mutation count (n = {n} cells)", pad=8)
    ax.legend(loc="upper left", frameon=True, framealpha=0.92, edgecolor="#d0cfca")
    ax.grid(axis="y", color="#e4e3df", lw=0.6); ax.set_axisbelow(True)
    for i, label in enumerate(BINS[1]):
        ax = fig.add_subplot(gs[2 + i])
        b = acc[acc.bin == label]
        violins(ax, [f"{name}\n({(b.data == key).sum()})" for name, _, key in DATA],
                [([col for _, col, _ in DATA], [b.loc[b.data == key, "accuracy"] for _, _, key in DATA])], width=0.6)
        ax.set_title(label, pad=6)
        if i == 0:
            ax.set_ylabel("Triplet accuracy")
    fig.text(0.735, 0.955, "XGBoost oracle accuracy on training vs held-out data, varying cell count", ha="center",
             va="bottom", fontsize=9.5)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"runtime_oracle.{ext}", bbox_inches="tight")
    print(f"wrote {OUT / 'runtime_oracle'}.png / .pdf")


if __name__ == "__main__":
    main()
