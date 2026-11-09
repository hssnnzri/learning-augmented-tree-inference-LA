"""Paper figures on the clean ISA test set, for ScisTree2, LA with the XGBoost (main) oracle and LA with the 95%
synthetic oracle (no SCITE: not rerun). Violins drawn like the paper's other figures (median bar, range whiskers, mean
dot, grey bands between groups).
  tree_accuracy.*     fig:noisy:rf   true clades recovered (RF accuracy) by cell count (left, mutation counts pooled)
                                     and by mutation count (right, cell counts pooled)
  mutation_metrics.*  Figs. 6-7      genotype, ancestor-descendant and different-lineage accuracy by cell count (top)
                                     and by mutation count (bottom) (ScisTree2: each mutation on its most likely edge;
                                     LA: placement with losses)
Reads results/old-paper.csv and results/noisy-oracle-0.05.csv; writes results/figures/<name>.png / .pdf.

Run: python figures_clean_isa.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = RESULTS / "figures"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"          # same colours as the paper's other figures
INK, MUTED, BAND = "#0b0b0b", "#52514e", "#ededeb"
SERIES = [("ScisTree2", ORANGE, "ScisTree2"), ("LA, XGBoost oracle", BLUE, "combined"),
          ("LA, 95% synthetic oracle", AQUA, "noisy")]
METRICS = [("genotypes", "Genotype accuracy"), ("ancestor-descendant", "Ancestor-descendant accuracy"),
           ("different-lineage", "Different-lineage accuracy")]
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "legend.fontsize": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "savefig.dpi": 200,
                     "font.family": "DejaVu Sans"})


def violins(ax, labels, series, width=0.8):
    """One violin per (group, series): series = [(colour, [values per group]), ...]; colour may be a list, one per
    group. Median bar, range whiskers and a mean dot; grey bands on every other group."""
    k = len(series); w = width / k
    rng = np.random.default_rng(0)
    for i in range(len(labels)):
        if i % 2:
            ax.axvspan(i - 0.5, i + 0.5, color=BAND, zorder=0, lw=0)
        for j, (colours, data) in enumerate(series):
            col = colours[i] if isinstance(colours, list) else colours
            x = np.asarray(data[i], dtype=float); x = x[~np.isnan(x)]
            pos = i - width / 2 + w * (j + 0.5)
            kde = x if np.ptp(x) > 1e-9 else x + rng.normal(0, 1e-5, len(x))   # all equal: draw a flat sliver
            v = ax.violinplot([kde], positions=[pos], widths=w * 0.9, showmedians=False, showextrema=False)
            for b in v["bodies"]:
                b.set_facecolor(col); b.set_edgecolor(col); b.set_alpha(0.35); b.set_linewidth(0.6)
            ax.vlines(pos, x.min(), x.max(), color=col, lw=1.0, zorder=3)
            ax.hlines([x.min(), x.max()], pos - w * 0.18, pos + w * 0.18, color=col, lw=1.0, zorder=3)
            ax.hlines(np.median(x), pos - w * 0.32, pos + w * 0.32, color=col, lw=1.8, zorder=4)
            ax.plot(pos, x.mean(), "o", ms=3.2, mfc="white", mec=col, mew=1.0, zorder=5)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.tick_params(length=2.5)


def title(vals, key):
    return f"Performance comparison for {', '.join(map(str, vals))} {key}"


def save(fig, name, series=SERIES):
    """One legend above all panels, then PNG and PDF."""
    fig.tight_layout()
    handles = [Patch(facecolor=c, edgecolor=c, alpha=0.5, label=n) for n, c, _ in series]
    handles += [Line2D([], [], color=MUTED, lw=1.8, label="median"),
                Line2D([], [], ls="", marker="o", ms=3.2, mfc="white", mec=MUTED, label="mean"),
                Line2D([], [], color=MUTED, lw=1.0, label="range")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles), frameon=False)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / name}.png / .pdf")


def tree_accuracy(d):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [4, 7]}, sharey=True)
    for ax, key in ((axes[0], "cells"), (axes[1], "mutations")):
        vals = sorted(d[key].unique())
        violins(ax, [f"{v}\n({(d[key] == v).sum()})" for v in vals],
                [(col, [d.loc[d[key] == v, f"{k} clades"] for v in vals]) for _, col, k in SERIES])
        ax.set_xlabel(f"{key.capitalize()[:-1]} count (datasets in brackets)")
        ax.set_title(title(vals, key), pad=8)
    axes[0].set_ylim(0, 1.04); axes[0].set_ylabel("Robinson–Foulds accuracy")
    save(fig, "tree_accuracy")


def mutation_metrics(d):
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.2))
    for row, key in zip(axes, ("cells", "mutations")):
        vals = sorted(d[key].unique())
        for ax, (metric, label) in zip(row, METRICS):
            violins(ax, [f"{v}\n({(d[key] == v).sum()})" for v in vals],
                    [(col, [d.loc[d[key] == v, f"{k} {metric}"] for v in vals]) for _, col, k in SERIES])
            lo = min(d[f"{k} {metric}"].min() for _, _, k in SERIES); pad = 0.03 * (1 - lo)
            ax.set_ylim(lo - pad, 1 + pad)
            ax.set_ylabel(label); ax.set_xlabel(f"{key.capitalize()[:-1]} count (datasets in brackets)")
            ax.set_title(title(vals, key), pad=8)
    save(fig, "mutation_metrics")


def main():
    d = pd.read_csv(RESULTS / "noisy-oracle-0.05.csv")
    cols = [f"{k} {x}" for k in ("combined", "ScisTree2") for x in ["clades"] + [m for m, _ in METRICS]]
    d = d.merge(pd.read_csv(RESULTS / "old-paper.csv")[["job"] + cols], on="job", how="left")
    assert len(d) == 3_360 and d.notna().all().all()
    tree_accuracy(d)
    mutation_metrics(d)
    for metric in ["clades"] + [m for m, _ in METRICS]:
        print(metric); print(d.groupby("cells")[[f"{k} {metric}" for _, _, k in SERIES]].agg(["mean", "median"]).round(3).to_string())


if __name__ == "__main__":
    main()
