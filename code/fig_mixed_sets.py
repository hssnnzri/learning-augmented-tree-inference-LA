"""Preview figure for the mixed test sets: true clades recovered (RF accuracy) by cell count and by mutations per cell,
ScisTree2 vs LA with the XGBoost (main) oracle, in the format of figures_clean_isa.py. Three versions: the general,
recoverable and realistic sets pooled (mixed_sets_pooled.*), the general set alone (mixed_sets_general.*), and every
test set in its own row, with the same bins in every row (all_test_sets.*).
Reads results/<set>.csv; writes results/figures/<name>.png / .pdf.

Run: python code/fig_mixed_sets.py
"""
import matplotlib.pyplot as plt
import pandas as pd

from figures_clean_isa import BLUE, ORANGE, RESULTS, save, violins

SERIES = [("ScisTree2", ORANGE, "ScisTree2"), ("LA, XGBoost oracle", BLUE, "combined")]
CELLS = ([0, 30, 50, 70, 100], ["20–30", "31–50", "51–70", "71–100"])
PER_CELL = ([0, 5, 20, 50, 100, 1e9], ["≤5", "5–20", "20–50", "50–100", ">100"])


def panels(axes, d, what):
    """By cell count (left) and by mutations per cell (right); empty bins left out."""
    d = d.assign(cb=pd.cut(d.cells, CELLS[0], labels=CELLS[1]),
                 mb=pd.cut(d.mutations / d.cells, PER_CELL[0], labels=PER_CELL[1]))
    for ax, key, labels, xlabel in ((axes[0], "cb", CELLS[1], "Cell count"), (axes[1], "mb", PER_CELL[1], "Mutations per cell")):
        groups = [d[d[key] == b] for b in labels]
        keep = [i for i, g in enumerate(groups) if len(g)]
        violins(ax, [f"{labels[i]}\n({len(groups[i])})" for i in keep],
                [(col, [groups[i][f"{k} clades"] for i in keep]) for _, col, k in SERIES])
        ax.set_xlabel(f"{xlabel} (datasets in brackets)")
        ax.set_title(f"{what}: by {xlabel.lower()}", pad=8)
    axes[0].set_ylim(0, 1.04); axes[0].set_ylabel("Robinson–Foulds accuracy")


def figure(d, name, what):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [4, 5]}, sharey=True)
    panels(axes, d, what)
    save(fig, name, SERIES)


def all_sets(sets):
    fig, axes = plt.subplots(len(sets), 2, figsize=(10, 3.0 * len(sets)), gridspec_kw={"width_ratios": [4, 5]},
                             sharey=True)
    for row, (label, d) in zip(axes, sets.items()):
        panels(row, d, f"{label} ({len(d):,} datasets)")
    save(fig, "all_test_sets", SERIES)


def main():
    sets = {t: pd.read_csv(RESULTS / f"{t}.csv") for t in ("general", "recoverable", "realistic")}
    figure(pd.concat(sets.values()), "mixed_sets_pooled", "Three mixed sets (500 datasets)")
    figure(sets["general"], "mixed_sets_general", "General set (200 datasets)")
    names = {"old-paper": "Clean ISA", "loss-grid": "Loss grid", "factor-grid": "Factor grid", "general": "General",
             "recoverable": "Recoverable", "realistic": "Realistic"}
    all_sets({label: pd.read_csv(RESULTS / f"{t}.csv") for t, label in names.items()})


if __name__ == "__main__":
    main()
