#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figures 4, 5 and 6: seasonality of the SOM classification.

Figure 4  Stacked bars of the dry, transitional and wet categories by month.
Figure 5  Bubble chart of month against node, sized and coloured by frequency.
Figure 6  Small multiples: monthly frequency of each node, arranged on the grid.

All three read the BMU table written by export_bmus_by_date.py and share the
categorical palette defined in plot_style.py, which ties them to the anomaly
maps of Figure 3.

Usage
-----
    python fig04_05_06_node_seasonality.py
    python fig04_05_06_node_seasonality.py --csv path/to/table.csv --outdir figures

Author: Gabriela Hernandez
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from plot_style import (ANCHO_1COL, ANCHO_2COL, MM, CMAP_SECUENCIAL,
                        COLORES_CAT, ORDEN_CAT, guardar, colorbar_limpia)


# ============================== CONFIGURATION ==============================

DEFAULT_CSV = os.path.join("som_results_ssim",
                           "som_4x4_pca_bmu-ssim_BMUs_por_fecha.csv")
DEFAULT_OUTDIR = "figuras_paper"

GRID = 4
MES_ORDEN = [10, 11, 12, 1, 2, 3]
MES_NOM = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar"]
MES_LETRA = ["O", "N", "D", "J", "F", "M"]

# Node categories, assigned in Section 3.6 from the sign, extent and relative
# magnitude of the mean anomaly field of each node.
NODOS_SECOS = {(0, 0), (0, 1), (1, 0)}
NODOS_HUMEDOS = {(2, 2), (2, 3), (3, 1), (3, 2), (3, 3)}

ESCALA_BURBUJA = 16     # marker area in points squared per year
ANCHO_F5 = 110 * MM


# ================================ FUNCTIONS ================================

def categorizar(i, j):
    """Return the category of a node: Dry, Wet, or Transitional otherwise."""
    if (i, j) in NODOS_SECOS:
        return "Dry"
    if (i, j) in NODOS_HUMEDOS:
        return "Wet"
    return "Transitional"


def load_table(csv_path):
    """Read the BMU table and derive the month, year and category columns."""
    table = pd.read_csv(csv_path)
    table["fecha"] = pd.to_datetime(table["fecha"])
    table["year"] = table["fecha"].dt.year
    table["month"] = table["fecha"].dt.month
    # A season runs from October of year t to March of year t+1, so the first
    # three calendar months of a year belong to the preceding season.
    table["season_year"] = np.where(table["month"] >= 10,
                                    table["year"], table["year"] - 1)
    table["node_order"] = table["nodo_i"] * GRID + table["nodo_j"]
    table["category"] = [categorizar(i, j)
                         for i, j in zip(table["nodo_i"], table["nodo_j"])]
    return table


def leyenda_categorias(target, **kwargs):
    """Draw the shared category legend on an axis or a figure."""
    handles = [Patch(facecolor=COLORES_CAT[c], edgecolor="0.4",
                     linewidth=0.5, label=c) for c in ORDEN_CAT]
    return target.legend(handles=handles, ncol=3, **kwargs)


def figura_04(table, outdir):
    """Stacked bars of category proportion by month."""
    counts = (table.groupby(["month", "category"]).size().unstack(fill_value=0)
              .reindex(index=MES_ORDEN, columns=ORDEN_CAT, fill_value=0))
    proportions = counts.div(counts.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(ANCHO_1COL, ANCHO_1COL * 0.78),
                           constrained_layout=True)
    base = np.zeros(len(MES_ORDEN))
    for category in ORDEN_CAT:
        ax.bar(MES_NOM, proportions[category], bottom=base,
               color=COLORES_CAT[category], edgecolor="white",
               linewidth=0.5, width=0.8)
        base += proportions[category].values

    ax.set_ylabel("Proportion of years (%)")
    ax.set_xlabel("Month")
    ax.set_ylim(0, 100)
    leyenda_categorias(ax, loc="upper center", bbox_to_anchor=(0.5, -0.22),
                       columnspacing=1.2, handlelength=1.4)

    guardar(fig, os.path.join(outdir, "fig04_categories_by_month"))
    plt.close(fig)


def figura_05(table, outdir):
    """
    Bubble chart of month against node.

    Frequency is encoded twice, by marker area and by colour, so the chart
    remains readable in greyscale. Both scales carry their own legend; the
    counts are not printed inside the markers, which are too small to hold
    legible text at final size.
    """
    counts = table.groupby(["month", "node_order"]).size().reset_index(name="n")
    position = {m: k for k, m in enumerate(MES_ORDEN)}
    counts["x"] = counts["month"].map(position)
    nmax = counts["n"].max()

    labels = [f"{i},{j}" for i in range(GRID) for j in range(GRID)]

    fig, ax = plt.subplots(figsize=(ANCHO_F5, ANCHO_F5 * 0.85),
                           constrained_layout=True)
    scatter = ax.scatter(counts["x"], counts["node_order"],
                         s=counts["n"] * ESCALA_BURBUJA, c=counts["n"],
                         cmap=CMAP_SECUENCIAL, vmin=0, vmax=nmax,
                         edgecolors="black", linewidths=0.4)

    ax.set_yticks(range(GRID * GRID))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xticks(range(len(MES_ORDEN)))
    ax.set_xticklabels(MES_NOM)
    ax.set_xlabel("Month")
    ax.set_ylabel("SOM node (i,j)")
    ax.set_xlim(-0.5, len(MES_ORDEN) - 0.5)
    ax.set_ylim(-0.7, GRID * GRID - 0.3)
    ax.grid(True, axis="both", linestyle="--", linewidth=0.3, color="0.88")
    ax.set_axisbelow(True)

    cbar = colorbar_limpia(fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.02))
    cbar.set_label("Number of years")

    for n in [2, 5, 10]:
        ax.scatter([], [], s=n * ESCALA_BURBUJA, facecolor="0.75",
                   edgecolor="black", linewidth=0.4, label=f"{n} yr")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=3,
              scatterpoints=1, handletextpad=0.3, columnspacing=1.2)

    guardar(fig, os.path.join(outdir, "fig05_bubbles_month_node"))
    plt.close(fig)


def figura_06(table, outdir):
    """Small multiples: monthly frequency of each node, arranged on the grid."""
    fig, axes = plt.subplots(GRID, GRID, figsize=(150 * MM, 120 * MM),
                             sharex=True, sharey=True, constrained_layout=True)

    for i in range(GRID):
        for j in range(GRID):
            ax = axes[i, j]
            subset = table[(table["nodo_i"] == i) & (table["nodo_j"] == j)]
            counts = (subset.groupby("month").size()
                      .reindex(MES_ORDEN, fill_value=0))
            ax.bar(range(len(MES_ORDEN)), counts.values,
                   color=COLORES_CAT[categorizar(i, j)],
                   edgecolor="none", width=0.7)
            ax.set_title(f"Node {i},{j}", pad=2.5)
            ax.set_ylim(0, 12)
            ax.set_yticks([0, 4, 8, 12])
            ax.set_xticks(range(len(MES_ORDEN)))
            ax.set_xticklabels(MES_LETRA, fontsize=7)
            ax.tick_params(length=2)
            if j == 0:
                ax.set_ylabel("Count")

    leyenda_categorias(fig, loc="outside lower center",
                       columnspacing=1.2, handlelength=1.4)

    guardar(fig, os.path.join(outdir, "fig06_small_multiples_nodes"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Draw Figures 4, 5 and 6.")
    parser.add_argument("--csv", default=DEFAULT_CSV,
                        help="BMU table written by export_bmus_by_date.py.")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR,
                        help="Directory receiving the figures.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    table = load_table(args.csv)

    figura_04(table, args.outdir)
    figura_05(table, args.outdir)
    figura_06(table, args.outdir)
    print("Done: Figures 4, 5 and 6.")


if __name__ == "__main__":
    main()
