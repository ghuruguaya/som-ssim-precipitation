#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 2: heatmap of months assigned to each node of the 4x4 SOM.

Counts are read directly from the BMU table produced by export_bmus_by_date.py,
so the figure always reflects the current classification. A fallback table is
kept in the script for the case where the CSV is unavailable.

Usage
-----
    python fig02_node_heatmap.py
    python fig02_node_heatmap.py --csv path/to/BMUs_por_fecha.csv --outdir figures

Author: Gabriela Hernandez
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plot_style import (ANCHO_1COL, CMAP_SECUENCIAL, guardar,
                        color_texto_contraste, colorbar_limpia)


# ============================== CONFIGURATION ==============================

DEFAULT_CSV = os.path.join("som_results_ssim",
                           "som_4x4_pca_bmu-ssim_BMUs_por_fecha.csv")
DEFAULT_OUTDIR = "figuras_paper"

GRID = 4

# Fallback counts, used only if the BMU table cannot be found.
MESES_RESPALDO = np.array([[44, 18, 15, 10],
                           [30, 21, 14, 12],
                           [15, 18, 12,  7],
                           [19, 12, 10,  7]])


# ================================ FUNCTIONS ================================

def load_counts(csv_path, grid=GRID):
    """
    Count the months assigned to each node.

    Reading from the BMU table avoids the transcription errors that come with
    a manually maintained count.
    """
    if not os.path.exists(csv_path):
        print(f"BMU table not found at {csv_path}; using the fallback counts.")
        return MESES_RESPALDO

    table = pd.read_csv(csv_path)
    counts = np.zeros((grid, grid), dtype=int)
    for (i, j), group in table.groupby(["nodo_i", "nodo_j"]):
        counts[int(i), int(j)] = len(group)
    print(f"Counts read from the BMU table. Total months: {counts.sum()}")
    return counts


def main():
    parser = argparse.ArgumentParser(description="Draw Figure 2.")
    parser.add_argument("--csv", default=DEFAULT_CSV,
                        help="BMU table written by export_bmus_by_date.py.")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR,
                        help="Directory receiving the figure.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    counts = load_counts(args.csv)
    percentages = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(ANCHO_1COL, ANCHO_1COL * 0.92),
                           constrained_layout=True)

    im = ax.imshow(counts, cmap=CMAP_SECUENCIAL, vmin=0, vmax=counts.max())

    # Thin white dividers between cells.
    ax.set_xticks(np.arange(-0.5, GRID, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, GRID, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.spines[:].set_visible(False)

    ax.set_xticks(range(GRID))
    ax.set_yticks(range(GRID))
    ax.set_xticklabels(range(GRID))
    ax.set_yticklabels(range(GRID))
    ax.set_xlabel("SOM grid column index (j)")
    ax.set_ylabel("SOM grid row index (i)")

    for i in range(GRID):
        for j in range(GRID):
            colour = color_texto_contraste(CMAP_SECUENCIAL,
                                           counts[i, j] / counts.max())
            ax.text(j, i - 0.13, f"{counts[i, j]}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=colour)
            ax.text(j, i + 0.22, f"{percentages[i, j]:.0f}%", ha="center",
                    va="center", fontsize=7, color=colour)

    cbar = colorbar_limpia(fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03))
    cbar.set_label("Number of months")

    guardar(fig, os.path.join(args.outdir, "fig02_node_heatmap"))
    plt.close(fig)


if __name__ == "__main__":
    main()
