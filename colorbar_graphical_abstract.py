#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone horizontal colorbar for the graphical abstract.

Draws the anomaly colour scale on its own, on a transparent background, so it
can be placed alongside maps assembled outside Matplotlib. The palette, range
and ticks come from plot_style.py, which keeps the graphical abstract
consistent with Figure 3.

Two labelling styles are available. The numeric style shows the scale in
millimetres and is more informative; the wording style replaces the ticks with
"drier" and "wetter" at the ends and reads more easily at small sizes.

Usage
-----
    python colorbar_graphical_abstract.py
    python colorbar_graphical_abstract.py --style words --out colorbar.png

Author: Gabriela Hernandez
"""

import os
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colorbar as mcb
from matplotlib.colors import BoundaryNorm

from plot_style import CMAP_ANOM, LEVELS, TICKS_CBAR


# ============================== CONFIGURATION ==============================

DEFAULT_OUT = "colorbar_ga.png"
DEFAULT_DPI = 600

ETIQUETA = "Precipitation anomaly (mm)"
EXTREMO_IZQUIERDO = "drier"
EXTREMO_DERECHO = "wetter"


# ================================== MAIN ===================================

def main():
    parser = argparse.ArgumentParser(
        description="Draw a standalone colorbar for the graphical abstract."
    )
    parser.add_argument("--style", choices=["numbers", "words"],
                        default="numbers",
                        help="Show tick values in millimetres, or only the "
                             "words drier and wetter at the ends.")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help="Output PNG path.")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI,
                        help="Output resolution.")
    args = parser.parse_args()

    fig = plt.figure(figsize=(4.2, 0.62))
    ax = fig.add_axes([0.06, 0.42, 0.88, 0.30])   # left, bottom, width, height

    norm = BoundaryNorm(LEVELS, CMAP_ANOM.N, extend="both")
    cbar = mcb.ColorbarBase(ax, cmap=CMAP_ANOM, norm=norm,
                            orientation="horizontal", extend="both")

    cbar.outline.set_linewidth(0.5)
    cbar.outline.set_edgecolor("#666666")

    if args.style == "numbers":
        cbar.set_ticks(TICKS_CBAR)
        cbar.ax.tick_params(labelsize=7, length=2, width=0.5, pad=2)
    else:
        cbar.set_ticks([])
        cbar.ax.text(-0.02, 0.5, EXTREMO_IZQUIERDO, transform=cbar.ax.transAxes,
                     ha="right", va="center", fontsize=8, style="italic")
        cbar.ax.text(1.02, 0.5, EXTREMO_DERECHO, transform=cbar.ax.transAxes,
                     ha="left", va="center", fontsize=8, style="italic")

    cbar.set_label(ETIQUETA, fontsize=8, labelpad=3)

    fig.savefig(args.out, dpi=args.dpi, transparent=True,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("saved:", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
