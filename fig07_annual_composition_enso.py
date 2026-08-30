#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 7: annual composition of SOM categories with the ENSO phase beneath.

The upper panel shows, for each season, the proportion of months falling in the
dry, transitional and wet categories. The lower strip gives the ENSO phase of
that season, classified from the December-January-February Oceanic Nino Index.
Placing the two on a shared time axis lets the correspondence be read directly.

Usage
-----
    python fig07_annual_composition_enso.py
    python fig07_annual_composition_enso.py --csv <table.csv> --enso <oni.csv>

Author: Gabriela Hernandez
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from plot_style import (ANCHO_2COL, COLORES_CAT, ORDEN_CAT,
                        COLORES_ENSO, guardar)


# ============================== CONFIGURATION ==============================

DEFAULT_CSV = os.path.join("som_results_ssim",
                           "som_4x4_pca_bmu-ssim_BMUs_por_fecha.csv")
DEFAULT_ENSO = "ENSO_ONI_DJF_1980_2023.csv"
DEFAULT_OUTDIR = "figuras_paper"

GRID = 4
ORDEN_ENSO = ["El Niño", "Neutral", "La Niña"]

NODOS_SECOS = {(0, 0), (0, 1), (1, 0)}
NODOS_HUMEDOS = {(2, 2), (2, 3), (3, 1), (3, 2), (3, 3)}


# ================================ FUNCTIONS ================================

def categorizar(i, j):
    """Return the category of a node: Dry, Wet, or Transitional otherwise."""
    if (i, j) in NODOS_SECOS:
        return "Dry"
    if (i, j) in NODOS_HUMEDOS:
        return "Wet"
    return "Transitional"


def load_table(csv_path):
    """Read the BMU table and assign each month to a season and a category."""
    table = pd.read_csv(csv_path)
    table["fecha"] = pd.to_datetime(table["fecha"])
    table["month"] = table["fecha"].dt.month
    table["year"] = table["fecha"].dt.year
    # October to December belong to the season labelled by that year; January
    # to March belong to the season that began the previous October.
    table["season_year"] = np.where(table["month"] >= 10,
                                    table["year"], table["year"] - 1)
    table["category"] = [categorizar(i, j)
                         for i, j in zip(table["nodo_i"], table["nodo_j"])]
    return table


def main():
    parser = argparse.ArgumentParser(description="Draw Figure 7.")
    parser.add_argument("--csv", default=DEFAULT_CSV,
                        help="BMU table written by export_bmus_by_date.py.")
    parser.add_argument("--enso", default=DEFAULT_ENSO,
                        help="CSV with columns season_year and ENSO_phase.")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR,
                        help="Directory receiving the figure.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    table = load_table(args.csv)
    enso = pd.read_csv(args.enso)
    phase_by_year = enso.set_index("season_year")["ENSO_phase"].to_dict()

    years = sorted(table["season_year"].unique())
    counts = (table.groupby(["season_year", "category"]).size()
              .unstack(fill_value=0)
              .reindex(index=years, columns=ORDEN_CAT, fill_value=0))
    proportions = counts.div(counts.sum(axis=1), axis=0) * 100

    fig, (ax_top, ax_strip) = plt.subplots(
        2, 1, figsize=(ANCHO_2COL, ANCHO_2COL * 0.44), sharex=True,
        gridspec_kw={"height_ratios": [6, 0.7], "hspace": 0.12},
        constrained_layout=True)

    base = np.zeros(len(years))
    for category in ORDEN_CAT:
        ax_top.bar(years, proportions[category], bottom=base,
                   color=COLORES_CAT[category], edgecolor="white",
                   linewidth=0.3, width=0.85)
        base += proportions[category].values

    ax_top.set_ylabel("Proportion of months (%)")
    ax_top.set_ylim(0, 100)
    ax_top.set_xlim(years[0] - 0.7, years[-1] + 0.7)

    handles_cat = [Patch(facecolor=COLORES_CAT[c], edgecolor="0.4",
                         linewidth=0.5, label=c) for c in ORDEN_CAT]
    ax_top.legend(handles=handles_cat, loc="lower left",
                  bbox_to_anchor=(0.0, 1.01), ncol=3,
                  columnspacing=1.2, handlelength=1.4)

    for year in years:
        phase = phase_by_year.get(year, "Neutral")
        ax_strip.bar(year, 1, width=0.85, color=COLORES_ENSO[phase],
                     edgecolor="0.6", linewidth=0.3)

    ax_strip.set_yticks([])
    ax_strip.set_ylabel("ENSO\n(DJF)", rotation=0, ha="right",
                        va="center", fontsize=7)
    ax_strip.set_xlabel("Season year (Oct–Mar)")
    ax_strip.set_xticks(np.arange(1985, years[-1] + 1, 5))
    ax_strip.spines["left"].set_visible(False)

    handles_enso = [Patch(facecolor=COLORES_ENSO[p], edgecolor="0.6",
                          linewidth=0.5, label=p) for p in ORDEN_ENSO]
    fig.legend(handles=handles_enso, loc="outside lower center",
               ncol=3, columnspacing=1.2, handlelength=1.4, fontsize=7)

    guardar(fig, os.path.join(args.outdir, "fig07_annual_composition_enso"))
    plt.close(fig)
    print("Done: Figure 7.")


if __name__ == "__main__":
    main()
