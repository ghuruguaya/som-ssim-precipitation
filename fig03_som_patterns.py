#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 3: the 16 SOM nodes, observed mean and denormalized prototype.

For each node the figure shows two maps side by side: the mean of the anomaly
fields assigned to it, and the prototype vector transformed back into
millimetres by inverting the per-pixel Min-Max normalization. Comparing the two
shows whether the prototype represents the months it groups.

The fields are computed and drawn in a single Cartopy figure rather than
assembled from separate PNGs. Cropping and pasting images was the source of
uneven margins and stray rules in earlier versions.

Layout: one landscape page, four node rows by eight map columns, with a single
colorbar. Every panel carries latitude and longitude labels with two ticks per
axis, so the labels do not crowd.

Usage
-----
    python fig03_som_patterns.py
    python fig03_som_patterns.py --cfg-dir <dir> --anom-nc <file.nc>

Author: Gabriela Hernandez
"""

import os
import json
import argparse

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from plot_style import (CMAP_ANOM, VMIN, VMAX, LEVELS, TICKS_CBAR,
                        MM, guardar, mapa_base, colorbar_limpia)


# ============================== CONFIGURATION ==============================

DEFAULT_ANOM_NC = "chirps_anom_oct_mar.nc"      # anomalies in millimetres
DEFAULT_TIME_NC = "chirps_normalized.nc"        # time axis of the SOM run
DEFAULT_CFG_DIR = os.path.join("som_results_ssim", "som_4x4_pca_bmu-ssim")
DEFAULT_OUTDIR = "figuras_paper"

LON_TICKS = [-62, -58, -54]
LAT_TICKS = [-25, -30, -35]

ANCHO_FIG = 228 * MM
ALTO_FIG = 168 * MM


# ================================ FUNCTIONS ================================

def primera_var(dataset):
    """Return the first data variable of a dataset."""
    return list(dataset.data_vars)[0]


def nombre(dataset, candidatos):
    """Return the first candidate name present among coordinates or dimensions."""
    return next((n for n in candidatos
                 if n in dataset.coords or n in dataset.dims), None)


def vec_to_2d(vector, shape, valid_idx):
    """Map a vector of land pixels back onto the 2D grid, filling with NaN."""
    field = np.full(shape, np.nan)
    field[valid_idx] = vector
    return field


def main():
    parser = argparse.ArgumentParser(description="Draw Figure 3.")
    parser.add_argument("--anom-nc", default=DEFAULT_ANOM_NC,
                        help="NetCDF of anomalies in millimetres.")
    parser.add_argument("--time-nc", default=DEFAULT_TIME_NC,
                        help="NetCDF used for training, read for its time axis.")
    parser.add_argument("--cfg-dir", default=DEFAULT_CFG_DIR,
                        help="Directory holding BMUs.npy, weights.npy and config.json.")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR,
                        help="Directory receiving the figure.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ---------------------------- Load the data ----------------------------
    print("Loading data...")
    dataset = xr.open_dataset(args.anom_nc)
    vname = primera_var(dataset)
    latn = nombre(dataset, ["latitude", "lat", "y"])
    lonn = nombre(dataset, ["longitude", "lon", "x"])
    timn = nombre(dataset, ["time", "t"])
    array = dataset[vname].transpose(timn, latn, lonn)
    lats, lons = dataset[latn].values, dataset[lonn].values

    # Restrict to the warm season; a no-op if the file is already subset.
    array = array.sel({timn: (array[timn].dt.month >= 10) |
                             (array[timn].dt.month <= 3)})

    # Ocean mask, and the per-pixel range needed to denormalize the prototypes.
    mask_ocean = ~np.isfinite(array.max(dim=timn, skipna=True)).values
    min_mm = array.min(dim=timn, skipna=True).values
    max_mm = array.max(dim=timn, skipna=True).values
    span = max_mm - min_mm
    span = np.where((span == 0) | ~np.isfinite(span), np.nan, span)

    # SOM output.
    with open(os.path.join(args.cfg_dir, "config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    rows, cols = config["grid_size"]

    bmus = np.load(os.path.join(args.cfg_dir, "BMUs.npy"))
    weights = np.load(os.path.join(args.cfg_dir, "weights.npy"))
    raw_idx = np.load(os.path.join(args.cfg_dir, "valid_idx.npy"), allow_pickle=True)
    valid_idx = (np.asarray(raw_idx[0]).astype(int),
                 np.asarray(raw_idx[1]).astype(int))

    # Training dates, aligned with the BMU entries.
    times = pd.to_datetime(xr.open_dataset(args.time_nc)["time"].values)
    n_use = min(len(times), len(bmus))
    times, bmus = times[:n_use], bmus[:n_use]
    in_season = (times.month >= 10) | (times.month <= 3)
    times, bmus = times[in_season], bmus[in_season]

    shape = (len(lats), len(lons))
    times_nc = pd.to_datetime(array[timn].values)
    year_month_nc = list(zip(times_nc.year, times_nc.month))

    # -------------------------- Fields per node ---------------------------
    def prototipo_mm(i, j):
        """Prototype weights in [0, 1] mapped back to millimetres."""
        field = vec_to_2d(weights[i, j], shape, valid_idx) * span + min_mm
        field[mask_ocean] = np.nan
        return np.clip(field, VMIN, VMAX)

    def promedio_mm(i, j):
        """
        Mean observed anomaly of the months assigned to a node.

        Months are matched by year and calendar month rather than by position,
        so the selection holds even if the two files are ordered differently.
        """
        dates = times[(bmus[:, 0] == i) & (bmus[:, 1] == j)]
        if len(dates) == 0:
            return None
        wanted = {(d.year, d.month) for d in dates}
        selection = np.array([ym in wanted for ym in year_month_nc])
        field = array.isel({timn: selection}).mean(dim=timn, skipna=True).values
        field[mask_ocean] = np.nan
        return np.clip(field, VMIN, VMAX)

    # -------------------------------- Figure -------------------------------
    fig = plt.figure(figsize=(ANCHO_FIG, ALTO_FIG))
    gs = fig.add_gridspec(rows, cols * 2, left=0.035, right=0.995,
                          top=0.975, bottom=0.105, wspace=0.34, hspace=0.28)

    im = None
    for i in range(rows):
        for j in range(cols):
            for m, field in enumerate([promedio_mm(i, j), prototipo_mm(i, j)]):
                ax = fig.add_subplot(gs[i, j * 2 + m], projection=ccrs.PlateCarree())
                mapa_base(ax, LON_TICKS, LAT_TICKS,
                          etiquetas_izq=True, etiquetas_abajo=True,
                          tam_etiquetas=5.5)
                if field is not None:
                    im = ax.contourf(lons, lats, np.ma.masked_invalid(field),
                                     levels=LEVELS, cmap=CMAP_ANOM,
                                     transform=ccrs.PlateCarree(),
                                     extend="both", antialiased=True)
                ax.set_extent([lons.min(), lons.max(), lats.min(), lats.max()],
                              crs=ccrs.PlateCarree())
                label = "mean" if m == 0 else "prototype"
                ax.set_title(f"Node {i},{j} {label}", fontsize=6, pad=2)

    cax = fig.add_axes([0.30, 0.062, 0.40, 0.013])
    cbar = colorbar_limpia(fig.colorbar(im, cax=cax, orientation="horizontal",
                                        ticks=TICKS_CBAR, extend="both"))
    cbar.set_label("Precipitation anomaly (mm)")

    guardar(fig, os.path.join(args.outdir, "fig03_som_patterns"))
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
