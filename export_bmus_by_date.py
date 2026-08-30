#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export the BMU assignment of every month, with its date, to CSV.

Reads the BMU array produced by train_som_full_ssim.py and pairs each entry
with the corresponding date taken from the NetCDF used for training. The
resulting table is the input of the figure scripts (fig02, fig04_05_06 and
fig07) and of the ENSO analysis.

Output columns
--------------
fecha    Date of the month, as stored in the NetCDF.
nodo_i   Row index of the assigned node.
nodo_j   Column index of the assigned node.
nodo     Flat node label, computed as nodo_i * n_columns + nodo_j.

Usage
-----
    python export_bmus_by_date.py --cfg-dir som_results_ssim/som_4x4_pca_bmu-ssim
    python export_bmus_by_date.py --cfg-dir <dir> --netcdf chirps_normalized.nc

Author: Gabriela Hernandez
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import xarray as xr


# ============================== CONFIGURATION ==============================

DEFAULT_CFG_DIR = os.path.join("som_results_ssim", "som_4x4_pca_bmu-ssim")
DEFAULT_NETCDF = "chirps_normalized.nc"

MONTHS = [10, 11, 12, 1, 2, 3]

AUXILIARY_VARS = {"time_bnds", "time_bounds", "lat_bnds", "lon_bnds",
                  "latitude_bnds", "longitude_bnds", "crs", "spatial_ref"}


# ================================ FUNCTIONS ================================

def load_dates(netcdf_path, months=MONTHS):
    """
    Read the time axis of the training NetCDF, restricted to the warm season.

    The filter is applied here rather than assumed, so that the function works
    whether the file already covers October-March only or holds the full year.
    """
    dataset = xr.open_dataset(netcdf_path)
    times = pd.to_datetime(dataset["time"].values)
    dataset.close()
    return times[times.month.isin(months)]


def main():
    parser = argparse.ArgumentParser(
        description="Export BMU assignments with their dates."
    )
    parser.add_argument("--cfg-dir", default=DEFAULT_CFG_DIR,
                        help="Directory holding BMUs.npy and config.json.")
    parser.add_argument("--netcdf", default=DEFAULT_NETCDF,
                        help="NetCDF used for training, read for its time axis.")
    parser.add_argument("--out", default=None,
                        help="Output CSV. Defaults to <config name>_BMUs_por_fecha.csv.")
    args = parser.parse_args()

    with open(os.path.join(args.cfg_dir, "config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    n_columns = int(config["grid_size"][1])

    bmus = np.load(os.path.join(args.cfg_dir, "BMUs.npy"))
    dates = load_dates(args.netcdf)

    if len(dates) != len(bmus):
        raise ValueError(
            f"Length mismatch: {len(dates)} dates in the NetCDF against "
            f"{len(bmus)} BMU entries. The two must come from the same run."
        )

    table = pd.DataFrame({
        "fecha": dates,
        "nodo_i": bmus[:, 0].astype(int),
        "nodo_j": bmus[:, 1].astype(int),
    })
    table["nodo"] = table["nodo_i"] * n_columns + table["nodo_j"]

    out_path = args.out or os.path.join(
        os.path.dirname(args.cfg_dir) or ".",
        f"{os.path.basename(args.cfg_dir)}_BMUs_por_fecha.csv")
    table.to_csv(out_path, index=False)

    print(f"Rows written: {len(table)}")
    print(f"Date range: {table['fecha'].iloc[0]} to {table['fecha'].iloc[-1]}")
    print(f"Occupied nodes: {table['nodo'].nunique()} of "
          f"{config['grid_size'][0] * n_columns}")
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
