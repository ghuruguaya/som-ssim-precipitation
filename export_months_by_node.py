#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export the months assigned to each node as an Excel workbook, one sheet per node.

A convenience utility for inspecting a classification by hand: each sheet lists
the year and month of every field that fell in that node. Nothing in the
manuscript depends on this output; it exists to make a classification easy to
read without loading the arrays.

Usage
-----
    python export_months_by_node.py --bmus <file.npy> --netcdf <file.nc>
    python export_months_by_node.py --bmus <file.npy> --out months.xlsx

Requires openpyxl to write the workbook.

Author: Gabriela Hernandez
"""

import os
import argparse

import numpy as np
import pandas as pd
import xarray as xr


# ============================== CONFIGURATION ==============================

DEFAULT_NETCDF = "anomalias_normalizadas_minmax_1981_2024_por_anio.nc"

MONTHS = [10, 11, 12, 1, 2, 3]

AUXILIARY_VARS = {"time_bnds", "time_bounds", "lat_bnds", "lon_bnds",
                  "latitude_bnds", "longitude_bnds", "crs", "spatial_ref"}


# ================================ FUNCTIONS ================================

def load_dates(netcdf_path, months=MONTHS):
    """Read the time axis, restricted to the warm season."""
    dataset = xr.open_dataset(netcdf_path)
    times = pd.to_datetime(dataset["time"].values)
    dataset.close()
    return times[times.month.isin(months)]


def main():
    parser = argparse.ArgumentParser(
        description="List the months assigned to each SOM node."
    )
    parser.add_argument("--bmus", required=True,
                        help="BMU array (.npy) of the configuration to inspect.")
    parser.add_argument("--netcdf", default=DEFAULT_NETCDF,
                        help="NetCDF used for training, read for its time axis.")
    parser.add_argument("--out", default=None,
                        help="Output workbook. Defaults to the BMU filename "
                             "with an .xlsx extension.")
    args = parser.parse_args()

    bmus = np.load(args.bmus)
    dates = load_dates(args.netcdf)

    if len(dates) != len(bmus):
        raise ValueError(
            f"Length mismatch: {len(dates)} dates against {len(bmus)} BMU "
            f"entries. The two must come from the same run."
        )

    out_path = args.out or (os.path.splitext(args.bmus)[0] +
                            "_months_by_node.xlsx")

    nodes = np.unique(bmus, axis=0)
    with pd.ExcelWriter(out_path) as writer:
        for node in nodes:
            selected = (bmus[:, 0] == node[0]) & (bmus[:, 1] == node[1])
            node_dates = dates[selected]
            table = pd.DataFrame({
                "year": node_dates.year,
                "month": node_dates.month,
            })
            # Sheet names cannot contain brackets, so nodes are written as i-j.
            table.to_excel(writer, sheet_name=f"node_{node[0]}-{node[1]}",
                           index=False)

    print(f"Nodes written: {len(nodes)}")
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
