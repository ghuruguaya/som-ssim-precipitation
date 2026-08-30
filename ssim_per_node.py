#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Node-mean SSIM: structural agreement between each field and its prototype.

For every month, the simplified global SSIM is computed between the observed
anomaly field and the prototype vector of the node it was assigned to. The
values are then averaged within each node.

This quantifies how well each prototype represents the months grouped under it.
Being SSIM-based, it compares configurations trained with SSIM against one
another; it is not a metric-neutral criterion and should not be used to rank
configurations trained with different BMU metrics (see Section 2.5 of the
manuscript).

The dynamic range is L = 1.0 throughout, consistent with input data normalized
to [0, 1].

Usage
-----
    python ssim_per_node.py --cfg-dir som_results_ssim/som_4x4_pca_bmu-ssim
    python ssim_per_node.py --cfg-dir <dir> --netcdf <file.nc> --out <file.csv>

Author: Gabriela Hernandez
"""

import os
import json
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import xarray as xr


# ============================== CONFIGURATION ==============================

DEFAULT_CFG_DIR = os.path.join("som_results_ssim", "som_4x4_pca_bmu-ssim")
DEFAULT_NETCDF = "chirps_normalized.nc"

MONTHS = [10, 11, 12, 1, 2, 3]

K1, K2 = 0.01, 0.03
DYNAMIC_RANGE = 1.0

AUXILIARY_VARS = {"time_bnds", "time_bounds", "lat_bnds", "lon_bnds",
                  "latitude_bnds", "longitude_bnds", "crs", "spatial_ref"}


# ================================ FUNCTIONS ================================

def detect_variable(dataset):
    """Choose the data variable, discarding known auxiliary ones."""
    candidates = [v for v in dataset.data_vars if v not in AUXILIARY_VARS]
    if not candidates:
        raise ValueError("No data variable found in the NetCDF.")
    if len(candidates) > 1:
        raise ValueError(f"Several data variables found: {candidates}. "
                         f"Specify one with --varname.")
    return candidates[0]


def ssim_global(x, y, dynamic_range=DYNAMIC_RANGE):
    """
    Simplified Structural Similarity Index between two vectors.

    Luminance, contrast and structure are computed from global statistics
    rather than within sliding windows, so a single index summarizes the pair.
    The stabilizing constants follow Wang et al. (2004).
    """
    mean_x, mean_y = np.mean(x), np.mean(y)
    var_x, var_y = np.var(x), np.var(y)
    covariance = np.mean((x - mean_x) * (y - mean_y))

    c1 = (K1 * dynamic_range) ** 2
    c2 = (K2 * dynamic_range) ** 2

    return (((2 * mean_x * mean_y + c1) * (2 * covariance + c2)) /
            ((mean_x ** 2 + mean_y ** 2 + c1) * (var_x + var_y + c2)))


def load_feature_matrix(netcdf_path, varname=None, months=MONTHS):
    """
    Build the (n_time, n_valid_pixels) matrix, keeping pixels that are valid
    at every retained time step.
    """
    dataset = xr.open_dataset(netcdf_path)
    if varname is None:
        varname = detect_variable(dataset)
        print(f"  Variable not specified; using '{varname}'")

    array = dataset[varname]
    array = array.sel(time=array["time"].dt.month.isin(months))
    values = array.values
    dataset.close()

    land_mask = ~np.isnan(values).any(axis=0)
    return values[:, land_mask]


def main():
    parser = argparse.ArgumentParser(
        description="Compute the node-mean SSIM of a trained SOM."
    )
    parser.add_argument("--cfg-dir", default=DEFAULT_CFG_DIR,
                        help="Directory holding BMUs.npy, weights.npy and config.json.")
    parser.add_argument("--netcdf", default=DEFAULT_NETCDF,
                        help="Normalized anomaly NetCDF used for training.")
    parser.add_argument("--varname", default=None,
                        help="Variable name inside the NetCDF (auto-detected if omitted).")
    parser.add_argument("--out", default=None,
                        help="Output CSV. Defaults to <config name>_ssim_per_node.csv.")
    args = parser.parse_args()

    with open(os.path.join(args.cfg_dir, "config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    rows, cols = config["grid_size"]

    bmus = np.load(os.path.join(args.cfg_dir, "BMUs.npy"))
    weights = np.load(os.path.join(args.cfg_dir, "weights.npy"))

    print("=== Node-mean SSIM ===")
    data = load_feature_matrix(args.netcdf, args.varname)

    if data.shape[0] != len(bmus):
        raise ValueError(
            f"Length mismatch: {data.shape[0]} time steps against "
            f"{len(bmus)} BMU entries."
        )
    if weights.shape[-1] != data.shape[1]:
        raise ValueError(
            f"Prototype length {weights.shape[-1]} does not match "
            f"{data.shape[1]} valid pixels. The land masks differ."
        )

    values_by_node = defaultdict(list)
    for index, field in enumerate(data):
        i, j = int(bmus[index, 0]), int(bmus[index, 1])
        values_by_node[(i, j)].append(ssim_global(field, weights[i, j]))

    records = []
    for i in range(rows):
        for j in range(cols):
            values = values_by_node.get((i, j), [])
            records.append({
                "nodo_i": i,
                "nodo_j": j,
                "n_meses": len(values),
                "ssim_promedio": float(np.mean(values)) if values else np.nan,
            })

    table = pd.DataFrame(records)

    out_path = args.out or os.path.join(
        os.path.dirname(args.cfg_dir) or ".",
        f"{os.path.basename(args.cfg_dir)}_ssim_per_node.csv")
    table.to_csv(out_path, index=False)

    node_mean = table["ssim_promedio"].mean(skipna=True)
    all_values = [v for values in values_by_node.values() for v in values]
    global_ssim = float(np.mean(all_values)) if all_values else np.nan

    print(f"\n  Node-mean SSIM (nodes weighted equally): {node_mean:.4f}")
    print(f"  Global SSIM (fields weighted equally):   {global_ssim:.4f}")
    print(f"  Occupied nodes: {int((table['n_meses'] > 0).sum())} of {rows * cols}")
    print(f"  Written: {out_path}")


if __name__ == "__main__":
    main()
