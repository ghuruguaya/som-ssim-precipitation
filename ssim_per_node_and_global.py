#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Within-node pairwise SSIM: structural agreement among the fields of a node.

For every node, the SSIM is computed between all pairs of anomaly fields
assigned to it, and the values are averaged. Unlike ssim_per_node.py, which
compares each field against its prototype, this script measures how similar the
grouped months are to one another, without reference to the prototype.

A global value per configuration is also reported, obtained by averaging the
per-node means with equal weight.

The dynamic range is L = 1.0, consistent with input data normalized to [0, 1].

Being SSIM-based, these quantities compare configurations trained with SSIM
against one another and are not a metric-neutral criterion (see Section 2.5 of
the manuscript).

Usage
-----
    python ssim_per_node_and_global.py --bmu-dir CORR/som_corr_pca_mascara_multisemilla
    python ssim_per_node_and_global.py --bmu-dir <dir> --netcdf <file.nc>

Author: Gabriela Hernandez
"""

import os
import re
import argparse
from itertools import combinations

import numpy as np
import pandas as pd
import xarray as xr


# ============================== CONFIGURATION ==============================

DEFAULT_NETCDF = "anomalias_normalizadas_minmax_1981_2024_por_anio.nc"

MONTHS = [10, 11, 12, 1, 2, 3]

K1, K2 = 0.01, 0.03
DYNAMIC_RANGE = 1.0

MIN_FIELDS_PER_NODE = 2

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

    Returns NaN when either vector is constant, since the structure term is
    then undefined.
    """
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    mean_x, mean_y = np.mean(x), np.mean(y)
    var_x, var_y = np.var(x), np.var(y)
    covariance = np.mean((x - mean_x) * (y - mean_y))

    c1 = (K1 * dynamic_range) ** 2
    c2 = (K2 * dynamic_range) ** 2

    return (((2 * mean_x * mean_y + c1) * (2 * covariance + c2)) /
            ((mean_x ** 2 + mean_y ** 2 + c1) * (var_x + var_y + c2)))


def load_feature_matrix(netcdf_path, varname=None, months=MONTHS):
    """Build the (n_time, n_valid_pixels) matrix over a constant land mask."""
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


def parse_bmu_filename(filename):
    """
    Recover the grid size and seed encoded in a multiseed BMU filename.

    Filenames follow the pattern written by the multiseed scripts, for example
    bmus_cor_pca_4x4_oct_mar_mascara_seed145.npy.

    Returns
    -------
    tuple of (str, int) or None
        Grid label and seed, or None when the pattern does not match.
    """
    grid = re.search(r"_(\d+x\d+)_", filename)
    seed = re.search(r"seed(\d+)", filename)
    if not grid or not seed:
        return None
    return grid.group(1), int(seed.group(1))


def mean_pairwise_ssim(fields):
    """
    Average SSIM over all distinct pairs of fields.

    Returns NaN when fewer than two fields are present, or when every pair is
    undefined.
    """
    if len(fields) < MIN_FIELDS_PER_NODE:
        return np.nan
    values = [ssim_global(a, b) for a, b in combinations(fields, 2)]
    values = [v for v in values if not np.isnan(v)]
    return float(np.mean(values)) if values else np.nan


def main():
    parser = argparse.ArgumentParser(
        description="Compute within-node pairwise SSIM for multiseed runs."
    )
    parser.add_argument("--bmu-dir", required=True,
                        help="Directory holding the multiseed BMU .npy files.")
    parser.add_argument("--netcdf", default=DEFAULT_NETCDF,
                        help="Normalized anomaly NetCDF used for training.")
    parser.add_argument("--varname", default=None,
                        help="Variable name inside the NetCDF (auto-detected if omitted).")
    parser.add_argument("--out-node", default="ssim_pairwise_per_node.csv",
                        help="Output CSV with one row per node and run.")
    parser.add_argument("--out-global", default="ssim_pairwise_global.csv",
                        help="Output CSV with one row per run.")
    args = parser.parse_args()

    print("=== Within-node pairwise SSIM ===")
    data = load_feature_matrix(args.netcdf, args.varname)
    print(f"  Fields: {data.shape[0]} x {data.shape[1]} valid pixels")

    files = sorted(f for f in os.listdir(args.bmu_dir) if f.endswith(".npy"))
    if not files:
        raise SystemExit(f"No .npy files found in {args.bmu_dir}")

    records = []
    for filename in files:
        parsed = parse_bmu_filename(filename)
        if parsed is None:
            print(f"  Skipping unrecognized filename: {filename}")
            continue
        grid, seed = parsed

        bmus = np.load(os.path.join(args.bmu_dir, filename))
        if len(bmus) != data.shape[0]:
            print(f"  Skipping {filename}: {len(bmus)} BMUs against "
                  f"{data.shape[0]} time steps.")
            continue

        by_node = {}
        for index, (i, j) in enumerate(bmus):
            by_node.setdefault((int(i), int(j)), []).append(data[index])

        for (i, j), fields in sorted(by_node.items()):
            records.append({
                "config_red": grid,
                "semilla": seed,
                "nodo_i": i,
                "nodo_j": j,
                "n_meses": len(fields),
                "ssim_promedio": mean_pairwise_ssim(fields),
            })

        print(f"  {filename}: {len(by_node)} occupied nodes")

    if not records:
        raise SystemExit("No BMU file could be processed.")

    per_node = pd.DataFrame(records)
    per_node.to_csv(args.out_node, index=False)

    per_run = (per_node
               .groupby(["config_red", "semilla"])["ssim_promedio"]
               .mean()
               .reset_index()
               .rename(columns={"ssim_promedio": "ssim_promedio_global"}))
    per_run.to_csv(args.out_global, index=False)

    print(f"\n  Written: {args.out_node}")
    print(f"  Written: {args.out_global}")
    print("\n=== Mean by grid size ===")
    print(per_run.groupby("config_red")["ssim_promedio_global"]
          .agg(["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    main()
