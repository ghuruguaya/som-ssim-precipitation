#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recompute internal cluster validity indices for all SOM configurations
under a common yardstick.

Motivation
----------
Each SOM configuration was trained with its own BMU similarity metric
(Euclidean, correlation, simplified SSIM, full SSIM). The internal validity
indices, however, must be computed under a single, fixed distance so that
configurations remain comparable. This script recomputes the indices from the
BMU assignments already saved during training, so no retraining is required.

For every configuration the script reports the Silhouette coefficient under
BOTH a Euclidean and a correlation distance. The Euclidean column is the
primary result: it is the reference distance most commonly used in the SOM
literature, and it is the yardstick least favourable to the SSIM-based
configurations. The correlation column is retained because the difference
between the two quantifies the extent to which the validity index itself,
rather than the partition, depends on the distance chosen to evaluate it.

Davies-Bouldin and Calinski-Harabasz are computed on the feature matrix.
Both are defined in terms of centroids and Euclidean dispersion and admit no
choice of distance, so they are reported once.

All four BMU metrics are covered: the three explored in the multiseed phase
(Euclidean, correlation, simplified global SSIM) and the full sliding-window
SSIM trained on the computing cluster. The two phases store their output
differently, so they are read by separate loaders, but every partition is
evaluated on the same feature matrix and the same pair of distance matrices.
Partitions are label vectors; evaluating them all in one common feature space
is what makes the configurations comparable, irrespective of the land mask
each training run happened to use.

Inputs
------
Normalized anomaly NetCDF (0-1, October-March) used for training, the
directories containing the BMU files saved by the multiseed scripts, and the
directory holding the cluster results (one subdirectory per configuration,
each with BMUs.npy and config.json).

Outputs
-------
A single tidy CSV with one row per (training metric, initialization, grid
size, seed) combination.

Usage
-----
    python recompute_validity_indices.py
    python recompute_validity_indices.py --netcdf data/chirps_normalized.nc

Author: Gabriela Hernandez
"""

import os
import json
import argparse
import numpy as np
import xarray as xr
import pandas as pd
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.metrics.pairwise import pairwise_distances


# ============================== CONFIGURATION ==============================

# Normalized anomaly file (0-1 per pixel, October-March) used for training.
DEFAULT_NETCDF = os.path.join("data", "chirps_normalized.nc")

# Name of the variable holding the normalized anomalies. Set to None to take
# the first data variable in the file.
DEFAULT_VARNAME = None

# Root directory under which the multiseed output folders live.
DEFAULT_ROOT = "."

# Months retained for the analysis (austral warm season).
MONTHS = [10, 11, 12, 1, 2, 3]

# Grid sizes and seeds explored.
GRID_SIZES = [3, 4, 6]
SEEDS = [1, 10, 22, 50, 77, 123, 145, 200, 321, 444, 555, 999]

# One entry per exploratory configuration:
#   (training metric, initialization, output directory, BMU filename prefix)
# The prefixes reproduce the naming used by the multiseed training scripts.
CONFIGURATIONS = [
    ("euclidean",  "pca",    os.path.join("EUCLID", "som_euclid_pca_mascara_multisemilla"),  "bmus_euclid_pca"),
    ("euclidean",  "random", os.path.join("EUCLID", "som_eucl_random_mascara_multisemilla"), "bmus_eucl_random"),
    ("correlation", "pca",    os.path.join("CORR", "som_corr_pca_mascara_multisemilla"),      "bmus_cor_pca"),
    ("correlation", "random", os.path.join("CORR", "som_corr_random_mascara_multisemilla"),   "bmus_cor_random"),
    ("ssim_simplified", "pca",    os.path.join("SSIM", "som_ssim_pca_mascara_multisemilla"),    "bmus_ssim_pca"),
    ("ssim_simplified", "random", os.path.join("SSIM", "som_ssim_random_mascara_multisemilla"), "bmus_ssim_random"),
]

# Directory holding the full sliding-window SSIM runs produced on the cluster.
# Each subdirectory is expected to contain BMUs.npy and config.json.
DEFAULT_SSIM_FULL_ROOT = "som_results_ssim"

OUTPUT_CSV = "validity_indices_unified.csv"


# ================================ FUNCTIONS ================================

def load_feature_matrix(netcdf_path, varname=None, months=MONTHS):
    """
    Build the (n_time, n_valid_pixels) feature matrix used for training.

    The land mask keeps only pixels that are valid at every retained time step,
    reproducing exactly the masking applied by the training scripts so that
    column ordering is identical and BMU labels remain meaningful.

    Parameters
    ----------
    netcdf_path : str
        Path to the normalized anomaly NetCDF.
    varname : str or None
        Variable name; the first data variable is used when None.
    months : list of int
        Calendar months to retain.

    Returns
    -------
    X : ndarray of shape (n_time, n_valid)
        Feature matrix, float64, free of NaN.
    n_valid : int
        Number of retained pixels.
    """
    ds = xr.open_dataset(netcdf_path)
    if varname is None:
        varname = list(ds.data_vars)[0]
        print(f"  Variable not specified; using '{varname}'")
    da = ds[varname]

    da = da.sel(time=da["time"].dt.month.isin(months))
    values = da.values  # (time, lat, lon)

    # A pixel is valid only if it is free of NaN across every retained month.
    land_mask = ~np.isnan(values).any(axis=0)
    X = values[:, land_mask].astype(np.float64)
    ds.close()

    if np.isnan(X).any():
        raise ValueError("Feature matrix still contains NaN after masking.")

    print(f"  Feature matrix: {X.shape[0]} time steps x {X.shape[1]} valid pixels")
    return X, X.shape[1]


def bmus_to_labels(bmus, grid_size):
    """
    Convert BMU coordinate pairs (i, j) into flat integer labels.

    Parameters
    ----------
    bmus : ndarray
        Either (n_time, 2) coordinate pairs or (n_time,) flat labels.
    grid_size : int
        Number of columns in the square SOM grid.

    Returns
    -------
    ndarray of shape (n_time,), dtype int
    """
    bmus = np.asarray(bmus)
    if bmus.ndim == 2 and bmus.shape[1] == 2:
        return (bmus[:, 0] * grid_size + bmus[:, 1]).astype(int)
    if bmus.ndim == 1:
        return bmus.astype(int)
    raise ValueError(f"Unexpected BMU array shape: {bmus.shape}")


def occupancy_statistics(labels, n_nodes):
    """
    Describe how the input fields are distributed over the map.

    Node occupancy is a count and therefore does not depend on any distance,
    which makes it usable for comparing configurations trained with different
    BMU metrics. Three complementary quantities are returned, because the
    number of occupied nodes alone does not distinguish a map that spreads the
    fields evenly from one that fills most nodes while crowding the majority of
    fields into a few.

    Parameters
    ----------
    labels : ndarray of shape (n_time,)
        Flat node label assigned to each field.
    n_nodes : int
        Total number of nodes in the map (rows x columns).

    Returns
    -------
    dict
        n_nodes_total : size of the map.
        n_nodes_empty : nodes to which no field was assigned.
        max_node_share : fraction of fields falling in the most populated node.
        occupancy_entropy : Shannon entropy of the occupancy distribution,
            normalized by log(n_nodes). Equals 1 when every node receives the
            same number of fields and approaches 0 as the fields concentrate in
            a single node. Unlike the raw count, it penalizes crowding.
    """
    counts = np.bincount(labels, minlength=n_nodes).astype(float)
    n_empty = int((counts == 0).sum())

    proportions = counts / counts.sum()
    nonzero = proportions[proportions > 0]
    entropy = -(nonzero * np.log(nonzero)).sum()
    normalized_entropy = entropy / np.log(n_nodes) if n_nodes > 1 else np.nan

    return {
        "n_nodes_total": int(n_nodes),
        "n_nodes_empty": n_empty,
        "max_node_share": float(proportions.max()),
        "occupancy_entropy": float(normalized_entropy),
    }


def compute_indices(X, labels, dist_euclidean, dist_correlation, n_nodes):
    """
    Compute validity indices for one partition.

    Silhouette is evaluated twice, on precomputed distance matrices, so that
    the effect of the evaluation distance can be isolated from the effect of
    the partition itself. Davies-Bouldin and Calinski-Harabasz are computed on
    the feature matrix; both are centroid-based and Euclidean by construction.

    Returns
    -------
    dict
        Index values, or NaN where fewer than two clusters are populated.
    """
    occupancy = occupancy_statistics(labels, n_nodes)

    n_clusters = np.unique(labels).size
    if n_clusters < 2:
        return {
            **occupancy,
            "silhouette_euclidean": np.nan,
            "silhouette_correlation": np.nan,
            "davies_bouldin": np.nan,
            "calinski_harabasz": np.nan,
            "n_clusters_occupied": n_clusters,
            "note": "Fewer than two occupied nodes; indices undefined.",
        }

    return {
        **occupancy,
        "silhouette_euclidean": float(
            silhouette_score(dist_euclidean, labels, metric="precomputed")
        ),
        "silhouette_correlation": float(
            silhouette_score(dist_correlation, labels, metric="precomputed")
        ),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
        "n_clusters_occupied": int(n_clusters),
        "note": "",
    }


def collect_multiseed_records(root, X, dist_euclidean, dist_correlation):
    """
    Read the BMU files written by the exploratory multiseed scripts.

    Each file encodes its configuration in the filename, so grid size and seed
    are recovered from the naming pattern rather than from a metadata file.

    Returns
    -------
    records : list of dict
    missing : int
        Number of expected files that were absent.
    """
    records = []
    missing = 0

    for train_metric, init, subdir, prefix in CONFIGURATIONS:
        directory = os.path.join(root, subdir)
        print(f"\n--- {train_metric} / {init} ---")

        if not os.path.isdir(directory):
            print(f"  Directory not found, skipping: {directory}")
            continue

        for grid in GRID_SIZES:
            found = 0
            for seed in SEEDS:
                filename = f"{prefix}_{grid}x{grid}_oct_mar_mascara_seed{seed}.npy"
                path = os.path.join(directory, filename)

                if not os.path.exists(path):
                    missing += 1
                    continue

                bmus = np.load(path)
                labels = bmus_to_labels(bmus, grid)

                if labels.shape[0] != X.shape[0]:
                    print(f"  Length mismatch in {filename}: "
                          f"{labels.shape[0]} BMUs vs {X.shape[0]} time steps. Skipped.")
                    continue

                indices = compute_indices(X, labels, dist_euclidean,
                                          dist_correlation, grid * grid)
                records.append({
                    "training_metric": train_metric,
                    "initialization": init,
                    "grid": f"{grid}x{grid}",
                    "seed": seed,
                    "phase": "exploratory",
                    **indices,
                })
                found += 1

            print(f"  {grid}x{grid}: {found}/{len(SEEDS)} seeds processed")

    return records, missing


def collect_full_ssim_records(root, X, dist_euclidean, dist_correlation):
    """
    Read the full sliding-window SSIM runs produced on the computing cluster.

    Unlike the exploratory output, these runs store one directory per
    configuration containing BMUs.npy and a config.json describing the grid
    size, initialization and seed actually used. Subdirectories are discovered
    automatically, so the naming convention does not matter.

    Returns
    -------
    records : list of dict
    """
    records = []
    print("\n--- ssim_full (cluster runs) ---")

    if not os.path.isdir(root):
        print(f"  Directory not found, skipping: {root}")
        return records

    subdirs = sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
        and os.path.exists(os.path.join(root, d, "config.json"))
        and os.path.exists(os.path.join(root, d, "BMUs.npy"))
    )

    if not subdirs:
        print(f"  No configuration directories with BMUs.npy and config.json in {root}")
        return records

    for name in subdirs:
        directory = os.path.join(root, name)

        with open(os.path.join(directory, "config.json"), "r", encoding="utf-8") as handle:
            cfg = json.load(handle)

        grid_size = cfg.get("grid_size") or cfg.get("grid")
        if grid_size is None:
            print(f"  {name}: no grid size in config.json, skipped.")
            continue
        rows, cols = int(grid_size[0]), int(grid_size[1])

        bmus = np.load(os.path.join(directory, "BMUs.npy"), allow_pickle=True)
        labels = bmus_to_labels(bmus, cols)

        if labels.shape[0] != X.shape[0]:
            print(f"  {name}: length mismatch, {labels.shape[0]} BMUs vs "
                  f"{X.shape[0]} time steps. Skipped.")
            continue

        indices = compute_indices(X, labels, dist_euclidean,
                                  dist_correlation, rows * cols)
        records.append({
            "training_metric": "ssim_full",
            "initialization": cfg.get("initialization", "unknown"),
            "grid": f"{rows}x{cols}",
            "seed": cfg.get("seed", np.nan),
            "phase": "final",
            **indices,
        })
        print(f"  {name}: {rows}x{cols}, seed {cfg.get('seed')}, "
              f"{indices['n_clusters_occupied']}/{rows * cols} nodes occupied")

    return records


def main():
    parser = argparse.ArgumentParser(
        description="Recompute SOM validity indices under a common distance."
    )
    parser.add_argument("--netcdf", default=DEFAULT_NETCDF,
                        help="Normalized anomaly NetCDF used for training.")
    parser.add_argument("--varname", default=DEFAULT_VARNAME,
                        help="Variable name inside the NetCDF.")
    parser.add_argument("--root", default=DEFAULT_ROOT,
                        help="Root directory containing the multiseed BMU folders.")
    parser.add_argument("--ssim-full-root", default=DEFAULT_SSIM_FULL_ROOT,
                        help="Directory containing the cluster runs (full SSIM).")
    parser.add_argument("--out", default=OUTPUT_CSV,
                        help="Path of the output CSV.")
    args = parser.parse_args()

    print("=== Recomputing internal validity indices ===")
    print(f"NetCDF: {args.netcdf}")

    X, n_valid = load_feature_matrix(args.netcdf, args.varname)

    # The two distance matrices are computed once and reused for every
    # configuration. This is what makes the exhaustive recomputation cheap:
    # the partitions change, the pairwise distances between fields do not.
    print("  Computing pairwise distance matrices (once)...")
    dist_euclidean = pairwise_distances(X, metric="euclidean")
    dist_correlation = pairwise_distances(X, metric="correlation")
    print("  Done.")

    records, missing = collect_multiseed_records(
        args.root, X, dist_euclidean, dist_correlation
    )

    full_root = args.ssim_full_root
    if not os.path.isabs(full_root):
        full_root = os.path.join(args.root, full_root)
    records += collect_full_ssim_records(
        full_root, X, dist_euclidean, dist_correlation
    )

    if not records:
        raise SystemExit(
            "No BMU files were found. Check --root and the paths in CONFIGURATIONS."
        )

    df = pd.DataFrame(records)
    df.to_csv(args.out, index=False)

    print(f"\nProcessed {len(df)} configurations "
          f"({missing} expected multiseed files were absent).")
    print(f"Written: {args.out}")

    # Summary by training metric and grid size, which is the comparison the
    # analysis rests on. Node occupancy is included because, unlike the
    # Silhouette coefficient, it is a count and therefore does not depend on
    # the distance chosen to evaluate the partition.
    summary = (
        df.groupby(["training_metric", "grid"])
        .agg(
            n_runs=("seed", "size"),
            sil_euclidean=("silhouette_euclidean", "mean"),
            sil_correlation=("silhouette_correlation", "mean"),
            nodes_occupied=("n_clusters_occupied", "mean"),
            nodes_empty=("n_nodes_empty", "mean"),
            max_share=("max_node_share", "mean"),
            evenness=("occupancy_entropy", "mean"),
        )
        .round(4)
    )
    print("\n=== Summary by training metric and grid size ===")
    print(summary.to_string())


if __name__ == "__main__":
    main()
