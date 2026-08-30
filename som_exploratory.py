#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared components for the exploratory multiseed SOM experiments.

The exploratory phase compares three BMU similarity metrics (Euclidean
distance, correlation-based distance and the simplified global SSIM) under two
prototype initialization schemes, across three grid sizes and twelve random
seeds. All eighteen combinations share the same training procedure and differ
only in how the BMU is selected and how prototypes are initialized, so those
pieces live here and each runner script supplies its own pair.

Training is stochastic: one field, drawn at random from the record, is
presented per iteration, for 5000 iterations. The learning rate and the
neighbourhood radius decay exponentially with iteration. This differs from the
epoch-based scheme used for the final configuration in train_som_full_ssim.py.

These experiments only train and store BMU assignments. Internal validity
indices are computed afterwards by recompute_validity_indices.py, so that every
partition is evaluated under the same distance and configurations trained with
different metrics remain comparable.

A note on the initialization schemes. They are not strictly equivalent across
metrics, and this is deliberate, since it reproduces the configurations that
were actually run:

- PCA initialization for the correlation metric places prototypes on a regular
  grid over [-1, 1] in component space; for the Euclidean and SSIM metrics it
  spans the observed range of the first two components.
- Random initialization for the Euclidean and correlation metrics draws
  prototypes from a uniform distribution over [-1, 1], which lies outside the
  [0, 1] range of the normalized input data; for the SSIM metric it samples
  observed fields instead.

The consequence is documented in TUTORIAL.md: the Euclidean metric under random
initialization collapses onto a single occupied node.

Author: Gabriela Hernandez
"""

import os
import time
import argparse
import numpy as np
import xarray as xr
from sklearn.decomposition import PCA


# ============================== CONFIGURATION ==============================

DEFAULT_NETCDF = "anomalias_normalizadas_minmax_1981_2024_por_anio.nc"

MONTHS = [10, 11, 12, 1, 2, 3]
GRID_SIZES = [3, 4, 6]
SEEDS = [1, 10, 22, 50, 77, 123, 145, 200, 321, 444, 555, 999]

N_ITERATIONS = 5000
INITIAL_SIGMA = 1.0
INITIAL_LEARNING_RATE = 0.5

# Auxiliary variables that may appear in a NetCDF but never hold the data.
AUXILIARY_VARS = {"time_bnds", "time_bounds", "lat_bnds", "lon_bnds",
                  "latitude_bnds", "longitude_bnds", "crs", "spatial_ref"}


# ============================== DATA LOADING ===============================

def detect_variable(dataset):
    """
    Choose the data variable to read from a NetCDF.

    Known auxiliary variables are discarded first. If exactly one candidate
    remains it is returned; if several do, the function raises rather than
    guessing, listing the options so the caller can pass --varname explicitly.
    """
    candidates = [v for v in dataset.data_vars if v not in AUXILIARY_VARS]
    if not candidates:
        raise ValueError("No data variable found in the NetCDF.")
    if len(candidates) > 1:
        raise ValueError(
            f"Several data variables found: {candidates}. "
            f"Specify one with --varname."
        )
    return candidates[0]


def load_feature_matrix(netcdf_path, varname=None, months=MONTHS):
    """
    Build the (n_time, n_valid_pixels) training matrix.

    Only pixels free of missing values at every retained time step are kept, so
    that the land mask is constant throughout the record and the columns of the
    matrix keep a fixed meaning.

    Parameters
    ----------
    varname : str or None
        Variable to read. Detected automatically when None.

    Returns
    -------
    ndarray of shape (n_time, n_valid)
    """
    dataset = xr.open_dataset(netcdf_path)

    if varname is None:
        varname = detect_variable(dataset)
        print(f"  Variable not specified; using '{varname}'")

    array = dataset[varname]
    array = array.sel(time=array["time"].dt.month.isin(months))

    values = array.values  # (time, lat, lon)
    land_mask = ~np.isnan(values).any(axis=0)
    matrix = values[:, land_mask]
    dataset.close()

    print(f"  Training matrix: {matrix.shape[0]} time steps x "
          f"{matrix.shape[1]} valid pixels")
    return matrix


# ============================ SIMILARITY METRICS ===========================

def euclidean_distance(x, w):
    """Straight Euclidean distance between a field and a prototype."""
    return np.linalg.norm(x - w)


def correlation_distance(x, w):
    """
    One minus the Pearson correlation between a field and a prototype.

    Returns the maximum distance of 1.0 when either vector is constant, since
    the correlation is undefined in that case.
    """
    if np.std(x) == 0 or np.std(w) == 0:
        return 1.0
    return 1 - np.corrcoef(x, w)[0, 1]


def ssim_global(x, y):
    """
    Simplified Structural Similarity Index over the whole domain.

    Luminance, contrast and structure terms are computed from global statistics
    rather than within sliding windows, so a single index summarizes each pair
    of fields. The stabilizing constants follow Wang et al. (2004) with a
    dynamic range of 1, consistent with data normalized to [0, 1].
    """
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    var_x = np.var(x)
    var_y = np.var(y)
    cov = np.mean((x - mean_x) * (y - mean_y))

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    return (((2 * mean_x * mean_y + c1) * (2 * cov + c2)) /
            ((mean_x**2 + mean_y**2 + c1) * (var_x + var_y + c2)))


def ssim_distance(x, w):
    """SSIM expressed as a distance, so that the BMU is always a minimum."""
    return -ssim_global(x, w)


METRICS = {
    "euclidean": euclidean_distance,
    "correlation": correlation_distance,
    "ssim": ssim_distance,
}


# ================================== MODEL ==================================

class MiniSom:
    """
    Minimal Self-Organizing Map with selectable BMU metric and initialization.

    Parameters
    ----------
    x, y : int
        Network dimensions.
    input_len : int
        Length of each prototype vector.
    metric : {'euclidean', 'correlation', 'ssim'}
        Criterion used to select the Best Matching Unit.
    init : {'pca', 'random'}
        Prototype initialization scheme.
    data : ndarray
        Training matrix, required by both initialization schemes.
    """

    def __init__(self, x, y, input_len, metric, init, data,
                 sigma=INITIAL_SIGMA, learning_rate=INITIAL_LEARNING_RATE):
        self.x = x
        self.y = y
        self.input_len = input_len
        self.metric_name = metric
        self.distance = METRICS[metric]
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.activation_map = np.zeros((x, y))

        if init == "pca":
            self._pca_initialization(data)
        elif init == "random":
            self._random_initialization(data)
        else:
            raise ValueError(f"Unknown initialization: {init}")

    def _pca_initialization(self, data):
        """
        Place prototypes along the first two principal components.

        The correlation metric spans a fixed [-1, 1] grid in component space,
        while the Euclidean and SSIM metrics span the observed range of the
        components. Both variants are kept as they were run.
        """
        pca = PCA(n_components=2)

        if self.metric_name == "correlation":
            pca.fit(data)
            grid_x, grid_y = np.meshgrid(np.linspace(-1, 1, self.x),
                                         np.linspace(-1, 1, self.y))
            grid = np.stack([grid_x.flatten(), grid_y.flatten()], axis=1)
            self.weights = pca.inverse_transform(grid).reshape(
                self.x, self.y, self.input_len)
        else:
            scores = pca.fit_transform(data)
            score_min = scores.min(axis=0)
            score_max = scores.max(axis=0)
            self.weights = np.zeros((self.x, self.y, self.input_len))
            for i in range(self.x):
                for j in range(self.y):
                    coord = np.array([i / (self.x - 1), j / (self.y - 1)])
                    point = score_min + coord * (score_max - score_min)
                    # inverse_transform expects a 2D array of samples; older
                    # scikit-learn releases accepted a bare 1D vector here.
                    self.weights[i, j] = pca.inverse_transform(
                        point.reshape(1, -1))[0]

    def _random_initialization(self, data):
        """
        Draw the initial prototypes.

        The SSIM metric samples observed fields, which keeps the prototypes
        inside the range of the data. The other two metrics draw from a uniform
        distribution over [-1, 1], which does not.
        """
        if self.metric_name == "ssim":
            picked = np.random.choice(data.shape[0], self.x * self.y, replace=True)
            self.weights = data[picked].reshape(self.x, self.y, self.input_len)
        else:
            self.weights = np.random.rand(self.x, self.y, self.input_len) * 2 - 1

    def _activate(self, x):
        """Fill the activation map with the distance from x to every node."""
        for i in range(self.x):
            for j in range(self.y):
                self.activation_map[i, j] = self.distance(x, self.weights[i, j])

    def winner(self, x):
        """Return the (row, column) index of the Best Matching Unit for x."""
        self._activate(x)
        return np.unravel_index(self.activation_map.argmin(),
                                self.activation_map.shape)

    def update(self, x, win, t, max_iter):
        """
        Move the winning prototype and its neighbours towards the input field.

        Both the learning rate and the neighbourhood radius decay as
        exp(-t / max_iter). Only nodes lying within the current radius of the
        winner are updated, weighted by a Gaussian influence function.
        """
        lr = self.learning_rate * np.exp(-t / max_iter)
        sigma = self.sigma * np.exp(-t / max_iter)
        for i in range(self.x):
            for j in range(self.y):
                dist_sq = (i - win[0]) ** 2 + (j - win[1]) ** 2
                if dist_sq <= sigma ** 2:
                    influence = np.exp(-dist_sq / (2 * sigma ** 2))
                    self.weights[i, j] += lr * influence * (x - self.weights[i, j])

    def train(self, data, num_iterations=N_ITERATIONS):
        """Present randomly drawn fields for the requested number of iterations."""
        for t in range(num_iterations):
            x = data[np.random.randint(0, data.shape[0])]
            win = self.winner(x)
            self.update(x, win, t, num_iterations)

    def map_vects(self, data):
        """Return the BMU coordinates for every field in data."""
        return [self.winner(x) for x in data]


# ================================= DRIVER ==================================

def build_parser(description):
    """Command-line interface shared by all six runner scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--netcdf", default=DEFAULT_NETCDF,
                        help="Normalized anomaly NetCDF.")
    parser.add_argument("--varname", default=None,
                        help="Variable name inside the NetCDF (auto-detected if omitted).")
    parser.add_argument("--outdir", default=None,
                        help="Directory where BMU files are written.")
    return parser


def run_multiseed(metric, init, default_outdir, prefix, args):
    """
    Train one metric-initialization combination over all grids and seeds.

    The random seed is set before the network is built, because random
    initialization consumes draws from the global NumPy state. Preserving that
    order is what makes the runs reproducible.

    Parameters
    ----------
    prefix : str
        Leading part of the BMU filenames, kept as originally written so that
        existing output directories remain readable.
    """
    outdir = args.outdir or default_outdir
    os.makedirs(outdir, exist_ok=True)

    print(f"=== SOM training: {metric} metric, {init} initialization ===")
    data = load_feature_matrix(args.netcdf, args.varname)

    total = len(GRID_SIZES) * len(SEEDS)
    done = 0
    start = time.time()

    for size in GRID_SIZES:
        print(f"\n--- Grid {size}x{size} ---")
        for seed in SEEDS:
            np.random.seed(seed)

            som = MiniSom(x=size, y=size, input_len=data.shape[1],
                          metric=metric, init=init, data=data)
            som.train(data)

            bmus = np.array(som.map_vects(data))

            filename = (f"{prefix}_{size}x{size}_oct_mar_mascara_seed{seed}.npy")
            np.save(os.path.join(outdir, filename), bmus)

            done += 1
            occupied = len(np.unique(bmus[:, 0] * size + bmus[:, 1]))
            elapsed = time.time() - start
            print(f"  seed {seed:>3}: {occupied}/{size * size} nodes occupied "
                  f"| {done}/{total} runs | {elapsed / 60:.1f} min elapsed")

    print(f"\nFinished in {(time.time() - start) / 60:.1f} minutes.")
    print(f"BMU files written to: {outdir}")
