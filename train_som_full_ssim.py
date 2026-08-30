#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final SOM training with the full sliding-window SSIM as BMU criterion.

Trains a Self-Organizing Map on normalized monthly precipitation anomaly
fields, using the Structural Similarity Index computed with sliding windows to
select the Best Matching Unit. Unlike the global SSIM used in the exploratory
phase, local statistics are estimated within Gaussian-weighted moving windows,
so the position and morphology of anomaly cores drive the topological
organization of the network.

Pipeline
--------
1. Read a NetCDF file (first data variable) and vectorize it, using the land
   mask derived from the first time step (NaN marks ocean pixels).
2. Train the network, selecting the BMU by SSIM at every field presentation.
3. Assign final BMUs, again by SSIM.
4. Compute node-mean and global SSIM, plus classical clustering indices.
5. Write per-configuration output and append a row to a summary CSV.

Note on the clustering indices: the Silhouette coefficient reported here is
computed under Euclidean distance only. For comparisons across configurations
trained with different BMU metrics, use recompute_validity_indices.py, which
evaluates every partition under a common set of distances.

Usage
-----
    python train_som_full_ssim.py \
      --netcdf chirps_normalized.nc \
      --grid 4 \
      --init pca \
      --epochs 8 \
      --lr 0.4 \
      --seed 145 \
      --window 11 \
      --sigma 1.5 \
      --out som_results_ssim

Author: Gabriela Hernandez
"""

import os
import sys
import json
import argparse
import numpy as np
import xarray as xr
from scipy import ndimage
import pandas as pd

# Optional, for the clustering indices.
try:
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False

# Optional, for the progress bar.
try:
    from tqdm import tqdm
    HAVE_TQDM = True
except Exception:
    HAVE_TQDM = False
    print("Warning: tqdm not available. Install with: pip install tqdm")


# ------------------------------- SSIM utilities -------------------------------

def create_gaussian_kernel(size=11, sigma=1.5):
    """Build a normalized 2D Gaussian kernel of the given odd size."""
    size = int(size)
    if size % 2 != 1:
        raise ValueError("Window size must be odd (e.g. 11).")
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax, indexing="ij")
    ker = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    ker /= ker.sum()
    return ker


def normalized_convolution(image, kernel, valid_mask):
    """
    Convolve an image while ignoring invalid pixels.

    The kernel is applied both to the image and to the validity mask, and the
    result is divided by the convolved mask. This keeps local statistics
    unbiased near the coastline, where part of the window falls over ocean.

    Parameters
    ----------
    valid_mask : ndarray of bool
        True marks valid (land) pixels.
    """
    vm = (~np.isnan(image)) & valid_mask
    img = np.where(vm, image, 0.0)
    conv_img = ndimage.convolve(img, kernel, mode="reflect")
    conv_msk = ndimage.convolve(vm.astype(float), kernel, mode="reflect")
    conv_msk = np.where(conv_msk > 0, conv_msk, 1.0)
    out = conv_img / conv_msk
    out = np.where(conv_msk > 1e-10, out, np.nan)
    return out


def ssim2d_masked(img1, img2, mask, window_size=11, sigma=1.5, k1=0.01, k2=0.03, L=1.0):
    """
    Structural Similarity Index between two masked 2D fields.

    Local means, variances and covariance are estimated within Gaussian-weighted
    sliding windows, and the resulting SSIM map is averaged over valid pixels.

    Parameters
    ----------
    window_size : int
        Side of the square moving window, in pixels.
    sigma : float
        Standard deviation of the Gaussian weighting inside the window.
    L : float
        Dynamic range of the data. Use 1.0 for fields normalized to [0, 1].

    Returns
    -------
    float
        Mean SSIM over valid pixels; 0.0 if no valid pixel remains.
    """
    if img1.shape != img2.shape or img1.shape != mask.shape:
        raise ValueError("Images and mask must have the same shape.")
    kernel = create_gaussian_kernel(window_size, sigma)
    mu1 = normalized_convolution(img1, kernel, mask)
    mu2 = normalized_convolution(img2, kernel, mask)

    img1_sq = img1 * img1
    img2_sq = img2 * img2
    img1_img2 = img1 * img2

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = normalized_convolution(img1_sq, kernel, mask) - mu1_sq
    sigma2_sq = normalized_convolution(img2_sq, kernel, mask) - mu2_sq
    sigma12 = normalized_convolution(img1_img2, kernel, mask) - mu1_mu2

    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2

    num = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)

    ssim_map = num / (den + 1e-12)
    valid = ~np.isnan(ssim_map)
    return float(np.mean(ssim_map[valid])) if valid.any() else 0.0


# --------------------------------- Data loading -------------------------------

def load_and_preprocess_data(netcdf_file):
    """
    Read the NetCDF and vectorize it over the land mask.

    The mask is taken from the first time step, where NaN marks ocean.

    Returns
    -------
    data_vec : ndarray of shape (n_time, n_valid)
    valid_idx : tuple of ndarray
        Row and column indices of the valid pixels, used to map vectors back
        onto the 2D grid.
    spatial_shape : tuple
        Shape of the original 2D field.
    """
    ds = xr.open_dataset(netcdf_file)
    varname = list(ds.data_vars)[0]
    data = ds[varname].values  # (time, y, x)
    first = data[0]
    valid_mask = ~np.isnan(first)
    valid_idx = np.where(valid_mask)
    n_times = data.shape[0]
    n_valid = int(valid_mask.sum())
    data_vec = np.zeros((n_times, n_valid), dtype=np.float32)
    for t in range(n_times):
        data_vec[t] = data[t][valid_mask]
    spatial_shape = first.shape
    return data_vec, valid_idx, spatial_shape


def vector_to_2d(vec, valid_idx, spatial_shape):
    """Map a vector of valid pixels back onto the 2D grid, filling with NaN."""
    arr = np.full(spatial_shape, np.nan, dtype=np.float32)
    arr[valid_idx] = vec
    return arr


# ------------------------------------- SOM ------------------------------------

class SOM:
    """
    Self-Organizing Map with a selectable BMU criterion.

    Training is epoch-based and sequential: all fields are presented once per
    epoch, and prototypes are updated after each field. The learning rate and
    the neighbourhood radius decay linearly with epoch.

    Parameters
    ----------
    grid_size : tuple of int
        Network dimensions (rows, columns).
    learning_rate : float
        Initial learning rate.
    sigma : float or None
        Initial neighbourhood radius in nodes. Defaults to max(grid_size)/2.
    bmu_metric : {'ssim', 'euclidean'}
        Criterion used to select the Best Matching Unit.
    valid_idx, spatial_shape : required when bmu_metric is 'ssim'
        Needed to map prototype vectors back onto the 2D grid.
    L : float
        Dynamic range passed to the SSIM.
    init_method : {'pca', 'random'}
        Prototype initialization scheme.
    """

    def __init__(self, grid_size, input_dim, learning_rate=0.5, sigma=None, seed=145,
                 valid_idx=None, spatial_shape=None, bmu_metric='ssim',
                 ssim_window=11, ssim_sigma=1.5, L=1.0, init_method='random'):
        rng = np.random.default_rng(seed)
        self.grid_size = (int(grid_size[0]), int(grid_size[1]))
        self.input_dim = int(input_dim)
        self.initial_learning_rate = float(learning_rate)
        self.initial_sigma = float(sigma) if sigma is not None else max(self.grid_size) / 2.0
        self.weights = rng.random((self.grid_size[0], self.grid_size[1], self.input_dim), dtype=np.float32)
        self.coords = np.array([[i, j] for i in range(self.grid_size[0]) for j in range(self.grid_size[1])], dtype=np.float32)

        # SSIM configuration
        self.bmu_metric = bmu_metric.lower()
        self.valid_idx = valid_idx
        self.spatial_shape = spatial_shape
        self.mask2d = None
        if (valid_idx is not None) and (spatial_shape is not None):
            mask = np.zeros(spatial_shape, dtype=bool)
            mask[valid_idx] = True
            self.mask2d = mask
        self.ssim_window = int(ssim_window)
        self.ssim_sigma = float(ssim_sigma)
        self.L = float(L)

        if init_method == 'pca':
            self.initialize_pca = True
        else:
            self.initialize_pca = False

    def _init_with_pca(self, data):
        """
        Place prototypes along the first two principal components.

        PCA is computed via SVD to avoid a dependency on scikit-learn. The
        reconstructed prototypes are clipped to [0, 1] to stay within the range
        of the normalized input data.
        """
        X = data[~np.isnan(data).any(axis=1)]
        if X.shape[0] < 2:
            return
        Xc = X - X.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        Vt = Vt[:2, :]  # two components
        mean = X.mean(axis=0)
        pc1_range = np.linspace(-2, 2, self.grid_size[0])
        pc2_range = np.linspace(-2, 2, self.grid_size[1])
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                coeffs = np.array([pc1_range[i], pc2_range[j]])
                if Vt.shape[0] == 2:
                    recon = mean + coeffs[0] * Vt[0] + coeffs[1] * Vt[1]
                else:
                    recon = mean + coeffs[0] * Vt[0]
                self.weights[i, j] = np.clip(recon, 0.0, 1.0).astype(np.float32)

    def gaussian_neighborhood(self, bmu_idx, sigma):
        """Return the Gaussian influence of the BMU over every node."""
        bmu = np.array(bmu_idx, dtype=np.float32)
        d2 = np.sum((self.coords - bmu)**2, axis=1)
        neigh = np.exp(-d2 / (2.0 * (sigma**2)))
        return neigh.reshape(self.grid_size)

    def _find_bmu_euclidean(self, input_vector):
        d2 = np.sum((self.weights - input_vector)**2, axis=2)
        return np.unravel_index(np.argmin(d2), d2.shape)

    def _find_bmu_ssim(self, input_vector):
        """
        Select the BMU by maximum SSIM.

        Both the input field and every prototype are mapped back onto the 2D
        grid, since the sliding-window SSIM requires spatial structure. The
        node minimizing 1 - SSIM is returned.
        """
        if self.mask2d is None or self.spatial_shape is None or self.valid_idx is None:
            raise ValueError("SSIM BMU requires valid_idx and spatial_shape.")
        in2d = vector_to_2d(input_vector, self.valid_idx, self.spatial_shape)
        best = None
        best_dist = np.inf
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                w2d = vector_to_2d(self.weights[i, j], self.valid_idx, self.spatial_shape)
                s = ssim2d_masked(in2d, w2d, self.mask2d,
                                  window_size=self.ssim_window,
                                  sigma=self.ssim_sigma, L=self.L)
                dist = 1.0 - s
                if dist < best_dist:
                    best_dist = dist
                    best = (i, j)
        return best

    def find_bmu(self, input_vector, metric=None):
        m = (metric or self.bmu_metric).lower()
        if m == 'euclidean':
            return self._find_bmu_euclidean(input_vector)
        elif m == 'ssim':
            return self._find_bmu_ssim(input_vector)
        else:
            raise ValueError("Unsupported BMU metric: use 'euclidean' or 'ssim'.")

    def train(self, data, epochs=8, metric=None):
        """
        Train the network for the requested number of epochs.

        Each epoch presents all fields once, in order, updating the prototypes
        after every field. The learning rate and the neighbourhood radius both
        decay linearly with epoch, reaching lr/8 and sigma/8 of their initial
        values in the last epoch when epochs = 8.
        """
        metric = (metric or self.bmu_metric).lower()
        if self.initialize_pca:
            print("   - PCA initialization")
            self._init_with_pca(data)
        else:
            print("   - Random initialization")

        for epoch in range(epochs):
            lr = self.initial_learning_rate * (1.0 - epoch / max(1, epochs))
            sig = self.initial_sigma * (1.0 - epoch / max(1, epochs))
            print(f"[epoch {epoch+1}/{epochs}] lr={lr:.4f} sigma={sig:.3f}")

            iterator = enumerate(data)
            if HAVE_TQDM:
                iterator = tqdm(iterator, total=len(data), desc="  Training", leave=False)

            for t, vec in iterator:
                if np.isnan(vec).any():
                    continue
                bmu = self.find_bmu(vec, metric=metric)
                neigh = self.gaussian_neighborhood(bmu, sig)
                for i in range(self.grid_size[0]):
                    for j in range(self.grid_size[1]):
                        self.weights[i, j] += lr * neigh[i, j] * (vec - self.weights[i, j])

    def get_bmus(self, data, metric=None):
        """Assign a BMU to every field. Fields containing NaN return (-1, -1)."""
        metric = (metric or self.bmu_metric).lower()
        b = []

        iterator = data
        if HAVE_TQDM:
            iterator = tqdm(data, desc="  Computing BMUs", leave=False)

        for vec in iterator:
            if np.isnan(vec).any():
                b.append((-1, -1))
            else:
                b.append(self.find_bmu(vec, metric=metric))
        return np.array(b, dtype=int)


# -------------------------------- SSIM metrics --------------------------------

def calculate_ssim_metrics(data, bmus, weights, valid_idx, spatial_shape,
                           window=11, sigma=1.5, L=1.0):
    """
    Structural coherence of the trained map.

    Returns
    -------
    avg_intra : float
        Node-mean SSIM: the mean SSIM between each field and the prototype of
        its node, averaged across nodes with equal weight.
    global_ssim : float
        Mean SSIM over all field-prototype pairs, weighting each field equally.

    Both are SSIM-based, so they compare configurations trained with SSIM
    rather than across BMU metrics.
    """
    mask = np.zeros(spatial_shape, dtype=bool)
    mask[valid_idx] = True
    groups = {}
    for k, (i, j) in enumerate(bmus):
        if i < 0 or j < 0:
            continue
        groups.setdefault((i, j), []).append(k)
    intra = []
    allv = []
    for (i, j), idxs in groups.items():
        w2d = vector_to_2d(weights[i, j], valid_idx, spatial_shape)
        vals = []
        for k in idxs:
            d2d = vector_to_2d(data[k], valid_idx, spatial_shape)
            s = ssim2d_masked(d2d, w2d, mask, window_size=window, sigma=sigma, L=L)
            vals.append(s)
            allv.append(s)
        if vals:
            intra.append(float(np.mean(vals)))
    avg_intra = float(np.mean(intra)) if intra else 0.0
    global_ssim = float(np.mean(allv)) if allv else 0.0
    return avg_intra, global_ssim


# ------------------------------ Clustering metrics ----------------------------

def calculate_clustering_metrics(data, bmus):
    """
    Classical internal validity indices for the resulting partition.

    The Silhouette coefficient is computed under Euclidean distance, which is
    the scikit-learn default. See the module docstring for why this value
    should not be used to compare configurations trained with different BMU
    metrics.
    """
    if not HAVE_SKLEARN:
        return None, None, None
    idx = np.where((bmus[:, 0] >= 0) & (bmus[:, 1] >= 0))[0]
    if idx.size < 2:
        return 0.0, float('inf'), 0.0
    X = data[idx]
    # Map BMU pairs (i, j) onto flat integer labels.
    labels_pairs = [tuple(x) for x in bmus[idx]]
    unique_labels = sorted(set(labels_pairs))
    map_lab = {lab: i for i, lab in enumerate(unique_labels)}
    labels = np.array([map_lab[p] for p in labels_pairs])
    if np.unique(labels).size < 2:
        return 0.0, float('inf'), 0.0
    sil = silhouette_score(X, labels)
    db = davies_bouldin_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    return float(sil), float(db), float(ch)


# ----------------------------------- Output -----------------------------------

def save_results(output_dir, name, bmus, weights, valid_idx, config, row):
    """Write BMUs, prototype weights, the land-mask indices and the config."""
    d = os.path.join(output_dir, name)
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "BMUs.npy"), bmus)
    np.save(os.path.join(d, "weights.npy"), weights)
    np.save(os.path.join(d, "valid_idx.npy"), np.array(valid_idx, dtype=object))
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    return row


# ------------------------------ Input validation ------------------------------

def validate_args(args):
    """Check the input parameters before any processing begins."""
    errors = []

    if not os.path.exists(args.netcdf):
        errors.append(f"ERROR: NetCDF file not found: {args.netcdf}")

    if args.grid <= 0:
        errors.append(f"ERROR: grid must be positive, got: {args.grid}")

    if args.epochs <= 0:
        errors.append(f"ERROR: epochs must be positive, got: {args.epochs}")

    if not (0 < args.lr <= 1):
        errors.append(f"ERROR: learning rate must lie in (0, 1], got: {args.lr}")

    if args.window % 2 == 0:
        errors.append(f"ERROR: window size must be odd (e.g. 11, 13, 15), got: {args.window}")

    if args.window < 3:
        errors.append(f"ERROR: window size must be >= 3, got: {args.window}")

    if args.sigma <= 0:
        errors.append(f"ERROR: sigma must be positive, got: {args.sigma}")

    if args.L <= 0:
        errors.append(f"ERROR: L (dynamic range) must be positive, got: {args.L}")

    if errors:
        print("=" * 72)
        print("Invalid parameters")
        print("=" * 72)
        for err in errors:
            print(err)
        print("=" * 72)
        sys.exit(1)

    # Non-fatal warnings.
    warnings = []

    if args.grid > 10:
        warnings.append(f"Warning: grid {args.grid}x{args.grid} = {args.grid**2} nodes. "
                        f"Training may be very slow with SSIM.")

    if args.window > 21:
        warnings.append(f"Warning: window size {args.window} is large and may oversmooth.")

    if warnings:
        print("\nWarnings:")
        for warn in warnings:
            print(warn)
        print()


# ------------------------------------ Main ------------------------------------

def main():
    ap = argparse.ArgumentParser(description="SOM with a sliding-window SSIM BMU criterion.")
    ap.add_argument("--netcdf", required=True,
                    help="Path to the NetCDF file (e.g. chirps_normalized.nc).")
    ap.add_argument("--grid", type=int, default=4, help="Network size (grid x grid).")
    ap.add_argument("--init", choices=["random", "pca"], default="pca",
                    help="Prototype initialization scheme.")
    ap.add_argument("--epochs", type=int, default=8, help="Training epochs.")
    ap.add_argument("--lr", type=float, default=0.4, help="Initial learning rate.")
    ap.add_argument("--seed", type=int, default=145, help="Random seed.")
    ap.add_argument("--window", type=int, default=11, help="SSIM window size in pixels.")
    ap.add_argument("--sigma", type=float, default=1.5, help="Gaussian sigma for the SSIM window.")
    ap.add_argument("--L", type=float, default=1.0,
                    help="SSIM dynamic range (1.0 for data normalized to [0, 1]).")
    ap.add_argument("--out", default="som_results_ssim", help="Output directory.")
    args = ap.parse_args()

    print("=" * 72)
    print("SOM training with sliding-window SSIM")
    print("=" * 72)
    print(f"NetCDF: {args.netcdf}")
    print(f"Grid: {args.grid}x{args.grid} | Init: {args.init} | Epochs: {args.epochs} "
          f"| LR: {args.lr} | Seed: {args.seed}")
    print(f"SSIM window={args.window} sigma={args.sigma} L={args.L}")
    print(f"Output: {args.out}")
    print("-" * 72)

    validate_args(args)

    # 1) Load data
    print("[1/5] Loading and preprocessing data...")
    try:
        data_vectors, valid_idx, spatial_shape = load_and_preprocess_data(args.netcdf)
    except Exception as e:
        print(f"ERROR: could not load NetCDF: {e}")
        sys.exit(1)

    print(f"   time={data_vectors.shape[0]}  valid_pixels={data_vectors.shape[1]}  "
          f"shape={spatial_shape}")
    print(f"   approximate memory: {data_vectors.nbytes / 1024**2:.1f} MB")

    # 2) Prepare output
    os.makedirs(args.out, exist_ok=True)

    # 3) Build the SOM
    print("[2/5] Initializing SOM...")
    som = SOM(
        grid_size=(args.grid, args.grid),
        input_dim=data_vectors.shape[1],
        learning_rate=args.lr,
        seed=args.seed,
        valid_idx=valid_idx,
        spatial_shape=spatial_shape,
        bmu_metric='ssim',
        ssim_window=args.window,
        ssim_sigma=args.sigma,
        L=args.L,
        init_method=args.init
    )

    # 4) Train
    print("[3/5] Training (BMU by SSIM)...")
    som.train(data_vectors, epochs=args.epochs, metric='ssim')

    # 5) Final assignment and metrics
    print("[4/5] Computing BMUs and metrics...")
    bmus = som.get_bmus(data_vectors, metric='ssim')
    avg_intra, global_ssim = calculate_ssim_metrics(
        data_vectors, bmus, som.weights, valid_idx, spatial_shape,
        window=args.window, sigma=args.sigma, L=args.L
    )
    sil, db, ch = calculate_clustering_metrics(data_vectors, bmus)

    # 6) Save
    print("[5/5] Writing results...")
    cfg_name = f"som_{args.grid}x{args.grid}_{args.init}_bmu-ssim"
    config = {
        "grid_size": [args.grid, args.grid],
        "initialization": args.init,
        "bmu_metric": "ssim",
        "epochs": args.epochs,
        "seed": args.seed,
        "learning_rate": args.lr,
        "ssim_window": args.window,
        "ssim_sigma": args.sigma,
        "L": args.L,
        "spatial_shape": spatial_shape,
        "n_valid_cells": int(data_vectors.shape[1]),
        "n_time_steps": int(data_vectors.shape[0]),
        "netcdf": args.netcdf,
    }
    row = {
        "configuration": cfg_name,
        "grid": f"{args.grid}x{args.grid}",
        "init": args.init,
        "bmu_metric": "ssim",
        "epochs": args.epochs,
        "seed": args.seed,
        "lr": args.lr,
        "avg_intra_ssim": avg_intra,
        "global_ssim": global_ssim,
        "silhouette_euclidean": sil if sil is not None else np.nan,
        "davies_bouldin": db if db is not None else np.nan,
        "calinski_harabasz": ch if ch is not None else np.nan,
    }
    save_results(args.out, cfg_name, bmus, som.weights, valid_idx, config, row)

    # Append to the summary CSV.
    summary_csv = os.path.join(args.out, "scores.csv")
    df = pd.DataFrame([row])
    if os.path.exists(summary_csv):
        old = pd.read_csv(summary_csv)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(summary_csv, index=False)

    print("-" * 72)
    print("Results:")
    print(f"  Node-mean SSIM      : {avg_intra:.4f}")
    print(f"  Global SSIM         : {global_ssim:.4f}")
    if sil is not None:
        print(f"  Silhouette (Euclid) : {sil:.4f}")
        print(f"  Davies-Bouldin      : {db:.4f}")
        print(f"  Calinski-Harabasz   : {ch:.0f}")
    else:
        print("  (scikit-learn not available: clustering metrics skipped)")
    print(f"  Written to          : {args.out}/{cfg_name}/")
    print(f"  Summary CSV         : {summary_csv}")
    print("=" * 72)
    print("Done.")


if __name__ == "__main__":
    main()
