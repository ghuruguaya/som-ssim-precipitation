#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SOM con BMU por SSIM (ventana 11x11) — Script único para servidor (Granito)

- Carga un NetCDF (primera variable del dataset).
- Preprocesa: vectoriza usando máscara de la primera fecha (NaN = océano).
- ENTRENAMIENTO: el BMU se elige con **SSIM** (no Euclídea).
- ASIGNACIÓN final: BMUs también con **SSIM**.
- Métricas: Silhouette/Davies/Calinski (sobre vectores) + SSIM intra-nodo y global.
- Guarda resultados por configuración + resumen CSV.

Ejemplo de uso:
  python som_ssim_granito.py \
    --netcdf chirps_normalized.nc \
    --grid 4 \
    --init pca \
    --epochs 10 \
    --lr 0.4 \
    --seed 145 \
    --window 11 \
    --sigma 1.5 \
    --out som_results_ssim
"""

import os
import sys
import json
import argparse
import numpy as np
import xarray as xr
from scipy import ndimage
import pandas as pd

# opcionales (para métricas de clustering)
try:
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False

# opcional (para barra de progreso)
try:
    from tqdm import tqdm
    HAVE_TQDM = True
except Exception:
    HAVE_TQDM = False
    print("⚠ tqdm no disponible. Instalar con: pip install tqdm")


# ------------------------------- Utilidades SSIM -------------------------------

def create_gaussian_kernel(size=11, sigma=1.5):
    size = int(size)
    if size % 2 != 1:
        raise ValueError("window size debe ser impar (p. ej., 11).")
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax, indexing="ij")
    ker = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    ker /= ker.sum()
    return ker

def normalized_convolution(image, kernel, valid_mask):
    # valid_mask True=válido
    vm = (~np.isnan(image)) & valid_mask
    img = np.where(vm, image, 0.0)
    conv_img = ndimage.convolve(img, kernel, mode="reflect")
    conv_msk = ndimage.convolve(vm.astype(float), kernel, mode="reflect")
    conv_msk = np.where(conv_msk > 0, conv_msk, 1.0)
    out = conv_img / conv_msk
    out = np.where(conv_msk > 1e-10, out, np.nan)
    return out

def ssim2d_masked(img1, img2, mask, window_size=11, sigma=1.5, k1=0.01, k2=0.03, L=1.0):
    if img1.shape != img2.shape or img1.shape != mask.shape:
        raise ValueError("Imágenes y máscara deben tener misma forma")
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
    sigma12   = normalized_convolution(img1_img2, kernel, mask) - mu1_mu2

    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2

    num = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)

    ssim_map = num / (den + 1e-12)
    valid = ~np.isnan(ssim_map)
    return float(np.mean(ssim_map[valid])) if valid.any() else 0.0


# ------------------------------- Carga de datos -------------------------------

def load_and_preprocess_data(netcdf_file):
    ds = xr.open_dataset(netcdf_file)
    varname = list(ds.data_vars)[0]
    data = ds[varname].values  # [time, y, x]
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
    arr = np.full(spatial_shape, np.nan, dtype=np.float32)
    arr[valid_idx] = vec
    return arr


# ------------------------------------ SOM -------------------------------------

class SOM:
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

        # SSIM config
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
        # PCA mínimo (SVD) sin dependencia de sklearn
        X = data[~np.isnan(data).any(axis=1)]
        if X.shape[0] < 2:
            return
        Xc = X - X.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        Vt = Vt[:2, :]  # 2 componentes
        mean = X.mean(axis=0)
        pc1_range = np.linspace(-2, 2, self.grid_size[0])
        pc2_range = np.linspace(-2, 2, self.grid_size[1])
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                coeffs = np.array([pc1_range[i], pc2_range[j]])
                if Vt.shape[0] == 2:
                    recon = mean + coeffs[0]*Vt[0] + coeffs[1]*Vt[1]
                else:
                    recon = mean + coeffs[0]*Vt[0]
                self.weights[i, j] = np.clip(recon, 0.0, 1.0).astype(np.float32)

    def gaussian_neighborhood(self, bmu_idx, sigma):
        bmu = np.array(bmu_idx, dtype=np.float32)
        d2 = np.sum((self.coords - bmu)**2, axis=1)
        neigh = np.exp(-d2 / (2.0 * (sigma**2)))
        return neigh.reshape(self.grid_size)

    def _find_bmu_euclidean(self, input_vector):
        d2 = np.sum((self.weights - input_vector)**2, axis=2)
        return np.unravel_index(np.argmin(d2), d2.shape)

    def _find_bmu_ssim(self, input_vector):
        if self.mask2d is None or self.spatial_shape is None or self.valid_idx is None:
            raise ValueError("SSIM BMU requiere valid_idx y spatial_shape.")
        in2d = vector_to_2d(input_vector, self.valid_idx, self.spatial_shape)
        best = None
        best_dist = np.inf
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                w2d = vector_to_2d(self.weights[i, j], self.valid_idx, self.spatial_shape)
                s = ssim2d_masked(in2d, w2d, self.mask2d,
                                  window_size=self.ssim_window,
                                  sigma=self.ssim_sigma, L=self.L)
                dist = 1.0 - s  # minimizar 1-SSIM
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
            raise ValueError("Métrica de BMU no soportada: use 'euclidean' o 'ssim'")

    def train(self, data, epochs=10, metric=None):
        metric = (metric or self.bmu_metric).lower()
        if self.initialize_pca:
            print("   • Inicialización PCA")
            self._init_with_pca(data)
        else:
            print("   • Inicialización aleatoria")

        for epoch in range(epochs):
            lr = self.initial_learning_rate * (1.0 - epoch / max(1, epochs))
            sig = self.initial_sigma * (1.0 - epoch / max(1, epochs))
            print(f"[epoch {epoch+1}/{epochs}] lr={lr:.4f} sigma={sig:.3f}")

            # Barra de progreso si tqdm disponible
            iterator = enumerate(data)
            if HAVE_TQDM:
                iterator = tqdm(iterator, total=len(data), desc=f"  Entrenando", leave=False)

            for t, vec in iterator:
                if np.isnan(vec).any():
                    continue
                bmu = self.find_bmu(vec, metric=metric)   # ← BMU por SSIM aquí
                neigh = self.gaussian_neighborhood(bmu, sig)
                for i in range(self.grid_size[0]):
                    for j in range(self.grid_size[1]):
                        self.weights[i, j] += lr * neigh[i, j] * (vec - self.weights[i, j])

    def get_bmus(self, data, metric=None):
        metric = (metric or self.bmu_metric).lower()
        b = []

        # Barra de progreso si tqdm disponible
        iterator = data
        if HAVE_TQDM:
            iterator = tqdm(data, desc="  Calculando BMUs", leave=False)

        for vec in iterator:
            if np.isnan(vec).any():
                b.append((-1, -1))
            else:
                b.append(self.find_bmu(vec, metric=metric))
        return np.array(b, dtype=int)


# ------------------------------- Métricas SSIM --------------------------------

def calculate_ssim_metrics(data, bmus, weights, valid_idx, spatial_shape,
                           window=11, sigma=1.5, L=1.0):
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


# ------------------------------- Métricas cluster ------------------------------

def calculate_clustering_metrics(data, bmus):
    if not HAVE_SKLEARN:
        return None, None, None
    idx = np.where((bmus[:,0] >= 0) & (bmus[:,1] >= 0))[0]
    if idx.size < 2:
        return 0.0, float('inf'), 0.0
    X = data[idx]
    # pasar BMU (i,j) a etiqueta 1D
    labels_pairs = [tuple(x) for x in bmus[idx]]
    unique_labels = sorted(set(labels_pairs))
    map_lab = {lab:i for i, lab in enumerate(unique_labels)}
    labels = np.array([map_lab[p] for p in labels_pairs])
    if np.unique(labels).size < 2:
        return 0.0, float('inf'), 0.0
    sil = silhouette_score(X, labels)
    db  = davies_bouldin_score(X, labels)
    ch  = calinski_harabasz_score(X, labels)
    return float(sil), float(db), float(ch)


# --------------------------------- Guardado -----------------------------------

def save_results(output_dir, name, bmus, weights, valid_idx, config, row):
    d = os.path.join(output_dir, name)
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "BMUs.npy"), bmus)
    np.save(os.path.join(d, "weights.npy"), weights)
    np.save(os.path.join(d, "valid_idx.npy"), np.array(valid_idx, dtype=object))
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    return row


# --------------------------- Validaciones de entrada --------------------------

def validate_args(args):
    """Valida los parámetros de entrada antes de empezar el procesamiento"""
    errors = []

    # Validar archivo NetCDF
    if not os.path.exists(args.netcdf):
        errors.append(f"✗ Archivo NetCDF no encontrado: {args.netcdf}")

    # Validar grid
    if args.grid <= 0:
        errors.append(f"✗ Grid debe ser positivo, recibido: {args.grid}")

    # Validar epochs
    if args.epochs <= 0:
        errors.append(f"✗ Epochs debe ser positivo, recibido: {args.epochs}")

    # Validar learning rate
    if not (0 < args.lr <= 1):
        errors.append(f"✗ Learning rate debe estar entre 0 y 1, recibido: {args.lr}")

    # Validar window (debe ser impar)
    if args.window % 2 == 0:
        errors.append(f"✗ Window size debe ser impar (ej: 11, 13, 15), recibido: {args.window}")

    if args.window < 3:
        errors.append(f"✗ Window size debe ser >= 3, recibido: {args.window}")

    # Validar sigma
    if args.sigma <= 0:
        errors.append(f"✗ Sigma debe ser positivo, recibido: {args.sigma}")

    # Validar L
    if args.L <= 0:
        errors.append(f"✗ L (rango dinámico) debe ser positivo, recibido: {args.L}")

    # Si hay errores, mostrarlos y salir
    if errors:
        print("=" * 72)
        print("ERROR: Parámetros inválidos")
        print("=" * 72)
        for err in errors:
            print(err)
        print("=" * 72)
        sys.exit(1)

    # Validaciones que son advertencias (no errores fatales)
    warnings = []

    if args.grid > 10:
        warnings.append(f"⚠ Grid {args.grid}x{args.grid} = {args.grid**2} nodos. Procesamiento puede ser muy lento con SSIM.")

    if args.window > 21:
        warnings.append(f"⚠ Window size {args.window} es muy grande. Puede suavizar demasiado.")

    if warnings:
        print("\nAdvertencias:")
        for warn in warnings:
            print(warn)
        print()


# --------------------------------- Main ---------------------------------------

def main():
    ap = argparse.ArgumentParser(description="SOM con BMU por SSIM (Granito)")
    ap.add_argument("--netcdf", required=True, help="Ruta al NetCDF (p. ej. chirps_normalized.nc o chirps_anom_oct_marzo.nc)")
    ap.add_argument("--grid", type=int, default=4, help="Tamaño de la red (grid x grid)")
    ap.add_argument("--init", choices=["random","pca"], default="pca", help="Inicialización")
    ap.add_argument("--epochs", type=int, default=10, help="Épocas de entrenamiento")
    ap.add_argument("--lr", type=float, default=0.4, help="Learning rate inicial")
    ap.add_argument("--seed", type=int, default=145, help="Semilla")
    ap.add_argument("--window", type=int, default=11, help="Tamaño de ventana SSIM")
    ap.add_argument("--sigma", type=float, default=1.5, help="Sigma gauss para SSIM")
    ap.add_argument("--L", type=float, default=1.0, help="Rango dinámico SSIM (1.0 si datos normalizados)")
    ap.add_argument("--out", default="som_results_ssim", help="Carpeta de salida")
    args = ap.parse_args()

    print("="*72)
    print("SOM con BMU por SSIM — Granito")
    print("="*72)
    print(f"NetCDF: {args.netcdf}")
    print(f"Grid: {args.grid}x{args.grid} | Init: {args.init} | Épocas: {args.epochs} | LR: {args.lr} | Seed: {args.seed}")
    print(f"SSIM window={args.window} sigma={args.sigma} L={args.L}")
    print(f"Salida: {args.out}")
    print("-"*72)

    # Validar argumentos
    validate_args(args)

    # 1) Cargar datos
    print("[1/5] Cargando y preprocesando datos...")
    try:
        data_vectors, valid_idx, spatial_shape = load_and_preprocess_data(args.netcdf)
    except Exception as e:
        print(f"✗ Error al cargar NetCDF: {e}")
        sys.exit(1)

    print(f"   ✓ time={data_vectors.shape[0]}  valid_pixels={data_vectors.shape[1]}  shape={spatial_shape}")
    print(f"   ✓ Memoria aprox: {data_vectors.nbytes/1024**2:.1f} MB")

    # 2) Preparar salida
    os.makedirs(args.out, exist_ok=True)

    # 3) Crear SOM
    print("[2/5] Inicializando SOM...")
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

    # 4) Entrenar (BMU por SSIM)
    print("[3/5] Entrenando SOM (BMU=SSIM)...")
    som.train(data_vectors, epochs=args.epochs, metric='ssim')

    # 5) BMUs y métricas
    print("[4/5] Calculando BMUs (SSIM) y métricas...")
    bmus = som.get_bmus(data_vectors, metric='ssim')
    avg_intra, global_ssim = calculate_ssim_metrics(
        data_vectors, bmus, som.weights, valid_idx, spatial_shape,
        window=args.window, sigma=args.sigma, L=args.L
    )
    sil, db, ch = calculate_clustering_metrics(data_vectors, bmus)

    # 6) Guardar
    print("[5/5] Guardando resultados...")
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
        "silhouette": sil if sil is not None else np.nan,
        "davies_bouldin": db if db is not None else np.nan,
        "calinski_harabasz": ch if ch is not None else np.nan,
    }
    save_results(args.out, cfg_name, bmus, som.weights, valid_idx, config, row)

    # CSV resumen (append seguro)
    summary_csv = os.path.join(args.out, "scores.csv")
    df = pd.DataFrame([row])
    if os.path.exists(summary_csv):
        old = pd.read_csv(summary_csv)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(summary_csv, index=False)

    print("-"*72)
    print("Resultados:")
    print(f"  • Avg Intra-SSIM : {avg_intra:.4f}")
    print(f"  • Global SSIM    : {global_ssim:.4f}")
    if sil is not None:
        print(f"  • Silhouette      : {sil:.4f}")
        print(f"  • Davies-Bouldin  : {db:.4f}")
        print(f"  • Calinski-Harab. : {ch:.0f}")
    else:
        print("  • (sklearn no disponible: métricas de clustering omitidas)")
    print(f"  • Guardado en    : {args.out}/{cfg_name}/")
    print(f"  • Resumen CSV    : {summary_csv}")
    print("="*72)
    print("✓ Proceso finalizado.")

if __name__ == "__main__":
    main()
