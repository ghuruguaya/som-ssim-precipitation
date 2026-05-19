# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 08:41:43 2026

@author: Gaby
"""

import os
import json
import numpy as np
import pandas as pd
import xarray as xr

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


# ==========================
# ========= CONFIG =========
# ==========================
BASE = r"C:\Users\Gaby\Desktop\SOM\SSIM-COMPLETO\PROYECTO_GABY"

# NetCDF normalizado (0-1) usado para entrenar el SOM
NC_NORM = os.path.join(BASE, "chirps_normalized.nc")

# Carpeta donde están las configuraciones (cada una con BMUs.npy, valid_idx.npy, config.json)
RESULTS_DIR = os.path.join(BASE, "som_results_ssim")

# Elegí una config puntual (o lista)
CFG_LIST = [
    #"som_3x3_pca_bmu-ssim",
    # "som_4x4_pca_bmu-ssim",
     "som_6x6_pca_bmu-ssim",
]

# Carpeta de salida
OUT_DIR = os.path.join(BASE, "metrics_posthoc")
os.makedirs(OUT_DIR, exist_ok=True)

# Variable en el netcdf (si no estás segura, dejar None y autodetecta)
VAR_NAME = None  # e.g. "precip"


# ==========================
# ====== UTILIDADES ========
# ==========================
def detect_var(ds: xr.Dataset, preferred=None):
    if preferred is not None and preferred in ds.data_vars:
        return preferred
    # primera variable data_var
    return list(ds.data_vars)[0]


def load_cfg(cfg_dir):
    with open(os.path.join(cfg_dir, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    bmus = np.load(os.path.join(cfg_dir, "BMUs.npy"))
    valid_idx = np.load(os.path.join(cfg_dir, "valid_idx.npy"), allow_pickle=True)
    return cfg, bmus, valid_idx


def bmus_to_labels(bmus, grid_size):
    """Convierte pares (i,j) a etiqueta entera 0..K-1."""
    rows, cols = grid_size
    bmus = np.asarray(bmus)
    if bmus.ndim != 2 or bmus.shape[1] != 2:
        raise ValueError(f"BMUs.npy debería ser (n_time, 2). Recibí: {bmus.shape}")
    i = bmus[:, 0].astype(int)
    j = bmus[:, 1].astype(int)
    return i * cols + j


def build_X_from_nc(nc_path, varname, valid_idx, fill_nan=False, fill_value=0.5):
    """
    Construye matriz X (n_time, n_features_valid) a partir de un NetCDF 3D
    y una máscara/índices de píxeles válidos (valid_idx).

    Soporta valid_idx como:
      A) máscara booleana (lat, lon) o (n_total,)
      B) índices planos (n_valid,)
      C) pares (ilat, ilon) como (2, n_valid) o (n_valid, 2), incluso dtype=object

    Parámetros
    ----------
    nc_path : str
        Ruta a NetCDF normalizado (0-1) usado en el SOM.
    varname : str or None
        Nombre de variable (si None autodetecta la primera data_var).
    valid_idx : np.ndarray
        Máscara/índices para seleccionar píxeles terrestres.
    fill_nan : bool
        Si True, reemplaza NaNs en X por fill_value (útil si quedan NaNs residuales).
        Por defecto False: si quedan NaNs, sklearn fallará (y eso ayuda a detectar problemas).
    fill_value : float
        Valor de reemplazo si fill_nan=True. Por defecto 0.5 (neutral en [0,1]).

    Returns
    -------
    X : np.ndarray float32
        Matriz (n_time, n_valid_pixels).
    t : np.ndarray
        Vector tiempo (np.datetime64).
    n_total : int
        Total de píxeles = nlat*nlon.
    vname : str
        Nombre de variable usada.
    """
    ds = xr.open_dataset(nc_path, decode_times=True)
    vname = detect_var(ds, preferred=varname)
    da = ds[vname]

    # Validar dims esperadas
    if "time" not in da.dims:
        ds.close()
        raise ValueError("El NetCDF no tiene dimensión/coord 'time'.")

    # Si usa latitude/longitude (como tu archivo), forzamos este orden
    # (si tuvieras lat/lon, ajustalo acá)
    needed = ["time", "latitude", "longitude"]
    if not all(d in da.dims for d in needed):
        ds.close()
        raise ValueError(f"Dims esperadas {needed}, pero encontré {da.dims}")

    da = da.transpose("time", "latitude", "longitude")

    n_time = da.sizes["time"]
    nlat = da.sizes["latitude"]
    nlon = da.sizes["longitude"]
    n_total = nlat * nlon

    # Vectorizar espacio de forma segura
    da2 = da.stack(space=("latitude", "longitude"))   # dims: (time, space)
    arr2 = da2.values.astype(np.float32)              # (n_time, n_total)

    # ---------- Interpretar valid_idx ----------
    # Lo pasamos a objeto para poder manejar listas/tuplas, etc.
    vi_raw = np.asarray(valid_idx, dtype=object)

    # Caso A: máscara booleana (puede venir bool puro)
    if np.asarray(valid_idx).dtype == bool:
        mask = np.asarray(valid_idx)
        mask = mask.ravel() if mask.ndim == 2 else mask
        if mask.size != n_total:
            ds.close()
            raise ValueError(f"valid_idx booleano size={mask.size} pero n_total={n_total}")
        cols_idx = np.where(mask)[0].astype(int)

    else:
        # Convertimos a numpy "numérico" si está en object.
        # Si es object 2D con pares, .tolist() suele preservar bien.
        try:
            vi = np.array(vi_raw.tolist())
        except Exception:
            vi = np.array(vi_raw)

        # Caso C: pares (ilat, ilon) en (2, N)
        if vi.ndim == 2 and vi.shape[0] == 2:
            ilat = vi[0, :].astype(int)
            ilon = vi[1, :].astype(int)
            cols_idx = ilat * nlon + ilon

        # Caso C alternativo: pares en (N, 2)
        elif vi.ndim == 2 and vi.shape[1] == 2:
            ilat = vi[:, 0].astype(int)
            ilon = vi[:, 1].astype(int)
            cols_idx = ilat * nlon + ilon

        # Caso B: índices planos (N,)
        else:
            cols_idx = np.array(vi_raw).ravel().astype(int)

        # Chequeo de rango
        if cols_idx.size == 0:
            ds.close()
            raise ValueError("cols_idx quedó vacío: valid_idx no contiene píxeles válidos.")

        if cols_idx.min() < 0 or cols_idx.max() >= n_total:
            ds.close()
            raise ValueError(
                f"valid_idx fuera de rango: min={cols_idx.min()} max={cols_idx.max()} n_total={n_total}"
            )

        cols_idx = cols_idx.astype(int)

    # ---------- Selección final (2D garantizado) ----------
    X = arr2[:, cols_idx]  # (n_time, n_valid)

    # Tiempo
    t = ds["time"].values
    ds.close()

    # Debug útil (podés borrarlo cuando esté ok)
    print("DEBUG da shape:", da.shape)
    print("DEBUG arr2 shape:", arr2.shape)
    print("DEBUG valid_idx dtype/ndim:", np.asarray(valid_idx).dtype, np.asarray(valid_idx).ndim)
    print("DEBUG cols_idx shape:", cols_idx.shape, "min/max:", int(cols_idx.min()), int(cols_idx.max()))
    print("DEBUG X shape:", X.shape)
    print("DEBUG NaNs en X:", int(np.isnan(X).sum()))

    # Si querés forzar métricas aunque quede algún NaN residual:
    if fill_nan and np.isnan(X).any():
        X = np.nan_to_num(X, nan=float(fill_value)).astype(np.float32)
        print("DEBUG NaNs reemplazados por", fill_value, "| NaNs finales:", int(np.isnan(X).sum()))

    return X, t, n_total, vname

def safe_metrics(X, labels):
    """
    Calcula métricas internas.
    Maneja el caso de etiquetas degeneradas (p.ej., un solo cluster).
    """
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    if uniq.size < 2:
        return {
            "silhouette": np.nan,
            "davies_bouldin": np.nan,
            "calinski_harabasz": np.nan,
            "n_clusters": int(uniq.size),
            "note": "Menos de 2 clusters presentes; métricas no definidas."
        }

    # Silhouette: para datasets grandes puede tardar; si querés, podés muestrear
    sil = silhouette_score(X, labels, metric="euclidean")
    db = davies_bouldin_score(X, labels)
    ch = calinski_harabasz_score(X, labels)

    return {
        "silhouette": float(sil),
        "davies_bouldin": float(db),
        "calinski_harabasz": float(ch),
        "n_clusters": int(uniq.size),
        "note": ""
    }


# ==========================
# =========== MAIN =========
# ==========================
def main():
    print("=== Métricas post-hoc (Spyder) ===")
    print("NC_NORM:", NC_NORM)
    print("RESULTS_DIR:", RESULTS_DIR)
    print("OUT_DIR:", OUT_DIR)

    for cfg_name in CFG_LIST:
        cfg_dir = os.path.join(RESULTS_DIR, cfg_name)
        print("\n-----------------------------------")
        print("CFG:", cfg_name)
        print("DIR:", cfg_dir)

        if not os.path.isdir(cfg_dir):
            raise FileNotFoundError(f"No existe la carpeta de config: {cfg_dir}")

        # cargar outputs SOM
        cfg, bmus, valid_idx = load_cfg(cfg_dir)
        grid_size = tuple(cfg.get("grid_size", cfg.get("grid", (None, None))))
        if grid_size[0] is None:
            raise ValueError("No encuentro 'grid_size' en config.json")

        # cargar X desde netcdf
        X, tvec, n_total, vname = build_X_from_nc(NC_NORM, VAR_NAME, valid_idx)

        # alinear longitudes (por si hubo recorte Oct–Mar antes de entrenar)
        nuse = min(len(bmus), X.shape[0])
        bmus = bmus[:nuse]
        X = X[:nuse, :]

        labels = bmus_to_labels(bmus, grid_size)

        # calcular métricas
        met = safe_metrics(X, labels)

        # armar registro
        rec = {
            "cfg_name": cfg_name,
            "grid_rows": int(grid_size[0]),
            "grid_cols": int(grid_size[1]),
            "n_time": int(X.shape[0]),
            "n_features_valid": int(X.shape[1]),
            "n_features_total": int(n_total),
            "netcdf_var": vname,
            "silhouette_euclidean": met["silhouette"],
            "davies_bouldin": met["davies_bouldin"],
            "calinski_harabasz": met["calinski_harabasz"],
            "n_clusters_present": met["n_clusters"],
            "note": met["note"],
        }

        # sumar algunos hiperparámetros del config.json si existen
        for k in ["metric", "init_method", "seed", "epochs", "learning_rate", "window", "sigma", "L"]:
            if k in cfg:
                rec[k] = cfg[k]

        # guardar
        out_csv = os.path.join(OUT_DIR, f"{cfg_name}_metrics.csv")
        out_json = os.path.join(OUT_DIR, f"{cfg_name}_metrics.json")

        pd.DataFrame([rec]).to_csv(out_csv, index=False)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)

        print("✓ Guardado:", out_csv)
        print("✓ Guardado:", out_json)
        print("  Silhouette:", rec["silhouette_euclidean"])
        print("  DB:", rec["davies_bouldin"])
        print("  CH:", rec["calinski_harabasz"])

    print("\nListo.")


if __name__ == "__main__":
    main()