# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 22:15:32 2025

@author: Gaby
"""

import os, json, glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import colormaps as cm
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter



# ==========================
# ========= CONFIG =========
# ==========================
BASE = r"C:\Users\Gaby\Desktop\SOM\SSIM-COMPLETO\PROYECTO_GABY"

RESULTS_DIR = os.path.join(BASE, "som_results_ssim")
CFG_NAME    = "som_4x4_pca_bmu-ssim"  # si dudás, dejalo "" y autodetecta

# NetCDF para obtener las FECHAS del SOM (solo coord 'time')
TIME_NC = os.path.join(BASE, "chirps_normalized.nc")

# NetCDF de ANOMALÍAS EN mm (el que querés promediar por nodo)
ANOM_NC = os.path.join(BASE, "chirps_anom_oct_mar.nc")

# Salida
OUT_DIR = RESULTS_DIR

# Estética
VMIN, VMAX = -150, 150
LEVELS = np.linspace(VMIN, VMAX, 21)
CMAP = cm.get_cmap("RdBu").copy(); CMAP.set_bad("white")
FIGSIZE = (6.0, 8.0)
DPI = 160

# ==========================
# ====== UTILIDADES  =======
# ==========================
def autodetect_cfg_dir(results_dir, cfg_name):
    if cfg_name:
        p = os.path.join(results_dir, cfg_name)
        if os.path.exists(os.path.join(p, "BMUs.npy")):
            return p
    cands = []
    for d in glob.glob(os.path.join(results_dir, "*")):
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "BMUs.npy")):
            cands.append(d)
    if not cands:
        raise FileNotFoundError("No hay subcarpetas con BMUs.npy dentro de som_results_ssim.")
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]

def pick_name(ds, candidates):
    return next((n for n in candidates if n in ds.variables or n in ds.coords or n in ds.dims), None)

def ajustar_lon_m180_180(da, lon_name):
    lon2 = ((da[lon_name] + 180) % 360) - 180
    return da.assign_coords({lon_name: lon2}).sortby(lon_name)

def build_year_month_mask(ds, tname, fechas_objetivo):
    """Máscara booleana para seleccionar por (año, mes) — soporta cftime y datetime64."""
    ym_set = {(f.year, f.month) for f in fechas_objetivo}
    ts = ds[tname].values  # np.datetime64 o cftime
    mask = []
    for t in ts:
        if hasattr(t, "year"):  # cftime
            y, m = t.year, t.month
        else:                   # datetime64
            s = np.datetime_as_string(t, unit='D')
            y, m = int(s[:4]), int(s[5:7])
        mask.append((y, m) in ym_set)
    return np.array(mask, dtype=bool)

def gridlines_labels(ax):
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                      alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = True
    gl.left_labels = True
    gl.xlabel_style = {'size': 11}
    gl.ylabel_style = {'size': 11}
    gl.xformatter = FuncFormatter(lambda x, pos: f"{int(x)}°")
    gl.yformatter = FuncFormatter(lambda y, pos: f"{int(y)}°")
    return gl

def plot_mapa(ax, LON, LAT, data2d, titulo):
    ax.set_global()
    gl = gridlines_labels(ax)
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.6)
    ax.add_feature(cfeature.STATES, linestyle=':', edgecolor='gray', linewidth=0.4)

    data2d = np.ma.masked_invalid(data2d)
    im = ax.contourf(LON, LAT, data2d, levels=LEVELS, cmap=CMAP,
                     norm=Normalize(VMIN, VMAX), transform=ccrs.PlateCarree(),
                     extend='both', antialiased=True)

    ax.set_extent([np.nanmin(LON), np.nanmax(LON), np.nanmin(LAT), np.nanmax(LAT)],
                  crs=ccrs.PlateCarree())
    ax.set_title(titulo, fontsize=13)
    return im

# ==========================
# =========== MAIN =========
# ==========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- CFG + BMUs
    cfg_dir = autodetect_cfg_dir(RESULTS_DIR, CFG_NAME)
    cfg_name = os.path.basename(cfg_dir)

    bmus = np.load(os.path.join(cfg_dir, "BMUs.npy"))
    with open(os.path.join(cfg_dir, "config.json")) as f:
        cfg = json.load(f)
    rows, cols = cfg["grid_size"]

    # --- Fechas del SOM
    ds_time = xr.open_dataset(TIME_NC)
    if "time" not in ds_time.coords:
        raise ValueError("El NetCDF de tiempo no tiene coord 'time'.")
    tvec = pd.to_datetime(ds_time["time"].values)
    # seguridad por si difieren longitudes
    nuse = min(len(tvec), len(bmus))
    tvec = tvec[:nuse]
    bmus = bmus[:nuse]
    # restringir a Oct–Mar (igual que tu entrenamiento/uso)
    is_oct_mar = (tvec.month >= 10) | (tvec.month <= 3)
    tvec_om = tvec[is_oct_mar]
    bmus_om = bmus[is_oct_mar]

    # Fechas por nodo
    fechas_por_nodo = {}
    for k, (i, j) in enumerate(bmus_om):
        fechas_por_nodo.setdefault((int(i), int(j)), []).append(tvec_om[k])

    # --- Abrir Anomalías (mm)
    ds = xr.open_dataset(ANOM_NC, decode_times=True, use_cftime=True)
    vname = pick_name(ds, ["precip","pr","anom","anomaly","chirps","variable"])
    if vname is None:
        # si no encuentra, toma la primera variable de datos
        for v in ds.data_vars:
            vname = v; break
    da = ds[vname]              # dims típicas: (time, lat, lon)
    latn = pick_name(ds, ["latitude","lat","y"])
    lonn = pick_name(ds, ["longitude","lon","x"])
    timn = pick_name(ds, ["time","t"])
    if not all([latn, lonn, timn]):
        raise ValueError("No pude identificar lat/lon/time en ANOM_NC.")

    lats = ds[latn].values
    lons = ds[lonn].values
    # LON/LAT grid para plot
    if lats.ndim == 1 and lons.ndim == 1:
        LON, LAT = np.meshgrid(lons, lats)
    else:
        LON, LAT = lons, lats

    # Máscara de océano: NaN donde todo es NaN en el tiempo
    mask_allnan = ~np.isfinite(da.max(dim=timn, skipna=True))
    # (usamos esto para que el promedio deje blanco el océano)
    out_cfg = os.path.join(OUT_DIR, cfg_name)
    os.makedirs(out_cfg, exist_ok=True)

    # --- Loop por nodo: promedio de anomalías
    resumen = []  # para CSV con cantidad de meses por nodo
    for i in range(rows):
        for j in range(cols):
            fechas = fechas_por_nodo.get((i, j), [])
            if len(fechas) == 0:
                continue

            # máscara año-mes sobre ANOM_NC
            mask = build_year_month_mask(ds, timn, fechas)
            if not mask.any():
                print(f"⚠️ Nodo ({i},{j}): no hay coincidencias de fechas en ANOM_NC.")
                continue

            sel = da.isel({timn: mask})
            # promedio (mm)
            prom = sel.mean(dim=timn, skipna=True)
            # aplicar océano a NaN
            prom = prom.where(~mask_allnan)

            # plot
            fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
            ax = plt.axes(projection=ccrs.PlateCarree())
            im = plot_mapa(ax, LON, LAT,
                           np.clip(prom.values, VMIN, VMAX),
                           f"Nodo ({i},{j})")
            cbar = plt.colorbar(im, orientation='horizontal', pad=0.05, extend='both')
            cbar.set_label("Anomalía de precipitación (mm)", fontsize=11)
            plt.tight_layout()

            node_dir = os.path.join(out_cfg, f"node_{i}_{j}")
            os.makedirs(node_dir, exist_ok=True)
            fout = os.path.join(node_dir, f"promedio_nodo_{i}_{j}.png")
            plt.savefig(fout)
            plt.close(fig)

            resumen.append({"nodo_i": i, "nodo_j": j, "n_meses": int(mask.sum())})
            print(f"✓ Nodo ({i},{j})  meses={int(mask.sum())}  →  {os.path.basename(fout)}")

    # CSV con cantidad de meses por nodo (útil para interpretar)
    if resumen:
        
        df = pd.DataFrame(resumen).sort_values(["nodo_i","nodo_j"])
        df.to_csv(os.path.join(out_cfg, f"{cfg_name}_cantidad_meses_por_nodo.csv"), index=False)

    print("Listo. Promedios por nodo en:", out_cfg)

if __name__ == "__main__":
    main()
