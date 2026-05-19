# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 20:37:53 2025

@author: Gaby
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import colormaps as cm
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Rectangle
from scipy.stats import ttest_1samp

# ==========================
# ========= CONFIG =========
# ==========================
BASE = os.path.dirname(os.path.abspath(__file__))

# 1) Resultados SOM
RESULTS_DIR = os.path.join(BASE, "som_results_ssim")
CFG_NAME = "som_6x6_pca_bmu-ssim"   # si dudás, dejalo "" y autodetecta

# 2) NetCDF con la coordenada de tiempo original (para mapear BMUs→fechas)
#    Puede ser chirps_normalized.nc o chirps_anom_oct_mar.nc (el script toma sólo 'time')
TIME_NC = os.path.join(BASE, "chirps_normalized.nc")

# 3) Datos externos
ERSST_NC = os.path.join(BASE, "ersst_v5_anom_1981_2024.nc")  # TSM anomalías (ERSST)
WIND_NC  = os.path.join(BASE, "UyV_global.nc")               # U/V mensual global

# Niveles de viento (top intentará éste; si no existe, usará 250)
LEVEL_TOP_TRY = 200
LEVEL_TOP_FALLBACK = 250
LEVEL_LOW = 850

# Salida
OUT_DIR = os.path.join(BASE, "compuestos_som")

# Estética
TSM_VMIN, TSM_VMAX = -2, 2
CMAP_TSM = cm.get_cmap("RdBu_r").copy(); CMAP_TSM.set_bad("white")
FIGSIZE = (12, 6)
DPI = 220
RECT_CX = (-65, -40, 13, 20)  # rectángulo de referencia (lon, lat, width, height) sobre Sudamérica


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
    lon = da[lon_name]
    lon2 = ((lon + 180) % 360) - 180
    return da.assign_coords({lon_name: lon2}).sortby(lon_name)

def gridlines_labels(ax):
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                      alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = True
    gl.left_labels = True
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}
    gl.xformatter = FuncFormatter(lambda x, pos: f"{int(x)}°")
    gl.yformatter = FuncFormatter(lambda y, pos: f"{int(y)}°")
    return gl

def plot_stream(ax, u, v, title, tsm=None):
    # coords esperadas: (lat, lon) y lon en [-180, 180]
    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.6)
    ax.add_feature(cfeature.STATES, linestyle=':', edgecolor='gray', linewidth=0.4)
    gridlines_labels(ax)

    if tsm is not None:
        tsm.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(),
                            cmap=CMAP_TSM, vmin=TSM_VMIN, vmax=TSM_VMAX,
                            add_labels=False,
                            cbar_kwargs={'label': 'Anomalía TSM (°C)'})
    # streamlines (u, v en la misma malla que lon/lat)
    lon2d, lat2d = np.meshgrid(u['lon'], u['lat'])
    ax.streamplot(lon2d, lat2d, u.values, v.values,
                  transform=ccrs.PlateCarree(),
                  color='k', density=2.4, linewidth=0.8, arrowsize=1)

    # rectángulo de referencia
    x0, y0, w, h = RECT_CX
    rect = Rectangle(xy=(x0, y0), width=w, height=h,
                     linewidth=1.2, edgecolor='green', facecolor='none',
                     transform=ccrs.PlateCarree())
    ax.add_patch(rect)

    # límites por datos
    ax.set_extent([float(u['lon'].min()), float(u['lon'].max()),
                   float(u['lat'].min()), float(u['lat'].max())],
                   crs=ccrs.PlateCarree())
    ax.set_title(title, fontsize=10)

def mean_anom_wind(ds_uv, level, months_mask, clim_period=("1991-01-01","2020-12-31")):
    """
    Promedio de anomalías de viento (u,v) en un nivel dado, usando una máscara booleana
    de tiempos (months_mask) y restando la climatología mensual 1991–2020.
    months_mask debe tener la misma longitud que ds_uv[tname].
    """
    # Detectar nombres
    tname = pick_name(ds_uv, ["valid_time","time","date"])
    latn  = pick_name(ds_uv, ["latitude","lat","y"])
    lonn  = pick_name(ds_uv, ["longitude","lon","x"])
    plev  = pick_name(ds_uv, ["pressure_level","level","plev"])
    if not all([tname, latn, lonn, plev]):
        raise ValueError("No pude detectar nombres de dims en WIND_NC.")

    # Selección nivel
    if level not in ds_uv[plev].values:
        raise KeyError(f"No existe nivel {level} hPa. Revisá el dataset.")
    U = ds_uv['u'].sel({plev: level})
    V = ds_uv['v'].sel({plev: level})

    # Climatología 1991–2020 por mes
    ds_c   = ds_uv.sel({tname: slice(clim_period[0], clim_period[1])})
    clim_u = ds_c['u'].sel({plev: level}).groupby(f"{tname}.month").mean(tname)
    clim_v = ds_c['v'].sel({plev: level}).groupby(f"{tname}.month").mean(tname)

    # Selección por máscara booleana (sin where)
    if len(months_mask) != U.sizes[tname]:
        raise ValueError(f"months_mask (len={len(months_mask)}) no coincide con {tname} (len={U.sizes[tname]}).")
    U_sel = U.isel({tname: months_mask})
    V_sel = V.isel({tname: months_mask})

    # Anomalías por mes y promedio
    U_an = U_sel.groupby(f"{tname}.month") - clim_u
    V_an = V_sel.groupby(f"{tname}.month") - clim_v
    U_m  = U_an.mean(tname)
    V_m  = V_an.mean(tname)

    # Asegurar nombres lat/lon
    if 'lon' not in U_m.coords: U_m = U_m.rename({lonn: 'lon'})
    if 'lat' not in U_m.coords: U_m = U_m.rename({latn: 'lat'})
    if 'lon' not in V_m.coords: V_m = V_m.rename({lonn: 'lon'})
    if 'lat' not in V_m.coords: V_m = V_m.rename({latn: 'lat'})

    # Ajustar longitudes a [-180, 180]
    U_m = ajustar_lon_m180_180(U_m, 'lon')
    V_m = ajustar_lon_m180_180(V_m, 'lon')
    return U_m, V_m

def mean_anom_sst(ds_sst, months_mask, tvec_comunes=None):
    """
    Promedio de anomalías de TSM (ERSST) para los tiempos indicados por months_mask.
    Aplica t-test 1 muestra vs 0 y devuelve sólo valores significativos (p<0.05).
    months_mask debe tener la misma longitud que ds_sst[time].
    """
    # Detectar nombres
    tname = pick_name(ds_sst, ["time"])
    latn  = pick_name(ds_sst, ["latitude","lat"])
    lonn  = pick_name(ds_sst, ["longitude","lon"])
    vname = pick_name(ds_sst, ["ssta","sst","anom"])
    if not all([tname, latn, lonn, vname]):
        raise ValueError("No pude detectar nombres/variable en ERSST_NC.")

    da = ds_sst[vname]
    if 'lev' in da.dims:
        da = da.sel(lev=0)

    # Selección por máscara booleana
    if len(months_mask) != da.sizes[tname]:
        raise ValueError(f"months_mask (len={len(months_mask)}) no coincide con {tname} (len={da.sizes[tname]}).")
    da_sel = da.isel({tname: months_mask})

    # t-test 1 muestra vs 0 + promedio
    arr = da_sel.transpose(tname, latn, lonn).values  # [time, lat, lon]
    tstat, pval = ttest_1samp(arr, popmean=0.0, axis=0, nan_policy='omit')
    mean_anom = np.nanmean(arr, axis=0)

    sst_m = xr.DataArray(mean_anom,
                         coords={latn: ds_sst[latn].values, lonn: ds_sst[lonn].values},
                         dims=(latn, lonn))
    mask_sig = xr.DataArray(pval < 0.05,
                            coords={latn: ds_sst[latn].values, lonn: ds_sst[lonn].values},
                            dims=(latn, lonn))

    # Renombres + longitudes a [-180, 180]
    if (latn != 'lat') or (lonn != 'lon'):
        sst_m    = sst_m.rename({latn: 'lat', lonn: 'lon'})
        mask_sig = mask_sig.rename({latn: 'lat', lonn: 'lon'})
    sst_m    = ajustar_lon_m180_180(sst_m, 'lon')
    mask_sig = ajustar_lon_m180_180(mask_sig, 'lon')

    # Aplicar máscara de significancia
    sst_m_sig = sst_m.where(mask_sig)
    return sst_m_sig

def build_year_month_mask(ds, tname, fechas_objetivo):
    """
    Devuelve un boolean mask para seleccionar en ds[tname] los tiempos cuyo (year, month)
    esté en fechas_objetivo. Funciona con datetime64 y con cftime (360_day).
    """
    ym_set = {(f.year, f.month) for f in fechas_objetivo}
    ts = ds[tname].values  # np.datetime64 o cftime
    mask = []
    for t in ts:
        if hasattr(t, "year"):  # cftime (360_day, etc.)
            y, m = t.year, t.month
        else:                   # datetime64
            s = np.datetime_as_string(t, unit='D')
            y, m = int(s[:4]), int(s[5:7])
        mask.append((y, m) in ym_set)
    return np.array(mask, dtype=bool)

# ==========================
# =========== MAIN =========
# ==========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # CFG
    cfg_dir = autodetect_cfg_dir(RESULTS_DIR, CFG_NAME)
    cfg_name = os.path.basename(cfg_dir)

    # BMUs y grid
    bmus = np.load(os.path.join(cfg_dir, "BMUs.npy"))
    with open(os.path.join(cfg_dir, "config.json")) as f:
        cfg = json.load(f)
    rows, cols = cfg["grid_size"]

    # Fechas base (del NetCDF con 'time')
    ds_time = xr.open_dataset(TIME_NC)
    if "time" not in ds_time.coords:
        raise ValueError("El NetCDF de tiempo no tiene coord 'time'.")
    tvec = pd.to_datetime(ds_time["time"].values)
    nuse = min(len(tvec), len(bmus))
    tvec = tvec[:nuse]
    bmus = bmus[:nuse]

    # Oct–Mar
    is_oct_mar = (tvec.month >= 10) | (tvec.month <= 3)
    tvec_om = tvec[is_oct_mar]
    bmus_om = bmus[is_oct_mar]

    fechas_por_nodo = {}
    for k, (i, j) in enumerate(bmus_om):
        fechas_por_nodo.setdefault((int(i), int(j)), []).append(tvec_om[k])

    # Datasets externos (usar cftime para ERSST por seguridad)
    ds_sst = xr.open_dataset(ERSST_NC, decode_times=True, use_cftime=True)
    ds_uv  = xr.open_dataset(WIND_NC,  decode_times=True, use_cftime=True)

    # Preparar niveles
    plev_name = pick_name(ds_uv, ["pressure_level","level","plev"])
    if plev_name is None:
        raise ValueError("No encontré dimensión de niveles en el archivo de vientos.")
    levels = list(ds_uv[plev_name].values)
    top_level = LEVEL_TOP_TRY if LEVEL_TOP_TRY in levels else (LEVEL_TOP_FALLBACK if LEVEL_TOP_FALLBACK in levels else None)
    if top_level is None:
        raise KeyError("No encontré ni 200 hPa ni 250 hPa en el archivo de vientos.")
    low_level = LEVEL_LOW
    if low_level not in levels:
        raise KeyError(f"No encontré {LEVEL_LOW} hPa en el archivo de vientos.")

    # Procesar por nodo
    out_cfg = os.path.join(OUT_DIR, cfg_name)
    os.makedirs(out_cfg, exist_ok=True)

    for (i, j), fechas in fechas_por_nodo.items():
        if len(fechas) == 0:
            continue

        # Máscara año-mes para TSM y Viento (sin pandas; compatible cftime)
        tname_sst = pick_name(ds_sst, ["time"])
        tname_uv  = pick_name(ds_uv,  ["valid_time","time","date"])
        mask_sst  = build_year_month_mask(ds_sst, tname_sst, fechas)
        mask_uv   = build_year_month_mask(ds_uv,  tname_uv,  fechas)

        if not mask_sst.any() and not mask_uv.any():
            print(f"⚠️ Nodo ({i},{j}): sin coincidencias de fechas en SST ni en Viento. Salto.")
            continue

        # Compuestos (si una máscara está vacía, se salta ese campo)
        sst_comp = None
        if mask_sst.any():
            sst_comp = mean_anom_sst(ds_sst, mask_sst)

        if mask_uv.any():
            u_top, v_top = mean_anom_wind(ds_uv, top_level, mask_uv)
            u_low, v_low = mean_anom_wind(ds_uv, low_level, mask_uv)
        else:
            print(f"⚠️ Nodo ({i},{j}): sin viento para esas fechas. Salto.")
            continue

        # Plots
        # 1) TSM + corrientes en top_level (si hay sst_comp)
        fig1, ax1 = plt.subplots(figsize=FIGSIZE,
                                 subplot_kw={'projection': ccrs.PlateCarree()})
        plot_stream(ax1, u_top, v_top,
                    f"TSM + Corriente {top_level} hPa — Nodo ({i},{j}) — {cfg_name}",
                    tsm=sst_comp)
        f1 = os.path.join(out_cfg, f"TSM_stream{top_level}_nodo_{i}_{j}.png")
        plt.savefig(f1, dpi=DPI, bbox_inches="tight")
        plt.close(fig1)

        # 2) Corrientes 850 hPa
        fig2, ax2 = plt.subplots(figsize=FIGSIZE,
                                 subplot_kw={'projection': ccrs.PlateCarree()})
        plot_stream(ax2, u_low, v_low,
                    f"Corriente {low_level} hPa — Nodo ({i},{j}) — {cfg_name}",
                    tsm=None)
        f2 = os.path.join(out_cfg, f"stream{low_level}_nodo_{i}_{j}.png")
        plt.savefig(f2, dpi=DPI, bbox_inches="tight")
        plt.close(fig2)

        print(f"✓ Nodo ({i},{j}) → {os.path.basename(f1)}, {os.path.basename(f2)}")

    print("Listo. PNGs en:", out_cfg)

if __name__ == "__main__":
    main()
