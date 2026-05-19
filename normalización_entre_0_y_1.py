# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 13:07:45 2025

@author: Gaby
"""

# -*- coding: utf-8 -*-
"""
Prepara CHIRPS mensual para SOM: detrend, anomalías vs. clim 1991–2020,
recorte Oct–Mar y normalización 0–1 por píxel (usando solo Oct–Mar).
Evita errores de tipos (datetime vs float) y 'time_bnds'.
"""

import os
import numpy as np
import xarray as xr

# ================== RUTAS ==================
BASE_DIR = r"C:\Users\Gaby\Desktop\SOM"
FN_RAW   = os.path.join(BASE_DIR, "chirps_mensual.nc")
FN_DET   = os.path.join(BASE_DIR, "chirps_mensual_detrend.nc")   # se usa si existe; si no, se crea
FN_CLIM  = os.path.join(BASE_DIR, "climatologia_mensual_1991_2020.nc")
FN_OUT   = os.path.join(BASE_DIR, "chirps_normalized.nc")
FN_ANOM  = os.path.join(BASE_DIR, "chirps_anom_oct_mar.nc")      # opcional: anomalías sin normalizar

# ================== CONFIG ==================
VAR = "precip"                  # fuerza la variable correcta
MONTHS_OM = [10, 11, 12, 1, 2, 3]
COMPRESSION = {VAR: dict(zlib=True, complevel=4)}
SAVE_ANOMALIES = True           # guarda también las anomalías Oct–Mar (no normalizadas)

# ================ HELPERS ===================
def drop_time_bnds(ds):
    """Quita variable 'time_bnds' si existe (evita mezclar fechas)."""
    return ds.drop_vars("time_bnds") if "time_bnds" in ds.variables else ds

def ensure_time_sorted(ds):
    """Ordena por tiempo si existe coord 'time'."""
    return ds.sortby("time") if "time" in ds.coords else ds

def detrend_linear_safe(da, dim="time"):
    """
    Detrend lineal por píxel usando un índice numérico (0..N-1) como predictor.
    Evita mezclar datetime con float (fuente del UFuncTypeError).
    """
    tnum = xr.DataArray(np.arange(da.sizes[dim]), dims=dim, coords={dim: da[dim]})
    fit = da.polyfit(dim=dim, deg=1, skipna=True, coord=tnum)
    trend = xr.polyval(tnum, fit.polyfit_coefficients)
    return da - trend

def interp_to(ds_src, ds_ref):
    """
    Interpola ds_src a la grilla de ds_ref si difieren latitude/longitude.
    Mantiene nombres de coords ('latitude', 'longitude') según tus archivos.
    """
    same_lat = np.array_equal(ds_src["latitude"].values,  ds_ref["latitude"].values)
    same_lon = np.array_equal(ds_src["longitude"].values, ds_ref["longitude"].values)
    if same_lat and same_lon:
        return ds_src
    return ds_src.interp(latitude=ds_ref["latitude"], longitude=ds_ref["longitude"])

def minmax_normalize_per_pixel(da_time_cut):
    """
    Normaliza a [0,1] por píxel usando SOLO los datos del período recortado (Oct–Mar).
    Sin rango (max==min) -> 0.5. Si toda la serie del píxel es NaN -> permanece NaN.
    """
    vmin  = da_time_cut.min("time", skipna=True)
    vmax  = da_time_cut.max("time", skipna=True)
    denom = vmax - vmin

    norm = (da_time_cut - vmin) / xr.where(denom > 0, denom, np.nan)
    # donde no hay rango, va 0.5
    norm = norm.where(~np.isnan(norm), 0.5)

    # si todo es NaN en el tiempo para el píxel, conservar NaN
    all_nan = da_time_cut.isnull().all("time")
    norm = norm.where(~all_nan)

    # asegurar [0,1]
    norm = xr.where(norm < 0, 0, xr.where(norm > 1, 1, norm))
    return norm

# ================== PIPELINE ==================
print("=== Preparación CHIRPS para SOM (anomalias, Oct–Mar, 0–1) ===")

# 1) Abrir dataset base (usar detrend existente si está; si no, calcular desde RAW)
if os.path.isfile(FN_DET):
    print(f"Usando detrendeado existente: {FN_DET}")
    ds = xr.open_dataset(FN_DET)
    ds = drop_time_bnds(ds)
    ds = ensure_time_sorted(ds)
    assert VAR in ds.data_vars, f"La variable '{VAR}' no está en {FN_DET}"
    da = ds[VAR]
else:
    print(f"Detrendiendo a partir de: {FN_RAW}")
    ds_raw = xr.open_dataset(FN_RAW)
    ds_raw = drop_time_bnds(ds_raw)
    ds_raw = ensure_time_sorted(ds_raw)
    assert VAR in ds_raw.data_vars, f"La variable '{VAR}' no está en {FN_RAW}"
    da_raw = ds_raw[VAR]
    da_det = detrend_linear_safe(da_raw, dim="time")
    da_det.name = VAR
    da_det.attrs.update(da_raw.attrs)
    ds = da_det.to_dataset()
    ds.to_netcdf(FN_DET, encoding=COMPRESSION)
    print(f"Detrendeado guardado en: {FN_DET}")
    da = ds[VAR]

# 2) Abrir climatología mensual externa (month, latitude, longitude) y alinear grilla
clim_ds = xr.open_dataset(FN_CLIM)
clim_ds = drop_time_bnds(clim_ds)
assert VAR in clim_ds.data_vars, f"La variable '{VAR}' no está en {FN_CLIM}"
clim = clim_ds[VAR]  # dims esperadas: (month, latitude, longitude)
# interpolar si la grilla difiere
clim = interp_to(clim, da)

# 3) Calcular anomalías: dato - climatología del mismo mes
#    Usa groupby por 'time.month' para broadcastear clim(month, lat, lon)
anom = da.groupby("time.month") - clim
anom = anom.rename(VAR)
anom.attrs["long_name"] = "Precipitación mensual - climatología (1991-2020)"
anom.attrs["note"] = "Anomalías respecto a climatología mensual externa (detrended)"

# 4) Filtrar meses Oct–Mar (mantiene todos los años)
anom_om = anom.where(anom["time"].dt.month.isin(MONTHS_OM), drop=True)

# 5) Normalización min–max por píxel a [0,1] (usando SOLO Oct–Mar)
norm_om = minmax_normalize_per_pixel(anom_om)
norm_om = norm_om.rename(VAR)
norm_om.attrs["long_name"] = "Anomalías normalizadas (min–max) 0–1, Oct–Mar"
norm_om.attrs["note"] = "Escala por píxel usando solo Oct–Mar; NaN preservados"

# 6) Guardar salida(s)
norm_om.to_netcdf(FN_OUT, encoding=COMPRESSION)
if SAVE_ANOMALIES:
    anom_om.to_netcdf(FN_ANOM, encoding=COMPRESSION)

# 7) Resumen
print("=== RESUMEN ===")
print(f"Fechas (Oct–Mar): {str(norm_om.time.values[0])[:10]} → {str(norm_om.time.values[-1])[:10]}")
print(f"Meses incluidos: {sorted(set(norm_om['time'].dt.month.values))}")
print(f"Rango global 0–1: {float(norm_om.min(skipna=True)):.3f} .. {float(norm_om.max(skipna=True)):.3f}")
print(f"Guardado normalizado: {FN_OUT}")
if SAVE_ANOMALIES:
    print(f"Guardado anomalías Oct–Mar (sin normalizar): {FN_ANOM}")
print("Listo para el script de SOM.")
