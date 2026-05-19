# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 12:07:33 2025

@author: Gaby
"""
import xarray as xr
import numpy as np

# === Cargar archivo ===
ruta = r'C:\Users\Gaby\Desktop\SOM\anomalias_normalizadas_minmax_1981_2024_por_anio.nc'
ds = xr.open_dataset(ruta)
anom = ds['anomalia_normalizada']

# === Máscara de tierra: primeros datos válidos
mascara_tierra = ~np.isnan(anom.isel(time=0).values)

# === Función para obtener mapa enmascarado
def obtener_mapa(anom, anio, mes):
    datos = anom.sel(time=f'{anio}-{mes:02d}').values
    return np.where(mascara_tierra, datos, np.nan)

# === Función para calcular SSIM a mano ===
def calcular_ssim_manual(x, y):
    # Aplanar y quitar NaNs
    x_flat = x[~np.isnan(x) & ~np.isnan(y)].flatten()
    y_flat = y[~np.isnan(x) & ~np.isnan(y)].flatten()

    # Constantes estándar
    L = max(y_flat.max() - y_flat.min(), 1e-6)  # rango dinámico
    K1, K2 = 0.01, 0.03
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    # Estadísticos
    mu_x = np.mean(x_flat)
    mu_y = np.mean(y_flat)
    sigma_x2 = np.var(x_flat)
    sigma_y2 = np.var(y_flat)
    sigma_xy = np.cov(x_flat, y_flat)[0, 1]

    # Fórmula SSIM
    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (sigma_x2 + sigma_y2 + C2)
    return num / den

# === Pares de fechas a comparar
pares = [
    ((2021, 11), (1992, 12)),
    ((2008, 3), (2020, 3)),
    ((2016, 11), (1987, 1)),
    ((1984, 12), (2023, 1)),
    ((2018, 1), (1985, 2)),
    ((2007, 2), (2022, 12)),
    ((1985, 2), (2018, 12))
]

# === Calcular SSIM
for (anio1, mes1), (anio2, mes2) in pares:
    mapa1 = obtener_mapa(anom, anio1, mes1)
    mapa2 = obtener_mapa(anom, anio2, mes2)

    ssim_valor = calcular_ssim_manual(mapa1, mapa2)
    print(f"SSIM manual entre {mes1:02d}/{anio1} y {mes2:02d}/{anio2}: {ssim_valor:.4f}")
