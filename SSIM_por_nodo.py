# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 19:36:01 2025

@author: Gaby
"""

import numpy as np
import xarray as xr
import os
from collections import defaultdict

# === CONFIGURACIÓN ===
archivo_nc = r"C:\Users\Gaby\Desktop\SOM\anomalias_normalizadas_minmax_1981_2024_por_anio.nc"
archivo_bmus = r"C:\Users\Gaby\Desktop\SOM\CORR\som_corr_pca_mascara_multisemilla\bmus_cor_pca_6x6_oct_mar_mascara_seed22.npy"
archivo_pesos = r"C:\Users\Gaby\Desktop\SOM\CORR\pesos_cor_pca_6x6_seed22.npy"
salida_csv = r"C:\Users\Gaby\Desktop\SOM\CORR\ssim_promedio_por_nodo_6x6_seed22.csv"

# === FUNCIÓN SSIM SIMPLIFICADA ===
def ssim_global(x, y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    var_x = np.var(x)
    var_y = np.var(y)
    cov = np.mean((x - mean_x) * (y - mean_y))

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    return ((2 * mean_x * mean_y + c1) * (2 * cov + c2)) / ((mean_x**2 + mean_y**2 + c1) * (var_x + var_y + c2))

# === CARGA DE DATOS ===
ds = xr.open_dataset(archivo_nc)
datos = ds['anomalia_normalizada']
meses_ok = datos['time'].dt.month.isin([10, 11, 12, 1, 2, 3])
datos_filtrados = datos.sel(time=meses_ok)

# Aplicar máscara de tierra
mascara_tierra = ~np.isnan(datos_filtrados.isel(time=0).values)
for i in range(1, datos_filtrados.shape[0]):
    mascara_tierra &= ~np.isnan(datos_filtrados.isel(time=i).values)

datos_masked = datos_filtrados.values[:, mascara_tierra]
bmus = np.load(archivo_bmus)
pesos = np.load(archivo_pesos)

# === CALCULAR SSIM PROMEDIO POR NODO ===
ssim_por_nodo = defaultdict(list)
for idx, x in enumerate(datos_masked):
    nodo = tuple(bmus[idx])
    peso = pesos[nodo[0], nodo[1]]
    ssim = ssim_global(x, peso)
    ssim_por_nodo[nodo].append(ssim)

# Calcular el promedio
resultado = []
for nodo, valores in ssim_por_nodo.items():
    promedio = np.mean(valores)
    resultado.append((nodo[0], nodo[1], len(valores), promedio))

# === EXPORTAR A CSV ===
import pandas as pd
df = pd.DataFrame(resultado, columns=['fila', 'columna', 'n_meses', 'ssim_promedio'])
df = df.sort_values(['fila', 'columna'])
df.to_csv(salida_csv, index=False)

print(f"\n✅ SSIM promedio por nodo exportado a:\n{salida_csv}")
