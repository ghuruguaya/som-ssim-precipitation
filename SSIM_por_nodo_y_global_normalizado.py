# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 10:59:20 2025

@author: Gaby
"""
import xarray as xr
import pandas as pd
import numpy as np
import os
from itertools import combinations
from skimage.metrics import structural_similarity as ssim

# === Rutas ===
archivo_csv = r"C:\Users\Gaby\Desktop\SOM\EUCLID\som_euclid_pca_mascara_multisemilla\nodos_oct_mar_todos_los_archivos.csv"
archivo_nc = r"C:\Users\Gaby\Desktop\SOM\anomalias_normalizadas_minmax_1981_2024_por_anio.nc"

# === Cargar CSV y NetCDF ===
df = pd.read_csv(archivo_csv)
ds = xr.open_dataset(archivo_nc)
data = ds['anomalia_normalizada']
fechas_nc = pd.to_datetime(data['time'].values)

# === Filtrar solo meses de octubre a marzo ===
meses_oct_mar = [10, 11, 12, 1, 2, 3]
mask_meses = data['time.month'].isin(meses_oct_mar)
data_oct_mar = data.sel(time=mask_meses)
fechas_filtradas = pd.to_datetime(data_oct_mar['time'].values)

# === Máscara de océano (NaNs en el primer mapa filtrado) ===
mascara = ~np.isnan(data_oct_mar[0].values)
data_masked = np.array([np.where(mascara, m.values, np.nan) for m in data_oct_mar])



# === Calcular SSIM promedio por nodo ===
resultados = []

for (config, semilla), grupo_config in df.groupby(["config_red", "semilla"]):
    print(f"\nProcesando config {config} - semilla {semilla}")
    for nodo, grupo_nodo in grupo_config.groupby(["nodo_x", "nodo_y"]):
        indices = grupo_nodo["indice_mes"].values
        print(f"  Nodo {nodo}: {len(indices)} mapas")

        if len(indices) < 2:
            print("    ❌ Menos de 2 mapas, se omite")
            continue

        mapas = data_masked[indices]

        ssim_vals = []
        for a, b in combinations(mapas, 2):
            diff_mask = ~np.isnan(a) & ~np.isnan(b)
            

            if np.sum(diff_mask) < 10:
                print("    ⚠️ Muy pocos puntos válidos comunes, se salta")
                continue

            if np.std(a[diff_mask]) == 0 or np.std(b[diff_mask]) == 0:
                print("    ⚠️ Al menos un mapa es constante, se salta")
                continue

            try:
                val = ssim(a[diff_mask], b[diff_mask], data_range=2.0)
                if not np.isnan(val):
                    ssim_vals.append(val)
                else:
                    print("    ⚠️ SSIM resultó nan")
            except Exception as e:
                print(f"    ⚠️ Error en SSIM: {e}")

        ssim_prom = np.mean(ssim_vals) if ssim_vals else np.nan
        print(f"    ✅ SSIM promedio: {ssim_prom}")

        resultados.append({
            "config_red": config,
            "semilla": semilla,
            "nodo_x": nodo[0],
            "nodo_y": nodo[1],
            "ssim_promedio": ssim_prom
        })

# === Exportar por nodo ===
df_resultados = pd.DataFrame(resultados)
csv_nodo = os.path.join(os.path.dirname(archivo_csv), "ssim_promedio_por_nodo.csv")
df_resultados.to_csv(csv_nodo, index=False)
print(f"✅ SSIM por nodo exportado: {csv_nodo}")

# === Exportar promedio global por red y semilla ===
promedios_globales = (
    df_resultados
    .groupby(["config_red", "semilla"])["ssim_promedio"]
    .mean()
    .reset_index()
    .rename(columns={"ssim_promedio": "ssim_promedio_global"})
)

csv_global = os.path.join(os.path.dirname(archivo_csv), "ssim_promedio_global_por_red.csv")
promedios_globales.to_csv(csv_global, index=False)
print(f"✅ SSIM global por red exportado: {csv_global}")
