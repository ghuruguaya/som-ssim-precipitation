# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 22:21:53 2025

@author: Gaby
"""
import numpy as np
import xarray as xr
import pandas as pd
import os
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score

# === Rutas de datos ===
ruta_nc = r"C:\Users\Gaby\Desktop\SOM\anomalias_normalizadas_minmax_1981_2024_por_anio.nc"
carpeta_bmus = r"C:\Users\Gaby\Desktop\SOM\EUCLID\som_euclid_pca_mascara_multisemilla"
semillas = [1, 10, 22, 50, 77, 123, 145, 200, 321, 444, 555, 999]
redes = ['3x3', '4x4', '6x6']

# === Cargar y filtrar los datos ===
ds = xr.open_dataset(ruta_nc)
data = ds["anomalia_normalizada"]
data_oct_mar = data.sel(time=data['time.month'].isin([10, 11, 12, 1, 2, 3]))

# Crear máscara para eliminar puntos que tienen algún NaN en todos los tiempos
mascara = ~np.isnan(data_oct_mar).any(dim="time")

# Aplicar máscara
data_masc = data_oct_mar.where(mascara)

# Convertir a 2D y eliminar puntos NaN
data_2d = data_masc.stack(puntos=('latitude', 'longitude')).transpose("time", "puntos").values
valid_mask = ~np.isnan(data_2d).any(axis=0)
data_2d = data_2d[:, valid_mask]

# === Cálculo de métricas ===
resultados = []

for red in redes:
    for seed in semillas:
        nombre_archivo = f"bmus_euclid_pca_{red}_oct_mar_mascara_seed{seed}.npy"
        ruta_bmu = os.path.join(carpeta_bmus, nombre_archivo)

        if not os.path.exists(ruta_bmu):
            print(f"❌ Archivo no encontrado: {nombre_archivo}")
            continue

        bmus = np.load(ruta_bmu)
        # Convertir coordenadas (i,j) a índice plano
        if bmus.ndim == 2 and bmus.shape[1] == 2:
            n_cols = int(red.split("x")[1])
            bmus = bmus[:, 0] * n_cols + bmus[:, 1]

        # Validar tamaño
        if len(bmus) != data_2d.shape[0]:
            print(f"❌ Error con {red}, semilla {seed}: tamaños incompatibles {len(bmus)} != {data_2d.shape[0]}")
            continue

        try:
            db = davies_bouldin_score(data_2d, bmus)
            ch = calinski_harabasz_score(data_2d, bmus)
            print(f"✅ {red} seed {seed} → DBI: {db:.3f}, CHI: {ch:.1f}")
        except Exception as e:
            print(f"⚠️ Error al calcular métricas para {red} seed {seed}: {e}")
            db = np.nan
            ch = np.nan

        resultados.append({
            "config_red": red,
            "semilla": seed,
            "Davies-Bouldin": db,
            "Calinski-Harabasz": ch
        })

# === Exportar ===
df_resultados = pd.DataFrame(resultados)
salida = os.path.join(carpeta_bmus, "indices_validez_DB_CH_octmar_MASCARA.xlsx")
df_resultados.to_excel(salida, index=False)
print(f"\n✅ Resultados exportados a: {salida}")
