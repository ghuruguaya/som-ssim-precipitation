# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 10:04:23 2025

@author: Gaby
"""
import os
import numpy as np
import pandas as pd
import xarray as xr

# === Rutas ===
carpeta_npy = r"C:\Users\Gaby\Desktop\SOM\EUCLID\som_euclid_pca_mascara_multisemilla"
archivo_anomalias = r"C:\Users\Gaby\Desktop\SOM\anomalias_mensuales_1981_2024_por_anio.nc"
salida_csv = os.path.join(carpeta_npy, "nodos_oct_mar_todos_los_archivos.csv")

# === Leer fechas del NetCDF ===
ds = xr.open_dataset(archivo_anomalias)
fechas_todas = pd.to_datetime(ds['time'].values)

# === Filtrar meses de octubre a marzo ===
meses_validos = [10, 11, 12, 1, 2, 3]
fechas_filtradas = pd.Series(fechas_todas[fechas_todas.month.isin(meses_validos)]).reset_index(drop=True)

# === Verificación importante ===
print(f"Total de fechas filtradas (oct-mar): {len(fechas_filtradas)}")  # Debe dar 264 si son 44 años

# === Procesar archivos .npy ===
registros = []

for archivo in os.listdir(carpeta_npy):
    if not archivo.endswith(".npy"):
        continue

    partes = archivo.split("_")
    size_str = [s for s in partes if "x" in s][0]       # Ej: '3x3'
    semilla = int(partes[-1].replace("seed", "").replace(".npy", ""))
    bmus = np.load(os.path.join(carpeta_npy, archivo))

    # === Asociar nodo y fecha ===
    for idx, (i, j) in enumerate(bmus):
        registros.append({
            "config_red": size_str,
            "semilla": semilla,
            "indice_mes": idx,
            "fecha": fechas_filtradas[idx].strftime("%Y-%m"),
            "nodo_x": i,
            "nodo_y": j
        })

# === Exportar CSV ===
df = pd.DataFrame(registros)
df.to_csv(salida_csv, index=False)

print(f"✅ CSV generado con fechas de octubre a marzo: {salida_csv}")
