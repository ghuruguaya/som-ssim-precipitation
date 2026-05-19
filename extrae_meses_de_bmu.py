# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 18:21:30 2025

@author: Gaby
"""

import numpy as np
import xarray as xr
import pandas as pd
import os

# === RUTAS DE ARCHIVOS ===
archivo_nc = r"C:\Users\Gaby\Desktop\SOM\anomalias_normalizadas_minmax_1981_2024_por_anio.nc"
archivo_bmus = r"C:\Users\Gaby\Desktop\SOM\CORR\som_corr_pca_mascara_multisemilla\bmus_cor_pca_6x6_oct_mar_mascara_seed22.npy"
salida_excel = r"C:\Users\Gaby\Desktop\SOM\CORR\meses_anios_por_nodo_corr_pca_6x6_seed22.xlsx"

# === CARGA DE DATOS ===
ds = xr.open_dataset(archivo_nc)
# Filtrar meses de octubre a marzo
tiempos_filtrados = ds['time'].sel(time=ds['time.month'].isin([10, 11, 12, 1, 2, 3]))
fechas = pd.to_datetime(tiempos_filtrados.values)

# === CARGA DE BMUs ===
bmus = np.load(archivo_bmus)

# === AGRUPAR MESES Y AÑOS POR NODO ===
nodos = np.unique(bmus, axis=0)
resultados = {}

for nodo in nodos:
    indices = np.where((bmus[:, 0] == nodo[0]) & (bmus[:, 1] == nodo[1]))[0]
    fechas_nodo = fechas[indices]
    lista_fechas = [(f.year, f.month) for f in fechas_nodo]
    resultados[f"Nodo ({nodo[0]},{nodo[1]})"] = lista_fechas

# === GUARDAR EN EXCEL ===
with pd.ExcelWriter(salida_excel) as writer:
    for nodo, fechas_lista in resultados.items():
        df = pd.DataFrame(fechas_lista, columns=['Año', 'Mes'])
        df.to_excel(writer, sheet_name=nodo, index=False)

print(f"Archivo exportado: {salida_excel}")
