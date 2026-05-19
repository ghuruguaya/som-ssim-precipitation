# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 08:23:50 2025

@author: Gaby
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import silhouette_score
import pandas as pd
import os

# === Clase SOM con distancia euclideana y pesos aleatorios ===
class MiniSom:
    def __init__(self, x, y, input_len, sigma=1.0, learning_rate=0.5):
        self.x = x
        self.y = y
        self.input_len = input_len
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.activation_map = np.zeros((x, y))
        self.weights = np.random.rand(x, y, input_len) * 2 - 1  # Pesos aleatorios
        self._init_coordinates()

    def _init_coordinates(self):
        self._xx, self._yy = np.meshgrid(np.arange(self.x), np.arange(self.y))
        self._locations = np.array(list(zip(self._xx.flatten(), self._yy.flatten())))

    def _euclidean_distance(self, x, w):
        return np.linalg.norm(x - w)

    def _activate(self, x):
        for i in range(self.x):
            for j in range(self.y):
                self.activation_map[i, j] = self._euclidean_distance(x, self.weights[i, j])

    def winner(self, x):
        self._activate(x)
        return np.unravel_index(self.activation_map.argmin(), self.activation_map.shape)

    def update(self, x, win, t, max_iter):
        lr = self.learning_rate * np.exp(-t / max_iter)
        sigma = self.sigma * np.exp(-t / max_iter)
        for i in range(self.x):
            for j in range(self.y):
                dist_sq = (i - win[0])**2 + (j - win[1])**2
                if dist_sq <= sigma**2:
                    influence = np.exp(-dist_sq / (2 * sigma**2))
                    self.weights[i, j] += lr * influence * (x - self.weights[i, j])

    def train(self, data, num_iterations):
        for t in range(num_iterations):
            x = data[np.random.randint(0, data.shape[0])]
            win = self.winner(x)
            self.update(x, win, t, num_iterations)

    def map_vects(self, data):
        return [self.winner(x) for x in data]

    def get_weights(self):
        return self.weights


# === MAIN ===
if __name__ == "__main__":
    archivo = r"C:\Users\Gaby\Desktop\SOM\anomalias_normalizadas_minmax_1981_2024_por_anio.nc"
    carpeta_salida = r"C:\Users\Gaby\Desktop\SOM\EUCLID\som_eucl_random_mascara_multisemilla"
    os.makedirs(carpeta_salida, exist_ok=True)

    ds = xr.open_dataset(archivo)
    datos = ds['anomalia_normalizada']
    meses_ok = datos['time'].dt.month.isin([10, 11, 12, 1, 2, 3])
    datos_filtrados = datos.sel(time=meses_ok)

    # Crear máscara de tierra
    mascara_tierra = ~np.isnan(datos_filtrados.isel(time=0).values)
    for i in range(1, datos_filtrados.shape[0]):
        mascara_tierra &= ~np.isnan(datos_filtrados.isel(time=i).values)

    data = datos_filtrados.values
    data_masked = data[:, mascara_tierra]
    data_2d = data_masked

    size = 6
    iteraciones = 5000
    semillas = [1, 10, 22, 50, 77, 123, 145, 200, 321, 444, 555, 999]

    resumen_silhouette = []

    for semilla in semillas:
        print(f"\n===> Ejecutando semilla {semilla}...")
        np.random.seed(semilla)

        som = MiniSom(x=size, y=size, input_len=data_2d.shape[1], sigma=1.0, learning_rate=0.5)
        som.train(data_2d, num_iterations=iteraciones)

        bmus = np.array(som.map_vects(data_2d))
        etiquetas = np.array([i * size + j for i, j in bmus])
        silhouette = silhouette_score(data_2d, etiquetas, metric='correlation')
        resumen_silhouette.append((semilla, silhouette))

        np.save(os.path.join(carpeta_salida, f"bmus_eucl_random_{size}x{size}_oct_mar_mascara_seed{semilla}.npy"), bmus)

        # Histograma de correlación entre datos y pesos
        correlaciones = []
        for x in data_2d:
            i, j = som.winner(x)
            w = som.weights[i, j]
            corr = 0 if np.std(x) == 0 or np.std(w) == 0 else np.corrcoef(x, w)[0, 1]
            correlaciones.append(corr)

        plt.hist(correlaciones, bins=30, color='lightgreen', edgecolor='black')
        plt.title(f"Distribución Correlación - Semilla {semilla}")
        plt.xlabel("Correlación")
        plt.ylabel("Frecuencia")
        plt.tight_layout()
        plt.savefig(os.path.join(carpeta_salida, f"hist_cor_seed{semilla}.png"), dpi=150)
        plt.close()

    # Guardar resumen
    df_silhouette = pd.DataFrame(resumen_silhouette, columns=["semilla", "silhouette"])
    df_silhouette.to_csv(os.path.join(carpeta_salida, f"resumen_silhouette_{size}x{size}_oct_mar.csv"), index=False)
    print("\nResumen exportado:", os.path.join(carpeta_salida, f"resumen_silhouette_{size}x{size}_oct_mar.csv"))
