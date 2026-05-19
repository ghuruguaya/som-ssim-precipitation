# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 17:17:48 2025

@author: Gaby
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import pandas as pd
import os

# === Clase SOM con correlación y PCA ===
class MiniSom:
    def __init__(self, x, y, input_len, data, sigma=1.0, learning_rate=0.5):
        self.x = x
        self.y = y
        self.input_len = input_len
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.activation_map = np.zeros((x, y))
        self._init_coordinates()
        self._pca_initialization(data)

    def _init_coordinates(self):
        self._xx, self._yy = np.meshgrid(np.arange(self.x), np.arange(self.y))
        self._locations = np.array(list(zip(self._xx.flatten(), self._yy.flatten())))

    def _pca_initialization(self, data):
        pca = PCA(n_components=2)
        pca.fit(data)
        grid_x, grid_y = np.meshgrid(np.linspace(-1, 1, self.x), np.linspace(-1, 1, self.y))
        grid = np.stack([grid_x.flatten(), grid_y.flatten()], axis=1)
        init_weights = pca.inverse_transform(grid)
        self.weights = init_weights.reshape(self.x, self.y, self.input_len)

    def _pearson_distance(self, x, w):
        if np.std(x) == 0 or np.std(w) == 0:
            return 1.0
        corr = np.corrcoef(x, w)[0, 1]
        return 1 - corr

    def _activate(self, x):
        for i in range(self.x):
            for j in range(self.y):
                self.activation_map[i, j] = self._pearson_distance(x, self.weights[i, j])

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
    carpeta_salida = r"C:\Users\Gaby\Desktop\SOM\CORR\som_corr_pca_mascara_multisemilla"
    os.makedirs(carpeta_salida, exist_ok=True)

    ds = xr.open_dataset(archivo)
    datos = ds['anomalia_normalizada']
    meses_ok = datos['time'].dt.month.isin([10, 11, 12, 1, 2, 3])
    datos_filtrados = datos.sel(time=meses_ok)

    # Crear máscara de tierra (válidos en todos los meses)
    mascara_tierra = ~np.isnan(datos_filtrados.isel(time=0).values)
    for i in range(1, datos_filtrados.shape[0]):
        mascara_tierra &= ~np.isnan(datos_filtrados.isel(time=i).values)

    # Aplicar máscara
    data = datos_filtrados.values  # (meses, lat, lon)
    data_masked = data[:, mascara_tierra]
    data_2d = data_masked

    size = 6
    iteraciones = 5000
    semillas = [1, 10, 22, 50, 77, 123, 145, 200, 321, 444, 555, 999]

    resumen_silhouette = []

    for semilla in semillas:
        print(f"\n===> Ejecutando semilla {semilla}...")

        np.random.seed(semilla)

        som = MiniSom(x=size, y=size, input_len=data_2d.shape[1], data=data_2d, sigma=1.0, learning_rate=0.5)
        som.train(data_2d, num_iterations=iteraciones)

        bmus = np.array(som.map_vects(data_2d))
        etiquetas = np.array([i * size + j for i, j in bmus])
        silhouette = silhouette_score(data_2d, etiquetas, metric='correlation')
        resumen_silhouette.append((semilla, silhouette))

        # Guardar BMUs
        np.save(os.path.join(carpeta_salida, f"bmus_cor_pca_{size}x{size}_oct_mar_mascara_seed{semilla}.npy"), bmus)

        # Guardar histograma de correlación
        correlaciones = []
        for x in data_2d:
            i, j = som.winner(x)
            w = som.weights[i, j]
            corr = 0 if np.std(x) == 0 or np.std(w) == 0 else np.corrcoef(x, w)[0, 1]
            correlaciones.append(corr)

        plt.hist(correlaciones, bins=30, color='skyblue', edgecolor='black')
        plt.title(f"Distribución Correlación - Semilla {semilla}")
        plt.xlabel("Correlación")
        plt.ylabel("Frecuencia")
        plt.tight_layout()
        plt.savefig(os.path.join(carpeta_salida, f"hist_cor_seed{semilla}.png"), dpi=150)
        plt.close()

    # Guardar resumen de valores de silhouette
    df_silhouette = pd.DataFrame(resumen_silhouette, columns=["semilla", "silhouette"])
    df_silhouette.to_csv(os.path.join(carpeta_salida, f"resumen_silhouette_{size}x{size}_oct_mar.csv"), index=False)
    print("\nResumen exportado:", os.path.join(carpeta_salida, f"resumen_silhouette_{size}x{size}_oct_mar.csv"))
