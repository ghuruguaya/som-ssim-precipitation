# SOM con SSIM para clasificación de patrones de precipitación — Sector Oriental de Sudamérica Subtropical

Este repositorio contiene el código Python utilizado en el trabajo:

> **"Clasificación de patrones espaciales de anomalía de precipitación estival en el sector oriental de Sudamérica subtropical mediante redes auto-organizativas (SOM) con métrica de similitud estructural (SSIM)"**
> Hernández, G.; Müller, G. V.; Vasconcellos, F. (en revisión)

El pipeline procesa datos CHIRPS mensuales (octubre–marzo, 1981–2024), entrena redes SOM con distintas métricas de similitud (SSIM completo, SSIM simplificado, correlación y distancia euclidiana), evalúa la calidad del agrupamiento y genera las figuras del paper.

---

## Estructura del pipeline

```
1. Preprocesamiento
   └── normalización_entre_0_y_1.py

2. Análisis de sensibilidad — fase exploratoria (ejecutado localmente)
   Métricas: SSIM simplificado, correlación, distancia euclidiana
   Inicializaciones: PCA y aleatoria | Tamaños: 3×3, 4×4, 6×6 | 12 semillas c/u
   ├── multiseed_SSIM_PCA_mejor_silhuette_3x3_4x4_9x9.py
   ├── multiseed_SSIM_init_aleat_mejor_silhuette_3x3_4x4_9x9.py
   ├── multiseed_corr_pca_mejor_silhuette_3x3_4x4_6x6.py
   ├── multiseed_corr_init_aleat_mejor_silhuette_3x3_4x4_6x6.py
   ├── multiseed_euclid_PCA_mejor_silhuette_3x3_4x4_9x9.py
   └── multiseed_euclid_init_aleat_mejor_silhuette_3x3_4x4_9x9.py

3. Entrenamiento final con SSIM completo (ejecutado en clúster Granito)
   └── som_ssim_granito_fixed.py

4. Métricas de validación
   ├── calculo_metricas.py                        ← Silhouette, DB, CH post-hoc
   ├── indice_DBI_CHI.py                          ← DB y CH para análisis multisemilla
   ├── SSIM_por_nodo.py                           ← Node-mean SSIM (versión simplificada)
   └── SSIM_por_nodo_y_global_normalizado.py      ← SSIM entre pares de campos por nodo

5. Visualización — Figura 2 del paper (ejecutar en orden)
   ├── promedio_amon_por_nodo.py                  ← promedio de anomalías por nodo
   ├── desnormalizar_BMU.py                       ← prototipo del nodo (pesos → mm)
   ├── mapas_promedios_desnormalizados_SSIMcompleto_4x4.py  ← ensambla filas
   └── union_de_mapas_SOM_SSIM_4x4.py            ← figura final

Utilidades
   ├── npy_a_csv.py                               ← convierte BMUs.npy a CSV con fechas
   ├── extrae_meses_de_bmu.py                     ← exporta fechas por nodo a Excel
   └── comparación_mapas_SSIM.py                  ← verificación manual de valores SSIM
```

---

## Datos de entrada requeridos

Los archivos de datos **no están incluidos** en el repositorio por su tamaño. Deben colocarse en la carpeta del proyecto con los siguientes nombres:

| Archivo | Descripción | Fuente |
|---------|-------------|--------|
| `chirps_mensual.nc` | Precipitación mensual CHIRPS v2.0 (0.05°) | [CHIRPS](https://www.chc.ucsb.edu/data/chirps) |
| `climatologia_mensual_1991_2020.nc` | Climatología mensual 1991–2020 | Generado desde CHIRPS |
| `chirps_normalized.nc` | Anomalías normalizadas 0–1 (Oct–Mar) | Generado por `normalización_entre_0_y_1.py` |
| `chirps_anom_oct_mar.nc` | Anomalías en mm (Oct–Mar) | Generado por `normalización_entre_0_y_1.py` |

---

## Orden de ejecución

### Paso 1 — Preprocesamiento
```bash
python normalización_entre_0_y_1.py
```
Genera `chirps_normalized.nc` y `chirps_anom_oct_mar.nc`.

### Paso 2 — Análisis de sensibilidad (fase exploratoria)
Correr los 6 scripts `multiseed_*.py`. Cada uno entrena el SOM con una métrica e inicialización específica, para los tamaños de red 3×3, 4×4 y 6×6, con 12 semillas distintas. Cada script guarda los archivos `bmus_*.npy` y un CSV con los valores de Silhouette por semilla.

### Paso 3 — Entrenamiento final (clúster Granito)
```bash
python som_ssim_granito_fixed.py \
  --netcdf chirps_normalized.nc \
  --grid 4 \
  --init pca \
  --epochs 10 \
  --lr 0.4 \
  --seed 145 \
  --window 11 \
  --sigma 1.5 \
  --out som_results_ssim
```
Guarda: `BMUs.npy`, `weights.npy`, `valid_idx.npy`, `config.json`.

### Paso 4 — Métricas de validación
```bash
python calculo_metricas.py
python indice_DBI_CHI.py
python SSIM_por_nodo.py
python SSIM_por_nodo_y_global_normalizado.py
```

### Paso 5 — Figura 2 del paper
Ejecutar en orden:
```bash
python promedio_amon_por_nodo.py
python desnormalizar_BMU.py
python mapas_promedios_desnormalizados_SSIMcompleto_4x4.py
python union_de_mapas_SOM_SSIM_4x4.py
```

---

## Dependencias

```
numpy
xarray
matplotlib
cartopy
scikit-learn
scikit-image
scipy
pandas
python-pptx  (opcional, para exportar PPT)
tqdm         (opcional, para barra de progreso en Granito)
```

Instalación:
```bash
pip install numpy xarray matplotlib cartopy scikit-learn scikit-image scipy pandas python-pptx tqdm
```

---

## Parámetros del entrenamiento final

| Parámetro | Valor |
|-----------|-------|
| Métrica de BMU | SSIM completo (ventanas 11×11, σ = 1.5) |
| Tamaño de red | 4×4 |
| Inicialización | PCA |
| Semilla | 145 |
| Épocas | 10 |
| Learning rate | 0.4 |
| Período | Octubre–Marzo, 1981–2024 |
| Dominio | 20°–40°S, 50°–65°O |
| Datos | CHIRPS v2.0 (0.05°) |
| Normalización | Min–Max por píxel (Oct–Mar) |

---

## Nota sobre rutas

Los scripts usan rutas absolutas de Windows (`C:\Users\Gaby\Desktop\SOM\...`).
Antes de correr, modificar la variable `BASE` (o equivalente) en la sección `CONFIG` de cada script.

---

## Contacto

Gabriela Hernández — Universidad Nacional del Centro de la Provincia de Buenos Aires
