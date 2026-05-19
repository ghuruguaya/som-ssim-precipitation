# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 21:04:23 2026

@author: Gaby
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# =========================
# CONFIG
# =========================
row_files = [
    r"C:\Users\Gaby\Desktop\SOM\figura_SOM_row_0.png",
    r"C:\Users\Gaby\Desktop\SOM\figura_SOM_row_1.png",
    r"C:\Users\Gaby\Desktop\SOM\figura_SOM_row_2.png",
    r"C:\Users\Gaby\Desktop\SOM\figura_SOM_row_3.png",
    r"C:\Users\Gaby\Desktop\SOM\figura_SOM_row_4.png",
    r"C:\Users\Gaby\Desktop\SOM\figura_SOM_row_5.png",
    r"C:\Users\Gaby\Desktop\SOM\figura_SOM_row_6.png",
    r"C:\Users\Gaby\Desktop\SOM\figura_SOM_row_7.png",
]

# usar un mapa original para extraer la barra
example_map = r"C:\Users\Gaby\Desktop\SOM\SSIM-COMPLETO\PROYECTO_GABY\som_results_ssim\som_4x4_pca_bmu-ssim\weights_mm\weight_mm_nodo_0_0.png"

out_png_A = r"C:\Users\Gaby\Desktop\SOM\figura_SOM_4x4_parte1.png"
out_pdf_A = r"C:\Users\Gaby\Desktop\SOM\figura_SOM_4x4_parte1.pdf"
out_png_B = r"C:\Users\Gaby\Desktop\SOM\figura_SOM_4x4_parte2.png"
out_pdf_B = r"C:\Users\Gaby\Desktop\SOM\figura_SOM_4x4_parte2.pdf"

# =========================
# FUNCION PARA EXTRAER BARRA
# =========================
def crop_colorbar(img, frac_bottom=0.18):
    """
    Extrae la barra de color desde la parte inferior de la imagen.
    """
    h, w = img.shape[:2]

    # zona inferior donde está la barra
    y0 = int(h * (1 - frac_bottom))
    bottom = img[y0:, :, :]

    # recorte de márgenes blancos
    gray = np.mean(bottom, axis=2)
    rows = np.where(np.min(gray, axis=1) < 250)[0]
    cols = np.where(np.min(gray, axis=0) < 250)[0]

    if len(rows) > 0 and len(cols) > 0:
        bottom = bottom[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1, :]

    return bottom

# =========================
# CARGAR LAS 4 FILAS
# =========================
print("Cargando imágenes...")
try:
    row_imgs = [np.array(Image.open(f)) for f in row_files]
except Exception as e:
    print("ERROR al cargar imágenes:", e)
    raise

# igualar ancho por seguridad
min_width = min(img.shape[1] for img in row_imgs)
row_imgs = [img[:, :min_width, :] for img in row_imgs]

# =========================
# EXTRAER BARRA
# =========================
img_example = np.array(Image.open(example_map))
bar = crop_colorbar(img_example, frac_bottom=0.18)

# =========================
# FIGURA PARTE 1 (filas 0-3)
# =========================
fig1 = plt.figure(figsize=(10, 14))
gs1 = fig1.add_gridspec(
    nrows=5, ncols=1,
    height_ratios=[1, 1, 1, 1, 0.18],
    hspace=0.02
)
for i in range(4):
    ax = fig1.add_subplot(gs1[i, 0])
    ax.imshow(row_imgs[i])
    ax.axis("off")
ax_bar1 = fig1.add_subplot(gs1[4, 0])
ax_bar1.imshow(bar)
ax_bar1.axis("off")
try:
    fig1.savefig(out_png_A, dpi=400, bbox_inches="tight", pad_inches=0)
    fig1.savefig(out_pdf_A, dpi=600, bbox_inches="tight", pad_inches=0)
    print("Parte 1 guardada en:", out_png_A)
except Exception as e:
    print("ERROR guardando parte 1:", e)
    raise
plt.close(fig1)

# =========================
# FIGURA PARTE 2 (filas 4-7)
# =========================
fig2 = plt.figure(figsize=(10, 14))
gs2 = fig2.add_gridspec(
    nrows=5, ncols=1,
    height_ratios=[1, 1, 1, 1, 0.18],
    hspace=0.02
)
for i in range(4):
    ax = fig2.add_subplot(gs2[i, 0])
    ax.imshow(row_imgs[i + 4])
    ax.axis("off")
ax_bar2 = fig2.add_subplot(gs2[4, 0])
ax_bar2.imshow(bar)
ax_bar2.axis("off")
try:
    fig2.savefig(out_png_B, dpi=400, bbox_inches="tight", pad_inches=0)
    fig2.savefig(out_pdf_B, dpi=600, bbox_inches="tight", pad_inches=0)
    print("Parte 2 guardada en:", out_png_B)
except Exception as e:
    print("ERROR guardando parte 2:", e)
    raise
plt.close(fig2)




#####################################DOS IMAGENES



import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# =========================
# CONFIG
# =========================
row_files = [
    r"C:\Users\Gaby\Desktop\SOM\figura_SOM_row_0.png",
    r"C:\Users\Gaby\Desktop\SOM\figura_SOM_row_1.png",
    r"C:\Users\Gaby\Desktop\SOM\figura_SOM_row_2.png",
    r"C:\Users\Gaby\Desktop\SOM\figura_SOM_row_3.png",
    r"C:\Users\Gaby\Desktop\SOM\figura_SOM_row_4.png",
    r"C:\Users\Gaby\Desktop\SOM\figura_SOM_row_5.png",
    r"C:\Users\Gaby\Desktop\SOM\figura_SOM_row_6.png",
    r"C:\Users\Gaby\Desktop\SOM\figura_SOM_row_7.png",
]

# usar un mapa original para extraer la barra
example_map = r"C:\Users\Gaby\Desktop\SOM\SSIM-COMPLETO\PROYECTO_GABY\som_results_ssim\som_4x4_pca_bmu-ssim\weights_mm\weight_mm_nodo_0_0.png"

out_png_A = r"C:\Users\Gaby\Desktop\SOM\figura_SOM_4x4_parte1.png"
out_pdf_A = r"C:\Users\Gaby\Desktop\SOM\figura_SOM_4x4_parte1.pdf"
out_png_B = r"C:\Users\Gaby\Desktop\SOM\figura_SOM_4x4_parte2.png"
out_pdf_B = r"C:\Users\Gaby\Desktop\SOM\figura_SOM_4x4_parte2.pdf"

# =========================
# FUNCION PARA EXTRAER BARRA
# =========================
def crop_colorbar(img, frac_bottom=0.18):
    """
    Extrae la barra de color desde la parte inferior de la imagen.
    """
    h, w = img.shape[:2]

    # zona inferior donde está la barra
    y0 = int(h * (1 - frac_bottom))
    bottom = img[y0:, :, :]

    # recorte de márgenes blancos
    gray = np.mean(bottom, axis=2)
    rows = np.where(np.min(gray, axis=1) < 250)[0]
    cols = np.where(np.min(gray, axis=0) < 250)[0]

    if len(rows) > 0 and len(cols) > 0:
        bottom = bottom[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1, :]

    return bottom

# =========================
# CARGAR LAS 4 FILAS
# =========================
print("Cargando imágenes...")
try:
    row_imgs = [np.array(Image.open(f)) for f in row_files]
except Exception as e:
    print("ERROR al cargar imágenes:", e)
    raise

# igualar ancho por seguridad
min_width = min(img.shape[1] for img in row_imgs)
row_imgs = [img[:, :min_width, :] for img in row_imgs]

# =========================
# EXTRAER BARRA
# =========================
img_example = np.array(Image.open(example_map))
bar = crop_colorbar(img_example, frac_bottom=0.18)

# =========================
# FIGURA PARTE 1 (filas 0-3)
# =========================
fig1 = plt.figure(figsize=(10, 14))
gs1 = fig1.add_gridspec(
    nrows=5, ncols=1,
    height_ratios=[1, 1, 1, 1, 0.18],
    hspace=0.02
)
for i in range(4):
    ax = fig1.add_subplot(gs1[i, 0])
    ax.imshow(row_imgs[i])
    ax.axis("off")
ax_bar1 = fig1.add_subplot(gs1[4, 0])
ax_bar1.imshow(bar)
ax_bar1.axis("off")
try:
    fig1.savefig(out_png_A, dpi=400, bbox_inches="tight", pad_inches=0)
    fig1.savefig(out_pdf_A, dpi=600, bbox_inches="tight", pad_inches=0)
    print("Parte 1 guardada en:", out_png_A)
except Exception as e:
    print("ERROR guardando parte 1:", e)
    raise
plt.close(fig1)

# =========================
# FIGURA PARTE 2 (filas 4-7)
# =========================
fig2 = plt.figure(figsize=(10, 14))
gs2 = fig2.add_gridspec(
    nrows=5, ncols=1,
    height_ratios=[1, 1, 1, 1, 0.18],
    hspace=0.02
)
for i in range(4):
    ax = fig2.add_subplot(gs2[i, 0])
    ax.imshow(row_imgs[i + 4])
    ax.axis("off")
ax_bar2 = fig2.add_subplot(gs2[4, 0])
ax_bar2.imshow(bar)
ax_bar2.axis("off")
try:
    fig2.savefig(out_png_B, dpi=400, bbox_inches="tight", pad_inches=0)
    fig2.savefig(out_pdf_B, dpi=600, bbox_inches="tight", pad_inches=0)
    print("Parte 2 guardada en:", out_png_B)
except Exception as e:
    print("ERROR guardando parte 2:", e)
    raise
plt.close(fig2)