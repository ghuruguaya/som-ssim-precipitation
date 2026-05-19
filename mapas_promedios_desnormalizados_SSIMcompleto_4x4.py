# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 20:35:32 2026

@author: Gaby """

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def trim_whitespace(img, threshold=250):
    """Recorta bordes blancos automáticamente"""
    if len(img.shape) == 3:
        gray = np.mean(img, axis=2)
    else:
        gray = img

    rows_with_content = np.where(np.min(gray, axis=1) < threshold)[0]
    cols_with_content = np.where(np.min(gray, axis=0) < threshold)[0]

    if len(rows_with_content) > 0 and len(cols_with_content) > 0:
        return img[
            rows_with_content[0]:rows_with_content[-1] + 1,
            cols_with_content[0]:cols_with_content[-1] + 1
        ]
    return img

def crop_bottom_fraction(img, frac=0.18):
    """Recorta un porcentaje de la parte inferior (barra de color y texto)"""
    h = img.shape[0]
    crop = int(h * frac)
    return img[:-crop, :, :]

base_folder = r"C:\Users\Gaby\Desktop\SOM\SSIM-COMPLETO\PROYECTO_GABY\som_results_ssim\som_4x4_pca_bmu-ssim"
weights_folder = os.path.join(base_folder, "weights_mm")

rows = 4
cols = 4
crop_frac = 0.18  # ajustar si hiciera falta

def resize_to_match(img1, img2):
    """Redimensiona ambas imágenes al mismo tamaño (mínimo común)"""
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])

    img1 = Image.fromarray(img1).resize((w, h))
    img2 = Image.fromarray(img2).resize((w, h))

    return np.array(img1), np.array(img2)


# Nodos por figura (2 nodos x 2 mapas = 4 mapas por fila)
half_cols = 2

# Genera 2 figuras por fila del SOM → 8 figuras en total
for row_idx in range(rows):
    for half_idx in range(cols // half_cols):
        col_start = half_idx * half_cols
        col_end = col_start + half_cols

        fig, axes = plt.subplots(1, half_cols * 2, figsize=(10, 5), constrained_layout=False)

        for i, col_idx in enumerate(range(col_start, col_end)):
            norm_path = os.path.join(
                base_folder,
                f"node_{row_idx}_{col_idx}",
                f"promedio_nodo_{row_idx}_{col_idx}.png"
            )
            mm_path = os.path.join(
                weights_folder,
                f"weight_mm_nodo_{row_idx}_{col_idx}.png"
            )

            img_norm = np.array(Image.open(norm_path))
            img_mm = np.array(Image.open(mm_path))

            # recortar parte inferior primero
            img_norm = crop_bottom_fraction(img_norm, frac=crop_frac)
            img_mm = crop_bottom_fraction(img_mm, frac=crop_frac)

            # recortar bordes blancos después
            img_norm = trim_whitespace(img_norm)
            img_mm = trim_whitespace(img_mm)

            img_norm, img_mm = resize_to_match(img_norm, img_mm)

            ax_norm = axes[i * 2]
            ax_mm = axes[i * 2 + 1]

            ax_norm.imshow(img_norm)
            ax_mm.imshow(img_mm)

            ax_norm.axis("off")
            ax_mm.axis("off")

            # Separador visual entre pares
            if i < half_cols - 1:
                ax_mm.spines['right'].set_visible(True)
                ax_mm.spines['right'].set_linewidth(2)
                ax_mm.spines['right'].set_color('gray')

        # Títulos centrados sobre cada par
        for i, col_idx in enumerate(range(col_start, col_end)):
            ax_left = axes[i * 2]
            ax_right = axes[i * 2 + 1]

            pos_left = ax_left.get_position()
            pos_right = ax_right.get_position()

            x_center = (pos_left.x0 + pos_right.x1) / 2
            y_top = pos_left.y1 + 0.01

            #fig.text(
             #   x_center,
              #  y_top,
               # f"Nodo ({row_idx},{col_idx})",
                #ha="center",
                #va="bottom",
                #fontsize=10
            #)

        plt.subplots_adjust(
            left=0.01,
            right=0.99,
            top=0.88,
            bottom=0.01,
            wspace=0.03
        )

        file_idx = row_idx * (cols // half_cols) + half_idx
        out = rf"C:\Users\Gaby\Desktop\SOM\figura_SOM_row_{file_idx}.png"
        plt.savefig(out, dpi=400, bbox_inches="tight", pad_inches=0.05)
        plt.close()
        print(f"Figura fila {row_idx} mitad {half_idx} guardada en:", out)

plt.show()