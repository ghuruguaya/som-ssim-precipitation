#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared style settings for every figure in the manuscript.

Import at the top of each figure script:  from plot_style import *

Centralizing the settings here keeps a single visual language across the paper:
one colour scale for anomaly maps, one categorical palette derived from it, one
typeface size, and one set of column widths. Changing a figure convention means
editing this file rather than seven scripts.

Defines
-------
- Column widths in inches, following the width and height limits of a
  two-column journal page.
- Typography at 8 pt at final size, with line weights no thinner than 0.5 pt.
- Palettes checked for colour-vision deficiency and consistent across figures.
- Helpers: guardar() for export, etiqueta_panel() for panel letters, and a
  common Cartopy map style.

To switch the anomaly maps back to a red-blue scale, change one line:
    CMAP_ANOM_NOMBRE = "RdBu"
and update the caption of Figure 3 accordingly (blue = surplus, red = deficit).

Author: Gabriela Hernandez
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colormaps


# ---------------------------------------------------------------------------
# 1) FIGURE WIDTHS (mm converted to inches).
#    Single column: 50-85 mm; double column: 105-170 mm; maximum height 228 mm.
# ---------------------------------------------------------------------------
MM = 1.0 / 25.4
ANCHO_1COL = 85 * MM      # single-column figure
ANCHO_15COL = 130 * MM    # intermediate width
ANCHO_2COL = 170 * MM     # double-column figure
ALTO_MAX = 228 * MM


# ---------------------------------------------------------------------------
# 2) TYPOGRAPHY AND LINE WEIGHTS
#    Figure text is 8 pt at final size; no line thinner than 0.5 pt.
# ---------------------------------------------------------------------------
def aplicar_estilo():
    """Apply the manuscript-wide Matplotlib settings."""
    mpl.rcParams.update({
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":         8,
        "axes.titlesize":    8,
        "axes.labelsize":    8,
        "xtick.labelsize":   7,
        "ytick.labelsize":   7,
        "legend.fontsize":   8,
        "axes.linewidth":    0.6,
        "lines.linewidth":   0.8,
        "patch.linewidth":   0.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size":  2.5,
        "ytick.major.size":  2.5,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         False,
        "grid.linewidth":    0.4,
        "grid.color":        "0.85",
        "legend.frameon":    False,
        "figure.facecolor":  "white",
        "savefig.facecolor": "white",
    })


aplicar_estilo()


# ---------------------------------------------------------------------------
# 3) PALETTES
#
#    Anomalies (diverging): BrBG, so brown marks a deficit and teal a surplus.
#    The association is intuitive for precipitation and safe for colour-vision
#    deficiency. The dry, transitional and wet categories take the extremes of
#    the same palette, which ties the maps to the categorical charts.
# ---------------------------------------------------------------------------
CMAP_ANOM_NOMBRE = "BrBG"
CMAP_ANOM = colormaps.get_cmap(CMAP_ANOM_NOMBRE).copy()
CMAP_ANOM.set_bad("white")          # ocean left blank

CMAP_SECUENCIAL = "viridis"         # heatmap and bubble chart

# Node categories: three pastels from the same BrBG range, without hatching.
# The teal is slightly stronger so that "Wet" stands out within the range
# rather than by contrast against it.
COL_DRY = "#d8b365"                 # pastel tan
COL_TRANS = "#e8e8e8"               # very light grey
COL_WET = "#5ab4ac"                 # pastel teal
COLORES_CAT = {"Dry": COL_DRY, "Transitional": COL_TRANS, "Wet": COL_WET}
HATCH_CAT = {"Dry": "", "Transitional": "", "Wet": ""}
ORDEN_CAT = ["Dry", "Transitional", "Wet"]

# ENSO phases: pastels harmonized with the above, drawn from RdBu.
COLORES_ENSO = {"El Niño": "#f4a582", "Neutral": "#f7f7f7", "La Niña": "#92c5de"}
HATCH_ENSO = {"El Niño": "", "Neutral": "", "La Niña": ""}

# Anomaly range shared by every map.
VMIN, VMAX = -150, 150
LEVELS = np.linspace(VMIN, VMAX, 21)
TICKS_CBAR = [-135, -90, -45, 0, 45, 90, 135]


# ---------------------------------------------------------------------------
# 4) HELPERS
# ---------------------------------------------------------------------------
def colorbar_limpia(cbar):
    """Strip the frame from a colorbar and thin its ticks."""
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(width=0.5, length=2.5)
    return cbar


def guardar(fig, ruta_sin_extension, dpi_png=600):
    """Export a vector PDF alongside a 600 dpi PNG."""
    fig.savefig(ruta_sin_extension + ".pdf")
    fig.savefig(ruta_sin_extension + ".png", dpi=dpi_png)
    print("saved:", ruta_sin_extension + ".pdf / .png")


def etiqueta_panel(ax, letra, dx=0.02, dy=0.98):
    """Place a panel letter (a), (b) in the upper left corner of an axis."""
    ax.text(dx, dy, f"({letra})", transform=ax.transAxes,
            ha="left", va="top", fontsize=9, fontweight="bold")


def color_texto_contraste(cmap, valor_norm):
    """
    Choose black or white text for legibility over a coloured cell.

    The decision uses the luminance of the actual colormap value rather than a
    fixed threshold on the data, so it holds whatever colormap is in use.
    """
    r, g, b, _ = colormaps.get_cmap(cmap)(valor_norm)
    luminancia = 0.299 * r + 0.587 * g + 0.114 * b
    return "black" if luminancia > 0.5 else "white"


def mapa_base(ax, lon_ticks, lat_ticks, etiquetas_izq=True, etiquetas_abajo=True,
              tam_etiquetas=6):
    """
    Apply the common Cartopy map style: coastlines, borders and a faint grid.

    Labels are drawn only where requested, so that in multi-panel figures they
    appear on the left column and bottom row alone.
    """
    import cartopy.feature as cfeature
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="0.25")
    ax.add_feature(cfeature.STATES, linewidth=0.25, edgecolor="0.55")

    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.75",
                      alpha=0.6, linestyle="--",
                      xlocs=lon_ticks, ylocs=lat_ticks)
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = etiquetas_abajo
    gl.left_labels = etiquetas_izq
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {"size": tam_etiquetas}
    gl.ylabel_style = {"size": tam_etiquetas}
    return gl
