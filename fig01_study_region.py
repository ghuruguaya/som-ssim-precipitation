#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 1: study region, in two panels.

    (a) South American context, with the domain outlined.
    (b) Study area (20-40 S, 50-65 W), with provincial boundaries, a scale bar
        and a north arrow.

The background is the Natural Earth shaded relief raster (HYP_HR_SR_OB_DR)
combined with country, province and ocean shapefiles. All of these are
downloaded on first use and cached, so later runs reuse the local copies.

If the raster or the shapefiles are unavailable and cannot be downloaded, the
script falls back to the basic Cartopy features rather than failing. The figure
is then plainer but still correct.

Credit the background in the caption as: shaded relief from Natural Earth
(naturalearthdata.com).

Extra dependencies
------------------
geopandas, shapely and rasterio, used only by this script. Without them the
fallback rendering is used.

Usage
-----
    python fig01_study_region.py
    python fig01_study_region.py --data-dir <cache> --outdir <figures>

Author: Gabriela Hernandez
"""

import os
import zipfile
import argparse
import warnings
import urllib.request

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
from matplotlib.colors import LightSource
from matplotlib.patches import Rectangle, ConnectionPatch
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from plot_style import MM, guardar

warnings.filterwarnings("ignore")


# ============================== CONFIGURATION ==============================

LON_MIN, LON_MAX = -65, -50
LAT_MIN, LAT_MAX = -40, -20
SA_LON_MIN, SA_LON_MAX = -82, -34
SA_LAT_MIN, SA_LAT_MAX = -56, 13

DEFAULT_DATA_DIR = "mapa_area_estudio"
DEFAULT_OUTDIR = "figuras_paper"

PROJ = ccrs.PlateCarree()
COLOR_DOMINIO = "#cc0000"

RELIEF_URL = "https://naciscdn.org/naturalearth/10m/raster/HYP_HR_SR_OB_DR.zip"
COUNTRIES_URL = ("https://naturalearth.s3.amazonaws.com/10m_cultural/"
                 "ne_10m_admin_0_countries.zip")
STATES_URL = ("https://naturalearth.s3.amazonaws.com/10m_cultural/"
              "ne_10m_admin_1_states_provinces.zip")
OCEAN_URL = ("https://naturalearth.s3.amazonaws.com/10m_physical/"
             "ne_10m_ocean.zip")

# Length of the scale bar, in degrees of longitude at 30 S.
ESCALA_GRADOS = 5.0
ESCALA_ETIQUETA = "430 km"


# ============================= DOWNLOAD CACHE ==============================

def download_file(url, destination, description=""):
    """Fetch a file unless it is already present. Failures are not fatal."""
    if os.path.exists(destination):
        return True
    print(f"Downloading {description}...")
    try:
        urllib.request.urlretrieve(url, destination)
        return True
    except Exception as error:
        print(f"  could not download ({error}); continuing without this layer.")
        return False


def download_and_unzip(url, zip_path, extract_dir, description=""):
    """Fetch and unpack a shapefile archive, reusing any local copy."""
    os.makedirs(extract_dir, exist_ok=True)
    found = [f for f in os.listdir(extract_dir) if f.endswith(".shp")]
    if found:
        return os.path.join(extract_dir, found[0])
    if download_file(url, zip_path, description):
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(extract_dir)
        found = [f for f in os.listdir(extract_dir) if f.endswith(".shp")]
        if found:
            return os.path.join(extract_dir, found[0])
    return None


# ================================== LAYERS =================================

def load_clipped(shp_path, xmin, xmax, ymin, ymax):
    """Read a shapefile and clip it to a bounding box; None if unavailable."""
    if shp_path is None or not os.path.exists(shp_path):
        return None
    try:
        import geopandas as gpd
        from shapely.geometry import box
        frame = gpd.read_file(shp_path)
        frame = (frame.set_crs("EPSG:4326") if frame.crs is None
                 else frame.to_crs("EPSG:4326"))
        window = gpd.GeoDataFrame(geometry=[box(xmin, ymin, xmax, ymax)],
                                  crs="EPSG:4326")
        try:
            return gpd.clip(frame, window)
        except Exception:
            return frame.cx[xmin:xmax, ymin:ymax]
    except Exception as error:
        print("  vector layer unavailable:", error)
        return None


def read_rgb_window(tif_path, xmin, xmax, ymin, ymax, max_px=2000):
    """
    Read an RGB window from the relief raster, downsampled if large.

    Returns None when rasterio is missing or the file is absent, which lets the
    caller fall back to plain Cartopy features.
    """
    if not os.path.exists(tif_path):
        return None
    try:
        import rasterio
        from rasterio.windows import from_bounds
        from scipy.ndimage import zoom as ndzoom
        with rasterio.open(tif_path) as source:
            window = from_bounds(xmin, ymin, xmax, ymax, source.transform)
            data = source.read(list(range(1, min(source.count, 3) + 1)),
                               window=window)
        _, height, width = data.shape
        if max(height, width) > max_px:
            factor = max_px / max(height, width)
            data = np.stack([ndzoom(data[i], factor, order=1)
                             for i in range(data.shape[0])])
        rgb = np.transpose(data[:3], (1, 2, 0)).astype(float) / 255.0
        return np.clip(rgb, 0, 1)
    except Exception as error:
        print("  relief raster unavailable:", error)
        return None


def hillshade(rgb):
    """Shade an RGB image using its own luminance as pseudo-elevation."""
    luminance = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    return LightSource(azdeg=315, altdeg=45).shade_rgb(
        rgb, elevation=luminance, blend_mode="overlay")


def plot_gdf(ax, frame, **kwargs):
    """Draw a GeoDataFrame if it exists and is not empty."""
    if frame is not None and len(frame) > 0:
        frame.plot(ax=ax, transform=PROJ, **kwargs)


# ================================== FIGURE =================================

def grilla(ax, lon_ticks, lat_ticks):
    """Draw a faint white graticule with edge labels."""
    gl = ax.gridlines(crs=PROJ, draw_labels=True, linewidth=0.3,
                      color="white", alpha=0.55, linestyle="--",
                      x_inline=False, y_inline=False, zorder=5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator(lon_ticks)
    gl.ylocator = mticker.FixedLocator(lat_ticks)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {"size": 6, "color": "#222222"}
    gl.ylabel_style = {"size": 6, "color": "#222222"}


def fondo(ax, rgb, extent):
    """Paint the shaded relief, or plain land and ocean if it is missing."""
    ax.set_facecolor("#e8e2d4")
    if rgb is not None:
        ax.imshow(hillshade(rgb), extent=extent, transform=PROJ,
                  origin="upper", interpolation="bilinear", zorder=0)
    else:
        ax.add_feature(cfeature.LAND, facecolor="0.93", zorder=0)
        ax.add_feature(cfeature.OCEAN, facecolor="#cfe3ef", zorder=0)


def barra_escala(ax):
    """Draw the scale bar and the north arrow on the study-area panel."""
    x0, y0 = LON_MIN + 0.7, LAT_MIN + 0.8
    x1 = x0 + ESCALA_GRADOS
    ax.plot([x0, x1], [y0, y0], color="black", linewidth=1.2, transform=PROJ,
            zorder=10, solid_capstyle="butt")
    for x in (x0, x1):
        ax.plot([x, x], [y0 - 0.12, y0 + 0.12], color="black", linewidth=1.0,
                transform=PROJ, zorder=10)
    ax.text((x0 + x1) / 2, y0 + 0.3, ESCALA_ETIQUETA, ha="center", va="bottom",
            fontsize=6.5, fontweight="bold", transform=PROJ, zorder=10,
            path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])

    nx, ny = LON_MAX - 1.3, LAT_MIN + 0.9
    ax.annotate("", xy=(nx, ny + 1.8), xytext=(nx, ny),
                xycoords=PROJ._as_mpl_transform(ax),
                textcoords=PROJ._as_mpl_transform(ax),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0,
                                mutation_scale=9), zorder=11)
    ax.text(nx, ny + 2.1, "N", ha="center", va="bottom", fontsize=8,
            fontweight="bold", transform=PROJ, zorder=11,
            path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])


def main():
    parser = argparse.ArgumentParser(description="Draw Figure 1.")
    parser.add_argument("--width", type=float, default=85,
                        help="Figure width in millimetres (85 single column, "
                             "130 intermediate, 170 double column).")
    parser.add_argument("--height", type=float, default=None,
                        help="Figure height in millimetres. Scales with width "
                             "when omitted.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help="Directory holding or receiving the map layers.")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR,
                        help="Directory receiving the figure.")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.outdir, exist_ok=True)

    # --------------------------- Fetch the layers --------------------------
    relief_zip = os.path.join(args.data_dir, "HYP_HR_SR_OB_DR.zip")
    relief_tif = os.path.join(args.data_dir, "HYP_HR_SR_OB_DR.tif")
    if not os.path.exists(relief_tif):
        if download_file(RELIEF_URL, relief_zip,
                         "Natural Earth shaded relief (about 400 MB)") \
                and os.path.exists(relief_zip):
            with zipfile.ZipFile(relief_zip) as archive:
                archive.extractall(args.data_dir)

    countries_shp = download_and_unzip(
        COUNTRIES_URL, os.path.join(args.data_dir, "ne_10m_countries.zip"),
        os.path.join(args.data_dir, "countries"), "country boundaries")
    states_shp = download_and_unzip(
        STATES_URL, os.path.join(args.data_dir, "ne_10m_states.zip"),
        os.path.join(args.data_dir, "states"), "provincial boundaries")
    ocean_shp = download_and_unzip(
        OCEAN_URL, os.path.join(args.data_dir, "ne_10m_ocean.zip"),
        os.path.join(args.data_dir, "ocean"), "oceans")

    print("Loading layers...")
    countries_sa = load_clipped(countries_shp, SA_LON_MIN, SA_LON_MAX,
                                SA_LAT_MIN, SA_LAT_MAX)
    countries_zoom = load_clipped(countries_shp, LON_MIN, LON_MAX,
                                  LAT_MIN, LAT_MAX)
    states_zoom = load_clipped(states_shp, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)
    ocean_sa = load_clipped(ocean_shp, SA_LON_MIN, SA_LON_MAX,
                            SA_LAT_MIN, SA_LAT_MAX)
    ocean_zoom = load_clipped(ocean_shp, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)
    rgb_sa = read_rgb_window(relief_tif, SA_LON_MIN, SA_LON_MAX,
                             SA_LAT_MIN, SA_LAT_MAX)
    rgb_zoom = read_rgb_window(relief_tif, LON_MIN, LON_MAX,
                               LAT_MIN, LAT_MAX, max_px=1500)

    # -------------------------------- Figure -------------------------------
    width = args.width * MM
    height = (args.height * MM) if args.height else (args.width * 1.18 * MM)
    fig = plt.figure(figsize=(width, height))

    # (a) South American context.
    # The left margin has to clear the latitude labels; at 8 pt a four-character
    # label such as 10 S needs roughly a tenth of the figure width.
    ax_context = fig.add_axes([0.11, 0.12, 0.63, 0.84], projection=PROJ)
    ax_context.set_extent([SA_LON_MIN, SA_LON_MAX, SA_LAT_MIN, SA_LAT_MAX],
                          crs=PROJ)
    fondo(ax_context, rgb_sa,
          [SA_LON_MIN, SA_LON_MAX, SA_LAT_MIN, SA_LAT_MAX])
    plot_gdf(ax_context, ocean_sa, facecolor="#7eb8d4", edgecolor="none",
             alpha=0.7, zorder=1)
    plot_gdf(ax_context, countries_sa, facecolor="none", edgecolor="#333333",
             linewidth=0.4, zorder=2)
    ax_context.add_patch(Rectangle(
        (LON_MIN, LAT_MIN), LON_MAX - LON_MIN, LAT_MAX - LAT_MIN,
        linewidth=0.9, edgecolor=COLOR_DOMINIO, facecolor="none",
        linestyle=(0, (4, 3)), transform=PROJ, zorder=10))
    grilla(ax_context, np.arange(-80, -30, 10), np.arange(-60, 20, 10))
    ax_context.set_title("Regional context", fontsize=8, pad=3)

    # (b) Study area.
    ax_area = fig.add_axes([0.575, 0.30, 0.39, 0.52], projection=PROJ, zorder=10)
    ax_area.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=PROJ)
    ax_area.spines["geo"].set_linewidth(0.8)
    fondo(ax_area, rgb_zoom, [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX])
    plot_gdf(ax_area, ocean_zoom, facecolor="#7eb8d4", edgecolor="none",
             alpha=0.7, zorder=1)
    plot_gdf(ax_area, countries_zoom, facecolor="none", edgecolor="#1a1a1a",
             linewidth=0.6, zorder=3)
    plot_gdf(ax_area, states_zoom, facecolor="none", edgecolor="#555555",
             linewidth=0.3, linestyle="--", zorder=3)
    grilla(ax_area, np.arange(-65, -49, 5), np.arange(-40, -19, 5))
    # Right-aligned: the inset overlaps the larger panel, so a centred title
    # would sit partly over the map behind it.
    ax_area.set_title("Study area", fontsize=8, pad=3, loc="right")
    barra_escala(ax_area)

    # Dashed leaders from the outlined domain to the zoom panel.
    for corner, anchor in [((LON_MAX, LAT_MAX), (0, 1)),
                           ((LON_MAX, LAT_MIN), (0, 0))]:
        fig.add_artist(ConnectionPatch(
            xyA=corner, coordsA=ax_context.transData,
            xyB=anchor, coordsB=ax_area.transAxes,
            color=COLOR_DOMINIO, linewidth=0.7,
            linestyle=(0, (4, 3)), zorder=9))

    guardar(fig, os.path.join(args.outdir, "fig01_study_region"))
    plt.close(fig)
    print("Done: Figure 1.")


if __name__ == "__main__":
    main()
