#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare monthly CHIRPS precipitation for SOM training.

Four steps are applied in order: a linear trend is removed pixel by pixel,
anomalies are computed against the 1991-2020 monthly climatology, the record is
restricted to October-March, and the resulting anomalies are rescaled to [0, 1]
with a per-pixel Min-Max transformation computed from the retained months only.

Detrending isolates interannual to interdecadal variability from the secular
trend in mean precipitation. The normalization gives all fields a common,
bounded data range, which the SSIM requires to be defined explicitly; the
dynamic range is consequently L = 1.0 throughout the pipeline.

Two files are written: the normalized fields used for training, and optionally
the anomalies in millimetres, which are needed later to map the prototypes back
to physical units.

Inputs
------
chirps_mensual.nc                    Monthly CHIRPS precipitation.
climatologia_mensual_1991_2020.nc    Monthly climatology, dims (month, lat, lon).

Outputs
-------
chirps_normalized.nc                 Normalized anomalies, October-March.
chirps_anom_oct_mar.nc               Anomalies in mm, October-March.
chirps_mensual_detrend.nc            Detrended precipitation, cached and reused.

Usage
-----
    python preprocess_anomalies.py
    python preprocess_anomalies.py --base-dir path/to/data

Author: Gabriela Hernandez
"""

import os
import argparse
import numpy as np
import xarray as xr


# ============================== CONFIGURATION ==============================

DEFAULT_BASE_DIR = "."

FILE_RAW = "chirps_mensual.nc"
FILE_DETRENDED = "chirps_mensual_detrend.nc"
FILE_CLIMATOLOGY = "climatologia_mensual_1991_2020.nc"
FILE_NORMALIZED = "chirps_normalized.nc"
FILE_ANOMALIES = "chirps_anom_oct_mar.nc"

VARIABLE = "precip"
MONTHS = [10, 11, 12, 1, 2, 3]

SAVE_ANOMALIES = True


# ================================ FUNCTIONS ================================

def drop_time_bounds(dataset):
    """Remove the time_bnds variable, which otherwise interferes with the
    date handling in later operations."""
    return dataset.drop_vars("time_bnds") if "time_bnds" in dataset.variables else dataset


def ensure_time_sorted(dataset):
    """Sort by time, so that the trend fit sees the record in order."""
    return dataset.sortby("time") if "time" in dataset.coords else dataset


def detrend_linear(array, dim="time"):
    """
    Remove a linear trend from every pixel by least squares.

    The fit is performed against a plain integer index rather than against the
    datetime coordinate, so that the slope is expressed per time step and the
    computation does not depend on how dates are encoded. The time coordinate
    is temporarily replaced by that index and restored afterwards.
    """
    index = np.arange(array.sizes[dim])
    original_coord = array[dim]

    indexed = array.assign_coords({dim: index})
    fit = indexed.polyfit(dim=dim, deg=1, skipna=True)
    trend = xr.polyval(indexed[dim], fit.polyfit_coefficients)

    detrended = indexed - trend
    return detrended.assign_coords({dim: original_coord})


def interpolate_to(source, reference):
    """Regrid source onto the grid of reference, if the two differ."""
    same_lat = np.array_equal(source["latitude"].values,
                              reference["latitude"].values)
    same_lon = np.array_equal(source["longitude"].values,
                              reference["longitude"].values)
    if same_lat and same_lon:
        return source
    return source.interp(latitude=reference["latitude"],
                         longitude=reference["longitude"])


def minmax_normalize_per_pixel(array):
    """
    Rescale each pixel to [0, 1] using its own temporal minimum and maximum.

    The range is computed over the retained months only, so the transformation
    reflects the variability of the season under study rather than that of the
    full year. Pixels with zero temporal range take the neutral value of 0.5;
    pixels that are missing throughout remain missing.
    """
    vmin = array.min("time", skipna=True)
    vmax = array.max("time", skipna=True)
    denominator = vmax - vmin

    normalized = (array - vmin) / xr.where(denominator > 0, denominator, np.nan)
    normalized = normalized.where(~np.isnan(normalized), 0.5)

    all_missing = array.isnull().all("time")
    normalized = normalized.where(~all_missing)

    return xr.where(normalized < 0, 0, xr.where(normalized > 1, 1, normalized))


def load_or_build_detrended(path_raw, path_detrended, variable, compression):
    """
    Return the detrended precipitation, reusing a cached file when present.

    Detrending the full record is the slowest step, so the result is written to
    disk the first time and read back on subsequent runs.
    """
    if os.path.isfile(path_detrended):
        print(f"Using cached detrended file: {path_detrended}")
        dataset = ensure_time_sorted(drop_time_bounds(xr.open_dataset(path_detrended)))
        if variable not in dataset.data_vars:
            raise ValueError(f"Variable '{variable}' not found in {path_detrended}")
        return dataset[variable]

    print(f"Detrending from: {path_raw}")
    raw = ensure_time_sorted(drop_time_bounds(xr.open_dataset(path_raw)))
    if variable not in raw.data_vars:
        raise ValueError(f"Variable '{variable}' not found in {path_raw}")

    detrended = detrend_linear(raw[variable], dim="time")
    detrended.name = variable
    detrended.attrs.update(raw[variable].attrs)
    detrended.to_dataset().to_netcdf(path_detrended, encoding=compression)
    print(f"Detrended file written: {path_detrended}")
    return detrended


def main():
    parser = argparse.ArgumentParser(
        description="Prepare CHIRPS anomalies for SOM training."
    )
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR,
                        help="Directory holding the input files and receiving the output.")
    parser.add_argument("--varname", default=VARIABLE,
                        help="Variable name inside the input files.")
    args = parser.parse_args()

    base = args.base_dir
    variable = args.varname
    compression = {variable: dict(zlib=True, complevel=4)}

    path_raw = os.path.join(base, FILE_RAW)
    path_detrended = os.path.join(base, FILE_DETRENDED)
    path_climatology = os.path.join(base, FILE_CLIMATOLOGY)
    path_normalized = os.path.join(base, FILE_NORMALIZED)
    path_anomalies = os.path.join(base, FILE_ANOMALIES)

    print("=== Preparing CHIRPS for SOM training ===")

    # 1) Detrended precipitation.
    precipitation = load_or_build_detrended(path_raw, path_detrended,
                                            variable, compression)

    # 2) Monthly climatology, regridded if necessary.
    climatology_ds = drop_time_bounds(xr.open_dataset(path_climatology))
    if variable not in climatology_ds.data_vars:
        raise ValueError(f"Variable '{variable}' not found in {path_climatology}")
    climatology = interpolate_to(climatology_ds[variable], precipitation)

    # 3) Anomalies against the climatology of the same calendar month.
    anomalies = precipitation.groupby("time.month") - climatology
    anomalies = anomalies.rename(variable)
    anomalies.attrs["long_name"] = "Monthly precipitation anomaly (1991-2020 base)"
    anomalies.attrs["note"] = "Computed from detrended precipitation"

    # 4) Restrict to the warm season, keeping every year.
    anomalies_season = anomalies.where(
        anomalies["time"].dt.month.isin(MONTHS), drop=True)

    # 5) Per-pixel Min-Max normalization over the retained months.
    normalized = minmax_normalize_per_pixel(anomalies_season).rename(variable)
    normalized.attrs["long_name"] = "Min-Max normalized anomaly, October-March"
    normalized.attrs["note"] = "Scaled per pixel using October-March only"

    # 6) Write output.
    normalized.to_netcdf(path_normalized, encoding=compression)
    if SAVE_ANOMALIES:
        anomalies_season.to_netcdf(path_anomalies, encoding=compression)

    # 7) Summary.
    print("=== Summary ===")
    print(f"Date range: {str(normalized.time.values[0])[:10]} to "
          f"{str(normalized.time.values[-1])[:10]}")
    print(f"Time steps: {normalized.sizes['time']}")
    print(f"Months included: {sorted(set(normalized['time'].dt.month.values))}")
    print(f"Value range: {float(normalized.min(skipna=True)):.3f} to "
          f"{float(normalized.max(skipna=True)):.3f}")
    print(f"Written: {path_normalized}")
    if SAVE_ANOMALIES:
        print(f"Written: {path_anomalies}")


if __name__ == "__main__":
    main()
