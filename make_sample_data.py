#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a small synthetic dataset for testing the pipeline.

The CHIRPS fields used in the study are too large to distribute with the code,
and are in any case publicly available from their provider. This script builds
a much smaller NetCDF with the same structure, so that the full pipeline can be
run end to end in a few minutes on a laptop: training, BMU assignment, validity
indices and figures.

The synthetic fields are not random noise. A small number of base patterns are
defined as Gaussian anomaly cores at different locations and with different
signs, and each month is drawn from one of them with added noise. The base
patterns are given unequal frequencies, mimicking the asymmetry between
frequent and rare configurations found in the real record. A SOM trained on
this dataset should therefore recover the base patterns, which makes the output
of the tutorial interpretable rather than arbitrary.

The output matches the structure the pipeline expects: dimensions
(time, latitude, longitude), monthly time steps restricted to October-March,
values normalized to [0, 1] per pixel, and NaN marking the ocean.

Usage
-----
    python make_sample_data.py
    python make_sample_data.py --output data/sample.nc --nlat 40 --nlon 26

Author: Gabriela Hernandez
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import xarray as xr


# ============================== CONFIGURATION ==============================

DEFAULT_OUTPUT = "sample_data.nc"
DEFAULT_VARNAME = "anomaly_normalized"

# Grid size. Much coarser than CHIRPS (0.05 degrees) so that the pipeline runs
# quickly; the domain covers the same geographical extent.
DEFAULT_NLAT = 40
DEFAULT_NLON = 26

LAT_RANGE = (-40.0, -20.0)
LON_RANGE = (-65.0, -50.0)

# Record length: 44 seasons of six months, matching the real dataset.
N_SEASONS = 44
FIRST_YEAR = 1981
MONTHS = [10, 11, 12, 1, 2, 3]

# Relative frequency of each base pattern. Deliberately uneven, so that the
# resulting node occupancy is asymmetric as in the real record.
PATTERN_WEIGHTS = [0.28, 0.20, 0.16, 0.14, 0.12, 0.10]

NOISE_LEVEL = 0.35


# ================================ FUNCTIONS ================================

def build_time_axis(n_seasons=N_SEASONS, first_year=FIRST_YEAR, months=MONTHS):
    """
    Build the monthly time axis, October to March, over consecutive seasons.

    A season spans October of year t to March of year t+1, so the last three
    months of each season fall in the following calendar year.
    """
    dates = []
    for s in range(n_seasons):
        year = first_year + s
        for month in months:
            calendar_year = year if month >= 10 else year + 1
            dates.append(pd.Timestamp(year=calendar_year, month=month, day=1))
    return pd.DatetimeIndex(dates)


def build_land_mask(nlat, nlon):
    """
    Build a land mask with an ocean corner.

    The southeastern corner is marked as ocean, which reproduces the situation
    that matters for the SSIM: sliding windows near the coast fall partly
    outside the valid domain and must be handled by the normalized convolution.
    """
    lat_idx, lon_idx = np.meshgrid(np.arange(nlat), np.arange(nlon), indexing="ij")
    # Diagonal coastline across the southeastern corner.
    ocean = (lat_idx / nlat + lon_idx / nlon) > 1.55
    return ~ocean


def gaussian_blob(nlat, nlon, center_lat, center_lon, width, amplitude):
    """Return a 2D Gaussian anomaly core centred at the given relative position."""
    lat_idx, lon_idx = np.meshgrid(
        np.linspace(0, 1, nlat), np.linspace(0, 1, nlon), indexing="ij"
    )
    d2 = ((lat_idx - center_lat) ** 2 + (lon_idx - center_lon) ** 2) / (2 * width ** 2)
    return amplitude * np.exp(-d2)


def build_base_patterns(nlat, nlon):
    """
    Define the base spatial patterns the synthetic months are drawn from.

    The set mirrors the structure found in the real classification: widespread
    deficits, widespread surpluses, and configurations with anomalies of
    opposite sign in different sectors of the domain.

    Returns
    -------
    ndarray of shape (n_patterns, nlat, nlon)
    """
    patterns = [
        # Widespread deficit centred on the north of the domain.
        gaussian_blob(nlat, nlon, 0.30, 0.45, 0.35, -1.0),
        # Widespread surplus covering most of the domain.
        gaussian_blob(nlat, nlon, 0.50, 0.50, 0.40, 1.0),
        # Deficit in the south.
        gaussian_blob(nlat, nlon, 0.75, 0.50, 0.28, -0.9),
        # Zonal dipole: deficit northwest, surplus southeast.
        gaussian_blob(nlat, nlon, 0.30, 0.25, 0.22, -0.8)
        + gaussian_blob(nlat, nlon, 0.65, 0.75, 0.22, 0.8),
        # Surplus concentrated in the northeast.
        gaussian_blob(nlat, nlon, 0.25, 0.70, 0.25, 0.9),
        # Weak, spatially heterogeneous configuration.
        gaussian_blob(nlat, nlon, 0.45, 0.30, 0.18, 0.4)
        + gaussian_blob(nlat, nlon, 0.60, 0.70, 0.18, -0.4),
    ]
    return np.array(patterns)


def normalize_per_pixel(values):
    """
    Rescale each pixel to [0, 1] using its own temporal minimum and maximum.

    This reproduces the Min-Max transformation applied to the real data. Pixels
    with zero temporal range are set to 0.5, and all-NaN pixels are preserved.
    """
    # Ocean pixels are NaN at every time step, which numpy reports as an
    # all-NaN slice. That is expected here, so the warning is suppressed.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        vmin = np.nanmin(values, axis=0)
        vmax = np.nanmax(values, axis=0)
    denom = vmax - vmin

    with np.errstate(invalid="ignore", divide="ignore"):
        normalized = (values - vmin) / np.where(denom > 0, denom, np.nan)

    # Constant pixels take the neutral value.
    constant = (denom == 0)
    normalized[:, constant] = 0.5

    return np.clip(normalized, 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a synthetic NetCDF for testing the SOM pipeline."
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output NetCDF path.")
    parser.add_argument("--varname", default=DEFAULT_VARNAME,
                        help="Name of the variable written to the file.")
    parser.add_argument("--nlat", type=int, default=DEFAULT_NLAT, help="Latitude points.")
    parser.add_argument("--nlon", type=int, default=DEFAULT_NLON, help="Longitude points.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    times = build_time_axis()
    n_time = len(times)
    latitudes = np.linspace(LAT_RANGE[1], LAT_RANGE[0], args.nlat)
    longitudes = np.linspace(LON_RANGE[0], LON_RANGE[1], args.nlon)

    land_mask = build_land_mask(args.nlat, args.nlon)
    patterns = build_base_patterns(args.nlat, args.nlon)

    print("=== Generating synthetic dataset ===")
    print(f"  Grid: {args.nlat} x {args.nlon} "
          f"({int(land_mask.sum())} land pixels of {land_mask.size})")
    print(f"  Time steps: {n_time} ({N_SEASONS} seasons, October-March)")
    print(f"  Base patterns: {len(patterns)}")

    # Draw one base pattern per month, with uneven frequencies, and add noise.
    weights = np.array(PATTERN_WEIGHTS[:len(patterns)])
    weights = weights / weights.sum()
    assignments = rng.choice(len(patterns), size=n_time, p=weights)

    values = np.empty((n_time, args.nlat, args.nlon), dtype=np.float32)
    for t, k in enumerate(assignments):
        noise = rng.normal(0.0, NOISE_LEVEL, size=(args.nlat, args.nlon))
        # A modest amplitude factor per month keeps intensity variable within
        # each pattern, as observed fields are never identical.
        amplitude = rng.uniform(0.7, 1.3)
        values[t] = amplitude * patterns[k] + noise

    values[:, ~land_mask] = np.nan
    normalized = normalize_per_pixel(values)

    dataset = xr.Dataset(
        {args.varname: (("time", "latitude", "longitude"), normalized.astype(np.float32))},
        coords={"time": times, "latitude": latitudes, "longitude": longitudes},
        attrs={
            "title": "Synthetic precipitation anomaly fields for pipeline testing",
            "description": (
                "Artificial data generated by make_sample_data.py. Not observations. "
                "Intended solely to allow the SOM pipeline to be run and inspected "
                "without downloading the full CHIRPS record."
            ),
            "n_base_patterns": len(patterns),
            "seed": args.seed,
        },
    )
    dataset[args.varname].attrs["long_name"] = "Normalized precipitation anomaly (synthetic)"
    dataset[args.varname].attrs["units"] = "1"

    dataset.to_netcdf(args.output)

    print(f"\n  Value range: {float(np.nanmin(normalized)):.3f} to "
          f"{float(np.nanmax(normalized)):.3f}")
    print(f"  Pattern frequencies: "
          f"{np.bincount(assignments, minlength=len(patterns)).tolist()}")
    print(f"  Written: {args.output}")
    print("\nThe file can now be used with train_som_full_ssim.py and the "
          "multiseed scripts. See TUTORIAL.md for a worked example.")


if __name__ == "__main__":
    main()
