#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exploratory SOM training: Euclidean BMU metric with PCA initialization.

Trains a Self-Organizing Map on normalized monthly precipitation anomaly fields
(October-March) for every combination of grid size and random seed, and writes
the resulting BMU assignments to disk. The training procedure, the similarity
metrics and the initialization schemes are defined in som_exploratory.py.

Usage
-----
    python multiseed_euclidean_pca.py
    python multiseed_euclidean_pca.py --netcdf path/to/file.nc --outdir path/to/output

Author: Gabriela Hernandez
"""

import os

from som_exploratory import build_parser, run_multiseed

DEFAULT_OUTDIR = os.path.join("EUCLID", "som_euclid_pca_mascara_multisemilla")


def main():
    parser = build_parser(
        "Train SOMs with a euclidean BMU metric and pca initialization."
    )
    args = parser.parse_args()
    run_multiseed(metric="euclidean", init="pca",
                  default_outdir=DEFAULT_OUTDIR,
                  prefix="bmus_euclid_pca", args=args)


if __name__ == "__main__":
    main()
