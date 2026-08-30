#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exploratory SOM training: Euclidean BMU metric with random initialization.

Trains a Self-Organizing Map on normalized monthly precipitation anomaly fields
(October-March) for every combination of grid size and random seed, and writes
the resulting BMU assignments to disk. The training procedure, the similarity
metrics and the initialization schemes are defined in som_exploratory.py.

This combination collapses onto a single occupied node in every run. The
behaviour is reproducible and is explained in TUTORIAL.md: the prototypes are
drawn from a uniform distribution over [-1, 1] while the input data lie in
[0, 1], so they start outside the range of the data.

Usage
-----
    python multiseed_euclidean_random.py
    python multiseed_euclidean_random.py --netcdf path/to/file.nc --outdir path/to/output

Author: Gabriela Hernandez
"""

import os

from som_exploratory import build_parser, run_multiseed

DEFAULT_OUTDIR = os.path.join("EUCLID", "som_eucl_random_mascara_multisemilla")


def main():
    parser = build_parser(
        "Train SOMs with a euclidean BMU metric and random initialization."
    )
    args = parser.parse_args()
    run_multiseed(metric="euclidean", init="random",
                  default_outdir=DEFAULT_OUTDIR,
                  prefix="bmus_eucl_random", args=args)


if __name__ == "__main__":
    main()
