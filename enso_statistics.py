#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Relationship between the SOM classification and the ENSO phase.

Each month is coded by the category of the node it was assigned to, and the
codes are averaged over the six months of each season to give a seasonal
dry-wet index. The index is then compared with the December-January-February
Oceanic Nino Index in two ways: across the three ENSO phases, and against the
ONI value itself.

Because the index is discrete, derived from three levels rather than measured
on a continuous scale, every test is reported alongside its distribution-free
counterpart. The analysis of variance is accompanied by the Kruskal-Wallis
test, and the Pearson correlation by the Spearman rank correlation, so the
conclusions do not rest on an assumption of normality. Shapiro-Wilk and Levene
tests are reported for the same reason.

Inputs
------
BMU table written by export_bmus_by_date.py, with columns fecha, nodo_i, nodo_j.
ENSO table with columns season_year, ONI_DJF and ENSO_phase.

Usage
-----
    python enso_statistics.py
    python enso_statistics.py --bmu-csv <table.csv> --enso-csv <oni.csv>

Author: Gabriela Hernandez
"""

import os
import argparse

import numpy as np
import pandas as pd
from scipy import stats


# ============================== CONFIGURATION ==============================

DEFAULT_BMU_CSV = os.path.join("som_results_ssim",
                               "som_4x4_pca_bmu-ssim_BMUs_por_fecha.csv")
DEFAULT_ENSO_CSV = "ENSO_ONI_DJF_1980_2023.csv"
DEFAULT_OUT = "resumen_supuestos_estadisticos_ENSO.csv"

ORDEN_ENSO = ["El Niño", "Neutral", "La Niña"]

# Node categories, assigned in Section 3.6 from the sign, extent and relative
# magnitude of the mean anomaly field of each node.
NODOS_SECOS = {(0, 0), (0, 1), (1, 0)}
NODOS_HUMEDOS = {(2, 2), (2, 3), (3, 1), (3, 2), (3, 3)}


# ================================ FUNCTIONS ================================

def categoria_a_valor(i, j):
    """
    Code a node as -1 for dry, +1 for wet and 0 for transitional.

    The classification is the one established in Section 3.6 by inspecting the
    mean anomaly field of each node; this function only translates it into
    numbers. Averaging the codes over a season then gives an index that runs
    from -1, every month dry, to +1, every month wet.
    """
    if (i, j) in NODOS_SECOS:
        return -1
    if (i, j) in NODOS_HUMEDOS:
        return 1
    return 0


def build_seasonal_index(bmu_csv, enso_csv):
    """
    Build the seasonal dry-wet index and attach the ENSO phase of each season.

    A season runs from October of year t to March of year t+1, so the first
    three calendar months of a year belong to the season labelled by the
    preceding year.
    """
    table = pd.read_csv(bmu_csv)
    table["fecha"] = pd.to_datetime(table["fecha"])
    table["year"] = table["fecha"].dt.year
    table["month"] = table["fecha"].dt.month
    table["season_year"] = np.where(table["month"] >= 10,
                                    table["year"], table["year"] - 1)
    table["dw_index"] = [categoria_a_valor(i, j)
                         for i, j in zip(table["nodo_i"], table["nodo_j"])]

    seasonal = (table.groupby("season_year")["dw_index"].mean()
                .reset_index().rename(columns={"dw_index": "mean_dw_index"}))

    enso = pd.read_csv(enso_csv)
    return seasonal.merge(enso[["season_year", "ONI_DJF", "ENSO_phase"]],
                          on="season_year")


def main():
    parser = argparse.ArgumentParser(
        description="Test the association between the SOM classification and ENSO."
    )
    parser.add_argument("--bmu-csv", default=DEFAULT_BMU_CSV,
                        help="BMU table written by export_bmus_by_date.py.")
    parser.add_argument("--enso-csv", default=DEFAULT_ENSO_CSV,
                        help="CSV with season_year, ONI_DJF and ENSO_phase.")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help="Output CSV summarizing every test.")
    args = parser.parse_args()

    merged = build_seasonal_index(args.bmu_csv, args.enso_csv)
    groups = {phase: merged.loc[merged["ENSO_phase"] == phase,
                                "mean_dw_index"].values
              for phase in ORDEN_ENSO}

    print("=== Sample size by phase ===")
    for phase, values in groups.items():
        print(f"  {phase}: n = {len(values)}")

    print("\n=== Normality within groups (Shapiro-Wilk) ===")
    shapiro_p = {}
    for phase, values in groups.items():
        statistic, pvalue = stats.shapiro(values)
        shapiro_p[phase] = pvalue
        verdict = "normal" if pvalue > 0.05 else "not normal"
        print(f"  {phase}: W = {statistic:.3f}, p = {pvalue:.3f} ({verdict})")

    print("\n=== Homogeneity of variance (Levene) ===")
    levene_stat, levene_p = stats.levene(*groups.values())
    verdict = "homogeneous" if levene_p > 0.05 else "not homogeneous"
    print(f"  statistic = {levene_stat:.3f}, p = {levene_p:.3f} ({verdict})")

    print("\n=== Difference among phases ===")
    anova_f, anova_p = stats.f_oneway(*groups.values())
    kruskal_h, kruskal_p = stats.kruskal(*groups.values())
    print(f"  ANOVA:          F = {anova_f:.3f}, p = {anova_p:.4f}")
    print(f"  Kruskal-Wallis: H = {kruskal_h:.3f}, p = {kruskal_p:.4f}")

    print("\n=== Normality of the full series ===")
    shapiro_index = stats.shapiro(merged["mean_dw_index"])
    shapiro_oni = stats.shapiro(merged["ONI_DJF"])
    print(f"  dry-wet index: W = {shapiro_index[0]:.3f}, p = {shapiro_index[1]:.3f}")
    print(f"  DJF ONI:       W = {shapiro_oni[0]:.3f}, p = {shapiro_oni[1]:.3f}")

    print("\n=== Correlation with the DJF ONI ===")
    pearson_r, pearson_p = stats.pearsonr(merged["ONI_DJF"],
                                          merged["mean_dw_index"])
    spearman_rho, spearman_p = stats.spearmanr(merged["ONI_DJF"],
                                               merged["mean_dw_index"])
    print(f"  Pearson:  r = {pearson_r:.3f}, p = {pearson_p:.2e}")
    print(f"  Spearman: rho = {spearman_rho:.3f}, p = {spearman_p:.2e}")

    summary = pd.DataFrame([{
        "n_el_nino": len(groups["El Niño"]),
        "n_neutral": len(groups["Neutral"]),
        "n_la_nina": len(groups["La Niña"]),
        "shapiro_p_el_nino": shapiro_p["El Niño"],
        "shapiro_p_neutral": shapiro_p["Neutral"],
        "shapiro_p_la_nina": shapiro_p["La Niña"],
        "levene_p": levene_p,
        "anova_F": anova_f,
        "anova_p": anova_p,
        "kruskal_H": kruskal_h,
        "kruskal_p": kruskal_p,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
    }])
    summary.to_csv(args.out, index=False)
    print(f"\nWritten: {args.out}")


if __name__ == "__main__":
    main()
