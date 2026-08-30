# som-ssim-precipitation

Code accompanying *"Evaluation of Similarity Metrics for Self-Organizing Maps of
Summer Precipitation Anomalies in Eastern Subtropical South America"*.

Hernández, G., Müller, G. V., Vasconcellos, F. C., Vasconcellos, E. C. (under
review).

The pipeline classifies monthly summer precipitation anomaly fields over eastern
subtropical South America with Self-Organizing Maps, comparing four similarity
metrics for the Best Matching Unit search: Euclidean distance, correlation-based
distance, and the Structural Similarity Index in its global and sliding-window
forms.

**Start with [TUTORIAL.md](TUTORIAL.md).** It runs the whole pipeline end to end
on a small synthetic dataset in about five minutes, so you can check that
everything works before downloading the CHIRPS record.

---

## Installation

```
git clone https://github.com/ghuruguaya/som-ssim-precipitation.git
cd som-ssim-precipitation
pip install -r requirements.txt
```

Python 3.9 or later. On Windows, `cartopy` installs more easily through conda:

```
conda install -c conda-forge cartopy
```

`fig01_study_region.py` additionally needs `geopandas`, `shapely` and
`rasterio`. Without them it falls back to a plainer rendering rather than
failing.

---

## Pipeline

Run in this order. Every script takes `--help`.

### 1. Preprocessing

| Script | Reads | Writes |
|---|---|---|
| `preprocess_anomalies.py` | Monthly CHIRPS, 1991–2020 climatology | `chirps_normalized.nc`, `chirps_anom_oct_mar.nc` |

Removes the linear trend pixel by pixel, computes anomalies against the monthly
climatology, restricts the record to October–March, and rescales each pixel to
[0, 1] with a Min–Max transformation. The normalization gives every field a
common bounded range, so the SSIM dynamic range is L = 1.0 throughout.

### 2. Exploratory phase

Six combinations of BMU metric and initialization, each trained over three grid
sizes and twelve seeds. All share `som_exploratory.py`.

| Script | Metric | Initialization |
|---|---|---|
| `multiseed_euclidean_pca.py` | Euclidean | PCA |
| `multiseed_euclidean_random.py` | Euclidean | random |
| `multiseed_correlation_pca.py` | correlation | PCA |
| `multiseed_correlation_random.py` | correlation | random |
| `multiseed_ssim_pca.py` | global SSIM | PCA |
| `multiseed_ssim_random.py` | global SSIM | random |

Each reads the normalized NetCDF and writes one BMU array per grid size and
seed. They deliberately compute no validity indices: comparing metrics requires
every partition to be evaluated under the same distance, which step 4 does.

Roughly one hour per script on the full dataset.

### 3. Final training

| Script | Reads | Writes |
|---|---|---|
| `train_som_full_ssim.py` | `chirps_normalized.nc` | `BMUs.npy`, `weights.npy`, `valid_idx.npy`, `config.json` |
| `export_bmus_by_date.py` | the above | `..._BMUs_por_fecha.csv` |

Trains with the sliding-window SSIM as BMU criterion. `config.json` records
every parameter actually used and is the authoritative account of a run.

### 4. Evaluation

| Script | Purpose |
|---|---|
| `recompute_validity_indices.py` | Silhouette, Davies–Bouldin, Calinski–Harabasz and node occupancy for every configuration |
| `ssim_per_node.py` | mean SSIM between each field and the prototype of its node |
| `ssim_per_node_and_global.py` | mean SSIM among the fields sharing a node |
| `enso_statistics.py` | association between the classification and the ENSO phase |

`recompute_validity_indices.py` is the central one. It reads the BMU
assignments already stored on disk, so nothing is retrained, and reports the
Silhouette coefficient under **both** Euclidean and correlation distance.

That is the point of the script. The coefficient requires a distance to be
specified, and that choice is not neutral with respect to the metric used for
training: the ranking of metrics inverts completely depending on which distance
is used to compute it. Node occupancy is a count, so it does not depend on any
distance and can be compared across metrics.

### 5. Figures

All import `plot_style.py`, which holds the shared palette, typography and
figure widths.

| Script | Produces |
|---|---|
| `fig01_study_region.py` | Figure 1, study region |
| `fig02_node_heatmap.py` | Figure 2, months per node |
| `fig03_som_patterns.py` | Figure 3, the 16 nodes, mean and prototype |
| `fig04_05_06_node_seasonality.py` | Figures 4, 5 and 6, seasonality |
| `fig07_annual_composition_enso.py` | Figure 7, annual composition and ENSO |
| `colorbar_graphical_abstract.py` | standalone colorbar |

### Utilities

| Script | Purpose |
|---|---|
| `make_sample_data.py` | generates the synthetic dataset used by the tutorial |
| `export_months_by_node.py` | writes the months of each node to an Excel workbook |

---

## Data

The scripts expect a NetCDF with dimensions `(time, latitude, longitude)`,
monthly time steps, and NaN marking pixels outside the domain. The data variable
is detected automatically; pass `--varname` to override.

CHIRPS v2.0 monthly precipitation is available from the
[Climate Hazards Center](https://www.chc.ucsb.edu/data/chirps). The files are
too large to distribute here, which is why `make_sample_data.py` exists.

`ENSO_ONI_DJF_1980_2023.csv` is included: the Oceanic Niño Index values are
published by the [NOAA Climate Prediction Center](https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php).

---

## Computational requirements

The tutorial runs on any laptop in a few minutes and needs well under 1 GB of
memory.

The full dataset is another matter. Each of the 264 fields holds 89 544 land
pixels, about 90 MB in single precision, which is manageable. What is expensive
is the sliding-window SSIM, which must be evaluated between every field and
every prototype at every presentation. The final training reported in the paper
was carried out on a computing cluster. The exploratory phase, which uses only
global metrics, runs on a desktop machine.

---

## Citation

If you use this code, please cite the paper and the archived release:

> Hernández, G., Müller, G. V., Vasconcellos, F. C., Vasconcellos, E. C.
> Evaluation of Similarity Metrics for Self-Organizing Maps of Summer
> Precipitation Anomalies in Eastern Subtropical South America. Under review.

Archived on Zenodo: https://doi.org/10.5281/zenodo.22181641

---

## License

MIT. See [LICENSE](LICENSE).

## Contact

Gabriela Hernández — Universidad Nacional del Centro de la Provincia de
Buenos Aires (UNCPBA), Argentina.
