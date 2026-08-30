# Tutorial: running the pipeline on synthetic data

This walkthrough runs the complete pipeline end to end on a small synthetic
dataset, so you can verify that the code works before downloading the CHIRPS
record. Every step below has been timed on a laptop; the whole tutorial takes
about five minutes.

The synthetic fields are not observations. They are built from six known base
patterns, which means you can check whether the SOM recovers something
sensible rather than staring at an arbitrary map.

---

## 0. Requirements

```
pip install -r requirements.txt
```

The pipeline needs `numpy`, `xarray`, `pandas`, `scipy`, `scikit-learn`,
`matplotlib` and `cartopy`. `netCDF4` is required to read and write NetCDF
files. `tqdm` is optional and only adds a progress bar during training.

---

## 1. Generate the sample dataset

```
python make_sample_data.py --output sample.nc
```

Takes a few seconds and writes a 1.1 MB file. Expected output:

```
=== Generating synthetic dataset ===
  Grid: 40 x 26 (949 land pixels of 1040)
  Time steps: 264 (44 seasons, October-March)
  Base patterns: 6

  Value range: 0.000 to 1.000
  Pattern frequencies: [75, 55, 38, 41, 33, 22]
```

The grid is much coarser than CHIRPS (949 land pixels instead of 89 544) but
covers the same domain, spans the same number of months, and includes an ocean
corner so that the sliding windows of the SSIM are exercised near the coast.

The six base patterns have deliberately uneven frequencies, reproducing the
asymmetry between frequent and rare configurations found in the real record.

---

## 2. Train the final SOM with the full sliding-window SSIM

```
python train_som_full_ssim.py --netcdf sample.nc --grid 4 --init pca --epochs 8 --seed 145 --out som_results_ssim
```

Takes about 90 seconds for a 4×4 network. A 3×3 network takes about 20 seconds;
a 6×6 network is considerably slower, since the SSIM must be evaluated between
each field and every prototype at every presentation.

The script writes `som_results_ssim/som_4x4_pca_bmu-ssim/` containing:

| File | Contents |
|---|---|
| `BMUs.npy` | Best Matching Unit coordinates `(i, j)` for each of the 264 months |
| `weights.npy` | Prototype vectors of every node |
| `valid_idx.npy` | Row and column indices of the land pixels |
| `config.json` | Every parameter actually used in the run |

`config.json` is the authoritative record of what was run. If the reported
parameters and the file ever disagree, the file is right.

To reproduce the three grid sizes used in the study, run the command three
times with `--grid 3`, `--grid 4` and `--grid 6`.

---

## 3. Train the exploratory configurations

The exploratory phase compares BMU metrics over twelve seeds and three grid
sizes. Each script covers one combination of metric and initialization:

```
python multiseed_euclidean_pca.py --netcdf sample.nc --varname anomaly_normalized
python multiseed_correlation_pca.py --netcdf sample.nc --varname anomaly_normalized
python multiseed_ssim_pca.py --netcdf sample.nc --varname anomaly_normalized
```

Each takes about 20 seconds on the synthetic data, covering all three grid
sizes and all twelve seeds. On the real dataset each script takes roughly an
hour.

These scripts only train and store BMU assignments. They deliberately do not
compute validity indices, because comparing metrics requires every partition to
be evaluated under the same distance, which is what the next step does.

**A note on the random-initialization variants.** Running
`multiseed_euclidean_random.py` on this dataset produces a map with a single
occupied node in every run:

```
seed 999: 1/36 nodes occupied
```

This is not a bug and it reproduces what happens on the real data. Random
initialization draws prototypes from a uniform distribution over [−1, 1] while
the input data are normalized to [0, 1], so the prototypes start outside the
range of the data, one of them wins every field by chance, and it is then
pulled further towards the data until it absorbs everything. The SSIM variant
avoids this by sampling its initial prototypes from the observed fields.

---

## 4. Recompute the validity indices under a common distance

```
python recompute_validity_indices.py --netcdf sample.nc --varname anomaly_normalized --ssim-full-root som_results_ssim
```

Takes a few seconds. It reads the BMU assignments already stored on disk, so
nothing is retrained, and writes `validity_indices_unified.csv` with one row
per configuration.

The Silhouette coefficient is reported twice, under Euclidean and under
correlation distance. This is the point of the script: the coefficient requires
a distance to be specified, and that choice is not neutral with respect to the
metric used for training. Comparing the two columns shows how much of the score
comes from the partition and how much from the yardstick.

Davies-Bouldin and Calinski-Harabasz are reported once, since both are defined
in terms of centroids and Euclidean dispersion and admit no choice of distance.

Node occupancy is a count and therefore does not depend on any distance, which
is what makes it usable for comparing configurations trained with different BMU
metrics.

The script prints a summary at the end:

```
=== Summary by training metric and grid size ===
                      n_runs  sil_euclidean  sil_correlation  nodes_occupied
ssim_full       4x4        1        -0.0086           0.0312            16.0
```

---

## 5. Produce the figures

```
python export_bmus_by_date.py --cfg-dir som_results_ssim/som_4x4_pca_bmu-ssim --netcdf sample.nc
python fig03_som_patterns.py --anom-nc sample.nc --time-nc sample.nc
python fig02_node_heatmap.py --csv som_results_ssim/som_4x4_pca_bmu-ssim_BMUs_por_fecha.csv
```

The first command pairs every BMU assignment with its date, producing the table
that the figure scripts read. The second draws the panel showing, for every
node, the mean anomaly field alongside the denormalized prototype. The third
draws the heatmap of months per node.

Both figures are written to `figuras_paper/` as a vector PDF and a 600 dpi PNG.

On the synthetic data the recovered nodes should resemble the six base patterns
defined in `make_sample_data.py`: a widespread deficit in the north, a
widespread surplus, a deficit in the south, a zonal dipole, a surplus in the
northeast, and a weak heterogeneous configuration. If they do, the pipeline is
working.

Figures 1, 4, 5, 6 and 7 are not part of this walkthrough. Figure 1 needs the
Natural Earth map layers, and the rest need the ENSO table and the real season
labels.

---

## Moving to real data

Replace `sample.nc` with your own file. The scripts detect the data variable
automatically and print which one they selected; pass `--varname` to override.
The expected structure is a NetCDF with dimensions `(time, latitude,
longitude)`, monthly time steps, and NaN marking pixels outside the domain.

The CHIRPS v2.0 fields used in the study are available from the Climate Hazards
Center. `preprocess_anomalies.py` performs the detrending, the anomaly
computation against the 1991–2020 climatology, the October–March subsetting and
the per-pixel Min–Max normalization.

---

## Computational requirements

The synthetic tutorial runs on any laptop and needs well under 1 GB of memory.

The real dataset is a different matter. The 264 fields of 89 544 pixels occupy
about 90 MB in single precision, but the full sliding-window SSIM must be
evaluated between every field and every prototype at every presentation, which
is what makes the final training expensive. The runs reported in the study were
carried out on a computing cluster. The exploratory phase, which uses only
global metrics, runs on a desktop machine.
