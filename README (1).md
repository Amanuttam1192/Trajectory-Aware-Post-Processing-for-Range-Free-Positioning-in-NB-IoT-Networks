# Trajectory-Aware Post-Processing for Range-Free Positioning in NB-IoT Networks

Code to reproduce every result in the paper *"Trajectory-Aware Post-Processing for Range-Free Positioning in NB-IoT Networks"* (A. Uttam, P. Kumar, NIT Rourkela).

Range-free positioning estimates where an NB-IoT device is by matching the signal strengths it measures against a database of measurements recorded at known places. The best published estimator for this, weighted *k* nearest neighbours (WkNN), reaches about 15.7 m in Oslo and 13.1 m in Rome, and five machine-learning alternatives failed to beat it [1].

This work keeps WkNN unchanged and adds a small correction after it. The measurements were collected from a moving vehicle, so consecutive points are only a few metres apart. An estimate that jumps far away from its neighbours in the sequence must be wrong. A rolling median along each drive removes those jumps without needing GPS, training or extra measurements.

## Results

Same data, features and train/test splits as the reference work [1], window *W* = 5:

| City | Method | Mean [m] | Std [m] | Median [m] | 95th pct [m] |
|---|---|---|---|---|---|
| Oslo | WkNN (Strategy I) [1] | 15.69 | 21.34 | 8.09 | 56.58 |
| Oslo | **WkNN + trajectory smoothing** | **12.12** | **14.22** | 7.56 | **38.62** |
| Rome | WkNN (Strategy I) [1] | 13.15 | 18.85 | 6.13 | 49.19 |
| Rome | **WkNN + trajectory smoothing** | **11.59** | **15.07** | 6.24 | **41.79** |

That is a **22.7 %** (Oslo) and **11.9 %** (Rome) lower mean error, and a 31.7 % and 15.0 % lower 95th-percentile error.

## Repository contents

| File | What it does | Paper result |
|---|---|---|
| `loader.py` | Reads the `.mat` data, builds the SINR fingerprints, computes distances | used by every script |
| `phase1_baseline.py` | WkNN baseline over the authors' 100 train/test splits | baseline reproduction, Section II-B |
| `phase2_loco.py` | WkNN under leave-one-campaign-out, with per-campaign errors | Section V-D |
| `phase3_xgboost.py` | XGBoost compared with WkNN, both protocols | Section V-C |
| `phase4_trajectory.py` | WkNN + trajectory smoothing, both protocols | **Table I** |
| `window_sweep.py` | Smoothing window *W* = 3, 5, 7 and the W = 15 Protocol B gains | **Table II**, Section V-D |
| `make_all_figures.py` | Runs everything again and saves all figures and a CSV of every number | **Fig. 1** and extra plots |
| `Positioning-NBIoT-Beyond-WkNN-main/` | Dataset and MATLAB reference code from the original authors [1], [3] | input data |

Each script imports from the ones above it, so keep them all in the same folder.

## Setup

You need **Python 3.10 or newer** and about 15 MB of disk space. The dataset is already included in this repository, so nothing else needs to be downloaded.

```bash
git clone https://github.com/Amanuttam1192/Trajectory-Aware-Post-Processing-for-Range-Free-Positioning-in-NB-IoT-Networks.git
cd Trajectory-Aware-Post-Processing-for-Range-Free-Positioning-in-NB-IoT-Networks
pip install -r requirements.txt
```

Every script must be **run from the repository's top folder**, because the data path is relative: `REPO = 'Positioning-NBIoT-Beyond-WkNN-main'`. If you keep the dataset somewhere else, change that line at the top of each script.

On Windows, use `python` or `py` in place of `python3` below.

## Reproducing the paper

Run the steps in order. Approximate run times are for a normal laptop.

### Step 1 — Reproduce the published baseline (under 1 min)

```bash
python3 phase1_baseline.py
```

Expected output, matching Table III of [1]:

```
Oslo:  mean 15.67 m  ...  std 21.33 m
Rome:  mean 13.12 m  ...  std 18.65 m
```

If these two numbers match, your setup is correct.

### Step 2 — Leakage-free evaluation (under 1 min)

```bash
python3 phase2_loco.py
```

Holds out one whole drive at a time. Expected overall mean: **96.62 m** (Oslo) and **256.82 m** (Rome). The per-campaign lines include the values quoted in Section V-D: Oslo campaigns 1 and 7 at 37.2 m and 36.8 m, and Rome campaign 3 at 1118 m.

### Step 3 — XGBoost comparison (about 15–30 min)

```bash
python3 phase3_xgboost.py
```

XGBoost should be about **three times worse** than WkNN under Protocol A. The paper reports 43.00 m (Oslo) and 39.55 m (Rome). These values can differ by about 1 m with the XGBoost version and the CPU; we measured 42.25 m and 38.70 m with XGBoost 3.4.1 on Linux. The conclusion does not change.

### Step 4 — Main result: trajectory smoothing (under 1 min)

```bash
python3 phase4_trajectory.py
```

Prints Table I. Expected Protocol A means: **12.12 m** (Oslo, −22.7 %) and **11.59 m** (Rome, −11.9 %).

### Step 5 — Window sensitivity (about 1 min)

```bash
python3 window_sweep.py
```

Expected output:

```
Oslo | Protocol A (Table II): W=1: 15.69   W=3: 12.59   W=5: 12.12   W=7: 12.20
Oslo | Protocol B gain (Sec. V-D): W=5: 3.7%   W=15: 5.1%
Rome | Protocol A (Table II): W=1: 13.15   W=3: 11.61   W=5: 11.59   W=7: 12.07
Rome | Protocol B gain (Sec. V-D): W=5: 2.5%   W=15: 8.5%
```

### Step 6 — All figures (about 5–10 min)

```bash
python3 make_all_figures.py
```

Creates a `figures/` folder:

| File | Content |
|---|---|
| `fig3_cdf_Oslo.png`, `fig3_cdf_Rome.png` | Error CDFs with the mean marked — **Fig. 1** of the paper |
| `fig1_baseline_reproduction.png` | Published vs reproduced baseline |
| `fig2_main_improvement.png` | Mean, median and 95th percentile, before and after smoothing |
| `fig4_cdf_all.png` | Both CDFs side by side |
| `fig5_window_sweep.png` | Error vs window size |
| `fig6_estimator_comparison.png` | WkNN and XGBoost, each with and without smoothing |
| `fig7_error_reduction.png` | Percentage improvement summary |
| `fig8_protocol_comparison.png` | Protocol A vs Protocol B |
| `fig9_percentile_bars.png` | Improvement at each percentile of the error |
| `fig10_trajectory_example.png` | A real drive: true path, raw estimates, smoothed estimates |
| `results_summary.csv` | Every number in one table |

`fig6` also shows that smoothing improves XGBoost (41.7 → 25.7 m in Oslo), which supports the claim in the abstract that the layer works with other estimators. Set `RUN_XGBOOST = False` at the top of the script to skip the slow XGBoost part.

The scripts also save `.npy` files of the raw errors. These are regenerated on every run and are ignored by git.

## Evaluation protocols

- **Protocol A — random split.** 70 % of points form the database and 30 % are tested, using the authors' pre-generated splits. This makes the results directly comparable with [1].
- **Protocol B — leave-one-campaign-out.** One entire drive is held out at a time. This removes leakage between nearly identical neighbouring points, but it also removes database coverage for streets visited by only one drive, so it should be read as a pessimistic bound.

## Data and credits

The dataset and the MATLAB reference implementation in `Positioning-NBIoT-Beyond-WkNN-main/` were released by De Nardis *et al.* under the **GNU GPL v3.0** licence (see the `LICENSE` file in that folder). The raw data is also available on Zenodo [3]. Please cite their work if you use the data.

## References

[1] L. De Nardis *et al.*, "Range-Free Positioning in NB-IoT Networks by Machine Learning: Beyond WkNN," *IEEE Journal of Indoor and Seamless Positioning and Navigation*, vol. 3, pp. 53–69, 2025. doi: [10.1109/JISPIN.2025.3558465](https://doi.org/10.1109/JISPIN.2025.3558465)

[2] L. De Nardis *et al.*, "Positioning by Multicell Fingerprinting in Urban NB-IoT Networks," *Sensors*, vol. 23, no. 9, p. 4266, 2023. doi: [10.3390/s23094266](https://doi.org/10.3390/s23094266)

[3] L. De Nardis *et al.*, "Outdoor NB-IoT and 5G Coverage and Channel Information Data in Urban Environments," Zenodo, 2025. doi: [10.5281/zenodo.7674298](https://doi.org/10.5281/zenodo.7674298)

[4] Original reference code: [github.com/lucadn/Positioning-NBIoT-Beyond-WkNN](https://github.com/lucadn/Positioning-NBIoT-Beyond-WkNN)

## Authors

Aman Uttam and Prof. Pawan Kumar, Department of Electronics and Communication Engineering, National Institute of Technology Rourkela, India.
