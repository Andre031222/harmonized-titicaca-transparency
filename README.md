# Harmonized Sentinel-2 / Landsat Retrieval of Water Transparency in Lake Titicaca

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Google Earth Engine](https://img.shields.io/badge/data-Google%20Earth%20Engine-success.svg)](https://earthengine.google.com)
[![Journal](https://img.shields.io/badge/target-Environmental%20Monitoring%20%26%20Assessment%20(Springer)-1a7f8e.svg)](https://link.springer.com/journal/10661)
[![Status](https://img.shields.io/badge/status-under%20peer%20review-orange.svg)](#citation)

Reproducible pipeline for retrieving **water transparency (Secchi disk depth, *Z*<sub>SD</sub>)**
in **Lake Titicaca** (3,810 m a.s.l.) from **harmonized Sentinel-2 MSI and
Landsat 8/9 OLI** surface reflectance, using machine learning validated against
**real in-situ measurements (2013–2024)**.

> **Companion code for the manuscript:**
> *Transferable Satellite-Based Monitoring Protocol for Water Transparency in
> High-Altitude Oligotrophic Lakes: Harmonized Sentinel-2/Landsat Retrieval,
> Rigorous Validation and Conformal Uncertainty in Lake Titicaca (2013–2024).*
> Under peer review at *Environmental Monitoring and Assessment* (Springer).

---

## Table of contents
- [Overview](#overview)
- [Key results](#key-results)
- [Repository structure](#repository-structure)
- [Data](#data)
- [Installation](#installation)
- [Reproducing the study](#reproducing-the-study)
- [Methodology](#methodology)
- [Figures](#figures)
- [Honest scope and limitations](#honest-scope-and-limitations)
- [Citation](#citation)
- [License & contact](#license--contact)

---

## Overview

High-altitude Andean lakes are critical freshwater reserves yet remain poorly
monitored. This repository builds, to our knowledge, the **first multi-mission
harmonized transparency retrieval for a high-altitude Andean lake**, using
**only real in-situ measurements** sampled at their true station coordinates.

- **1,002 real satellite–field match-ups** (2013–2024) were assembled by pairing
  cloud-free Sentinel-2 and Landsat 8/9 imagery with **881 in-situ measurements**
  from IMARPE limnological campaigns, across the whole lake (156 stations).
- A **Random Forest** trained on twelve harmonized spectral features retrieves
  *Z*<sub>SD</sub> under **strict station-wise (GroupKFold) cross-validation** and
  a **temporal hold-out**.
- **SHAP** interpretability shows the retrieval is driven by green/blue
  reflectance ratios — physically consistent with water-clarity optics.
- The study **transparently delimits what is *not* retrievable** (chlorophyll-*a*,
  suspended solids, temperature) in this clear oligotrophic regime.

---

## Key results

All metrics are **out-of-fold**, under station-wise GroupKFold (a station never
appears in both train and test). RMSE and MAE in metres.

| Configuration | R² | RMSE | MAE | n |
|---|:---:|:---:|:---:|:---:|
| Random Forest — Sentinel-2 only | **0.64** | 1.85 | 1.46 | 274 |
| Random Forest — combined (S2 + Landsat) | 0.60 | 1.81 | 1.40 | 812 |
| Random Forest — Landsat only | 0.58 | 1.75 | 1.33 | 538 |
| XGBoost — combined | 0.57 | 1.87 | 1.44 | 812 |
| Classical blue/green band ratio | 0.39 | 2.42 | 1.94 | 812 |
| **Temporal hold-out** (train ≤2021, test 2022–2024) | 0.51 | 2.63 | 2.13 | 226 |

**Validation hierarchy (transferable protocol, demonstrated empirically):**
- **No spatial leakage:** random K-fold and station-wise GroupKFold give an
  *identical* R² (0.60) — the model does not memorize location.
- **Extrapolation collapses:** leave-one-zone-out (withholding an entire trophic
  zone) drops R² to ~0 — the model interpolates within sampled optical regimes
  but cannot extrapolate to an unseen water type. Random splits alone hide this.
- **Range restriction:** restricting to homogeneous clear/turbid subsets halves
  R² (to ~0.29) even though RMSE *improves* — so **RMSE is the fair cross-study
  metric**, not R² alone.

**Calibrated uncertainty:** split-conformal prediction intervals are well
calibrated (empirical coverage **89.8%** at a 90% nominal level), giving a
per-pixel confidence band that a single R² cannot convey.

---

## Repository structure

```
.
├── scripts/
│   ├── 10_build_matchups.py         Sentinel-2 satellite–field match-ups (GEE, real in-situ)
│   ├── 12_build_matchups_landsat.py Landsat 8/9 harmonized match-ups (Roy et al. 2016) + thermal band
│   ├── 11_train_secchi.py           Secchi model (GroupKFold, no leakage)
│   ├── 13_train_multimission.py     Combined S2+Landsat model + SHAP + temperature
│   ├── 14_improve_secchi.py         Honest improvement experiments
│   ├── 15_recalibrate_landsat.py    Local Landsat→S2 water recalibration (CDF matching)
│   ├── 16_extract_nearest.py        Nearest-image match-up variant (tested, not adopted)
│   ├── 17_validation_experiments.py Leakage & range-restriction experiments
│   ├── 18_results_and_figures.py    Production model, figures, temporal trends
│   ├── 19_map_and_harmonization.py  Study-area map, spectral signature
│   ├── 20_transparency_map.py       Lake-wide transparency map (model applied to imagery)
│   ├── 21_validation_q1.py          Validation hierarchy (random/station/temporal/zone-out) + range restriction
│   ├── 22_uncertainty_benchmark.py  Conformal prediction intervals + head-to-head benchmark
│   ├── 23_figures_q1.py             Validation-hierarchy and uncertainty-calibration figures
│   ├── 24_dataflow_graphical_abstract.py  Data-flow diagram + graphical abstract
│   └── 25_protocol_diagram.py       Conceptual monitoring-protocol diagram
├── data/
│   ├── processed/                   Match-up tables (matchups_s2.csv, matchups_ls.csv)
│   ├── insitu/reference/            OEFA 2016 reference measurements + source metadata
│   └── lake_boundary/               Lake geometry (Natural Earth / GPKG)
├── results/
│   ├── figures/                     All manuscript figures (600 dpi)
│   └── metrics/                     CV metrics, SHAP importance, validation experiments
├── manuscript/                      LaTeX source, compiled PDF, cover letter, references
└── environment.yml / requirements.txt
```

---

## Data

| Source | What | Availability |
|---|---|---|
| **IMARPE** limnological campaigns | In-situ Secchi, chlorophyll-*a*, nutrients, etc. (881 measurements, 2011–2024) | Published records; July 2019 campaign in Siguayro & Franco (2022). Raw compilation available on request. |
| **OEFA (2016)** | Reference measurements, Inner Puno Bay | Public government report |
| **Sentinel-2 / Landsat 8-9** | Surface reflectance | Free via [Google Earth Engine](https://earthengine.google.com) |
| **Match-up tables** (`data/processed/`) | Satellite reflectance ↔ in-situ pairs (1,002) | Included in this repository |

> **Provenance check:** our 2019 zone-level means reproduce the published IMARPE
> values almost exactly (e.g. Lago Mayor conductivity 1513.4 vs 1513.5 µS/cm;
> Lago Menor chlorophyll-*a* 1.52 vs 1.52 mg/m³), confirming the in-situ data
> originate from IMARPE monitoring campaigns. **No synthetic or
> spectrally-derived labels are used at any stage.**

---

## Installation

```bash
# 1. Clone
git clone https://github.com/Andre031222/harmonized-titicaca-transparency.git
cd harmonized-titicaca-transparency

# 2. Environment (conda recommended)
conda env create -f environment.yml
conda activate titicaca-wq
#   or:  pip install -r requirements.txt

# 3. Authenticate Google Earth Engine (only needed to re-extract imagery)
earthengine authenticate
```

**Core requirements:** Python 3.11 · `earthengine-api` · `scikit-learn` ·
`xgboost` · `shap` · `geopandas` · `rasterio` · `matplotlib`.

---

## Reproducing the study

The match-up tables and all manuscript figures are already included. The Random
Forest trains in seconds from the bundled match-ups, so you can **retrain the
model and regenerate every figure without Earth Engine**:

```bash
# Train the transparency model + figures + temporal trends
python scripts/18_results_and_figures.py

# Multi-mission model + SHAP + temperature validation
python scripts/13_train_multimission.py

# Study-area map and cross-sensor spectral signature
python scripts/19_map_and_harmonization.py

# --- Q1 validation suite (regenerates the manuscript's core results) ---
# Validation hierarchy (random/station/temporal/zone-out) + range restriction
python scripts/21_validation_q1.py
# Conformal prediction intervals + head-to-head benchmark
python scripts/22_uncertainty_benchmark.py
# Validation-hierarchy and uncertainty-calibration figures
python scripts/23_figures_q1.py
# Data-flow diagram + graphical abstract
python scripts/24_dataflow_graphical_abstract.py
```

To **rebuild the match-ups from scratch** (requires Earth Engine authentication):

```bash
python scripts/10_build_matchups.py            # Sentinel-2 match-ups
python scripts/12_build_matchups_landsat.py    # Landsat match-ups (harmonized)
python scripts/20_transparency_map.py          # lake-wide transparency map
```

---

## Methodology

1. **Match-ups** — for each in-situ measurement, cloud-masked median reflectance
   is extracted in a 30 m buffer at the true station coordinate, within ±10 days.
2. **Harmonization** — Landsat SR is transformed to Sentinel-2-equivalent
   reflectance using the per-band coefficients of Roy et al. (2016); the six
   common bands (B2, B3, B4, B8/NIR, B11, B12) are retained.
3. **Features** — twelve sensor-agnostic features (bands + NDWI, NDTI and
   blue/green, green/blue, red/green, blue/red ratios).
4. **Model** — Random Forest (500–600 trees, depth 12); robust scaler fitted
   **only on the training fold** (no leakage). XGBoost reported for comparison.
5. **Validation hierarchy** — random K-fold → station-wise GroupKFold →
   temporal hold-out (2022–2024, including the 2023–2024 El Niño) →
   leave-one-zone-out, to separate genuine generalization from
   spatial-autocorrelation leakage and expose where extrapolation fails.
6. **Uncertainty** — per-pixel split-conformal prediction intervals
   (empirical coverage 89.8% at a 90% nominal level).
7. **Interpretation** — SHAP (TreeExplainer); benchmarked against a classical
   blue/green band-ratio algorithm and a multiband linear model.

---

## Figures

All figures are in [`results/figures/`](results/figures/) at 600 dpi:

| File | Content |
|---|---|
| `fig_study_area.png` | Monitoring stations coloured by measured transparency |
| `fig_scatter_secchi.png` | Out-of-fold predicted vs in-situ Secchi depth |
| `fig_validation_hierarchy.png` | Validation hierarchy (random → station → temporal → zone-out) |
| `fig_uncertainty.png` | Conformal prediction intervals + coverage calibration |
| `fig_shap_importance.png` | SHAP feature importance (green/blue dominate) |
| `fig_transparency_map.png` | Lake-wide modelled transparency (2022 composite) |
| `fig_trends.png` | In-situ transparency trends by zone (Mann–Kendall) |
| `fig_spectral_signature.png` | Cross-sensor water-leaving reflectance |
| `fig_protocol.png` | Conceptual diagram of the transferable monitoring protocol |

---

## Honest scope and limitations

This study is deliberately transparent about its envelope:

- **Water transparency (Secchi depth) is retrievable** — R² ≈ 0.64, RMSE ≈ 1.8 m.
- **Chlorophyll-*a*, suspended solids and thermal temperature are *not*
  retrievable** in this clear oligotrophic regime; their in-situ ranges lie below
  the optical/thermal detection limit. We report this explicitly to caution
  against transferring turbid-water algorithms to clear high-altitude lakes.
- **Caveats:** the field data are dry-season biased; transparency above ~15 m
  saturates the signal; the harmonized Landsat NIR shows a known positive bias
  over dark water and should be locally recalibrated in future work.

---

## Citation

If you use this code or data, please cite the manuscript:

```bibtex
@article{vilcasolorzano2026titicaca,
  author  = {Mamani-Calisaya, Milton Vladimir and Vilca-Solorzano, Richar Andre
             and Yana-Yucra, Dina Maribel and Torres-Cruz, Fred},
  title   = {Transferable Satellite-Based Monitoring Protocol for Water
             Transparency in High-Altitude Oligotrophic Lakes: Harmonized
             {Sentinel-2}/{Landsat} Retrieval, Rigorous Validation and
             Conformal Uncertainty in {Lake Titicaca} (2013--2024)},
  journal = {Environmental Monitoring and Assessment},
  year    = {2026},
  note    = {Under review}
}
```

---

## License & contact

Released under the [MIT License](LICENSE).

**Corresponding author:** Richar Andre Vilca-Solorzano — `75521963@est.unap.edu.pe`
Faculty of Statistical and Computer Engineering, Universidad Nacional del
Altiplano (UNAP), Puno, Peru.

*In-situ data courtesy of IMARPE and OEFA monitoring programmes. Satellite imagery
via Google Earth Engine.*
