# Pipeline

Each script reads what the previous one writes and leaves its outputs in
`results/metrics/` (aggregate JSON, versioned) and `results/tidy/`
(per-observation tables for the figures; not versioned because they carry
IMARPE field values).

| Script | What it does |
|---|---|
| `p00_config.py` | Paths, bands, features, seed, quality control and the **validation designs**: `splits(d, design)`, `cv_oof`, `cluster_boot` |
| `p01_extract_matchups.py` | Extracts the HLS v2.0 product and each agency's reflectance (Sen2Cor, LaSRC) in Earth Engine at every field measurement, within ±1, 3, 5 and 10 days. Needs Earth Engine credentials and the IMARPE record |
| `p02_build_dataset.py` | Match-up table, locations (coordinates clustered at 500 m), campaigns, spatial blocks and quality control |
| `p03_harmonization.py` | Sensor two-sample test, with negative controls and recalibrations fitted within each fold |
| `p04_validation.py` | Hierarchical null models under six designs, Random Forest under each design, zone extrapolation, temporal hold-out |
| `p05_baselines.py` | Kloiber et al. (2002), blue/green ratio and multiband linear model |
| `p06_model_selection.py` | Random Forest, ExtraTrees, XGBoost and LightGBM tuned with Optuna in nested CV (slow) |
| `p07_uncertainty.py` | Conformal intervals (constant, Mondrian, CQR), marginal and conditional coverage |
| `p08_interpretability.py` | Out-of-fold SHAP and retrievability of other variables |
| `p09_spatial.py` | Moran's I, correlogram, variogram, LISA, Gi* and distance to shore, by location |
| `p10_trends.py` | Trends in the field record (exact Mann–Kendall, Sen, Gilbert) and their sensitivity |
| `p11_windows.py` | Sensitivity to the match-up window, on each window's own sample and on the common sample |
| `p12_multilake_hls.py` | Same-day HLSL30/HLSS30 pairs over large lakes worldwide (`--eval` re-evaluates the cache) |
| `p15_recalibrated_retrieval.py` | Effect of recalibrating one sensor to the other (OLS, RMA, quantile mapping) on retrieval |
| `p16_lake_maps.py` | Lake-wide transparency maps for the 2024 dry season with both sensors (Earth Engine if the composites are missing) |
| `p17_glint.py` | Sun glint angle of each pair and the separability test without glint (`--eval` uses the cache) |
| `p18_toa_separability.py` | The separability test on top-of-atmosphere reflectance, before atmospheric correction (`--eval` uses the cache) |
| `p13_export_numbers.py` | Writes `manuscript/rsase/numbers.tex`: the manuscript does not type numbers by hand |
| `p14_verify_manuscript.py` | Checks that every number and figure in the manuscript comes from `results/` |
| `harmonization_check.py` | The two-sample test as a reusable function (numpy, pandas, scikit-learn) |

## Design

- Spatial unit: the **location**, not the IMARPE station code, which was
  reassigned between years.
- Primary blocking unit: the **campaign** (one per year), leaving one whole
  campaign out at a time. `p04` justifies the choice.
- 95% intervals: bootstrap over whole campaigns.

## Running

```bash
bash pipeline/run_all.sh            # everything (p06 takes ~30 min)
bash pipeline/run_all.sh --fast     # without p06
bash pipeline/run_all.sh --no-tex   # without compiling the manuscript
```

`p13` and `p14` write to and read from `manuscript/rsase/`, which is not part of
this repository; outside the authors' copy they can be skipped.
