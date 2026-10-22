# Pipeline

Cada script lee lo que escribe el anterior y deja sus salidas en
`results/metrics/` (JSON agregados, versionados) y `results/tidy/` (tablas por
observacion para las figuras; no se versionan porque llevan valores de campo
de IMARPE).

| Script | Que hace |
|---|---|
| `p00_config.py` | Rutas, bandas, features, semilla, control de calidad y **diseno de validacion**: `splits(d, diseno)`, `cv_oof`, `cluster_boot` |
| `p01_extract_matchups.py` | Extrae en Earth Engine el producto HLS v2.0 y la reflectancia de cada agencia (Sen2Cor, LaSRC) en cada medicion de campo, a +-1, 3, 5 y 10 dias. Necesita credenciales y el registro de IMARPE |
| `p02_build_dataset.py` | Tabla de match-ups, ubicaciones (coordenadas agrupadas a 500 m), campanas, bloques espaciales y control de calidad |
| `p03_harmonization.py` | Prueba de dos muestras entre sensores, con controles negativos y recalibraciones ajustadas dentro de cada fold |
| `p04_validation.py` | Nulos jerarquicos bajo seis disenos, RF bajo cada diseno, extrapolacion por zona, hold-out temporal |
| `p05_baselines.py` | Kloiber (2002), cociente azul/verde y modelo lineal multibanda |
| `p06_model_selection.py` | RF, ExtraTrees, XGBoost y LightGBM con Optuna en CV anidada (lento) |
| `p07_uncertainty.py` | Intervalos conformales (constante, Mondrian, CQR), cobertura marginal y condicional |
| `p08_interpretability.py` | SHAP fuera de fold y recuperabilidad de otras variables |
| `p09_spatial.py` | Moran, correlograma, variograma, LISA, Gi* y distancia a la orilla, por ubicacion |
| `p10_trends.py` | Tendencias del registro de campo (Mann-Kendall exacto, Sen, Gilbert) y su sensibilidad |
| `p11_windows.py` | Sensibilidad a la ventana del match-up, en la muestra propia y en la comun |
| `p12_multilake_hls.py` | Pares del mismo dia HLSL30/HLSS30 en lagos grandes del mundo (`--eval` reevalua la cache) |
| `p15_recalibrated_retrieval.py` | Efecto de recalibrar un sensor al otro (OLS, RMA, mapeo de cuantiles) sobre la recuperacion |
| `p16_lake_maps.py` | Mapas de transparencia del lago, estacion seca 2024, con ambos sensores (Earth Engine si faltan los compuestos) |
| `p17_glint.py` | Angulo de brillo solar de cada par y prueba de separabilidad sin glint (`--eval` usa la cache) |
| `p18_toa_separability.py` | La prueba de separabilidad en reflectancia TOA, antes de la correccion atmosferica (`--eval` usa la cache) |
| `p13_export_numbers.py` | Escribe `manuscript/rsase/numbers.tex`: el texto no teclea cifras |
| `p14_verify_manuscript.py` | Comprueba que cada cifra y figura del manuscrito sale de `results/` |
| `harmonization_check.py` | La prueba de dos muestras como funcion reutilizable (numpy, pandas, sklearn) |

## Diseno

- Unidad espacial: la **ubicacion**, no el codigo de estacion de IMARPE, que
  se reasigno entre anios.
- Unidad de bloqueo primaria: la **campana** (una por anio), dejando fuera
  una entera cada vez. Lo justifica p04.
- Intervalos del 95 %: bootstrap de campanas enteras.

## Ejecutar

```bash
bash pipeline/run_all.sh            # todo (p06 tarda ~30 min)
bash pipeline/run_all.sh --fast     # sin p06
bash pipeline/run_all.sh --no-tex   # sin compilar el manuscrito
```

Los trabajos se fijan a los nucleos de eficiencia (`taskset -c 8-15`) porque
el portatil se calienta con todos los nucleos.
