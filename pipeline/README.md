# Pipeline corregido (agosto 2026)

Reemplaza a `scripts/10..25_*.py`, que quedan como referencia histórica pero
**cuyos números no deben reutilizarse**: la auditoría del 2026-08-22 encontró
cuatro defectos reales en ellos (y uno más a punto de cometerse).

## Orden

| Script | Qué corrige / hace |
|---|---|
| `p00_config.py` | Rutas, features, semilla y **diseño de validación** compartidos |
| `p01_build_dataset.py` | Dataset de análisis + inventario real (años y meses que faltan) |
| `p02_fix1_harmonization.py` | **Error 1** — la armonización Roy (2016) es inválida sobre agua oscura |
| `p03_fix2_baselines.py` | **Error 2** — el baseline clásico era un espantapájaros |
| `p04_fix3_validation.py` | **Error 3** — la unidad de bloqueo es la campaña, no la estación |
| `p05_fix4_temporal.py` | **Error 4** — hold-out mal descrito + **error 5**: sensibilidad al año inicial |
| `p06_model_selection.py` | ¿Se pueden mejorar los números? CV anidada + Optuna (lento) |
| `p07_uncertainty.py` | Intervalos conformales, cobertura marginal **y condicional** |
| `p08_interpretability.py` | SHAP fuera de fold + retrievabilidad de las demás variables |
| `p09_export_manuscript_numbers.py` | Exporta `numbers.tex`: el manuscrito no teclea cifras a mano |

`p03` importa de `p04`, así que en `run_all.sh` va después.

## Ejecutar

```bash
bash pipeline/run_all.sh           # todo (~1.5 h por p06)
bash pipeline/run_all.sh --fast    # salta p06
bash pipeline/run_all.sh --no-tex  # sin compilar el PDF
```

## Los cinco defectos

1. **Armonización rota.** Los interceptos de Roy et al. (2016), ajustados sobre
   tierra, superan la señal nativa sobre el agua por factores de 1.4× (verde) a
   69.8× (NIR). Un clasificador distingue los sensores al **99.8 %** después de
   armonizar (azar 66.3 %): el modelo leía la identidad del sensor, no la óptica.
2. **Baseline espantapájaros.** `log1p`/`expm1` sin acotar producía una
   predicción de 143.5 m (máximo real 16.5 m) que hundía el R² a −2.46. Bien
   implementado da +0.10.
3. **Bloqueo en la unidad equivocada.** Bajo folds bloqueados por campaña el
   nulo media-de-estación aún saca R²=0.21 y el de zona 0.29; el de campaña
   colapsa a −0.01. Titular honesto: **R²=0.574, RMSE 1.86 m**.
4. **Hold-out mal descrito.** No hay campañas en 2020, 2021 ni 2023, así que
   «train ≤2021 / test 2022–2024» es «train ≤2019 / test {2022, 2024}». Y todos
   los match-ups son de julio–octubre: sólo época seca.
5. **Tendencias frágiles** (detectado, no cometido). Empezar la serie en 2013 en
   vez de 2011 vuelve «significativas» dos zonas. Se reporta el registro
   completo y la tabla de sensibilidad.

## Figuras

En R con ggplot2, en `figures/R/`. Tema y paleta compartidos en
`theme_titicaca.R`; salidas en `results/figures_r/` (PNG 300 dpi + PDF vectorial,
anchos de columna Elsevier: 90 / 140 / 190 mm).
