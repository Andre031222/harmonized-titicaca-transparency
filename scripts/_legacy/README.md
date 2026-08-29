# Scripts históricos (NO USAR)

Estos son los scripts `10_*` a `25_*` que produjeron las versiones MDPI y
EMA-Springer del artículo, más `26_harmonization_audit.py`, que fue el script
de auditoría con el que se detectaron los defectos.

**Sus números no deben reutilizarse.** La auditoría del 2026-08-22 encontró
cuatro defectos reales en ellos:

1. `12_build_matchups_landsat.py` aplica los coeficientes de Roy et al. (2016),
   ajustados sobre superficies terrestres, a agua oligotrófica oscura, donde el
   intercepto llega a ser 69.8× la señal.
2. `22_uncertainty_benchmark.py` implementa el baseline clásico con
   `log1p`/`expm1` sin acotar; una sola predicción de 143.5 m hundía su R² a
   −2.46 y hacía parecer al Random Forest mucho mejor de lo que es.
3. `21_validation_q1.py` bloquea la validación cruzada por estación, que en un
   programa de campañas no retiene nada, y concluye erróneamente «no hay fuga»
   a partir de que el resultado coincide con el K-fold aleatorio.
4. `18_results_and_figures.py` y el manuscrito describían un hold-out temporal
   «2013–2021 / 2022–2024» que no existe: no hay campañas en 2020, 2021 ni 2023.

El análisis vigente está en [`../../pipeline/`](../../pipeline/), que reproduce
todo desde los mismos match-ups con el diseño corregido. Se conservan aquí por
trazabilidad y porque documentan cómo se construyeron los match-ups en Google
Earth Engine (`10_*` y `12_*`), paso que el pipeline nuevo no repite.
