# Métricas históricas (NO USAR)

Producidas por `scripts/_legacy/`. Corresponden al análisis con los cuatro
defectos descritos en `scripts/_legacy/README.md`.

Equivalencias con el análisis vigente (`results/metrics/`):

| Antiguo | Vigente | Cambio principal |
|---|---|---|
| `secchi_multimission_metrics.csv` | `validation.json` | Bloqueo por campaña, no por estación |
| `benchmark_models.csv` | `benchmark.json` | Baseline clásico acotado: −2.74 → +0.10 |
| `validation_q1.csv` | `validation.json` | Se añaden modelos nulos |
| `uncertainty_metrics.json` | `uncertainty.json` | Se añade cobertura condicional |
| `transparency_trends.csv` | `temporal.json` | Registro in-situ completo + sensibilidad al año inicial |
