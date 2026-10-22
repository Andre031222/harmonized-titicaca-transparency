#!/usr/bin/env bash
# ============================================================================
# run_all.sh -- reproduce el analisis completo, en orden, desde la tabla de
#               match-ups hasta las figuras y el PDF del manuscrito.
#
#   bash pipeline/run_all.sh            # todo
#   bash pipeline/run_all.sh --fast     # salta p06 (seleccion de modelos, ~3 h)
#   bash pipeline/run_all.sh --no-tex   # no compila el manuscrito
#
# p01 (extraccion en Earth Engine) no se ejecuta aqui: necesita credenciales y
# el registro de campo de IMARPE, que no se redistribuye. Las demas etapas
# arrancan de data/processed/matchups.csv.
#
# El portatil se calienta con todos los nucleos: los trabajos van fijados a
# los nucleos de eficiencia (8-15) y con prioridad baja.
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"; else PY="python3"; fi
if command -v taskset >/dev/null && [ "$(nproc)" -ge 16 ]; then
  RUN=(taskset -c 8-15 nice -n 10 "$PY")
else
  RUN=(nice -n 10 "$PY")
fi
cd "$ROOT/pipeline"

FAST=0; NO_TEX=0
for a in "$@"; do
  case "$a" in
    --fast)   FAST=1 ;;
    --no-tex) NO_TEX=1 ;;
    *) echo "opcion desconocida: $a"; exit 2 ;;
  esac
done

step() { printf '\n\033[1m>>> %s\033[0m\n' "$1"; }

step "p02  tabla de match-ups, ubicaciones, campanas y control de calidad"
"${RUN[@]}" p02_build_dataset.py
step "p03  prueba de dos muestras entre sensores (HLS y via de cada agencia)"
"${RUN[@]}" p03_harmonization.py
step "p04  diseno de validacion: nulos jerarquicos y RF bajo cada diseno"
"${RUN[@]}" p04_validation.py
step "p05  algoritmos clasicos y lineal"
"${RUN[@]}" p05_baselines.py
if [ "$FAST" -eq 0 ]; then
  step "p06  seleccion de modelos (CV anidada + Optuna) -- lento"
  "${RUN[@]}" p06_model_selection.py
else
  step "p06  OMITIDO (--fast)"
fi
step "p07  intervalos conformales"
"${RUN[@]}" p07_uncertainty.py
step "p08  SHAP fuera de fold y recuperabilidad de otras variables"
"${RUN[@]}" p08_interpretability.py
step "p09  estadistica espacial de los errores"
"${RUN[@]}" p09_spatial.py
step "p10  tendencias del registro de campo"
"${RUN[@]}" p10_trends.py
step "p11  ventana temporal del match-up"
"${RUN[@]}" p11_windows.py
step "p12  pares del mismo dia en otros lagos (reevalua la cache)"
"${RUN[@]}" p12_multilake_hls.py --eval
step "p15  recuperacion con recalibracion entre sensores"
"${RUN[@]}" p15_recalibrated_retrieval.py
step "p16  mapas del lago, estacion seca 2024 (descarga de Earth Engine si faltan los compuestos)"
"${RUN[@]}" p16_lake_maps.py
step "p17  brillo solar (reevalua la cache)"
"${RUN[@]}" p17_glint.py --eval
step "p18  separabilidad en reflectancia TOA (reevalua la cache)"
"${RUN[@]}" p18_toa_separability.py --eval
step "p13  exporta las cifras del manuscrito"
mkdir -p "$ROOT/manuscript/latex"
"${RUN[@]}" p13_export_numbers.py
if [ -f "$ROOT/manuscript/latex/titicaca_secchi.tex" ]; then
  step "p14  verifica el manuscrito contra results/"
  "${RUN[@]}" p14_verify_manuscript.py
fi

step "figuras (R / ggplot2)"
(cd "$ROOT/figures/R" && for f in fig[0-9S]*.R; do Rscript "$f"; done)
"$PY" "$ROOT/figures/graphical_abstract/ga_3d.py"

if [ "$NO_TEX" -eq 0 ] && [ -d "$ROOT/manuscript/latex" ]; then
  step "manuscrito"
  (cd "$ROOT/manuscript/latex" && latexmk -pdf -interaction=nonstopmode -quiet titicaca_secchi.tex)
fi
