#!/usr/bin/env bash
# ============================================================================
# run_all.sh -- reproduce el analisis completo, en orden, desde los match-ups
#               hasta el PDF del manuscrito.
#
#   bash pipeline/run_all.sh            # todo
#   bash pipeline/run_all.sh --fast     # salta p06 (seleccion de modelo, ~1 h)
#   bash pipeline/run_all.sh --no-tex   # no compila el manuscrito
#
# Requisitos: .venv con requirements.txt, R con los paquetes de figures/R,
# y una distribucion LaTeX si se quiere el PDF.
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
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

step "p01  dataset de analisis e inventario"
"$PY" p01_build_dataset.py

step "p02  ERROR 1 -- armonizacion Roy sobre agua"
"$PY" p02_fix1_harmonization.py

step "p04  ERROR 3 -- unidad de bloqueo y modelos nulos"
"$PY" p04_fix3_validation.py

step "p03  ERROR 2 -- baseline clasico corregido"
"$PY" p03_fix2_baselines.py     # importa de p04, por eso va despues

step "p05  ERROR 4 -- hold-out temporal, estacionalidad y tendencias"
"$PY" p05_fix4_temporal.py

if [ "$FAST" -eq 0 ]; then
  step "p06  seleccion de modelo (CV anidada + Optuna) -- lento, ~1 h"
  "$PY" p06_model_selection.py
else
  step "p06  OMITIDO (--fast)"
fi

step "p07  incertidumbre conformal"
"$PY" p07_uncertainty.py

step "p08  SHAP fuera de fold y retrievabilidad"
"$PY" p08_interpretability.py

step "p09  exportacion de numeros al manuscrito"
"$PY" p09_export_manuscript_numbers.py

step "p10  verificacion: el manuscrito coincide con los datos"
"$PY" p10_verify_manuscript.py

step "figuras (R / ggplot2)"
cd "$ROOT/figures/R"
for f in fig0*.R; do
  echo "  -> $f"
  Rscript "$f" 2>&1 | grep -E "guardado|Error" || true
done

if [ "$NO_TEX" -eq 0 ] && command -v pdflatex >/dev/null 2>&1; then
  step "manuscrito (LaTeX)"
  cd "$ROOT/manuscript/jglr"
  cp "$ROOT"/results/figures_r/*.png figures/ 2>/dev/null || true
  pdflatex -interaction=nonstopmode titicaca_transparency_jglr.tex >/dev/null
  bibtex titicaca_transparency_jglr >/dev/null 2>&1 || true
  pdflatex -interaction=nonstopmode titicaca_transparency_jglr.tex >/dev/null
  pdflatex -interaction=nonstopmode titicaca_transparency_jglr.tex >/dev/null
  echo "  -> $(pdfinfo titicaca_transparency_jglr.pdf | grep Pages)"
fi

printf '\n\033[1mListo.\033[0m\n'
echo "  metricas : results/metrics/"
echo "  tablas   : results/tidy/"
echo "  figuras  : results/figures_r/"
echo "  articulo : manuscript/jglr/titicaca_transparency_jglr.pdf"
echo
echo "  Las cifras del texto salen de numbers.tex (p09) y las de las tablas"
echo "  quedan verificadas contra results/tidy/ por p10."
