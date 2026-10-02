"""
==============================================================================
p10_verify_manuscript.py -- comprueba que el manuscrito dice lo que dicen los
datos.

El texto corrido no teclea cifras: usa las macros de numbers.tex, que genera
p09. Las TABLAS si estan tecleadas a mano, porque su maquetacion no se presta
a generarlas. Ese es justo el sitio donde una cifra se queda obsoleta sin que
nadie lo note: al reordenar el analisis en agosto de 2026 el MAE del modelo
lineal quedo en 1.78 cuando el CSV decia 1.797.

Este script cierra ese hueco con dos comprobaciones:

  (1) Toda macro \\num* usada en el .tex esta definida en numbers.tex. Una
      macro no definida se expande a nada, asi que la cifra simplemente
      desaparece del PDF sin error visible. Ya paso una vez
      (\\numRoyRatioSwirone por \\numRoyRatioSwirOne).

  (2) Toda celda numerica de las tablas coincide con su origen en
      results/tidy/, a la precision con la que esta impresa. El mapeo es
      explicito (MANIFEST): cada celda declara de que fichero, que fila y que
      columna sale.

Devuelve codigo 1 si algo no cuadra, para que run_all.sh se detenga.

    python pipeline/p10_verify_manuscript.py
==============================================================================
"""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TIDY = ROOT / "results" / "tidy"
TEX = ROOT / "manuscript" / "jglr" / "titicaca_transparency_jglr.tex"
NUMBERS = ROOT / "manuscript" / "jglr" / "numbers.tex"


# --- (2) de donde sale cada celda tecleada -----------------------------------
# (csv, {columna_clave: valor}, columna, valor_impreso_en_el_tex)
MANIFEST = [
    # -- Table: retrieval under each validation design: macros (p09) --------
    # -- Tables: HLS adjustment and local recalibration (p02) --------------
    ("harmonization_bands.csv", {"band_label": "Blue"}, "hls_slope", 0.9778),
    ("harmonization_bands.csv", {"band_label": "Blue"}, "hls_intercept", -0.004),
    ("harmonization_bands.csv", {"band_label": "Blue"}, "native_ls_median", 0.0072),
    ("harmonization_bands.csv", {"band_label": "Blue"}, "s2_raw_median", 0.0142),
    ("harmonization_bands.csv", {"band_label": "Blue"}, "s2_hls_median", 0.0099),
    ("harmonization_bands.csv", {"band_label": "Blue"}, "hls_shift_pct", -30.4),
    ("harmonization_bands.csv", {"band_label": "Blue"}, "paired_ls_over_s2_hls", 0.76),
    ("harmonization_bands.csv", {"band_label": "Green"}, "hls_slope", 1.0064),
    ("harmonization_bands.csv", {"band_label": "Green"}, "hls_intercept", -0.00085),
    ("harmonization_bands.csv", {"band_label": "Green"}, "native_ls_median", 0.0089),
    ("harmonization_bands.csv", {"band_label": "Green"}, "s2_raw_median", 0.0128),
    ("harmonization_bands.csv", {"band_label": "Green"}, "s2_hls_median", 0.012),
    ("harmonization_bands.csv", {"band_label": "Green"}, "hls_shift_pct", -6.0),
    ("harmonization_bands.csv", {"band_label": "Green"}, "paired_ls_over_s2_hls", 0.75),
    ("harmonization_bands.csv", {"band_label": "Red"}, "hls_slope", 0.9763),
    ("harmonization_bands.csv", {"band_label": "Red"}, "hls_intercept", 0.00095),
    ("harmonization_bands.csv", {"band_label": "Red"}, "native_ls_median", 0.0012),
    ("harmonization_bands.csv", {"band_label": "Red"}, "s2_raw_median", 0.005),
    ("harmonization_bands.csv", {"band_label": "Red"}, "s2_hls_median", 0.0059),
    ("harmonization_bands.csv", {"band_label": "Red"}, "hls_shift_pct", 16.5),
    ("harmonization_bands.csv", {"band_label": "Red"}, "paired_ls_over_s2_hls", 0.2),
    ("harmonization_bands.csv", {"band_label": "NIR"}, "hls_slope", 0.9974),
    ("harmonization_bands.csv", {"band_label": "NIR"}, "hls_intercept", -5e-05),
    ("harmonization_bands.csv", {"band_label": "NIR"}, "native_ls_median", -0.0006),
    ("harmonization_bands.csv", {"band_label": "NIR"}, "s2_raw_median", 0.0026),
    ("harmonization_bands.csv", {"band_label": "NIR"}, "s2_hls_median", 0.0025),
    ("harmonization_bands.csv", {"band_label": "NIR"}, "hls_shift_pct", -2.2),
    ("harmonization_bands.csv", {"band_label": "NIR"}, "paired_ls_over_s2_hls", -0.26),
    ("harmonization_bands.csv", {"band_label": "SWIR1"}, "hls_slope", 0.9993),
    ("harmonization_bands.csv", {"band_label": "SWIR1"}, "hls_intercept", -0.0007),
    ("harmonization_bands.csv", {"band_label": "SWIR1"}, "native_ls_median", 0.0013),
    ("harmonization_bands.csv", {"band_label": "SWIR1"}, "s2_raw_median", 0.0024),
    ("harmonization_bands.csv", {"band_label": "SWIR1"}, "s2_hls_median", 0.0017),
    ("harmonization_bands.csv", {"band_label": "SWIR1"}, "hls_shift_pct", -29.5),
    ("harmonization_bands.csv", {"band_label": "SWIR1"}, "paired_ls_over_s2_hls", 0.99),
    ("harmonization_bands.csv", {"band_label": "SWIR2"}, "hls_slope", 0.9949),
    ("harmonization_bands.csv", {"band_label": "SWIR2"}, "hls_intercept", -0.0004),
    ("harmonization_bands.csv", {"band_label": "SWIR2"}, "native_ls_median", 0.0014),
    ("harmonization_bands.csv", {"band_label": "SWIR2"}, "s2_raw_median", 0.0023),
    ("harmonization_bands.csv", {"band_label": "SWIR2"}, "s2_hls_median", 0.0018),
    ("harmonization_bands.csv", {"band_label": "SWIR2"}, "hls_shift_pct", -18.2),
    ("harmonization_bands.csv", {"band_label": "SWIR2"}, "paired_ls_over_s2_hls", 0.93),
    ("harmonization_local_coefficients.csv", {"band_label": "Blue"}, "intercept", 0.0053),
    ("harmonization_local_coefficients.csv", {"band_label": "Blue"}, "slope", 0.307),
    ("harmonization_local_coefficients.csv", {"band_label": "Blue"}, "pearson_r", 0.26),
    ("harmonization_local_coefficients.csv", {"band_label": "Green"}, "intercept", 0.0008),
    ("harmonization_local_coefficients.csv", {"band_label": "Green"}, "slope", 0.81),
    ("harmonization_local_coefficients.csv", {"band_label": "Green"}, "pearson_r", 0.71),
    ("harmonization_local_coefficients.csv", {"band_label": "Red"}, "intercept", -0.0017),
    ("harmonization_local_coefficients.csv", {"band_label": "Red"}, "slope", 0.703),
    ("harmonization_local_coefficients.csv", {"band_label": "Red"}, "pearson_r", 0.52),
    ("harmonization_local_coefficients.csv", {"band_label": "NIR"}, "intercept", -0.0015),
    ("harmonization_local_coefficients.csv", {"band_label": "NIR"}, "slope", 0.681),
    ("harmonization_local_coefficients.csv", {"band_label": "NIR"}, "pearson_r", 0.62),
    ("harmonization_local_coefficients.csv", {"band_label": "SWIR1"}, "intercept", 0.0018),
    ("harmonization_local_coefficients.csv", {"band_label": "SWIR1"}, "slope", 0.616),
    ("harmonization_local_coefficients.csv", {"band_label": "SWIR1"}, "pearson_r", 0.64),
    ("harmonization_local_coefficients.csv", {"band_label": "SWIR2"}, "intercept", 0.0014),
    ("harmonization_local_coefficients.csv", {"band_label": "SWIR2"}, "slope", 0.642),
    ("harmonization_local_coefficients.csv", {"band_label": "SWIR2"}, "pearson_r", 0.6),
    # -- Table: head-to-head benchmark ---------------------------------------
    ("benchmark_models.csv", {"family": "classical"}, "R2", -0.012),
    ("benchmark_models.csv", {"family": "classical"}, "RMSE", 2.86),
    ("benchmark_models.csv", {"family": "classical"}, "MAE", 2.29),
    ("benchmark_models.csv", {"family": "classical"}, "bias", -0.51),
    ("benchmark_models.csv", {"family": "kloiber"}, "R2", 0.012),
    ("benchmark_models.csv", {"family": "kloiber"}, "RMSE", 2.83),
    ("benchmark_models.csv", {"family": "kloiber"}, "MAE", 2.23),
    ("benchmark_models.csv", {"family": "kloiber"}, "bias", -0.49),
    ("benchmark_models.csv", {"family": "linear"}, "R2", 0.466),
    ("benchmark_models.csv", {"family": "linear"}, "RMSE", 2.08),
    ("benchmark_models.csv", {"family": "linear"}, "MAE", 1.62),
    ("benchmark_models.csv", {"family": "linear"}, "bias", -0.02),
    ("benchmark_models.csv", {"family": "rf"}, "R2", 0.548),
    ("benchmark_models.csv", {"family": "rf"}, "RMSE", 1.91),
    ("benchmark_models.csv", {"family": "rf"}, "MAE", 1.49),
    ("benchmark_models.csv", {"family": "rf"}, "bias", 0.03),
    # -- Table: match-up window sensitivity (p11; solo si se corrio) ---------
    # Los valores salen de macros, no tecleados, asi que aqui basta con que el
    # CSV exista y cuadre con lo que exporto p09.
    # -- Table: model selection (p06) ----------------------------------------
    ("model_selection.csv", {"label": "Ensemble (RF+ET+XGB+LGBM optimizados)"}, "R2", 0.564),
    ("model_selection.csv", {"label": "Ensemble (RF+ET+XGB+LGBM optimizados)"}, "RMSE", 1.88),
    ("model_selection.csv", {"label": "XGB optimizado (anidado)"}, "R2", 0.562),
    ("model_selection.csv", {"label": "XGB optimizado (anidado)"}, "RMSE", 1.88),
    ("model_selection.csv", {"label": "LGBM optimizado (anidado)"}, "R2", 0.555),
    ("model_selection.csv", {"label": "LGBM optimizado (anidado)"}, "RMSE", 1.90),
    ("model_selection.csv", {"label": "ET por defecto"}, "R2", 0.555),
    ("model_selection.csv", {"label": "ET por defecto"}, "RMSE", 1.90),
    ("model_selection.csv", {"label": "RF optimizado (anidado)"}, "R2", 0.551),
    ("model_selection.csv", {"label": "RF optimizado (anidado)"}, "RMSE", 1.91),
    ("model_selection.csv", {"label": "XGB optimizado + log(Zsd)"}, "R2", 0.551),
    ("model_selection.csv", {"label": "XGB optimizado + log(Zsd)"}, "RMSE", 1.91),
    ("model_selection.csv", {"label": "RF por defecto"}, "R2", 0.548),
    ("model_selection.csv", {"label": "RF por defecto"}, "RMSE", 1.91),
    ("model_selection.csv", {"label": "ET optimizado (anidado)"}, "R2", 0.547),
    ("model_selection.csv", {"label": "ET optimizado (anidado)"}, "RMSE", 1.92),
    ("model_selection.csv", {"label": "RF optimizado + log(Zsd)"}, "R2", 0.544),
    ("model_selection.csv", {"label": "RF optimizado + log(Zsd)"}, "RMSE", 1.92),
    ("model_selection.csv", {"label": "XGB optimizado -- visible_only"}, "R2", 0.528),
    ("model_selection.csv", {"label": "XGB optimizado -- visible_only"}, "RMSE", 1.96),
    ("model_selection.csv", {"label": "XGB por defecto"}, "R2", 0.520),
    ("model_selection.csv", {"label": "XGB por defecto"}, "RMSE", 1.97),
    ("model_selection.csv", {"label": "LGBM por defecto"}, "R2", 0.499),
    ("model_selection.csv", {"label": "LGBM por defecto"}, "RMSE", 2.02),
    ("model_selection.csv", {"label": "XGB optimizado -- ratios_only"}, "R2", 0.381),
    ("model_selection.csv", {"label": "XGB optimizado -- ratios_only"}, "RMSE", 2.24),
    # -- Table: error by trophic zone ----------------------------------------
    ("error_by_zone.csv", {"zone": "BAHIA PUNO"}, "R2", 0.243),
    ("error_by_zone.csv", {"zone": "BAHIA PUNO"}, "RMSE", 1.87),
    ("error_by_zone.csv", {"zone": "BAHIA PUNO"}, "bias", 1.01),
    ("error_by_zone.csv", {"zone": "BAHIA PUNO"}, "n", 88),
    ("error_by_zone.csv", {"zone": "LAGO MENOR"}, "R2", 0.099),
    ("error_by_zone.csv", {"zone": "LAGO MENOR"}, "RMSE", 2.08),
    ("error_by_zone.csv", {"zone": "LAGO MENOR"}, "bias", 0.19),
    ("error_by_zone.csv", {"zone": "LAGO MENOR"}, "n", 133),
    ("error_by_zone.csv", {"zone": "LAGO MAYOR"}, "R2", 0.418),
    ("error_by_zone.csv", {"zone": "LAGO MAYOR"}, "RMSE", 1.88),
    ("error_by_zone.csv", {"zone": "LAGO MAYOR"}, "bias", -0.15),
    ("error_by_zone.csv", {"zone": "LAGO MAYOR"}, "n", 591),
    # -- Table: el diagnostico en otros lagos grandes (p13) -------------------
    ("multilake_summary.csv", {"lake": "Lake Tahoe"}, "s2_green_median", 0.0087),
    ("multilake_summary.csv", {"lake": "Lake Tahoe"}, "blue_ratio", 1.13),
    ("multilake_summary.csv", {"lake": "Lake Tahoe"}, "green_ratio", 0.91),
    ("multilake_summary.csv", {"lake": "Lake Tahoe"}, "red_ratio", -0.03),
    ("multilake_summary.csv", {"lake": "Lake Tahoe"}, "no_signal_bands", 2),
    ("multilake_summary.csv", {"lake": "Lake Titicaca"}, "s2_green_median", 0.01449),
    ("multilake_summary.csv", {"lake": "Lake Titicaca"}, "blue_ratio", 0.51),
    ("multilake_summary.csv", {"lake": "Lake Titicaca"}, "green_ratio", 0.56),
    ("multilake_summary.csv", {"lake": "Lake Titicaca"}, "red_ratio", 0.1),
    ("multilake_summary.csv", {"lake": "Lake Titicaca"}, "no_signal_bands", 1),
    ("multilake_summary.csv", {"lake": "Lake Baikal"}, "s2_green_median", 0.01714),
    ("multilake_summary.csv", {"lake": "Lake Baikal"}, "blue_ratio", 0.58),
    ("multilake_summary.csv", {"lake": "Lake Baikal"}, "green_ratio", 0.69),
    ("multilake_summary.csv", {"lake": "Lake Baikal"}, "red_ratio", 0.14),
    ("multilake_summary.csv", {"lake": "Lake Baikal"}, "no_signal_bands", 1),
    ("multilake_summary.csv", {"lake": "Lake Malawi"}, "s2_green_median", 0.02618),
    ("multilake_summary.csv", {"lake": "Lake Malawi"}, "blue_ratio", 0.31),
    ("multilake_summary.csv", {"lake": "Lake Malawi"}, "green_ratio", 0.44),
    ("multilake_summary.csv", {"lake": "Lake Malawi"}, "red_ratio", 0.01),
    ("multilake_summary.csv", {"lake": "Lake Malawi"}, "no_signal_bands", 1),
    ("multilake_summary.csv", {"lake": "Lake Erie"}, "s2_green_median", 0.04066),
    ("multilake_summary.csv", {"lake": "Lake Erie"}, "blue_ratio", 0.29),
    ("multilake_summary.csv", {"lake": "Lake Erie"}, "green_ratio", 0.43),
    ("multilake_summary.csv", {"lake": "Lake Erie"}, "red_ratio", 0.12),
    ("multilake_summary.csv", {"lake": "Lake Erie"}, "no_signal_bands", 1),
    ("multilake_summary.csv", {"lake": "Lake Okeechobee"}, "s2_green_median", 0.04099),
    ("multilake_summary.csv", {"lake": "Lake Okeechobee"}, "blue_ratio", 0.29),
    ("multilake_summary.csv", {"lake": "Lake Okeechobee"}, "green_ratio", 0.63),
    ("multilake_summary.csv", {"lake": "Lake Okeechobee"}, "red_ratio", 0.62),
    ("multilake_summary.csv", {"lake": "Lake Okeechobee"}, "no_signal_bands", 0),
]


def check_macros(tex, numbers):
    """Toda macro \\num* usada tiene que estar definida."""
    used = set(re.findall(r"\\(num[A-Za-z]+)", tex))
    defined = set(re.findall(r"\\newcommand\{\\(num[A-Za-z]+)\}", numbers))
    missing = sorted(used - defined)
    return used, defined, missing


def check_tables():
    """Cada celda tecleada tiene que coincidir con su origen en results/tidy/."""
    cache, bad = {}, []
    for csv, sel, col, printed in MANIFEST:
        if csv not in cache:
            cache[csv] = pd.read_csv(TIDY / csv)
        df = cache[csv]
        for k, v in sel.items():
            df_sel = df[df[k].astype(str) == str(v)]
        if len(df_sel) != 1:
            bad.append((csv, sel, col, printed, f"{len(df_sel)} filas coinciden"))
            continue
        actual = float(df_sel[col].iloc[0])
        # tolerancia = media unidad del ultimo decimal impreso
        s = f"{printed}"
        dec = len(s.split(".")[1]) if "." in s else 0
        if abs(actual - printed) > 0.5 * 10 ** (-dec) + 1e-12:
            bad.append((csv, sel, col, printed, f"CSV dice {actual}"))
    return len(MANIFEST), bad


def main():
    print("=" * 78)
    print("p10 -- VERIFICACION DEL MANUSCRITO")
    print("=" * 78)
    if not TEX.exists():
        print("\n  El manuscrito no esta en este arbol (manuscript/ no forma "
              "parte\n  del repositorio publico). No hay nada que verificar.")
        print("=" * 78)
        return 0
    tex = TEX.read_text(encoding="utf-8")
    numbers = NUMBERS.read_text(encoding="utf-8")

    used, defined, missing = check_macros(tex, numbers)
    print(f"\n(1) Macros: {len(used)} usadas, {len(defined)} definidas.")
    if missing:
        print("    NO DEFINIDAS (se expanden a nada y la cifra desaparece):")
        for m in missing:
            print(f"      \\{m}")
    else:
        print("    Todas las macros usadas estan definidas.")

    n_cells, bad = check_tables()
    print(f"\n(2) Celdas de tabla verificadas contra results/tidy/: {n_cells}")
    if bad:
        print("    DISCREPANCIAS:")
        for csv, sel, col, printed, why in bad:
            print(f"      {csv} {sel} {col}: el tex dice {printed}, {why}")
    else:
        print("    Todas coinciden con su origen a la precision impresa.")

    ok = not missing and not bad
    print("\n" + "=" * 78)
    print("OK: el manuscrito coincide con los datos." if ok
          else "FALLO: corregir lo anterior antes de enviar.")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
