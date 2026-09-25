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
    # -- Table: retrieval under each validation design ------------------------
    ("validation_hierarchy.csv", {"case": "random_kfold"}, "R2", 0.596),
    ("validation_hierarchy.csv", {"case": "random_kfold"}, "RMSE", 1.81),
    ("validation_hierarchy.csv", {"case": "by_station"}, "R2", 0.596),
    ("validation_hierarchy.csv", {"case": "by_station"}, "RMSE", 1.81),
    ("validation_hierarchy.csv", {"case": "by_campaign_date"}, "R2", 0.574),
    ("validation_hierarchy.csv", {"case": "by_campaign_date"}, "RMSE", 1.86),
    ("validation_hierarchy.csv", {"case": "by_year"}, "R2", 0.495),
    ("validation_hierarchy.csv", {"case": "by_year"}, "RMSE", 2.02),
    ("validation_hierarchy.csv", {"case": "one_record_per_event"}, "R2", 0.571),
    ("validation_hierarchy.csv", {"case": "one_record_per_event"}, "RMSE", 1.77),
    ("validation_hierarchy.csv", {"case": "one_record_per_event"}, "n", 540),
    ("validation_hierarchy.csv", {"case": "LAGO MENOR"}, "R2", 0.247),
    ("validation_hierarchy.csv", {"case": "LAGO MENOR"}, "RMSE", 1.90),
    ("validation_hierarchy.csv", {"case": "LAGO MENOR"}, "n", 133),
    ("validation_hierarchy.csv", {"case": "BAHIA PUNO"}, "R2", 0.036),
    ("validation_hierarchy.csv", {"case": "BAHIA PUNO"}, "RMSE", 2.11),
    ("validation_hierarchy.csv", {"case": "BAHIA PUNO"}, "n", 88),
    ("validation_hierarchy.csv", {"case": "LAGO MAYOR"}, "R2", -0.047),
    ("validation_hierarchy.csv", {"case": "LAGO MAYOR"}, "RMSE", 2.52),
    ("validation_hierarchy.csv", {"case": "LAGO MAYOR"}, "n", 591),
    ("temporal_holdout.csv", {"cut_year": 2019}, "R2", 0.507),
    ("temporal_holdout.csv", {"cut_year": 2019}, "RMSE", 2.63),
    ("temporal_holdout.csv", {"cut_year": 2019}, "n", 226),
    # -- Table: Roy transform over Titicaca water ----------------------------
    ("harmonization_bands.csv", {"band_label": "Blue"}, "roy_intercept", 0.0183),
    ("harmonization_bands.csv", {"band_label": "Blue"}, "native_oli_median", 0.0072),
    ("harmonization_bands.csv", {"band_label": "Blue"}, "intercept_over_signal", 2.5),
    ("harmonization_bands.csv", {"band_label": "Blue"}, "harmonized_ls_median", 0.0247),
    ("harmonization_bands.csv", {"band_label": "Blue"}, "ls_over_s2", 1.74),
    ("harmonization_bands.csv", {"band_label": "Green"}, "roy_intercept", 0.0123),
    ("harmonization_bands.csv", {"band_label": "Green"}, "native_oli_median", 0.0089),
    ("harmonization_bands.csv", {"band_label": "Green"}, "intercept_over_signal", 1.4),
    ("harmonization_bands.csv", {"band_label": "Green"}, "harmonized_ls_median", 0.0206),
    ("harmonization_bands.csv", {"band_label": "Green"}, "ls_over_s2", 1.61),
    ("harmonization_bands.csv", {"band_label": "Red"}, "roy_intercept", 0.0123),
    ("harmonization_bands.csv", {"band_label": "Red"}, "native_oli_median", 0.0012),
    ("harmonization_bands.csv", {"band_label": "Red"}, "intercept_over_signal", 10.6),
    ("harmonization_bands.csv", {"band_label": "Red"}, "harmonized_ls_median", 0.0134),
    ("harmonization_bands.csv", {"band_label": "Red"}, "ls_over_s2", 2.66),
    ("harmonization_bands.csv", {"band_label": "NIR"}, "roy_intercept", 0.0448),
    ("harmonization_bands.csv", {"band_label": "NIR"}, "native_oli_median", -0.0006),
    ("harmonization_bands.csv", {"band_label": "NIR"}, "intercept_over_signal", 69.8),
    ("harmonization_bands.csv", {"band_label": "NIR"}, "harmonized_ls_median", 0.0443),
    ("harmonization_bands.csv", {"band_label": "NIR"}, "ls_over_s2", 15.42),
    ("harmonization_bands.csv", {"band_label": "SWIR1"}, "roy_intercept", 0.0306),
    ("harmonization_bands.csv", {"band_label": "SWIR1"}, "native_oli_median", 0.0013),
    ("harmonization_bands.csv", {"band_label": "SWIR1"}, "intercept_over_signal", 23.2),
    ("harmonization_bands.csv", {"band_label": "SWIR1"}, "harmonized_ls_median", 0.0317),
    ("harmonization_bands.csv", {"band_label": "SWIR1"}, "ls_over_s2", 13.34),
    ("harmonization_bands.csv", {"band_label": "SWIR2"}, "roy_intercept", 0.0116),
    ("harmonization_bands.csv", {"band_label": "SWIR2"}, "native_oli_median", 0.0014),
    ("harmonization_bands.csv", {"band_label": "SWIR2"}, "intercept_over_signal", 8.2),
    ("harmonization_bands.csv", {"band_label": "SWIR2"}, "harmonized_ls_median", 0.0129),
    ("harmonization_bands.csv", {"band_label": "SWIR2"}, "ls_over_s2", 5.70),
    # -- Table: water-specific recalibration ---------------------------------
    ("harmonization_local_coefficients.csv", {"band_label": "Blue"}, "intercept", 0.0084),
    ("harmonization_local_coefficients.csv", {"band_label": "Blue"}, "slope", 0.258),
    ("harmonization_local_coefficients.csv", {"band_label": "Blue"}, "pearson_r", 0.26),
    ("harmonization_local_coefficients.csv", {"band_label": "Green"}, "intercept", -0.0009),
    ("harmonization_local_coefficients.csv", {"band_label": "Green"}, "slope", 0.665),
    ("harmonization_local_coefficients.csv", {"band_label": "Green"}, "pearson_r", 0.71),
    ("harmonization_local_coefficients.csv", {"band_label": "Red"}, "intercept", -0.0005),
    ("harmonization_local_coefficients.csv", {"band_label": "Red"}, "slope", 0.427),
    ("harmonization_local_coefficients.csv", {"band_label": "Red"}, "pearson_r", 0.52),
    ("harmonization_local_coefficients.csv", {"band_label": "NIR"}, "intercept", -0.0250),
    ("harmonization_local_coefficients.csv", {"band_label": "NIR"}, "slope", 0.631),
    ("harmonization_local_coefficients.csv", {"band_label": "NIR"}, "pearson_r", 0.62),
    ("harmonization_local_coefficients.csv", {"band_label": "SWIR1"}, "intercept", -0.0223),
    ("harmonization_local_coefficients.csv", {"band_label": "SWIR1"}, "slope", 0.770),
    ("harmonization_local_coefficients.csv", {"band_label": "SWIR1"}, "pearson_r", 0.64),
    ("harmonization_local_coefficients.csv", {"band_label": "SWIR2"}, "intercept", -0.0057),
    ("harmonization_local_coefficients.csv", {"band_label": "SWIR2"}, "slope", 0.615),
    ("harmonization_local_coefficients.csv", {"band_label": "SWIR2"}, "pearson_r", 0.60),
    # -- Table: head-to-head benchmark ---------------------------------------
    ("benchmark_models.csv", {"family": "artefact"}, "R2", -2.458),
    ("benchmark_models.csv", {"family": "artefact"}, "RMSE", 5.29),
    ("benchmark_models.csv", {"family": "artefact"}, "MAE", 1.91),
    ("benchmark_models.csv", {"family": "artefact"}, "bias", -0.19),
    ("benchmark_models.csv", {"family": "classical"}, "R2", 0.097),
    ("benchmark_models.csv", {"family": "classical"}, "RMSE", 2.71),
    ("benchmark_models.csv", {"family": "classical"}, "MAE", 2.13),
    ("benchmark_models.csv", {"family": "classical"}, "bias", -0.46),
    ("benchmark_models.csv", {"family": "linear"}, "R2", 0.323),
    ("benchmark_models.csv", {"family": "linear"}, "RMSE", 2.34),
    ("benchmark_models.csv", {"family": "linear"}, "MAE", 1.80),
    ("benchmark_models.csv", {"family": "linear"}, "bias", -0.01),
    ("benchmark_models.csv", {"family": "rf"}, "R2", 0.574),
    ("benchmark_models.csv", {"family": "rf"}, "RMSE", 1.86),
    ("benchmark_models.csv", {"family": "rf"}, "MAE", 1.44),
    ("benchmark_models.csv", {"family": "rf"}, "bias", -0.02),
    # -- Table: model selection ----------------------------------------------
    ("model_selection.csv", {"label": "Ensemble (RF+ET+XGB+LGBM optimizados)"}, "R2", 0.583),
    ("model_selection.csv", {"label": "Ensemble (RF+ET+XGB+LGBM optimizados)"}, "RMSE", 1.84),
    ("model_selection.csv", {"label": "LGBM optimizado (anidado)"}, "R2", 0.581),
    ("model_selection.csv", {"label": "LGBM optimizado (anidado)"}, "RMSE", 1.84),
    ("model_selection.csv", {"label": "XGB optimizado (anidado)"}, "R2", 0.577),
    ("model_selection.csv", {"label": "XGB optimizado (anidado)"}, "RMSE", 1.85),
    ("model_selection.csv", {"label": "RF optimizado (anidado)"}, "R2", 0.576),
    ("model_selection.csv", {"label": "RF optimizado (anidado)"}, "RMSE", 1.85),
    ("model_selection.csv", {"label": "RF por defecto"}, "R2", 0.574),
    ("model_selection.csv", {"label": "RF por defecto"}, "RMSE", 1.86),
    ("model_selection.csv", {"label": "ET por defecto"}, "R2", 0.573),
    ("model_selection.csv", {"label": "ET por defecto"}, "RMSE", 1.86),
    ("model_selection.csv", {"label": "XGB optimizado + log(Zsd)"}, "R2", 0.561),
    ("model_selection.csv", {"label": "XGB optimizado + log(Zsd)"}, "RMSE", 1.89),
    ("model_selection.csv", {"label": "LGBM optimizado -- visible_only"}, "R2", 0.544),
    ("model_selection.csv", {"label": "LGBM optimizado -- visible_only"}, "RMSE", 1.92),
    ("model_selection.csv", {"label": "LGBM optimizado -- ratios_only"}, "R2", 0.488),
    ("model_selection.csv", {"label": "LGBM optimizado -- ratios_only"}, "RMSE", 2.04),
    # -- Table: match-up window sensitivity (p11; solo si se corrio) ---------
    # Los valores salen de macros, no tecleados, asi que aqui basta con que el
    # CSV exista y cuadre con lo que exporto p09.
    # -- Table: error by trophic zone ----------------------------------------
    ("error_by_zone.csv", {"zone": "BAHIA PUNO"}, "R2", 0.221),
    ("error_by_zone.csv", {"zone": "BAHIA PUNO"}, "RMSE", 1.90),
    ("error_by_zone.csv", {"zone": "BAHIA PUNO"}, "bias", 1.02),
    ("error_by_zone.csv", {"zone": "BAHIA PUNO"}, "n", 88),
    ("error_by_zone.csv", {"zone": "LAGO MENOR"}, "R2", 0.211),
    ("error_by_zone.csv", {"zone": "LAGO MENOR"}, "RMSE", 1.94),
    ("error_by_zone.csv", {"zone": "LAGO MENOR"}, "bias", 0.08),
    ("error_by_zone.csv", {"zone": "LAGO MENOR"}, "n", 133),
    ("error_by_zone.csv", {"zone": "LAGO MAYOR"}, "R2", 0.448),
    ("error_by_zone.csv", {"zone": "LAGO MAYOR"}, "RMSE", 1.83),
    ("error_by_zone.csv", {"zone": "LAGO MAYOR"}, "bias", -0.19),
    ("error_by_zone.csv", {"zone": "LAGO MAYOR"}, "n", 591),
    # -- Table: el diagnostico en otros lagos grandes (p13) -------------------
    ("multilake_summary.csv", {"lake": "Lake Tahoe"}, "median_band_ratio", 5.78),
    ("multilake_summary.csv", {"lake": "Lake Tahoe"}, "worst_band_ratio", 17.95),
    ("multilake_summary.csv", {"lake": "Lake Tahoe"}, "s2_green_median", 0.0087),
    ("multilake_summary.csv", {"lake": "Lake Titicaca"}, "median_band_ratio", 2.95),
    ("multilake_summary.csv", {"lake": "Lake Titicaca"}, "worst_band_ratio", 9.15),
    ("multilake_summary.csv", {"lake": "Lake Titicaca"}, "s2_green_median", 0.01449),
    ("multilake_summary.csv", {"lake": "Lake Baikal"}, "median_band_ratio", 3.89),
    ("multilake_summary.csv", {"lake": "Lake Baikal"}, "worst_band_ratio", 12.22),
    ("multilake_summary.csv", {"lake": "Lake Baikal"}, "s2_green_median", 0.01714),
    ("multilake_summary.csv", {"lake": "Lake Malawi"}, "median_band_ratio", 1.75),
    ("multilake_summary.csv", {"lake": "Lake Malawi"}, "worst_band_ratio", 4.53),
    ("multilake_summary.csv", {"lake": "Lake Malawi"}, "s2_green_median", 0.02618),
    ("multilake_summary.csv", {"lake": "Lake Erie"}, "median_band_ratio", 1.05),
    ("multilake_summary.csv", {"lake": "Lake Erie"}, "worst_band_ratio", 2.59),
    ("multilake_summary.csv", {"lake": "Lake Erie"}, "s2_green_median", 0.04066),
    ("multilake_summary.csv", {"lake": "Lake Okeechobee"}, "median_band_ratio", 1.31),
    ("multilake_summary.csv", {"lake": "Lake Okeechobee"}, "worst_band_ratio", 2.98),
    ("multilake_summary.csv", {"lake": "Lake Okeechobee"}, "s2_green_median", 0.04099),
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
