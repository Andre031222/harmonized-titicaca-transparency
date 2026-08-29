"""
p00_config.py -- configuracion compartida del pipeline corregido.

Todo el pipeline (p01..p09) importa de aqui. Un solo lugar donde viven las
rutas, las features, la semilla y el DISENO DE VALIDACION, para que no vuelva
a ocurrir lo del manuscrito anterior: numeros de distintos disenos mezclados
en la misma frase.

DECISION DE DISENO (ver p04): la unidad de bloqueo primaria es la FECHA DE
CAMPANA, no la estacion. En un programa de monitoreo por campanas todo el
lago se muestrea en pocos dias bajo las mismas condiciones atmosfericas y
limnologicas, asi que la dependencia vive entre campanas. Bloquear por
estacion no retiene nada (el nulo media-de-estacion sigue sacando R2=0.21).
"""
from pathlib import Path

import numpy as np

# --- rutas -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PROC = DATA / "processed"
MET = ROOT / "results/metrics"
TIDY = ROOT / "results/tidy"        # CSV largos y limpios que consume R
FIGR = ROOT / "results/figures_r"   # salidas de ggplot2
MODELS = ROOT / "results/models"
for _d in (MET, TIDY, FIGR, MODELS):
    _d.mkdir(parents=True, exist_ok=True)

# --- reproducibilidad ------------------------------------------------------
SEED = 42
N_SPLITS = 5

# --- bandas y features -----------------------------------------------------
# Seis bandas comunes a Sentinel-2 MSI y Landsat 8/9 OLI.
BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
BAND_LABELS = {"B2": "Blue", "B3": "Green", "B4": "Red",
               "B8": "NIR", "B11": "SWIR1", "B12": "SWIR2"}

RATIOS = ["NDWI", "NDTI", "B2_B3", "B4_B3", "B3_B2", "B2_B4"]
FEATURES = BANDS + RATIOS                       # las 12 del estudio
FEATURES_VIS = ["B2", "B3", "B4", "NDTI", "B2_B3", "B4_B3", "B3_B2", "B2_B4"]
FEATURES_RATIO_ONLY = RATIOS

FEATURE_LABELS = {"B2": "Blue", "B3": "Green", "B4": "Red", "B8": "NIR",
                  "B11": "SWIR1", "B12": "SWIR2", "NDWI": "NDWI",
                  "NDTI": "NDTI", "B2_B3": "Blue/Green", "B4_B3": "Red/Green",
                  "B3_B2": "Green/Blue", "B2_B4": "Blue/Red"}

# --- coeficientes Roy et al. (2016), OLI -> MSI: rho_MSI = a + b*rho_OLI ----
# Ajustados sobre superficies TERRESTRES. p02 demuestra que no son validos
# sobre agua oligotrofica oscura.
ROY = {"B2": (0.0183, 0.8850), "B3": (0.0123, 0.9317), "B4": (0.0123, 0.9372),
       "B8": (0.0448, 0.8339), "B11": (0.0306, 0.8639), "B12": (0.0116, 0.9165)}

# --- zonas trofica ---------------------------------------------------------
ZONES = ["BAHIA PUNO", "LAGO MENOR", "LAGO MAYOR"]
ZONE_LABELS = {"BAHIA PUNO": "Bahia de Puno", "LAGO MENOR": "Lago Menor",
               "LAGO MAYOR": "Lago Mayor"}

# --- diseno de validacion --------------------------------------------------
PRIMARY_BLOCK = "campaign_date"   # ver p04
SECCHI_CLIP_LO = 0.05             # las predicciones nunca son negativas


def add_indices(d):
    """Anade indices espectrales y ratios. Sensor-agnostico por construccion."""
    e = 1e-9
    d = d.copy()
    d["NDWI"] = (d.B3 - d.B8) / (d.B3 + d.B8 + e)
    d["NDTI"] = (d.B4 - d.B3) / (d.B4 + d.B3 + e)
    d["B2_B3"] = d.B2 / (d.B3 + e)
    d["B4_B3"] = d.B4 / (d.B3 + e)
    d["B3_B2"] = d.B3 / (d.B2 + e)
    d["B2_B4"] = d.B2 / (d.B4 + e)
    return d


def metrics(y_true, y_pred, decimals=3):
    """R2, RMSE, MAE, bias y n. Un solo sitio para no discrepar entre scripts."""
    from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                                 r2_score)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {"R2": round(float(r2_score(y_true, y_pred)), decimals),
            "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), decimals),
            "MAE": round(float(mean_absolute_error(y_true, y_pred)), decimals),
            "bias": round(float(np.mean(y_pred - y_true)), decimals),
            "n": int(len(y_true))}


def banner(text, char="=", width=78):
    print("\n" + char * width)
    print(text)
    print(char * width)
