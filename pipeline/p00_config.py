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

# --- armonizacion entre sensores --------------------------------------------
# Las tablas de match-ups guardan Landsat transformado con los coeficientes OLS
# OLI -> ETM+ de Roy et al. (2016, Tabla 2): llevan Landsat 8 al espacio de
# Landsat 7, NO al de Sentinel-2 (el script oficial de Earth Engine los llama
# `oliToEtmOls`). Aqui solo sirven para deshacer esa transformacion, que es
# lineal y se invierte exacta (los valores se guardaron con 5 decimales).
ROY_OLI_TO_ETM = {"B2": (0.0183, 0.8850), "B3": (0.0123, 0.9317),
                  "B4": (0.0123, 0.9372), "B8": (0.0448, 0.8339),
                  "B11": (0.0306, 0.8639), "B12": (0.0116, 0.9165)}

# Ajuste de banda de HLS v2.0 (Masek et al. 2021, Tabla 5; Claverie et al.
# 2018): lleva MSI al espacio de OLI, OLI_like = slope * MSI + intercept. Es la
# direccion oficial de HLS. Las tablas no registran si cada escena es S2A o S2B,
# asi que se usa la media de ambos (difieren en <0.5 % de pendiente). El NIR de
# OLI (B5) se empareja con B8A de MSI, no con B8: es la banda equivalente.
_HLS_S2A = {"B2": (0.9778, -0.0040), "B3": (1.0053, -0.0009),
            "B4": (0.9765, 0.0009), "B8A": (0.9983, -0.0001),
            "B11": (0.9987, -0.0011), "B12": (1.0030, -0.0012)}
_HLS_S2B = {"B2": (0.9778, -0.0040), "B3": (1.0075, -0.0008),
            "B4": (0.9761, 0.0010), "B8A": (0.9966, 0.0000),
            "B11": (1.0000, -0.0003), "B12": (0.9867, 0.0004)}
HLS_MSI_TO_OLI = {b: ((_HLS_S2A[b][0] + _HLS_S2B[b][0]) / 2,
                      (_HLS_S2A[b][1] + _HLS_S2B[b][1]) / 2) for b in _HLS_S2A}
# banda de MSI que ocupa cada posicion comun (el NIR sale de B8A)
MSI_SOURCE = {"B2": "B2", "B3": "B3", "B4": "B4", "B8": "B8A",
              "B11": "B11", "B12": "B12"}


def native_oli(d):
    """Reflectancia nativa de Landsat 8/9 desde las tablas de match-ups, que la
    guardan transformada OLI -> ETM+. Solo toca filas sensor == 'LS'."""
    d = d.copy()
    m = d.sensor == "LS"
    for b in BANDS:
        a, slope = ROY_OLI_TO_ETM[b]
        d.loc[m, b] = (d.loc[m, b] - a) / slope
    return d


def harmonize(d, adjust_msi=True, ls_native=False):
    """Deja ambos sensores en el espacio de OLI: Landsat nativo y Sentinel-2
    con el ajuste de banda de HLS (NIR desde B8A). Con adjust_msi=False deja
    Sentinel-2 sin ajustar, para medir que aporta el ajuste. ls_native=True
    cuando Landsat ya viene nativo (extracciones de p11 y p13)."""
    d = d.copy() if ls_native else native_oli(d)
    m = d.sensor == "S2"
    for b in BANDS:
        src = MSI_SOURCE[b]
        if adjust_msi:
            slope, icpt = HLS_MSI_TO_OLI[src]
            d.loc[m, b] = slope * d.loc[m, src] + icpt
        else:
            d.loc[m, b] = d.loc[m, src]
    return d

# --- zonas trofica ---------------------------------------------------------
ZONES = ["BAHIA PUNO", "LAGO MENOR", "LAGO MAYOR"]
ZONE_LABELS = {"BAHIA PUNO": "Bahia de Puno", "LAGO MENOR": "Lago Menor",
               "LAGO MAYOR": "Lago Mayor"}

# --- diseno de validacion --------------------------------------------------
PRIMARY_BLOCK = "campaign_date"   # ver p04
SECCHI_CLIP_LO = 0.05             # las predicciones nunca son negativas


def paired_events(d):
    """Eventos de campo vistos por ambos sensores (misma estacion y fecha de
    muestreo): tabla ancha con columnas (banda, sensor)."""
    return (d.pivot_table(index=["station", "campaign_date"], columns="sensor",
                          values=BANDS, aggfunc="median").dropna())


def fit_local_recal(d):
    """Recalibracion sobre agua, banda a banda, de Sentinel-2 (ya con HLS) a
    Landsat nativo: LS = a + b * S2, ajustada en los eventos pareados de d.
    Para validar hay que llamarla solo con el fold de entrenamiento."""
    piv = paired_events(d)
    coefs = {}
    for b in BANDS:
        x, y = piv[(b, "S2")].values, piv[(b, "LS")].values
        slope, icpt = np.polyfit(x, y, 1)
        coefs[b] = (float(icpt), float(slope))
    return coefs, len(piv)


def apply_local_recal(d, coefs):
    """Aplica la recalibracion a las filas de Sentinel-2 y recalcula features."""
    out = d.copy()
    m = out.sensor == "S2"
    for b, (icpt, slope) in coefs.items():
        out.loc[m, b] = icpt + slope * out.loc[m, b]
    return add_indices(out)


# Piso de los denominadores de los ratios: sobre agua clara la reflectancia de
# superficie de Landsat en el rojo es <= 0 en ~14 % de los match-ups
# (sobrecorreccion atmosferica), y un ratio con denominador nulo o negativo no
# significa nada. 1e-4 esta por debajo del ruido radiometrico de ambos sensores.
RATIO_FLOOR = 1e-4


def add_indices(d):
    """Anade indices espectrales y ratios. Sensor-agnostico por construccion."""
    e = 1e-9
    f = RATIO_FLOOR
    d = d.copy()
    d["NDWI"] = (d.B3 - d.B8) / (d.B3 + d.B8 + e)
    d["NDTI"] = (d.B4 - d.B3) / (d.B4 + d.B3 + e)
    d["B2_B3"] = d.B2 / d.B3.clip(lower=f)
    d["B4_B3"] = d.B4 / d.B3.clip(lower=f)
    d["B3_B2"] = d.B3 / d.B2.clip(lower=f)
    d["B2_B4"] = d.B2 / d.B4.clip(lower=f)
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
