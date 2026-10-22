"""
p00_config.py -- configuracion compartida del pipeline.

Un solo lugar para rutas, bandas, features, semilla y DISENO DE VALIDACION.
Ningun script fija a mano su particion: todos piden `splits(d, design)`.

Decisiones que sostienen el articulo (ver p04):
  * Dato principal: el producto NASA HLS v2.0 (HLSS30 + HLSL30), que corrige
    ambos sensores con LaSRC, normaliza la BRDF y aplica el ajuste de banda.
    La via de reflectancia de superficie de cada sensor (Sen2Cor + LaSRC) con
    solo los coeficientes de banda de HLS queda como comparacion.
  * Unidad espacial: la UBICACION real (coordenadas agrupadas a 500 m), no el
    codigo de estacion, que IMARPE reasigno entre anios.
  * Unidad de bloqueo primaria: la CAMPANA (una por anio en los match-ups),
    validada dejando fuera una campana entera cada vez.
"""
from pathlib import Path

import numpy as np

# --- rutas -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PROC = DATA / "processed"
FIELD_RAW = DATA / "insitu/raw/imarpe_titicaca_wq.csv"   # no se redistribuye
MET = ROOT / "results/metrics"
TIDY = ROOT / "results/tidy"
FIGR = ROOT / "results/figures_r"
for _d in (MET, TIDY, FIGR):
    _d.mkdir(parents=True, exist_ok=True)

GEE_PROJECT = "black-display-445217-p3"

# --- reproducibilidad ------------------------------------------------------
SEED = 42
N_SPLITS = 5
N_BOOT = 2000

# --- bandas y features -----------------------------------------------------
# Seis bandas comunes, con el nombre de la posicion en OLI/MSI.
BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
BAND_LABELS = {"B2": "Blue", "B3": "Green", "B4": "Red",
               "B8": "NIR", "B11": "SWIR1", "B12": "SWIR2"}
RATIOS = ["NDWI", "NDTI", "B2_B3", "B4_B3", "B3_B2", "B2_B4"]
FEATURES = BANDS + RATIOS
FEATURES_VIS = ["B2", "B3", "B4", "NDTI", "B2_B3", "B4_B3", "B3_B2", "B2_B4"]
FEATURES_RATIO_ONLY = RATIOS
FEATURE_LABELS = {"B2": "Blue", "B3": "Green", "B4": "Red", "B8": "NIR",
                  "B11": "SWIR1", "B12": "SWIR2", "NDWI": "NDWI",
                  "NDTI": "NDTI", "B2_B3": "Blue/Green", "B4_B3": "Red/Green",
                  "B3_B2": "Green/Blue", "B2_B4": "Blue/Red"}

# Bandas de cada producto en las posiciones comunes. En HLS el NIR de
# Sentinel-2 es B8A, la banda equivalente al B5 de OLI.
HLS_S30 = {"B2": "B2", "B3": "B3", "B4": "B4", "B8": "B8A", "B11": "B11", "B12": "B12"}
HLS_L30 = {"B2": "B2", "B3": "B3", "B4": "B4", "B8": "B5", "B11": "B6", "B12": "B7"}
LS_C2 = {"B2": "SR_B2", "B3": "SR_B3", "B4": "SR_B4", "B8": "SR_B5",
         "B11": "SR_B6", "B12": "SR_B7"}

# Ajuste de banda de HLS v2.0 (guia de usuario, Tabla 5; Claverie et al. 2018):
# OLI_like = slope * MSI + intercept. Solo se usa en la via de comparacion
# (Sen2Cor + coeficientes). Las tablas no registran S2A/S2B: media de ambos,
# que difieren hasta 1.6 % en la pendiente de SWIR2.
_HLS_S2A = {"B2": (0.9778, -0.0040), "B3": (1.0053, -0.0009),
            "B4": (0.9765, 0.0009), "B8A": (0.9983, -0.0001),
            "B11": (0.9987, -0.0011), "B12": (1.0030, -0.0012)}
_HLS_S2B = {"B2": (0.9778, -0.0040), "B3": (1.0075, -0.0008),
            "B4": (0.9761, 0.0010), "B8A": (0.9966, 0.0000),
            "B11": (1.0000, -0.0003), "B12": (0.9867, 0.0004)}
HLS_MSI_TO_OLI = {b: ((_HLS_S2A[b][0] + _HLS_S2B[b][0]) / 2,
                      (_HLS_S2A[b][1] + _HLS_S2B[b][1]) / 2) for b in _HLS_S2A}
MSI_SOURCE = {"B2": "B2", "B3": "B3", "B4": "B4", "B8": "B8A",
              "B11": "B11", "B12": "B12"}


def bandpass_adjust(d, adjust=True):
    """Via de comparacion: Sentinel-2 Sen2Cor al espacio OLI con los
    coeficientes de HLS (NIR desde B8A). Landsat ya es Collection 2 nativo."""
    d = d.copy()
    m = d.sensor == "S2"
    for b in BANDS:
        src = MSI_SOURCE[b]
        slope, icpt = HLS_MSI_TO_OLI[src] if adjust else (1.0, 0.0)
        d.loc[m, b] = slope * d.loc[m, src] + icpt
    return d


# Piso de los denominadores de los ratios: sobre agua clara la reflectancia
# roja puede ser <= 0 (sobrecorreccion atmosferica).
RATIO_FLOOR = 1e-4


def add_indices(d):
    """Indices y ratios a partir de las seis bandas comunes."""
    e, f = 1e-9, RATIO_FLOOR
    d = d.copy()
    d["NDWI"] = (d.B3 - d.B8) / (d.B3 + d.B8 + e)
    d["NDTI"] = (d.B4 - d.B3) / (d.B4 + d.B3 + e)
    d["B2_B3"] = d.B2 / d.B3.clip(lower=f)
    d["B4_B3"] = d.B4 / d.B3.clip(lower=f)
    d["B3_B2"] = d.B3 / d.B2.clip(lower=f)
    d["B2_B4"] = d.B2 / d.B4.clip(lower=f)
    return d


# --- zonas -----------------------------------------------------------------
ZONES = ["BAHIA PUNO", "LAGO MENOR", "LAGO MAYOR"]
ZONE_LABELS = {"BAHIA PUNO": "Bahia de Puno", "LAGO MENOR": "Lago Menor",
               "LAGO MAYOR": "Lago Mayor"}

# --- unidades del registro ---------------------------------------------------
LOCATION_RADIUS_M = 500      # dos lecturas son la misma ubicacion si distan <= 500 m
N_SPATIAL_BLOCKS = 8         # bloques espaciales (k-medias sobre las ubicaciones)
UTM = "EPSG:32719"           # UTM 19S
SECCHI_CLIP_LO = 0.05

# Control de calidad de la reflectancia sobre agua: el agua no refleja en el
# SWIR, asi que SWIR1 > 0.02 delata nube o bruma residual no enmascarada, o
# tierra (una coordenada erronea); NDWI <= 0 delata un pixel que no es agua.
QC_SWIR1_MAX = 0.02


def qc_pass(d):
    ndwi = (d.B3 - d.B8) / (d.B3 + d.B8 + 1e-9)
    return (d.B11 <= QC_SWIR1_MAX) & (ndwi > 0)


def load_matchups(all_rows=False):
    """Tabla larga de match-ups (p02). Por defecto solo las filas que pasan el
    control de calidad."""
    import pandas as pd
    m = pd.read_csv(PROC / "matchups.csv", dtype={"campaign": str})
    return m if all_rows else m[m.qc_pass].reset_index(drop=True)

# Disenos de validacion, de menos a mas exigente. El primario es "campaign".
DESIGNS = ["random", "location", "sampling_date", "spatial_block", "campaign",
           "campaign_block"]
DESIGN_LABELS = {"random": "Random K-fold", "location": "Location-blocked",
                 "sampling_date": "Sampling-date-blocked",
                 "spatial_block": "Spatial-block",
                 "campaign": "Campaign-blocked (primary)",
                 "campaign_block": "Campaign x spatial block"}
PRIMARY = "campaign"


def to_utm(lat, lon):
    from pyproj import Transformer
    t = Transformer.from_crs("EPSG:4326", UTM, always_xy=True)
    x, y = t.transform(np.asarray(lon), np.asarray(lat))
    return np.asarray(x), np.asarray(y)


def splits(d, design=PRIMARY):
    """Particiones (train, test) en posiciones de d (indice 0..n-1).

    campaign       : deja fuera una campana entera (9 folds).
    campaign_block : el test es una campana dentro de un bloque espacial; el
                     entrenamiento excluye esa campana Y ese bloque (espacio-
                     tiempo). Solo se evalua cada fila una vez.
    """
    from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut
    n = len(d)
    idx = np.arange(n)
    if design == "random":
        return list(KFold(N_SPLITS, shuffle=True, random_state=SEED).split(idx))
    if design == "location":
        return list(GroupKFold(N_SPLITS).split(idx, groups=d.location.values))
    if design == "sampling_date":
        return list(GroupKFold(N_SPLITS).split(idx, groups=d.sampling_date.values))
    if design == "spatial_block":
        return list(LeaveOneGroupOut().split(idx, groups=d.spatial_block.values))
    if design == "campaign":
        return list(LeaveOneGroupOut().split(idx, groups=d.campaign.values))
    if design == "campaign_block":
        out = []
        c, b = d.campaign.values, d.spatial_block.values
        for cc in np.unique(c):
            for bb in np.unique(b):
                te = idx[(c == cc) & (b == bb)]
                if len(te):
                    out.append((idx[(c != cc) & (b != bb)], te))
        return out
    raise ValueError(design)


def make_rf(params=None, seed=SEED):
    from sklearn.ensemble import RandomForestRegressor
    p = {"n_estimators": 500, "max_depth": 12, "min_samples_leaf": 3}
    p.update(params or {})
    return RandomForestRegressor(random_state=seed, n_jobs=-1, **p)


def cv_oof(d, feats=FEATURES, design=PRIMARY, model_fn=make_rf, target="secchi",
           folds=None):
    """Predicciones fuera de fold. El escalado robusto se ajusta solo con el
    entrenamiento de cada fold."""
    from sklearn.preprocessing import RobustScaler
    d = d.reset_index(drop=True)
    X, y = d[feats].values, d[target].values
    oof = np.full(len(y), np.nan)
    for tr, te in (folds or splits(d, design)):
        sc = RobustScaler().fit(X[tr])
        m = model_fn()
        m.fit(sc.transform(X[tr]), y[tr])
        oof[te] = m.predict(sc.transform(X[te]))
    if target == "secchi":
        oof = np.clip(oof, SECCHI_CLIP_LO, None)
    return y, oof


def metrics(y_true, y_pred, decimals=3):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {"R2": round(float(r2_score(y_true, y_pred)), decimals),
            "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), decimals),
            "MAE": round(float(mean_absolute_error(y_true, y_pred)), decimals),
            "bias": round(float(np.mean(y_pred - y_true)), decimals),
            "n": int(len(y_true))}


def cluster_boot(y, p, groups, stat="R2", n_boot=N_BOOT, seed=SEED, p2=None):
    """IC del 95 % por bootstrap de conglomerados (se remuestrean campanas
    enteras). Con p2, el IC es de la diferencia stat(p) - stat(p2)."""
    from sklearn.metrics import mean_squared_error, r2_score
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    fn = {"R2": r2_score,
          "RMSE": lambda a, b: float(np.sqrt(mean_squared_error(a, b)))}[stat]
    ug = np.unique(groups)
    pos = {g: np.flatnonzero(groups == g) for g in ug}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        ix = np.concatenate([pos[g] for g in rng.choice(ug, len(ug))])
        v = fn(y[ix], p[ix])
        if p2 is not None:
            v -= fn(y[ix], np.asarray(p2)[ix])
        vals.append(v)
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return round(float(lo), 3), round(float(hi), 3)


def banner(text, char="=", width=78):
    print("\n" + char * width)
    print(text)
    print(char * width)


def write_json(obj, path):
    """JSON valido: NaN e infinitos pasan a null y los tipos de numpy a nativos."""
    import json
    import math

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, np.generic):
            o = o.item()
        if isinstance(o, float) and not math.isfinite(o):
            return None
        return o
    with open(path, "w") as f:
        json.dump(clean(obj), f, indent=2, allow_nan=False, default=str)
