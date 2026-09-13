"""
p11_window_sensitivity.py -- sensibilidad de la recuperacion a la ventana del
match-up (+-1, +-3, +-5, +-10 dias).

A 3810 m el viento, la lluvia y las descargas fluviales pueden cambiar la
transparencia en dias, asi que +-10 dias podria ser demasiado ancho. Aqui se
vuelve a extraer con cada ventana y se reevalua bajo el diseno primario.

Requiere Earth Engine:  earthengine authenticate

    python pipeline/p11_window_sensitivity.py           # extrae y evalua
    python pipeline/p11_window_sensitivity.py --eval    # solo reevalua la cache
"""
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from p00_config import (BANDS, DATA, FEATURES, MET, N_SPLITS, PROC, ROY, TIDY,
                        add_indices, banner, metrics)
from p04_fix3_validation import cv_oof

WINDOWS = [1, 3, 5, 10]
BUFFER = 30
CLDPRB = 30
CACHE = PROC / "window_sensitivity_raw.csv"

S2_START, LS_START = 2016, 2013
ROY_SR = {"B2": "SR_B2", "B3": "SR_B3", "B4": "SR_B4",
          "B8": "SR_B5", "B11": "SR_B6", "B12": "SR_B7"}
MONTH_MAP = {"ENE": 1, "FEB": 2, "MAR": 3, "ABR": 4, "MAY": 5, "JUN": 6,
             "JUL": 7, "AGO": 8, "SET": 9, "SEP": 9, "OCT": 10, "NOV": 11,
             "DIC": 12}


def load_field():
    j = pd.read_csv(DATA / "external_validation/jairo_titicaca_wq.csv")
    j.columns = [c.replace("\xf1", "n").replace("�", "n") for c in j.columns]
    j["mon"] = j["Mes"].astype(str).str.upper().str[:3].map(MONTH_MAP)
    j["date"] = pd.to_datetime(dict(year=j["Ano"], month=j["mon"], day=j["Dia"]),
                               errors="coerce")
    j = j.rename(columns={"Latitud": "lat", "Longitud": "lon",
                          "Transparencia_m": "secchi", "Zona": "zona",
                          "Estacion": "station"})
    j = j.dropna(subset=["date", "lat", "lon", "secchi"])
    # el registro de IMARPE repite E59 del 2017-07-16 con la misma lectura
    j = j.drop_duplicates(subset=["station", "date"])
    return j[["station", "zona", "date", "lat", "lon", "secchi"]].reset_index(drop=True)


def _point(ee, lat, lon, date):
    return ee.Geometry.Point([lon, lat]), ee.Date(date.strftime("%Y-%m-%d"))


def _composite(ee, col, bands, mask_fn):
    """Mediana de la coleccion; imagen enmascarada si la ventana esta vacia.

    Sin este respaldo, median() sobre una coleccion vacia devuelve una imagen
    sin bandas y select() aborta la llamada entera.
    """
    empty = ee.Image.constant([0] * len(bands)).rename(bands).updateMask(ee.Image(0))
    return ee.Image(ee.Algorithms.If(col.size().gt(0),
                                     col.map(mask_fn).median().select(bands),
                                     empty))


def s2_windows(ee, lat, lon, date):
    pt, d0 = _point(ee, lat, lon, date)
    base = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(pt)

    def mask(img):
        scl = img.select("SCL")
        good = (scl.neq(1).And(scl.neq(3)).And(scl.neq(8)).And(scl.neq(9))
                .And(scl.neq(10)).And(scl.neq(11)))
        return img.updateMask(good.And(img.select("MSK_CLDPRB").lt(CLDPRB)))

    # El diccionario de reduceRegion se devuelve entero: Dictionary.set aborta
    # si el valor es nulo, y una banda enmascarada lo es.
    out = {}
    for w in WINDOWS:
        col = base.filterDate(d0.advance(-w, "day"), d0.advance(w, "day"))
        out[f"w{w}_n"] = col.size()
        out[f"w{w}"] = (_composite(ee, col, BANDS, mask).divide(10000)
                        .reduceRegion(ee.Reducer.mean(), pt.buffer(BUFFER), 20))
    return ee.Dictionary(out).getInfo()


def ls_windows(ee, lat, lon, date):
    pt, d0 = _point(ee, lat, lon, date)
    base = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")).filterBounds(pt))

    def mask(img):
        qa = img.select("QA_PIXEL")
        clear = (qa.bitwiseAnd(1 << 1).eq(0).And(qa.bitwiseAnd(1 << 3).eq(0))
                 .And(qa.bitwiseAnd(1 << 4).eq(0)))
        return img.updateMask(clear)

    srb = list(ROY_SR.values())
    out = {}
    for w in WINDOWS:
        col = base.filterDate(d0.advance(-w, "day"), d0.advance(w, "day"))
        out[f"w{w}_n"] = col.size()
        out[f"w{w}"] = (_composite(ee, col, srb, mask)
                        .reduceRegion(ee.Reducer.mean(), pt.buffer(BUFFER), 30))
    return ee.Dictionary(out).getInfo()


def bands_at(v, w, sensor):
    """Reflectancia equivalente Sentinel-2 para una ventana, o None si falta."""
    red = v.get(f"w{w}") or {}
    out = {}
    for b in BANDS:
        raw = red.get(b if sensor == "S2" else ROY_SR[b])
        if raw is None:
            return None
        if sensor == "S2":
            out[b] = raw
        else:
            a, slope = ROY[b]
            out[b] = a + slope * (raw * 0.0000275 - 0.2)
    return out


def extract():
    import ee
    ee.Initialize(project="black-display-445217-p3")
    field = load_field()
    done = set()
    if CACHE.exists():
        prev = pd.read_csv(CACHE)
        done = set(zip(prev.sensor, prev.station.astype(str), prev.date.astype(str)))
        print(f"  cache: {len(prev)} filas ya extraidas")

    rows, t0 = [], time.time()
    jobs = [("S2", s2_windows, S2_START), ("LS", ls_windows, LS_START)]
    for sensor, fn, y0 in jobs:
        sub = field[field.date.dt.year >= y0]
        print(f"\n  {sensor}: {len(sub)} mediciones desde {y0}")
        for i, r in enumerate(sub.itertuples(), 1):
            key = (sensor, str(r.station), r.date.strftime("%Y-%m-%d"))
            if key in done:
                continue
            try:
                v = fn(ee, r.lat, r.lon, r.date)
            except Exception as exc:
                print(f"    [{i}] {r.station} {key[2]}: {repr(exc)[:60]}")
                time.sleep(2)
                continue
            rec = {"sensor": sensor, "station": r.station, "zona": r.zona,
                   "date": key[2], "lat": r.lat, "lon": r.lon, "secchi": r.secchi}
            for w in WINDOWS:
                vals = bands_at(v, w, sensor)
                rec[f"w{w}_n"] = v.get(f"w{w}_n")
                for b in BANDS:
                    rec[f"w{w}_{b}"] = None if vals is None else vals[b]
            rows.append(rec)
            if len(rows) % 25 == 0:
                flush(rows)
                print(f"    {i}/{len(sub)}  {time.time() - t0:.0f}s")
                rows = []
    flush(rows)
    print(f"\n  extraccion terminada en {time.time() - t0:.0f}s -> {CACHE}")


def flush(rows):
    if not rows:
        return
    pd.DataFrame(rows).to_csv(CACHE, mode="a", index=False,
                              header=not CACHE.exists(), float_format="%.6f")


def agreement_with_primary(raw):
    """Contraste de la ventana +-10d contra los match-ups del analisis principal.

    Comprueba que esta extraccion reproduce el dataset del articulo antes de
    leer nada de las ventanas estrechas.
    """
    ref = TIDY / "insitu_distribution.csv"
    if not ref.exists():
        return {}
    o = pd.read_csv(ref)
    o["key"] = o.sensor + "|" + o.station.astype(str) + "|" + o.campaign_date.astype(str)
    # el registro de IMARPE trae E59 del 2017-07-16 por duplicado; sin esto el
    # merge devolveria mas coincidencias que match-ups
    o = o.drop_duplicates("key")
    n = raw[raw.w10_n.fillna(0) > 0].drop_duplicates(["sensor", "station", "date"]).copy()
    n["key"] = n.sensor + "|" + n.station.astype(str) + "|" + n.date.astype(str)
    m = o.merge(n[["key"] + [f"w10_{b}" for b in BANDS]], on="key")
    rs = {b: float(np.corrcoef(m[f"w10_{b}"], m[b])[0, 1]) for b in BANDS}
    out = {"primary_n": int(len(o)), "recovered_n": int(len(m)),
           "recovered_pct": round(100 * len(m) / len(o), 1),
           "reflectance_r_min": round(min(rs.values()), 4),
           "reflectance_r_by_band": {b: round(v, 4) for b, v in rs.items()}}
    print(f"\n  Contraste con el dataset principal: {out['recovered_n']} de "
          f"{out['primary_n']} match-ups unicos recuperados "
          f"({out['recovered_pct']}%), reflectancia r>="
          f"{out['reflectance_r_min']:.3f} en las seis bandas.")
    return out


def evaluate():
    if not CACHE.exists():
        sys.exit("Falta la cache; ejecuta sin --eval para extraer primero.")
    raw = pd.read_csv(CACHE)
    raw["campaign_date"] = raw["date"]
    rows = []
    for w in WINDOWS:
        d = raw.rename(columns={f"w{w}_{b}": b for b in BANDS}).copy()
        d = d[d[f"w{w}_n"].fillna(0) > 0]
        d = add_indices(d).dropna(subset=FEATURES + ["secchi"])
        if len(d) < 50:
            print(f"    +-{w:2d} dias: solo {len(d)} match-ups, se omite")
            continue
        y, oof = cv_oof(d, FEATURES, d.campaign_date.values, GroupKFold(N_SPLITS))
        m = metrics(y, oof)
        rows.append({"window_days": w, "n_matchups": int(len(d)),
                     "n_campaigns": int(d.campaign_date.nunique()),
                     "median_images": float(d[f"w{w}_n"].median()), **m})
        print(f"    +-{w:2d} dias  n={len(d):4d}  R2={m['R2']:+.3f}  "
              f"RMSE={m['RMSE']:.2f} m  imagenes/match-up={d[f'w{w}_n'].median():.0f}")

    out = pd.DataFrame(rows)
    out.to_csv(TIDY / "window_sensitivity.csv", index=False, float_format="%.4f")
    ref = out[out.window_days == 10]
    summary = {"design": "GroupKFold(5) blocked by campaign date, RF, 12 features",
               "windows": rows, **agreement_with_primary(raw)}
    if not ref.empty:
        best = out.loc[out.R2.idxmax()]
        summary["r2_gain_vs_10d"] = float(best.R2 - ref.R2.iloc[0])
        summary["n_lost_vs_10d"] = int(ref.n_matchups.iloc[0] - best.n_matchups)
        summary["best_window_days"] = int(best.window_days)
    pd.Series(summary).to_json(MET / "window_sensitivity.json", indent=2)
    print(f"\n  {TIDY / 'window_sensitivity.csv'}")
    print(f"  {MET / 'window_sensitivity.json'}")


def main():
    banner("p11 -- SENSIBILIDAD A LA VENTANA TEMPORAL DEL MATCH-UP")
    if "--eval" not in sys.argv:
        extract()
    banner("EVALUACION BAJO EL DISENO PRIMARIO (bloqueo por campana)", "-")
    evaluate()


if __name__ == "__main__":
    main()
