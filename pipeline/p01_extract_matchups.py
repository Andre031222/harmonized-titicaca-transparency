"""
p01_extract_matchups.py -- extrae la reflectancia satelital en cada medicion
de campo. Es la UNICA etapa que necesita Earth Engine y el registro crudo de
IMARPE; el resto del pipeline arranca de data/processed/matchups.csv.

Dos productos, cuatro ventanas (+-1, +-3, +-5, +-10 dias):
  hls : NASA HLS v2.0 (HLSS30 y HLSL30). Ambos sensores con LaSRC, BRDF
        normalizada a nadir y ajuste de banda; grilla comun de 30 m. Mascara
        Fmask (nube, adyacente, sombra, nieve). Es el dato principal.
  sr  : reflectancia de superficie de cada sensor tal como la distribuye su
        agencia: Sentinel-2 L2A (Sen2Cor, COPERNICUS/S2_SR_HARMONIZED, /10000)
        y Landsat 8/9 Collection 2 L2 (LaSRC, x2.75e-5 - 0.2). Es la via de
        comparacion (p03 le aplica los coeficientes de banda de HLS).

En cada ventana la reflectancia es la mediana por pixel de todas las imagenes
limpias, promediada en un buffer de 30 m alrededor de la coordenada real.

    python pipeline/p01_extract_matchups.py         # extrae lo que falte
"""
import time

import numpy as np
import pandas as pd

from p00_config import (BANDS, FIELD_RAW, GEE_PROJECT, HLS_L30, HLS_S30, LS_C2,
                        PROC, banner)

WINDOWS = [1, 3, 5, 10]
BUFFER = 30
CHUNK = 40
HLS_CACHE = PROC / "hls_matchups_raw.csv"
SR_CACHE = PROC / "sr_matchups_raw.csv"
FMASK_BAD = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4)
S2_CLDPRB = 30
MONTHS = {"ENE": 1, "FEB": 2, "MAR": 3, "ABR": 4, "MAY": 5, "JUN": 6, "JUL": 7,
          "AGO": 8, "SET": 9, "SEP": 9, "OCT": 10, "NOV": 11, "DIC": 12}


def load_field(secchi_only=False):
    """Registro de campo de IMARPE, una fila por estacion y fecha."""
    j = pd.read_csv(FIELD_RAW)
    j.columns = [c.replace("Año", "Ano") for c in j.columns]
    j["date"] = pd.to_datetime(dict(year=j.Ano, month=j.Mes.str.upper().str[:3].map(MONTHS),
                                    day=j.Dia), errors="coerce")
    j = j.rename(columns={"Latitud": "lat", "Longitud": "lon", "Zona": "zona",
                          "Estacion": "station", "Transparencia_m": "secchi",
                          "Clorofila-A": "chl", "SST": "tss",
                          "Temperatura": "temp_insitu"})
    j = j.dropna(subset=["date", "lat", "lon"])
    # E59 del 2017-07-16 esta duplicado en el registro con la misma lectura
    j = j.drop_duplicates(subset=["station", "date"])
    j = fix_2011_zones(j)
    if secchi_only:
        j = j.dropna(subset=["secchi"])
    j["key"] = j.station.astype(str) + "|" + j.date.dt.strftime("%Y-%m-%d")
    return j[["key", "station", "zona", "date", "lat", "lon", "secchi", "chl",
              "tss", "temp_insitu"]].reset_index(drop=True)


def fix_2011_zones(j):
    """En 2011 el registro rotula las 86 lecturas como BAHIA PUNO, aunque 76
    estan fuera de la bahia (desde 2012 la bahia queda al oeste de 69.77 W).
    Se reasigna la zona de 2011 con la de la lectura etiquetada mas cercana de
    los demas anios. Los match-ups (2013 en adelante) no cambian."""
    from scipy.spatial import cKDTree
    ref = j[j.date.dt.year >= 2012]
    m = j.date.dt.year == 2011
    if m.any():
        _, i = cKDTree(ref[["lon", "lat"]].values).query(j.loc[m, ["lon", "lat"]].values)
        j = j.copy()
        j["zona_as_issued"] = j.zona
        j.loc[m, "zona"] = ref.zona.values[i]
    return j


# --- HLS ---------------------------------------------------------------------
def _hls_fc(ee, rows, name, bands, extra):
    src = list(bands.values()) + extra

    def per_point(f):
        pt = f.geometry()
        d0 = ee.Date(f.get("date"))
        # filterBounds devuelve teselas T01K* espurias junto al antimeridiano
        base = (ee.ImageCollection(name).filterBounds(pt)
                .filter(ee.Filter.stringStartsWith("system:index", "T19")))
        res = {}
        for w in WINDOWS:
            col = base.filterDate(d0.advance(-w, "day"), d0.advance(w + 1, "day"))
            masked = col.map(lambda i: i.updateMask(
                i.select("Fmask").bitwiseAnd(FMASK_BAD).eq(0)))
            empty = ee.Image.constant([0] * len(src)).rename(src).updateMask(ee.Image(0))
            comp = ee.Image(ee.Algorithms.If(col.size().gt(0),
                                             masked.select(src).median(), empty))
            clear = ee.Image(ee.Algorithms.If(
                col.size().gt(0), masked.select(src[0]).count(), ee.Image(0)))
            res[f"w{w}"] = comp.reduceRegion(ee.Reducer.mean(), pt.buffer(BUFFER), 30)
            res[f"w{w}_n"] = clear.reduceRegion(ee.Reducer.max(), pt.buffer(BUFFER), 30).values().get(0)
        return f.set(res)

    fc = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([r.lon, r.lat]),
                   {"key": r.key, "date": r.date.strftime("%Y-%m-%d")})
        for r in rows.itertuples()])
    return fc.map(per_point), bands, extra


def extract_hls(ee, field):
    done = set()
    if HLS_CACHE.exists():
        prev = pd.read_csv(HLS_CACHE)
        done = set(zip(prev.sensor, prev.key))
    jobs = [("S2", "NASA/HLS/HLSS30/v002", HLS_S30, [], 2015),
            ("LS", "NASA/HLS/HLSL30/v002", HLS_L30, ["B10"], 2013)]
    t0 = time.time()
    for sensor, name, bands, extra, y0 in jobs:
        sub = field[(field.date.dt.year >= y0)
                    & ~field.key.map(lambda k: (sensor, k) in done)]
        print(f"  HLS {sensor}: {len(sub)} mediciones por extraer")
        for i in range(0, len(sub), CHUNK):
            part = sub.iloc[i:i + CHUNK]
            fc, bmap, ex = _hls_fc(ee, part, name, bands, extra)
            for attempt in range(5):
                try:
                    info = fc.getInfo()
                    break
                except Exception as e:  # cuotas o red
                    print(f"    reintento {attempt + 1}: {str(e)[:100]}")
                    time.sleep(15 * (attempt + 1))
            else:
                raise RuntimeError("Earth Engine no respondio")
            rows = []
            for f in info["features"]:
                p = f["properties"]
                for w in WINDOWS:
                    v = p.get(f"w{w}") or {}
                    row = {"product": "hls", "sensor": sensor, "key": p["key"],
                           "window": w, "n_img": p.get(f"w{w}_n")}
                    for b, s in bmap.items():
                        row[b] = v.get(s)
                    row["B10"] = v.get("B10") if ex else None
                    rows.append(row)
            pd.DataFrame(rows).to_csv(HLS_CACHE, mode="a", index=False,
                                      header=not HLS_CACHE.exists(), float_format="%.6f")
            print(f"    {sensor} {i + len(part)}/{len(sub)}  {time.time() - t0:.0f} s")


# --- reflectancia de superficie de cada agencia -------------------------------
def _sr_point(ee, sensor, lat, lon, date):
    pt = ee.Geometry.Point([lon, lat])
    d0 = ee.Date(date.strftime("%Y-%m-%d"))
    if sensor == "S2":
        base = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(pt)
        src = BANDS + ["B8A"]

        def mask(img):
            scl = img.select("SCL")
            good = (scl.neq(1).And(scl.neq(3)).And(scl.neq(8)).And(scl.neq(9))
                    .And(scl.neq(10)).And(scl.neq(11)))
            return img.updateMask(good.And(img.select("MSK_CLDPRB").lt(S2_CLDPRB)))
        scale, k, off = 20, 1e-4, 0.0
    else:
        base = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
                .merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")).filterBounds(pt))
        src = list(LS_C2.values())

        def mask(img):
            qa = img.select("QA_PIXEL")
            return img.updateMask(qa.bitwiseAnd(1 << 1).eq(0)
                                  .And(qa.bitwiseAnd(1 << 3).eq(0))
                                  .And(qa.bitwiseAnd(1 << 4).eq(0)))
        scale, k, off = 30, 2.75e-5, -0.2
    out = {}
    for w in WINDOWS:
        col = base.filterDate(d0.advance(-w, "day"), d0.advance(w, "day"))
        empty = ee.Image.constant([0] * len(src)).rename(src).updateMask(ee.Image(0))
        comp = ee.Image(ee.Algorithms.If(col.size().gt(0),
                                         col.map(mask).median().select(src), empty))
        out[f"w{w}_n"] = col.size()
        out[f"w{w}"] = comp.multiply(k).add(off).reduceRegion(
            ee.Reducer.mean(), pt.buffer(BUFFER), scale)
    return ee.Dictionary(out).getInfo(), src


def extract_sr(ee, field):
    """La cache sr_matchups_raw.csv ya cubre todas las mediciones con Secchi;
    solo se extrae lo que falte (formato ancho w{ventana}_{banda})."""
    done = set()
    if SR_CACHE.exists():
        prev = pd.read_csv(SR_CACHE)
        done = set(zip(prev.sensor, prev.station.astype(str), prev.date.astype(str)))
    rows = []
    for sensor, y0 in [("S2", 2016), ("LS", 2013)]:
        sub = field[field.date.dt.year >= y0]
        for r in sub.itertuples():
            k = (sensor, str(r.station), r.date.strftime("%Y-%m-%d"))
            if k in done:
                continue
            v, src = _sr_point(ee, sensor, r.lat, r.lon, r.date)
            rec = {"sensor": sensor, "station": r.station, "zona": r.zona,
                   "date": k[2], "lat": r.lat, "lon": r.lon, "secchi": r.secchi}
            for w in WINDOWS:
                red = v.get(f"w{w}") or {}
                rec[f"w{w}_n"] = v.get(f"w{w}_n")
                names = BANDS + ["B8A"] if sensor == "S2" else BANDS
                for b in names:
                    s = b if sensor == "S2" else LS_C2[b]
                    rec[f"w{w}_{b}"] = red.get(s)
            rows.append(rec)
    if rows:
        pd.DataFrame(rows).to_csv(SR_CACHE, mode="a", index=False,
                                  header=not SR_CACHE.exists(), float_format="%.6f")
    print(f"  SR: {len(rows)} mediciones nuevas")


def main():
    banner("p01 -- EXTRACCION DE MATCH-UPS (Earth Engine)")
    import ee
    ee.Initialize(project=GEE_PROJECT)
    field = load_field()
    field = field[field.date.dt.year >= 2013]
    extract_hls(ee, field)
    sec = load_field(secchi_only=True)
    extract_sr(ee, sec[sec.date.dt.year >= 2013])


if __name__ == "__main__":
    main()
