"""
p12_multilake_hls.py -- la diferencia residual entre sensores en el producto
HLS v2.0, ¿es del Titicaca o de toda el agua?

p14 muestra que sobre el Titicaca Landsat 8/9 (HLSL30) y Sentinel-2 (HLSS30)
siguen siendo separables aun despues de LaSRC comun, BRDF y ajuste de banda.
Aqui se repite la prueba en lagos grandes de todo el mundo, de ultra claros a
turbios, con PARES DEL MISMO DIA: en cada fecha en que ambos productos cubren
el lago se muestrea el mismo punto en las dos imagenes. Asi no queda
confusion temporal (estacion, floraciones, nivel del lago) entre sensores.

Muestreo: poligono de HydroLAKES encogido hacia adentro (1-3 km segun el
tamano), N_POINTS puntos aleatorios con semilla; en cada fecha, mosaico de
las teselas de ese dia de cada producto, Fmask limpio (sin nube, adyacente,
sombra ni nieve/hielo) y bit de agua obligatorio en AMBOS productos; muestra
del pixel de 30 m (ambos productos comparten la grilla MGRS).

Por lago: prueba de dos muestras LS vs S2 agrupada por fecha, control negativo
(mitades de fechas dentro de un sensor), y estadisticos por banda de los
pares. Ademas una prueba conjunta agrupada por lago-fecha y otra que deja
lagos enteros fuera, y la relacion con la claridad (verde mediano de S2).

Requiere Earth Engine:  earthengine authenticate

    python pipeline/p12_multilake_hls.py           # extrae (reanudable) y evalua
    python pipeline/p12_multilake_hls.py --eval    # solo reevalua la cache

Salidas: data/processed/multilake_hls_raw.csv (cache de la extraccion)
         results/tidy/multilake_hls_separability.csv
         results/tidy/multilake_hls_bands.csv
         results/metrics/multilake_hls.json
"""
import json
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             roc_auc_score)
from sklearn.model_selection import GroupKFold

from p00_config import (write_json, BANDS, FEATURES, GEE_PROJECT, MET, PROC, SEED, TIDY,
                        add_indices, banner)

CACHE = PROC / "multilake_hls_raw.csv.gz"   # gzip: anexa miembros, pandas los lee
HYDROLAKES = "projects/sat-io/open-datasets/HydroLakes/lake_poly_v10"
L30_ID, S30_ID = "NASA/HLS/HLSL30/v002", "NASA/HLS/HLSS30/v002"
START, END = "2019-01-01", "2025-01-01"
N_POINTS = 200
MAX_DATES = 40          # fechas utiles (>= MIN_PAIRS pares) por lago
MAX_CANDIDATES = 100    # fechas candidatas revisadas como maximo por lago
MIN_PAIRS = 5
MAX_CLOUD = 70          # CLOUD_COVERAGE medio de las teselas del dia (%)
BATCH = 8               # fechas por llamada getInfo
POOL_CAP = 60           # puntos por fecha en la prueba conjunta
MIN_DATES_TEST = 8
N_SPLITS = 5

# bandas de cada producto HLS en las posiciones comunes del estudio
S30 = {"B2": "B2", "B3": "B3", "B4": "B4", "B8": "B8A", "B11": "B11", "B12": "B12"}
L30 = {"B2": "B2", "B3": "B3", "B4": "B4", "B8": "B5", "B11": "B6", "B12": "B7"}
# Fmask v2.0: bit 1 nube, 2 adyacente, 3 sombra, 4 nieve/hielo, 5 agua
FMASK_BAD = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4)
FMASK_WATER = 1 << 5

# Titicaca se parte en Lago Mayor y Lago Menor con una caja al sur del
# estrecho de Tiquina; el resto son poligonos enteros de HydroLAKES.
MENOR_BOX = [-69.12, -16.80, -68.50, -16.22]
LAKES = {  # nombre: (Hylak_id, recorte)
    "Titicaca Mayor": (78, "mayor"), "Titicaca Menor": (78, "menor"),
    "Tahoe": (792, None), "Baikal": (11, None), "Malawi": (18, None),
    "Tanganyika": (17, None), "Victoria": (16, None), "Superior": (5, None),
    "Michigan": (6, None), "Erie": (9, None), "Ontario": (7, None),
    "Okeechobee": (69, None), "Winnipeg": (4, None), "Great Slave": (3, None),
    "Ladoga": (10, None), "Vanern": (105, None), "Geneva": (1261, None),
    "Garda": (1282, None), "Balaton": (1251, None), "Qinghai": (143, None),
    "Taihu": (148, None), "Poyang": (151, None), "Issyk-Kul": (136, None),
    "Van": (141, None), "Nasser": (152, None), "Nicaragua": (72, None),
    "Taupo": (1698, None), "Chapala": (70, None),
}
LAT_BANDS = "CDEFGHJKLMNPQRSTUVWX"
ANGLE_FILL = -99999
ONLY = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--only=")]
SAMPLE_COLS = (["date", "pid"] + [f"LS_{b}" for b in BANDS]
               + [f"S2_{b}" for b in BANDS]
               + ["LS_VZA", "S2_VZA", "LS_SZA", "S2_SZA"])


# --------------------------------------------------------------------------
# extraccion
# --------------------------------------------------------------------------
def _retry(fn, what):
    for attempt in range(3):
        try:
            return fn()
        except Exception as e:  # cuotas y cortes de red: reintentar
            print(f"    reintento {attempt + 1} ({what}): {str(e)[:150]}")
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"Earth Engine no respondio: {what}")


def _lake_geometry(ee, hid, clip):
    g = (ee.FeatureCollection(HYDROLAKES).filter(ee.Filter.eq("Hylak_id", hid))
         .first().geometry())
    box = ee.Geometry.Rectangle(MENOR_BOX)
    if clip == "menor":
        g = g.intersection(box, 50)
    elif clip == "mayor":
        g = g.difference(box, 50)
    return g


def _lake_info(ee, name):
    hid, clip = LAKES[name]
    g = _lake_geometry(ee, hid, clip)
    area = g.area(100).divide(1e6)
    # encoger hacia adentro: 1 km en lagos chicos, 2-3 km en los grandes
    dist = ee.Number(ee.Algorithms.If(area.lt(1500), -1000,
                                      ee.Algorithms.If(area.lt(15000), -2000, -3000)))
    # tolerancia de 500 m: los puntos quedan a 1-3 km de la orilla, y con
    # contornos muy detallados (Winnipeg) un buffer fino agota el tiempo de EE
    inner = g.simplify(500).buffer(dist, 500)
    pts = ee.FeatureCollection.randomPoints(inner, N_POINTS, SEED, 10)
    info = ee.Dictionary({
        "area": area, "buffer_m": dist, "bounds": g.bounds(100).coordinates(),
        "pts": pts.geometry().coordinates()}).getInfo()
    xy = np.array(info["pts"], dtype=float).reshape(-1, 2)
    b = np.array(info["bounds"][0])
    return {"area_km2": info["area"], "buffer_m": info["buffer_m"],
            "lon_min": b[:, 0].min(), "lon_max": b[:, 0].max(),
            "lat_min": b[:, 1].min(), "lat_max": b[:, 1].max(),
            "points": pd.DataFrame({"pid": np.arange(len(xy)),
                                    "lon": xy[:, 0], "lat": xy[:, 1]})}


def _tile_prefixes(info):
    """Prefijos MGRS (zona UTM + banda de latitud) que pueden cubrir el lago.
    filterBounds sobre HLSS30 devuelve teselas espurias (T01*, T60* de altas
    latitudes) por huellas mal definidas cerca del antimeridiano; filtrar por
    prefijo las descarta de forma robusta."""
    zones = range(int((info["lon_min"] - 0.5 + 180) // 6) + 1,
                  int((info["lon_max"] + 0.5 + 180) // 6) + 2)
    bands = LAT_BANDS[int((info["lat_min"] - 0.5 + 80) // 8):
                      int((info["lat_max"] + 0.5 + 80) // 8) + 1]
    return [f"T{((z - 1) % 60) + 1:02d}{b}" for z in zones for b in bands]


def _scenes(ee, name, geom, prefixes):
    """Escenas del producto sobre el lago. Se piden anio por anio: en lagos
    grandes (muchas teselas) una sola consulta de seis anios agota el tiempo
    de computo de Earth Engine."""
    filt = ee.Filter.Or(*[ee.Filter.stringStartsWith("system:index", p)
                          for p in prefixes])
    parts = []
    for y in range(int(START[:4]), int(END[:4]) + 1):
        a, b = max(START, f"{y}-01-01"), min(END, f"{y + 1}-01-01")
        if a >= b:
            continue
        col = (ee.ImageCollection(name).filterBounds(geom).filterDate(a, b)
               .filter(filt))
        parts.append(pd.DataFrame(_retry(lambda: ee.Dictionary({
            "id": col.aggregate_array("system:index"),
            "t": col.aggregate_array("system:time_start"),
            "cc": col.aggregate_array("CLOUD_COVERAGE")}).getInfo(), f"escenas {y}")))
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df.t, unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    return df


def _candidate_dates(ls, s2):
    """Fechas con escena de ambos productos, ordenadas para repartirse en
    anos y trimestres: ronda sobre los estratos (ano, trimestre), dentro de
    cada estrato de menor a mayor nubosidad. Determinista."""
    cl = ls.groupby("date").cc.mean()
    cs = s2.groupby("date").cc.mean()
    common = sorted(set(cl.index) & set(cs.index))
    d = pd.DataFrame({"date": common})
    if d.empty:
        return [], 0
    d["cc"] = np.maximum(cl.loc[common].values, cs.loc[common].values)
    n_common = len(d)
    d = d[d.cc <= MAX_CLOUD].copy()
    dt = pd.to_datetime(d.date)
    d["stratum"] = dt.dt.year.astype(str) + "Q" + dt.dt.quarter.astype(str)
    d = d.sort_values(["stratum", "cc", "date"])
    d["rank"] = d.groupby("stratum").cumcount()
    d = d.sort_values(["rank", "cc", "date"])
    return d.date.tolist(), n_common


def _product_image(ee, cid, ids, bands, prefix):
    src = list(bands.values())
    col = ee.ImageCollection(cid).filter(ee.Filter.inList("system:index", ids))

    def prep(i):
        fm = i.select("Fmask")
        ok = fm.bitwiseAnd(FMASK_BAD).eq(0).And(fm.bitwiseAnd(FMASK_WATER).neq(0))
        # las bandas de angulos faltan en algunas escenas L30: sin unmask, un
        # angulo enmascarado haria que sampleRegions descarte el punto entero
        ang = (i.select(["VZA", "SZA"], [f"{prefix}_VZA", f"{prefix}_SZA"])
               .unmask(ANGLE_FILL, False).toFloat())
        out = i.select(src, [f"{prefix}_{b}" for b in bands]).addBands(ang)
        return out.updateMask(ok)

    return col.map(prep).mosaic(), col.first().select("B3").projection()


def _sample_dates(ee, dates, ls, s2, pts_fc):
    out = []
    for d in dates:
        lsi, _ = _product_image(ee, L30_ID, ls[ls.date == d].id.tolist(), L30, "LS")
        s2i, proj = _product_image(ee, S30_ID, s2[s2.date == d].id.tolist(), S30, "S2")
        img = lsi.addBands(s2i)
        valid = lsi.select("LS_B3").mask().And(s2i.select("S2_B3").mask())
        img = img.updateMask(valid)
        smp = img.sampleRegions(collection=pts_fc, properties=["pid"],
                                scale=30, projection=proj, geometries=False)
        out.append(smp.map(lambda f, d=d: f.set("date", d)))
    return ee.FeatureCollection(out).flatten()


def _platforms(ls, s2, d):
    """Satelite del dia: L8/L9 desde el id de producto Landsat, S2A/S2B no
    viene en el id, se deja el id de la primera tesela."""
    return ls[ls.date == d].id.iloc[0], s2[s2.date == d].id.iloc[0]


def extract():
    import ee
    ee.Initialize(project=GEE_PROJECT)
    prev = pd.read_csv(CACHE) if CACHE.exists() else pd.DataFrame()
    t0 = time.time()
    for name in LAKES:
        if ONLY and name not in ONLY:
            continue
        done = prev[prev.lake == name] if len(prev) else pd.DataFrame()
        if len(done) and done.status.eq("complete").any():
            continue
        print(f"  {name}: preparando puntos y escenas  {time.time() - t0:.0f} s")
        try:
            _extract_lake(ee, name, done, t0)
        except Exception as e:  # un lago que no responde no detiene a los demas
            print(f"    {name}: OMITIDO ({str(e)[:120]})")
            _append(pd.DataFrame([{"lake": name, "hylak_id": LAKES[name][0],
                                   "date": "", "pid": -3, "status": "failed"}]))


def _extract_lake(ee, name, done, t0):
    if True:
        info = _retry(lambda: _lake_info(ee, name), f"{name} puntos")
        pref = _tile_prefixes(info)
        # escenas que tocan los puntos de muestreo (multipunto): filtrar por
        # el poligono completo agota el tiempo de calculo en lagos complejos
        mp = ee.Geometry.MultiPoint(info["points"][["lon", "lat"]].values.tolist())
        ls = _retry(lambda: _scenes(ee, L30_ID, mp, pref), f"{name} L30")
        s2 = _retry(lambda: _scenes(ee, S30_ID, mp, pref), f"{name} S30")
        cands, n_common = ([], 0) if ls.empty or s2.empty else _candidate_dates(ls, s2)
        cands = cands[:MAX_CANDIDATES]
        pts = info["points"]
        pts_fc = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([r.lon, r.lat]), {"pid": int(r.pid)})
            for r in pts.itertuples()])
        meta = {"lake": name, "hylak_id": LAKES[name][0],
                "area_km2": round(info["area_km2"], 1),
                "buffer_m": info["buffer_m"], "n_points": len(pts),
                "n_common_dates": n_common, "n_candidates": len(cands)}
        seen = set(done.date) if len(done) else set()
        n_useful = (done[done.pid >= 0].groupby("date").size() >= MIN_PAIRS).sum() \
            if len(done) else 0
        todo = [d for d in cands if d not in seen]
        print(f"    {len(ls)} L30, {len(s2)} S30, {n_common} dias comunes, "
              f"{len(cands)} candidatos, {n_useful} utiles en cache")
        i = 0
        while n_useful < MAX_DATES and i < len(todo):
            batch = todo[i:i + BATCH]
            i += BATCH
            fc = _sample_dates(ee, batch, ls, s2, pts_fc)
            feats = _retry(lambda: fc.getInfo()["features"], f"{name} muestreo")
            rows = []
            for f in feats:
                p = f["properties"]
                rows.append({k: p.get(k) for k in SAMPLE_COLS})
            got = pd.DataFrame(rows, columns=SAMPLE_COLS)
            empty = [d for d in batch if d not in set(got.date)]
            if empty:  # fechas revisadas sin pares: centinela pid = -1
                got = pd.concat([got, pd.DataFrame({"date": empty, "pid": -1})],
                                ignore_index=True)
            got = got.merge(pts, on="pid", how="left")
            lsid, s2id = zip(*[_platforms(ls, s2, d) for d in got.date])
            got["ls_scene"], got["s2_scene"] = lsid, s2id
            for k, v in meta.items():
                got[k] = v
            got["status"] = "partial"
            n_useful += int((got[got.pid >= 0].groupby("date").size()
                             >= MIN_PAIRS).sum())
            _append(got)
            print(f"    {name}: {i}/{len(todo)} fechas revisadas, {n_useful} utiles, "
                  f"{(got.pid >= 0).sum()} pares en el lote  {time.time() - t0:.0f} s")
        mark = pd.DataFrame([{**meta, "date": "", "pid": -2, "status": "complete"}])
        _append(mark)


COLS = (["lake", "hylak_id", "area_km2", "buffer_m", "n_points", "n_common_dates",
         "n_candidates", "status", "date", "pid", "lon", "lat", "ls_scene",
         "s2_scene"] + [f"LS_{b}" for b in BANDS] + [f"S2_{b}" for b in BANDS]
        + ["LS_VZA", "S2_VZA", "LS_SZA", "S2_SZA"])


def _append(df):
    df = df.reindex(columns=COLS)
    df.to_csv(CACHE, mode="a", header=not CACHE.exists(), index=False)


# --------------------------------------------------------------------------
# evaluacion
# --------------------------------------------------------------------------
def sep_test(df, feats, groups, label_col="sensor", positive="S2"):
    """Clasificador de dos muestras con validacion agrupada (copia local de la
    prueba de p02: GroupKFold(5), RandomForest(300)). Devuelve exactitud,
    exactitud balanceada, AUC y el nivel de azar (clase mayoritaria)."""
    df = df.reset_index(drop=True)
    y = (df[label_col] == positive).astype(int).values
    proba = np.zeros(len(df))
    for tr, te in GroupKFold(N_SPLITS).split(df, y, groups):
        clf = RandomForestClassifier(300, random_state=SEED, n_jobs=-1)
        clf.fit(df.iloc[tr][feats].values, y[tr])
        proba[te] = clf.predict_proba(df.iloc[te][feats].values)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {"n": int(len(df)), "chance": round(float(max(y.mean(), 1 - y.mean())), 4),
            "accuracy": round(float(accuracy_score(y, pred)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
            "auc": round(float(roc_auc_score(y, proba)), 4)}


def load_pairs():
    raw = pd.read_csv(CACHE)
    p = raw[raw.pid >= 0].dropna(subset=[f"{s}_{b}" for s in ("LS", "S2")
                                         for b in BANDS])
    p = (p.drop_duplicates(["lake", "date", "pid"])
         .sort_values(["lake", "date", "pid"]).reset_index(drop=True))
    for c in ("LS_VZA", "S2_VZA", "LS_SZA", "S2_SZA"):
        p[c] = p[c].where(p[c] > ANGLE_FILL, np.nan)  # GEE ya da grados
    # solo fechas con suficientes pares validos
    n = p.groupby(["lake", "date"]).pid.transform("size")
    return raw, p[n >= MIN_PAIRS].reset_index(drop=True)


def to_long(p):
    keep = ["lake", "date", "pid"]
    parts = []
    for s in ("LS", "S2"):
        t = p[keep + [f"{s}_{b}" for b in BANDS]].rename(
            columns={f"{s}_{b}": b for b in BANDS})
        t["sensor"] = s
        parts.append(t)
    d = add_indices(pd.concat(parts, ignore_index=True))
    d["lake_date"] = d.lake + "|" + d.date
    return d.dropna(subset=FEATURES).reset_index(drop=True)


def band_stats(p):
    rows = []
    for b in BANDS:
        ls, s2 = p[f"LS_{b}"], p[f"S2_{b}"]
        # razones solo con senal: en NIR/SWIR la mediana sobre agua es ~0
        den = s2.median() if abs(s2.median()) >= 5e-4 else np.nan
        rows.append({"band": b, "n_pairs": len(p),
                     "median_LS": round(float(ls.median()), 5),
                     "median_S2": round(float(s2.median()), 5),
                     "ratio_LS_S2": round(float(ls.median() / den), 3),
                     "pearson_r": round(float(np.corrcoef(ls, s2)[0, 1]), 3),
                     "median_diff": round(float((ls - s2).median()), 5),
                     "median_rel_diff": round(float((ls - s2).median() / den), 3)})
    return rows


def blue_green_departure(p):
    f = 1e-4
    r_ls = (p.LS_B2 / p.LS_B3.clip(lower=f)).median()
    r_s2 = (p.S2_B2 / p.S2_B3.clip(lower=f)).median()
    return float(r_ls / r_s2 - 1)


def control_halves(d, sensor, rng):
    sub = d[d.sensor == sensor].copy()
    dates = np.sort(sub.date.unique())
    half = set(rng.choice(dates, len(dates) // 2, replace=False))
    sub["half"] = np.where(sub.date.isin(half), "A", "B")
    return sep_test(sub, FEATURES, sub.date.values, label_col="half", positive="A")


def main():
    t0 = time.time()
    banner("p12 -- PARES DEL MISMO DIA HLSL30 vs HLSS30 EN LAGOS DEL MUNDO")
    if "--eval" not in sys.argv:
        extract()
    t_ext = time.time() - t0
    raw, p = load_pairs()
    meta = (raw.groupby("lake")[["hylak_id", "area_km2", "buffer_m",
                                 "n_common_dates", "n_candidates"]].first())
    reviewed = raw[raw.pid != -2].groupby("lake").date.nunique()

    rng = np.random.default_rng(SEED)
    sep_rows, band_rows = [], []
    for lake in LAKES:
        pl = p[p.lake == lake]
        nd = pl.date.nunique()
        base = {"lake": lake, "hylak_id": int(meta.loc[lake, "hylak_id"])
                if lake in meta.index else LAKES[lake][0],
                "area_km2": meta.area_km2.get(lake), "n_common_dates":
                meta.n_common_dates.get(lake), "n_dates_reviewed": reviewed.get(lake, 0),
                "n_dates": nd, "n_pairs": len(pl)}
        if nd < MIN_DATES_TEST:
            print(f"  {lake}: {nd} fechas con pares, no se prueba")
            sep_rows.append({**base, "tested": False})
            continue
        d = to_long(pl)
        main_t = sep_test(d, FEATURES, d.date.values)
        c_ls = control_halves(d, "LS", rng)
        c_s2 = control_halves(d, "S2", rng)
        bs = band_stats(pl)
        for r in bs:
            band_rows.append({"lake": lake, **r})
        bsd = {r["band"]: r for r in bs}
        row = {**base, "tested": True,
               "s2_green_median": bsd["B3"]["median_S2"],
               "balanced_accuracy": main_t["balanced_accuracy"],
               "auc": main_t["auc"], "accuracy": main_t["accuracy"],
               "control_LS_bacc": c_ls["balanced_accuracy"],
               "control_LS_auc": c_ls["auc"],
               "control_S2_bacc": c_s2["balanced_accuracy"],
               "control_S2_auc": c_s2["auc"],
               "bg_departure": round(blue_green_departure(pl), 4),
               "s2_vza_median": round(float(pl.S2_VZA.median()), 2),
               "ls_vza_median": round(float(pl.LS_VZA.median()), 2),
               "sza_median": round(float(pl.S2_SZA.median()), 2)}
        for b in BANDS:
            row[f"ratio_{b}"] = bsd[b]["ratio_LS_S2"]
            row[f"r_{b}"] = bsd[b]["pearson_r"]
        sep_rows.append(row)
        print(f"  {lake:15s} fechas {nd:3d} pares {len(pl):5d}  "
              f"bacc {row['balanced_accuracy']:.3f} auc {row['auc']:.3f}  "
              f"ctrl {row['control_LS_bacc']:.2f}/{row['control_S2_bacc']:.2f}  "
              f"B2 {row['ratio_B2']:.2f} B3 {row['ratio_B3']:.2f} "
              f"B4 {row['ratio_B4']:.2f}  {time.time() - t0:.0f} s")
    sep = pd.DataFrame(sep_rows)
    tested = sep[sep.tested]

    # prueba conjunta: hasta POOL_CAP puntos por lago-fecha (semilla fija)
    pp = p[p.lake.isin(tested.lake)]
    pp = pp.sample(frac=1, random_state=SEED)
    pp = (pp[pp.groupby(["lake", "date"]).cumcount() < POOL_CAP]
          .sort_values(["lake", "date", "pid"]))
    dl = to_long(pp)
    pooled = sep_test(dl, FEATURES, dl.lake_date.values)
    pooled_lake = (sep_test(dl, FEATURES, dl.lake.values)
                   if dl.lake.nunique() >= N_SPLITS else None)
    ctl = dl[dl.sensor == "S2"].copy()
    ld = np.sort(ctl.lake_date.unique())
    half = set(rng.choice(ld, len(ld) // 2, replace=False))
    ctl["half"] = np.where(ctl.lake_date.isin(half), "A", "B")
    pooled_ctrl = sep_test(ctl, FEATURES, ctl.lake_date.values,
                           label_col="half", positive="A")
    print(f"  conjunta por lago-fecha: {pooled}")
    print(f"  conjunta dejando lagos fuera: {pooled_lake}")
    print(f"  control conjunto (S2, mitades de lago-fecha): {pooled_ctrl}")

    corr = {}
    for col in ("balanced_accuracy", "auc", "bg_departure", "ratio_B2", "ratio_B3"):
        x = tested.s2_green_median
        y = tested[col].abs() if col == "bg_departure" else tested[col]
        rho, pv = spearmanr(x, y)
        corr[f"{col}{'_abs' if col == 'bg_departure' else ''}_vs_clarity"] = {
            "rho": round(float(rho), 3), "p": float(f"{pv:.3g}"),
            "n_lakes": int(len(tested))}
    print("  Spearman vs verde mediano S2:", corr)
    med = tested.s2_green_median.median()
    clear, turbid = tested[tested.s2_green_median <= med], tested[tested.s2_green_median > med]
    halves = {"clear": {"n": int(len(clear)),
                        "bacc_median": round(float(clear.balanced_accuracy.median()), 3),
                        "bacc_min": round(float(clear.balanced_accuracy.min()), 3)},
              "turbid": {"n": int(len(turbid)),
                         "bacc_median": round(float(turbid.balanced_accuracy.median()), 3),
                         "bacc_min": round(float(turbid.balanced_accuracy.min()), 3)},
              "green_split": round(float(med), 5)}
    print("  claros vs turbios:", halves)

    sep.to_csv(TIDY / "multilake_hls_separability.csv", index=False)
    pd.DataFrame(band_rows).to_csv(TIDY / "multilake_hls_bands.csv", index=False)
    out = {"period": [START, END], "n_points_per_lake": N_POINTS,
           "max_dates": MAX_DATES, "min_pairs_per_date": MIN_PAIRS,
           "max_cloud_tile_pct": MAX_CLOUD, "pool_cap_per_date": POOL_CAP,
           "n_lakes": len(LAKES), "n_lakes_tested": int(len(tested)),
           "lakes_not_tested": sep.loc[~sep.tested, "lake"].tolist(),
           "n_pairs_total": int(len(p)), "n_dates_total": int(p.groupby("lake").date.nunique().sum()),
           "pooled_lake_date": pooled, "pooled_leave_lake_out": pooled_lake,
           "pooled_control_S2": pooled_ctrl,
           "bacc_range": [float(tested.balanced_accuracy.min()),
                          float(tested.balanced_accuracy.max())],
           "n_lakes_bacc_ge_0p8": int((tested.balanced_accuracy >= 0.8).sum()),
           "n_lakes_bacc_lt_0p65": int((tested.balanced_accuracy < 0.65).sum()),
           "spearman_vs_s2_green": corr, "clear_vs_turbid": halves,
           "runtime_s": {"extraction": round(t_ext), "total": round(time.time() - t0)},
           "lakes": sep.to_dict(orient="records")}
    write_json(out, MET / "multilake_hls.json")


if __name__ == "__main__":
    main()
