"""
p17_glint.py -- ¿explica el destello solar (sun glint) la separabilidad
HLSL30 vs HLSS30 encontrada por p12 en 28 cuerpos de agua?

La cache de p12 no trae azimuts. Aqui se vuelven a armar, para cada lago-fecha
de la cache, los mismos mosaicos del dia de cada producto (mismas teselas,
mismo Fmask, misma mascara conjunta) y se muestrean SZA, SAA, VZA, VAA y B3
(control de que el pixel es el mismo) en los puntos de cada par.

Angulo de destello (especular): cos(tg) = cos SZA cos VZA - sin SZA sin VZA
cos(SAA - VAA); tg = 0 cuando VZA = SZA y VAA = SAA + 180.

Prueba de dos muestras de p12 (RF 300, 12 variables, folds agrupados por
fecha) por lago y conjunta: (a) todos los pares, (b) solo pares con tg > 30
en AMBOS sensores, (c) tg > 40. Mas: correlacion entre la exactitud
balanceada por lago y el destello mediano, y regresion por par de la
diferencia LS - S2 en verde con los angulos de destello como covariables
(efectos fijos de lago, errores agrupados por lago-fecha).

    python pipeline/p17_glint.py          # extrae (reanudable) y evalua
    python pipeline/p17_glint.py --eval   # solo evalua la cache

Entradas: data/processed/multilake_hls_raw.csv.gz (p12)
Salidas:  data/processed/multilake_angles.csv.gz (cache de angulos)
          results/tidy/glint_separability.csv
          results/metrics/glint.json
"""
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from p00_config import FEATURES, GEE_PROJECT, MET, PROC, SEED, TIDY, banner, write_json
from p12_multilake_hls import (ANGLE_FILL, FMASK_BAD, FMASK_WATER, L30_ID, LAKES,
                               MIN_DATES_TEST, MIN_PAIRS, N_SPLITS, POOL_CAP, S30_ID,
                               END, START, _lake_geometry, _retry, _tile_prefixes,
                               load_pairs, sep_test, to_long)

CACHE = PROC / "multilake_angles.csv.gz"
BATCH = 8
ANG = ["SZA", "SAA", "VZA", "VAA"]
MEAN = ["MEAN_SUN_ZENITH_ANGLE", "MEAN_SUN_AZIMUTH_ANGLE", "MEAN_VIEW_ZENITH_ANGLE",
        "MEAN_VIEW_AZIMUTH_ANGLE"]
MCOLS = ["tile", "t", "mSZA", "mSAA", "mVZA", "mVAA"]   # escena que dio el pixel
ACOLS = (["lake", "date", "pid"] + [f"{s}_{a}" for s in ("LS", "S2") for a in ANG]
         + ["LS_B3_chk", "S2_B3_chk"] + [f"{s}_{c}" for s in ("LS", "S2") for c in MCOLS])
THRESHOLDS = [None, 30, 40]


# --------------------------------------------------------------------------
# geometria
# --------------------------------------------------------------------------
def glint_angle(sza, vza, saa, vaa):
    """Angulo entre la direccion de vision y la direccion especular (grados)."""
    r = np.deg2rad
    c = (np.cos(r(sza)) * np.cos(r(vza))
         - np.sin(r(sza)) * np.sin(r(vza)) * np.cos(r(np.asarray(saa) - np.asarray(vaa))))
    return np.rad2deg(np.arccos(np.clip(c, -1, 1)))


def solar_position(t_ms, lat, lon):
    """Cenit y azimut solar (grados, azimut desde el norte en sentido horario)
    con el algoritmo de NOAA; t_ms en milisegundos UTC."""
    t_ms, lat, lon = (np.asarray(x, float) for x in (t_ms, lat, lon))
    r, dg = np.deg2rad, np.rad2deg
    jc = (t_ms / 86400000 + 2440587.5 - 2451545) / 36525
    L = (280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360
    M = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    e = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
    C = (np.sin(r(M)) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
         + np.sin(r(2 * M)) * (0.019993 - 0.000101 * jc) + np.sin(r(3 * M)) * 0.000289)
    om = 125.04 - 1934.136 * jc
    lam = L + C - 0.00569 - 0.00478 * np.sin(r(om))
    eps = (23 + (26 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60) / 60
           + 0.00256 * np.cos(r(om)))
    dec = np.arcsin(np.sin(r(eps)) * np.sin(r(lam)))
    y = np.tan(r(eps / 2)) ** 2
    eot = 4 * dg(y * np.sin(2 * r(L)) - 2 * e * np.sin(r(M))
                 + 4 * e * y * np.sin(r(M)) * np.cos(2 * r(L))
                 - 0.5 * y * y * np.sin(4 * r(L)) - 1.25 * e * e * np.sin(2 * r(M)))
    minutes = (t_ms % 86400000) / 60000
    ha = r(((minutes + eot + 4 * lon) % 1440) / 4 - 180)
    la = r(lat)
    cz = np.clip(np.sin(la) * np.sin(dec) + np.cos(la) * np.cos(dec) * np.cos(ha), -1, 1)
    zen = np.arccos(cz)
    ca = np.clip((np.sin(la) * cz - np.sin(dec)) / (np.cos(la) * np.sin(zen)), -1, 1)
    az = np.where(ha > 0, (dg(np.arccos(ca)) + 180) % 360, (540 - dg(np.arccos(ca))) % 360)
    return dg(zen), az


def _sanity():
    assert abs(glint_angle(35, 35, 120, 300)) < 1e-6          # especular
    assert abs(glint_angle(35, 35, 120, 120) - 70) < 1e-6     # retrodispersion
    assert abs(glint_angle(30, 0, 0, 0) - 30) < 1e-6          # nadir: tg = SZA
    assert abs(glint_angle(20, 10, 0, 180) - 10) < 1e-6       # mismo plano, opuesto


# --------------------------------------------------------------------------
# extraccion
# --------------------------------------------------------------------------
def _product_image(ee, cid, ids, prefix):
    col = ee.ImageCollection(cid).filter(ee.Filter.inList("system:index", ids))

    def prep(i):
        fm = i.select("Fmask")
        ok = fm.bitwiseAnd(FMASK_BAD).eq(0).And(fm.bitwiseAnd(FMASK_WATER).neq(0))
        ang = (i.select(ANG, [f"{prefix}_{a}" for a in ANG])
               .unmask(ANGLE_FILL, False).toFloat())
        # indice de la tesela en la lista: identifica la escena del pixel
        idx = (ee.Image.constant(ee.List(ids).indexOf(i.get("system:index")))
               .toInt16().rename(f"{prefix}_idx"))
        return (i.select(["B3"], [f"{prefix}_B3_chk"]).addBands(ang).addBands(idx)
                .updateMask(ok))

    return col.map(prep).mosaic(), col.first().select("B3").projection()


def _sample(ee, batch, ls, s2):
    out = []
    for d, sub in batch:
        lsi, _ = _product_image(ee, L30_ID, ls[ls.date == d].id.tolist(), "LS")
        s2i, proj = _product_image(ee, S30_ID, s2[s2.date == d].id.tolist(), "S2")
        valid = lsi.select("LS_B3_chk").mask().And(s2i.select("S2_B3_chk").mask())
        img = lsi.addBands(s2i).updateMask(valid)
        fc = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([r.lon, r.lat]), {"pid": int(r.pid)})
            for r in sub.itertuples()])
        smp = img.sampleRegions(collection=fc, properties=["pid"], scale=30,
                                projection=proj, geometries=False)
        out.append(smp.map(lambda f, d=d: f.set("date", d)))
    return ee.FeatureCollection(out).flatten()


def _scenes_meta(ee, cid, geom, prefixes):
    """Como _scenes de p12 (anio por anio, filtro por prefijo MGRS) pero con
    los angulos medios de cada escena."""
    filt = ee.Filter.Or(*[ee.Filter.stringStartsWith("system:index", p)
                          for p in prefixes])
    parts = []
    for y in range(int(START[:4]), int(END[:4]) + 1):
        a, b = max(START, f"{y}-01-01"), min(END, f"{y + 1}-01-01")
        if a >= b:
            continue
        col = (ee.ImageCollection(cid).filterBounds(geom).filterDate(a, b).filter(filt))
        props = {"id": col.aggregate_array("system:index"),
                 "t": col.aggregate_array("system:time_start")}
        for k, c in zip(MEAN, MCOLS[2:]):
            props[c] = col.aggregate_array(k)
        parts.append(pd.DataFrame(_retry(lambda: ee.Dictionary(props).getInfo(),
                                         f"escenas {y}")))
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(df):
        df["date"] = pd.to_datetime(df.t, unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    return df


def _attach_scene(got, sc, prefix):
    """Del indice de tesela muestreado a la escena y sus angulos medios."""
    rows = []
    for d, idx in zip(got.date, got[f"{prefix}_idx"]):
        ids = sc[sc.date == d].reset_index(drop=True)
        if pd.isna(idx) or int(idx) < 0 or int(idx) >= len(ids):
            rows.append({c: np.nan for c in MCOLS})
            continue
        r = ids.iloc[int(idx)]
        rows.append({"tile": r.id, **{c: r[c] for c in MCOLS[1:]}})
    m = pd.DataFrame(rows, index=got.index)
    for c in MCOLS:
        got[f"{prefix}_{c}"] = m[c]
    return got


def _lake_bounds(ee, name):
    hid, clip = LAKES[name]
    b = np.array(_lake_geometry(ee, hid, clip).bounds(100).coordinates().getInfo()[0])
    return {"lon_min": b[:, 0].min(), "lon_max": b[:, 0].max(),
            "lat_min": b[:, 1].min(), "lat_max": b[:, 1].max()}


def _append(df):
    df.reindex(columns=ACOLS).to_csv(CACHE, mode="a", header=not CACHE.exists(),
                                     index=False)


def extract():
    import ee
    ee.Initialize(project=GEE_PROJECT)
    raw = pd.read_csv(PROC / "multilake_hls_raw.csv.gz")
    pairs = raw[raw.pid >= 0]
    prev = pd.read_csv(CACHE) if CACHE.exists() else pd.DataFrame(columns=ACOLS)
    done = set(zip(prev.lake, prev.date))
    t0 = time.time()
    only = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--only=")]
    for name in LAKES:
        if only and name not in only:
            continue
        pl = pairs[pairs.lake == name]
        todo = sorted(d for d in pl.date.unique() if (name, d) not in done)
        if not todo:
            continue
        print(f"  {name}: {len(todo)} fechas  {time.time() - t0:.0f} s", flush=True)
        try:
            # puntos tomados de la cache por fecha (en Nicaragua algunos pid
            # tienen dos posiciones segun la corrida de p12)
            pts = pl.drop_duplicates(["date", "pid"])[["date", "pid", "lon", "lat"]]
            info = _retry(lambda: _lake_bounds(ee, name), f"{name} limites")
            pref = _tile_prefixes(info)
            mp = ee.Geometry.MultiPoint(
                pts.drop_duplicates(["lon", "lat"])[["lon", "lat"]].values.tolist())
            ls = _retry(lambda: _scenes_meta(ee, L30_ID, mp, pref), f"{name} L30")
            s2 = _retry(lambda: _scenes_meta(ee, S30_ID, mp, pref), f"{name} S30")
            for i in range(0, len(todo), BATCH):
                batch = [(d, pts[pts.date == d]) for d in todo[i:i + BATCH]]
                fc = _sample(ee, batch, ls, s2)
                feats = _retry(lambda: fc.getInfo()["features"], f"{name} muestreo")
                keys = ACOLS + ["LS_idx", "S2_idx"]
                got = pd.DataFrame([{k: f["properties"].get(k) for k in keys}
                                    for f in feats], columns=keys)
                got = _attach_scene(_attach_scene(got, ls, "LS"), s2, "S2")
                got["lake"] = name
                empty = [d for d, _ in batch if d not in set(got.date)]
                if empty:  # fecha sin pixeles validos: centinela pid = -1
                    got = pd.concat([got, pd.DataFrame({"lake": name, "date": empty,
                                                        "pid": -1})], ignore_index=True)
                _append(got)
                print(f"    {name}: {min(i + BATCH, len(todo))}/{len(todo)} fechas, "
                      f"{(got.pid >= 0).sum()} puntos  {time.time() - t0:.0f} s", flush=True)
        except Exception as e:  # un lago que falla no detiene a los demas
            print(f"    {name}: OMITIDO ({str(e)[:150]})", flush=True)


# --------------------------------------------------------------------------
# evaluacion
# --------------------------------------------------------------------------
def load_glint_pairs():
    raw, p = load_pairs()
    a = pd.read_csv(CACHE)
    a = a[a.pid >= 0].drop_duplicates(["lake", "date", "pid"])
    for c in [f"{s}_{x}" for s in ("LS", "S2") for x in ANG]:
        a[c] = a[c].where(a[c] > ANGLE_FILL, np.nan)
    m = p.drop(columns=["LS_VZA", "S2_VZA", "LS_SZA", "S2_SZA"]).merge(
        a, on=["lake", "date", "pid"], how="left")
    # bandas de angulos enmascaradas (todas las L30 al sur de ~0 grados en
    # Earth Engine): sol por pixel con NOAA a la hora de la escena, vision con
    # los angulos medios de la escena que dio el pixel
    for s in ("LS", "S2"):
        sz, sa = solar_position(m[f"{s}_t"], m.lat, m.lon)
        m[f"{s}_glint_pix"] = glint_angle(m[f"{s}_SZA"], m[f"{s}_VZA"],
                                          m[f"{s}_SAA"], m[f"{s}_VAA"])
        m[f"{s}_glint_mod"] = glint_angle(sz, m[f"{s}_mVZA"], sa, m[f"{s}_mVAA"])
        m[f"{s}_SZA_mod"] = sz
        m[f"{s}_SAA_mod"] = sa
        pix = m[f"{s}_glint_pix"].notna()
        m[f"{s}_glint"] = m[f"{s}_glint_pix"].where(pix, m[f"{s}_glint_mod"])
        m[f"{s}_angle_source"] = np.where(pix, "pixel", np.where(
            m[f"{s}_glint_mod"].notna(), "modeled", "missing"))
        for a in ("SZA", "VZA"):  # para la regresion
            src = m[f"{s}_{a}_mod"] if a == "SZA" else m[f"{s}_mVZA"]
            m[f"{s}_{a}"] = m[f"{s}_{a}"].where(pix, src)
    # mismo pixel que p12: B3 reproducido
    m["b3_match"] = ((m.LS_B3_chk - m.LS_B3).abs() < 1e-5) & \
                    ((m.S2_B3_chk - m.S2_B3).abs() < 1e-5)
    return raw, p, m


def subset(m, thr):
    if thr is None:
        s = m
    else:
        s = m[(m.LS_glint > thr) & (m.S2_glint > thr)]
    n = s.groupby(["lake", "date"]).pid.transform("size")
    return s[n >= MIN_PAIRS].reset_index(drop=True)


def pool(pp, seed=SEED):
    pp = pp.sample(frac=1, random_state=seed)
    pp = (pp[pp.groupby(["lake", "date"]).cumcount() < POOL_CAP]
          .sort_values(["lake", "date", "pid"]))
    return to_long(pp)


def ols_cluster(y, X, groups):
    """MCO con errores robustos agrupados (CR1)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    e = y - X @ beta
    meat = np.zeros((X.shape[1], X.shape[1]))
    g = pd.factorize(groups)[0]
    G = g.max() + 1
    for k in range(G):
        s = X[g == k].T @ e[g == k]
        meat += np.outer(s, s)
    n, kx = X.shape
    adj = G / (G - 1) * (n - 1) / (n - kx)
    se = np.sqrt(np.diag(adj * xtx_inv @ meat @ xtx_inv))
    r2 = 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return beta, se, r2, G


def regression(m, ycol, covs):
    d = m.dropna(subset=[ycol] + covs).reset_index(drop=True)
    fe = pd.get_dummies(d.lake, drop_first=True, dtype=float)
    X = np.column_stack([np.ones(len(d)), d[covs].values, fe.values])
    beta, se, r2, G = ols_cluster(d[ycol].values, X, (d.lake + "|" + d.date).values)
    out = {"n_pairs": int(len(d)), "n_clusters": int(G), "r2": round(float(r2), 4),
           "lake_fixed_effects": True}
    for j, c in enumerate(covs, start=1):
        out[c] = {"coef": float(f"{beta[j]:.4g}"), "se": float(f"{se[j]:.3g}"),
                  "t": round(float(beta[j] / se[j]), 2)}
    return out


def main():
    t0 = time.time()
    _sanity()
    banner("p17 -- DESTELLO SOLAR Y SEPARABILIDAD HLSL30 vs HLSS30")
    if "--eval" not in sys.argv:
        extract()
    if "--extract" in sys.argv:
        print("EXTRACCION TERMINADA")
        return
    t_ext = time.time() - t0
    raw, p, m = load_glint_pairs()
    n_ang = int((m.LS_glint.notna() & m.S2_glint.notna()).sum())
    print(f"  pares {len(m)}, con angulos de ambos {n_ang}, "
          f"B3 reproducido {m.b3_match.mean():.4f}")
    cover = {"n_pairs": int(len(m)), "n_pairs_with_glint": n_ang,
             "frac_b3_reproduced": round(float(m.b3_match.mean()), 4)}
    for s in ("LS", "S2"):
        cover[f"{s}_angle_source"] = m[f"{s}_angle_source"].value_counts().to_dict()
        mod = m[m[f"{s}_angle_source"] == "modeled"]
        cover[f"{s}_lakes_modeled"] = sorted(
            mod.lake.value_counts()[lambda x: x > 0].index.tolist())
    # validacion del modelo de angulos donde hay angulos por pixel
    val = {}
    for s in ("LS", "S2"):
        v = m[(m[f"{s}_angle_source"] == "pixel") & m[f"{s}_glint_mod"].notna()]
        eg = v[f"{s}_glint_mod"] - v[f"{s}_glint_pix"]
        ez = v[f"{s}_SZA_mod"] - v[f"{s}_SZA"]
        ea = (v[f"{s}_SAA_mod"] - v[f"{s}_SAA"] + 180) % 360 - 180
        val[s] = {"n_pairs": int(len(v)),
                  "sza_err_median_abs": round(float(ez.abs().median()), 3),
                  "sza_err_p95_abs": round(float(ez.abs().quantile(.95)), 3),
                  "saa_err_median_abs": round(float(ea.abs().median()), 3),
                  "glint_err_median": round(float(eg.median()), 3),
                  "glint_err_median_abs": round(float(eg.abs().median()), 3),
                  "glint_err_p95_abs": round(float(eg.abs().quantile(.95)), 3),
                  "glint_err_max_abs": round(float(eg.abs().max()), 3),
                  # coincidencia de la clasificacion por umbral
                  "agree_gt30": round(float(((v[f"{s}_glint_mod"] > 30)
                                             == (v[f"{s}_glint_pix"] > 30)).mean()), 4),
                  "agree_gt40": round(float(((v[f"{s}_glint_mod"] > 40)
                                             == (v[f"{s}_glint_pix"] > 40)).mean()), 4)}
    print("  validacion del modelo de angulos:", val)
    # solo pares con el mismo pixel que p12 y angulos completos
    m = m[m.b3_match & m.LS_glint.notna() & m.S2_glint.notna()].reset_index(drop=True)
    m["glint_diff"] = m.S2_glint - m.LS_glint
    m["glint_min"] = np.minimum(m.LS_glint, m.S2_glint)

    p12 = pd.read_csv(TIDY / "multilake_hls_separability.csv")
    tested_all = p12.loc[p12.tested == True, "lake"].tolist()  # noqa: E712

    rows, pooled = [], {}
    rng = np.random.default_rng(SEED)
    for thr in THRESHOLDS:
        lab = "all" if thr is None else f"glint_gt_{thr}"
        s = subset(m, thr)
        for lake in LAKES:
            sl = s[s.lake == lake]
            nd = sl.date.nunique()
            r = {"subset": lab, "threshold_deg": thr, "lake": lake, "n_dates": nd,
                 "n_pairs": int(len(sl)),
                 "frac_pairs_kept": round(len(sl) / max((m.lake == lake).sum(), 1), 4),
                 "glint_LS_median": round(float(sl.LS_glint.median()), 2) if len(sl) else np.nan,
                 "glint_S2_median": round(float(sl.S2_glint.median()), 2) if len(sl) else np.nan,
                 "glint_diff_median": round(float(sl.glint_diff.median()), 2) if len(sl) else np.nan,
                 "frac_glint_lt30_any": round(float((sl.glint_min < 30).mean()), 4) if len(sl) else np.nan,
                 "green_diff_median": round(float((sl.LS_B3 - sl.S2_B3).median()), 5) if len(sl) else np.nan,
                 "frac_LS_angles_modeled": round(float((sl.LS_angle_source == "modeled").mean()), 4) if len(sl) else np.nan,
                 "tested": nd >= MIN_DATES_TEST}
            if r["tested"]:
                d = to_long(sl)
                t = sep_test(d, FEATURES, d.date.values)
                r.update({"balanced_accuracy": t["balanced_accuracy"], "auc": t["auc"],
                          "accuracy": t["accuracy"], "chance": t["chance"]})
            rows.append(r)
            if r["tested"]:
                print(f"  {lab:12s} {lake:15s} fechas {nd:3d} pares {len(sl):5d} "
                      f"bacc {r['balanced_accuracy']:.3f}  gLS {r['glint_LS_median']:.1f} "
                      f"gS2 {r['glint_S2_median']:.1f}  {time.time() - t0:.0f} s", flush=True)
        sp = s[s.lake.isin(tested_all)]
        dl = pool(sp)
        res = {"n_lakes": int(sp.lake.nunique()), "n_lake_dates": int(dl.lake_date.nunique()),
               **sep_test(dl, FEATURES, dl.lake_date.values)}
        if thr is not None:
            # control de tamano: mismos pares por lago, elegidos al azar de todos
            full = subset(m, None)
            full = full[full.lake.isin(tested_all)]
            k = sp.groupby("lake").size()
            rnd = pd.concat([full[full.lake == lk].sample(int(n), random_state=int(rng.integers(1e9)))
                             for lk, n in k.items() if (full.lake == lk).sum() >= n])
            rnd = subset(rnd, None)
            dr = pool(rnd)
            res["size_matched_random"] = sep_test(dr, FEATURES, dr.lake_date.values)
        pooled[lab] = res
        print(f"  conjunta {lab}: {res}", flush=True)

    tidy = pd.DataFrame(rows)
    tidy.to_csv(TIDY / "glint_separability.csv", index=False)

    # p12 replicado en el subconjunto "all"
    a = tidy[(tidy.subset == "all") & tidy.tested].merge(
        p12[["lake", "balanced_accuracy"]], on="lake", suffixes=("", "_p12"))
    rep = {"n_lakes": int(len(a)), "max_abs_diff_bacc":
           round(float((a.balanced_accuracy - a.balanced_accuracy_p12).abs().max()), 4)}

    # correlaciones por lago (todos los pares)
    corr = {}
    for col in ("glint_LS_median", "glint_S2_median", "glint_diff_median"):
        rho, pv = spearmanr(a[col], a.balanced_accuracy)
        corr[f"bacc_vs_{col}"] = {"rho": round(float(rho), 3), "p": float(f"{pv:.3g}"),
                                  "n_lakes": int(len(a))}
    absd = m.groupby("lake").glint_diff.apply(lambda x: x.abs().median())
    gmin = m.groupby("lake").glint_min.median()
    for nm, ser in (("abs_glint_diff_median", absd), ("glint_min_median", gmin)):
        rho, pv = spearmanr(ser.loc[a.lake].values, a.balanced_accuracy)
        corr[f"bacc_vs_{nm}"] = {"rho": round(float(rho), 3), "p": float(f"{pv:.3g}"),
                                 "n_lakes": int(len(a))}
    print("  Spearman:", corr)

    # cambio por lago al quitar pares con destello
    paired = {}
    for thr in (30, 40):
        b = tidy[(tidy.subset == f"glint_gt_{thr}") & tidy.tested].merge(
            a[["lake", "balanced_accuracy"]], on="lake", suffixes=("", "_all"))
        dlt = b.balanced_accuracy - b.balanced_accuracy_all
        paired[f"glint_gt_{thr}"] = {
            "n_lakes_tested": int(len(b)),
            "bacc_median_all": round(float(b.balanced_accuracy_all.median()), 3),
            "bacc_median_low_glint": round(float(b.balanced_accuracy.median()), 3),
            "delta_median": round(float(dlt.median()), 3),
            "delta_min": round(float(dlt.min()), 3), "delta_max": round(float(dlt.max()), 3),
            "n_lakes_bacc_ge_0p8": int((b.balanced_accuracy >= 0.8).sum()),
            "n_lakes_bacc_lt_0p65": int((b.balanced_accuracy < 0.65).sum())}
    print("  por lago:", paired)

    # regresion por par de la diferencia en verde
    m["green_diff"] = m.LS_B3 - m.S2_B3
    pos = (m.LS_B3 > 0) & (m.S2_B3 > 0)
    m["green_logratio"] = np.where(pos, np.log(m.LS_B3.where(pos) / m.S2_B3.where(pos)), np.nan)
    m["LS_lowglint"] = (m.LS_glint < 30).astype(float)
    m["S2_lowglint"] = (m.S2_glint < 30).astype(float)
    reg = {}
    for y in ("green_diff", "green_logratio"):
        reg[y] = {
            "glint_linear": regression(m, y, ["LS_glint", "S2_glint"]),
            "glint_lt30_dummies": regression(m, y, ["LS_lowglint", "S2_lowglint"]),
            "glint_plus_vza": regression(m, y, ["LS_glint", "S2_glint", "LS_VZA", "S2_VZA"])}
    print("  regresion:", reg)
    gd = {}
    for thr in THRESHOLDS:
        s = subset(m, thr)
        gd["all" if thr is None else f"glint_gt_{thr}"] = {
            "n_pairs": int(len(s)),
            "green_diff_median": round(float(s.green_diff.median()), 5),
            "green_logratio_median": round(float(np.nanmedian(s.green_logratio)), 4)}

    dist = {f"{s}_glint_pct": {q: round(float(m[f"{s}_glint"].quantile(q / 100)), 1)
                               for q in (5, 25, 50, 75, 95)} for s in ("LS", "S2")}
    dist["frac_pairs_both_gt30"] = round(float(((m.LS_glint > 30) & (m.S2_glint > 30)).mean()), 4)
    dist["frac_pairs_both_gt40"] = round(float(((m.LS_glint > 40) & (m.S2_glint > 40)).mean()), 4)

    pa, p30, p40 = (pooled[k]["balanced_accuracy"] for k in ("all", "glint_gt_30", "glint_gt_40"))
    persists = p40 >= 0.65 and paired["glint_gt_40"]["bacc_median_low_glint"] >= 0.65
    interp = (f"Pooled balanced accuracy {pa:.3f} with all pairs, {p30:.3f} when both sensors "
              f"have glint angle > 30 deg and {p40:.3f} when > 40 deg "
              f"(size-matched random subsets: "
              f"{pooled['glint_gt_30']['size_matched_random']['balanced_accuracy']:.3f}, "
              f"{pooled['glint_gt_40']['size_matched_random']['balanced_accuracy']:.3f}). "
              + ("Separability persists in low-glint-risk pairs: sun glint does not explain it."
                 if persists else
                 "Separability drops toward chance in low-glint pairs: sun glint is a plausible driver."))
    print(" ", interp)

    out = {"definition": "theta_g = arccos(cos SZA cos VZA - sin SZA sin VZA cos(SAA - VAA)); "
                         "0 deg = specular",
           "thresholds_deg": [30, 40], "min_pairs_per_date": MIN_PAIRS,
           "min_dates_test": MIN_DATES_TEST, "pool_cap_per_date": POOL_CAP,
           "n_splits": N_SPLITS, "coverage": cover, "angle_model_validation": val, "glint_distribution": dist,
           "p12_replication": rep, "pooled": pooled, "per_lake_change": paired,
           "spearman_per_lake": corr, "green_difference_by_subset": gd,
           "regression": reg, "interpretation": interp, "separability_persists": bool(persists),
           "runtime_s": {"extraction": round(t_ext), "total": round(time.time() - t0)}}
    write_json(out, MET / "glint.json")


if __name__ == "__main__":
    main()
