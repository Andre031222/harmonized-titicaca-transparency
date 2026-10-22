"""
p16_lake_maps.py -- mapas de Secchi de todo el lago con HLS v2.0 (aplicacion).

Compuestos de mediana por pixel de la temporada seca 2024 (1 jul - 31 oct;
la ultima campana de campo fue ago-sep 2024) de HLSL30 (Landsat 8/9) y
HLSS30 (Sentinel-2), Fmask (nube, adyacente, sombra, nieve) y bit de agua.
Se aplica el mismo control de calidad que en los match-ups (qc_pass) y se
predice Secchi con RF entrenados en los 815 match-ups: modelo conjunto y
modelos de un solo sensor. La diferencia S2 - LS del mismo modelo conjunto
es la consecuencia de que los sensores no sean intercambiables.

Entradas : GEE (NASA/HLS/HLSL30/v002, NASA/HLS/HLSS30/v002),
           data/lake_boundary/titicaca.gpkg, data/processed/analysis_dataset.csv,
           results/tidy/location_units.csv, results/metrics/uncertainty.json
Salidas  : data/processed/lake_composites/hls_{LS,S2}_2024_dry.tif (bandas x1e4 + n claras)
           data/processed/lake_composites/lake_secchi_maps_2024.tif (mapas predichos)
           results/tidy/lake_map_{zones,field,summary,hist}.csv
           results/metrics/lake_maps.json
Uso      : python p16_lake_maps.py [--redownload]
"""
import json
import sys

import numpy as np
import pandas as pd

from p00_config import (BANDS, FEATURES, GEE_PROJECT, HLS_L30, HLS_S30, MET, PROC,
                        QC_SWIR1_MAX, ROOT, SECCHI_CLIP_LO, SEED, TIDY, UTM,
                        add_indices, banner, make_rf, metrics, qc_pass, to_utm,
                        write_json)

OUT = PROC / "lake_composites"
OUT.mkdir(parents=True, exist_ok=True)
LAKE = ROOT / "data/lake_boundary/titicaca.gpkg"
START, END = "2024-07-01", "2024-11-01"     # END exclusivo: hasta el 31 oct
FIELD_CAMPAIGN = "2024"
SHRINK_M = 300
SCALE = 90
NATIVE = 30
NTILE = 3                                    # teselas por lado para la descarga
NODATA = -32768
FMASK_BAD = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4)
FMASK_WATER = 1 << 5
SENSORS = {"LS": ("NASA/HLS/HLSL30/v002", HLS_L30),
           "S2": ("NASA/HLS/HLSS30/v002", HLS_S30)}
SENSOR_LABEL = {"LS": "Landsat 8/9", "S2": "Sentinel-2"}


def comp_path(s):
    return OUT / f"hls_{s}_2024_dry.tif"


# --- descarga GEE ------------------------------------------------------------
def lake_inner():
    import geopandas as gpd
    g = gpd.read_file(LAKE).to_crs(UTM)
    inner = g.geometry.buffer(-SHRINK_M).union_all()
    return inner


def download(redownload=False):
    import ee
    import requests
    import rasterio
    from rasterio.merge import merge
    from shapely.geometry import box, mapping

    ee.Initialize(project=GEE_PROJECT)
    inner = lake_inner()
    x0, y0, x1, y1 = inner.bounds
    x0, y0 = np.floor(x0 / SCALE) * SCALE, np.floor(y0 / SCALE) * SCALE
    x1, y1 = np.ceil(x1 / SCALE) * SCALE, np.ceil(y1 / SCALE) * SCALE
    nx, ny = int((x1 - x0) / SCALE), int((y1 - y0) / SCALE)
    # geometria del lago en EE (simplificada a 30 m para aligerar la peticion)
    lake_ee = ee.Geometry(mapping(inner.simplify(NATIVE)), UTM, False)
    proj30 = ee.Projection(UTM).atScale(NATIVE)
    info = {}
    for s, (name, bmap) in SENSORS.items():
        if comp_path(s).exists() and not redownload:
            print(f"  {s}: ya descargado ({comp_path(s).name})")
            continue
        src = [bmap[b] for b in BANDS]
        col = (ee.ImageCollection(name).filterBounds(lake_ee).filterDate(START, END)
               .filter(ee.Filter.stringStartsWith("system:index", "T19")))

        def clean(i):
            f = i.select("Fmask")
            ok = f.bitwiseAnd(FMASK_BAD).eq(0).And(f.bitwiseAnd(FMASK_WATER).neq(0))
            return i.select(src, BANDS).updateMask(ok)

        masked = col.map(clean)
        # mediana a 30 m (Fmask evaluado a resolucion nativa, vecino mas
        # proximo) y luego media de los pixeles de 30 m dentro de cada 90 m
        med = masked.median().setDefaultProjection(proj30)
        cnt = masked.select("B2").count().rename("n_clear").setDefaultProjection(proj30)
        img = (med.multiply(10000).addBands(cnt.toFloat())
               .reduceResolution(ee.Reducer.mean(), maxPixels=64)
               .reproject(crs=UTM, scale=SCALE)
               .round().toInt16().clip(lake_ee).unmask(NODATA, False))
        n_scenes = col.size().getInfo()
        info[s] = {"collection": name, "n_granules": n_scenes}
        print(f"  {s}: {n_scenes} granulos HLS en {START}..{END}")
        tiles = []
        xs = np.linspace(0, nx, NTILE + 1).astype(int)
        ys = np.linspace(0, ny, NTILE + 1).astype(int)
        for i in range(NTILE):
            for j in range(NTILE):
                tx0, tx1 = x0 + xs[i] * SCALE, x0 + xs[i + 1] * SCALE
                ty1, ty0 = y1 - ys[j] * SCALE, y1 - ys[j + 1] * SCALE
                if not box(tx0, ty0, tx1, ty1).intersects(inner):
                    continue
                fn = OUT / f"_tile_{s}_{i}{j}.tif"
                if not fn.exists():
                    url = img.getDownloadURL({
                        "crs": UTM,
                        "crs_transform": [SCALE, 0, float(tx0), 0, -SCALE, float(ty1)],
                        "dimensions": f"{xs[i + 1] - xs[i]}x{ys[j + 1] - ys[j]}",
                        "format": "GEO_TIFF"})
                    r = requests.get(url, timeout=900)
                    r.raise_for_status()
                    fn.write_bytes(r.content)
                    print(f"    tesela {i}{j}: {len(r.content) / 1e6:.1f} MB")
                tiles.append(fn)
        srcs = [rasterio.open(t) for t in tiles]
        arr, tr = merge(srcs, nodata=NODATA)
        prof = srcs[0].profile.copy()
        for x in srcs:
            x.close()
        prof.update(height=arr.shape[1], width=arr.shape[2], transform=tr,
                    nodata=NODATA, compress="deflate", count=arr.shape[0])
        with rasterio.open(comp_path(s), "w", **prof) as dst:
            dst.write(arr)
            dst.descriptions = tuple(BANDS + ["n_clear"])
        for t in tiles:
            t.unlink()
    if info:
        (OUT / "download_info.json").write_text(json.dumps(info, indent=2))


# --- prediccion --------------------------------------------------------------
def read_comp(s):
    import rasterio
    with rasterio.open(comp_path(s)) as r:
        a = r.read().astype("float64")
        tr, crs, shape = r.transform, r.crs, (r.height, r.width)
    nd = (a == NODATA).any(axis=0) | (a[-1] <= 0)
    bands = a[:-1] / 10000.0
    bands[:, nd] = np.nan
    n_clear = np.where(nd, np.nan, a[-1])
    return bands, n_clear, tr, crs, shape


def fit(d):
    from sklearn.preprocessing import RobustScaler
    sc = RobustScaler().fit(d[FEATURES].values)
    m = make_rf()
    m.fit(sc.transform(d[FEATURES].values), d.secchi.values)
    return sc, m


def predict(model, X):
    sc, m = model
    return np.clip(m.predict(sc.transform(X)), SECCHI_CLIP_LO, None)


def bin_halfwidth(unc):
    """Semi-ancho al 90 % por tramo de Secchi predicho (Mondrian, p07)."""
    labs = unc["mondrian_bins"]["bins_m"]
    w = unc["coverage90_by_predicted_bin"]["mondrian"]
    edges = []
    for lab in labs[:-1]:
        edges.append(float(lab.replace("<", "").split("-")[-1]))
    half = np.array([w[l]["mean_width"] / 2 for l in labs])
    return np.array(edges), half, labs


def r3(x, k=3):
    return None if x is None or not np.isfinite(x) else round(float(x), k)


def main():
    import rasterio
    from scipy.spatial import cKDTree

    if not (comp_path("LS").exists() and comp_path("S2").exists()) or "--redownload" in sys.argv:
        banner("p16 | descarga de compuestos HLS (GEE)")
        download("--redownload" in sys.argv)

    banner("p16 | mapas de Secchi temporada seca 2024")
    d = pd.read_csv(PROC / "analysis_dataset.csv", dtype={"campaign": str})
    d = d.reset_index(drop=True)
    unc = json.loads((MET / "uncertainty.json").read_text())
    half_marg = unc["marginal"]["split"]["90"]["mean_width_m"] / 2
    half_mond = unc["marginal"]["mondrian"]["90"]["mean_width_m"] / 2
    edges, half_bin, bin_labs = bin_halfwidth(unc)

    comps, nclr, valid_raw = {}, {}, {}
    for s in SENSORS:
        bands, n_clear, tr, crs, shape = read_comp(s)
        comps[s], nclr[s] = bands, n_clear
    # pixeles con dato en el lago (union) y con QC en cada sensor
    pix = {}
    for s in SENSORS:
        b = comps[s]
        df = pd.DataFrame({k: b[i].ravel() for i, k in enumerate(BANDS)})
        has = df[BANDS].notna().all(axis=1)
        ok = has & qc_pass(df.fillna(0)).values
        valid_raw[s] = has.values
        df = add_indices(df)
        pix[s] = (df, ok.values)
    lake_px = valid_raw["LS"] | valid_raw["S2"]

    # modelos: conjunto, solo LS, solo S2; y las mismas versiones sin 2024
    models = {"pooled": fit(d), "LS_only": fit(d[d.sensor == "LS"]),
              "S2_only": fit(d[d.sensor == "S2"])}
    no24 = d[d.campaign != FIELD_CAMPAIGN]
    models_no24 = {"pooled": fit(no24), "LS_only": fit(no24[no24.sensor == "LS"]),
                   "S2_only": fit(no24[no24.sensor == "S2"])}

    def pred_map(model, s):
        df, ok = pix[s]
        out = np.full(len(df), np.nan)
        out[ok] = predict(model, df.loc[ok, FEATURES].values)
        return out.reshape(shape)

    maps = {"LS_pooled": pred_map(models["pooled"], "LS"),
            "S2_pooled": pred_map(models["pooled"], "S2"),
            "LS_single": pred_map(models["LS_only"], "LS"),
            "S2_single": pred_map(models["S2_only"], "S2")}
    maps["diff_pooled"] = maps["S2_pooled"] - maps["LS_pooled"]
    maps["diff_single"] = maps["S2_single"] - maps["LS_single"]
    maps_no24 = {"LS_pooled": pred_map(models_no24["pooled"], "LS"),
                 "S2_pooled": pred_map(models_no24["pooled"], "S2"),
                 "LS_single": pred_map(models_no24["LS_only"], "LS"),
                 "S2_single": pred_map(models_no24["S2_only"], "S2")}

    # semi-ancho Mondrian por pixel segun el tramo del valor predicho
    def hw(p):
        h = half_bin[np.clip(np.searchsorted(edges, p, side="right"), 0, len(half_bin) - 1)]
        return np.where(np.isfinite(p), h, np.nan)

    # extrapolacion: alguna feature fuera del rango [min, max] de entrenamiento
    def outside(s, train):
        df, ok = pix[s]
        lo, hi = train[FEATURES].min().values, train[FEATURES].max().values
        X = df[FEATURES].values
        o = ((X < lo) | (X > hi)).any(axis=1) & ok
        return o.reshape(shape)

    out_pool = {s: outside(s, d) for s in SENSORS}

    # zonas: la de la ubicacion de campo mas cercana
    loc = pd.read_csv(TIDY / "location_units.csv")
    lx, ly = to_utm(loc.lat.values, loc.lon.values)
    rows, cols = np.indices(shape)
    px, py = rasterio.transform.xy(tr, rows.ravel(), cols.ravel())
    px, py = np.asarray(px), np.asarray(py)
    dist, k = cKDTree(np.c_[lx, ly]).query(np.c_[px, py])
    zone = np.where(lake_px, loc.zona.values[k], None).reshape(shape)
    dist_km = (dist / 1000).reshape(shape)

    # --- resumenes -----------------------------------------------------------
    both = np.isfinite(maps["diff_pooled"])
    dp, ds = maps["diff_pooled"][both], maps["diff_single"][both]
    hw_pair = np.maximum(hw(maps["LS_pooled"]), hw(maps["S2_pooled"]))[both]

    def q(x):
        return [r3(v) for v in np.percentile(x, [25, 50, 75])]

    px_km2 = (SCALE / 1000) ** 2
    cov = {}
    for s in SENSORS:
        ok = pix[s][1].reshape(shape)
        nc = nclr[s][valid_raw[s].reshape(shape)]
        cov[s] = {"lake_pixels_with_data": int(valid_raw[s].sum()),
                  "pixels_pass_qc": int(ok.sum()),
                  "pct_of_lake_pixels_pass_qc": r3(100 * ok.sum() / lake_px.sum(), 1),
                  "area_pass_qc_km2": r3(ok.sum() * px_km2, 0),
                  "qc_fail_pixels": int(valid_raw[s].sum() - ok.sum()),
                  "n_clear_obs_median": r3(np.nanmedian(nc), 1),
                  "n_clear_obs_iqr": [r3(v, 1) for v in np.nanpercentile(nc, [25, 75])],
                  "pct_pixels_outside_training_range": r3(100 * out_pool[s].sum() / ok.sum(), 1)}

    summary = {}
    for key in ["LS_pooled", "S2_pooled", "LS_single", "S2_single"]:
        v = maps[key][np.isfinite(maps[key])]
        summary[key] = {"n_pixels": int(v.size), "mean": r3(v.mean()),
                        "median": r3(np.median(v)), "p05": r3(np.percentile(v, 5)),
                        "p95": r3(np.percentile(v, 95)),
                        "pct_above_training_max": r3(100 * (v > d.secchi.max()).mean(), 2)}
    diff = {}
    for key, x in [("pooled", dp), ("single", ds)]:
        diff[key] = {"n_pixels_both": int(x.size), "mean": r3(x.mean()), "sd": r3(x.std()),
                     "median": r3(np.median(x)), "iqr": q(x),
                     "p05_p95": [r3(v) for v in np.percentile(x, [5, 95])],
                     "mean_abs": r3(np.abs(x).mean()),
                     "pct_abs_gt_marginal_halfwidth90": r3(100 * (np.abs(x) > half_marg).mean(), 2),
                     "pct_abs_gt_mondrian_bin_halfwidth90": r3(100 * (np.abs(x) > hw_pair).mean(), 2),
                     "pct_abs_gt_1m": r3(100 * (np.abs(x) > 1).mean(), 1),
                     "pct_abs_gt_2m": r3(100 * (np.abs(x) > 2).mean(), 1),
                     "pct_positive": r3(100 * (x > 0).mean(), 1)}
    # correlacion espacial entre los mapas LS y S2 del modelo conjunto
    r_maps = float(np.corrcoef(maps["LS_pooled"][both], maps["S2_pooled"][both])[0, 1])

    # zonas
    zrows = []
    zflat = zone.ravel()
    for z in ["BAHIA PUNO", "LAGO MENOR", "LAGO MAYOR"]:
        mz = zflat == z
        for key in ["LS_pooled", "S2_pooled", "LS_single", "S2_single",
                    "diff_pooled", "diff_single"]:
            v = maps[key].ravel()[mz]
            v = v[np.isfinite(v)]
            zrows.append({"zona": z, "map": key, "n_pixels": int(v.size),
                          "area_km2": round(v.size * px_km2, 1),
                          "mean": v.mean() if v.size else np.nan,
                          "median": np.median(v) if v.size else np.nan,
                          "q25": np.percentile(v, 25) if v.size else np.nan,
                          "q75": np.percentile(v, 75) if v.size else np.nan})
        f = d[(d.zona == z) & (d.campaign == FIELD_CAMPAIGN)].drop_duplicates("event")
        zrows.append({"zona": z, "map": "field_2024", "n_pixels": int(len(f)),
                      "area_km2": np.nan, "mean": f.secchi.mean(),
                      "median": f.secchi.median(), "q25": f.secchi.quantile(.25),
                      "q75": f.secchi.quantile(.75)})
    zt = pd.DataFrame(zrows)
    zt.to_csv(TIDY / "lake_map_zones.csv", index=False, float_format="%.4f")

    # lecturas de campo 2024: media de los pixeles validos en 3 x 3 (270 m)
    f = (d[d.campaign == FIELD_CAMPAIGN].drop_duplicates("event")
         .reset_index(drop=True))
    fx, fy = to_utm(f.lat.values, f.lon.values)
    rr, cc = rasterio.transform.rowcol(tr, fx, fy)
    rr, cc = np.asarray(rr), np.asarray(cc)

    def sample(a):
        out = np.full(len(f), np.nan)
        for i, (r, c) in enumerate(zip(rr, cc)):
            w = a[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2]
            w = w[np.isfinite(w)]
            if w.size:
                out[i] = w.mean()
        return out

    frows, fmet = [], {}
    for train_set, mp in [("all_campaigns", maps), ("without_2024", maps_no24)]:
        for key in ["LS_pooled", "S2_pooled", "LS_single", "S2_single"]:
            p = sample(mp[key])
            sensor, model = key.split("_")
            for i in range(len(f)):
                frows.append({"location": f.location[i], "zona": f.zona[i],
                              "sampling_date": f.sampling_date[i], "lat": f.lat[i],
                              "lon": f.lon[i], "secchi": f.secchi[i],
                              "composite": sensor, "composite_label": SENSOR_LABEL[sensor],
                              "model": model, "training": train_set,
                              "predicted": p[i]})
            ok = np.isfinite(p)
            m = metrics(f.secchi.values[ok], p[ok]) if ok.sum() > 2 else {}
            fmet.setdefault(train_set, {})[key] = m
    ft = pd.DataFrame(frows)
    ft.to_csv(TIDY / "lake_map_field.csv", index=False, float_format="%.4f")

    # histograma de la diferencia (para la figura)
    bins = np.arange(np.floor(np.percentile(np.r_[dp, ds], 0.1)),
                     np.ceil(np.percentile(np.r_[dp, ds], 99.9)) + 0.25, 0.25)
    hrows = []
    for key, x in [("pooled", dp), ("single", ds)]:
        h, e = np.histogram(x, bins)
        hrows += [{"model": key, "lo": e[i], "hi": e[i + 1], "mid": (e[i] + e[i + 1]) / 2,
                   "n": int(h[i]), "frac": h[i] / x.size} for i in range(len(h))]
    pd.DataFrame(hrows).to_csv(TIDY / "lake_map_diff_hist.csv", index=False,
                               float_format="%.5f")

    # resumen largo para el texto y la figura
    srows = []
    for key, v in summary.items():
        for st in ["mean", "median", "p05", "p95"]:
            srows.append({"quantity": key, "stat": st, "value": v[st]})
    for key, v in diff.items():
        for st in ["mean", "median", "pct_abs_gt_marginal_halfwidth90"]:
            srows.append({"quantity": f"diff_{key}", "stat": st, "value": v[st]})
        srows.append({"quantity": f"diff_{key}", "stat": "q25", "value": v["iqr"][0]})
        srows.append({"quantity": f"diff_{key}", "stat": "q75", "value": v["iqr"][2]})
    srows.append({"quantity": "halfwidth90", "stat": "marginal_split", "value": r3(half_marg)})
    srows.append({"quantity": "halfwidth90", "stat": "marginal_mondrian", "value": r3(half_mond)})
    pd.DataFrame(srows).to_csv(TIDY / "lake_map_summary.csv", index=False)

    # GeoTIFF de los mapas para la figura
    names = ["LS_pooled", "S2_pooled", "diff_pooled", "LS_single", "S2_single",
             "diff_single", "outside_range_LS", "outside_range_S2", "n_clear_LS",
             "n_clear_S2"]
    stack = np.stack([maps[n] for n in names[:6]]
                     + [np.where(pix[s][1].reshape(shape), out_pool[s], np.nan).astype(float)
                        for s in SENSORS] + [nclr["LS"], nclr["S2"]]).astype("float32")
    prof = {"driver": "GTiff", "dtype": "float32", "count": len(names),
            "height": shape[0], "width": shape[1], "transform": tr, "crs": crs,
            "nodata": np.nan, "compress": "deflate"}
    with rasterio.open(OUT / "lake_secchi_maps_2024.tif", "w", **prof) as dst:
        dst.write(stack)
        dst.descriptions = tuple(names)

    dl = OUT / "download_info.json"
    res = {
        "design": ("Per-pixel median composites of HLS v2.0 over Lake Titicaca, dry season "
                   f"{START} to 2024-10-31; Fmask bits 1-4 masked, water bit 5 required; "
                   f"median at {NATIVE} m then mean-aggregated to {SCALE} m (UTM 19S); lake "
                   f"polygon (HydroLAKES) shrunk {SHRINK_M} m inward. QC as match-ups "
                   f"(SWIR1 <= {QC_SWIR1_MAX}, NDWI > 0). RF make_rf() on RobustScaler "
                   "features, trained on all 815 match-ups (pooled) or on one sensor; "
                   "predictions clipped at the Secchi floor."),
        "period": [START, "2024-10-31"], "scale_m": SCALE, "shrink_m": SHRINK_M,
        "granules": json.loads(dl.read_text()) if dl.exists() else None,
        "training_n": {"pooled": int(len(d)), "LS_only": int((d.sensor == "LS").sum()),
                       "S2_only": int((d.sensor == "S2").sum())},
        "training_secchi_range_m": [float(d.secchi.min()), float(d.secchi.max())],
        "lake_pixels": int(lake_px.sum()), "lake_area_km2": r3(lake_px.sum() * px_km2, 0),
        "coverage": cov,
        "maps": summary,
        "difference_S2_minus_LS": diff,
        "pearson_r_LS_vs_S2_pooled_maps": r3(r_maps),
        "halfwidth90_m": {"marginal_split": r3(half_marg), "marginal_mondrian": r3(half_mond),
                          "mondrian_by_predicted_bin": {l: r3(h) for l, h in zip(bin_labs, half_bin)},
                          "note": ("Half-width = mean 90 % interval width / 2 from p07 "
                                   "(leave-one-campaign-out conformal); per-pixel interval = "
                                   "prediction +/- half-width of its predicted-Secchi bin.")},
        "zones": {z: {r["map"]: {"mean": r3(r["mean"]), "median": r3(r["median"]),
                              "n": int(r.n_pixels)}
                      for _, r in zt[zt.zona == z].iterrows()}
                  for z in zt.zona.unique()},
        "nearest_location_distance_km": {"median": r3(np.nanmedian(dist_km[lake_px.reshape(shape)]), 1),
                                         "max": r3(np.nanmax(dist_km[lake_px.reshape(shape)]), 1)},
        "field_2024": {"n_readings": int(len(f)),
                       "dates": [f.sampling_date.min(), f.sampling_date.max()],
                       "sample": "mean of valid 90 m pixels in a 3 x 3 window (270 m) around each reading",
                       "metrics": fmet,
                       "note": ("'all_campaigns' models include the 2024 match-ups in training "
                                "(not independent); 'without_2024' models exclude the 2024 campaign "
                                "(out-of-campaign check).")},
    }
    write_json(res, MET / "lake_maps.json")
    print(json.dumps({k: res[k] for k in ["coverage", "maps", "difference_S2_minus_LS",
                                          "pearson_r_LS_vs_S2_pooled_maps", "halfwidth90_m"]},
                     indent=1))
    print(json.dumps(res["field_2024"]["metrics"], indent=1))
    print(zt.to_string())


if __name__ == "__main__":
    main()
