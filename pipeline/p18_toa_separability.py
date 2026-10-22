"""
p18_toa_separability.py -- la separabilidad Landsat / Sentinel-2 de HLS sobre
lagos, ¿la introduce la correccion atmosferica de tierra o ya esta en el techo
de la atmosfera (TOA)?

Usa los MISMOS pares del mismo dia que p12 (lago, fecha, punto; cache
multilake_hls_raw.csv.gz). Para un subconjunto fijo con semilla (hasta
N_PER_DATE puntos por lago-fecha) extrae en esa fecha:
  * Landsat 8/9 Collection 2 Tier 1 TOA (B2-B7), mascara QA_PIXEL (nube
    dilatada, cirros, nube, sombra, nieve);
  * Sentinel-2 L1C (COPERNICUS/S2_HARMONIZED / 10000; B2,B3,B4,B8A,B11,B12),
    mascara S2_CLOUD_PROBABILITY < 30 (QA60 donde no hay probabilidad),
    agregado por media a la grilla Landsat de 30 m.
Mosaico de las escenas del dia sobre los puntos; muestra en la grilla de
Landsat.

Evaluacion (igual que p12: RF 300, 12 features, GroupKFold(5) por fecha):
por lago HLS (reflectancia de superficie) vs TOA crudo vs TOA con los
coeficientes de banda de HLS aplicados a S2, todos sobre exactamente los mismos
pares; controles negativos por mitades de fechas; prueba conjunta por
lago-fecha y dejando lagos fuera; relacion con la claridad (verde mediano S2
de HLS).

    python pipeline/p18_toa_separability.py          # extrae (reanudable) y evalua
    python pipeline/p18_toa_separability.py --eval   # solo evalua la cache

Salidas: data/processed/multilake_toa.csv.gz (cache)
         results/tidy/toa_separability.csv
         results/metrics/toa.json
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from p00_config import (BANDS, FEATURES, GEE_PROJECT, HLS_MSI_TO_OLI, MET,
                        MSI_SOURCE, PROC, SEED, TIDY, add_indices, banner,
                        write_json)
from p12_multilake_hls import (LAKES, MIN_DATES_TEST, MIN_PAIRS, N_SPLITS,
                               POOL_CAP, load_pairs, sep_test)

CACHE = PROC / "multilake_toa.csv.gz"
L8_ID, L9_ID = "LANDSAT/LC08/C02/T1_TOA", "LANDSAT/LC09/C02/T1_TOA"
S2_ID, S2P_ID = "COPERNICUS/S2_HARMONIZED", "COPERNICUS/S2_CLOUD_PROBABILITY"
N_PER_DATE = 25        # puntos por lago-fecha (subconjunto con semilla)
MAX_PROB = 30          # probabilidad de nube S2 maxima (%)
BATCH = 6              # fechas por llamada getInfo
WORKERS = 4            # llamadas concurrentes a Earth Engine
LS_SRC = {"B2": "B2", "B3": "B3", "B4": "B4", "B8": "B5", "B11": "B6", "B12": "B7"}
S2_SRC = {"B2": "B2", "B3": "B3", "B4": "B4", "B8": "B8A", "B11": "B11", "B12": "B12"}
# QA_PIXEL C2: 1 nube dilatada, 2 cirros, 3 nube, 4 sombra, 5 nieve
QA_BAD = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5)
COLS = (["lake", "date", "pid"] + [f"LS_{b}" for b in BANDS]
        + [f"S2_{b}" for b in BANDS]
        + ["ls_time", "s2_time", "ls_sun_elev", "s2_sza", "n_ls_img", "n_s2_img"])


# --------------------------------------------------------------------------
# extraccion
# --------------------------------------------------------------------------
def subset_pairs():
    """Subconjunto fijo de los pares de p12: hasta N_PER_DATE por lago-fecha."""
    _, p = load_pairs()
    p = p.sample(frac=1, random_state=SEED)
    p = p[p.groupby(["lake", "date"]).cumcount() < N_PER_DATE]
    return p.sort_values(["lake", "date", "pid"]).reset_index(drop=True)


def _retry(fn, what):
    for attempt in range(4):
        try:
            return fn()
        except Exception as e:  # cuotas y cortes de red
            print(f"    reintento {attempt + 1} ({what}): {str(e)[:150]}", flush=True)
            time.sleep(15 * (attempt + 1))
    raise RuntimeError(f"Earth Engine no respondio: {what}")


def _date_fc(ee, date, pts):
    t0 = ee.Date(date)
    t1 = t0.advance(1, "day")
    geom = ee.Geometry.MultiPoint(pts[["lon", "lat"]].values.tolist())
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([r.lon, r.lat]),
                                          {"pid": int(r.pid)})
                               for r in pts.itertuples()])
    ls = (ee.ImageCollection(L8_ID).merge(ee.ImageCollection(L9_ID))
          .filterBounds(geom).filterDate(t0, t1))
    s2 = ee.ImageCollection(S2_ID).filterBounds(geom).filterDate(t0, t1)
    prob = ee.ImageCollection(S2P_ID).filterBounds(geom).filterDate(t0, t1)

    def prep_ls(i):
        ok = i.select("QA_PIXEL").bitwiseAnd(QA_BAD).eq(0)
        return (i.select(list(LS_SRC.values()), [f"LS_{b}" for b in LS_SRC])
                .updateMask(ok))

    def sampled():
        lsproj = ls.first().select("B3").projection()
        lsi = ls.map(prep_ls).mosaic()
        s2m = s2.mosaic()
        qa_ok = s2m.select("QA60").eq(0)
        ok = ee.Image(ee.Algorithms.If(
            prob.size().gt(0),
            prob.mosaic().select("probability").lt(MAX_PROB).unmask(qa_ok),
            qa_ok))
        s2i = (s2m.select(list(S2_SRC.values()), [f"S2_{b}" for b in S2_SRC])
               .divide(10000).updateMask(ok)
               .setDefaultProjection(s2.first().select("B2").projection())
               .reduceResolution(ee.Reducer.mean(), False, 64)
               .reproject(lsproj))
        img = lsi.addBands(s2i)
        valid = lsi.select("LS_B3").mask().And(s2i.select("S2_B3").mask())
        smp = img.updateMask(valid).sampleRegions(
            collection=fc, properties=["pid"], scale=30, projection=lsproj,
            geometries=False)
        props = {"date": date,
                 "ls_time": ls.first().get("system:time_start"),
                 "s2_time": s2.first().get("system:time_start"),
                 "ls_sun_elev": ls.first().get("SUN_ELEVATION"),
                 "s2_sza": s2.first().get("MEAN_SOLAR_ZENITH_ANGLE"),
                 "n_ls_img": ls.size(), "n_s2_img": s2.size()}
        return smp.map(lambda f: f.set(props))

    return ee.FeatureCollection(ee.Algorithms.If(
        ls.size().gt(0).And(s2.size().gt(0)), sampled(),
        ee.FeatureCollection([])))


def _fetch(ee, lake, batch, sub):
    fcs = [_date_fc(ee, d, sub[sub.date == d]) for d in batch]
    feats = _retry(lambda: ee.FeatureCollection(fcs).flatten().getInfo()["features"],
                   f"{lake} {batch[0]}")
    rows = [{k: f["properties"].get(k) for k in COLS[1:]} for f in feats]
    got = pd.DataFrame(rows, columns=COLS[1:])
    empty = [d for d in batch if d not in set(got.date)]
    if empty:  # fechas sin datos TOA: centinela pid = -1
        got = pd.concat([got, pd.DataFrame({"date": empty, "pid": -1})],
                        ignore_index=True)
    got["lake"] = lake
    return got.reindex(columns=COLS)


def extract(sub):
    import ee
    ee.Initialize(project=GEE_PROJECT)
    prev = pd.read_csv(CACHE) if CACHE.exists() else pd.DataFrame(columns=COLS)
    seen = set(zip(prev.lake, prev.date))
    jobs = []
    for lake in LAKES:
        dates = [d for d in sorted(sub[sub.lake == lake].date.unique())
                 if (lake, d) not in seen]
        for i in range(0, len(dates), BATCH):
            jobs.append((lake, dates[i:i + BATCH]))
    print(f"  {len(jobs)} lotes pendientes ({sum(len(b) for _, b in jobs)} fechas)")
    t0 = time.time()
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(_fetch, ee, lake, b,
                          sub[(sub.lake == lake) & sub.date.isin(b)])
                for lake, b in jobs]
        for k, (f, (lake, b)) in enumerate(zip(futs, jobs)):
            try:
                got = f.result()
            except Exception as e:  # el lote queda pendiente para otra corrida
                print(f"    {lake} {b[0]}: OMITIDO ({str(e)[:120]})", flush=True)
                continue
            got.to_csv(CACHE, mode="a", header=not CACHE.exists(), index=False)
            print(f"    [{k + 1}/{len(jobs)}] {lake} {b[0]}..{b[-1]}: "
                  f"{(got.pid >= 0).sum()} pares  {time.time() - t0:.0f} s", flush=True)


# --------------------------------------------------------------------------
# evaluacion
# --------------------------------------------------------------------------
def to_long(p, prefix_ls, prefix_s2, bandpass=False):
    keep = ["lake", "date", "pid"]
    parts = []
    for s, pre in (("LS", prefix_ls), ("S2", prefix_s2)):
        t = p[keep + [f"{pre}{b}" for b in BANDS]].rename(
            columns={f"{pre}{b}": b for b in BANDS})
        if s == "S2" and bandpass:  # coeficientes de banda HLS (MSI -> OLI)
            for b in BANDS:
                slope, icpt = HLS_MSI_TO_OLI[MSI_SOURCE[b]]
                t[b] = slope * t[b] + icpt
        t["sensor"] = s
        parts.append(t)
    d = add_indices(pd.concat(parts, ignore_index=True))
    d["lake_date"] = d.lake + "|" + d.date
    return d.dropna(subset=FEATURES).reset_index(drop=True)


VARIANTS = {"hls_sr": ("LS_", "S2_", False),
            "toa_raw": ("LS_TOA_", "S2_TOA_", False),
            "toa_bandpass": ("LS_TOA_", "S2_TOA_", True)}


def control(d, sensor, rng, group="date"):
    sub = d[d.sensor == sensor].copy()
    units = np.sort(sub[group].unique())
    half = set(rng.choice(units, len(units) // 2, replace=False))
    sub["half"] = np.where(sub[group].isin(half), "A", "B")
    return sep_test(sub, FEATURES, sub[group].values, label_col="half", positive="A")


def load_joined(sub):
    toa = pd.read_csv(CACHE)
    toa = toa[toa.pid >= 0].drop_duplicates(["lake", "date", "pid"])
    toa = toa.rename(columns={f"{s}_{b}": f"{s}_TOA_{b}" for s in ("LS", "S2")
                              for b in BANDS})
    j = sub.merge(toa, on=["lake", "date", "pid"], how="inner")
    j = j.dropna(subset=[f"{s}_TOA_{b}" for s in ("LS", "S2") for b in BANDS])
    n = j.groupby(["lake", "date"]).pid.transform("size")
    j = j[n >= MIN_PAIRS].reset_index(drop=True)
    j["dt_min"] = (j.s2_time - j.ls_time) / 60000.0
    j["ls_sza"] = 90 - j.ls_sun_elev
    return toa, j


def band_ratios(pl, pre_ls, pre_s2):
    out = {}
    for b in BANDS:
        den = pl[f"{pre_s2}{b}"].median()
        out[b] = (round(float(pl[f"{pre_ls}{b}"].median() / den), 3)
                  if abs(den) >= 5e-4 else np.nan)
    return out


def main():
    t0 = time.time()
    banner("p18 -- SEPARABILIDAD LANDSAT / SENTINEL-2 EN TOA vs HLS (MISMOS PARES)")
    sub = subset_pairs()
    print(f"  subconjunto: {len(sub)} pares, {sub.groupby('lake').date.nunique().sum()} "
          f"lago-fechas, {sub.lake.nunique()} lagos")
    if "--eval" not in sys.argv:
        extract(sub)
    t_ext = time.time() - t0
    toa, j = load_joined(sub)
    rng = np.random.default_rng(SEED)

    rows, band_rows = [], []
    for lake in LAKES:
        pl = j[j.lake == lake]
        nd = pl.date.nunique()
        base = {"lake": lake, "n_dates_subset": int(sub[sub.lake == lake].date.nunique()),
                "n_dates": nd, "n_pairs": len(pl)}
        if nd < MIN_DATES_TEST:
            print(f"  {lake}: {nd} fechas con pares TOA, no se prueba")
            rows.append({**base, "tested": False})
            continue
        row = {**base, "tested": True,
               "s2_green_median": round(float(pl.S2_B3.median()), 5),
               "s2_toa_green_median": round(float(pl.S2_TOA_B3.median()), 5),
               "dt_min_median": round(float(pl.dt_min.median()), 1),
               "ls_sza_median": round(float(pl.ls_sza.median()), 2),
               "s2_sza_median": round(float(pl.s2_sza.median()), 2)}
        for v, (a, b, bp) in VARIANTS.items():
            d = to_long(pl, a, b, bp)
            t = sep_test(d, FEATURES, d.date.values)
            row[f"{v}_bacc"], row[f"{v}_auc"] = t["balanced_accuracy"], t["auc"]
            if v != "toa_bandpass":
                row[f"{v}_ctrl_LS_bacc"] = control(d, "LS", rng)["balanced_accuracy"]
                row[f"{v}_ctrl_S2_bacc"] = control(d, "S2", rng)["balanced_accuracy"]
        for v, (a, b) in {"hls_sr": ("LS_", "S2_"),
                          "toa": ("LS_TOA_", "S2_TOA_")}.items():
            r = band_ratios(pl, a, b)
            for bd in BANDS:
                row[f"{v}_ratio_{bd}"] = r[bd]
                band_rows.append({"lake": lake, "level": v, "band": bd, "ratio_LS_S2": r[bd]})
        row["delta_bacc_toa_minus_sr"] = round(row["toa_raw_bacc"] - row["hls_sr_bacc"], 4)
        rows.append(row)
        print(f"  {lake:15s} f {nd:3d} p {len(pl):5d}  SR {row['hls_sr_bacc']:.3f}  "
              f"TOA {row['toa_raw_bacc']:.3f}  TOA+bp {row['toa_bandpass_bacc']:.3f}  "
              f"ctrl TOA {row['toa_raw_ctrl_LS_bacc']:.2f}/{row['toa_raw_ctrl_S2_bacc']:.2f}"
              f"  dt {row['dt_min_median']:.0f} min  {time.time() - t0:.0f} s", flush=True)
    sep = pd.DataFrame(rows)
    tested = sep[sep.tested]

    # prueba conjunta sobre los lagos probados
    jp = j[j.lake.isin(tested.lake)]
    jp = jp.sample(frac=1, random_state=SEED)
    jp = jp[jp.groupby(["lake", "date"]).cumcount() < POOL_CAP].sort_values(
        ["lake", "date", "pid"])
    pooled = {}
    for v, (a, b, bp) in VARIANTS.items():
        d = to_long(jp, a, b, bp)
        pooled[v] = {"lake_date": sep_test(d, FEATURES, d.lake_date.values),
                     "leave_lake_out": sep_test(d, FEATURES, d.lake.values),
                     "control_S2_lake_date": control(d, "S2", rng, "lake_date")}
        print(f"  conjunta {v}: lago-fecha {pooled[v]['lake_date']['balanced_accuracy']}"
              f"  dejando lagos {pooled[v]['leave_lake_out']['balanced_accuracy']}"
              f"  control {pooled[v]['control_S2_lake_date']['balanced_accuracy']}")

    # comparacion pareada por lago y relacion con la claridad
    x = tested.s2_green_median
    corr = {}
    for col in ("hls_sr_bacc", "toa_raw_bacc", "toa_bandpass_bacc",
                "delta_bacc_toa_minus_sr"):
        rho, pv = spearmanr(x, tested[col])
        corr[col] = {"rho": round(float(rho), 3), "p": float(f"{pv:.3g}")}
    rho, pv = spearmanr(tested.hls_sr_bacc, tested.toa_raw_bacc)
    w = wilcoxon(tested.toa_raw_bacc, tested.hls_sr_bacc)
    wb = wilcoxon(tested.toa_bandpass_bacc, tested.toa_raw_bacc)
    paired = {"n_lakes": int(len(tested)),
              "median_bacc": {c: round(float(tested[f"{c}_bacc"].median()), 3)
                              for c in VARIANTS},
              "range_bacc": {c: [float(tested[f"{c}_bacc"].min()),
                                 float(tested[f"{c}_bacc"].max())] for c in VARIANTS},
              "median_delta_toa_minus_sr": round(float(tested.delta_bacc_toa_minus_sr.median()), 3),
              "n_lakes_toa_lower": int((tested.delta_bacc_toa_minus_sr < 0).sum()),
              "n_lakes_toa_ge_0p8": int((tested.toa_raw_bacc >= 0.8).sum()),
              "n_lakes_sr_ge_0p8": int((tested.hls_sr_bacc >= 0.8).sum()),
              "wilcoxon_toa_vs_sr_p": float(f"{w.pvalue:.3g}"),
              "wilcoxon_bandpass_vs_raw_toa_p": float(f"{wb.pvalue:.3g}"),
              "spearman_toa_vs_sr_across_lakes": {"rho": round(float(rho), 3),
                                                  "p": float(f"{pv:.3g}")},
              "control_bacc_max": {
                  "hls_sr": float(tested[["hls_sr_ctrl_LS_bacc", "hls_sr_ctrl_S2_bacc"]].max().max()),
                  "toa_raw": float(tested[["toa_raw_ctrl_LS_bacc", "toa_raw_ctrl_S2_bacc"]].max().max())}}
    timing = {"dt_min_median_all": round(float(j.dt_min.median()), 1),
              "dt_min_iqr": [round(float(j.dt_min.quantile(q)), 1) for q in (0.25, 0.75)],
              "sza_diff_median": round(float((j.s2_sza - j.ls_sza).median()), 2)}
    med_sr, med_toa = paired["median_bacc"]["hls_sr"], paired["median_bacc"]["toa_raw"]
    # veredicto graduado: nivel de TOA, diferencia pareada y su significancia
    if med_toa <= 0.6:
        verdict = "toa_not_separable"
    elif abs(paired["median_delta_toa_minus_sr"]) <= 0.02 and w.pvalue > 0.05:
        verdict = "toa_as_separable"
    elif med_toa < med_sr:
        verdict = "toa_separable_but_less_than_sr"
    else:
        verdict = "toa_more_separable_than_sr"
    print("  pareado:", paired)
    print("  claridad:", corr, " tiempos:", timing, " veredicto:", verdict)

    sep.to_csv(TIDY / "toa_separability.csv", index=False)
    out = {"design": {"source_pairs": "p12 multilake_hls_raw (same-day HLSL30/HLSS30 pairs)",
                      "subset_points_per_lake_date": N_PER_DATE, "seed": SEED,
                      "landsat_toa": [L8_ID, L9_ID], "landsat_mask": "QA_PIXEL bits 1-5",
                      "s2_toa": S2_ID, "s2_mask": f"{S2P_ID} < {MAX_PROB} (QA60 fallback)",
                      "s2_resampling": "mean to Landsat 30 m grid",
                      "classifier": "RF 300, 12 features, GroupKFold(5) by date",
                      "min_pairs_per_date": MIN_PAIRS, "min_dates_per_lake": MIN_DATES_TEST,
                      "pool_cap_per_date": POOL_CAP, "n_splits": N_SPLITS},
           "n_pairs_subset": int(len(sub)),
           "n_pairs_with_toa": int(len(j)),
           "n_lake_dates_subset": int(sub.groupby("lake").date.nunique().sum()),
           "n_lake_dates_with_toa": int(j.groupby("lake").date.nunique().sum()),
           "n_lakes_tested": int(len(tested)),
           "lakes_not_tested": sep.loc[~sep.tested, "lake"].tolist(),
           "pooled": pooled, "paired_per_lake": paired,
           "spearman_vs_hls_s2_green": corr, "acquisition_geometry": timing,
           "band_ratio_median_over_lakes": {
               lvl: {b: round(float(tested[f"{lvl}_ratio_{b}"].median()), 3)
                     for b in BANDS} for lvl in ("hls_sr", "toa")},
           "verdict": verdict,
           "interpretation": (
               "toa_not_separable: the land atmospheric correction (LaSRC in HLS) "
               "introduces the Landsat/Sentinel-2 difference over water. "
               "toa_as_separable: the difference is already present at top of atmosphere "
               "and arises from sensor calibration, spectral response and the acquisition-"
               "time difference. toa_separable_but_less_than_sr: most of the separability "
               "already exists at TOA; the atmospheric correction adds (does not remove) a "
               "smaller part. The HLS MSI->OLI bandpass coefficients were derived for "
               "surface reflectance; applied to TOA they move S2 blue away from OLI TOA and "
               "raise separability, so toa_bandpass is reported only as a check. "
               "Caveat: TOA includes path radiance (Rayleigh + aerosol, most of the TOA "
               "signal over water in the blue-green) that differs between the two "
               "acquisitions (solar/view geometry, aerosol change between overpasses), so "
               "TOA separability is not a pure sensor effect; the classifier can also use "
               "this geometry-driven path-radiance difference."),
           "runtime_s": {"extraction": round(t_ext), "total": round(time.time() - t0)},
           "lakes": sep.to_dict(orient="records")}
    write_json(out, MET / "toa.json")


if __name__ == "__main__":
    main()
