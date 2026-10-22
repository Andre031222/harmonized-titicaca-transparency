"""
p02_build_dataset.py -- tabla de match-ups y dataset de analisis.

Entradas : data/processed/hls_matchups_raw.csv, sr_matchups_raw.csv (p01)
           data/insitu/raw/imarpe_titicaca_wq.csv (solo si existe)
Salidas  : data/processed/matchups.csv        todos los productos y ventanas,
                                              con los valores de campo (versionado)
           data/processed/analysis_dataset.csv HLS, +-10 d, con Secchi (primario)
           results/tidy/dataset_inventory.csv, insitu_distribution.csv,
                 location_units.csv
           results/metrics/dataset_summary.json

Unidades:
  * ubicacion  : IMARPE reasigno los codigos de estacion entre anios (un mismo
                 codigo aparece a mas de 80 km de si mismo), asi que la unidad
                 espacial es la coordenada: lecturas a <= 500 m (enlace completo)
                 son la misma ubicacion.
  * campana    : en los match-ups hay una campana por anio.
  * bloque     : k-medias (k = 8) sobre las ubicaciones, para validacion espacial.
"""
import json

import numpy as np
import pandas as pd

from p00_config import (qc_pass, BANDS, FEATURES, FIELD_RAW, LOCATION_RADIUS_M, MET,
                        N_SPATIAL_BLOCKS, PROC, SEED, TIDY, ZONES, add_indices,
                        banner, to_utm)

WINDOWS = [1, 3, 5, 10]
FIELD_COLS = ["station", "zona", "date", "lat", "lon", "secchi", "chl", "tss",
              "temp_insitu"]


def field_units(field):
    """Ubicacion (agrupamiento de coordenadas) y bloque espacial de cada
    medicion de campo."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from sklearn.cluster import KMeans
    f = field.copy()
    f["x"], f["y"] = to_utm(f.lat.values, f.lon.values)
    xy = f[["x", "y"]].values
    lab = fcluster(linkage(xy, "complete"), LOCATION_RADIUS_M, "distance")
    f["location"] = pd.Series(lab).map(lambda v: f"L{v:03d}").values
    loc = f.groupby("location").agg(x=("x", "mean"), y=("y", "mean")).reset_index()
    km = KMeans(N_SPATIAL_BLOCKS, n_init=20, random_state=SEED).fit(loc[["x", "y"]])
    # bloques numerados de oeste a este para que sean legibles en los mapas
    order = np.argsort(np.argsort(km.cluster_centers_[:, 0]))
    loc["spatial_block"] = [f"B{order[c] + 1}" for c in km.labels_]
    f = f.merge(loc[["location", "spatial_block"]], on="location")
    f["campaign"] = f.date.dt.year.astype(str)
    f["sampling_date"] = f.date.dt.strftime("%Y-%m-%d")
    return f


def load_field_table():
    """Registro de campo 2011-2024 con su ubicacion, bloque y campana."""
    from p01_extract_matchups import load_field
    return field_units(load_field())


def spread_km(g):
    """Distancia maxima entre dos lecturas con el mismo codigo de estacion."""
    from scipy.spatial.distance import pdist
    xy = g[["x", "y"]].values
    return float(pdist(xy).max() / 1000) if len(xy) > 1 else 0.0


def long_hls():
    h = pd.read_csv(PROC / "hls_matchups_raw.csv")
    h = h.drop_duplicates(["sensor", "key", "window"])
    return h


def long_sr():
    s = pd.read_csv(PROC / "sr_matchups_raw.csv")
    s["key"] = s.station.astype(str) + "|" + s.date.astype(str)
    s = s.drop_duplicates(["sensor", "key"])
    out = []
    for w in WINDOWS:
        cols = {f"w{w}_{b}": b for b in BANDS + ["B8A"]}
        t = s[["sensor", "key", f"w{w}_n"] + list(cols)].rename(columns=cols)
        t = t.rename(columns={f"w{w}_n": "n_img"})
        t["window"], t["product"] = w, "sr"
        out.append(t)
    return pd.concat(out, ignore_index=True)


def main():
    banner("p02 -- TABLA DE MATCH-UPS Y DATASET DE ANALISIS")
    if not FIELD_RAW.exists():
        # sin el registro de IMARPE: se parte de la tabla versionada
        print("  (registro crudo ausente: se usa data/processed/matchups.csv)")
        m = pd.read_csv(PROC / "matchups.csv")
        return analysis(m, None)
    field = load_field_table()
    fcols = ["key", "station", "zona", "date", "sampling_date", "campaign", "lat",
             "lon", "location", "spatial_block", "secchi", "chl", "tss", "temp_insitu"]
    raw = pd.concat([long_hls(), long_sr()], ignore_index=True)
    m = raw.merge(field[fcols], on="key", how="inner")
    m = m[(m.n_img.fillna(0) > 0)].dropna(subset=BANDS)
    m["qc_pass"] = qc_pass(m)
    m["year"] = m.date.dt.year
    m["month"] = m.date.dt.month
    m = m.sort_values(["product", "window", "sensor", "sampling_date", "station"])
    keep = ["product", "window", "sensor", "station", "location", "spatial_block",
            "zona", "sampling_date", "campaign", "year", "month", "lat", "lon",
            "n_img", "qc_pass", "secchi", "chl", "tss", "temp_insitu"] + BANDS + ["B8A", "B10"]
    m[keep].to_csv(PROC / "matchups.csv", index=False, float_format="%.6g")
    analysis(pd.read_csv(PROC / "matchups.csv"), field[field.date.dt.year >= 2013])


def analysis(m, field):

    prim = m[(m["product"] == "hls") & (m.window == 10)].dropna(subset=["secchi"])
    qc_out = prim[~prim.qc_pass]
    qc_out[["location", "station", "sampling_date", "sensor", "zona", "secchi"] + BANDS].to_csv(
        TIDY / "qc_excluded.csv", index=False)
    print(f"  control de calidad: {len(qc_out)} match-ups excluidos (SWIR1 > 0.02 o NDWI <= 0)")
    d = prim[prim.qc_pass]
    d = add_indices(d).dropna(subset=FEATURES).reset_index(drop=True)
    d["event"] = d.location + "|" + d.sampling_date
    d.drop(columns=["B8A"]).to_csv(PROC / "analysis_dataset.csv", index=False,
                                   float_format="%.6g")

    # --- resumen -------------------------------------------------------------
    if field is not None:
        ids = field.groupby("station")[["x", "y"]].apply(spread_km)
        zone_multi = field.groupby("station").zona.nunique()
        ids.rename("max_spread_km").reset_index().to_csv(
            TIDY / "station_id_spread.csv", index=False)
    else:
        ids = zone_multi = pd.Series(dtype=float)
    inv = (d.groupby("campaign")
           .agg(n=("secchi", "size"), n_locations=("location", "nunique"),
                n_sampling_dates=("sampling_date", "nunique"),
                first=("sampling_date", "min"), last=("sampling_date", "max"),
                n_s2=("sensor", lambda s: int((s == "S2").sum())),
                n_ls=("sensor", lambda s: int((s == "LS").sum())),
                secchi_median=("secchi", "median")).reset_index())
    inv["span_days"] = (pd.to_datetime(inv["last"]) - pd.to_datetime(inv["first"])).dt.days + 1
    print(inv.to_string(index=False))
    inv.to_csv(TIDY / "dataset_inventory.csv", index=False)
    d[["station", "location", "spatial_block", "zona", "sampling_date", "campaign",
       "year", "month", "sensor", "lat", "lon", "secchi"]].to_csv(
        TIDY / "insitu_distribution.csv", index=False)
    loc = (d.groupby("location").agg(lat=("lat", "mean"), lon=("lon", "mean"),
                                     zona=("zona", lambda z: z.mode().iloc[0]),
                                     spatial_block=("spatial_block", "first"),
                                     n=("secchi", "size"),
                                     n_campaigns=("campaign", "nunique"),
                                     secchi_mean=("secchi", "mean")).reset_index())
    loc.to_csv(TIDY / "location_units.csv", index=False)

    frac_half = float(np.mean(np.isclose(d.secchi * 2, (d.secchi * 2).round())))
    paired = d.groupby("event").sensor.nunique()
    summary = {
        "product": "NASA HLS v2.0 (HLSS30, HLSL30), +-10 d",
        "n_field_measurements_2013_2024": int(len(field)) if field is not None else None,
        "n_analysis": int(len(d)),
        "n_qc_excluded": int(len(qc_out)),
        "qc_excluded_by_campaign": qc_out.campaign.value_counts().sort_index().to_dict(),
        "n_s2": int((d.sensor == "S2").sum()), "n_ls": int((d.sensor == "LS").sum()),
        "n_events": int(d.event.nunique()), "n_paired_events": int((paired == 2).sum()),
        "n_station_ids": int(d.station.nunique()),
        "n_locations": int(d.location.nunique()),
        "n_sampling_dates": int(d.sampling_date.nunique()),
        "n_campaigns": int(d.campaign.nunique()),
        "campaigns": sorted(d.campaign.unique().tolist()),
        "campaign_span_days_min": int(inv.span_days.min()),
        "campaign_span_days_max": int(inv.span_days.max()),
        "campaign_dates_median": float(inv.n_sampling_dates.median()),
        "station_ids_moving_gt10km": int((ids > 10).sum()) if len(ids) else None,
        "station_id_max_spread_km": round(float(ids.max()), 1) if len(ids) else None,
        "station_ids_in_two_zones": int((zone_multi > 1).sum()) if len(ids) else None,
        "images_per_matchup_s2": float(d.loc[d.sensor == "S2", "n_img"].median()),
        "images_per_matchup_ls": float(d.loc[d.sensor == "LS", "n_img"].median()),
        "months_present": sorted(int(v) for v in d.month.unique()),
        "secchi_min": float(d.secchi.min()), "secchi_max": float(d.secchi.max()),
        "secchi_n_unique_values": int(d.secchi.nunique()),
        "secchi_frac_half_metre_multiples": round(frac_half, 3),
        "by_zone": {z: {"n": int((d.zona == z).sum()),
                        "secchi_mean": round(float(d.loc[d.zona == z, "secchi"].mean()), 2),
                        "secchi_sd": round(float(d.loc[d.zona == z, "secchi"].std()), 2)}
                    for z in ZONES}}
    json.dump(summary, open(MET / "dataset_summary.json", "w"), indent=2)
    for k, v in summary.items():
        if k != "by_zone":
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
