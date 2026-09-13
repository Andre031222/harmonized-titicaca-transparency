"""
p01_build_dataset.py -- construye el dataset de analisis y su inventario.

Entradas : data/processed/matchups_s2.csv, matchups_ls.csv
Salidas  : data/processed/analysis_dataset.csv   (el unico dataset que usa
                                                  el resto del pipeline)
           results/tidy/dataset_inventory.csv    (para ggplot2)
           results/tidy/insitu_distribution.csv
           results/metrics/dataset_summary.json

Aqui se documenta explicitamente lo que el manuscrito anterior escondia:
  * los anios REALMENTE presentes (no hay campanas en 2020, 2021 ni 2023)
  * los meses REALMENTE presentes (julio-octubre; solo epoca seca)
  * la granularidad del Secchi in-situ (lecturas a 0.5-1 m, 73 valores unicos)
"""
import json

import numpy as np
import pandas as pd

from p00_config import (BANDS, FEATURES, MET, PROC, TIDY, ZONES, add_indices,
                        banner)


def load_matchups():
    frames = []
    for fname, sensor in [("matchups_s2.csv", "S2"), ("matchups_ls.csv", "LS")]:
        d = pd.read_csv(PROC / fname)
        d["sensor"] = sensor
        d["source_file"] = fname
        frames.append(d)
    raw = pd.concat(frames, ignore_index=True)
    raw["date"] = pd.to_datetime(raw["date"])
    raw["year"] = raw["date"].dt.year
    raw["month"] = raw["date"].dt.month
    raw["campaign_date"] = raw["date"].dt.strftime("%Y-%m-%d")
    return raw


def main():
    banner("p01 -- CONSTRUCCION DEL DATASET DE ANALISIS")
    raw = load_matchups()
    print(f"  filas crudas (todas las estaciones-fecha): {len(raw)}")
    print(f"    Sentinel-2 : {int((raw.sensor == 'S2').sum())}")
    print(f"    Landsat 8/9: {int((raw.sensor == 'LS').sum())}")

    matched = raw[raw.matched == 1].copy()
    print(f"\n  con imagen limpia dentro de +-10 dias (matched==1): {len(matched)}")
    print(f"    Sentinel-2 : {int((matched.sensor == 'S2').sum())}")
    print(f"    Landsat 8/9: {int((matched.sensor == 'LS').sum())}")

    df = add_indices(matched)
    sec = df[df.secchi.notna()].dropna(subset=FEATURES).reset_index(drop=True)
    print(f"\n  con Secchi valido y todas las features: {len(sec)}")

    # --- pseudo-replicas cruzadas: un evento visto por ambos sensores -------
    sec["event_id"] = (sec.station.astype(str) + "|" + sec.campaign_date
                       + "|" + sec.secchi.round(3).astype(str))
    n_events = sec.event_id.nunique()
    n_dup = len(sec) - n_events
    print(f"    eventos in-situ independientes : {n_events}")
    print(f"    match-ups que son pseudo-replica cruzada S2<->LS: {n_dup}")

    # --- inventario temporal -----------------------------------------------
    banner("INVENTARIO TEMPORAL REAL (error 4 del manuscrito anterior)", "-")
    inv = (sec.groupby("year")
           .agg(n=("secchi", "size"), n_stations=("station", "nunique"),
                n_campaigns=("campaign_date", "nunique"),
                sensors=("sensor", lambda s: "+".join(sorted(set(s)))),
                secchi_median=("secchi", "median"))
           .reset_index())
    print(inv.to_string(index=False))
    span = range(int(sec.year.min()), int(sec.year.max()) + 1)
    missing = sorted(set(span) - set(sec.year.unique()))
    print(f"\n  ANIOS SIN CAMPANA dentro del rango: {missing}")
    print("  => 'train <=2021' es en realidad 'train <=2019'.")
    print("  => NO existe dato de 2023: cualquier mencion a El Nino 2023-24 es falsa.")

    months = sorted(sec.month.unique())
    print(f"\n  MESES presentes: {months}  (1=ene ... 12=dic)")
    print("  => TODO el dataset es epoca seca austral (julio-octubre).")
    print("  => No se puede sostener ninguna afirmacion sobre epoca de lluvias.")

    # --- granularidad del Secchi -------------------------------------------
    banner("GRANULARIDAD DE LA MEDIDA DE REFERENCIA", "-")
    vc = sec.secchi.value_counts().sort_index()
    frac_half = float(np.mean(np.isclose(sec.secchi * 2, (sec.secchi * 2).round())))
    print(f"  Secchi: min={sec.secchi.min()} max={sec.secchi.max()} "
          f"media={sec.secchi.mean():.2f} sd={sec.secchi.std():.2f}")
    print(f"  valores distintos: {sec.secchi.nunique()} en {len(sec)} match-ups")
    print(f"  fraccion de lecturas en multiplos de 0.5 m: {frac_half * 100:.1f}%")
    print("  => el RMSE se acerca a la granularidad de la propia medida de campo.")

    # --- exportaciones para R ----------------------------------------------
    inv.to_csv(TIDY / "dataset_inventory.csv", index=False)

    dist = sec[["station", "zona", "campaign_date", "year", "month", "sensor",
                "lat", "lon", "secchi"] + FEATURES].copy()
    dist.to_csv(TIDY / "insitu_distribution.csv", index=False)

    keep = (["row_id", "station", "zona", "campaign_date", "date", "year",
             "month", "lat", "lon", "sensor", "event_id", "secchi", "chl",
             "tss", "temp_insitu"] + FEATURES)
    keep = [c for c in keep if c in sec.columns]
    sec[keep].to_csv(PROC / "analysis_dataset.csv", index=False)

    summary = {
        "n_rows_raw": int(len(raw)),
        "n_matched": int(len(matched)),
        "n_matched_s2": int((matched.sensor == "S2").sum()),
        "n_matched_ls": int((matched.sensor == "LS").sum()),
        "n_analysis": int(len(sec)),
        "n_stations": int(sec.station.nunique()),
        "n_campaign_dates": int(sec.campaign_date.nunique()),
        "n_independent_events": int(n_events),
        # imagenes compuestas por match-up: la reflectancia es la mediana de
        # todas las imagenes limpias de la ventana, no la mas cercana
        "images_per_matchup_s2": int(sec.loc[sec.sensor == "S2", "n_img"].median()),
        "images_per_matchup_ls": int(sec.loc[sec.sensor == "LS", "n_img"].median()),
        "n_cross_sensor_pseudoreplicates": int(n_dup),
        "years_present": sorted(int(v) for v in sec.year.unique()),
        "years_missing_in_span": [int(v) for v in missing],
        "months_present": [int(v) for v in months],
        "season": "austral dry season only (July-October)",
        "secchi_min": float(sec.secchi.min()), "secchi_max": float(sec.secchi.max()),
        "secchi_mean": round(float(sec.secchi.mean()), 3),
        "secchi_sd": round(float(sec.secchi.std()), 3),
        "secchi_n_unique_values": int(sec.secchi.nunique()),
        "secchi_frac_half_metre_multiples": round(frac_half, 3),
        "by_zone": {z: {"n": int((sec.zona == z).sum()),
                        "secchi_mean": round(float(sec.loc[sec.zona == z, "secchi"].mean()), 2),
                        "secchi_sd": round(float(sec.loc[sec.zona == z, "secchi"].std()), 2)}
                    for z in ZONES}}
    json.dump(summary, open(MET / "dataset_summary.json", "w"), indent=2)

    banner("SALIDAS")
    print(f"  {PROC / 'analysis_dataset.csv'}")
    print(f"  {TIDY / 'dataset_inventory.csv'}")
    print(f"  {TIDY / 'insitu_distribution.csv'}")
    print(f"  {MET / 'dataset_summary.json'}")


if __name__ == "__main__":
    main()
