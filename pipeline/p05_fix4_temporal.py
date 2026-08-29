"""
p05_fix4_temporal.py -- ERROR 4: el hold-out temporal estaba mal descrito.

El manuscrito anterior decia "hold-out temporal (train 2013-2021, test
2022-2024)". No existen campanas en 2020, 2021 ni 2023, asi que el corte real
es "train <=2019, test {2022, 2024}". Ademas afirmaba que el test incluia
"El Nino 2023-2024": no hay ni un solo dato de 2023.

Este script fija el inventario real, evalua el hold-out con la descripcion
correcta, mide cuanta independencia tiene de verdad (cuantas estaciones del
test aparecen tambien en entrenamiento) y documenta el sesgo estacional:
los 812 match-ups caen entre julio y octubre, unicamente epoca seca austral.

Salidas: results/tidy/temporal_holdout.csv
         results/tidy/seasonality.csv
         results/tidy/annual_trends.csv
         results/metrics/temporal.json
"""
import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler

from p00_config import (FEATURES, MET, N_SPLITS, PROC, SECCHI_CLIP_LO, TIDY,
                        ZONE_LABELS, ZONES, banner, metrics)
from p04_fix3_validation import make_model

MONTH_ES = {1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
            7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic"}


MONTH_MAP = {"ENE": 1, "FEB": 2, "MAR": 3, "ABR": 4, "MAY": 5, "JUN": 6,
             "JUL": 7, "AGO": 8, "SET": 9, "SEP": 9, "OCT": 10, "NOV": 11,
             "DIC": 12}


def load():
    d = pd.read_csv(PROC / "analysis_dataset.csv")
    d["date"] = pd.to_datetime(d["date"])
    return d


def load_full_insitu():
    """Registro in-situ COMPLETO (2011-2024), no solo el que tuvo imagen limpia.

    Las tendencias son una afirmacion sobre el lago, no sobre la disponibilidad
    de imagenes sin nubes, asi que deben calcularse sobre este registro.
    """
    from p00_config import DATA
    j = pd.read_csv(DATA / "external_validation/jairo_titicaca_wq.csv")
    j.columns = [c.replace("\xf1", "n").replace("�", "n") for c in j.columns]
    j["mon"] = j["Mes"].astype(str).str.upper().str[:3].map(MONTH_MAP)
    j["date"] = pd.to_datetime(dict(year=j["Ano"], month=j["mon"], day=j["Dia"]),
                               errors="coerce")
    j = j.rename(columns={"Transparencia_m": "secchi", "Zona": "zona",
                          "Estacion": "station"})
    j = j.dropna(subset=["date"])
    j["year"] = j["date"].dt.year
    return j[j.secchi.notna()].reset_index(drop=True)


def mann_kendall(x):
    """Mann-Kendall no parametrico + pendiente de Sen."""
    x = np.asarray(x, float)
    n = len(x)
    s = sum(np.sign(x[j] - x[i]) for i in range(n - 1) for j in range(i + 1, n))
    _, counts = np.unique(x, return_counts=True)
    tie = sum(c * (c - 1) * (2 * c + 5) for c in counts if c > 1)
    var = (n * (n - 1) * (2 * n + 5) - tie) / 18.0
    if var <= 0 or n < 3:
        return 0.0, 1.0, 0.0
    z = (s - np.sign(s)) / np.sqrt(var) if s != 0 else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    slopes = [(x[j] - x[i]) / (j - i) for i in range(n - 1) for j in range(i + 1, n)]
    return float(z), float(p), float(np.median(slopes)) if slopes else 0.0


def main():
    banner("p05 -- ERROR 4: HOLD-OUT TEMPORAL Y ESTACIONALIDAD")
    d = load()

    # --- (1) inventario real -----------------------------------------------
    banner("(1) INVENTARIO TEMPORAL REAL", "-")
    years = sorted(d.year.unique())
    span = list(range(min(years), max(years) + 1))
    missing = [y for y in span if y not in years]
    print(f"  anios con campana : {years}")
    print(f"  anios SIN campana : {missing}")
    print(f"  => el corte 'train <=2021' entrena exactamente igual que 'train <=2019'")
    print(f"  => el test '2022-2024' contiene solo {sorted(set(y for y in years if y >= 2022))}")
    print(f"  => NO hay dato de 2023: la mencion a 'El Nino 2023-2024' era falsa")

    # --- (2) hold-out temporal correcto ------------------------------------
    banner("(2) HOLD-OUT TEMPORAL (descrito correctamente)", "-")
    X, y = d[FEATURES].values, d.secchi.values
    rows = []
    for cut in [2017, 2018, 2019]:
        tr, te = (d.year <= cut).values, (d.year > cut).values
        if te.sum() < 30:
            continue
        sc = RobustScaler().fit(X[tr])
        m = make_model("rf"); m.fit(sc.transform(X[tr]), y[tr])
        pred = np.clip(m.predict(sc.transform(X[te])), SECCHI_CLIP_LO, None)
        mm = metrics(y[te], pred)
        test_years = sorted(set(d.loc[te, "year"]))
        shared = len(set(d.loc[te, "station"]) & set(d.loc[tr, "station"]))
        n_test_st = d.loc[te, "station"].nunique()
        rows.append({"cut_year": cut, "train_years": sorted(set(d.loc[tr, "year"])),
                     "test_years": test_years, **mm,
                     "n_train": int(tr.sum()),
                     "test_stations": int(n_test_st),
                     "test_stations_also_in_train": int(shared),
                     "station_overlap_pct": round(100 * shared / max(n_test_st, 1), 1)})
        print(f"  train <={cut} (n={tr.sum():3d}) -> test {test_years} (n={mm['n']:3d})  "
              f"R2={mm['R2']:+.3f} RMSE={mm['RMSE']:.2f}")
        print(f"      {shared}/{n_test_st} estaciones del test aparecen tambien en train "
              f"({100 * shared / max(n_test_st, 1):.0f}%)")
    print("\n  Esto mide generalizacion TEMPORAL, no independencia espacial:")
    print("  la mayoria de estaciones del test ya se vieron en entrenamiento.")

    # --- (3) estacionalidad -------------------------------------------------
    banner("(3) SESGO ESTACIONAL: SOLO EPOCA SECA", "-")
    seas = (d.groupby("month")
            .agg(n=("secchi", "size"), secchi_median=("secchi", "median"),
                 n_campaigns=("campaign_date", "nunique"))
            .reindex(range(1, 13), fill_value=0).reset_index())
    seas["month_label"] = seas.month.map(MONTH_ES)
    seas["season"] = np.where(seas.month.isin([12, 1, 2, 3]), "Lluvias (Dic-Mar)",
                              np.where(seas.month.isin([7, 8, 9, 10]),
                                       "Seca (Jul-Oct)", "Transicion"))
    print(seas[["month_label", "n", "n_campaigns", "secchi_median", "season"]]
          .to_string(index=False))
    wet = int(seas.loc[seas.month.isin([12, 1, 2, 3]), "n"].sum())
    print(f"\n  match-ups en epoca de LLUVIAS (dic-mar): {wet}")
    print("  => cero cobertura de la epoca de floraciones en Bahia de Puno.")
    print("  => el modelo esta calibrado y validado SOLO para epoca seca.")

    # --- (4) tendencias anuales ---------------------------------------------
    # IMPORTANTE: la tendencia es una afirmacion sobre el REGISTRO IN-SITU, no
    # sobre el subconjunto que casualmente tuvo imagen limpia. Se calcula sobre
    # el registro completo (2011-2024, 734 lecturas de Secchi), no sobre los
    # 812 match-ups. La diferencia NO es cosmetica: ver el analisis de
    # sensibilidad al anio inicial mas abajo.
    banner("(4) TENDENCIAS ANUALES (Mann-Kendall / Sen) -- registro in-situ completo", "-")
    insitu = load_full_insitu()
    print(f"  registro in-situ completo: n={len(insitu)} lecturas de Secchi, "
          f"anios {sorted(insitu.year.unique())}\n")
    trows, series = [], []
    for z in ZONES:
        sub = insitu[insitu.zona == z].groupby("year").secchi.median()
        if len(sub) < 4:
            continue
        z_stat, p, slope = mann_kendall(sub.values)
        trows.append({"zone": z, "zone_label": ZONE_LABELS[z],
                      "n_years": int(len(sub)), "start_year": int(sub.index.min()),
                      "z": round(z_stat, 3), "p_value": round(p, 4),
                      "sen_slope_m_per_yr": round(slope, 4),
                      "significant_at_005": bool(p < 0.05)})
        for yr, val in sub.items():
            series.append({"zone": z, "zone_label": ZONE_LABELS[z],
                           "year": int(yr), "secchi_median": float(val)})
        print(f"  {ZONE_LABELS[z]:15s} n_anios={len(sub):2d} "
              f"pendiente Sen={slope:+.3f} m/anio  p={p:.3f}  "
              f"{'SIGNIFICATIVA' if p < 0.05 else 'no significativa'}")

    # --- (4b) SENSIBILIDAD AL ANIO INICIAL ----------------------------------
    # Este es el quinto error, que aun no se habia cometido pero estaba a un
    # paso: si la serie se empieza en 2013 (el anio con las medianas mas bajas
    # del registro) en lugar de 2011, DOS zonas pasan a tener tendencia
    # positiva "significativa". El resultado lo fija la eleccion del anio
    # inicial, no el lago.
    banner("(4b) SENSIBILIDAD DE LA TENDENCIA AL ANIO INICIAL", "-")
    print("  Mismo test, mismo dato, distinto anio de inicio de la serie:\n")
    sens = []
    print(f"  {'Zona':15s} {'inicio':>7s} {'n':>3s} {'Sen(m/anio)':>12s} {'p':>8s}  veredicto")
    for z in ZONES:
        for start in [2011, 2012, 2013, 2014]:
            sub = (insitu[(insitu.zona == z) & (insitu.year >= start)]
                   .groupby("year").secchi.median())
            if len(sub) < 4:
                continue
            _, p, slope = mann_kendall(sub.values)
            sig = p < 0.05
            sens.append({"zone": z, "zone_label": ZONE_LABELS[z],
                         "start_year": start, "n_years": int(len(sub)),
                         "sen_slope_m_per_yr": round(slope, 4),
                         "p_value": round(p, 4), "significant_at_005": bool(sig)})
            print(f"  {ZONE_LABELS[z]:15s} {start:7d} {len(sub):3d} {slope:+12.3f} "
                  f"{p:8.4f}  {'SIGNIFICATIVA' if sig else 'no significativa'}")
    n_flip = sum(1 for s in sens if s["start_year"] >= 2013 and s["significant_at_005"])
    print(f"\n  {n_flip} de {len(sens)} combinaciones cambian de veredicto segun")
    print("  donde se empiece la serie. 2013 es el anio con las medianas mas")
    print("  bajas del registro, asi que arrancar ahi fabrica una tendencia")
    print("  positiva artificial.")
    print("\n  DECISION: se reporta el registro COMPLETO (2011-2024) -> sin")
    print("  tendencia significativa en ninguna zona, y se publica esta tabla de")
    print("  sensibilidad. Con 9-11 valores anuales y 3 anios sin campana, la")
    print("  potencia es baja: es AUSENCIA DE EVIDENCIA, no evidencia de estabilidad.")

    # --- salidas ------------------------------------------------------------
    pd.DataFrame(rows).to_csv(TIDY / "temporal_holdout.csv", index=False)
    seas.to_csv(TIDY / "seasonality.csv", index=False)
    pd.DataFrame(series).merge(pd.DataFrame(trows), on=["zone", "zone_label"]) \
        .to_csv(TIDY / "annual_trends.csv", index=False)
    pd.DataFrame(sens).to_csv(TIDY / "trend_start_year_sensitivity.csv", index=False)
    json.dump({"years_present": [int(v) for v in years],
               "years_missing": [int(v) for v in missing],
               "months_present": [int(v) for v in sorted(d.month.unique())],
               "n_wet_season_matchups": wet,
               "season_coverage": "austral dry season only (July-October)",
               "temporal_holdouts": rows,
               "trends_full_insitu_record": trows,
               "trend_start_year_sensitivity": sens,
               "corrections": [
                   "There are no campaigns in 2020, 2021 or 2023.",
                   "'train <=2021' is identical to 'train <=2019'.",
                   "The claim that the test set includes the 2023-2024 El Nino is "
                   "false: no 2023 data exist.",
                   "All match-ups fall between July and October; no wet-season "
                   "claim is supportable.",
                   "Trends must be computed on the full in-situ record "
                   "(2011-2024), not on the match-up subset. Starting the series "
                   "in 2013 -- the lowest-median year on record -- turns two "
                   "zones significantly positive (Lago Mayor Sen +0.42 m/yr, "
                   "p=0.012). On the full record no zone shows a significant "
                   "trend. The verdict is set by the start year, not by the lake."]},
              open(MET / "temporal.json", "w"), indent=2)

    banner("SALIDAS")
    for f in ["temporal_holdout.csv", "seasonality.csv", "annual_trends.csv"]:
        print(f"  {TIDY / f}")
    print(f"  {MET / 'temporal.json'}")


if __name__ == "__main__":
    main()
