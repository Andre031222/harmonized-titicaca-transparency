"""
p10_trends.py -- tendencias de la transparencia in situ, 2011-2024.

La tendencia es una afirmacion sobre el registro de campo de IMARPE (no sobre
los match-ups). Serie: mediana anual de Secchi por zona.
  * primaria       : solo epoca seca (jul-oct)
  * sensibilidad 1 : + la unica campana de lluvias (dic 2012)
  * sensibilidad 2 : ajuste por ubicacion. El numero de ubicaciones por
                     zona-anio varia mucho (Bahia de Puno 2011: 79 lecturas,
                     otros anios 3-12), asi que la mediana puede moverse solo
                     por DONDE se muestreo. Serie de anomalias: mediana anual de
                     (secchi - climatologia de la ubicacion), solo ubicaciones
                     con >= 3 anios.
  * sensibilidad 3 : anio inicial 2011..2014, contando solo series DISTINTAS y
                     CAMBIOS reales de veredicto.
Mann-Kendall con P exacta (distribucion de S por permutacion completa: numeros
de Mahonian sin empates, enumeracion/Monte Carlo con empates) y aproximacion
normal; pendiente de Sen en anios calendario con IC 95 % de Gilbert (1987).

El registro crudo no se redistribuye: si existe se reescriben los agregados
versionados; si no, se leen.

Entradas : data/insitu/raw/imarpe_titicaca_wq.csv (opcional)
           data/processed/insitu_annual_medians.csv,
           data/processed/insitu_location_anomaly_medians.csv (versionados)
Salidas  : results/tidy/annual_trends.csv, trend_tests.csv,
           trend_start_year_sensitivity.csv, trend_verdict_changes.csv,
           trend_n_zone_year.csv
           results/metrics/temporal.json
"""
import itertools
import json

import numpy as np
import pandas as pd
from scipy import stats

from p00_config import write_json, FIELD_RAW, MET, PROC, SEED, TIDY, ZONE_LABELS, ZONES, banner

DRY_MONTHS = [7, 8, 9, 10]
WET_MONTHS = [12, 1, 2, 3]
MIN_YEARS_LOC = 3
START_YEARS = [2011, 2012, 2013, 2014]
ALPHA = 0.05
N_MC = 200000
AGG_MED = PROC / "insitu_annual_medians.csv"
AGG_ANOM = PROC / "insitu_location_anomaly_medians.csv"
SERIES = {"dry": "Dry season (Jul-Oct)", "dry_wet": "Dry season + Dec 2012 wet campaign"}
ADJUST = {"raw": "Annual median Secchi", "location": "Location-adjusted (anomaly) median"}


# --- agregados -----------------------------------------------------------------
def build_aggregates():
    from p01_extract_matchups import load_field
    from p02_build_dataset import field_units
    # ubicaciones sobre el registro completo (igual que p02), luego solo Secchi
    f = field_units(load_field())
    f = f[f.secchi.notna()].copy()
    f["year"], f["month"] = f.date.dt.year, f.date.dt.month
    f["season"] = np.where(f.month.isin(WET_MONTHS), "wet",
                           np.where(f.month.isin(DRY_MONTHS), "dry", "other"))
    assert (f.season != "other").all(), "meses fuera de las epocas definidas"
    med = (f.groupby(["zona", "year", "season"], as_index=False)
           .agg(secchi_median=("secchi", "median"), n=("secchi", "size"),
                n_locations=("location", "nunique")))
    assert not med.duplicated(["zona", "year"]).any()

    rows = []
    for series, seasons in [("dry", ["dry"]), ("dry_wet", ["dry", "wet"])]:
        g = f[f.season.isin(seasons)].copy()
        # una lectura por ubicacion-anio (mediana) para que la climatologia no
        # pese mas los anios con lecturas repetidas
        ly = g.groupby(["zona", "location", "year"], as_index=False).secchi.median()
        ny = ly.groupby("location").year.nunique()
        keep = ny[ny >= MIN_YEARS_LOC].index
        ly = ly[ly.location.isin(keep)]
        ly["clim"] = ly.groupby("location").secchi.transform("mean")
        ly["anom"] = ly.secchi - ly.clim
        a = (ly.groupby(["zona", "year"], as_index=False)
             .agg(anomaly_median=("anom", "median"), n_locations=("location", "nunique")))
        a["series"] = series
        a["season"] = a.year.map(g.groupby("year").season.first())
        rows.append(a)
    anom = pd.concat(rows)[["series", "zona", "year", "season", "anomaly_median", "n_locations"]]
    med.to_csv(AGG_MED, index=False, float_format="%.4f")
    anom.to_csv(AGG_ANOM, index=False, float_format="%.4f")
    print(f"  registro crudo: {len(f)} lecturas de Secchi, {f.location.nunique()} ubicaciones "
          f"-> {AGG_MED.name}, {AGG_ANOM.name}")
    return med, anom


def load_aggregates():
    if FIELD_RAW.exists():
        return build_aggregates()
    print("  registro crudo no disponible (no se redistribuye): se leen los agregados versionados")
    return pd.read_csv(AGG_MED), pd.read_csv(AGG_ANOM)


def series_table(med, anom):
    """Tabla larga: serie x ajuste x zona x anio -> valor."""
    out = []
    for s, seasons in [("dry", ["dry"]), ("dry_wet", ["dry", "wet"])]:
        m = med[med.season.isin(seasons)]
        out.append(pd.DataFrame({"series": s, "adjust": "raw", "zona": m.zona, "year": m.year,
                                 "season": m.season, "value": m.secchi_median, "n": m.n,
                                 "n_locations": m.n_locations}))
        a = anom[anom.series == s]
        out.append(pd.DataFrame({"series": s, "adjust": "location", "zona": a.zona,
                                 "year": a.year, "season": a.season, "value": a.anomaly_median,
                                 "n": np.nan, "n_locations": a.n_locations}))
    t = pd.concat(out, ignore_index=True)
    t["zone_label"] = t.zona.map(ZONE_LABELS)
    return t.sort_values(["series", "adjust", "zona", "year"]).reset_index(drop=True)


# --- Mann-Kendall --------------------------------------------------------------
def s_stat(x):
    x = np.asarray(x, float)
    i, j = np.triu_indices(len(x), 1)
    return int(np.sign(x[j] - x[i]).sum())


def mahonian(n):
    """Numero de permutaciones de n elementos por numero de inversiones."""
    c = np.array([1], dtype=object)
    for m in range(2, n + 1):
        new = np.zeros(len(c) + m - 1, dtype=object)
        for k in range(m):
            new[k:k + len(c)] += c
        c = new
    return c


def mk_exact_p(x, rng):
    """P bilateral exacta de S bajo H0 (todas las permutaciones equiprobables)."""
    x = np.asarray(x, float)
    n = len(x)
    s = s_stat(x)
    if len(np.unique(x)) == n:
        cnt = mahonian(n)
        npairs = n * (n - 1) // 2
        svals = npairs - 2 * np.arange(len(cnt))
        tot = sum(cnt)
        return float(sum(c for c, v in zip(cnt, svals) if abs(v) >= abs(s)) / tot), "exact"
    i, j = np.triu_indices(n, 1)
    if n <= 9:
        P = x[np.array(list(itertools.permutations(range(n))))]
        ss = np.sign(P[:, j] - P[:, i]).sum(1)
        return float(np.mean(np.abs(ss) >= abs(s))), "exact_enumeration_ties"
    P = rng.permuted(np.tile(x, (N_MC, 1)), axis=1)
    ss = np.sign(P[:, j] - P[:, i]).sum(1)
    return float((1 + np.sum(np.abs(ss) >= abs(s))) / (N_MC + 1)), "monte_carlo_ties"


def mann_kendall(x, t, rng):
    x, t = np.asarray(x, float), np.asarray(t, float)
    o = np.argsort(t)
    x, t = x[o], t[o]
    n = len(x)
    s = s_stat(x)
    _, cnt = np.unique(x, return_counts=True)
    var = (n * (n - 1) * (2 * n + 5) - sum(c * (c - 1) * (2 * c + 5) for c in cnt)) / 18.0
    z = (s - np.sign(s)) / np.sqrt(var) if s != 0 else 0.0
    p_norm = 2 * stats.norm.sf(abs(z))
    p_exact, method = mk_exact_p(x, rng)
    i, j = np.triu_indices(n, 1)
    slopes = np.sort((x[j] - x[i]) / (t[j] - t[i]))
    N = len(slopes)
    sen = float(np.median(slopes))
    # Gilbert (1987): C = z_{1-a/2} sqrt(VAR(S)); M1 = (N-C)/2, M2 = (N+C)/2;
    # limite inferior = pendiente de rango M1, superior = rango M2 + 1
    # (rangos 1..N, interpolando si no son enteros)
    C = stats.norm.ppf(1 - ALPHA / 2) * np.sqrt(var)
    M1, M2 = (N - C) / 2, (N + C) / 2
    ranks = np.arange(1, N + 1)
    lo = float(np.interp(M1, ranks, slopes)) if M1 >= 1 else np.nan
    hi = float(np.interp(M2 + 1, ranks, slopes)) if M2 + 1 <= N else np.nan
    tau = s / (n * (n - 1) / 2)
    return {"n_years": n, "start_year": int(t[0]), "end_year": int(t[-1]), "S": s,
            "var_S": float(var), "tau": float(tau), "z": float(z), "p_normal": float(p_norm),
            "p_exact": p_exact, "p_method": method, "sen_slope": sen,
            "sen_lo95": lo, "sen_hi95": hi}


def verdict(r):
    if r["p_exact"] >= ALPHA:
        return "ns"
    return "increasing" if r["S"] > 0 else "decreasing"


# --- principal -----------------------------------------------------------------
def main():
    banner("p10  TENDENCIAS DE TRANSPARENCIA IN SITU (Mann-Kendall / Sen)")
    rng = np.random.default_rng(SEED)
    med, anom = load_aggregates()
    ser = series_table(med, anom)

    # n por zona-anio
    nzy = med.assign(zone_label=med.zona.map(ZONE_LABELS))
    nzy = nzy.merge(anom[anom.series == "dry_wet"][["zona", "year", "n_locations"]]
                    .rename(columns={"n_locations": "n_locations_in_anomaly"}),
                    on=["zona", "year"], how="left")
    print(nzy.pivot(index="year", columns="zone_label", values="n").fillna(0).astype(int)
          .to_string())

    # pruebas: serie x ajuste x zona x anio inicial
    rows = []
    for (s, a, z), g in ser.groupby(["series", "adjust", "zona"]):
        prev_years = None
        for start in START_YEARS:
            sub = g[g.year >= start]
            if len(sub) < 4:
                continue
            yrs = tuple(sub.year)
            r = mann_kendall(sub.value.values, sub.year.values, rng)
            r.update({"series": s, "adjust": a, "zona": z, "zone_label": ZONE_LABELS[z],
                      "requested_start": start, "years": " ".join(map(str, yrs)),
                      "distinct_series": yrs != prev_years})
            r["verdict"] = verdict(r)
            rows.append(r)
            prev_years = yrs
    tests = pd.DataFrame(rows)
    full = tests[tests.requested_start == START_YEARS[0]].copy()
    full["test_label"] = full.series.map(SERIES) + " | " + full.adjust.map(ADJUST)

    banner("TENDENCIAS (serie completa)", "-")
    for _, r in full.iterrows():
        print(f"  {r.series:8s} {r.adjust:8s} {r.zone_label:13s} n={r.n_years:2d} "
              f"{r.start_year}-{r.end_year} Sen={r.sen_slope:+.3f} "
              f"[{r.sen_lo95:+.3f},{r.sen_hi95:+.3f}] m/anio  tau={r.tau:+.2f} "
              f"P_exacta={r.p_exact:.4f} P_normal={r.p_normal:.4f}  {r.verdict}")

    # cambios de veredicto con el anio inicial (solo series distintas)
    banner("SENSIBILIDAD AL ANIO INICIAL (series distintas)", "-")
    sens = tests[tests.distinct_series].copy()
    ch = []
    for (s, a, z), g in sens.groupby(["series", "adjust", "zona"]):
        v = list(g.verdict)
        nchg = sum(1 for i in range(1, len(v)) if v[i] != v[i - 1])
        ch.append({"series": s, "adjust": a, "zona": z, "zone_label": ZONE_LABELS[z],
                   "n_distinct_series": len(v), "start_years_distinct": " ".join(str(y) for y in g.start_year),
                   "verdicts": " -> ".join(v), "n_verdict_changes": nchg,
                   "stable": nchg == 0})
        print(f"  {s:8s} {a:8s} {ZONE_LABELS[z]:13s} series distintas={len(v)} "
              f"({' '.join(str(y) for y in g.start_year)}): {' -> '.join(v)}  cambios={nchg}")
    chg = pd.DataFrame(ch)

    # que tendencias sobreviven
    banner("SUPERVIVENCIA", "-")
    surv = {}
    for z in ZONES:
        zr = full[full.zona == z].set_index(["series", "adjust"])
        prim = zr.loc[("dry", "raw")]
        allv = tests[tests.zona == z]
        rec = {
            "primary": {"verdict": prim.verdict, "sen_slope": round(prim.sen_slope, 3),
                        "sen_ci95": [round(prim.sen_lo95, 3), round(prim.sen_hi95, 3)],
                        "p_exact": round(prim.p_exact, 4), "p_normal": round(prim.p_normal, 4),
                        "n_years": int(prim.n_years), "years": prim.years},
            "with_wet_campaign": zr.loc[("dry_wet", "raw")].verdict,
            "location_adjusted": zr.loc[("dry", "location")].verdict,
            "location_adjusted_with_wet": zr.loc[("dry_wet", "location")].verdict,
            "location_adjusted_slope": round(zr.loc[("dry", "location")].sen_slope, 3),
            "location_adjusted_ci95": [round(zr.loc[("dry", "location")].sen_lo95, 3),
                                       round(zr.loc[("dry", "location")].sen_hi95, 3)],
            "location_adjusted_p_exact": round(zr.loc[("dry", "location")].p_exact, 4),
            "start_year_verdict_changes_primary": int(chg[(chg.zona == z) & (chg.series == "dry") &
                                                          (chg.adjust == "raw")].n_verdict_changes.iloc[0]),
            "n_distinct_start_series_primary": int(chg[(chg.zona == z) & (chg.series == "dry") &
                                                       (chg.adjust == "raw")].n_distinct_series.iloc[0]),
            "n_tests_total": int(len(allv[allv.distinct_series])),
            "n_tests_significant_same_sign_as_primary": int(
                (allv[allv.distinct_series].verdict == prim.verdict).sum()) if prim.verdict != "ns" else None,
        }
        rec["survives_all"] = bool(prim.verdict != "ns" and
                                   (allv[allv.distinct_series].verdict == prim.verdict).all())
        surv[ZONE_LABELS[z]] = rec
        print(f"  {ZONE_LABELS[z]:13s} primaria={prim.verdict:10s} | +lluvias={rec['with_wet_campaign']:10s}"
              f" | ajuste ubicacion={rec['location_adjusted']:10s} | cambios por anio inicial="
              f"{rec['start_year_verdict_changes_primary']} | sobrevive todo={rec['survives_all']}")

    # --- salidas ---------------------------------------------------------------
    ann = ser.merge(full[["series", "adjust", "zona", "sen_slope", "sen_lo95", "sen_hi95",
                          "p_exact", "p_normal", "tau", "verdict"]],
                    on=["series", "adjust", "zona"], how="left")
    ann["series_label"], ann["adjust_label"] = ann.series.map(SERIES), ann.adjust.map(ADJUST)
    ann.to_csv(TIDY / "annual_trends.csv", index=False, float_format="%.4f")
    cols = ["series", "adjust", "zona", "zone_label", "requested_start", "start_year", "end_year",
            "n_years", "years", "distinct_series", "S", "var_S", "tau", "z", "p_normal", "p_exact",
            "p_method", "sen_slope", "sen_lo95", "sen_hi95", "verdict"]
    full[cols].to_csv(TIDY / "trend_tests.csv", index=False, float_format="%.5f")
    tests[cols].to_csv(TIDY / "trend_start_year_sensitivity.csv", index=False, float_format="%.5f")
    chg.to_csv(TIDY / "trend_verdict_changes.csv", index=False)
    nzy.to_csv(TIDY / "trend_n_zone_year.csv", index=False, float_format="%.4f")

    def rec(r):
        return {"n_years": int(r.n_years), "years": r.years, "S": int(r.S), "tau": round(r.tau, 3),
                "p_exact": round(r.p_exact, 4), "p_method": r.p_method,
                "p_normal": round(r.p_normal, 4), "sen_slope_m_per_yr": round(r.sen_slope, 3),
                "sen_ci95_gilbert": [None if np.isnan(r.sen_lo95) else round(r.sen_lo95, 3),
                                     None if np.isnan(r.sen_hi95) else round(r.sen_hi95, 3)],
                "verdict": r.verdict}

    out = {
        "source": "IMARPE in-situ record (not the match-ups); annual median Secchi by zone",
        "n_readings_with_secchi": int(med.n.sum()),
        "years_with_data": sorted(int(y) for y in med.year.unique()),
        "wet_season_campaign": {"years": sorted(int(y) for y in med.loc[med.season == "wet", "year"].unique()),
                                "months": "December"},
        "dry_season_first_year": {ZONE_LABELS[z]: int(med[(med.zona == z) & (med.season == "dry")].year.min())
                                  for z in ZONES},
        "n_per_zone_year": {ZONE_LABELS[z]: {int(y): int(n) for y, n in
                                             med[med.zona == z][["year", "n"]].values} for z in ZONES},
        "location_adjustment": {"min_years_per_location": MIN_YEARS_LOC,
                                "climatology": "mean of per-year location medians over the same series",
                                "n_locations_used_dry": {ZONE_LABELS[z]: int(anom[(anom.series == "dry") &
                                                                                  (anom.zona == z)].n_locations.max())
                                                         for z in ZONES}},
        "methods": {"mk_p": "two-sided exact permutation distribution of S (Mahonian numbers; "
                            "full enumeration or Monte Carlo when ties); normal approximation with "
                            "continuity correction for reference",
                    "sen_ci": "Gilbert (1987) rank-based 95% CI, ranks M1=(N-C)/2 and M2+1, "
                              "interpolated; slopes on calendar years",
                    "alpha": ALPHA},
        "trends": {f"{s}|{a}": {ZONE_LABELS[z]: rec(full[(full.series == s) & (full.adjust == a) &
                                                         (full.zona == z)].iloc[0]) for z in ZONES}
                   for s in SERIES for a in ADJUST},
        "start_year_sensitivity": chg.drop(columns="zona").to_dict("records"),
        "survival": surv,
    }
    surviving = [k for k, v in surv.items() if v["survives_all"]]
    sig_primary = [k for k, v in surv.items() if v["primary"]["verdict"] != "ns"]
    out["summary"] = {
        "significant_primary": sig_primary,
        "robust_to_all_sensitivities": surviving,
        "text": ("Primary dry-season trends significant (exact MK, P<0.05): "
                 + (", ".join(f"{k} ({surv[k]['primary']['sen_slope']:+.3f} m/yr, "
                              f"95% CI {surv[k]['primary']['sen_ci95'][0]:+.3f} to "
                              f"{surv[k]['primary']['sen_ci95'][1]:+.3f}, P={surv[k]['primary']['p_exact']:.3f})"
                              for k in sig_primary) or "none")
                 + ". Robust to season inclusion, start year and location adjustment: "
                 + (", ".join(surviving) or "none") + "."),
    }
    write_json(out, MET / "temporal.json")
    banner("SALIDAS")
    for f in ["annual_trends.csv", "trend_tests.csv", "trend_start_year_sensitivity.csv",
              "trend_verdict_changes.csv", "trend_n_zone_year.csv"]:
        print(f"  {TIDY / f}")
    print(f"  {MET / 'temporal.json'}")
    print("\n" + out["summary"]["text"])


if __name__ == "__main__":
    main()
