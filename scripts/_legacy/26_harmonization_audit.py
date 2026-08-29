"""
==============================================================================
26_harmonization_audit.py
AUDITORIA Y CORRECCION del retrieval de transparencia (Secchi) en Titicaca.

Este script reemplaza los numeros defectuosos de 13/17/21/22 y genera las
metricas defendibles para la version JGLR. Corrige cuatro errores reales
detectados en la auditoria de 2026-08-22:

  (E1) ARMONIZACION ROTA. Los coeficientes de Roy et al. (2016) OLI->MSI se
       derivaron sobre superficies terrestres y tienen interceptos de
       0.012-0.045 en reflectancia. Sobre agua oligotrofica oscura la senal
       nativa de Landsat es ~0.001-0.010, asi que el intercepto DOMINA:
       en B8 el intercepto es ~72x la senal nativa mediana. Resultado: las
       bandas "armonizadas" NIR/SWIR de Landsat son practicamente una
       constante que identifica al sensor, no una medida del agua.
       => Se cuantifica el fallo y se deriva una recalibracion LOCAL sobre
          agua a partir de los eventos observados por ambos sensores el
          mismo dia en la misma estacion.

  (E2) BASELINE CLASICO ESPANTAPAJAROS. El benchmark original ajustaba
       LinearRegression sobre log1p(y) y deshacia con expm1 sin acotar; UNA
       sola prediccion de 150 m hundia el R2 a -2.74 y subia el RMSE a 5.51.
       => Se implementa la forma estandar ln(Zsd) ~ ln(B2/B3) (Kloiber et
          al. 2002) con acotado fisico al rango observable.

  (E3) BLOQUEO DE VALIDACION EN LA UNIDAD EQUIVOCADA. La CV se agrupaba por
       ESTACION, pero la varianza real de Secchi esta entre CAMPANAS, no
       entre estaciones (un nulo "media de la fecha" ya alcanza R2=0.31,
       mientras que un nulo "media de la estacion" da R2=-0.01). Agrupar por
       estacion no retiene nada.
       => La validacion primaria pasa a ser GroupKFold por FECHA DE CAMPANA.

  (E4) HOLD-OUT TEMPORAL MAL DESCRITO. No existen campanas en 2020, 2021 ni
       2023. "train<=2021 / test 2022-2024" es en realidad
       "train<=2019 / test {2022, 2024}".
       => Se reporta el inventario real de anios.

Salidas (results/metrics/):
  harmonization_audit.json      fallo de Roy cuantificado + coef. locales
  benchmark_models_fixed.csv    benchmark con el baseline clasico correcto
  validation_fixed.csv          jerarquia de validacion + modelos nulos
  retrievability.json           chl-a / TSS / LST (lo que NO es recuperable)
==============================================================================
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_score
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data/processed"
MET = ROOT / "results/metrics"
MET.mkdir(parents=True, exist_ok=True)

# Roy et al. (2016) OLI -> MSI: S2 = a + b * OLI. Derivados sobre tierra.
ROY = {"B2": (0.0183, 0.8850), "B3": (0.0123, 0.9317), "B4": (0.0123, 0.9372),
       "B8": (0.0448, 0.8339), "B11": (0.0306, 0.8639), "B12": (0.0116, 0.9165)}

BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
RATIOS = ["NDWI", "NDTI", "B2_B3", "B4_B3", "B3_B2", "B2_B4"]
ALL_F = BANDS + RATIOS
VIS_F = ["B2", "B3", "B4", "NDTI", "B2_B3", "B4_B3", "B3_B2", "B2_B4"]
ZONES = ["BAHIA PUNO", "LAGO MENOR", "LAGO MAYOR"]


def add_indices(d):
    e = 1e-9
    d["NDWI"] = (d.B3 - d.B8) / (d.B3 + d.B8 + e)
    d["NDTI"] = (d.B4 - d.B3) / (d.B4 + d.B3 + e)
    d["B2_B3"] = d.B2 / (d.B3 + e)
    d["B4_B3"] = d.B4 / (d.B3 + e)
    d["B3_B2"] = d.B3 / (d.B2 + e)
    d["B2_B4"] = d.B2 / (d.B4 + e)
    return d


def load_raw():
    frames = []
    for fn, sensor in [("matchups_s2.csv", "S2"), ("matchups_ls.csv", "LS")]:
        d = pd.read_csv(PROC / fn)
        d = d[d.matched == 1].copy()
        d["sensor"] = sensor
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df.date)
    df["year"] = df.date.dt.year
    return df


def rf():
    return RandomForestRegressor(n_estimators=500, max_depth=12, min_samples_leaf=3,
                                 random_state=42, n_jobs=-1)


def scores(yt, yp):
    return dict(R2=round(float(r2_score(yt, yp)), 3),
                RMSE=round(float(np.sqrt(mean_squared_error(yt, yp))), 3),
                MAE=round(float(mean_absolute_error(yt, yp)), 3),
                bias=round(float(np.mean(yp - yt)), 3), n=int(len(yt)))


def cv_predict(X, y, groups, splitter, model_fn=rf):
    """Out-of-fold predictions con scaler ajustado solo en train."""
    yt, yp = [], []
    it = splitter.split(X, y, groups) if groups is not None else splitter.split(X, y)
    for tr, te in it:
        sc = RobustScaler().fit(X[tr])
        m = model_fn()
        m.fit(sc.transform(X[tr]), y[tr])
        yp.append(np.clip(m.predict(sc.transform(X[te])), 0.05, None))
        yt.append(y[te])
    return np.concatenate(yt), np.concatenate(yp)


# ---------------------------------------------------------------------------
# (E1) Diagnostico de la armonizacion de Roy sobre agua
# ---------------------------------------------------------------------------
def audit_harmonization(raw):
    print("=" * 78)
    print("(E1) ARMONIZACION ROY et al. (2016) SOBRE AGUA OLIGOTROFICA")
    print("=" * 78)
    report = {"roy_intercept_vs_signal": {}, "local_water_coefficients": {},
              "sensor_separability": {}}

    for b, (a, slope) in ROY.items():
        ls_h = raw.loc[raw.sensor == "LS", b].dropna()
        s2 = raw.loc[raw.sensor == "S2", b].dropna()
        native = (ls_h - a) / slope          # deshace Roy -> SR nativo OLI
        ratio = float(a / max(abs(native.median()), 1e-9))
        report["roy_intercept_vs_signal"][b] = {
            "roy_intercept": a, "roy_slope": slope,
            "native_OLI_median": round(float(native.median()), 6),
            "harmonized_LS_median": round(float(ls_h.median()), 6),
            "S2_median": round(float(s2.median()), 6),
            "intercept_over_native_signal": round(ratio, 1),
            "harmonized_over_S2": round(float(ls_h.median() / max(s2.median(), 1e-9)), 2)}
        print(f"  {b:4s} intercepto={a:.4f} es {ratio:6.1f}x la senal nativa "
              f"(mediana OLI={native.median():+.5f}) | LS_harm/S2 = "
              f"{ls_h.median() / max(s2.median(), 1e-9):5.2f}x")

    # Separabilidad de sensores: si la armonizacion funcionara, deberia ser ~azar
    d = add_indices(raw.copy()).dropna(subset=ALL_F)
    ys = (d.sensor == "S2").astype(int).values
    chance = float(max(ys.mean(), 1 - ys.mean()))
    print(f"\n  Un clasificador deberia ser incapaz de distinguir los sensores "
          f"si estuvieran armonizados (azar = {chance * 100:.1f}%):")
    for name, F in [("todas las 12 feats", ALL_F), ("solo visible", VIS_F),
                    ("solo ratios", RATIOS), ("solo NIR+SWIR", ["B8", "B11", "B12"])]:
        acc = float(cross_val_score(
            RandomForestClassifier(300, random_state=42, n_jobs=-1),
            d[F].values, ys, cv=5, scoring="accuracy").mean())
        report["sensor_separability"][name] = round(acc, 4)
        print(f"    {name:20s} -> {acc * 100:5.1f}% de acierto")
    report["sensor_separability"]["chance_level"] = round(chance, 4)

    # Recalibracion LOCAL sobre agua: eventos vistos por ambos sensores
    piv = (add_indices(raw.copy())
           .pivot_table(index=["station", "date"], columns="sensor",
                        values=BANDS, aggfunc="median").dropna())
    print(f"\n  Recalibracion local LS->S2 sobre {len(piv)} eventos pareados "
          f"(misma estacion, misma fecha):")
    coefs = {}
    for b in BANDS:
        xx = piv[(b, "LS")].values.reshape(-1, 1)
        yy = piv[(b, "S2")].values
        lr = LinearRegression().fit(xx, yy)
        r = float(np.corrcoef(xx.ravel(), yy)[0, 1])
        coefs[b] = (float(lr.intercept_), float(lr.coef_[0]))
        report["local_water_coefficients"][b] = {
            "intercept": round(float(lr.intercept_), 6),
            "slope": round(float(lr.coef_[0]), 4), "pearson_r": round(r, 3),
            "n_paired": int(len(piv))}
        print(f"    {b:4s} S2 = {lr.intercept_:+.5f} + {lr.coef_[0]:+.4f}*LS_roy"
              f"   r={r:+.3f}")
    print("    NOTA: r(B2)~0.26 -> el azul, que es la banda que el modelo mas usa,")
    print("    es justo donde ambos sensores MENOS concuerdan sobre esta agua.")
    return report, coefs


def apply_local_coefs(raw, coefs):
    d = raw.copy()
    m = d.sensor == "LS"
    for b, (a, s) in coefs.items():
        d.loc[m, b] = a + s * d.loc[m, b]
    return d


# ---------------------------------------------------------------------------
# (E2) Baseline clasico correcto
# ---------------------------------------------------------------------------
def classical_kloiber(sec, groups, splitter):
    """ln(Zsd) ~ ln(B2/B3), acotado al rango fisicamente observable."""
    Xl = np.log(np.clip(sec[["B2_B3"]].values, 1e-6, None))
    y = sec.secchi.values
    hi = float(y.max() * 1.5)
    yt, yp = [], []
    for tr, te in splitter.split(Xl, y, groups):
        lr = LinearRegression().fit(Xl[tr], np.log(np.clip(y[tr], 1e-3, None)))
        yp.append(np.clip(np.exp(lr.predict(Xl[te])), 0.05, hi))
        yt.append(y[te])
    return np.concatenate(yt), np.concatenate(yp)


def classical_as_published(sec, groups, splitter):
    """Reproduce el bug original: log1p/expm1 sin acotar."""
    Xb = sec[["B2_B3"]].values
    y = sec.secchi.values
    yt, yp = [], []
    for tr, te in splitter.split(Xb, y, groups):
        sc = RobustScaler().fit(Xb[tr])
        lr = LinearRegression().fit(sc.transform(Xb[tr]), np.log1p(y[tr]))
        yp.append(np.clip(np.expm1(lr.predict(sc.transform(Xb[te]))), 0.05, None))
        yt.append(y[te])
    return np.concatenate(yt), np.concatenate(yp)


# ---------------------------------------------------------------------------
def main():
    raw = load_raw()
    harm_report, coefs = audit_harmonization(raw)

    sec = add_indices(raw.copy())
    sec = sec[sec.secchi.notna()].dropna(subset=ALL_F).reset_index(drop=True)
    y = sec.secchi.values
    g_station = sec.station.values
    g_date = sec.date.astype(str).values
    gkf = GroupKFold(5)

    # -- (E2) benchmark corregido ------------------------------------------
    print("\n" + "=" * 78)
    print("(E2) BENCHMARK CON EL BASELINE CLASICO CORREGIDO (GroupKFold por campana)")
    print("=" * 78)
    rows = []
    yt, yp = classical_as_published(sec, g_date, gkf)
    s = scores(yt, yp)
    s["model"] = "Classical blue/green -- AS PUBLISHED (buggy, unbounded expm1)"
    s["max_prediction_m"] = round(float(yp.max()), 1)
    rows.append(s)
    print(f"  como se publico  R2={s['R2']:+.3f} RMSE={s['RMSE']:.2f}  "
          f"(prediccion maxima = {yp.max():.1f} m; el maximo observado es {y.max():.1f} m)")

    yt, yp = classical_kloiber(sec, g_date, gkf)
    s = scores(yt, yp)
    s["model"] = "Classical blue/green (Kloiber-style, bounded)"
    s["max_prediction_m"] = round(float(yp.max()), 1)
    rows.append(s)
    print(f"  corregido        R2={s['R2']:+.3f} RMSE={s['RMSE']:.2f}")

    X_all = sec[ALL_F].values
    yt, yp = cv_predict(X_all, y, g_date, gkf,
                        model_fn=lambda: LinearRegression())
    s = scores(yt, yp); s["model"] = "Multiband linear"; rows.append(s)
    print(f"  lineal multibanda R2={s['R2']:+.3f} RMSE={s['RMSE']:.2f}")

    yt, yp = cv_predict(X_all, y, g_date, gkf)
    s = scores(yt, yp); s["model"] = "Random Forest (this study)"; rows.append(s)
    print(f"  Random Forest     R2={s['R2']:+.3f} RMSE={s['RMSE']:.2f} "
          f"bias={s['bias']:+.2f}")
    pd.DataFrame(rows)[["model", "R2", "RMSE", "MAE", "bias", "n",
                        "max_prediction_m"]].to_csv(
        MET / "benchmark_models_fixed.csv", index=False)

    # -- (E3) jerarquia de validacion + modelos nulos ------------------------
    print("\n" + "=" * 78)
    print("(E3) JERARQUIA DE VALIDACION Y MODELOS NULOS")
    print("=" * 78)
    vrows = []

    def add(exp, case, yt, yp, note=""):
        s = scores(yt, yp)
        vrows.append({"experiment": exp, "case": case, **s, "note": note})
        print(f"  {exp:22s} {case:28s} R2={s['R2']:+.3f} RMSE={s['RMSE']:.2f} n={s['n']}")

    # nulos: cuanta habilidad es solo "que campana / que zona es esta"
    for label, key in [("zone-mean", "zona"), ("station-mean", "station"),
                       ("campaign-date-mean", "date")]:
        yt_, yp_ = [], []
        for tr, te in gkf.split(X_all, y, g_date):
            t = pd.DataFrame({"k": sec[key].astype(str).values, "y": y})
            mp = t.iloc[tr].groupby("k").y.mean()
            yp_.append(t.iloc[te].k.map(mp).fillna(y[tr].mean()).values)
            yt_.append(y[te])
        add("null_model", label, np.concatenate(yt_), np.concatenate(yp_),
            "grouped by campaign date")

    # bloqueo por distintas unidades
    for case, grp in [("groupkfold_station", g_station),
                      ("groupkfold_campaign_date", g_date),
                      ("groupkfold_year", sec.year.astype(str).values)]:
        if len(np.unique(grp)) < 5:
            continue
        yt_, yp_ = cv_predict(X_all, y, grp, gkf)
        add("blocking_unit", case, yt_, yp_, f"{len(np.unique(grp))} groups")
    yt_, yp_ = cv_predict(X_all, y, None, KFold(5, shuffle=True, random_state=42))
    add("blocking_unit", "random_kfold", yt_, yp_, "leakage-prone reference")

    # leave-one-zone-out
    for z in ZONES:
        trm, tem = sec.zona.values != z, sec.zona.values == z
        if tem.sum() < 10:
            continue
        sc = RobustScaler().fit(X_all[trm])
        m = rf(); m.fit(sc.transform(X_all[trm]), y[trm])
        add("leave_one_zone_out", z, y[tem],
            np.clip(m.predict(sc.transform(X_all[tem])), 0.05, None))

    # hold-out temporal (con el inventario real de anios)
    inv = sec.groupby("year").size().to_dict()
    print(f"\n  Inventario real de anios: {inv}")
    print("  -> NO hay campanas en 2020, 2021 ni 2023. 'train<=2021' == 'train<=2019'.")
    tr, te = (sec.year <= 2021).values, (sec.year >= 2022).values
    sc = RobustScaler().fit(X_all[tr]); m = rf(); m.fit(sc.transform(X_all[tr]), y[tr])
    add("temporal_holdout", "train<=2019_test_2022_2024", y[te],
        np.clip(m.predict(sc.transform(X_all[te])), 0.05, None),
        f"no campaigns 2020/2021/2023; {len(set(sec.loc[te,'station']) & set(sec.loc[tr,'station']))} of "
        f"{sec.loc[te,'station'].nunique()} test stations also in train")

    # -- (E1) impacto de la armonizacion en el modelo ------------------------
    print("\n  Efecto de la armonizacion sobre el modelo (GroupKFold por campana):")
    for tag, F, frame in [
            ("roy_harmonized_all_bands", ALL_F, sec),
            ("visible_only_no_broken_bands", VIS_F, sec),
            ("scale_invariant_ratios_only", RATIOS, sec),
            ("locally_reharmonized_all", ALL_F,
             add_indices(apply_local_coefs(raw, coefs)).pipe(
                 lambda d: d[d.secchi.notna()].dropna(subset=ALL_F).reset_index(drop=True)))]:
        yy = frame.secchi.values
        gg = frame.date.astype(str).values
        yt_, yp_ = cv_predict(frame[F].values, yy, gg, gkf)
        add("harmonization_variant", tag, yt_, yp_)

    for sensor in ["S2", "LS"]:
        sub = sec[sec.sensor == sensor]
        yt_, yp_ = cv_predict(sub[ALL_F].values, sub.secchi.values,
                              sub.date.astype(str).values, gkf)
        add("single_sensor", f"{sensor}_only", yt_, yp_, "no harmonization needed")

    pd.DataFrame(vrows).to_csv(MET / "validation_fixed.csv", index=False)

    # -- (E4) retrievability: lo que NO se recupera --------------------------
    print("\n" + "=" * 78)
    print("(E4) RETRIEVABILIDAD DE OTRAS VARIABLES (artefactos que faltaban en disco)")
    print("=" * 78)
    dd = add_indices(raw.copy())
    dd["NDCI_proxy"] = (dd.B4 - dd.B3) / (dd.B4 + dd.B3 + 1e-9)
    retr = {}
    for var, label in [("chl", "chlorophyll-a (mg m-3)"), ("tss", "TSS (mg L-1)"),
                       ("temp_insitu", "in-situ water temperature (C)")]:
        sub = dd[dd[var].notna()].dropna(subset=["NDCI_proxy"])
        if len(sub) < 20:
            continue
        r = float(np.corrcoef(sub.NDCI_proxy, sub[var])[0, 1])
        # ademas: un RF completo, validado por campana, sobre esa variable
        yv = sub[var].values
        yt_, yp_ = cv_predict(sub[ALL_F].values, yv, sub.date.astype(str).values, gkf)
        retr[var] = {"label": label, "n": int(len(sub)),
                     "pearson_r_single_index": round(r, 3),
                     "rf_R2_campaign_cv": round(float(r2_score(yt_, yp_)), 3),
                     "insitu_min": round(float(yv.min()), 2),
                     "insitu_max": round(float(yv.max()), 2)}
        print(f"  {label:32s} n={len(sub):4d} r(indice)={r:+.3f} "
              f"R2(RF, CV campana)={r2_score(yt_, yp_):+.3f}")
    if "lst_C" in raw.columns:
        sub = raw[(raw.lst_C.notna()) & (raw.temp_insitu.notna())]
        r = float(np.corrcoef(sub.lst_C, sub.temp_insitu)[0, 1])
        rmse = float(np.sqrt(mean_squared_error(sub.temp_insitu, sub.lst_C)))
        retr["lst_thermal"] = {"label": "Landsat ST_B10 vs in-situ temperature",
                               "n": int(len(sub)), "pearson_r": round(r, 3),
                               "rmse_C": round(rmse, 2)}
        print(f"  {'Landsat termico ST_B10':32s} n={len(sub):4d} r={r:+.3f} "
              f"RMSE={rmse:.2f} C")

    # -- inventario del dataset ---------------------------------------------
    harm_report["dataset"] = {
        "n_matchups_total": int((raw.matched == 1).sum()),
        "n_with_secchi": int(len(sec)),
        "n_stations": int(sec.station.nunique()),
        "n_campaign_dates": int(sec.date.nunique()),
        "years_present": sorted(int(v) for v in sec.year.unique()),
        "months_present": sorted(int(v) for v in sec.date.dt.month.unique()),
        "secchi_min": float(y.min()), "secchi_max": float(y.max()),
        "secchi_mean": round(float(y.mean()), 2), "secchi_sd": round(float(y.std()), 2),
        "n_unique_secchi_values": int(len(np.unique(y)))}
    print(f"\n  Dataset: {harm_report['dataset']['n_with_secchi']} match-ups con Secchi | "
          f"{harm_report['dataset']['n_campaign_dates']} fechas de campana | "
          f"meses presentes = {harm_report['dataset']['months_present']} (SOLO epoca seca)")

    json.dump(harm_report, open(MET / "harmonization_audit.json", "w"), indent=2)
    json.dump(retr, open(MET / "retrievability.json", "w"), indent=2)
    print(f"\nGuardado en {MET}:")
    print("  harmonization_audit.json | benchmark_models_fixed.csv")
    print("  validation_fixed.csv     | retrievability.json")


if __name__ == "__main__":
    main()
