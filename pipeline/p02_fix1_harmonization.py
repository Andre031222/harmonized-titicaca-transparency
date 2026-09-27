"""
p02_fix1_harmonization.py -- ¿basta el ajuste de banda de HLS para que Landsat
8/9 y Sentinel-2 sean intercambiables sobre agua oscura?

HLS v2.0 (Masek et al. 2021; Claverie et al. 2018) lleva MSI al espacio de OLI
con un ajuste lineal por banda derivado de espectros TERRESTRES (vegetacion,
desierto, urbano, nieve). Sus interceptos son pequenos (|b| <= 0.004) y sus
pendientes ~1: corrige diferencias de respuesta espectral, no de correccion
atmosferica. Sobre agua clara, cuya senal es de 0.001-0.01, la pregunta es si
lo que separa a los dos sensores es la banda o el procesamiento.

Nota historica: una version anterior aplicaba los coeficientes OLS OLI -> ETM+
de Roy et al. (2016), que llevan Landsat 8 al espacio de Landsat 7 y no al de
Sentinel-2. p00 los usa ahora solo para deshacer esa transformacion en las
tablas de match-ups.

Este script:
  (A) mide cuanto mueve HLS a Sentinel-2 y cuanta diferencia queda entre los
      sensores despues, banda a banda y sobre los eventos pareados
  (B) prueba de dos muestras con clasificador, con los controles que un
      revisor pediria: validacion agrupada por fecha de muestreo, exactitud
      balanceada y AUC, solo eventos pareados, pares a +-3 dias con la
      extraccion de p11, antes y despues de la recalibracion local (ajustada
      dentro de cada fold) y un control negativo con el mismo sensor
  (C) recalibracion local sobre agua (Sentinel-2 -> Landsat) en los eventos
      pareados, descriptiva
  (D) firma espectral por sensor en cada etapa

Salidas: results/metrics/harmonization.json
         results/tidy/harmonization_bands.csv
         results/tidy/harmonization_paired.csv
         results/tidy/harmonization_local_coefficients.csv
         results/tidy/sensor_separability.csv
         results/tidy/spectral_signature.csv
"""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from p00_config import (BAND_LABELS, BANDS, FEATURES, FEATURES_RATIO_ONLY,
                        FEATURES_VIS, HLS_MSI_TO_OLI, MET, MSI_SOURCE, N_SPLITS,
                        PROC, SEED, TIDY, add_indices, apply_local_recal, banner,
                        fit_local_recal, harmonize, paired_events)


def load():
    """Dataset de analisis (ambos sensores en espacio OLI, HLS aplicado) y la
    misma seleccion de filas con Sentinel-2 SIN ajustar, para medir el ajuste."""
    d = pd.read_csv(PROC / "analysis_dataset.csv")
    raw = []
    for fn, s in [("matchups_s2.csv", "S2"), ("matchups_ls.csv", "LS")]:
        t = pd.read_csv(PROC / fn)
        t = t[t.matched == 1].copy()
        t["sensor"] = s
        raw.append(t)
    raw = pd.concat(raw, ignore_index=True)
    un = harmonize(raw, adjust_msi=False)
    un["campaign_date"] = pd.to_datetime(un["date"]).dt.strftime("%Y-%m-%d")
    un = add_indices(un).merge(d[["row_id", "sensor"]], on=["row_id", "sensor"])
    return d, un


# --------------------------------------------------------------------------
def part_a_bandpass_vs_gap(d, un):
    banner("(A) CUANTO MUEVE HLS A SENTINEL-2 Y CUANTO QUEDA ENTRE SENSORES")
    piv = paired_events(d)
    piv_un = paired_events(un)
    print(f"  eventos pareados (misma estacion y fecha de muestreo): {len(piv)}\n")
    print(f"  {'Banda':7s} {'pend.':>7s} {'icpt':>8s} {'LS nat':>8s} {'S2 crudo':>9s} "
          f"{'ajuste %':>9s} {'LS/S2 par':>10s} {'r par':>7s}")
    rows = []
    for b in BANDS:
        slope, icpt = HLS_MSI_TO_OLI[MSI_SOURCE[b]]
        ls = d.loc[d.sensor == "LS", b]
        s2_raw = un.loc[un.sensor == "S2", b]
        s2_hls = d.loc[d.sensor == "S2", b]
        shift = float((s2_hls.median() - s2_raw.median()) / s2_raw.median() * 100)
        x, y = piv[(b, "S2")].values, piv[(b, "LS")].values
        ratio = float(np.median(y) / np.median(x))
        ratio_raw = float(np.median(piv_un[(b, "LS")]) / np.median(piv_un[(b, "S2")]))
        r = float(np.corrcoef(x, y)[0, 1])
        rows.append({"band": b, "band_label": BAND_LABELS[b],
                     "msi_band": MSI_SOURCE[b],
                     "hls_slope": round(slope, 5), "hls_intercept": round(icpt, 5),
                     "native_ls_median": round(float(ls.median()), 6),
                     "s2_raw_median": round(float(s2_raw.median()), 6),
                     "s2_hls_median": round(float(s2_hls.median()), 6),
                     "hls_shift_pct": round(shift, 1),
                     "intercept_over_s2_signal": round(abs(icpt) / float(s2_raw.median()), 2),
                     "paired_ls_over_s2_raw": round(ratio_raw, 2),
                     "paired_ls_over_s2_hls": round(ratio, 2),
                     "paired_r": round(r, 3), "n_paired": int(len(piv))})
        print(f"  {BAND_LABELS[b]:7s} {slope:7.4f} {icpt:+8.5f} {ls.median():8.5f} "
              f"{s2_raw.median():9.5f} {shift:+8.1f}% {ratio:10.2f} {r:+7.3f}")
    print("\n  HLS mueve Sentinel-2 unos pocos por ciento; la diferencia entre")
    print("  sensores sobre el mismo evento es de otro orden. Lo que los separa sobre")
    print("  agua oscura no es la respuesta espectral sino el procesamiento.")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
def sep_test(df, feats, groups, label_col="sensor", positive="S2", recal=False):
    """Clasificador de dos muestras con validacion agrupada. Devuelve exactitud,
    exactitud balanceada, AUC y el nivel de azar (clase mayoritaria)."""
    df = df.reset_index(drop=True)
    y = (df[label_col] == positive).astype(int).values
    proba = np.zeros(len(df))
    for tr, te in GroupKFold(N_SPLITS).split(df, y, groups):
        trn, tst = df.iloc[tr], df.iloc[te]
        if recal:
            coefs, _ = fit_local_recal(trn)
            trn, tst = apply_local_recal(trn, coefs), apply_local_recal(tst, coefs)
        clf = RandomForestClassifier(300, random_state=SEED, n_jobs=-1)
        clf.fit(trn[feats].values, y[tr])
        proba[te] = clf.predict_proba(tst[feats].values)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {"n": int(len(df)), "chance": round(float(max(y.mean(), 1 - y.mean())), 4),
            "accuracy": round(float(accuracy_score(y, pred)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
            "auc": round(float(roc_auc_score(y, proba)), 4)}


def pairs_from_p11(window=3):
    """Pares de ambos sensores con imagen a +-window dias del evento de campo,
    desde la re-extraccion de p11 (Landsat nativo, Sentinel-2 con B8A)."""
    f = PROC / "window_sensitivity_raw.csv"
    if not f.exists():
        return None
    w = pd.read_csv(f)
    cols = {f"w{window}_{b}": b for b in BANDS + ["B8A"]}
    if not set(cols) <= set(w.columns):
        return None
    t = w.rename(columns=cols)
    t = t[t[f"w{window}_n"].fillna(0) > 0].dropna(subset=BANDS)
    t = add_indices(harmonize(t, ls_native=True)).dropna(subset=FEATURES)
    t["campaign_date"] = t["date"]
    both = t.groupby(["station", "date"]).sensor.nunique()
    keep = both[both == 2].index
    return t.set_index(["station", "date"]).loc[keep].reset_index()


def part_b_sensor_separability(d, un):
    banner("(B) PRUEBA DE DOS MUESTRAS: ¿SIGUEN SIENDO DISTINGUIBLES LOS SENSORES?")
    print("  Validacion GroupKFold(5) agrupada por fecha de muestreo; si la")
    print("  armonizacion funcionara, exactitud balanceada y AUC estarian en 0.5.\n")
    rng = np.random.default_rng(SEED)
    rows = []

    def run(key, label, frame, feats, groups, **kw):
        m = sep_test(frame, feats, groups, **kw)
        rows.append({"test": key, "label": label, **m})
        print(f"  {label:48s} n={m['n']:4d}  exact={m['accuracy'] * 100:5.1f}%  "
              f"bal={m['balanced_accuracy'] * 100:5.1f}%  AUC={m['auc']:.3f}  "
              f"(azar {m['chance'] * 100:.1f}%)")

    g = d.campaign_date.values
    run("all_hls", "Todos, HLS, 12 features", d, FEATURES, g)
    run("all_raw", "Todos, sin ajuste HLS, 12 features", un, FEATURES, un.campaign_date.values)
    run("visible_only", "Todos, HLS, solo visible", d, FEATURES_VIS, g)
    run("ratios_only", "Todos, HLS, solo ratios", d, FEATURES_RATIO_ONLY, g)
    run("nir_swir_only", "Todos, HLS, solo NIR+SWIR", d, ["B8", "B11", "B12"], g)

    piv = paired_events(d)
    pe = d.set_index(["station", "campaign_date"]).loc[piv.index].reset_index()
    run("paired_hls", "Solo eventos pareados, HLS", pe, FEATURES, pe.campaign_date.values)
    run("paired_recal", "Solo eventos pareados, recalibracion local*", pe,
        FEATURES, pe.campaign_date.values, recal=True)
    run("all_recal", "Todos, recalibracion local*", d, FEATURES, g, recal=True)

    p3 = pairs_from_p11(3)
    if p3 is not None and len(p3) >= 60:
        run("pm3d_hls", "Pares con imagen a +-3 dias (p11), HLS", p3, FEATURES,
            p3.campaign_date.values)
        run("pm3d_recal", "Pares a +-3 dias (p11), recalibracion local*", p3,
            FEATURES, p3.campaign_date.values, recal=True)
    else:
        print("  (pares a +-3 dias: la extraccion de p11 aun no esta disponible)")

    # control negativo: dos mitades al azar del mismo sensor deben dar ~50 %
    for s, name in [("LS", "Landsat"), ("S2", "Sentinel-2")]:
        one = d[d.sensor == s].copy()
        one["half"] = np.where(rng.random(len(one)) < 0.5, "A", "B")
        run(f"negative_{s.lower()}", f"Control negativo: {name} contra {name}",
            one, FEATURES, one.campaign_date.values, label_col="half", positive="A")
    print("\n  * recalibracion ajustada solo con los eventos pareados del fold de")
    print("    entrenamiento y aplicada a entrenamiento y prueba.")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
def part_c_local_recalibration(d):
    banner("(C) RECALIBRACION LOCAL SOBRE AGUA (descriptiva, todos los pares)")
    coefs, n = fit_local_recal(d)
    piv = paired_events(d)
    rows, paired_long = [], []
    print(f"  LS = a + b * S2(HLS), {n} eventos pareados\n")
    print(f"  {'Banda':7s} {'a':>10s} {'b':>8s} {'r':>7s}")
    for b in BANDS:
        x, y = piv[(b, "S2")].values, piv[(b, "LS")].values
        r = float(np.corrcoef(x, y)[0, 1])
        icpt, slope = coefs[b]
        verdict = ("unusable" if r < 0.35 else "weak" if r < 0.55 else
                   "acceptable" if r < 0.70 else "good")
        rows.append({"band": b, "band_label": BAND_LABELS[b],
                     "intercept": round(icpt, 6), "slope": round(slope, 4),
                     "pearson_r": round(r, 3), "r2": round(r * r, 3),
                     "n_paired": int(n), "verdict": verdict})
        for xi, yi in zip(x, y):
            paired_long.append({"band": b, "band_label": BAND_LABELS[b],
                                "s2_hls": xi, "ls_native": yi})
        print(f"  {BAND_LABELS[b]:7s} {icpt:+10.5f} {slope:8.4f} {r:+7.3f}  {verdict}")
    return coefs, pd.DataFrame(rows), pd.DataFrame(paired_long)


# --------------------------------------------------------------------------
def part_d_spectral_signature(d, un, coefs):
    banner("(D) FIRMA ESPECTRAL POR SENSOR EN CADA ETAPA")
    stages = [("Sentinel-2 unadjusted", un),
              ("HLS bandpass adjustment", d),
              ("Local water recalibration", apply_local_recal(d, coefs))]
    rows = []
    for tag, frame in stages:
        for b in BANDS:
            for s in ["S2", "LS"]:
                v = frame.loc[frame.sensor == s, b].dropna()
                rows.append({"stage": tag, "band": b, "band_label": BAND_LABELS[b],
                             "sensor": "Sentinel-2" if s == "S2" else "Landsat 8/9",
                             "median": round(float(v.median()), 6),
                             "q25": round(float(v.quantile(.25)), 6),
                             "q75": round(float(v.quantile(.75)), 6),
                             "n": int(len(v))})
    sig = pd.DataFrame(rows)
    for tag in sig.stage.unique():
        p = sig[sig.stage == tag].pivot_table(index="band_label", columns="sensor",
                                              values="median")
        p["LS/S2"] = (p["Landsat 8/9"] / p["Sentinel-2"]).round(2)
        print(f"\n  {tag}:")
        print(p.round(5).to_string())
    return sig


# --------------------------------------------------------------------------
def main():
    d, un = load()
    bands_df = part_a_bandpass_vs_gap(d, un)
    sep_df = part_b_sensor_separability(d, un)
    coefs, coef_df, paired_df = part_c_local_recalibration(d)
    sig_df = part_d_spectral_signature(d, un, coefs)

    bands_df.to_csv(TIDY / "harmonization_bands.csv", index=False)
    sep_df.to_csv(TIDY / "sensor_separability.csv", index=False)
    coef_df.to_csv(TIDY / "harmonization_local_coefficients.csv", index=False)
    paired_df.to_csv(TIDY / "harmonization_paired.csv", index=False)
    sig_df.to_csv(TIDY / "spectral_signature.csv", index=False)

    json.dump({"harmonization": "HLS v2.0 bandpass adjustment, MSI -> OLI (mean of S2A/S2B)",
               "bands": bands_df.to_dict("records"),
               "sensor_separability": sep_df.to_dict("records"),
               "local_coefficients": coef_df.to_dict("records"),
               "n_paired_events": int(coef_df.n_paired.iloc[0])},
              open(MET / "harmonization.json", "w"), indent=2)

    banner("SALIDAS")
    for f in ["harmonization_bands.csv", "sensor_separability.csv",
              "harmonization_local_coefficients.csv", "harmonization_paired.csv",
              "spectral_signature.csv"]:
        print(f"  {TIDY / f}")
    print(f"  {MET / 'harmonization.json'}")


if __name__ == "__main__":
    main()
