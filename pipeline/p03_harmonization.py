"""
p03_harmonization.py -- son intercambiables Landsat 8/9 y Sentinel-2 sobre el
agua del Titicaca despues de armonizar?

Prueba de dos muestras con clasificador (bosque aleatorio, 300 arboles, 12
features) en tres vias:
  hls    : producto NASA HLS v2.0 (LaSRC en ambos sensores, BRDF, ajuste de
           banda, grilla comun). Via principal.
  sr     : reflectancia de superficie de cada agencia (Sen2Cor, Landsat C2) con
           solo los coeficientes de banda de HLS aplicados a Sentinel-2.
  sr_raw : la misma sin coeficientes (que hacen los coeficientes por si solos).

Pruebas (validacion dejando fuera una campana; la disponibilidad de S2 esta
confundida con la campana, S2 falta en 2013-2015):
  all (+-10 d; tambien agrupada por fecha de muestreo), eventos pareados
  (misma ubicacion y fecha vistas por ambos), pares casi simultaneos (+-3 d y
  una sola imagen por sensor; +-1 d si hay >= 40 pares), recalibracion por
  banda S2 -> LS ajustada dentro de cada fold (OLS, eje mayor reducido,
  mapeo de cuantiles por banda y por feature), subconjuntos de features (hls).
  IC 95 % de BA y AUC: bootstrap de conglomerados por campana.
  Controles negativos: dentro de un sensor, mitades al azar de fechas de
  muestreo o de filas, 100 repeticiones -> distribucion de BA y AUC.
Descriptivos por via y banda (eventos pareados): LS/S2, r, diferencia relativa,
% de reflectancia <= 0, y cuanto mueven los coeficientes de HLS al agua de S2.

Entrada : data/processed/matchups.csv
Salidas : results/tidy/harmonization_separability.csv, harmonization_controls.csv,
          harmonization_control_draws.csv, harmonization_bands.csv,
          harmonization_bandpass_shift.csv, harmonization_local_coefficients.csv,
          harmonization_paired.csv, spectral_signature.csv
          results/metrics/harmonization.json
"""
import json
from functools import partial

import numpy as np
import pandas as pd

from harmonization_check import (negative_controls, summarize_controls,
                                 two_sample_test)
from p00_config import (load_matchups, BAND_LABELS, BANDS, FEATURES, FEATURES_RATIO_ONLY,
                        FEATURES_VIS, HLS_MSI_TO_OLI, MET, MSI_SOURCE, N_BOOT,
                        PROC, SEED, TIDY, add_indices, bandpass_adjust, banner,
                        splits)

W_ALL, W_NEAR, W_SIM = 10, 3, 1
MIN_PAIRS = 40
N_CONTROLS = 100
ROUTES = ["hls", "sr", "sr_raw"]
ROUTE_LABELS = {"hls": "NASA HLS v2.0 product",
                "sr": "Agency SR + HLS bandpass coefficients",
                "sr_raw": "Agency SR, no adjustment"}
FEATSETS = {"all12": FEATURES, "visible": FEATURES_VIS,
            "ratios": FEATURES_RATIO_ONLY, "nir_swir": ["B8", "B11", "B12"]}
RECALS = ["ols", "rma", "qm_bands", "qm_features"]
SENSOR_LABELS = {"S2": "Sentinel-2", "LS": "Landsat 8/9"}

M = load_matchups()


# --- datos -------------------------------------------------------------------
def route_table(route, window):
    prod = "hls" if route == "hls" else "sr"
    m = M[(M["product"] == prod) & (M.window == window)].copy()
    if route != "hls":
        m = bandpass_adjust(m, adjust=(route == "sr"))
    m = add_indices(m).dropna(subset=FEATURES)
    m["event"] = m.location + "|" + m.sampling_date
    return m.reset_index(drop=True)


def paired(d):
    ev = d.groupby("event").sensor.nunique()
    return d[d.event.isin(ev[ev == 2].index)].reset_index(drop=True)


def single_image_pairs(d):
    return paired(d[d.n_img == 1])


def pivot(d, cols):
    return paired(d).pivot(index="event", columns="sensor", values=cols)


# --- recalibracion S2 -> LS por banda ------------------------------------------
def fit_maps(train, kind):
    """Mapa por columna ajustado en los eventos pareados de `train`."""
    cols = FEATURES if kind == "qm_features" else BANDS
    piv = pivot(train, cols)
    maps = {}
    for c in cols:
        x, y = piv[(c, "S2")].values, piv[(c, "LS")].values
        if kind == "ols":
            b = np.cov(x, y)[0, 1] / np.var(x, ddof=1)
            maps[c] = ("lin", y.mean() - b * x.mean(), b)
        elif kind == "rma":   # eje mayor reducido: iguala media y varianza
            b = np.sign(np.corrcoef(x, y)[0, 1]) * y.std(ddof=1) / x.std(ddof=1)
            maps[c] = ("lin", y.mean() - b * x.mean(), b)
        else:                 # mapeo de cuantiles (monotono)
            q = np.linspace(0, 1, 101)
            maps[c] = ("qm", np.quantile(x, q), np.quantile(y, q))
    return maps, kind


def apply_maps(d, fitted):
    maps, kind = fitted
    d = d.copy()
    s = (d.sensor == "S2").values
    for c, (typ, a, b) in maps.items():
        v = d.loc[s, c].values
        d.loc[s, c] = a + b * v if typ == "lin" else np.interp(v, a, b)
    return d if kind == "qm_features" else add_indices(d)


def recal_transform(train, test, kind):
    f = fit_maps(train, kind)
    return apply_maps(train, f), apply_maps(test, f)


# --- pruebas -------------------------------------------------------------------
ROWS = []


def run(route, test, data, grouping="campaign", featset="all12", recal="none"):
    data = data.reset_index(drop=True)
    tf = None if recal == "none" else partial(recal_transform, kind=recal)
    res, _ = two_sample_test(data, FEATSETS[featset], "sensor", "S2",
                             data.campaign.values, folds=splits(data, grouping),
                             transform=tf, n_boot=N_BOOT, seed=SEED)
    row = {"route": route, "route_label": ROUTE_LABELS[route], "test": test,
           "grouping": grouping, "featset": featset, "recal": recal,
           "n": res["n"], "n_s2": res["n_positive"], "n_ls": res["n"] - res["n_positive"],
           "n_events_paired": int(paired(data).event.nunique()),
           "n_campaigns": res["n_groups"], "n_folds": res["n_folds"],
           "chance": round(res["chance"], 3),
           "balanced_accuracy": round(res["balanced_accuracy"], 3),
           "ba_lo": round(res["ba_lo"], 3), "ba_hi": round(res["ba_hi"], 3),
           "auc": round(res["auc"], 3),
           "auc_lo": round(res["auc_lo"], 3), "auc_hi": round(res["auc_hi"], 3),
           "accuracy": round(res["accuracy"], 3)}
    ROWS.append(row)
    print(f"  {route:6s} {test:11s} {grouping:13s} {featset:8s} {recal:11s} "
          f"n={row['n']:4d}  BA={row['balanced_accuracy']:.3f} "
          f"[{row['ba_lo']:.3f},{row['ba_hi']:.3f}]  AUC={row['auc']:.3f} "
          f"[{row['auc_lo']:.3f},{row['auc_hi']:.3f}]")
    return row


def separability(tables):
    banner("PRUEBA DE DOS MUESTRAS (BA, AUC; IC 95 % bootstrap por campana)")
    counts = {}
    for r in ROUTES:
        d, d3, d1 = tables[r]
        pr, near = paired(d), single_image_pairs(d3)
        sim_any, sim_one = paired(d1), single_image_pairs(d1)
        counts[r] = {"n_all": len(d), "n_s2": int((d.sensor == "S2").sum()),
                     "n_ls": int((d.sensor == "LS").sum()),
                     "n_campaigns_all": int(d.campaign.nunique()),
                     "n_paired_events": int(pr.event.nunique()),
                     "n_pairs_pm3d_any_nimg": int(paired(d3).event.nunique()),
                     "n_pairs_pm3d_single_image": int(near.event.nunique()),
                     "n_campaigns_pm3d_single_image": int(near.campaign.nunique()),
                     "n_pairs_pm1d_any_nimg": int(sim_any.event.nunique()),
                     "n_pairs_pm1d_single_image": int(sim_one.event.nunique())}
        run(r, "all", d)
        run(r, "all", d, grouping="sampling_date")
        run(r, "paired", pr)
        run(r, "paired", pr, grouping="sampling_date")
        run(r, "pm3d", near)
        if sim_one.event.nunique() >= MIN_PAIRS:
            run(r, "pm1d", sim_one)
        elif sim_any.event.nunique() >= MIN_PAIRS:
            run(r, "pm1d_anyimg", sim_any)
        if r == "sr_raw":
            continue
        for k in RECALS:
            run(r, "paired", pr, recal=k)
            run(r, "pm3d", near, recal=k)
        if r == "hls":
            for fs in ["visible", "ratios", "nir_swir"]:
                run(r, "all", d, featset=fs)
                run(r, "paired", pr, featset=fs)
                run(r, "pm3d", near, featset=fs)
    return pd.DataFrame(ROWS), counts


def controls(tables):
    banner(f"CONTROLES NEGATIVOS ({N_CONTROLS} repeticiones, folds por campana)")
    summ, draws = [], []
    for r in ["hls", "sr"]:
        d, d3, _ = tables[r]
        sets = {"all": d, "pm3d": single_image_pairs(d3)}
        for sample, data in sets.items():
            for s in ["LS", "S2"]:
                one = data[data.sensor == s].reset_index(drop=True)
                for kind, unit in [("dates", "sampling_date"), ("rows", None)]:
                    if sample == "pm3d" and kind == "rows":
                        continue
                    ba, auc = negative_controls(one, FEATURES, one.campaign.values,
                                                unit, N_CONTROLS, SEED,
                                                folds=splits(one, "campaign"))
                    c = summarize_controls(ba, auc)
                    summ.append({"route": r, "sample": sample, "sensor": s,
                                 "control": kind, "n": len(one),
                                 "n_units": int(one[unit].nunique()) if unit else len(one),
                                 "n_campaigns": int(one.campaign.nunique()),
                                 **{k: (round(v, 3) if isinstance(v, float) else v)
                                    for k, v in c.items()}})
                    draws += [{"route": r, "sample": sample, "sensor": s,
                               "control": kind, "rep": i, "balanced_accuracy": b,
                               "auc": a} for i, (b, a) in enumerate(zip(ba, auc))]
                    x = summ[-1]
                    print(f"  {r:4s} {sample:5s} {s} {kind:5s} n={len(one):4d}  "
                          f"BA {x['ba_mean']:.3f} [{x['ba_lo']:.3f},{x['ba_hi']:.3f}]  "
                          f"AUC {x['auc_mean']:.3f} [{x['auc_lo']:.3f},{x['auc_hi']:.3f}]")
    return pd.DataFrame(summ), pd.DataFrame(draws)


# --- descriptivos ----------------------------------------------------------------
def band_stats(tables):
    banner("DESCRIPTIVOS POR BANDA (LS frente a S2)")
    rows, long = [], []
    for r in ROUTES:
        d, d3, _ = tables[r]
        for b in BANDS:
            ls, s2 = d.loc[d.sensor == "LS", b], d.loc[d.sensor == "S2", b]
            rows.append({"route": r, "sample": "all_w10", "band": b,
                         "band_label": BAND_LABELS[b], "n": len(d),
                         "ls_median": ls.median(), "s2_median": s2.median(),
                         "ratio_ls_s2": ls.median() / s2.median(),
                         "pct_ls_le0": 100 * (ls <= 0).mean(),
                         "pct_s2_le0": 100 * (s2 <= 0).mean()})
        for sample, data in [("paired_w10", paired(d)),
                             ("pm3d_single", single_image_pairs(d3))]:
            piv = pivot(data, BANDS)
            camp = data.drop_duplicates("event").set_index("event").campaign
            for b in BANDS:
                x, y = piv[(b, "S2")], piv[(b, "LS")]
                rel = (y - x) / x
                rows.append({"route": r, "sample": sample, "band": b,
                             "band_label": BAND_LABELS[b], "n": len(piv),
                             "ls_median": y.median(), "s2_median": x.median(),
                             "ratio_ls_s2": y.median() / x.median(),
                             "pearson_r": float(np.corrcoef(x, y)[0, 1]),
                             "spearman_r": float(x.rank().corr(y.rank())),
                             "median_rel_diff_pct": 100 * float(rel.median()),
                             "median_abs_diff": float((y - x).median()),
                             "pct_ls_le0": 100 * (y <= 0).mean(),
                             "pct_s2_le0": 100 * (x <= 0).mean()})
                long += [{"route": r, "sample": sample, "event": e,
                          "campaign": camp[e], "band": b, "band_label": BAND_LABELS[b],
                          "s2": xi, "ls": yi} for e, xi, yi in zip(piv.index, x, y)]
    out = pd.DataFrame(rows)
    num = out.select_dtypes("number").columns.drop("n")
    out[num] = out[num].round(6)
    for c in ["ratio_ls_s2", "pearson_r", "spearman_r", "median_rel_diff_pct",
              "pct_ls_le0", "pct_s2_le0"]:
        out[c] = out[c].round(3)
    show = out[out["sample"] == "paired_w10"]
    print(show[["route", "band_label", "n", "ls_median", "s2_median", "ratio_ls_s2",
                "pearson_r", "median_rel_diff_pct"]].to_string(index=False))
    return out, pd.DataFrame(long)


def bandpass_shift(tables):
    """Cuanto mueven los coeficientes de HLS a la reflectancia Sen2Cor sobre agua."""
    banner("CUANTO MUEVEN LOS COEFICIENTES DE HLS A SENTINEL-2 (agua)")
    raw, adj = tables["sr_raw"][0], tables["sr"][0]
    praw, padj = pivot(raw, BANDS), pivot(adj, BANDS)
    rows = []
    for b in BANDS:
        slope, icpt = HLS_MSI_TO_OLI[MSI_SOURCE[b]]
        s_raw = raw.loc[raw.sensor == "S2", b]
        s_adj = adj.loc[adj.sensor == "S2", b]
        ls = raw.loc[raw.sensor == "LS", b]
        gap_raw = float((praw[(b, "LS")] - praw[(b, "S2")]).median())
        gap_adj = float((padj[(b, "LS")] - padj[(b, "S2")]).median())
        rows.append({"band": b, "band_label": BAND_LABELS[b], "msi_band": MSI_SOURCE[b],
                     "hls_slope": round(slope, 5), "hls_intercept": round(icpt, 5),
                     "s2_sen2cor_median": round(float(s_raw.median()), 6),
                     "s2_adjusted_median": round(float(s_adj.median()), 6),
                     "ls_c2_median": round(float(ls.median()), 6),
                     "median_shift_pct": round(100 * float((s_adj / s_raw - 1).median()), 1),
                     "shift_of_median_pct": round(100 * float(s_adj.median() / s_raw.median() - 1), 1),
                     "abs_intercept_over_s2_median": round(abs(icpt) / float(s_raw.median()), 3),
                     "paired_gap_raw": round(gap_raw, 6),
                     "paired_gap_adjusted": round(gap_adj, 6),
                     "paired_gap_closed_pct": round(100 * (1 - abs(gap_adj) / abs(gap_raw)), 1),
                     "n_s2": int(len(s_raw)), "n_paired": int(len(praw))})
    out = pd.DataFrame(rows)
    print(out[["band_label", "hls_slope", "hls_intercept", "s2_sen2cor_median",
               "median_shift_pct", "abs_intercept_over_s2_median",
               "paired_gap_raw", "paired_gap_adjusted", "paired_gap_closed_pct"]]
          .to_string(index=False))
    return out


def local_coefficients(tables):
    """Recalibracion S2 -> LS ajustada con TODOS los eventos pareados (descriptiva)."""
    rows, fitted = [], {}
    for r in ROUTES:
        d = tables[r][0]
        piv = pivot(d, BANDS)
        for kind in ["ols", "rma"]:
            maps, _ = fit_maps(d, kind)
            fitted[(r, kind)] = (maps, kind)
            for b in BANDS:
                _, a, s = maps[b]
                rows.append({"route": r, "method": kind, "band": b,
                             "band_label": BAND_LABELS[b], "intercept": round(a, 6),
                             "slope": round(s, 4),
                             "pearson_r": round(float(np.corrcoef(piv[(b, "S2")],
                                                                  piv[(b, "LS")])[0, 1]), 3),
                             "n_paired": int(len(piv))})
    return pd.DataFrame(rows), fitted


def spectral_signature(tables, fitted):
    stages = [(r, r, ROUTE_LABELS[r], tables[r][0]) for r in ROUTES]
    for r in ["hls", "sr"]:
        stages.append((r, f"{r}_ols", ROUTE_LABELS[r] + " + local OLS recalibration",
                       apply_maps(tables[r][0], fitted[(r, "ols")])))
    rows = []
    for route, stage, label, frame in stages:
        for sample, data in [("all", frame), ("paired", paired(frame))]:
            for s in ["S2", "LS"]:
                for b in BANDS:
                    v = data.loc[data.sensor == s, b]
                    rows.append({"route": route, "stage": stage, "stage_label": label,
                                 "sample": sample, "sensor": s,
                                 "sensor_label": SENSOR_LABELS[s], "band": b,
                                 "band_label": BAND_LABELS[b],
                                 "median": round(float(v.median()), 6),
                                 "q25": round(float(v.quantile(.25)), 6),
                                 "q75": round(float(v.quantile(.75)), 6),
                                 "n": int(len(v))})
    return pd.DataFrame(rows)


# --- principal -------------------------------------------------------------------
def main():
    banner("p03 -- ARMONIZACION: PRUEBA DE DOS MUESTRAS LANDSAT 8/9 vs SENTINEL-2")
    tables = {r: (route_table(r, W_ALL), route_table(r, W_NEAR), route_table(r, W_SIM))
              for r in ROUTES}
    sep, counts = separability(tables)
    ctrl, draws = controls(tables)
    bands, paired_long = band_stats(tables)
    shift = bandpass_shift(tables)
    coefs, fitted = local_coefficients(tables)
    sig = spectral_signature(tables, fitted)

    # cada prueba real frente a la banda superior (97.5 %) de los controles por
    # fechas del mismo tamano de muestra (all o pares casi simultaneos)
    ref = ctrl[ctrl.control == "dates"].groupby(["route", "sample"]).ba_hi.max()
    sep["control_ba_hi_dates"] = [
        ref[("sr" if r == "sr_raw" else r, "pm3d" if t.startswith("pm") else "all")]
        for r, t in zip(sep.route, sep.test)]
    sep["exceeds_control"] = sep.balanced_accuracy > sep.control_ba_hi_dates

    sep.to_csv(TIDY / "harmonization_separability.csv", index=False)
    ctrl.to_csv(TIDY / "harmonization_controls.csv", index=False)
    draws.round(4).to_csv(TIDY / "harmonization_control_draws.csv", index=False)
    bands.to_csv(TIDY / "harmonization_bands.csv", index=False)
    shift.to_csv(TIDY / "harmonization_bandpass_shift.csv", index=False)
    coefs.to_csv(TIDY / "harmonization_local_coefficients.csv", index=False)
    paired_long.to_csv(TIDY / "harmonization_paired.csv", index=False, float_format="%.6g")
    sig.to_csv(TIDY / "spectral_signature.csv", index=False)

    keyed = {}
    for x in sep.to_dict("records"):
        k = f"{x['route']}|{x['test']}|{x['grouping']}|{x['featset']}|{x['recal']}"
        keyed[k] = {c: x[c] for c in ["n", "n_s2", "n_ls", "n_events_paired",
                                      "n_campaigns", "balanced_accuracy", "ba_lo",
                                      "ba_hi", "auc", "auc_lo", "auc_hi", "accuracy",
                                      "chance", "exceeds_control"]}
    red = bands[(bands["sample"] == "all_w10") & (bands.band == "B4")]
    out = {
        "design": {"classifier": "RandomForestClassifier, 300 trees, seed 42",
                   "features": FEATURES, "window_all_days": W_ALL,
                   "window_near_days": W_NEAR, "window_sim_days": W_SIM,
                   "near_rule": "both sensors with a clear image within the window and "
                                "n_img == 1 for both",
                   "min_pairs_to_test": MIN_PAIRS,
                   "primary_grouping": "leave-one-campaign-out",
                   "ci": f"cluster bootstrap over campaigns of OOF predictions, "
                         f"{N_BOOT} draws",
                   "n_controls": N_CONTROLS,
                   "recalibration": {"ols": "per-band OLS S2->LS",
                                     "rma": "per-band reduced major axis (matches "
                                            "mean and variance)",
                                     "qm_bands": "per-band quantile mapping, indices "
                                                 "recomputed",
                                     "qm_features": "quantile mapping of all 12 features",
                                     "fit": "inside each training fold, on its paired "
                                            "events; applied to S2 rows of train and test"}},
        "counts": counts,
        "tests": keyed,
        "controls": ctrl.to_dict("records"),
        "pct_red_le0_all_w10": {r["route"]: {"LS": r["pct_ls_le0"], "S2": r["pct_s2_le0"]}
                                for r in red.to_dict("records")},
        "bands": bands.to_dict("records"),
        "bandpass_shift_sr": shift.to_dict("records"),
        "local_coefficients": coefs.to_dict("records"),
    }
    (MET / "harmonization.json").write_text(json.dumps(out, indent=2, default=float))

    banner("RESUMEN (12 features, campana)")
    cols = ["route", "test", "grouping", "recal", "n", "n_events_paired",
            "balanced_accuracy", "ba_lo", "ba_hi", "auc", "auc_lo", "auc_hi",
            "exceeds_control"]
    print(sep[sep.featset == "all12"][cols].to_string(index=False))
    banner("SALIDAS")
    for f in ["harmonization_separability", "harmonization_controls",
              "harmonization_control_draws", "harmonization_bands",
              "harmonization_bandpass_shift", "harmonization_local_coefficients",
              "harmonization_paired", "spectral_signature"]:
        print(f"  {TIDY / (f + '.csv')}")
    print(f"  {MET / 'harmonization.json'}")


if __name__ == "__main__":
    main()
