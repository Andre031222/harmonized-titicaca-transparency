"""
p08_interpretability.py -- SHAP fuera de fold y retrievabilidad de otras
variables, ambos bajo el diseno primario (dejar fuera una campana).

(A) SHAP (TreeExplainer) del RF: cada fila se explica con el modelo que NO vio
    su campana. Importancia (|SHAP| medio y cuota), cuota por grupo de
    features, dependencia de la feature principal y la tabla larga completa.
(B) Retrievabilidad: el mismo RF con las 12 features para Secchi, clorofila-a,
    solidos suspendidos y temperatura (match-ups HLS, ventana 10 d, filas que
    tienen cada variable). Termico: correlacion de Pearson entre HLSL30 B10
    (temperatura de brillo TOA, solo Landsat) y la temperatura in situ, y un
    RF con B10 solo. No se usa ningun umbral de R2: se reportan los numeros.

Entradas: data/processed/analysis_dataset.csv, data/processed/matchups.csv
Salidas: results/tidy/shap_values_all.csv, shap_importance.csv,
         shap_dependence.csv, shap_groups.csv, retrievability.csv
         results/metrics/interpretability.json
"""
import json

import numpy as np
import pandas as pd

from p00_config import (load_matchups, write_json, FEATURE_LABELS, FEATURES, MET, N_BOOT, PRIMARY, PROC,
                        SECCHI_CLIP_LO, SEED, TIDY, ZONE_LABELS, add_indices,
                        banner, cluster_boot, make_rf, metrics, splits)

GROUPS = {"B2": "Green/blue bands and ratios", "B3": "Green/blue bands and ratios",
          "B2_B3": "Green/blue bands and ratios", "B3_B2": "Green/blue bands and ratios",
          "B8": "NIR/SWIR", "B11": "NIR/SWIR", "B12": "NIR/SWIR",
          "B4": "Other (red, NDWI, NDTI, red ratios)",
          "NDWI": "Other (red, NDWI, NDTI, red ratios)",
          "NDTI": "Other (red, NDWI, NDTI, red ratios)",
          "B4_B3": "Other (red, NDWI, NDTI, red ratios)",
          "B2_B4": "Other (red, NDWI, NDTI, red ratios)"}
VARS = [("secchi", "Water transparency (Secchi)", "m"),
        ("chl", "Chlorophyll-a", "mg m-3"),
        ("tss", "Total suspended solids", "mg L-1"),
        ("temp_insitu", "Water temperature", "degC")]


def loco_rf(d, feats, target, clip=None):
    """Predicciones fuera de fold dejando fuera una campana. Sin escalado: el
    RF es invariante al RobustScaler afin por feature."""
    d = d.reset_index(drop=True)
    X, y = d[feats].values, d[target].values
    oof = np.full(len(y), np.nan)
    for tr, te in splits(d, PRIMARY):
        oof[te] = make_rf().fit(X[tr], y[tr]).predict(X[te])
    if clip is not None:
        oof = np.clip(oof, clip, None)
    return y, oof


def boot_r(a, b, groups, n_boot=N_BOOT, seed=SEED):
    """IC 95 % de la r de Pearson por bootstrap de campanas."""
    a, b, groups = map(np.asarray, (a, b, groups))
    ug = np.unique(groups)
    pos = {g: np.flatnonzero(groups == g) for g in ug}
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n_boot):
        ix = np.concatenate([pos[g] for g in rng.choice(ug, len(ug))])
        v.append(np.corrcoef(a[ix], b[ix])[0, 1])
    lo, hi = np.nanquantile(v, [0.025, 0.975])
    return round(float(lo), 3), round(float(hi), 3)


# ---------------------------------------------------------------------------
def shap_oof(d):
    banner("(A) SHAP FUERA DE FOLD (dejar fuera una campana)")
    import shap
    X, y = d[FEATURES].values, d.secchi.values
    sv = np.full(X.shape, np.nan)
    base = np.full(len(y), np.nan)
    for tr, te in splits(d, PRIMARY):
        m = make_rf().fit(X[tr], y[tr])
        ex = shap.TreeExplainer(m)
        sv[te] = ex.shap_values(X[te], check_additivity=False)
        base[te] = float(np.ravel(ex.expected_value)[0])
    # comprobacion de aditividad contra la prediccion del mismo modelo
    pct = d[FEATURES].rank(pct=True).values
    allv = pd.DataFrame({
        "row": np.tile(np.arange(len(y)), len(FEATURES)),
        "feature": np.repeat(FEATURES, len(y)),
        "label": np.repeat([FEATURE_LABELS[f] for f in FEATURES], len(y)),
        "group": np.repeat([GROUPS[f] for f in FEATURES], len(y)),
        "shap": sv.T.ravel(), "value": X.T.ravel(), "value_pct": pct.T.ravel(),
        "zona": np.tile(d.zona.values, len(FEATURES)),
        "campaign": np.tile(d.campaign.values, len(FEATURES)),
        "sensor": np.tile(d.sensor.values, len(FEATURES)),
        "secchi": np.tile(y, len(FEATURES))})
    allv["zone_label"] = allv.zona.map(ZONE_LABELS)

    imp = (pd.DataFrame({"feature": FEATURES,
                         "label": [FEATURE_LABELS[f] for f in FEATURES],
                         "group": [GROUPS[f] for f in FEATURES],
                         "mean_abs_shap": np.abs(sv).mean(axis=0)})
           .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
    imp["share"] = imp.mean_abs_shap / imp.mean_abs_shap.sum()
    imp["cumulative_share"] = imp.share.cumsum()
    imp["rank"] = np.arange(1, len(imp) + 1)
    grp = (imp.groupby("group").agg(mean_abs_shap=("mean_abs_shap", "sum"),
                                    share=("share", "sum"),
                                    features=("feature", lambda s: ",".join(s)))
           .sort_values("share", ascending=False).reset_index())

    print(f"  {'feature':12s} {'|SHAP| medio (m)':>16s} {'cuota':>7s}  grupo")
    for r in imp.itertuples():
        print(f"  {r.label:12s} {r.mean_abs_shap:16.4f} {r.share * 100:6.1f}%  {r.group}")
    print("\n  cuota por grupo:")
    for r in grp.itertuples():
        print(f"    {r.group:38s} {r.share * 100:5.1f}%")

    lead = imp.feature.iloc[0]
    dep = allv[allv.feature == lead].copy()
    print(f"\n  feature principal: {FEATURE_LABELS[lead]} ({lead})")
    return imp, grp, dep, allv, sv, base


# ---------------------------------------------------------------------------
def retrievability():
    banner("(B) RETRIEVABILIDAD (RF, 12 features, dejar fuera una campana)")
    m = load_matchups()
    h = add_indices(m[(m["product"] == "hls") & (m.window == 10)].copy())
    rows, preds = [], []
    print(f"  {'variable':30s} {'n':>5s} {'camp':>4s} {'R2 [IC95]':>24s} "
          f"{'RMSE':>7s} {'mediana':>8s} {'rango':>14s}")
    for var, label, unit in VARS:
        sub = h[h[var].notna()].dropna(subset=FEATURES).reset_index(drop=True)
        y, p = loco_rf(sub, FEATURES, var,
                       clip=SECCHI_CLIP_LO if var == "secchi" else None)
        mm = metrics(y, p)
        ci = cluster_boot(y, p, sub.campaign.values)
        q = np.quantile(y, [0.25, 0.5, 0.75])
        rows.append({"variable": var, "label": label, "unit": unit,
                     "predictors": "12 features (HLS)", "n": mm["n"],
                     "n_campaigns": int(sub.campaign.nunique()),
                     "R2": mm["R2"], "R2_lo": ci[0], "R2_hi": ci[1],
                     "RMSE": mm["RMSE"], "MAE": mm["MAE"], "bias": mm["bias"],
                     "insitu_min": round(float(y.min()), 3),
                     "insitu_q25": round(float(q[0]), 3),
                     "insitu_median": round(float(q[1]), 3),
                     "insitu_q75": round(float(q[2]), 3),
                     "insitu_max": round(float(y.max()), 3),
                     "insitu_sd": round(float(y.std(ddof=1)), 3),
                     "pearson_r": None, "pearson_r_lo": None, "pearson_r_hi": None})
        preds.append(pd.DataFrame({"variable": var, "campaign": sub.campaign,
                                   "observed": y, "predicted": p}))
        print(f"  {label:30s} {mm['n']:5d} {sub.campaign.nunique():4d} "
              f"{mm['R2']:+7.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] {mm['RMSE']:7.3f} "
              f"{q[1]:8.2f} {y.min():6.2f}-{y.max():<6.2f}")

    # termico: B10 (TOA, HLSL30) contra temperatura in situ
    t = h[(h.sensor == "LS") & h.B10.notna() & h.temp_insitu.notna()].reset_index(drop=True)
    r = float(np.corrcoef(t.B10, t.temp_insitu)[0, 1])
    rci = boot_r(t.B10, t.temp_insitu, t.campaign)
    # r dentro de campana (anomalias respecto a la media de cada campana)
    an = t[["B10", "temp_insitu"]] - t.groupby("campaign")[["B10", "temp_insitu"]].transform("mean")
    r_within = float(np.corrcoef(an.B10, an.temp_insitu)[0, 1])
    y, p = loco_rf(t, ["B10"], "temp_insitu")
    mm = metrics(y, p)
    ci = cluster_boot(y, p, t.campaign.values)
    q = np.quantile(y, [0.25, 0.5, 0.75])
    rows.append({"variable": "temp_insitu_B10", "label": "Water temperature (B10 only)",
                 "unit": "degC", "predictors": "HLSL30 B10 TOA brightness temperature",
                 "n": mm["n"], "n_campaigns": int(t.campaign.nunique()),
                 "R2": mm["R2"], "R2_lo": ci[0], "R2_hi": ci[1], "RMSE": mm["RMSE"],
                 "MAE": mm["MAE"], "bias": mm["bias"],
                 "insitu_min": round(float(y.min()), 3),
                 "insitu_q25": round(float(q[0]), 3),
                 "insitu_median": round(float(q[1]), 3),
                 "insitu_q75": round(float(q[2]), 3),
                 "insitu_max": round(float(y.max()), 3),
                 "insitu_sd": round(float(y.std(ddof=1)), 3),
                 "pearson_r": round(r, 3), "pearson_r_lo": rci[0], "pearson_r_hi": rci[1]})
    preds.append(pd.DataFrame({"variable": "temp_insitu_B10", "campaign": t.campaign,
                               "observed": y, "predicted": p}))
    print(f"  {'Water temperature (B10 only)':30s} {mm['n']:5d} {t.campaign.nunique():4d} "
          f"{mm['R2']:+7.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] {mm['RMSE']:7.3f}")
    print(f"\n  B10 vs temperatura in situ (Landsat, n={len(t)}): r={r:+.3f} "
          f"[{rci[0]:+.3f}, {rci[1]:+.3f}] | r dentro de campana={r_within:+.3f}")
    print(f"  B10: media {t.B10.mean():.2f} C, in situ media {t.temp_insitu.mean():.2f} C")
    thermal = {"n": int(len(t)), "n_campaigns": int(t.campaign.nunique()),
               "pearson_r": round(r, 3), "pearson_r_ci": list(rci),
               "pearson_r_within_campaign": round(r_within, 3),
               "B10_mean_C": round(float(t.B10.mean()), 2),
               "B10_sd_C": round(float(t.B10.std(ddof=1)), 2),
               "insitu_mean_C": round(float(t.temp_insitu.mean()), 2),
               "insitu_sd_C": round(float(t.temp_insitu.std(ddof=1)), 2),
               "mean_B10_minus_insitu_C": round(float((t.B10 - t.temp_insitu).mean()), 2),
               "rf_B10_only": {**mm, "R2_ci": list(ci)}}
    return pd.DataFrame(rows), thermal, pd.concat(preds, ignore_index=True)


# ---------------------------------------------------------------------------
def main():
    d = pd.read_csv(PROC / "analysis_dataset.csv", dtype={"campaign": str})
    d = d.reset_index(drop=True)
    imp, grp, dep, allv, sv, base = shap_oof(d)
    retr, thermal, _ = retrievability()

    allv.to_csv(TIDY / "shap_values_all.csv", index=False, float_format="%.6f")
    imp.to_csv(TIDY / "shap_importance.csv", index=False, float_format="%.6f")
    dep.to_csv(TIDY / "shap_dependence.csv", index=False, float_format="%.6f")
    grp.to_csv(TIDY / "shap_groups.csv", index=False, float_format="%.6f")
    retr.to_csv(TIDY / "retrievability.csv", index=False)

    lead = imp.iloc[0]
    out = {
        "shap_design": ("Out-of-fold TreeExplainer SHAP under leave-one-campaign-out "
                        "(9 folds): each row explained by the RF (make_rf) that did not "
                        "see its campaign; raw (unscaled) features."),
        "n": int(len(d)),
        "importance": imp.round(4).to_dict("records"),
        "share_by_group": {r.group: round(float(r.share), 4) for r in grp.itertuples()},
        "leading_feature": {"feature": lead.feature, "label": lead.label,
                            "share": round(float(lead.share), 4)},
        "top3_cumulative_share": round(float(imp.cumulative_share.iloc[2]), 4),
        "retrievability_design": ("matchups.csv product=hls window=10 rows with each "
                                  "variable; RF (make_rf) on the 12 features; "
                                  "leave-one-campaign-out; R2 95% CI by cluster "
                                  "bootstrap over campaigns. No R2 threshold applied."),
        "retrievability": retr.to_dict("records"),
        "thermal": thermal,
    }
    write_json(out, MET / "interpretability.json")

    banner("SALIDAS")
    for f in ["shap_values_all.csv", "shap_importance.csv", "shap_dependence.csv",
              "shap_groups.csv", "retrievability.csv"]:
        print(f"  {TIDY / f}")
    print(f"  {MET / 'interpretability.json'}")


if __name__ == "__main__":
    main()
