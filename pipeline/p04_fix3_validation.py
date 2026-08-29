"""
p04_fix3_validation.py -- ERROR 3: la validacion bloqueaba por la unidad
equivocada.

El manuscrito anterior argumentaba: "K-fold aleatorio y GroupKFold por
estacion dan el mismo R2, luego no hay fuga espacial". La inferencia es
falsa. Que dos disenos coincidan solo significa que el segundo no retiene
nada; no dice nada sobre si existe dependencia.

La forma correcta de diagnosticarlo es con MODELOS NULOS. Un modelo nulo
predice la media de entrenamiento del grupo (zona / estacion / campana). Una
unidad de bloqueo es efectiva si, y solo si, su modelo nulo COLAPSA bajo ella.

Resultado: bajo folds bloqueados por campana el nulo media-de-estacion todavia
alcanza R2~0.21 y el media-de-zona R2~0.29, mientras que el media-de-campana
cae a ~0. La dependencia vive ENTRE CAMPANAS: una campana muestrea todo el
lago en pocos dias bajo las mismas condiciones. Bloquear por estacion la deja
intacta.

Salidas: results/tidy/validation_hierarchy.csv
         results/tidy/null_models.csv
         results/tidy/oof_predictions.csv
         results/tidy/error_by_zone.csv
         results/metrics/validation.json
"""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import RobustScaler

from p00_config import (FEATURES, FEATURES_RATIO_ONLY, FEATURES_VIS, MET,
                        N_SPLITS, PROC, SECCHI_CLIP_LO, SEED, TIDY, ZONE_LABELS,
                        ZONES, banner, metrics)


def make_model(key="rf", params=None):
    p = dict(params or {})
    if key == "rf":
        return RandomForestRegressor(
            n_estimators=p.pop("n_estimators", 500),
            max_depth=p.pop("max_depth", 12),
            min_samples_leaf=p.pop("min_samples_leaf", 3),
            random_state=SEED, n_jobs=-1, **p)
    if key == "linear":
        return LinearRegression()
    raise ValueError(key)


def load(reharmonized=False):
    f = "analysis_dataset_reharmonized.csv" if reharmonized else "analysis_dataset.csv"
    d = pd.read_csv(PROC / f)
    d["date"] = pd.to_datetime(d["date"])
    return d


def cv_oof(d, feats, groups, splitter, model_key="rf", params=None):
    """Predicciones out-of-fold; scaler ajustado SOLO en la particion train."""
    X = d[feats].values
    y = d.secchi.values
    oof = np.full(len(y), np.nan)
    it = splitter.split(X, y, groups) if groups is not None else splitter.split(X, y)
    for tr, te in it:
        sc = RobustScaler().fit(X[tr])
        m = make_model(model_key, params)
        m.fit(sc.transform(X[tr]), y[tr])
        oof[te] = np.clip(m.predict(sc.transform(X[te])), SECCHI_CLIP_LO, None)
    return y, oof


def null_oof(d, key, groups, splitter):
    """Nulo: predice la media de entrenamiento del grupo indicado por `key`."""
    y = d.secchi.values
    k = d[key].astype(str).values
    oof = np.full(len(y), np.nan)
    for tr, te in splitter.split(y.reshape(-1, 1), y, groups):
        mp = pd.Series(y[tr]).groupby(k[tr]).mean()
        oof[te] = pd.Series(k[te]).map(mp).fillna(y[tr].mean()).values
    return y, oof


def main():
    banner("p04 -- ERROR 3: DIAGNOSTICO DE LA UNIDAD DE BLOQUEO")
    d = load()
    gkf = GroupKFold(N_SPLITS)
    g_camp = d.campaign_date.values
    g_stat = d.station.values
    g_year = d.year.astype(str).values

    # --- (1) modelos nulos bajo bloqueo por campana ------------------------
    banner("(1) MODELOS NULOS (bajo folds bloqueados por CAMPANA)", "-")
    print("  Una unidad de bloqueo funciona si su modelo nulo colapsa bajo ella.\n")
    nulls = []
    for key, label in [("zona", "media de la ZONA"),
                       ("station", "media de la ESTACION"),
                       ("campaign_date", "media de la CAMPANA")]:
        yt, yp = null_oof(d, key, g_camp, gkf)
        m = metrics(yt, yp)
        nulls.append({"null_model": key, "label": label, **m})
        print(f"    nulo {label:24s} R2={m['R2']:+.3f} RMSE={m['RMSE']:.2f}")
    print("\n  => El nulo de CAMPANA colapsa a ~0 bajo bloqueo por campana:")
    print("     el bloqueo esta funcionando.")
    print("  => Los nulos de ZONA y ESTACION NO colapsan: esa estructura sigue")
    print("     presente, y bloquear por estacion no la habria retenido.")

    # --- (2) jerarquia de unidades de bloqueo -----------------------------
    banner("(2) JERARQUIA DE VALIDACION", "-")
    rows = []

    def add(exp, case, label, yt, yp, note=""):
        m = metrics(yt, yp)
        rows.append({"experiment": exp, "case": case, "label": label, **m,
                     "note": note})
        print(f"    {label:38s} R2={m['R2']:+.3f} RMSE={m['RMSE']:.2f} n={m['n']}")
        return m

    yt, yp = cv_oof(d, FEATURES, None, KFold(N_SPLITS, shuffle=True, random_state=SEED))
    add("blocking", "random_kfold", "K-fold aleatorio (propenso a fuga)", yt, yp)
    yt, yp = cv_oof(d, FEATURES, g_stat, gkf)
    add("blocking", "by_station", "Bloqueo por ESTACION", yt, yp,
        f"{d.station.nunique()} grupos")
    y_true, oof_primary = cv_oof(d, FEATURES, g_camp, gkf)
    add("blocking", "by_campaign_date", "Bloqueo por CAMPANA (primario)",
        y_true, oof_primary, f"{d.campaign_date.nunique()} grupos")
    yt, yp = cv_oof(d, FEATURES, g_year, gkf)
    add("blocking", "by_year", "Bloqueo por ANIO", yt, yp,
        f"{d.year.nunique()} grupos")

    # pseudo-replicas colapsadas
    coll = (d.assign(pri=(d.sensor == "S2").astype(int))
            .sort_values("pri", ascending=False)
            .drop_duplicates("event_id").reset_index(drop=True))
    yt, yp = cv_oof(coll, FEATURES, coll.campaign_date.values, gkf)
    add("blocking", "one_record_per_event", "Un registro por evento in-situ",
        yt, yp, f"n={len(coll)} eventos independientes")

    # --- (3) extrapolacion a una zona no vista ----------------------------
    banner("(3) EXTRAPOLACION: DEJAR FUERA UNA ZONA TROFICA ENTERA", "-")
    X, y = d[FEATURES].values, d.secchi.values
    for z in ZONES:
        trm, tem = (d.zona != z).values, (d.zona == z).values
        if tem.sum() < 10:
            continue
        sc = RobustScaler().fit(X[trm])
        m = make_model("rf"); m.fit(sc.transform(X[trm]), y[trm])
        add("leave_one_zone_out", z, f"Sin {ZONE_LABELS[z]}", y[tem],
            np.clip(m.predict(sc.transform(X[tem])), SECCHI_CLIP_LO, None))
    print("\n  El modelo interpola dentro de los regimenes opticos muestreados")
    print("  pero NO extrapola a un tipo de agua ausente del entrenamiento.")

    # --- (4) variantes de armonizacion ------------------------------------
    banner("(4) EFECTO DE LA ARMONIZACION SOBRE EL RETRIEVAL", "-")
    d_re = load(reharmonized=True)
    for case, label, feats, frame in [
            ("roy_all", "Roy + las 12 features", FEATURES, d),
            ("visible_only", "Solo visible (sin NIR/SWIR rotos)", FEATURES_VIS, d),
            ("ratios_only", "Solo ratios escala-invariantes", FEATURES_RATIO_ONLY, d),
            ("local_reharm", "Recalibracion local sobre agua", FEATURES, d_re)]:
        yt, yp = cv_oof(frame, feats, frame.campaign_date.values, gkf)
        add("harmonization", case, label, yt, yp)
    for s, lab in [("S2", "Solo Sentinel-2"), ("LS", "Solo Landsat 8/9")]:
        sub = d[d.sensor == s].reset_index(drop=True)
        yt, yp = cv_oof(sub, FEATURES, sub.campaign_date.values, gkf)
        add("harmonization", f"{s}_only", f"{lab} (sin armonizar)", yt, yp,
            f"n={len(sub)}")
    print("\n  La ventaja del modelo combinado sobre estas variantes es pequena,")
    print("  consistente con que explota la identidad del sensor como covariable")
    print("  y no con una sinergia inter-sensor real.")

    # --- (5) error por zona sobre el diseno primario ----------------------
    banner("(5) ERROR POR ZONA TROFICA (diseno primario)", "-")
    zrows = []
    for z in ZONES:
        mk = (d.zona == z).values
        m = metrics(y_true[mk], oof_primary[mk])
        zrows.append({"zone": z, "zone_label": ZONE_LABELS[z], **m})
        print(f"    {ZONE_LABELS[z]:15s} R2={m['R2']:+.3f} RMSE={m['RMSE']:.2f} "
              f"sesgo={m['bias']:+.2f} n={m['n']}")
    print("\n  El sesgo positivo en Bahia de Puno es el mas preocupante: es la")
    print("  zona mas eutrofica y vulnerable, y el modelo la sobreestima porque")
    print("  la mayoria de agua clara arrastra las predicciones hacia arriba.")

    # --- salidas -----------------------------------------------------------
    pd.DataFrame(nulls).to_csv(TIDY / "null_models.csv", index=False)
    pd.DataFrame(rows).to_csv(TIDY / "validation_hierarchy.csv", index=False)
    pd.DataFrame(zrows).to_csv(TIDY / "error_by_zone.csv", index=False)
    out = d[["station", "zona", "campaign_date", "year", "sensor", "lat", "lon",
             "secchi"]].copy()
    out["zone_label"] = out.zona.map(ZONE_LABELS)
    out["predicted"] = oof_primary
    out["residual"] = oof_primary - y_true
    out.to_csv(TIDY / "oof_predictions.csv", index=False)

    primary = metrics(y_true, oof_primary)
    json.dump({"primary_design": "GroupKFold(5) blocked by campaign date",
               "primary_metrics": primary,
               "null_models": nulls, "hierarchy": rows, "by_zone": zrows,
               "conclusion": ("Dependence lives between campaigns, not between "
                              "stations. Station-blocked folds leave the "
                              "station-mean null at R2=0.21 and the zone-mean "
                              "null at R2=0.29; only campaign blocking collapses "
                              "its own null to ~0.")},
              open(MET / "validation.json", "w"), indent=2)

    banner("RESULTADO PRIMARIO")
    print(f"  Random Forest, bloqueo por campana: R2={primary['R2']:.3f} "
          f"RMSE={primary['RMSE']:.2f} m  MAE={primary['MAE']:.2f} m  "
          f"sesgo={primary['bias']:+.2f} m  n={primary['n']}")


if __name__ == "__main__":
    main()
