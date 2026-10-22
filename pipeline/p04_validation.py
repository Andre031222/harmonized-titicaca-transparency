"""
p04_validation.py -- donde vive la dependencia y que exactitud es defendible.

1. Descomposicion de la varianza del Secchi (efectos fijos anidados, descriptiva):
   cuanto explican zona, campana, zona x campana y ubicacion.
2. Grilla de nulos jerarquicos bajo cada diseno. Cada nulo es el gradiente de
   zona mas la anomalia media de un agrupamiento (ubicacion, fecha de muestreo o
   campana), ajustados solo con el entrenamiento. Un diseno retiene la
   dependencia de una unidad si el nulo de esa unidad no supera al de zona.
   Como 109 de 120 fechas de muestreo cubren una sola zona, un nulo de "media
   de la fecha" sin la zona confunde la fecha con el gradiente espacial.
3. Random Forest bajo cada diseno, con IC por bootstrap de campanas, y su
   ganancia sobre el mejor nulo que el diseno deja vivo.
4. Extrapolacion (fuera una zona), hold-out temporal, error por campana y por
   zona, un registro por evento, sensores por separado y subconjuntos de
   features, y la via de comparacion (Sen2Cor + coeficientes de HLS).

Salidas: results/tidy/variance_decomposition.csv, null_models_grid.csv,
         validation_hierarchy.csv, oof_predictions.csv, error_by_zone.csv,
         error_by_campaign.csv, temporal_holdout.csv
         results/metrics/validation.json
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from p00_config import (load_matchups, write_json, DESIGN_LABELS, DESIGNS, FEATURES, FEATURES_RATIO_ONLY,
                        FEATURES_VIS, MET, PRIMARY, PROC, SEED, TIDY, ZONES,
                        add_indices, banner, bandpass_adjust, cluster_boot,
                        cv_oof, metrics, splits)

NULLS = {"zone": None, "zone+location": "location",
         "zone+sampling_date": "sampling_date", "zone+campaign": "campaign"}
NULL_LABELS = {"zone": "Zone mean", "zone+location": "Zone + location anomaly",
               "zone+sampling_date": "Zone + sampling-date anomaly",
               "zone+campaign": "Zone + campaign anomaly"}
DESIGN_UNIT = {"location": "zone+location", "sampling_date": "zone+sampling_date",
               "campaign": "zone+campaign"}


def load():
    d = pd.read_csv(PROC / "analysis_dataset.csv", dtype={"campaign": str})
    return d.reset_index(drop=True)


def null_oof(d, key, folds):
    """Media de la zona en entrenamiento mas la anomalia media del grupo `key`."""
    y = d.secchi.values
    oof = np.full(len(y), np.nan)
    for tr, te in folds:
        t, s = d.iloc[tr], d.iloc[te]
        zm = t.groupby("zona").secchi.mean()
        base = s.zona.map(zm).fillna(t.secchi.mean()).values
        if key is not None:
            an = (t.secchi - t.zona.map(zm)).groupby(t[key]).mean()
            base = base + s[key].map(an).fillna(0).values
        oof[te] = base
    return y, oof


def variance_decomposition(d):
    """R2 en muestra de modelos de efectos fijos anidados (descriptivo)."""
    zc = d.zona + "|" + d.campaign
    steps = [("zone", [d.zona]), ("+ campaign", [d.zona, d.campaign]),
             ("+ zone x campaign", [d.zona, d.campaign, zc]),
             ("+ location", [d.zona, d.campaign, zc, d.location])]
    y = d.secchi.values
    rows, prev = [], 0.0
    for name, facs in steps:
        X = np.column_stack([np.ones(len(d))] + [
            pd.get_dummies(f, drop_first=True).values.astype(float) for f in facs])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        r2 = float(1 - np.sum((y - X @ beta) ** 2) / np.sum((y - y.mean()) ** 2))
        rows.append({"step": name, "R2_cumulative": round(r2, 3),
                     "R2_added": round(r2 - prev, 3)})
        prev = r2
    return pd.DataFrame(rows)


def main():
    banner("p04 -- DISENO DE VALIDACION")
    d = load()
    camp = d.campaign.values
    folds = {k: splits(d, k) for k in DESIGNS}
    zones_per_date = d.groupby("sampling_date").zona.nunique()
    print(f"  n={len(d)}  ubicaciones={d.location.nunique()}  fechas="
          f"{d.sampling_date.nunique()}  campanas={d.campaign.nunique()}")
    print(f"  fechas con una sola zona: {(zones_per_date == 1).sum()} de {len(zones_per_date)}")

    # --- (1) descomposicion de la varianza ------------------------------------
    banner("(1) DESCOMPOSICION DE LA VARIANZA (efectos fijos anidados)", "-")
    vd = variance_decomposition(d)
    print(vd.to_string(index=False))
    vd.to_csv(TIDY / "variance_decomposition.csv", index=False)

    # --- (2) nulos jerarquicos bajo cada diseno -------------------------------
    banner("(2) NULOS JERARQUICOS BAJO CADA DISENO", "-")
    grid = []
    for des in DESIGNS:
        for nk, key in NULLS.items():
            y, p = null_oof(d, key, folds[des])
            grid.append({"design": des, "design_label": DESIGN_LABELS[des],
                         "n_folds": len(folds[des]), "null_model": nk,
                         "null_label": NULL_LABELS[nk], **metrics(y, p)})
    grid = pd.DataFrame(grid)
    grid.to_csv(TIDY / "null_models_grid.csv", index=False)
    print(grid.pivot(index="null_model", columns="design", values="R2")
          [DESIGNS].to_string())

    # --- (3) Random Forest bajo cada diseno -----------------------------------
    banner("(3) RANDOM FOREST BAJO CADA DISENO", "-")
    hier, oof_by = [], {}
    for des in DESIGNS:
        y, p = cv_oof(d, FEATURES, folds=folds[des])
        oof_by[des] = p
        m = metrics(y, p)
        lo, hi = cluster_boot(y, p, camp)
        g = grid[grid.design == des].set_index("null_model").R2
        best_null = g.idxmax()
        fold_r2 = [r2_score(y[te], p[te]) for _, te in folds[des] if len(te) > 1]
        hier.append({"setting": "design", "design": des, "label": DESIGN_LABELS[des],
                     "n_folds": len(folds[des]), **m, "R2_lo": lo, "R2_hi": hi,
                     "fold_R2_min": round(min(fold_r2), 3),
                     "fold_R2_max": round(max(fold_r2), 3),
                     "best_null": best_null, "best_null_R2": float(g.max()),
                     "gain_over_null": round(m["R2"] - float(g.max()), 3)})
        print(f"  {DESIGN_LABELS[des]:30s} R2={m['R2']:+.3f} [{lo:+.3f}, {hi:+.3f}] "
              f"RMSE={m['RMSE']:.2f}  mejor nulo {best_null} {g.max():+.3f}")
    y = d.secchi.values
    p = oof_by[PRIMARY]

    # el sesgo de diseno: cuanto infla cada diseno respecto al primario
    for h in hier:
        h["R2_minus_primary"] = round(h["R2"] - metrics(y, p)["R2"], 3)
        if h["design"] != PRIMARY:
            h["diff_lo"], h["diff_hi"] = cluster_boot(y, oof_by[h["design"]], camp, p2=p)

    # --- (4) extrapolacion, temporal, subconjuntos -----------------------------
    banner("(4) EXTRAPOLACION, HOLD-OUT TEMPORAL Y SUBCONJUNTOS", "-")
    idx = np.arange(len(d))
    zone_folds = [(idx[d.zona.values != z], idx[d.zona.values == z]) for z in ZONES]
    yz, pz = cv_oof(d, FEATURES, folds=zone_folds)
    for z, (_, te) in zip(ZONES, zone_folds):
        m = metrics(yz[te], pz[te])
        hier.append({"setting": "zone_out", "design": z, "label": f"Leave out {z}", **m})
        print(f"  fuera {z:12s} R2={m['R2']:+.3f} RMSE={m['RMSE']:.2f} bias={m['bias']:+.2f}")

    cut = 2019
    tr, te = idx[d.year.values <= cut], idx[d.year.values > cut]
    yt, pt = cv_oof(d, FEATURES, folds=[(tr, te)])
    mt = metrics(yt[te], pt[te])
    shared = set(d.location.iloc[te]) & set(d.location.iloc[tr])
    hier.append({"setting": "temporal", "design": f"train<={cut}",
                 "label": f"Train <= {cut}, test 2022 & 2024", **mt})
    temporal = {"train_until": cut, "test_years": sorted(d.year.iloc[te].unique().tolist()),
                **mt, "test_locations": int(d.location.iloc[te].nunique()),
                "test_locations_in_train": len(shared),
                "pct_shared": round(100 * len(shared) / d.location.iloc[te].nunique(), 1)}
    pd.DataFrame([temporal]).to_csv(TIDY / "temporal_holdout.csv", index=False)
    print(f"  temporal R2={mt['R2']:+.3f}; ubicaciones de test vistas en train: "
          f"{len(shared)} de {d.location.iloc[te].nunique()}")

    def sub_eval(label, frame, feats=FEATURES):
        frame = frame.reset_index(drop=True)
        yy, pp = cv_oof(frame, feats, PRIMARY)
        m = metrics(yy, pp)
        lo, hi = cluster_boot(yy, pp, frame.campaign.values)
        hier.append({"setting": "subset", "design": PRIMARY, "label": label, **m,
                     "R2_lo": lo, "R2_hi": hi})
        print(f"  {label:44s} R2={m['R2']:+.3f} [{lo:+.3f}, {hi:+.3f}] n={m['n']}")
        return yy, pp

    rng = np.random.default_rng(SEED)
    one = d.loc[d.groupby("event").sample(1, random_state=SEED).index]
    sub_eval("One record per field event", one)
    sub_eval("Sentinel-2 only", d[d.sensor == "S2"])
    sub_eval("Landsat only", d[d.sensor == "LS"])
    sub_eval("Visible bands and indices only", d, FEATURES_VIS)
    sub_eval("Band ratios only", d, FEATURES_RATIO_ONLY)
    s2c = d.campaign.isin(d.loc[d.sensor == "S2", "campaign"].unique())
    sub_eval("Pooled, campaigns with both sensors", d[s2c])
    del rng

    # la via de comparacion: Sen2Cor + LaSRC con solo los coeficientes de banda
    mu = load_matchups()
    sr = mu[(mu["product"] == "sr") & (mu.window == 10)].dropna(subset=["secchi"])
    sr = add_indices(bandpass_adjust(sr)).dropna(subset=FEATURES)
    sub_eval("Agency SR + HLS bandpass coefficients", sr)

    hier = pd.DataFrame(hier)
    hier.to_csv(TIDY / "validation_hierarchy.csv", index=False)

    # --- (5) error por zona y por campana, OOF primario ------------------------
    oof = d[["station", "location", "spatial_block", "zona", "sampling_date",
             "campaign", "year", "sensor", "lat", "lon", "secchi"]].copy()
    oof["pred"] = p
    oof["residual"] = oof.pred - oof.secchi
    for des in DESIGNS:
        oof[f"pred_{des}"] = oof_by[des]
    oof.to_csv(TIDY / "oof_predictions.csv", index=False, float_format="%.4f")

    ez = [{"zona": z, **metrics(oof.secchi[oof.zona == z], oof.pred[oof.zona == z])}
          for z in ZONES]
    pd.DataFrame(ez).to_csv(TIDY / "error_by_zone.csv", index=False)
    ec = [{"campaign": c, "n_s2": int(((oof.campaign == c) & (oof.sensor == "S2")).sum()),
           **metrics(oof.secchi[oof.campaign == c], oof.pred[oof.campaign == c])}
          for c in sorted(oof.campaign.unique())]
    ec = pd.DataFrame(ec)
    ec.to_csv(TIDY / "error_by_campaign.csv", index=False)
    print(ec.to_string(index=False))

    mp = metrics(y, p)
    lo, hi = cluster_boot(y, p, camp)
    rlo, rhi = cluster_boot(y, p, camp, stat="RMSE")
    gnull = grid.set_index(["design", "null_model"]).R2
    out = {
        "primary_design": "leave-one-campaign-out (9 folds)",
        "n": len(d), "n_campaigns": int(d.campaign.nunique()),
        "n_sampling_dates": int(d.sampling_date.nunique()),
        "n_locations": int(d.location.nunique()),
        "sampling_dates_single_zone": int((zones_per_date == 1).sum()),
        "headline": {**mp, "R2_lo": lo, "R2_hi": hi, "RMSE_lo": rlo, "RMSE_hi": rhi},
        "variance_decomposition": vd.to_dict(orient="records"),
        "null_grid": {f"{a}|{b}": float(v) for (a, b), v in gnull.items()},
        "hierarchy": hier.replace({np.nan: None}).to_dict(orient="records"),
        "temporal_holdout": temporal,
        "error_by_zone": ez,
        "error_by_campaign": ec.to_dict(orient="records"),
    }
    write_json(out, MET / "validation.json")
    print(f"\n  PRIMARIO (fuera una campana): R2={mp['R2']:.3f} [{lo:.3f}, {hi:.3f}] "
          f"RMSE={mp['RMSE']:.2f} m bias={mp['bias']:+.2f}")


if __name__ == "__main__":
    main()
