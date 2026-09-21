"""
p08_interpretability.py -- SHAP y retrievabilidad.

(A) SHAP sobre el Random Forest. IMPORTANTE: se calcula FUERA DE FOLD (cada
    observacion se explica con el modelo que NO la vio) y no sobre un modelo
    ajustado a todo el dataset, como se hacia antes. Explicar en muestra
    exagera la importancia de las features que el modelo memoriza.

    El resultado tiene una lectura incomoda que el articulo debe afrontar:
    los ratios verde/azul dominan la atribucion, y el azul es justo la banda
    en la que ambos sensores menos concuerdan (p02). La interpretacion fisica
    y el problema de armonizacion apuntan a la misma banda.

(B) Retrievabilidad de las demas variables: clorofila-a, solidos suspendidos y
    temperatura. Se evalua con el mismo diseno honesto (RF + bloqueo por
    campana), no solo con la correlacion de un indice suelto, para que la
    afirmacion "no es recuperable" sea tan rigurosa como la afirmacion
    positiva sobre el Secchi.

Salidas: results/tidy/shap_importance.csv
         results/tidy/shap_dependence.csv
         results/tidy/retrievability.csv
         results/metrics/interpretability.json
"""
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler

from p00_config import (FEATURE_LABELS, FEATURES, MET, N_SPLITS, PROC,
                        SECCHI_CLIP_LO, TIDY, banner, metrics)
from p04_fix3_validation import make_model


def load():
    d = pd.read_csv(PROC / "analysis_dataset.csv")
    d["date"] = pd.to_datetime(d["date"])
    return d


# ---------------------------------------------------------------------------
def shap_out_of_fold(d):
    banner("(A) SHAP FUERA DE FOLD (cada punto explicado por un modelo que no lo vio)")
    import shap
    X, y = d[FEATURES].values, d.secchi.values
    g = d.campaign_date.values
    sv = np.full((len(y), len(FEATURES)), np.nan)
    for tr, te in GroupKFold(N_SPLITS).split(X, y, g):
        sc = RobustScaler().fit(X[tr])
        m = make_model("rf")
        m.fit(sc.transform(X[tr]), y[tr])
        expl = shap.TreeExplainer(m)
        sv[te] = expl.shap_values(sc.transform(X[te]), check_additivity=False)

    imp = (pd.DataFrame({"feature": FEATURES,
                         "label": [FEATURE_LABELS[f] for f in FEATURES],
                         "mean_abs_shap": np.abs(sv).mean(axis=0)})
           .sort_values("mean_abs_shap", ascending=False)
           .reset_index(drop=True))
    imp["share"] = imp.mean_abs_shap / imp.mean_abs_shap.sum()
    imp["cumulative_share"] = imp.share.cumsum()
    imp["band_group"] = np.where(
        imp.feature.isin(["B2", "B3", "B2_B3", "B3_B2"]), "Verde/azul",
        np.where(imp.feature.isin(["B8", "B11", "B12"]), "NIR/SWIR", "Otras"))

    print(f"  {'Feature':12s} {'|SHAP| medio':>13s} {'cuota':>7s} {'grupo':>12s}")
    for _, r in imp.iterrows():
        print(f"  {r.label:12s} {r.mean_abs_shap:13.4f} {r.share * 100:6.1f}% "
              f"{r.band_group:>12s}")

    gb = imp.groupby("band_group").share.sum().sort_values(ascending=False)
    print(f"\n  Cuota por grupo de bandas:")
    for k, v in gb.items():
        print(f"    {k:12s} {v * 100:5.1f}%")
    print("\n  LECTURA CRITICA: los ratios verde/azul dominan la atribucion, lo")
    print("  cual es fisicamente coherente con la optica de la transparencia,")
    print("  PERO el azul es la banda con peor acuerdo entre sensores (r=0.26).")
    print("  La fisica y el problema de armonizacion senalan a la misma banda.")

    # dependencia SHAP de las dos features dominantes
    top = imp.feature.head(2).tolist()
    dep = []
    for f in top:
        j = FEATURES.index(f)
        for val, s, zn in zip(X[:, j], sv[:, j], d.zona.values):
            dep.append({"feature": f, "label": FEATURE_LABELS[f],
                        "value": float(val), "shap": float(s), "zona": zn})
    # todas las features, una fila por match-up: el resumen tipo enjambre
    allv = pd.DataFrame({"feature": np.repeat(FEATURES, len(X)),
                         "value": X.T.ravel(), "shap": sv.T.ravel()})
    allv["label"] = allv.feature.map(FEATURE_LABELS)
    return imp, pd.DataFrame(dep), gb.to_dict(), allv


# ---------------------------------------------------------------------------
def retrievability(d_all):
    banner("(B) RETRIEVABILIDAD: ¿que OTRA variable se puede recuperar?")
    rows = []
    print(f"  {'Variable':26s} {'n':>5s} {'R2 (RF, CV campana)':>21s} {'rango in-situ':>18s}")
    for var, label, unit in [("secchi", "Transparencia (Secchi)", "m"),
                             ("chl", "Clorofila-a", "mg m-3"),
                             ("tss", "Solidos suspendidos", "mg L-1"),
                             ("temp_insitu", "Temperatura del agua", "C")]:
        if var not in d_all.columns:
            continue
        sub = d_all[d_all[var].notna()].dropna(subset=FEATURES).reset_index(drop=True)
        if len(sub) < 40 or sub.campaign_date.nunique() < N_SPLITS:
            continue
        X, y = sub[FEATURES].values, sub[var].values
        g = sub.campaign_date.values
        oof = np.full(len(y), np.nan)
        for tr, te in GroupKFold(N_SPLITS).split(X, y, g):
            sc = RobustScaler().fit(X[tr])
            m = make_model("rf")
            m.fit(sc.transform(X[tr]), y[tr])
            oof[te] = m.predict(sc.transform(X[te]))
        mm = metrics(y, oof)
        cv_y = float(np.std(y) / max(np.mean(y), 1e-9))
        rows.append({"variable": var, "label": label, "unit": unit,
                     "R2": mm["R2"], "RMSE": mm["RMSE"], "MAE": mm["MAE"],
                     "n": mm["n"], "insitu_min": round(float(y.min()), 2),
                     "insitu_max": round(float(y.max()), 2),
                     "insitu_cv": round(cv_y, 3),
                     "retrievable": bool(mm["R2"] > 0.30)})
        print(f"  {label:26s} {mm['n']:5d} {mm['R2']:+20.3f} "
              f"{y.min():8.2f} – {y.max():<7.2f}")

    # termico Landsat contra temperatura in-situ
    if "lst_C" in d_all.columns:
        sub = d_all[(d_all.lst_C.notna()) & (d_all.temp_insitu.notna())]
        if len(sub) > 40:
            r = float(np.corrcoef(sub.lst_C, sub.temp_insitu)[0, 1])
            rmse = float(np.sqrt(np.mean((sub.lst_C - sub.temp_insitu) ** 2)))
            rows.append({"variable": "lst_thermal",
                         "label": "Temperatura (termico ST_B10)", "unit": "C",
                         "R2": round(r * r, 3), "RMSE": round(rmse, 3),
                         "MAE": None, "n": int(len(sub)),
                         "insitu_min": round(float(sub.temp_insitu.min()), 2),
                         "insitu_max": round(float(sub.temp_insitu.max()), 2),
                         "insitu_cv": None, "retrievable": False})
            print(f"  {'Termico ST_B10 (directo)':26s} {len(sub):5d} "
                  f"{r * r:+20.3f} {sub.temp_insitu.min():8.2f} – "
                  f"{sub.temp_insitu.max():<7.2f}   (r={r:+.3f})")

    print("\n  Solo la transparencia supera el umbral de utilidad. Clorofila-a y")
    print("  solidos suspendidos tienen rangos in-situ por debajo del limite de")
    print("  deteccion optica en agua tan clara, y el termico no distingue")
    print("  variaciones de 1-2 C. Reportarlo delimita el sobre operativo.")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def main():
    d = load()
    imp, dep, groups, allv = shap_out_of_fold(d)

    # el dataset completo con matched==1 (incluye filas sin Secchi)
    raw = []
    for fn, s in [("matchups_s2.csv", "S2"), ("matchups_ls.csv", "LS")]:
        t = pd.read_csv(PROC / fn)
        t = t[t.matched == 1].copy()
        t["sensor"] = s
        raw.append(t)
    from p00_config import add_indices
    d_all = add_indices(pd.concat(raw, ignore_index=True))
    d_all["campaign_date"] = pd.to_datetime(d_all.date).dt.strftime("%Y-%m-%d")
    retr = retrievability(d_all)

    imp.to_csv(TIDY / "shap_importance.csv", index=False)
    dep.to_csv(TIDY / "shap_dependence.csv", index=False)
    allv.to_csv(TIDY / "shap_values_all.csv", index=False, float_format="%.6f")
    retr.to_csv(TIDY / "retrievability.csv", index=False)
    json.dump({"shap_design": "out-of-fold, GroupKFold(5) by campaign date",
               "importance": imp.to_dict("records"),
               "share_by_band_group": {k: round(v, 4) for k, v in groups.items()},
               "retrievability": retr.to_dict("records"),
               "caveat": ("Green/blue ratios dominate the attribution, which is "
                          "physically consistent, but blue is the band with the "
                          "worst cross-sensor agreement (r=0.26). Physics and the "
                          "harmonization problem point at the same band.")},
              open(MET / "interpretability.json", "w"), indent=2, default=str)

    banner("SALIDAS")
    for f in ["shap_importance.csv", "shap_dependence.csv", "shap_values_all.csv",
              "retrievability.csv"]:
        print(f"  {TIDY / f}")
    print(f"  {MET / 'interpretability.json'}")


if __name__ == "__main__":
    main()
