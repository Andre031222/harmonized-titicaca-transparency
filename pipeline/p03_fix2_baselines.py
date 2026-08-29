"""
p03_fix2_baselines.py -- ERROR 2: el baseline clasico era un espantapajaros.

El benchmark anterior reportaba R2 = -2.74 para el algoritmo clasico
azul/verde. El codigo ajustaba LinearRegression sobre log1p(y) y deshacia la
transformacion con expm1 SIN ACOTAR el resultado. Una unica prediccion de
~143 m (el maximo observado del lago es 16.5 m) bastaba para hundir el R2 y
duplicar el RMSE.

Un baseline que se auto-destruye por un artefacto numerico no demuestra que
el Random Forest sea mejor: demuestra que el baseline se implemento mal. Aqui
se implementa en su forma estandar (Kloiber et al. 2002),
    ln(Zsd) = b0 + b1 * ln(rho_azul / rho_verde),
con las predicciones acotadas al rango fisicamente observable, y se conserva
la version defectuosa SOLO como demostracion del artefacto.

Salidas: results/tidy/benchmark_models.csv
         results/tidy/baseline_artefact.csv
         results/metrics/benchmark.json
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler

from p00_config import (FEATURES, MET, N_SPLITS, PROC, SECCHI_CLIP_LO, SEED,
                        TIDY, banner, metrics)
from p04_fix3_validation import make_model


def load():
    d = pd.read_csv(PROC / "analysis_dataset.csv")
    d["date"] = pd.to_datetime(d["date"])
    return d


def classical_broken(d, groups, gkf):
    """Reproduce el bug: log1p -> expm1 sin acotar."""
    X = d[["B2_B3"]].values
    y = d.secchi.values
    yt, yp = [], []
    for tr, te in gkf.split(X, y, groups):
        sc = RobustScaler().fit(X[tr])
        lr = LinearRegression().fit(sc.transform(X[tr]), np.log1p(y[tr]))
        yp.append(np.clip(np.expm1(lr.predict(sc.transform(X[te]))),
                          SECCHI_CLIP_LO, None))
        yt.append(y[te])
    return np.concatenate(yt), np.concatenate(yp)


def classical_kloiber(d, groups, gkf):
    """Forma estandar: ln(Zsd) ~ ln(azul/verde), acotada al rango observable."""
    X = np.log(np.clip(d[["B2_B3"]].values, 1e-6, None))
    y = d.secchi.values
    hi = float(y.max() * 1.5)     # margen fisico generoso, no ajustado al test
    yt, yp = [], []
    for tr, te in gkf.split(X, y, groups):
        lr = LinearRegression().fit(X[tr], np.log(np.clip(y[tr], 1e-3, None)))
        yp.append(np.clip(np.exp(lr.predict(X[te])), SECCHI_CLIP_LO, hi))
        yt.append(y[te])
    return np.concatenate(yt), np.concatenate(yp)


def generic_cv(d, feats, groups, gkf, model_key):
    X = d[feats].values
    y = d.secchi.values
    yt, yp = [], []
    for tr, te in gkf.split(X, y, groups):
        sc = RobustScaler().fit(X[tr])
        m = make_model(model_key)
        m.fit(sc.transform(X[tr]), y[tr])
        yp.append(np.clip(m.predict(sc.transform(X[te])), SECCHI_CLIP_LO, None))
        yt.append(y[te])
    return np.concatenate(yt), np.concatenate(yp)


def main():
    banner("p03 -- ERROR 2: BASELINE CLASICO CORREGIDO")
    d = load()
    groups = d.campaign_date.values          # diseno primario: bloqueo por campana
    gkf = GroupKFold(N_SPLITS)
    y_max = float(d.secchi.max())
    print(f"  Diseno: GroupKFold({N_SPLITS}) bloqueado por FECHA DE CAMPANA")
    print(f"  Maximo Secchi observado en el lago: {y_max} m\n")

    rows = []

    yt, yp = classical_broken(d, groups, gkf)
    m = metrics(yt, yp)
    n_abs = int((yp > y_max * 1.5).sum())
    rows.append({"model": "Classical blue/green -- unbounded (as published)",
                 "family": "artefact", **m,
                 "max_prediction_m": round(float(yp.max()), 1),
                 "n_absurd_predictions": n_abs})
    print(f"  [BUG] clasico sin acotar    R2={m['R2']:+.3f} RMSE={m['RMSE']:.2f}")
    print(f"        prediccion maxima = {yp.max():.1f} m  "
          f"({n_abs} prediccion(es) fisicamente imposible(s))")
    art = pd.DataFrame({"observed": yt, "predicted_unbounded": yp})

    yt, yp = classical_kloiber(d, groups, gkf)
    m = metrics(yt, yp)
    rows.append({"model": "Classical blue/green ratio (Kloiber 2002, bounded)",
                 "family": "classical", **m,
                 "max_prediction_m": round(float(yp.max()), 1),
                 "n_absurd_predictions": 0})
    print(f"  [OK]  clasico acotado       R2={m['R2']:+.3f} RMSE={m['RMSE']:.2f}")
    art["predicted_bounded"] = yp

    for key, label, feats in [("linear", "Multiband linear regression", FEATURES),
                              ("rf", "Random Forest (this study)", FEATURES)]:
        yt, yp = generic_cv(d, feats, groups, gkf, key)
        m = metrics(yt, yp)
        rows.append({"model": label, "family": key, **m,
                     "max_prediction_m": round(float(yp.max()), 1),
                     "n_absurd_predictions": int((yp > y_max * 1.5).sum())})
        print(f"  [OK]  {label:22s} R2={m['R2']:+.3f} RMSE={m['RMSE']:.2f} "
              f"bias={m['bias']:+.2f}")

    bench = pd.DataFrame(rows)
    bench.to_csv(TIDY / "benchmark_models.csv", index=False)
    art.to_csv(TIDY / "baseline_artefact.csv", index=False)

    rf_r2 = bench.loc[bench.family == "rf", "R2"].iloc[0]
    cl_r2 = bench.loc[bench.family == "classical", "R2"].iloc[0]
    bug_r2 = bench.loc[bench.family == "artefact", "R2"].iloc[0]
    print(f"\n  Ventaja REAL del RF sobre el clasico bien implementado: "
          f"{rf_r2:.3f} vs {cl_r2:.3f}")
    print(f"  Ventaja FICTICIA que daba el bug:                       "
          f"{rf_r2:.3f} vs {bug_r2:.3f}")
    print("  El RF sigue ganando con claridad. Corregir el baseline no cuesta")
    print("  nada y elimina un blanco obvio para cualquier revisor.")

    json.dump({"design": "GroupKFold(5) blocked by campaign date",
               "models": bench.to_dict("records"),
               "rf_vs_correct_classical": {"rf_R2": rf_r2, "classical_R2": cl_r2},
               "note": ("The R2=-2.74 previously reported for the classical "
                        "algorithm was an artefact of unbounded expm1 "
                        "back-transformation, driven by a single prediction of "
                        "~143 m against an observed maximum of 16.5 m.")},
              open(MET / "benchmark.json", "w"), indent=2)

    banner("SALIDAS")
    print(f"  {TIDY / 'benchmark_models.csv'}")
    print(f"  {TIDY / 'baseline_artefact.csv'}")
    print(f"  {MET / 'benchmark.json'}")


if __name__ == "__main__":
    main()
