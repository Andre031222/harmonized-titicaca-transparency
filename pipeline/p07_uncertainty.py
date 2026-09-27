"""
p07_uncertainty.py -- Incertidumbre por prediccion (prediccion conformal).

Dentro de cada fold del diseno primario (GroupKFold por fecha de muestreo), una
parte de la particion de entrenamiento se reserva para calibrar. Esa parte se
toma por FECHAS DE MUESTREO COMPLETAS, no al azar: las observaciones de una
misma fecha estan correlacionadas (p04) y calibrar con ellas repartidas
subestimaria el error. El cuantil usa la correccion de muestra finita,
ceil((n+1)(1-alpha))/n.

Tres variantes:
  split     ancho constante: el cuantil global de los residuos absolutos
  mondrian  el cuantil se calcula por separado en cada tramo del valor
            PREDICHO, asi que el ancho se adapta a lo que el usuario ve
  cqr       regresion cuantilica conformalizada (Romano et al. 2019): dos
            modelos de cuantiles dan un intervalo asimetrico que se corrige
            con el conjunto de calibracion

Se evalua a 50, 80, 90 y 95 % y la cobertura se condiciona tanto al Secchi
medido como al predicho.

Salidas: results/tidy/conformal_coverage.csv
         results/tidy/conformal_intervals.csv
         results/tidy/interval_width_by_secchi.csv
         results/metrics/uncertainty.json
"""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import RobustScaler

from p00_config import (FEATURES, MET, N_SPLITS, PROC, SECCHI_CLIP_LO, SEED,
                        TIDY, ZONE_LABELS, banner, metrics)
from p04_fix3_validation import make_model

LEVELS = [0.50, 0.80, 0.90, 0.95]
METHODS = ["split", "mondrian", "cqr"]
CAL_FRACTION = 0.2          # fraccion de fechas de entrenamiento para calibrar
BINS = [0, 5, 7.5, 10, 12.5, 20]
BIN_LABELS = ["<5", "5–7.5", "7.5–10", "10–12.5", ">12.5"]


def load():
    d = pd.read_csv(PROC / "analysis_dataset.csv")
    d["date"] = pd.to_datetime(d["date"])
    return d


def q_finite(scores, alpha):
    """Cuantil conformal con la correccion de muestra finita."""
    n = len(scores)
    k = min(n, int(np.ceil((n + 1) * (1 - alpha))))
    return float(np.sort(scores)[k - 1])


def qgbr(alpha_q):
    return GradientBoostingRegressor(loss="quantile", alpha=alpha_q,
                                     n_estimators=300, max_depth=3,
                                     learning_rate=0.05, random_state=SEED)


def conformal_oof(d, alpha, method):
    """Predicciones y bandas conformales fuera de fold."""
    X, y = d[FEATURES].values, d.secchi.values
    g = d.campaign_date.values
    n = len(y)
    pred, lo, hi = (np.full(n, np.nan) for _ in range(3))
    for tr, te in GroupKFold(N_SPLITS).split(X, y, g):
        sc = RobustScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        fit, cal = next(GroupShuffleSplit(1, test_size=CAL_FRACTION,
                                          random_state=SEED).split(Xtr, y[tr], g[tr]))
        m = make_model("rf")
        m.fit(Xtr[fit], y[tr][fit])
        p_te = np.clip(m.predict(Xte), SECCHI_CLIP_LO, None)
        p_cal = m.predict(Xtr[cal])
        pred[te] = p_te
        if method == "split":
            q = q_finite(np.abs(y[tr][cal] - p_cal), alpha)
            lo[te], hi[te] = p_te - q, p_te + q
        elif method == "mondrian":
            b_cal = np.digitize(p_cal, BINS[1:-1])
            b_te = np.digitize(p_te, BINS[1:-1])
            res = np.abs(y[tr][cal] - p_cal)
            q_all = q_finite(res, alpha)
            for b in np.unique(b_te):
                r_b = res[b_cal == b]
                # con menos de 20 residuos en el tramo se usa el cuantil global
                q = q_finite(r_b, alpha) if len(r_b) >= 20 else q_all
                lo[te[b_te == b]] = p_te[b_te == b] - q
                hi[te[b_te == b]] = p_te[b_te == b] + q
        else:
            ql, qh = qgbr(alpha / 2), qgbr(1 - alpha / 2)
            ql.fit(Xtr[fit], y[tr][fit])
            qh.fit(Xtr[fit], y[tr][fit])
            s = np.maximum(ql.predict(Xtr[cal]) - y[tr][cal],
                           y[tr][cal] - qh.predict(Xtr[cal]))
            q = q_finite(s, alpha)
            lo[te], hi[te] = ql.predict(Xte) - q, qh.predict(Xte) + q
    lo = np.clip(lo, SECCHI_CLIP_LO, None)
    return y, pred, lo, hi


def conditional(iv, by):
    col = "secchi" if by == "measured" else "predicted"
    b = pd.cut(iv[col], bins=BINS, labels=BIN_LABELS)
    out = (iv.assign(bin=b).groupby(["method", "bin"], observed=True)
           .agg(n=("secchi", "size"), mean_width=("width", "mean"),
                coverage=("covered", "mean"))
           .reset_index())
    out["conditioned_on"] = by
    return out


def main():
    banner("p07 -- INCERTIDUMBRE CONFORMAL (bloqueo por fecha de muestreo)")
    d = load()
    print(f"  n={len(d)} | {d.campaign_date.nunique()} fechas de muestreo; "
          f"calibracion con el {CAL_FRACTION:.0%} de las fechas de cada fold\n")
    rows, ivs = [], []
    print(f"  {'metodo':9s} {'nominal':>8s} {'empirica':>9s} {'ancho medio':>12s}")
    for method in METHODS:
        for lev in LEVELS:
            y, pred, lo, hi = conformal_oof(d, 1 - lev, method)
            cov = float(np.mean((y >= lo) & (y <= hi)))
            width = float(np.mean(hi - lo))
            rows.append({"method": method, "nominal": lev,
                         "empirical_coverage": round(cov, 4),
                         "mean_width_m": round(width, 3),
                         "calibration_error": round(cov - lev, 4)})
            print(f"  {method:9s} {lev * 100:7.0f}% {cov * 100:8.1f}% {width:11.2f} m")
            if abs(lev - 0.90) < 1e-9:
                iv = d[["station", "zona", "campaign_date", "year", "sensor",
                        "secchi"]].copy()
                iv["zone_label"] = iv.zona.map(ZONE_LABELS)
                iv["method"] = method
                iv["predicted"], iv["lower"], iv["upper"] = pred, lo, hi
                ivs.append(iv)
        print()
    cov_df = pd.DataFrame(rows)
    iv = pd.concat(ivs, ignore_index=True)
    iv["width"] = iv.upper - iv.lower
    iv["covered"] = (iv.secchi >= iv.lower) & (iv.secchi <= iv.upper)

    banner("COBERTURA DEL INTERVALO AL 90% CONDICIONADA", "-")
    cond = pd.concat([conditional(iv, "measured"), conditional(iv, "predicted")],
                     ignore_index=True)
    for by in ["measured", "predicted"]:
        t = cond[cond.conditioned_on == by].pivot(index="bin", columns="method",
                                                  values="coverage")
        print(f"\n  por Secchi {'medido' if by == 'measured' else 'predicho'}:")
        print((t * 100).round(1).reindex(BIN_LABELS).dropna(how="all").to_string())

    spread = (cond[cond.conditioned_on == "measured"].groupby("method")
              .coverage.agg(lambda c: float(c.max() - c.min())))
    print("\n  rango de cobertura entre tramos de Secchi medido (puntos):")
    print((spread * 100).round(1).to_string())

    m = metrics(iv[iv.method == "split"].secchi.values,
                iv[iv.method == "split"].predicted.values)
    cov_df.to_csv(TIDY / "conformal_coverage.csv", index=False)
    iv.to_csv(TIDY / "conformal_intervals.csv", index=False, float_format="%.6f")
    cond.to_csv(TIDY / "interval_width_by_secchi.csv", index=False,
                float_format="%.6f")
    by_zone = (iv.groupby(["method", "zone_label"])
               .agg(n=("secchi", "size"), mean_width=("width", "mean"),
                    coverage=("covered", "mean")).reset_index())
    json.dump({"design": ("GroupKFold(5) by sampling date; calibration on whole "
                          "sampling dates (20 %); finite-sample quantile"),
               "levels": rows,
               "max_calibration_error": {k: round(float(v), 4) for k, v in
                                         cov_df.groupby("method").calibration_error
                                         .apply(lambda c: c.abs().max()).items()},
               "coverage_spread_measured_90": {k: round(v, 4) for k, v in spread.items()},
               "model_with_cal_split": m,
               "mean_width_90_m": {k: round(float(v), 3) for k, v in
                                   iv.groupby("method").width.mean().items()},
               "by_zone": by_zone.round(4).to_dict("records")},
              open(MET / "uncertainty.json", "w"), indent=2)

    banner("SALIDAS")
    for f in ["conformal_coverage.csv", "conformal_intervals.csv",
              "interval_width_by_secchi.csv"]:
        print(f"  {TIDY / f}")
    print(f"  {MET / 'uncertainty.json'}")


if __name__ == "__main__":
    main()
