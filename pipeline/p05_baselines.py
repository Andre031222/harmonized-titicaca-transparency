"""
p05_baselines.py -- baselines clasicos y lineales frente al Random Forest.

Todo bajo el diseno primario (dejar fuera una campana entera, 9 folds) sobre
analysis_dataset.csv (HLS v2.0, +-10 d):
  (a) Kloiber et al. (2002), forma original:  ln(Zsd) = a*(azul/rojo) + b*azul + c
  (b) razon de dos bandas:                    ln(Zsd) = a + b*ln(azul/verde)
      ambos ajustados en cada fold sobre ln(Zsd), devueltos con exp y acotados
      a [0.05, 1.5 x maximo del entrenamiento]; tambien sin acotar, para ver
      si la cota importa.
  (c) regresion lineal multibanda (12 features), winsorizada a los
      percentiles 1-99 del entrenamiento (limites solo del train, aplicados
      a una COPIA).
  (d) Random Forest (make_rf).
Cada uno se ajusta tambien con un solo sensor (S2 o LS) para ver si
ajustar a un sensor por separado ayuda.

Salidas: results/tidy/benchmark_models.csv
         results/tidy/model_comparison_ci.csv
         results/metrics/benchmark.json
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler

from p00_config import (FEATURES, MET, PRIMARY, PROC, SECCHI_CLIP_LO, TIDY,
                        banner, cluster_boot, cv_oof, metrics, splits)

# piso para ln(azul/verde): dos filas tienen azul <= 0
LOG_RATIO_FLOOR = 1e-3
ABSURD_FACTOR = 1.5

MODELS = {
    "kloiber": "Kloiber et al. (2002): ln(Z) ~ blue/red + blue",
    "two_band": "Two-band: ln(Z) ~ ln(blue/green)",
    "linear": "Multiband linear regression (12 features, winsorized)",
    "rf": "Random Forest (12 features)",
}


def load():
    d = pd.read_csv(PROC / "analysis_dataset.csv", dtype={"campaign": str})
    return d.reset_index(drop=True)


def log_linear(X, d, folds):
    """ln(Zsd) lineal en X. Devuelve (y, acotada, sin acotar)."""
    y = d.secchi.values
    bounded, raw = np.full(len(y), np.nan), np.full(len(y), np.nan)
    for tr, te in folds:
        lr = LinearRegression().fit(X[tr], np.log(y[tr]))
        p = np.exp(lr.predict(X[te]))
        raw[te] = p
        bounded[te] = np.clip(p, SECCHI_CLIP_LO, ABSURD_FACTOR * y[tr].max())
    return y, bounded, raw


def kloiber_X(d):
    return np.column_stack([d.B2_B4.values, d.B2.values])


def two_band_X(d):
    return np.log(np.clip(d[["B2_B3"]].values, LOG_RATIO_FLOOR, None))


def linear_winsor(d, folds):
    X, y = d[FEATURES].values, d.secchi.values
    oof = np.full(len(y), np.nan)
    for tr, te in folds:
        lo, hi = np.percentile(X[tr], [1, 99], axis=0)
        Xtr, Xte = np.clip(X[tr], lo, hi), np.clip(X[te], lo, hi)   # copias
        sc = RobustScaler().fit(Xtr)
        lr = LinearRegression().fit(sc.transform(Xtr), y[tr])
        oof[te] = lr.predict(sc.transform(Xte))
    return y, np.clip(oof, SECCHI_CLIP_LO, None)


def run_all(d):
    """Predicciones OOF de los cuatro modelos sobre d (diseno primario)."""
    folds = splits(d, PRIMARY)
    out = {}
    y, b, r = log_linear(kloiber_X(d), d, folds)
    out["kloiber"], out["kloiber_unbounded"] = b, r
    y, b, r = log_linear(two_band_X(d), d, folds)
    out["two_band"], out["two_band_unbounded"] = b, r
    out["linear"] = linear_winsor(d, folds)[1]
    out["rf"] = cv_oof(d, FEATURES, PRIMARY, folds=folds)[1]
    return y, out


def summarise(y, preds, groups, subset, ymax):
    rows = []
    for key, p in preds.items():
        base = key.replace("_unbounded", "")
        m = metrics(y, p)
        lo, hi = cluster_boot(y, p, groups)
        rows.append({"subset": subset, "model": key, "family": base,
                     "label": MODELS[base] + (" -- unbounded" if key != base else ""),
                     "bounded": key == base, **m, "R2_lo": lo, "R2_hi": hi,
                     "n_campaigns": int(len(np.unique(groups))),
                     "max_prediction_m": round(float(np.max(p)), 2),
                     "n_absurd": int((p > ABSURD_FACTOR * ymax).sum())})
    return rows


def main():
    banner("p05 -- BASELINES CLASICOS Y LINEALES vs RANDOM FOREST")
    d = load()
    ymax = float(d.secchi.max())
    print(f"  n={len(d)} | {d.campaign.nunique()} campanas | Secchi max {ymax} m")
    print(f"  filas con azul/verde < {LOG_RATIO_FLOOR} (piso en ln): "
          f"{int((d.B2_B3 < LOG_RATIO_FLOOR).sum())}")

    rows, comps = [], []
    for subset, sub in [("pooled", d), ("S2", d[d.sensor == "S2"]),
                        ("LS", d[d.sensor == "LS"])]:
        sub = sub.reset_index(drop=True)
        y, preds = run_all(sub)
        g = sub.campaign.values
        r = summarise(y, preds, g, subset, ymax)
        rows += r
        banner(f"{subset}: n={len(sub)}, {sub.campaign.nunique()} campanas", "-")
        for x in r:
            print(f"  {x['model']:20s} R2={x['R2']:+.3f} [{x['R2_lo']:+.3f}, "
                  f"{x['R2_hi']:+.3f}] RMSE={x['RMSE']:.2f} MAE={x['MAE']:.2f} "
                  f"bias={x['bias']:+.2f} max={x['max_prediction_m']:.1f} "
                  f"absurd={x['n_absurd']}")
        # diferencias pareadas de R2 contra el RF
        for other in ["linear", "kloiber", "two_band"]:
            dr2 = metrics(y, preds["rf"])["R2"] - metrics(y, preds[other])["R2"]
            lo, hi = cluster_boot(y, preds["rf"], g, p2=preds[other])
            comps.append({"subset": subset, "comparison": f"rf - {other}",
                          "dR2": round(dr2, 3), "dR2_lo": lo, "dR2_hi": hi,
                          "excludes_0": bool(lo > 0 or hi < 0)})
            print(f"  RF - {other:9s} dR2={dr2:+.3f} [{lo:+.3f}, {hi:+.3f}]")

    # modelo conjunto evaluado por sensor: mismas filas que el ajuste a un
    # solo sensor, para ver si separar los sensores ayuda
    y, preds = run_all(d)
    joint = []
    for s in ["S2", "LS"]:
        m_ = (d.sensor == s).values
        for key in MODELS:
            mm = metrics(y[m_], preds[key][m_])
            lo, hi = cluster_boot(y[m_], preds[key][m_], d.campaign.values[m_])
            joint.append({"sensor": s, "model": key, "fit_on": "both sensors",
                          **mm, "R2_lo": lo, "R2_hi": hi})
    res = pd.DataFrame(rows)
    sep = res[res.subset != "pooled"][["subset", "model", "R2", "R2_lo", "R2_hi",
                                       "RMSE", "n"]].rename(columns={"subset": "sensor"})
    sep = sep[~sep.model.str.endswith("unbounded")].assign(fit_on="single sensor")
    by_sensor = pd.concat([pd.DataFrame(joint), sep], ignore_index=True)
    banner("AJUSTE CONJUNTO vs UN SOLO SENSOR (mismas filas)", "-")
    print(by_sensor.pivot_table(index=["sensor", "model"], columns="fit_on",
                                values="R2").to_string())

    res.to_csv(TIDY / "benchmark_models.csv", index=False)
    comp = pd.DataFrame(comps)
    comp.to_csv(TIDY / "model_comparison_ci.csv", index=False)

    def rec(subset, model):
        r = res[(res.subset == subset) & (res.model == model)].iloc[0]
        return {k: (r[k].item() if hasattr(r[k], "item") else r[k])
                for k in ["R2", "R2_lo", "R2_hi", "RMSE", "MAE", "bias", "n",
                          "max_prediction_m", "n_absurd"]}

    out = {
        "design": "leave-one-campaign-out (9 folds), analysis_dataset (HLS, +-10 d)",
        "n": int(len(d)), "secchi_max_m": ymax,
        "bound": f"[{SECCHI_CLIP_LO}, {ABSURD_FACTOR} x training-fold max]",
        "absurd_threshold_m": round(ABSURD_FACTOR * ymax, 2),
        "log_ratio_floor": LOG_RATIO_FLOOR,
        "n_rows_at_log_ratio_floor": int((d.B2_B3 < LOG_RATIO_FLOOR).sum()),
        "pooled": {k: rec("pooled", k) for k in res[res.subset == "pooled"].model},
        "per_sensor_fit": {s: {k: rec(s, k) for k in res[res.subset == s].model}
                           for s in ["S2", "LS"]},
        "joint_fit_scored_by_sensor": pd.DataFrame(joint).to_dict("records"),
        "paired_dR2": comp.to_dict("records"),
        "bounding_matters": {
            k: {"R2_bounded": rec("pooled", k)["R2"],
                "R2_unbounded": rec("pooled", f"{k}_unbounded")["R2"],
                "n_absurd_unbounded": rec("pooled", f"{k}_unbounded")["n_absurd"],
                "max_unbounded_m": rec("pooled", f"{k}_unbounded")["max_prediction_m"]}
            for k in ["kloiber", "two_band"]},
    }
    json.dump(out, open(MET / "benchmark.json", "w"), indent=2)
    banner("SALIDAS")
    for f in [TIDY / "benchmark_models.csv", TIDY / "model_comparison_ci.csv",
              MET / "benchmark.json"]:
        print(f"  {f}")


if __name__ == "__main__":
    main()
