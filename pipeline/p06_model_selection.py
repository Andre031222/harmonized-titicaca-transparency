"""
p06_model_selection.py -- ¿SE PUEDEN MEJORAR LOS NUMEROS, O AHI SE QUEDAN?

Responde la pregunta empiricamente en lugar de suponerla. Prueba, todo bajo
el mismo diseno honesto (GroupKFold bloqueado por CAMPANA):

  (A) Familias de modelo con hiperparametros optimizados por Optuna dentro de
      VALIDACION ANIDADA. La optimizacion ocurre en un GroupKFold interno
      sobre la particion de entrenamiento de cada fold externo, asi que el
      fold externo nunca influye en la eleccion de hiperparametros. Sin esto
      el "modelo optimizado" siempre parece mejor de lo que es.
  (B) Transformaciones del objetivo (identidad vs logaritmo).
  (C) Conjuntos de features.
  (D) Un ensemble por promedio de las mejores familias.

La conclusion que salga de aqui es la que va al manuscrito. Si la mejora es
menor que la variabilidad entre semillas, NO es una mejora.

Salidas: results/tidy/model_selection.csv
         results/tidy/seed_stability.csv
         results/metrics/model_selection.json
         results/models/best_params.json
"""
import json
import warnings

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler

from p00_config import (FEATURES, FEATURES_RATIO_ONLY, FEATURES_VIS, MET,
                        MODELS, N_SPLITS, PROC, SECCHI_CLIP_LO, SEED, TIDY,
                        banner, metrics)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = 18
INNER_SPLITS = 3


def load():
    d = pd.read_csv(PROC / "analysis_dataset.csv")
    d["date"] = pd.to_datetime(d["date"])
    return d


# --- espacios de busqueda ---------------------------------------------------
def suggest(trial, family):
    if family == "rf":
        return dict(n_estimators=trial.suggest_int("n_estimators", 200, 900, step=100),
                    max_depth=trial.suggest_int("max_depth", 4, 24),
                    min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 12),
                    max_features=trial.suggest_float("max_features", 0.2, 1.0))
    if family == "et":
        return dict(n_estimators=trial.suggest_int("n_estimators", 200, 900, step=100),
                    max_depth=trial.suggest_int("max_depth", 4, 24),
                    min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 12),
                    max_features=trial.suggest_float("max_features", 0.2, 1.0))
    if family == "xgb":
        return dict(n_estimators=trial.suggest_int("n_estimators", 200, 1200, step=100),
                    max_depth=trial.suggest_int("max_depth", 2, 10),
                    learning_rate=trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                    subsample=trial.suggest_float("subsample", 0.5, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
                    min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
                    reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True))
    if family == "lgbm":
        return dict(n_estimators=trial.suggest_int("n_estimators", 200, 1200, step=100),
                    num_leaves=trial.suggest_int("num_leaves", 7, 127),
                    learning_rate=trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                    min_child_samples=trial.suggest_int("min_child_samples", 5, 60),
                    subsample=trial.suggest_float("subsample", 0.5, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
                    reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True))
    raise ValueError(family)


def build(family, params, seed=SEED):
    p = dict(params)
    if family == "rf":
        return RandomForestRegressor(random_state=seed, n_jobs=-1, **p)
    if family == "et":
        return ExtraTreesRegressor(random_state=seed, n_jobs=-1, **p)
    if family == "xgb":
        from xgboost import XGBRegressor
        return XGBRegressor(random_state=seed, n_jobs=-1, tree_method="hist",
                            verbosity=0, **p)
    if family == "lgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(random_state=seed, n_jobs=-1, verbose=-1, **p)
    raise ValueError(family)


DEFAULTS = {"rf": dict(n_estimators=500, max_depth=12, min_samples_leaf=3),
            "et": dict(n_estimators=500, max_depth=12, min_samples_leaf=3),
            "xgb": dict(n_estimators=600, max_depth=5, learning_rate=0.03,
                        subsample=0.8, colsample_bytree=0.8),
            "lgbm": dict(n_estimators=600, num_leaves=31, learning_rate=0.03,
                         subsample=0.8, colsample_bytree=0.8)}


# --- nucleo de evaluacion ---------------------------------------------------
def fit_predict(family, params, Xtr, ytr, Xte, log_target, seed=SEED):
    sc = RobustScaler().fit(Xtr)
    m = build(family, params, seed)
    m.fit(sc.transform(Xtr), np.log(ytr) if log_target else ytr)
    p = m.predict(sc.transform(Xte))
    return np.clip(np.exp(p) if log_target else p, SECCHI_CLIP_LO, None)


def nested_cv(d, feats, family, log_target=False, tune=True, seed=SEED,
              n_trials=N_TRIALS):
    """CV anidada: Optuna dentro de cada fold externo, nunca sobre el test."""
    X = d[feats].values
    y = d.secchi.values
    g = d.campaign_date.values
    oof = np.full(len(y), np.nan)
    chosen = []
    for tr, te in GroupKFold(N_SPLITS).split(X, y, g):
        if not tune:
            params = DEFAULTS[family]
        else:
            Xi, yi, gi = X[tr], y[tr], g[tr]

            def objective(trial):
                pr = suggest(trial, family)
                errs = []
                for itr, ite in GroupKFold(INNER_SPLITS).split(Xi, yi, gi):
                    pred = fit_predict(family, pr, Xi[itr], yi[itr], Xi[ite],
                                       log_target, seed)
                    errs.append(np.sqrt(np.mean((yi[ite] - pred) ** 2)))
                return float(np.mean(errs))

            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=seed))
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            params = study.best_params
        chosen.append(params)
        oof[te] = fit_predict(family, params, X[tr], y[tr], X[te], log_target, seed)
    return y, oof, chosen


# ---------------------------------------------------------------------------
def main():
    banner("p06 -- ¿SE PUEDEN MEJORAR LOS NUMEROS?")
    d = load()
    print(f"  n={len(d)} match-ups | {d.campaign_date.nunique()} campanas")
    print(f"  Diseno: GroupKFold({N_SPLITS}) por CAMPANA, con Optuna "
          f"({N_TRIALS} trials) en un GroupKFold({INNER_SPLITS}) INTERNO.")
    print("  El fold externo nunca influye en la eleccion de hiperparametros.\n")

    rows, best_params = [], {}

    def record(label, family, feats_name, tuned, log_t, yt, yp, params=None):
        m = metrics(yt, yp)
        rows.append({"label": label, "family": family, "features": feats_name,
                     "tuned": tuned, "log_target": log_t, **m})
        flag = "*" if m["R2"] > 0.60 else " "
        print(f" {flag} {label:44s} R2={m['R2']:+.3f} RMSE={m['RMSE']:.2f} "
              f"MAE={m['MAE']:.2f}")
        if params is not None:
            best_params[label] = params
        return m

    # --- (A) referencia sin optimizar --------------------------------------
    banner("(A) REFERENCIA: hiperparametros por defecto (lo del manuscrito)", "-")
    base = None
    for fam in ["rf", "et", "xgb", "lgbm"]:
        yt, yp, _ = nested_cv(d, FEATURES, fam, tune=False)
        m = record(f"{fam.upper()} por defecto", fam, "all12", False, False, yt, yp)
        if fam == "rf":
            base = m

    # --- (B) con optimizacion anidada --------------------------------------
    banner("(B) CON OPTUNA EN VALIDACION ANIDADA", "-")
    for fam in ["rf", "et", "xgb", "lgbm"]:
        yt, yp, ch = nested_cv(d, FEATURES, fam, tune=True)
        record(f"{fam.upper()} optimizado (anidado)", fam, "all12", True, False,
               yt, yp, ch)

    # --- (C) objetivo logaritmico ------------------------------------------
    banner("(C) OBJETIVO LOGARITMICO", "-")
    for fam in ["rf", "xgb"]:
        yt, yp, ch = nested_cv(d, FEATURES, fam, log_target=True, tune=True)
        record(f"{fam.upper()} optimizado + log(Zsd)", fam, "all12", True, True,
               yt, yp, ch)

    # --- (D) conjuntos de features -----------------------------------------
    banner("(D) CONJUNTOS DE FEATURES (con el mejor modelo optimizado)", "-")
    best_fam = max((r for r in rows if r["tuned"]), key=lambda r: r["R2"])["family"]
    print(f"  mejor familia hasta ahora: {best_fam.upper()}\n")
    for fname, feats in [("visible_only", FEATURES_VIS),
                         ("ratios_only", FEATURES_RATIO_ONLY)]:
        yt, yp, ch = nested_cv(d, feats, best_fam, tune=True)
        record(f"{best_fam.upper()} optimizado -- {fname}", best_fam, fname,
               True, False, yt, yp, ch)

    # --- (E) ensemble -------------------------------------------------------
    banner("(E) ENSEMBLE (promedio de las familias optimizadas)", "-")
    preds = {}
    for fam in ["rf", "et", "xgb", "lgbm"]:
        yt, yp, _ = nested_cv(d, FEATURES, fam, tune=True)
        preds[fam] = yp
    ens = np.mean(np.column_stack(list(preds.values())), axis=1)
    record("Ensemble (RF+ET+XGB+LGBM optimizados)", "ensemble", "all12", True,
           False, yt, ens)

    # --- (F) estabilidad entre semillas ------------------------------------
    banner("(F) ESTABILIDAD ENTRE SEMILLAS -- ¿cuanto es ruido?", "-")
    print("  Cualquier 'mejora' menor que esta dispersion no es una mejora.\n")
    srows = []
    for seed in [0, 7, 42, 123, 2024]:
        yt, yp, _ = nested_cv(d, FEATURES, "rf", tune=False, seed=seed)
        m = metrics(yt, yp)
        srows.append({"seed": seed, **m})
        print(f"    RF por defecto, semilla {seed:<5d} R2={m['R2']:+.3f} "
              f"RMSE={m['RMSE']:.2f}")
    sd_r2 = float(np.std([r["R2"] for r in srows]))
    rng_r2 = float(max(r["R2"] for r in srows) - min(r["R2"] for r in srows))
    print(f"\n    sd(R2) entre semillas = {sd_r2:.4f} | rango = {rng_r2:.4f}")

    # --- veredicto ----------------------------------------------------------
    res = pd.DataFrame(rows).sort_values("R2", ascending=False)
    banner("VEREDICTO")
    print(res[["label", "R2", "RMSE", "MAE", "bias"]].to_string(index=False))
    best = res.iloc[0]
    gain = float(best.R2) - base["R2"]
    print(f"\n  Mejor configuracion : {best.label}")
    print(f"  R2 {base['R2']:.3f} -> {best.R2:.3f}  (ganancia {gain:+.3f})")
    print(f"  Ruido entre semillas: sd={sd_r2:.3f}, rango={rng_r2:.3f}")
    if gain <= rng_r2:
        verdict = ("La ganancia por optimizacion cae DENTRO del ruido entre "
                   "semillas. No es una mejora real: el techo lo pone el dato, "
                   "no el modelo. Se mantiene el Random Forest por defecto, que "
                   "es mas simple y mas facil de reproducir.")
    elif gain < 0.03:
        verdict = ("La ganancia es real pero marginal (<0.03 de R2). No cambia "
                   "ninguna conclusion del articulo; se reporta como analisis de "
                   "sensibilidad y se mantiene el Random Forest.")
    else:
        verdict = (f"La ganancia es sustancial ({gain:+.3f} de R2). Merece la pena "
                   f"adoptar {best.label} como modelo principal.")
    print(f"\n  => {verdict}")

    res.to_csv(TIDY / "model_selection.csv", index=False)
    pd.DataFrame(srows).to_csv(TIDY / "seed_stability.csv", index=False)
    json.dump({"design": "nested CV; GroupKFold(5) outer by campaign date, "
                         "GroupKFold(3) inner, Optuna TPE",
               "n_trials": N_TRIALS,
               "baseline_rf_default": base,
               "best": best.to_dict(),
               "gain_R2": round(gain, 4),
               "seed_noise_sd_R2": round(sd_r2, 4),
               "seed_noise_range_R2": round(rng_r2, 4),
               "verdict": verdict,
               "all_configurations": rows},
              open(MET / "model_selection.json", "w"), indent=2)
    json.dump({k: [dict(p) for p in v] for k, v in best_params.items()},
              open(MODELS / "best_params.json", "w"), indent=2, default=str)

    banner("SALIDAS")
    print(f"  {TIDY / 'model_selection.csv'}")
    print(f"  {TIDY / 'seed_stability.csv'}")
    print(f"  {MET / 'model_selection.json'}")


if __name__ == "__main__":
    main()
