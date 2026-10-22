"""
p06_model_selection.py -- ¿mejora algo afinar el modelo o cambiar de algoritmo?

Validacion ANIDADA sobre analysis_dataset.csv (HLS, +-10 d):
  externa = dejar fuera una campana (9 folds, diseno primario);
  interna = GroupKFold(4) por campana sobre el entrenamiento externo, donde
            Optuna (TPE, semilla fija) elige hiperparametros minimizando RMSE.
Configuraciones: RF, ExtraTrees, XGBoost y LightGBM sin afinar y afinados;
objetivo Zsd y ln(Zsd) para RF y XGB; conjuntos de features (12 / visibles /
solo ratios) con el mejor algoritmo afinado; ensemble de igual peso de los
cuatro afinados.

Para cada configuracion: R2/RMSE OOF con IC bootstrap por campanas y el IC
pareado de la diferencia de R2 contra el RF sin afinar. La particion es fija,
asi que la variacion entre semillas no mide la incertidumbre de la seleccion;
el IC pareado si.

Las predicciones OOF de cada configuracion se guardan en logs_local/p06_cache
y se reutilizan si el script se relanza.

Salidas: results/tidy/model_selection.csv
         results/metrics/model_selection.json
         results/models/best_params.json
"""
import json
import time
import warnings

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler

from p00_config import (FEATURES, FEATURES_RATIO_ONLY, FEATURES_VIS, MET,
                        PRIMARY, PROC, ROOT, SECCHI_CLIP_LO, SEED, TIDY, banner,
                        cluster_boot, metrics, splits)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = 25
INNER_SPLITS = 4
N_JOBS = 4
MODELS = ROOT / "results/models"
CACHE = ROOT / "logs_local/p06_cache"
MODELS.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

FAMILIES = ["rf", "et", "xgb", "lgbm"]
FAMILY_LABELS = {"rf": "Random Forest", "et": "Extra Trees",
                 "xgb": "XGBoost", "lgbm": "LightGBM", "ensemble": "Ensemble"}
FEATURE_SETS = {"all12": FEATURES, "visible": FEATURES_VIS,
                "ratios": FEATURES_RATIO_ONLY}
# por defecto: RF igual a make_rf de p00
DEFAULTS = {"rf": dict(n_estimators=500, max_depth=12, min_samples_leaf=3),
            "et": dict(n_estimators=500, max_depth=12, min_samples_leaf=3),
            "xgb": dict(n_estimators=600, max_depth=5, learning_rate=0.03,
                        subsample=0.8, colsample_bytree=0.8),
            "lgbm": dict(n_estimators=600, num_leaves=31, learning_rate=0.03,
                         subsample=0.8, colsample_bytree=0.8)}


def suggest(trial, family):
    if family in ("rf", "et"):
        return dict(n_estimators=trial.suggest_int("n_estimators", 100, 600, step=100),
                    max_depth=trial.suggest_int("max_depth", 3, 24),
                    min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
                    max_features=trial.suggest_float("max_features", 0.2, 1.0))
    if family == "xgb":
        return dict(n_estimators=trial.suggest_int("n_estimators", 100, 600, step=100),
                    max_depth=trial.suggest_int("max_depth", 2, 6),
                    learning_rate=trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                    subsample=trial.suggest_float("subsample", 0.5, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
                    min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
                    reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True))
    if family == "lgbm":
        return dict(n_estimators=trial.suggest_int("n_estimators", 100, 1000, step=100),
                    num_leaves=trial.suggest_int("num_leaves", 4, 64),
                    learning_rate=trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                    min_child_samples=trial.suggest_int("min_child_samples", 5, 60),
                    subsample=trial.suggest_float("subsample", 0.5, 1.0),
                    subsample_freq=1,
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
                    reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True))
    raise ValueError(family)


def build(family, params, seed=SEED):
    if family == "rf":
        return RandomForestRegressor(random_state=seed, n_jobs=N_JOBS, **params)
    if family == "et":
        return ExtraTreesRegressor(random_state=seed, n_jobs=N_JOBS, **params)
    if family == "xgb":
        from xgboost import XGBRegressor
        return XGBRegressor(random_state=seed, n_jobs=N_JOBS, tree_method="hist",
                            verbosity=0, **params)
    if family == "lgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(random_state=seed, n_jobs=N_JOBS, verbose=-1, **params)
    raise ValueError(family)


def fit_predict(family, params, Xtr, ytr, Xte, log_target):
    sc = RobustScaler().fit(Xtr)
    m = build(family, params)
    m.fit(sc.transform(Xtr), np.log(ytr) if log_target else ytr)
    p = m.predict(sc.transform(Xte))
    return np.clip(np.exp(p) if log_target else p, SECCHI_CLIP_LO, None)


def tune(family, X, y, g, log_target):
    """Estudio Optuna sobre un conjunto de entrenamiento (CV interna por campana)."""
    inner = list(GroupKFold(INNER_SPLITS).split(X, y, g))

    def objective(trial):
        pr = suggest(trial, family)
        sq, n = 0.0, 0
        for k, (itr, ite) in enumerate(inner):
            p = fit_predict(family, pr, X[itr], y[itr], X[ite], log_target)
            sq += float(np.sum((y[ite] - p) ** 2))
            n += len(ite)
            trial.report(np.sqrt(sq / n), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return np.sqrt(sq / n)

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    return dict(study.best_params), float(study.best_value)


def nested(d, cfg):
    """OOF de una configuracion; con cache en disco."""
    f = CACHE / f"{cfg['key']}.json"
    if f.exists():
        return json.load(open(f))
    t0 = time.time()
    X, y, g = d[FEATURE_SETS[cfg["features"]]].values, d.secchi.values, d.campaign.values
    oof = np.full(len(y), np.nan)
    chosen = []
    for k, (tr, te) in enumerate(splits(d, PRIMARY)):
        if cfg["tuned"]:
            params, inner_rmse = tune(cfg["family"], X[tr], y[tr], g[tr], cfg["log"])
        else:
            params, inner_rmse = DEFAULTS[cfg["family"]], None
        oof[te] = fit_predict(cfg["family"], params, X[tr], y[tr], X[te], cfg["log"])
        chosen.append({"held_out_campaign": str(g[te][0]), "inner_RMSE":
                       None if inner_rmse is None else round(inner_rmse, 4), **params})
        if cfg["tuned"]:
            print(f"      fold {k + 1}/9 ({g[te][0]}) {time.time() - t0:.0f}s", flush=True)
    out = {"oof": oof.tolist(), "params": chosen,
           "seconds": round(time.time() - t0, 1)}
    json.dump(out, open(f, "w"))
    return out


def cfg(family, tuned, log=False, features="all12"):
    key = f"{family}_{'tuned' if tuned else 'default'}_{'log' if log else 'raw'}_{features}"
    return {"key": key, "family": family, "tuned": tuned, "log": log,
            "features": features}


def main():
    banner("p06 -- SELECCION DE MODELO EN VALIDACION ANIDADA")
    d = pd.read_csv(PROC / "analysis_dataset.csv", dtype={"campaign": str})
    d = d.reset_index(drop=True)
    y, g = d.secchi.values, d.campaign.values
    print(f"  n={len(d)} | {d.campaign.nunique()} campanas | externa: dejar fuera "
          f"una campana | interna: GroupKFold({INNER_SPLITS}) | Optuna TPE "
          f"{N_TRIALS} trials", flush=True)

    plan = ([cfg(f, False) for f in FAMILIES]
            + [cfg(f, False, log=True) for f in ("rf", "xgb")]
            + [cfg(f, True) for f in FAMILIES]
            + [cfg(f, True, log=True) for f in ("rf", "xgb")])
    preds, info = {}, {}
    t_all = time.time()

    def run(c):
        print(f"  -> {c['key']}", flush=True)
        r = nested(d, c)
        preds[c["key"]], info[c["key"]] = np.asarray(r["oof"]), {**c, **r}
        m = metrics(y, preds[c["key"]])
        print(f"     R2={m['R2']:+.3f} RMSE={m['RMSE']:.3f} ({r['seconds']:.0f}s; "
              f"total {time.time() - t_all:.0f}s)", flush=True)

    for c in plan:
        run(c)

    # mejor algoritmo afinado (Zsd, 12 features) para los conjuntos de features
    tuned_raw = [c for c in plan if c["tuned"] and not c["log"]]
    best_fam = max(tuned_raw, key=lambda c: metrics(y, preds[c["key"]])["R2"])["family"]
    print(f"  mejor algoritmo afinado: {best_fam}", flush=True)
    feat_cfgs = [cfg(best_fam, True, features=fs) for fs in ("visible", "ratios")]
    for c in feat_cfgs:
        run(c)

    ens_key = "ensemble_tuned_raw_all12"
    preds[ens_key] = np.mean([preds[c["key"]] for c in tuned_raw], axis=0)
    info[ens_key] = {"key": ens_key, "family": "ensemble", "tuned": True,
                     "log": False, "features": "all12", "params": None}

    # --- tabla con IC -------------------------------------------------------
    ref = "rf_default_raw_all12"
    rows = []
    for key, p in preds.items():
        c = info[key]
        m = metrics(y, p)
        lo, hi = cluster_boot(y, p, g)
        rlo, rhi = cluster_boot(y, p, g, stat="RMSE")
        dr2 = round(m["R2"] - metrics(y, preds[ref])["R2"], 3)
        dlo, dhi = (0.0, 0.0) if key == ref else cluster_boot(y, p, g, p2=preds[ref])
        rows.append({"config": key, "family": c["family"],
                     "learner": FAMILY_LABELS[c["family"]], "tuned": c["tuned"],
                     "target": "ln(Zsd)" if c["log"] else "Zsd",
                     "features": c["features"], **m, "R2_lo": lo, "R2_hi": hi,
                     "RMSE_lo": rlo, "RMSE_hi": rhi, "dR2_vs_rf_default": dr2,
                     "dR2_lo": dlo, "dR2_hi": dhi,
                     "dR2_excludes_0": bool(key != ref and (dlo > 0 or dhi < 0))})
    res = pd.DataFrame(rows).sort_values("R2", ascending=False)
    res.to_csv(TIDY / "model_selection.csv", index=False)

    banner("RESULTADOS (dR2 pareado contra RF sin afinar)")
    print(res[["config", "R2", "R2_lo", "R2_hi", "RMSE", "dR2_vs_rf_default",
               "dR2_lo", "dR2_hi", "dR2_excludes_0"]].to_string(index=False))

    best = res.iloc[0]
    any_sig_better = bool(((res.dR2_lo > 0)).any())
    any_sig_worse = bool(((res.dR2_hi < 0)).any())
    print(f"\n  Mejor configuracion: {best.config} (dR2 {best.dR2_vs_rf_default:+.3f} "
          f"[{best.dR2_lo:+.3f}, {best.dR2_hi:+.3f}])")
    print(f"  Alguna configuracion significativamente mejor que el RF sin afinar: "
          f"{any_sig_better}; alguna peor: {any_sig_worse}")
    print(f"  Tiempo total: {time.time() - t_all:.0f}s")

    def r(x):
        return {k: (v.item() if hasattr(v, "item") else v) for k, v in x.items()}

    json.dump({
        "design": (f"nested CV: outer leave-one-campaign-out (9 folds), inner "
                   f"GroupKFold({INNER_SPLITS}) by campaign; Optuna TPE seed {SEED}, "
                   f"{N_TRIALS} trials, MedianPruner; objective pooled inner RMSE (m)"),
        "n": int(len(d)), "n_trials": N_TRIALS,
        "reference": r(res[res.config == ref].iloc[0].to_dict()),
        "best": r(best.to_dict()),
        "best_tuned_learner_for_feature_sets": best_fam,
        "feature_set_note": ("the learner for the feature-set comparison was chosen "
                             "on outer OOF R2; a sensitivity analysis, not a selection"),
        "any_config_dR2_CI_above_0": any_sig_better,
        "any_config_dR2_CI_below_0": any_sig_worse,
        "configs_dR2_CI_excludes_0": res.loc[res.dR2_excludes_0, "config"].tolist(),
        "runtime_s": {k: info[k].get("seconds") for k in info},
        "all_configurations": [r(x) for x in res.to_dict("records")],
    }, open(MET / "model_selection.json", "w"), indent=2)
    json.dump({k: info[k]["params"] for k in info if info[k].get("params") is not None
               and info[k]["tuned"]},
              open(MODELS / "best_params.json", "w"), indent=2)
    banner("SALIDAS")
    for f in [TIDY / "model_selection.csv", MET / "model_selection.json",
              MODELS / "best_params.json"]:
        print(f"  {f}")


if __name__ == "__main__":
    main()
