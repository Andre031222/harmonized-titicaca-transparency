"""
harmonization_check.py -- prueba de dos muestras con clasificador, reutilizable.

Pregunta si un clasificador (bosque aleatorio, 300 arboles) todavia distingue
dos sensores despues de armonizarlos. Si fueran intercambiables, la exactitud
balanceada (BA) y el AUC fuera de fold estarian en el nivel de azar (0.5).

Dos salvaguardas que una prueba ingenua no tiene:
  * validacion agrupada por `group_col` (dejando fuera un grupo entero cada
    vez). El grupo debe ser la unidad que confunde sensor y contexto: si un
    sensor falta en ciertas campanas, folds al azar dejarian que el modelo
    reconozca la campana en lugar del sensor.
  * una DISTRIBUCION de controles negativos: dentro de un mismo sensor, las
    etiquetas son mitades al azar de unidades enteras (`unit_col`, p. ej. la
    fecha de muestreo; la etiqueta es constante dentro de la unidad), repetido
    `n_controls` veces. Dice cuanto se aleja del azar la prueba por si sola
    cuando las dos "muestras" solo difieren en fechas.
El IC del 95 % de la prueba real es un bootstrap de conglomerados por
`group_col` sobre las predicciones fuera de fold.

Solo depende de numpy, pandas y scikit-learn (y joblib, que viene con
scikit-learn); se puede copiar a otro proyecto:

    from harmonization_check import harmonization_check
    res = harmonization_check(df, features=["B2", "B3", "B4"],
                              sensor_col="sensor", group_col="campaign",
                              unit_col="sampling_date")

    python harmonization_check.py   # demo sobre data/processed/analysis_dataset.csv
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.utils.parallel import Parallel, delayed

N_TREES = 300


def group_folds(groups, n_splits=None):
    """Folds agrupados: deja fuera un grupo entero (n_splits=None) o
    GroupKFold(n_splits)."""
    groups = np.asarray(groups)
    idx = np.arange(len(groups))
    cv = LeaveOneGroupOut() if n_splits is None else GroupKFold(n_splits)
    return list(cv.split(idx, groups=groups))


def _fit_fold(df, features, y, tr, te, transform, seed):
    a, b = df.iloc[tr], df.iloc[te]
    if transform is not None:
        a, b = transform(a, b)
    if len(np.unique(y[tr])) < 2:        # fold de entrenamiento con una sola clase
        return te, np.full(len(te), float(y[tr][0]))
    clf = RandomForestClassifier(N_TREES, random_state=seed, n_jobs=1)
    clf.fit(a[features].values, y[tr])
    return te, clf.predict_proba(b[features].values)[:, 1]


def oof_proba(df, features, y, folds, transform=None, seed=42, n_jobs=-1):
    """Probabilidad fuera de fold. `transform(train_df, test_df)` permite
    ajustar algo (p. ej. una recalibracion) solo con el entrenamiento."""
    df = df.reset_index(drop=True)
    if n_jobs == 1:
        res = [_fit_fold(df, features, y, tr, te, transform, seed) for tr, te in folds]
    else:
        res = Parallel(n_jobs=n_jobs)(delayed(_fit_fold)(
            df, features, y, tr, te, transform, seed) for tr, te in folds)
    p = np.full(len(df), np.nan)
    for te, pr in res:
        p[te] = pr
    return p


def scores(y, proba):
    y, proba = np.asarray(y), np.asarray(proba)
    pred = (proba >= 0.5).astype(int)
    return {"balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "auc": float(roc_auc_score(y, proba)) if len(np.unique(y)) == 2 else np.nan,
            "accuracy": float(accuracy_score(y, pred))}


def cluster_boot_cls(y, proba, groups, n_boot=2000, seed=42):
    """IC 95 % de BA y AUC remuestreando grupos enteros (con reemplazo)."""
    y, proba, groups = np.asarray(y), np.asarray(proba), np.asarray(groups)
    ug = np.unique(groups)
    pos = {g: np.flatnonzero(groups == g) for g in ug}
    rng = np.random.default_rng(seed)
    ba, auc = [], []
    for _ in range(n_boot):
        ix = np.concatenate([pos[g] for g in rng.choice(ug, len(ug))])
        if len(np.unique(y[ix])) < 2:
            continue                      # remuestra con una sola clase
        s = scores(y[ix], proba[ix])
        ba.append(s["balanced_accuracy"])
        auc.append(s["auc"])
    q = [0.025, 0.975]
    return {"ba_lo": float(np.quantile(ba, q[0])), "ba_hi": float(np.quantile(ba, q[1])),
            "auc_lo": float(np.quantile(auc, q[0])), "auc_hi": float(np.quantile(auc, q[1])),
            "n_boot_valid": len(ba)}


def two_sample_test(df, features, label_col, positive, groups, folds=None,
                    transform=None, n_boot=2000, seed=42, n_jobs=-1):
    """Prueba real: BA y AUC fuera de fold + IC por conglomerados (`groups`).
    Sin `folds`, deja fuera un grupo entero cada vez."""
    df = df.reset_index(drop=True)
    groups = np.asarray(groups)
    y = (df[label_col] == positive).astype(int).values
    folds = folds if folds is not None else group_folds(groups)
    p = oof_proba(df, features, y, folds, transform, seed, n_jobs)
    out = {"n": int(len(df)), "n_positive": int(y.sum()),
           "n_groups": int(len(np.unique(groups))), "n_folds": int(len(folds)),
           "chance": float(max(y.mean(), 1 - y.mean())), **scores(y, p)}
    if n_boot:
        out.update(cluster_boot_cls(y, p, groups, n_boot, seed))
    return out, p


def _halves(units, rng):
    u = np.unique(units)
    pick = set(rng.choice(u, len(u) // 2, replace=False))
    return np.array([v in pick for v in units], dtype=int)


def _one_control(df, features, units, folds, seed_rep, seed):
    rng = np.random.default_rng(seed_rep)
    y = _halves(units, rng)
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    p = oof_proba(df, features, y, folds, None, seed, n_jobs=1)
    s = scores(y, p)
    return s["balanced_accuracy"], s["auc"]


def negative_controls(df, features, groups, unit_col=None, n_controls=100,
                      seed=42, folds=None, n_jobs=-1):
    """Distribucion de BA y AUC cuando las dos muestras son mitades al azar
    del MISMO conjunto. Etiqueta constante dentro de `unit_col` (None: filas).
    Devuelve dos arrays de largo n_controls."""
    df = df.reset_index(drop=True)
    units = df[unit_col].values if unit_col else np.arange(len(df))
    folds = folds if folds is not None else group_folds(groups)
    seeds = np.random.SeedSequence(seed).generate_state(n_controls)
    res = Parallel(n_jobs=n_jobs)(delayed(_one_control)(
        df, features, units, folds, int(s), seed) for s in seeds)
    res = np.array(res, dtype=float)
    return res[:, 0], res[:, 1]


def summarize_controls(ba, auc):
    ba, auc = ba[~np.isnan(ba)], auc[~np.isnan(auc)]
    return {"n_controls": int(len(ba)),
            "ba_mean": float(ba.mean()), "ba_lo": float(np.quantile(ba, .025)),
            "ba_hi": float(np.quantile(ba, .975)), "ba_max": float(ba.max()),
            "auc_mean": float(auc.mean()), "auc_lo": float(np.quantile(auc, .025)),
            "auc_hi": float(np.quantile(auc, .975)), "auc_max": float(auc.max())}


def harmonization_check(df, features, sensor_col="sensor", group_col="campaign",
                        n_controls=100, seed=42, unit_col="sampling_date",
                        n_boot=2000, n_jobs=-1):
    """Prueba real entre sensores y la distribucion de controles negativos.

    Devuelve una fila `sensors` (BA y AUC con IC 95 % por bootstrap de
    conglomerados sobre `group_col`) y una fila `control_<sensor>` por sensor
    (media y percentiles 2.5/97.5 de `n_controls` controles con mitades al azar
    de `unit_col`; si la columna no existe, mitades de filas). `p_vs_control`
    es la fraccion (+1) de controles con BA >= la de la prueba real.
    Lectura: los sensores son intercambiables si la BA real cae dentro de la
    banda de los controles.
    """
    features = list(features)
    df = df.dropna(subset=features).reset_index(drop=True)
    if unit_col is not None and unit_col not in df.columns:
        unit_col = None
    sensors = sorted(pd.unique(df[sensor_col]))
    if len(sensors) != 2:
        raise ValueError(f"se esperaban dos sensores en '{sensor_col}', hay {sensors}")
    real, _ = two_sample_test(df, features, sensor_col, sensors[1],
                              df[group_col].values, n_boot=n_boot, seed=seed,
                              n_jobs=n_jobs)
    rows = [{"test": "sensors", "positive": sensors[1], "n": real["n"],
             "n_groups": real["n_groups"],
             "balanced_accuracy": real["balanced_accuracy"], "auc": real["auc"],
             "ba_lo": real["ba_lo"], "ba_hi": real["ba_hi"],
             "auc_lo": real["auc_lo"], "auc_hi": real["auc_hi"],
             "interval": f"cluster bootstrap over {group_col}"}]
    for s in sensors:
        one = df[df[sensor_col] == s].reset_index(drop=True)
        ba, auc = negative_controls(one, features, one[group_col].values, unit_col,
                                    n_controls, seed, n_jobs=n_jobs)
        c = summarize_controls(ba, auc)
        rows.append({"test": f"control_{s}", "positive": None, "n": int(len(one)),
                     "n_groups": int(one[group_col].nunique()),
                     "balanced_accuracy": c["ba_mean"], "auc": c["auc_mean"],
                     "ba_lo": c["ba_lo"], "ba_hi": c["ba_hi"],
                     "auc_lo": c["auc_lo"], "auc_hi": c["auc_hi"],
                     "p_vs_control": float((1 + np.sum(ba >= real["balanced_accuracy"]))
                                           / (1 + np.sum(~np.isnan(ba)))),
                     "interval": f"2.5-97.5 % of {c['n_controls']} controls "
                                 f"(halves of {unit_col or 'rows'})"})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from pathlib import Path

    feats = ["B2", "B3", "B4", "B8", "B11", "B12",
             "NDWI", "NDTI", "B2_B3", "B4_B3", "B3_B2", "B2_B4"]
    path = Path(__file__).resolve().parent.parent / "data/processed/analysis_dataset.csv"
    d = pd.read_csv(path, dtype={"campaign": str})
    res = harmonization_check(d, feats, sensor_col="sensor", group_col="campaign",
                              n_controls=100, seed=42, unit_col="sampling_date")
    with pd.option_context("display.width", 200):
        print(res.round(3).to_string(index=False))
