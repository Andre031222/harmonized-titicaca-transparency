"""
harmonization_check.py -- La prueba de dos muestras de p02, reutilizable.

Pregunta si un clasificador todavia distingue los sensores despues de
armonizar. Si la armonizacion funcionara, la exactitud balanceada y el AUC
estarian en el nivel de azar; los controles negativos (dos mitades al azar del
MISMO sensor) muestran cuanto se aleja del azar la prueba por si sola.

Solo depende de numpy, pandas y scikit-learn, asi que se puede copiar a otro
proyecto sin el resto del pipeline:

    from harmonization_check import harmonization_check
    res = harmonization_check(df, features=["B2", "B3", "B4"],
                              sensor_col="sensor", group_col="sampling_date")
    print(res)

La validacion se agrupa por `group_col`: con folds al azar, observaciones del
mismo dia caerian a ambos lados y el clasificador podria reconocer el dia en
lugar del sensor.

    python harmonization_check.py     # corre la prueba sobre el dataset de este estudio
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold


def two_sample_test(df, features, label_col, positive, group_col,
                    n_splits=5, seed=42):
    """Clasificador de dos muestras con validacion agrupada."""
    df = df.reset_index(drop=True)
    y = (df[label_col] == positive).astype(int).values
    groups = df[group_col].values
    proba = np.zeros(len(df))
    for tr, te in GroupKFold(n_splits).split(df, y, groups):
        clf = RandomForestClassifier(300, random_state=seed, n_jobs=-1)
        clf.fit(df.loc[tr, features].values, y[tr])
        proba[te] = clf.predict_proba(df.loc[te, features].values)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {"n": int(len(df)),
            "chance": round(float(max(y.mean(), 1 - y.mean())), 4),
            "accuracy": round(float(accuracy_score(y, pred)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
            "auc": round(float(roc_auc_score(y, proba)), 4)}


def harmonization_check(df, features, sensor_col="sensor", group_col="date",
                        n_splits=5, seed=42):
    """Prueba real entre sensores y un control negativo por sensor.

    Devuelve una tabla con una fila por prueba. Se lee comparando la fila
    `sensors` con las filas `control_*`: si la armonizacion funciono, las tres
    estan cerca de 0.5 de exactitud balanceada y de AUC.
    """
    df = df.dropna(subset=list(features)).reset_index(drop=True)
    sensors = list(pd.unique(df[sensor_col]))
    if len(sensors) != 2:
        raise ValueError(f"se esperaban dos sensores en '{sensor_col}', hay {sensors}")
    rows = [{"test": "sensors",
             **two_sample_test(df, features, sensor_col, sensors[1], group_col,
                               n_splits, seed)}]
    rng = np.random.default_rng(seed)
    for s in sensors:
        one = df[df[sensor_col] == s].copy()
        one["_half"] = np.where(rng.random(len(one)) < 0.5, "A", "B")
        rows.append({"test": f"control_{s}",
                     **two_sample_test(one, features, "_half", "A", group_col,
                                       n_splits, seed)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from p00_config import FEATURES, PROC, SEED

    d = pd.read_csv(PROC / "analysis_dataset.csv")
    print(harmonization_check(d, FEATURES, sensor_col="sensor",
                              group_col="campaign_date", seed=SEED).to_string(index=False))
