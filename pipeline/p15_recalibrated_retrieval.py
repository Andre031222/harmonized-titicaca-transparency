"""
p15_recalibrated_retrieval.py -- mejora la recuperacion de Secchi agrupada si
antes se recalibra Sentinel-2 al espacio de Landsat (o al reves)?

p03 muestra que un mapeo de cuantiles por banda S2 -> LS, ajustado en pares,
vuelve indistinguibles a los sensores para un clasificador, y OLS/RMA no. Aqui
se pregunta si eso se traduce en mejor Secchi. Todo bajo el diseno primario
(dejando fuera una campana, splits(d, "campaign")) sobre analysis_dataset.csv.

Variantes (RF por defecto de p00, 12 features, escalado robusto en el fold):
  baseline          : HLS crudo (reproduce p04).
  {ols,rma,qm}_s2ls_paired : mapa por banda S2 -> LS ajustado SOLO con los
                      eventos pareados (misma ubicacion y fecha, ambos
                      sensores; matchups hls, +-10 d, incluye filas sin
                      Secchi) de las campanas de entrenamiento; se aplica a
                      las filas S2 de train y test y se recalculan indices.
  {ols,rma,qm}_s2ls_pm3d : igual con pares casi simultaneos (+-3 d, una
                      imagen por sensor; p03).
  qmfeat_s2ls_paired: mapeo de cuantiles de las 12 features (p03).
  {ols,rma,qm}_lss2_paired : direccion inversa, LS -> S2 (filas LS).
  sensor_feature    : HLS crudo + columna binaria del sensor.
  Referencia: modelos de un solo sensor en sus propias filas, y el modelo
  agrupado evaluado por sensor.
Por variante: R2 (IC bootstrap por campana), RMSE, sesgo, R2 por sensor, IC
pareado de la diferencia de R2 frente a baseline, y exactitud balanceada del
clasificador de sensor (harmonization_check.two_sample_test, folds por
campana) sobre las features recalibradas (todas las filas y eventos pareados).

Entrada : data/processed/analysis_dataset.csv, data/processed/matchups.csv
Salidas : results/tidy/recalibrated_retrieval.csv
          results/metrics/recalibrated_retrieval.json
"""
from functools import partial

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from harmonization_check import two_sample_test
from p00_config import (FEATURES, MET, N_BOOT, PRIMARY, PROC, SECCHI_CLIP_LO,
                        SEED, TIDY, banner, cluster_boot, make_rf, metrics,
                        splits, write_json)
from p03_harmonization import (apply_maps, fit_maps, paired, route_table,
                               single_image_pairs, W_ALL, W_NEAR)

SWAP = {"S2": "LS", "LS": "S2"}

# fuentes de pares: eventos pareados de matchups (hls), por campana
PAIRS = {"paired": paired(route_table("hls", W_ALL)),
         "pm3d": single_image_pairs(route_table("hls", W_NEAR))}


# --- recalibracion dentro del fold -------------------------------------------------
def recal(train, test, kind, source, direction):
    """Ajusta el mapa con los pares de las campanas de `train` y lo aplica a
    train y test. direction 's2ls' corrige S2; 'lsS2' corrige LS (se
    intercambian las etiquetas para reutilizar fit_maps/apply_maps de p03)."""
    pr = PAIRS[source]
    pr = pr[pr.campaign.isin(train.campaign.unique())]
    if direction == "lss2":
        sw = lambda x: x.assign(sensor=x.sensor.map(SWAP))
        f = fit_maps(sw(pr), kind)
        return (sw(apply_maps(sw(train), f)), sw(apply_maps(sw(test), f)))
    f = fit_maps(pr, kind)
    return apply_maps(train, f), apply_maps(test, f)


def n_train_pairs(d, source):
    pr = PAIRS[source]
    return [int(pr[pr.campaign.isin(d.campaign.iloc[tr].unique())].event.nunique())
            for tr, _ in splits(d, PRIMARY)]


def oof(d, feats, transform=None):
    """Como cv_oof de p00, con transformacion ajustada en el fold."""
    d = d.reset_index(drop=True)
    y = d.secchi.values
    p = np.full(len(d), np.nan)
    for tr, te in splits(d, PRIMARY):
        a, b = d.iloc[tr], d.iloc[te]
        if transform is not None:
            a, b = transform(a, b)
        sc = RobustScaler().fit(a[feats].values)
        m = make_rf()
        m.fit(sc.transform(a[feats].values), y[tr])
        p[te] = m.predict(sc.transform(b[feats].values))
    return y, np.clip(p, SECCHI_CLIP_LO, None)


VARIANTS = [  # nombre, etiqueta, metodo, pares, direccion
    ("baseline", "Raw HLS (baseline)", None, None, None),
    ("ols_s2ls_paired", "OLS S2->LS, paired events", "ols", "paired", "s2ls"),
    ("rma_s2ls_paired", "RMA S2->LS, paired events", "rma", "paired", "s2ls"),
    ("qm_s2ls_paired", "Quantile mapping S2->LS, paired events", "qm_bands", "paired", "s2ls"),
    ("qmfeat_s2ls_paired", "Quantile mapping of 12 features S2->LS, paired events",
     "qm_features", "paired", "s2ls"),
    ("ols_s2ls_pm3d", "OLS S2->LS, near-simultaneous pairs", "ols", "pm3d", "s2ls"),
    ("rma_s2ls_pm3d", "RMA S2->LS, near-simultaneous pairs", "rma", "pm3d", "s2ls"),
    ("qm_s2ls_pm3d", "Quantile mapping S2->LS, near-simultaneous pairs", "qm_bands", "pm3d", "s2ls"),
    ("ols_lss2_paired", "OLS LS->S2, paired events", "ols", "paired", "lss2"),
    ("rma_lss2_paired", "RMA LS->S2, paired events", "rma", "paired", "lss2"),
    ("qm_lss2_paired", "Quantile mapping LS->S2, paired events", "qm_bands", "paired", "lss2"),
    ("sensor_feature", "Raw HLS + sensor as feature", None, None, None),
]


def sensor_test(d, transform):
    """BA/AUC del clasificador de sensor (folds por campana) con la misma
    recalibracion; en todas las filas y en los eventos pareados de d."""
    out = {}
    for tag, data in [("all", d), ("paired", paired(d))]:
        data = data.reset_index(drop=True)
        res, _ = two_sample_test(data, FEATURES, "sensor", "S2", data.campaign.values,
                                 folds=splits(data, PRIMARY), transform=transform,
                                 n_boot=N_BOOT, seed=SEED)
        out.update({f"ba_{tag}": round(res["balanced_accuracy"], 3),
                    f"ba_{tag}_lo": round(res["ba_lo"], 3),
                    f"ba_{tag}_hi": round(res["ba_hi"], 3),
                    f"auc_{tag}": round(res["auc"], 3), f"n_{tag}_cls": res["n"]})
    return out


def main():
    banner("p15 -- RECALIBRACION ENTRE SENSORES Y RECUPERACION DE SECCHI (LOCO)")
    d = pd.read_csv(PROC / "analysis_dataset.csv", dtype={"campaign": str})
    d = d.reset_index(drop=True)
    d["is_s2"] = (d.sensor == "S2").astype(int)
    camp, s2 = d.campaign.values, (d.sensor == "S2").values
    print(f"  n={len(d)} (S2 {s2.sum()}, LS {(~s2).sum()}), campanas {d.campaign.nunique()}")
    npairs = {k: n_train_pairs(d, k) for k in PAIRS}
    for k, v in npairs.items():
        print(f"  pares de entrenamiento por fold ({k}): min {min(v)}, max {max(v)}")

    rows, preds = [], {}
    for name, label, kind, source, direction in VARIANTS:
        tf = None if kind is None else partial(recal, kind=kind, source=source,
                                               direction=direction)
        feats = FEATURES + ["is_s2"] if name == "sensor_feature" else FEATURES
        y, p = oof(d, feats, tf)
        preds[name] = p
        m = metrics(y, p)
        lo, hi = cluster_boot(y, p, camp)
        row = {"variant": name, "variant_label": label, "group": "pooled",
               "method": kind or ("sensor_feature" if name == "sensor_feature" else "none"),
               "pairs": source or "", "direction": direction or "",
               "n_train_pairs_min": min(npairs[source]) if source else np.nan,
               **m, "R2_lo": lo, "R2_hi": hi}
        for s, msk in [("S2", s2), ("LS", ~s2)]:
            ms = metrics(y[msk], p[msk])
            row[f"R2_{s}"], row[f"RMSE_{s}"], row[f"bias_{s}"] = ms["R2"], ms["RMSE"], ms["bias"]
            row[f"R2_{s}_lo"], row[f"R2_{s}_hi"] = cluster_boot(y[msk], p[msk], camp[msk])
        if name != "baseline":
            row["dR2"] = round(m["R2"] - rows[0]["R2"], 3)
            row["dR2_lo"], row["dR2_hi"] = cluster_boot(y, p, camp, p2=preds["baseline"])
            row["dRMSE"] = round(m["RMSE"] - rows[0]["RMSE"], 3)
            row["dRMSE_lo"], row["dRMSE_hi"] = cluster_boot(y, p, camp, "RMSE",
                                                            p2=preds["baseline"])
        # el clasificador ve las 12 features; con sensor como feature seria trivial,
        # asi que su BA es la de baseline (las features espectrales no cambian)
        row.update(sensor_test(d, tf) if name != "sensor_feature" else
                   {k: v for k, v in rows[0].items() if k.startswith(("ba_", "auc_", "n_all", "n_paired"))})
        rows.append(row)
        print(f"  {name:20s} R2={m['R2']:.3f} [{lo:.3f},{hi:.3f}] RMSE={m['RMSE']:.3f} "
              f"bias={m['bias']:+.3f}  R2 S2={row['R2_S2']:.3f} LS={row['R2_LS']:.3f}  "
              f"dR2={row.get('dR2', 0):+.3f} [{row.get('dR2_lo', 0):+.3f},"
              f"{row.get('dR2_hi', 0):+.3f}]  BA all={row['ba_all']:.3f} "
              f"paired={row['ba_paired']:.3f}")

    # referencia: un modelo por sensor en sus propias filas
    for s in ["S2", "LS"]:
        sub = d[d.sensor == s].reset_index(drop=True)
        y, p = oof(sub, FEATURES)
        m = metrics(y, p)
        lo, hi = cluster_boot(y, p, sub.campaign.values)
        pooled = preds["baseline"][(d.sensor == s).values]
        dlo, dhi = cluster_boot(y, p, sub.campaign.values, p2=pooled)
        rows.append({"variant": f"single_{s}", "variant_label": f"{s}-only model, own rows",
                     "group": "single_sensor", "method": "none", "pairs": "", "direction": "",
                     **m, "R2_lo": lo, "R2_hi": hi, f"R2_{s}": m["R2"], f"RMSE_{s}": m["RMSE"],
                     f"bias_{s}": m["bias"], f"R2_{s}_lo": lo, f"R2_{s}_hi": hi,
                     "dR2_vs_pooled_same_rows": round(m["R2"] - metrics(y, pooled)["R2"], 3),
                     "dR2_vs_pooled_lo": dlo, "dR2_vs_pooled_hi": dhi,
                     "n_campaigns": int(sub.campaign.nunique())})
        print(f"  single_{s:13s} R2={m['R2']:.3f} [{lo:.3f},{hi:.3f}] RMSE={m['RMSE']:.3f}  "
              f"pooled on same rows R2={metrics(y, pooled)['R2']:.3f}  "
              f"d=[{dlo:+.3f},{dhi:+.3f}]")

    out = pd.DataFrame(rows)
    out.to_csv(TIDY / "recalibrated_retrieval.csv", index=False)

    # --- interpretacion ---------------------------------------------------------------
    pool = out[(out.group == "pooled") & (out.variant != "baseline")]
    rpool = pool[pool.variant != "sensor_feature"]       # solo recalibraciones
    sig_up = rpool[rpool.dR2_lo > 0].variant.tolist()
    sig_down = rpool[rpool.dR2_hi < 0].variant.tolist()
    sf = out.set_index("variant").loc["sensor_feature"]
    rec = out[(out.group == "pooled") & (out.variant != "sensor_feature")]
    rho_all = float(rec.ba_all.rank().corr(rec.R2.rank()))
    rho_pair = float(rec.ba_paired.rank().corr(rec.R2.rank()))
    base = out.iloc[0]
    best = rpool.loc[rpool.R2.idxmax()]
    lowest_ba = rec.loc[rec.ba_paired.idxmin()]
    res = {
        "design": {"validation": "leave-one-campaign-out (splits(d, 'campaign'), 9 folds)",
                   "model": "p00 make_rf defaults, 12 features, RobustScaler in fold",
                   "recalibration_fit": "inside each outer training fold, on paired events "
                                        "(matchups, product hls) of the training campaigns "
                                        "only; applied to the corrected sensor's rows of train "
                                        "and test; indices recomputed (except qm_features)",
                   "pairs": {"paired": "same location and sampling date seen by both sensors "
                                       "within +-10 d (incl. rows without Secchi)",
                             "pm3d": "both sensors within +-3 d with a single image each"},
                   "n_train_pairs_per_fold": npairs,
                   "sensor_classifier": "harmonization_check.two_sample_test, RF 300 trees, "
                                        "folds by campaign, same in-fold recalibration; "
                                        "'all' = 815 rows (S2 absent 2013-2015, so campaign "
                                        "confounds), 'paired' = paired events of the dataset",
                   "ci": f"cluster bootstrap over campaigns, {N_BOOT} draws; dR2 paired "
                         "(same resamples for variant and baseline)",
                   "sensor_feature_ba": "not recomputed (spectral features unchanged; BA "
                                        "copied from baseline)"},
        "n": int(len(d)), "n_s2": int(s2.sum()), "n_ls": int((~s2).sum()),
        "variants": {r["variant"]: {k: v for k, v in r.items()
                                    if k not in ("variant",) and not (isinstance(v, float) and np.isnan(v))}
                     for r in out.to_dict("records")},
        "interpretation": {
            "baseline_R2": float(base.R2), "baseline_R2_ci": [float(base.R2_lo), float(base.R2_hi)],
            "baseline_reproduces_p04": bool(abs(base.R2 - 0.486) < 0.0015),
            "any_recalibration_significantly_improves_R2": bool(len(sig_up) > 0),
            "variants_significantly_better": sig_up,
            "variants_significantly_worse": sig_down,
            "best_variant": str(best.variant), "best_R2": float(best.R2),
            "best_dR2": float(best.dR2), "best_dR2_ci": [float(best.dR2_lo), float(best.dR2_hi)],
            "max_abs_dR2_recalibrations": float(rpool.dR2.abs().max()),
            "sensor_feature_dR2": float(sf.dR2),
            "sensor_feature_dR2_ci": [float(sf.dR2_lo), float(sf.dR2_hi)],
            "sensor_feature_significant": bool(sf.dR2_lo > 0 or sf.dR2_hi < 0),
            "lowest_separability_variant_paired": str(lowest_ba.variant),
            "lowest_ba_paired": float(lowest_ba.ba_paired),
            "lowest_ba_paired_dR2": (float(lowest_ba.dR2) if lowest_ba.variant != "baseline"
                                     else 0.0),
            "spearman_ba_all_vs_R2": round(rho_all, 3),
            "spearman_note": "positive rho = variants with LOWER sensor separability have "
                             "LOWER retrieval R2 (rank correlation over baseline + 10 "
                             "recalibrations)",
            "spearman_ba_paired_vs_R2": round(rho_pair, 3),
            "reduced_separability_translates_into_retrieval_gain": bool(
                len(sig_up) > 0 and lowest_ba.variant in sig_up),
        },
    }
    write_json(res, MET / "recalibrated_retrieval.json")

    banner("RESUMEN")
    cols = ["variant", "R2", "R2_lo", "R2_hi", "RMSE", "bias", "R2_S2", "R2_LS", "dR2",
            "dR2_lo", "dR2_hi", "ba_all", "ba_paired"]
    print(out[cols].to_string(index=False))
    print(res["interpretation"])
    banner("SALIDAS")
    print(f"  {TIDY / 'recalibrated_retrieval.csv'}\n  {MET / 'recalibrated_retrieval.json'}")


if __name__ == "__main__":
    main()
