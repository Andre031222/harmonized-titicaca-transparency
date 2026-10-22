"""
p07_uncertainty.py -- Incertidumbre por prediccion (prediccion conformal)
bajo el diseno primario: dejar fuera una campana entera.

Para cada campana c que se deja fuera, el entrenamiento T son las otras 8.
La calibracion es de tipo CV+ / cross-conformal: un leave-one-campaign-out
INTERNO sobre T da un residuo fuera de fold para cada fila de T, producido
por un modelo que no vio su campana. La prediccion puntual para c sale del
RF ajustado con todo T. Asi la calibracion respeta la misma estructura de
dependencia (campanas completas) que la evaluacion.

Variantes (cuantil con correccion de muestra finita ceil((n+1)(1-a))/n):
  split     ancho constante: cuantil de |residuos internos|
  mondrian  el cuantil se calcula por tramo del valor PREDICHO; un tramo con
            menos de MIN_BIN residuos se fusiona con su vecino (se registra)
  cqr       regresion cuantilica conformalizada (Romano et al. 2019) con
            gradient boosting de cuantiles a/2 y 1-a/2; puntuacion
            max(lo - y, y - hi) fuera de fold interno

Niveles 50, 80, 90 y 95 %. Cobertura marginal con IC por bootstrap de
campanas; cobertura al 90 % por tramo de Secchi medido y predicho, por
campana y por zona; sesgo de la prediccion puntual en los extremos.

Entrada: data/processed/analysis_dataset.csv
Salidas: results/tidy/conformal_intervals.csv
         results/tidy/conformal_coverage.csv
         results/tidy/interval_width_by_secchi.csv
         results/metrics/uncertainty.json
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from p00_config import (write_json, FEATURES, MET, N_BOOT, PRIMARY, PROC, SECCHI_CLIP_LO,
                        SEED, TIDY, ZONE_LABELS, banner, cluster_boot, make_rf,
                        metrics, splits)

LEVELS = [0.50, 0.80, 0.90, 0.95]
METHODS = ["split", "mondrian", "cqr"]
METHOD_LABELS = {"split": "Split (constant width)",
                 "mondrian": "Mondrian (by predicted Secchi)",
                 "cqr": "CQR (quantile GBM)"}
BINS = [-np.inf, 5, 7.5, 10, 12.5, np.inf]
BIN_LABELS = ["<5", "5-7.5", "7.5-10", "10-12.5", ">12.5"]
MIN_BIN = 30                 # residuos minimos por tramo Mondrian
INNER_TREES = 300            # RF de los ajustes internos (9 x 8)
FOCUS = 0.90


def load():
    d = pd.read_csv(PROC / "analysis_dataset.csv", dtype={"campaign": str})
    return d.reset_index(drop=True)


def q_finite(scores, alpha):
    """Cuantil conformal con correccion de muestra finita."""
    s = np.sort(np.asarray(scores))
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(s[min(k, n) - 1]) if k <= n else float("inf")


def qgbr(aq):
    return GradientBoostingRegressor(loss="quantile", alpha=aq, n_estimators=300,
                                     max_depth=3, learning_rate=0.05,
                                     subsample=1.0, random_state=SEED)


def bin_of(v):
    return np.digitize(v, BINS[1:-1])          # 0..4


def merge_bins(cal_bins, n_bins=len(BIN_LABELS)):
    """Agrupa tramos adyacentes hasta que todos tengan >= MIN_BIN residuos.
    Devuelve un mapa tramo original -> grupo y la lista de fusiones."""
    groups = [[b] for b in range(n_bins)]
    log = []
    cnt = lambda g: int(np.isin(cal_bins, g).sum())
    while len(groups) > 1:
        small = [i for i, g in enumerate(groups) if cnt(g) < MIN_BIN]
        if not small:
            break
        i = min(small, key=lambda k: cnt(groups[k]))
        # vecino con menos residuos (o el unico que exista)
        nb = [j for j in (i - 1, i + 1) if 0 <= j < len(groups)]
        j = min(nb, key=lambda k: cnt(groups[k]))
        log.append(f"{'+'.join(BIN_LABELS[b] for b in groups[i])} (n={cnt(groups[i])})"
                   f" -> {'+'.join(BIN_LABELS[b] for b in groups[j])}")
        a, b = sorted((i, j))
        groups[a] = groups[a] + groups[b]
        del groups[b]
    m = {b: gi for gi, g in enumerate(groups) for b in g}
    return m, groups, log


def boot_mean(x, groups, n_boot=N_BOOT, seed=SEED):
    """IC 95 % de una media por bootstrap de conglomerados (campanas)."""
    x, groups = np.asarray(x, float), np.asarray(groups)
    ug = np.unique(groups)
    pos = {g: np.flatnonzero(groups == g) for g in ug}
    rng = np.random.default_rng(seed)
    v = [x[np.concatenate([pos[g] for g in rng.choice(ug, len(ug))])].mean()
         for _ in range(n_boot)]
    lo, hi = np.quantile(v, [0.025, 0.975])
    return round(float(lo), 4), round(float(hi), 4)


# ---------------------------------------------------------------------------
def run(d):
    X, y = d[FEATURES].values, d.secchi.values
    n = len(y)
    pred = np.full(n, np.nan)
    lo = {(m, l): np.full(n, np.nan) for m in METHODS for l in LEVELS}
    hi = {(m, l): np.full(n, np.nan) for m in METHODS for l in LEVELS}
    merges, n_cal = [], []
    outer = splits(d, PRIMARY)
    for k, (tr, te) in enumerate(outer):
        t0 = time.time()
        camp = d.campaign.values[te[0]]
        dT = d.iloc[tr].reset_index(drop=True)
        XT, yT = X[tr], y[tr]
        inner = splits(dT, PRIMARY)
        # modelo puntual con todo T
        rf = make_rf().fit(XT, yT)
        p_te = np.clip(rf.predict(X[te]), SECCHI_CLIP_LO, None)
        pred[te] = p_te
        # residuos internos fuera de fold (campanas completas)
        p_in = np.full(len(yT), np.nan)
        for itr, ite in inner:
            m = make_rf({"n_estimators": INNER_TREES}).fit(XT[itr], yT[itr])
            p_in[ite] = np.clip(m.predict(XT[ite]), SECCHI_CLIP_LO, None)
        res = np.abs(yT - p_in)
        n_cal.append(len(res))
        cal_b, te_b = bin_of(p_in), bin_of(p_te)
        bmap, bgroups, blog = merge_bins(cal_b)
        merges.append({"held_out": camp, "n_cal": int(len(res)),
                       "cal_counts": {BIN_LABELS[b]: int((cal_b == b).sum())
                                      for b in range(len(BIN_LABELS))},
                       "groups": ["+".join(BIN_LABELS[b] for b in g) for g in bgroups],
                       "merges": blog})
        cal_g = np.array([bmap[b] for b in cal_b])
        te_g = np.array([bmap[b] for b in te_b])
        for lev in LEVELS:
            a = 1 - lev
            q = q_finite(res, a)
            lo[("split", lev)][te], hi[("split", lev)][te] = p_te - q, p_te + q
            for g in np.unique(te_g):
                qg = q_finite(res[cal_g == g], a)
                s = te[te_g == g]
                lo[("mondrian", lev)][s] = pred[s] - qg
                hi[("mondrian", lev)][s] = pred[s] + qg
            # CQR: cuantiles con todo T, puntuaciones con LOCO interno
            sc = np.full(len(yT), np.nan)
            for itr, ite in inner:
                ql = qgbr(a / 2).fit(XT[itr], yT[itr])
                qh = qgbr(1 - a / 2).fit(XT[itr], yT[itr])
                sc[ite] = np.maximum(ql.predict(XT[ite]) - yT[ite],
                                     yT[ite] - qh.predict(XT[ite]))
            qc = q_finite(sc, a)
            ql = qgbr(a / 2).fit(XT, yT)
            qh = qgbr(1 - a / 2).fit(XT, yT)
            lq, hq = ql.predict(X[te]), qh.predict(X[te])
            lo[("cqr", lev)][te] = np.minimum(lq, hq) - qc
            hi[("cqr", lev)][te] = np.maximum(lq, hq) + qc
        print(f"  campana {camp}: test n={len(te):3d} | cal n={len(res):3d} | "
              f"tramos Mondrian {merges[-1]['groups']} | {time.time() - t0:5.1f} s")
    for kk in lo:
        lo[kk] = np.clip(lo[kk], SECCHI_CLIP_LO, None)
    return pred, lo, hi, merges, n_cal


# ---------------------------------------------------------------------------
def main():
    banner("p07 -- PREDICCION CONFORMAL (dejar fuera una campana; calibracion CV+ por campanas)")
    d = load()
    print(f"  n={len(d)} | {d.campaign.nunique()} campanas | RF interno con "
          f"{INNER_TREES} arboles\n")
    pred, lo, hi, merges, n_cal = run(d)
    y, g = d.secchi.values, d.campaign.values

    # intervalos por observacion (todas las variantes y niveles)
    base = d[["location", "zona", "campaign", "sampling_date", "sensor",
              "secchi"]].copy()
    base["zone_label"] = base.zona.map(ZONE_LABELS)
    base["predicted"] = pred
    base["bin_measured"] = pd.Categorical(np.array(BIN_LABELS)[bin_of(y)],
                                          BIN_LABELS, ordered=True)
    base["bin_predicted"] = pd.Categorical(np.array(BIN_LABELS)[bin_of(pred)],
                                           BIN_LABELS, ordered=True)
    ivs = []
    for m in METHODS:
        for lev in LEVELS:
            t = base.copy()
            t["method"], t["method_label"], t["nominal"] = m, METHOD_LABELS[m], lev
            t["lower"], t["upper"] = lo[(m, lev)], hi[(m, lev)]
            ivs.append(t)
    iv = pd.concat(ivs, ignore_index=True)
    iv["width"] = iv.upper - iv.lower
    iv["covered"] = ((iv.secchi >= iv.lower) & (iv.secchi <= iv.upper)).astype(int)

    # cobertura en formato largo
    rows = []

    def add(sub, m, lev, cond, level_name):
        c = sub.covered.values
        lo_, hi_ = (boot_mean(c, sub.campaign.values)
                    if sub.campaign.nunique() > 1 else (np.nan, np.nan))
        rows.append({"method": m, "method_label": METHOD_LABELS[m],
                     "nominal": lev, "conditioned_on": cond, "group": level_name,
                     "n": int(len(sub)), "coverage": round(float(c.mean()), 4),
                     "coverage_lo": lo_, "coverage_hi": hi_,
                     "mean_width": round(float(sub.width.mean()), 3),
                     "median_width": round(float(sub.width.median()), 3)})

    for (m, lev), sub in iv.groupby(["method", "nominal"], sort=False):
        add(sub, m, lev, "marginal", "all")
        if abs(lev - FOCUS) > 1e-9:
            continue
        for col, cond in [("bin_measured", "measured_secchi"),
                          ("bin_predicted", "predicted_secchi")]:
            for b, s in sub.groupby(col, observed=True):
                add(s, m, lev, cond, str(b))
        for c, s in sub.groupby("campaign"):
            add(s, m, lev, "campaign", c)
        for z, s in sub.groupby("zone_label"):
            add(s, m, lev, "zone", z)
    cov = pd.DataFrame(rows)

    # ancho y cobertura por tramo (90 %), medido y predicho
    wb = []
    for col, cond in [("bin_measured", "measured"), ("bin_predicted", "predicted")]:
        t = (iv[iv.nominal == FOCUS].groupby(["method", col], observed=True)
             .agg(n=("secchi", "size"), mean_width=("width", "mean"),
                  median_width=("width", "median"), coverage=("covered", "mean"),
                  mean_bias=("predicted", "mean"))
             .reset_index().rename(columns={col: "bin"}))
        mean_y = (iv[iv.nominal == FOCUS].groupby(["method", col], observed=True)
                  .secchi.mean().values)
        t["mean_bias"] = t.mean_bias - mean_y
        t["conditioned_on"] = cond
        t["method_label"] = t.method.map(METHOD_LABELS)
        wb.append(t)
    wb = pd.concat(wb, ignore_index=True)

    # ---- consola
    print()
    marg = cov[cov.conditioned_on == "marginal"]
    print(f"  {'metodo':9s} {'nominal':>7s} {'cobertura [IC95 campanas]':>28s} "
          f"{'ancho medio':>11s} {'mediana':>8s}")
    for _, r in marg.iterrows():
        print(f"  {r.method:9s} {r.nominal * 100:6.0f}% {r.coverage * 100:7.1f}% "
              f"[{r.coverage_lo * 100:5.1f}, {r.coverage_hi * 100:5.1f}]"
              f"{'':6s} {r.mean_width:8.2f} m {r.median_width:7.2f} m")
    for cond in ["measured_secchi", "predicted_secchi", "zone", "campaign"]:
        t = cov[cov.conditioned_on == cond].pivot(index="group", columns="method",
                                                  values="coverage")
        if cond.endswith("secchi"):
            t = t.reindex([b for b in BIN_LABELS if b in t.index])
        print(f"\n  cobertura al 90 % por {cond}:")
        print((t[METHODS] * 100).round(1).to_string())
    pm = metrics(y, pred)
    bias_meas = (pd.DataFrame({"b": base.bin_measured, "e": pred - y})
                 .groupby("b", observed=True).e.agg(["mean", "size"]))
    print("\n  sesgo medio de la prediccion puntual por tramo de Secchi MEDIDO (m):")
    print(bias_meas.round(2).to_string())

    # ---- JSON
    def cond_dict(cond):
        t = cov[cov.conditioned_on == cond]
        return {m: {r.group: {"n": r.n, "coverage": r.coverage,
                              "ci": [r.coverage_lo, r.coverage_hi],
                              "mean_width": r.mean_width}
                    for r in t[t.method == m].itertuples()} for m in METHODS}

    ext = {}
    for b in ["<5", ">12.5"]:
        s = base.bin_measured == b
        ext[b] = {"n": int(s.sum()),
                  "mean_measured": round(float(y[s].mean()), 3),
                  "mean_predicted": round(float(pred[s].mean()), 3),
                  "mean_bias_pred_minus_obs": round(float((pred[s] - y[s]).mean()), 3),
                  "coverage_90": {m: float(cov[(cov.conditioned_on == "measured_secchi")
                                               & (cov.method == m) & (cov.group == b)]
                                           .coverage.iloc[0]) for m in METHODS}}
    worst = {m: float(cov[(cov.conditioned_on == "measured_secchi")
                          & (cov.method == m)].coverage.min()) for m in METHODS}
    fails = {m: bool(worst[m] < FOCUS - 0.10) for m in METHODS}
    out = {
        "design": ("Leave-one-campaign-out (9 folds). Point prediction: RF fitted on "
                   "the 8 training campaigns. Calibration: CV+/cross-conformal-style "
                   "inner leave-one-campaign-out on the training campaigns, so every "
                   "calibration residual comes from a model that did not see its "
                   "campaign. Finite-sample quantile ceil((n+1)(1-a))/n."),
        "point_model": "make_rf() (500 trees) for the outer fit; inner fits use "
                       f"n_estimators={INNER_TREES} for speed; no feature scaling "
                       "(tree models are invariant to the per-feature affine RobustScaler).",
        "cqr_model": "GradientBoostingRegressor(loss=quantile, 300 trees, depth 3, lr 0.05)",
        "n": int(len(y)), "n_campaigns": int(d.campaign.nunique()),
        "calibration_n_per_fold": {"min": int(min(n_cal)), "max": int(max(n_cal))},
        "point_prediction_metrics": pm,
        "point_prediction_R2_ci": list(cluster_boot(y, pred, g)),
        "marginal": {m: {f"{int(r.nominal * 100)}": {
            "coverage": r.coverage, "ci": [r.coverage_lo, r.coverage_hi],
            "calibration_error": round(r.coverage - r.nominal, 4),
            "mean_width_m": r.mean_width, "median_width_m": r.median_width}
            for r in marg[marg.method == m].itertuples()} for m in METHODS},
        "coverage90_by_measured_bin": cond_dict("measured_secchi"),
        "coverage90_by_predicted_bin": cond_dict("predicted_secchi"),
        "coverage90_by_zone": cond_dict("zone"),
        "coverage90_by_campaign": cond_dict("campaign"),
        "coverage90_campaign_range": {m: [float(cov[(cov.conditioned_on == "campaign")
                                                    & (cov.method == m)].coverage.min()),
                                          float(cov[(cov.conditioned_on == "campaign")
                                                    & (cov.method == m)].coverage.max())]
                                      for m in METHODS},
        "mondrian_bins": {"bins_m": BIN_LABELS, "min_residuals_per_bin": MIN_BIN,
                          "per_fold": merges},
        "measured_extremes": ext,
        "conditional_on_measured_fails": {
            "worst_bin_coverage_90": worst, "fails_by_more_than_10pts": fails,
            "why": (
                "Conformal guarantees are marginal (and, for Mondrian, conditional on "
                "the PREDICTED value). The point prediction is biased in the measured "
                f"extremes (mean pred-obs {ext['<5']['mean_bias_pred_minus_obs']:+.2f} m "
                f"in <5 m, {ext['>12.5']['mean_bias_pred_minus_obs']:+.2f} m in >12.5 m), "
                "i.e. it compresses them toward the mean, so residuals there exceed the "
                "calibration quantile and coverage conditional on the MEASURED value drops.")},
    }
    write_json(out, MET / "uncertainty.json")
    iv.to_csv(TIDY / "conformal_intervals.csv", index=False, float_format="%.4f")
    cov.to_csv(TIDY / "conformal_coverage.csv", index=False)
    wb.to_csv(TIDY / "interval_width_by_secchi.csv", index=False, float_format="%.4f")

    banner("SALIDAS")
    for f in ["conformal_intervals.csv", "conformal_coverage.csv",
              "interval_width_by_secchi.csv"]:
        print(f"  {TIDY / f}")
    print(f"  {MET / 'uncertainty.json'}")


if __name__ == "__main__":
    main()
