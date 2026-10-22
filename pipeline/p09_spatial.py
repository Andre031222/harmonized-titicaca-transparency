"""
p09_spatial.py -- estructura espacial del error de recuperacion.

Unidad: la UBICACION (212; residuo medio por ubicacion), no el codigo de
estacion, que IMARPE reasigno entre anios. Residuo = predicho - observado, con
las predicciones fuera de fold del diseno primario (una campana fuera).

  (1) I de Moran global, kNN estandarizado por fila (k = 4..10, primario 6),
      permutacion; tambien sobre el residuo sin la media de su zona
  (2) correlograma de Moran por anillos de distancia, con envolvente
  (3) semivariograma empirico + modelo esferico/exponencial (pepita, meseta,
      alcance)
  (4) LISA con permutacion condicional + FDR de Benjamini-Hochberg
  (5) Gi* de Getis-Ord + FDR
  (6) distancia a la orilla (poligono del lago, UTM): Spearman global, en la
      Bahia de Puno, dentro de cada zona y regresion residuo ~ zona + log(dist)

Solo numpy/scipy (sin PySAL).

Entradas : data/processed/analysis_dataset.csv, results/tidy/location_units.csv,
           data/lake_boundary/titicaca.gpkg
Salidas  : results/tidy/moran_global.csv, moran_correlogram.csv,
           variogram_empirical.csv, variogram_model.csv, lisa_locations.csv,
           shore_distance.csv
           results/metrics/spatial.json
"""
import json

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

from p00_config import (write_json, DATA, FEATURES, MET, PROC, SEED, TIDY, UTM, ZONE_LABELS,
                        ZONES, banner, cv_oof, metrics, to_utm)

K_PRIMARY = 6
K_SENS = [4, 5, 6, 7, 8, 10]
N_PERM = 9999
ALPHA = 0.05
RINGS_KM = [0, 5, 10, 20, 40, 80, 160]
VARIO_EDGES_KM = [0, 2.5, 5, 7.5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100]


# --- datos -------------------------------------------------------------------
def location_table():
    d = pd.read_csv(PROC / "analysis_dataset.csv", dtype={"campaign": str})
    d = d.reset_index(drop=True)
    y, p = cv_oof(d, FEATURES, "campaign")
    d["pred"], d["residual"] = p, p - y
    m = metrics(y, p)
    lu = pd.read_csv(TIDY / "location_units.csv")[["location", "lat", "lon", "zona"]]
    loc = (d.groupby("location")
           .agg(n=("residual", "size"), n_campaigns=("campaign", "nunique"),
                residual=("residual", "mean"), secchi_mean=("secchi", "mean"))
           .reset_index().merge(lu, on="location", how="left"))
    loc["zone_label"] = loc.zona.map(ZONE_LABELS)
    x, yy = to_utm(loc.lat.values, loc.lon.values)
    loc["x_km"], loc["y_km"] = x / 1000, yy / 1000
    loc["residual_zone_demeaned"] = loc.residual - loc.groupby("zona").residual.transform("mean")
    # diagnostico: el RF regresa a la media (subestima agua clara), y la claridad
    # tiene su propio patron espacial. Residuo sin la parte lineal en Secchi medio.
    X = np.column_stack([np.ones(len(loc)), loc.secchi_mean.values])
    b = np.linalg.lstsq(X, loc.residual.values, rcond=None)[0]
    loc["residual_clarity_adjusted"] = loc.residual - X @ b
    return d, loc, m


def station_table(d):
    """Unidad OBSOLETA (codigo de estacion), solo para comparar con el analisis
    anterior: promedia residuos de lecturas que pueden distar > 80 km."""
    st = d.groupby("station").agg(lat=("lat", "mean"), lon=("lon", "mean"),
                                  residual=("residual", "mean")).reset_index()
    x, y = to_utm(st.lat.values, st.lon.values)
    st["x_km"], st["y_km"] = x / 1000, y / 1000
    return st


def dist_matrix(t):
    p = t[["x_km", "y_km"]].values
    return np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1))


# --- pesos y Moran -------------------------------------------------------------
def knn_binary(D, k):
    n = len(D)
    W = np.zeros((n, n))
    nb = np.argsort(D + np.eye(n) * 1e12, axis=1)[:, :k]
    W[np.repeat(np.arange(n), k), nb.ravel()] = 1.0
    return W


def row_std(W):
    s = W.sum(1, keepdims=True)
    return np.divide(W, s, out=np.zeros_like(W), where=s > 0)


def moran_many(Z, W):
    """I de Moran para cada fila de Z (vectorizado)."""
    Z = Z - Z.mean(1, keepdims=True)
    n = Z.shape[1]
    return n / W.sum() * np.einsum("ij,ij->i", Z @ W, Z) / np.einsum("ij,ij->i", Z, Z)


def moran_test(y, W, rng, n_perm=N_PERM):
    obs = float(moran_many(y[None, :], W)[0])
    sims = np.concatenate([moran_many(np.array([rng.permutation(y) for _ in range(min(2000, n_perm - i))]), W)
                           for i in range(0, n_perm, 2000)])
    ei = -1.0 / (len(y) - 1)
    # bilateral respecto de E[I]; y unilateral (autocorrelacion positiva)
    p2 = (1 + np.sum(np.abs(sims - ei) >= abs(obs - ei))) / (n_perm + 1)
    p1 = (1 + np.sum(sims >= obs)) / (n_perm + 1)
    zs = (obs - sims.mean()) / sims.std()
    lo, hi = np.quantile(sims, [0.025, 0.975])
    return {"I": obs, "E_I": ei, "z_perm": float(zs), "p_two_sided": float(p2),
            "p_one_sided_pos": float(p1), "env_lo": float(lo), "env_hi": float(hi)}


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(q, 1)
    return out


# --- semivariograma ------------------------------------------------------------
def semivariogram(z, D, edges):
    iu = np.triu_indices(len(z), 1)
    h, g = D[iu], 0.5 * (z[iu[0]] - z[iu[1]]) ** 2
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (h > lo) & (h <= hi)
        if m.sum():
            rows.append({"lo_km": lo, "hi_km": hi, "h_km": float(h[m].mean()),
                         "gamma": float(g[m].mean()), "pairs": int(m.sum())})
    return pd.DataFrame(rows)


def sph(h, c0, c, a):
    r = np.clip(h / a, 0, 1)
    return c0 + c * (1.5 * r - 0.5 * r ** 3)


def expo(h, c0, c, a):
    # a = alcance practico (95 % de la meseta)
    return c0 + c * (1 - np.exp(-3 * h / a))


def fit_variogram(ev, var):
    out = {}
    for name, f in [("spherical", sph), ("exponential", expo)]:
        try:
            par, _ = curve_fit(f, ev.h_km, ev.gamma, p0=[var * 0.5, var * 0.5, 20],
                               sigma=1 / np.sqrt(ev.pairs), absolute_sigma=False,
                               bounds=([0, 0, 0.5], [var * 5, var * 5, 300]),
                               maxfev=20000)
        except RuntimeError:
            continue
        wsse = float(np.sum(ev.pairs * (ev.gamma - f(ev.h_km, *par)) ** 2))
        c0, c, a = (float(v) for v in par)
        out[name] = {"nugget": c0, "partial_sill": c, "sill": c0 + c,
                     "range_km": a, "nugget_sill_ratio": c0 / (c0 + c) if c0 + c > 0 else np.nan,
                     "wsse": wsse}
    best = min(out, key=lambda k: out[k]["wsse"])
    return out, best


# --- LISA y Gi* ----------------------------------------------------------------
def lisa(y, W, rng, n_perm=N_PERM):
    """Moran local con permutacion condicional (pseudo-p unilateral en la
    direccion observada, convencion de GeoDa). W estandarizada por fila."""
    z = (y - y.mean()) / y.std()
    lag = W @ z
    Ii = z * lag
    n = len(z)
    p = np.empty(n)
    for i in range(n):
        nb = np.nonzero(W[i])[0]
        others = np.delete(z, i)
        idx = rng.permuted(np.tile(np.arange(n - 1), (n_perm, 1)), axis=1)[:, :len(nb)]
        sim = z[i] * (others[idx] * W[i, nb]).sum(1)
        ext = np.sum(sim >= Ii[i]) if Ii[i] >= 0 else np.sum(sim <= Ii[i])
        p[i] = (1 + ext) / (n_perm + 1)
    quad = np.where(z > 0, np.where(lag > 0, "HH", "HL"), np.where(lag > 0, "LH", "LL"))
    return Ii, z, lag, p, quad


def gi_star(y, Wb):
    """Gi* de Getis-Ord (1995) con pesos binarios que incluyen al propio sitio.
    Devuelve z (Gi* estandarizado) y P bilateral normal."""
    W = Wb.copy()
    np.fill_diagonal(W, 1.0)
    n = len(y)
    xbar, s = y.mean(), np.sqrt((y ** 2).mean() - y.mean() ** 2)
    sw, sw2 = W.sum(1), (W ** 2).sum(1)
    num = W @ y - xbar * sw
    den = s * np.sqrt((n * sw2 - sw ** 2) / (n - 1))
    z = num / den
    return z, 2 * stats.norm.sf(np.abs(z))


# --- orilla --------------------------------------------------------------------
def shore_distance(loc):
    lake = gpd.read_file(DATA / "lake_boundary" / "titicaca.gpkg").to_crs(UTM)
    poly = lake.union_all()
    pts = gpd.GeoSeries(gpd.points_from_xy(loc.x_km * 1000, loc.y_km * 1000), crs=UTM)
    dist = pts.distance(poly.boundary).values / 1000
    inside = pts.within(poly).values
    # contorno de HydroLAKES; la unica ubicacion que cae fuera esta en agua
    # pegada a la orilla real -> distancia 0
    return np.where(inside, dist, 0.0), inside


def ols(y, X, names):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    dof = len(y) - X.shape[1]
    s2 = res @ res / dof
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), dof)
    return {nm: {"coef": float(b), "se": float(e), "p": float(pp)}
            for nm, b, e, pp in zip(names, beta, se, p)}


def shore_analysis(loc):
    rows, out = [], {}
    sets = ([("Lake-wide", loc), ("Lake-wide, inside polygon", loc[loc.inside_polygon])] +
            [(ZONE_LABELS[z], loc[loc.zona == z]) for z in ZONES])
    for name, sub in sets:
        r = {"subset": name, "n": int(len(sub)),
             "shore_km_median": float(sub.shore_km.median())}
        for lab, col in [("signed", sub.residual), ("abs", sub.residual.abs()),
                         ("zone_demeaned", sub.residual_zone_demeaned)]:
            rho, p = stats.spearmanr(sub.shore_km, col)
            r[f"rho_{lab}"], r[f"p_{lab}"] = float(rho), float(p)
        rows.append(r)
    tab = pd.DataFrame(rows)
    # regresion parcial: residuo ~ zona + log(distancia)
    X = np.column_stack([np.ones(len(loc))] +
                        [(loc.zona == z).astype(float).values for z in ZONES[1:]] +
                        [np.log(loc.shore_km.clip(lower=0.05)).values])
    names = ["intercept"] + [f"zone_{ZONE_LABELS[z]}" for z in ZONES[1:]] + ["log_shore_km"]
    out["ols_signed"] = ols(loc.residual.values, X, names)
    out["ols_abs"] = ols(loc.residual.abs().values, X, names)
    # control adicional por la claridad media observada (regresion a la media)
    Xc = np.column_stack([X, loc.secchi_mean.values])
    out["ols_signed_clarity"] = ols(loc.residual.values, Xc, names + ["secchi_mean"])
    out["ols_abs_clarity"] = ols(loc.residual.abs().values, Xc, names + ["secchi_mean"])
    Xz = X[:, :-1]
    out["ols_zone_only_r2"] = float(1 - np.sum((loc.residual - Xz @ np.linalg.lstsq(Xz, loc.residual, rcond=None)[0]) ** 2)
                                    / np.sum((loc.residual - loc.residual.mean()) ** 2))
    return tab, out


# --- principal -----------------------------------------------------------------
def r(v, k=4):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), k)


def main():
    banner("p09  ESTRUCTURA ESPACIAL DEL ERROR (ubicaciones, diseno primario)")
    rng = np.random.default_rng(SEED)
    d, loc, m = location_table()
    print(f"  OOF campana-fuera: R2={m['R2']:.3f} RMSE={m['RMSE']:.3f} bias={m['bias']:+.3f} n={m['n']}")
    D = dist_matrix(loc)
    nn = np.sort(D, 1)[:, 1]
    print(f"  ubicaciones: {len(loc)}  | vecino mas cercano, mediana {np.median(nn):.2f} km")
    zone_mean = loc.groupby("zone_label").residual.agg(["mean", "size"]).round(3)
    print(zone_mean.to_string())

    # (1) Moran global
    banner("(1) MORAN GLOBAL", "-")
    grows = []
    for var in ["residual", "residual_zone_demeaned", "residual_clarity_adjusted"]:
        y = loc[var].values
        for k in K_SENS:
            t = moran_test(y, row_std(knn_binary(D, k)), rng)
            grows.append({"unit": "location", "variable": var, "k": k, "n": len(y), **t})
            print(f"  {var:24s} k={k:2d}: I={t['I']:+.3f} z={t['z_perm']:+.2f} "
                  f"p2={t['p_two_sided']:.4f}")
    st = station_table(d)
    Ds = dist_matrix(st)
    t = moran_test(st.residual.values, row_std(knn_binary(Ds, K_PRIMARY)), rng)
    grows.append({"unit": "station_id_obsolete", "variable": "residual", "k": K_PRIMARY,
                  "n": len(st), **t})
    print(f"  [obsoleto] codigo de estacion (n={len(st)}) k=6: I={t['I']:+.3f} "
          f"p2={t['p_two_sided']:.4f}")
    glob = pd.DataFrame(grows)

    # (2) correlograma
    banner("(2) CORRELOGRAMA", "-")
    crow = []
    for var in ["residual", "residual_zone_demeaned"]:
        y = loc[var].values
        for lo, hi in zip(RINGS_KM[:-1], RINGS_KM[1:]):
            W = ((D > lo) & (D <= hi)).astype(float)
            np.fill_diagonal(W, 0)
            if W.sum() == 0:
                continue
            t = moran_test(y, W, rng, n_perm=4999)
            crow.append({"variable": var, "lo_km": lo, "hi_km": hi,
                         "mid_km": (lo + hi) / 2, "pairs": int(W.sum() / 2),
                         "n_with_neighbours": int((W.sum(1) > 0).sum()), **t})
            print(f"  {var:24s} {lo:3d}-{hi:3d} km: I={t['I']:+.3f} "
                  f"[{t['env_lo']:+.3f},{t['env_hi']:+.3f}] p={t['p_two_sided']:.4f} "
                  f"pares={int(W.sum() / 2)}")
    corr = pd.DataFrame(crow)

    # (3) semivariograma
    banner("(3) SEMIVARIOGRAMA", "-")
    evs, vms, vario = [], [], {}
    for var in ["residual", "residual_zone_demeaned"]:
        z = loc[var].values
        ev = semivariogram(z, D, VARIO_EDGES_KM)
        fits, best = fit_variogram(ev, float(z.var()))
        vario[var] = {"sample_variance": float(z.var()), "best_model": best,
                      "models": fits}
        ev["variable"] = var
        evs.append(ev)
        h = np.linspace(0, VARIO_EDGES_KM[-1], 201)
        for name, pars in fits.items():
            f = sph if name == "spherical" else expo
            vms.append(pd.DataFrame({"variable": var, "model": name, "h_km": h,
                                     "gamma": f(h, pars["nugget"], pars["partial_sill"],
                                                pars["range_km"]),
                                     "best": name == best}))
            print(f"  {var:24s} {name:11s} pepita={pars['nugget']:.3f} "
                  f"meseta={pars['sill']:.3f} alcance={pars['range_km']:.1f} km "
                  f"pepita/meseta={pars['nugget_sill_ratio']:.2f}"
                  f"{'  <- mejor' if name == best else ''}")
        print(f"     varianza muestral {z.var():.3f}")
    ev_tab, vm_tab = pd.concat(evs), pd.concat(vms)

    # (4) LISA, (5) Gi*
    banner("(4-5) LISA Y Gi*", "-")
    Wb = knn_binary(D, K_PRIMARY)
    Ii, zz, lag, p_l, quad = lisa(loc.residual.values, row_std(Wb), rng)
    loc["local_I"], loc["z"], loc["lag"], loc["p_lisa"] = Ii, zz, lag, p_l
    loc["q_lisa"] = bh(p_l)
    loc["lisa_quadrant"] = quad
    loc["lisa_class"] = np.where(loc.q_lisa < ALPHA, quad, "ns")
    loc["lisa_class_unadjusted"] = np.where(loc.p_lisa < ALPHA, quad, "ns")
    gz, gp = gi_star(loc.residual.values, Wb)
    loc["gi_z"], loc["gi_p"], loc["gi_q"] = gz, gp, bh(gp)
    loc["gi_class"] = np.where(loc.gi_q < ALPHA, np.where(gz > 0, "hot", "cold"), "ns")
    # LISA sobre el residuo sin media de zona
    Ii2, _, _, p2, quad2 = lisa(loc.residual_zone_demeaned.values, row_std(Wb), rng)
    loc["lisa_class_zone_demeaned"] = np.where(bh(p2) < ALPHA, quad2, "ns")
    lc = loc.lisa_class.value_counts().to_dict()
    print(f"  LISA FDR q<{ALPHA}: {lc} | sin ajustar: "
          f"{loc.lisa_class_unadjusted.value_counts().to_dict()}")
    print(f"  LISA zona-demeaned FDR: {loc.lisa_class_zone_demeaned.value_counts().to_dict()}")
    print(f"  Gi* FDR: {loc.gi_class.value_counts().to_dict()}")
    print(pd.crosstab(loc.zone_label, loc.lisa_class).to_string())
    print(pd.crosstab(loc.zone_label, loc.gi_class).to_string())

    # (6) orilla
    banner("(6) DISTANCIA A LA ORILLA", "-")
    loc["shore_km"], loc["inside_polygon"] = shore_distance(loc)
    print(f"  ubicaciones fuera del poligono: {(~loc.inside_polygon).sum()}")
    shore_tab, shore_ols = shore_analysis(loc)
    for _, rr in shore_tab.iterrows():
        print(f"  {rr.subset:14s} n={rr.n:3d} dist med {rr.shore_km_median:5.1f} km | "
              f"signo rho={rr.rho_signed:+.2f} P={rr.p_signed:.3f} | "
              f"abs rho={rr.rho_abs:+.2f} P={rr.p_abs:.3f}")
    for k in ["ols_signed", "ols_abs", "ols_signed_clarity", "ols_abs_clarity"]:
        c = shore_ols[k]["log_shore_km"]
        print(f"  {k}: log(dist) coef={c['coef']:+.3f} SE={c['se']:.3f} P={c['p']:.3f} (con zona)")
    print(f"  R2 del residuo explicado solo por zona: {shore_ols['ols_zone_only_r2']:.3f}")

    # --- salidas ---------------------------------------------------------------
    glob.to_csv(TIDY / "moran_global.csv", index=False, float_format="%.5f")
    corr.to_csv(TIDY / "moran_correlogram.csv", index=False, float_format="%.5f")
    ev_tab.to_csv(TIDY / "variogram_empirical.csv", index=False, float_format="%.5f")
    vm_tab.to_csv(TIDY / "variogram_model.csv", index=False, float_format="%.5f")
    shore_tab.to_csv(TIDY / "shore_distance.csv", index=False, float_format="%.5f")
    cols = ["location", "lat", "lon", "x_km", "y_km", "zona", "zone_label", "n",
            "n_campaigns", "secchi_mean", "residual", "residual_zone_demeaned",
            "residual_clarity_adjusted",
            "local_I", "z", "lag", "p_lisa", "q_lisa", "lisa_quadrant", "lisa_class",
            "lisa_class_unadjusted", "lisa_class_zone_demeaned", "gi_z", "gi_p", "gi_q",
            "gi_class", "shore_km", "inside_polygon"]
    loc[cols].to_csv(TIDY / "lisa_locations.csv", index=False, float_format="%.6f")

    def g(var, k=K_PRIMARY, unit="location"):
        return glob[(glob.unit == unit) & (glob.variable == var) & (glob.k == k)].iloc[0]

    prim, dem, sid = g("residual"), g("residual_zone_demeaned"), g("residual", unit="station_id_obsolete")
    cla = g("residual_clarity_adjusted")
    sig_k = glob[(glob.unit == "location") & (glob.variable == "residual")]
    sig_kd = glob[(glob.unit == "location") & (glob.variable == "residual_zone_demeaned")]
    cr = corr[corr.variable == "residual"]
    crd = corr[corr.variable == "residual_zone_demeaned"]
    sig_bands = [f"{int(a)}-{int(b)}" for a, b, p in zip(cr.lo_km, cr.hi_km, cr.p_two_sided) if p < ALPHA]
    sig_bands_d = [f"{int(a)}-{int(b)}" for a, b, p in zip(crd.lo_km, crd.hi_km, crd.p_two_sided) if p < ALPHA]
    bv, bvd = vario["residual"], vario["residual_zone_demeaned"]
    bvm, bvdm = bv["models"][bv["best_model"]], bvd["models"][bvd["best_model"]]
    lake = shore_tab.set_index("subset")
    zone_struct = prim["p_two_sided"] < ALPHA and dem["p_two_sided"] >= ALPHA

    out = {
        "residual_definition": "predicted - observed Secchi (m), out-of-fold, leave-one-campaign-out RF, FEATURES",
        "oof_metrics": m,
        "n_locations": int(len(loc)), "k_primary": K_PRIMARY, "n_perm": N_PERM, "alpha": ALPHA,
        "nearest_neighbour_km_median": r(np.median(nn), 2),
        "zone_mean_residual": {ZONE_LABELS[z]: {"mean": r(loc.loc[loc.zona == z, "residual"].mean(), 3),
                                                 "n_locations": int((loc.zona == z).sum())} for z in ZONES},
        "zone_only_r2_of_location_residual": r(shore_ols["ols_zone_only_r2"], 3),
        "moran_global": {
            "residual": {"I": r(prim.I), "E_I": r(prim.E_I), "z_perm": r(prim.z_perm, 2),
                         "p_two_sided": r(prim.p_two_sided), "p_one_sided": r(prim.p_one_sided_pos)},
            "zone_demeaned": {"I": r(dem.I), "z_perm": r(dem.z_perm, 2),
                              "p_two_sided": r(dem.p_two_sided), "p_one_sided": r(dem.p_one_sided_pos)},
            "clarity_adjusted": {"I": r(cla.I), "p_two_sided": r(cla.p_two_sided),
                                 "note": "residual minus its linear fit on location-mean observed Secchi"},
            "k_sensitivity_residual": {int(k): {"I": r(i), "p": r(p)} for k, i, p in
                                       zip(sig_k.k, sig_k.I, sig_k.p_two_sided)},
            "k_sensitivity_zone_demeaned": {int(k): {"I": r(i), "p": r(p)} for k, i, p in
                                            zip(sig_kd.k, sig_kd.I, sig_kd.p_two_sided)},
            "I_range_over_k": [r(sig_k.I.min()), r(sig_k.I.max())],
            "station_id_obsolete": {"n_stations": int(sid.n), "I": r(sid.I),
                                    "p_two_sided": r(sid.p_two_sided)},
        },
        "correlogram_significant_bands_km": sig_bands,
        "correlogram_significant_bands_km_zone_demeaned": sig_bands_d,
        "variogram": {
            "residual": {"model": bv["best_model"], "nugget": r(bvm["nugget"]), "sill": r(bvm["sill"]),
                         "range_km": r(bvm["range_km"], 1), "nugget_sill_ratio": r(bvm["nugget_sill_ratio"], 2),
                         "sample_variance": r(bv["sample_variance"])},
            "zone_demeaned": {"model": bvd["best_model"], "nugget": r(bvdm["nugget"]), "sill": r(bvdm["sill"]),
                              "range_km": r(bvdm["range_km"], 1),
                              "nugget_sill_ratio": r(bvdm["nugget_sill_ratio"], 2),
                              "sample_variance": r(bvd["sample_variance"])},
        },
        "lisa": {"fdr_counts": {c: int((loc.lisa_class == c).sum()) for c in ["HH", "LL", "HL", "LH"]},
                 "n_fdr_significant": int((loc.lisa_class != "ns").sum()),
                 "n_unadjusted_significant": int((loc.lisa_class_unadjusted != "ns").sum()),
                 "expected_by_chance_unadjusted": r(ALPHA * len(loc), 1),
                 "fdr_by_zone": {ZONE_LABELS[z]: {c: int(((loc.zona == z) & (loc.lisa_class == c)).sum())
                                                  for c in ["HH", "LL", "HL", "LH"]} for z in ZONES},
                 "n_fdr_significant_zone_demeaned": int((loc.lisa_class_zone_demeaned != "ns").sum())},
        "gi_star": {"n_hot": int((loc.gi_class == "hot").sum()), "n_cold": int((loc.gi_class == "cold").sum()),
                    "by_zone": {ZONE_LABELS[z]: {c: int(((loc.zona == z) & (loc.gi_class == c)).sum())
                                                 for c in ["hot", "cold"]} for z in ZONES}},
        "shore_distance": {
            "n_outside_polygon": int((~loc.inside_polygon).sum()),
            "polygon_source": "HydroLAKES v1.0, Hylak_id 78 (data/lake_boundary/titicaca.gpkg)",
            "spearman": {s: {"n": int(lake.loc[s, "n"]), "shore_km_median": r(lake.loc[s, "shore_km_median"], 2),
                             "rho_signed": r(lake.loc[s, "rho_signed"], 3), "p_signed": r(lake.loc[s, "p_signed"]),
                             "rho_abs": r(lake.loc[s, "rho_abs"], 3), "p_abs": r(lake.loc[s, "p_abs"]),
                             "rho_zone_demeaned": r(lake.loc[s, "rho_zone_demeaned"], 3),
                             "p_zone_demeaned": r(lake.loc[s, "p_zone_demeaned"])}
                         for s in lake.index},
            "ols_residual_on_zone_and_log_shore": {
                k: {"coef": r(v["log_shore_km"]["coef"]), "se": r(v["log_shore_km"]["se"]),
                    "p": r(v["log_shore_km"]["p"])} for k, v in
                [("signed", shore_ols["ols_signed"]), ("abs", shore_ols["ols_abs"]),
                 ("signed_plus_secchi_mean", shore_ols["ols_signed_clarity"]),
                 ("abs_plus_secchi_mean", shore_ols["ols_abs_clarity"])]},
            "outside_polygon_rule": "locations falling outside the HydroLAKES outline get shore_km = 0",
        },
    }
    shore_p_lake = lake.loc["Lake-wide", "p_signed"]
    shore_p_part = shore_ols["ols_signed"]["log_shore_km"]["p"]
    out["interpretation"] = {
        "residual_spatial_structure": bool(prim.p_two_sided < ALPHA),
        "explained_by_zone": bool(zone_struct),
        "within_zone_structure": bool(dem.p_two_sided < ALPHA),
        "range_km": r(bvm["range_km"], 1),
        "shore_effect_lake_wide": bool(shore_p_lake < ALPHA),
        "shore_effect_after_zone": bool(shore_p_part < ALPHA),
        "shore_effect_after_zone_and_clarity": bool(
            shore_ols["ols_signed_clarity"]["log_shore_km"]["p"] < ALPHA),
        "clarity_adjusted_moran_significant": bool(cla.p_two_sided < ALPHA),
        "text": (
            f"Location-mean residuals are {'positively autocorrelated' if prim.p_two_sided < ALPHA else 'not autocorrelated'} "
            f"(Moran I = {prim.I:.3f}, k={K_PRIMARY}, permutation P = {prim.p_two_sided:.4f}; "
            f"I = {sig_k.I.min():.3f}-{sig_k.I.max():.3f} over k = {min(K_SENS)}-{max(K_SENS)}). "
            f"After removing each zone's mean residual, I = {dem.I:.3f} (P = {dem.p_two_sided:.4f}): "
            + ("the structure is mostly a zone-level bias. " if zone_struct else
               ("structure persists within zones. " if dem.p_two_sided < ALPHA else "")) +
            f"Best-fit {bv['best_model']} variogram: range {bvm['range_km']:.1f} km, nugget/sill "
            f"{bvm['nugget_sill_ratio']:.2f}. Removing the regression-to-the-mean component (linear in "
            f"location-mean Secchi) leaves I = {cla.I:.3f} (P = {cla.p_two_sided:.4f}). Using IMARPE station IDs as the unit (obsolete) gives "
            f"I = {sid.I:.3f} (P = {sid.p_two_sided:.3f}), because IDs average readings from distant sites."),
        "design_implication": (
            "Residual error is spatially structured at the scale of zones/clusters of sites, so random or "
            "location-blocked CV can leak spatial information; spatial-block and campaign x spatial-block "
            "designs (p04) are needed to bound the effect." if prim.p_two_sided < ALPHA else
            "No residual spatial structure; spatial blocking is a robustness check only."),
    }
    write_json(out, MET / "spatial.json")
    banner("SALIDAS")
    for f in ["moran_global.csv", "moran_correlogram.csv", "variogram_empirical.csv",
              "variogram_model.csv", "lisa_locations.csv", "shore_distance.csv"]:
        print(f"  {TIDY / f}")
    print(f"  {MET / 'spatial.json'}")
    print("\n" + out["interpretation"]["text"])


if __name__ == "__main__":
    main()
