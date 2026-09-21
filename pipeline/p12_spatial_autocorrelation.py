"""
p12_spatial_autocorrelation.py -- autocorrelacion espacial de los residuos.

Si el error del modelo se agrupa en el espacio, la validacion bloqueada por
fecha de campana podria seguir dejando informacion espacial compartida entre
folds, y el mapa de sesgo de la fig04 esconderia un patron. Aqui se mide sobre
el residuo medio por estacion del diseno primario:

  - I de Moran global (k vecinos mas cercanos, pesos estandarizados por fila),
    con prueba de permutacion y sensibilidad a k
  - correlograma: I de Moran por anillos de distancia
  - LISA (Moran local) con permutacion condicional, clases HH/LL/HL/LH

Moran y LISA se implementan aqui en numpy para no depender de PySAL.

    python pipeline/p12_spatial_autocorrelation.py
"""
import json

import geopandas as gpd
import numpy as np
import pandas as pd

from p00_config import MET, SEED, TIDY, banner

K_PRIMARY = 6
K_SENS = [4, 6, 8, 10]
N_PERM = 9999
N_PERM_LOCAL = 9999
ALPHA = 0.05
RINGS_KM = [0, 5, 10, 20, 40, 80, 160]


def station_table():
    oof = pd.read_csv(TIDY / "oof_predictions.csv")
    st = (oof.groupby("station")
          .agg(lat=("lat", "mean"), lon=("lon", "mean"),
               residual=("residual", "mean"), n=("residual", "size"),
               zone_label=("zone_label", "first"))
          .reset_index())
    xy = gpd.GeoSeries(gpd.points_from_xy(st.lon, st.lat), crs=4326).to_crs(32719)
    st["x_km"], st["y_km"] = xy.x / 1000, xy.y / 1000
    return st


def dist_matrix(st):
    p = st[["x_km", "y_km"]].values
    return np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1))


def knn_weights(D, k):
    n = len(D)
    W = np.zeros((n, n))
    order = np.argsort(D + np.eye(n) * 1e12, axis=1)[:, :k]
    for i in range(n):
        W[i, order[i]] = 1.0
    return W / W.sum(1, keepdims=True)


def ring_weights(D, lo, hi):
    W = ((D > lo) & (D <= hi)).astype(float)
    np.fill_diagonal(W, 0)
    return W


def moran(z, W):
    z = z - z.mean()
    return len(z) / W.sum() * (z @ W @ z) / (z @ z)


def moran_test(y, W, rng, n_perm=N_PERM):
    obs = moran(y, W)
    sims = np.array([moran(rng.permutation(y), W) for _ in range(n_perm)])
    # bilateral: extremos a cualquier lado de la media de la referencia
    dev = np.abs(sims - sims.mean())
    p = (1 + np.sum(dev >= abs(obs - sims.mean()))) / (n_perm + 1)
    return obs, p, float(-1 / (len(y) - 1)), np.quantile(sims, [0.025, 0.975])


def lisa(y, W, rng, n_perm=N_PERM_LOCAL):
    """Moran local con permutacion condicional (el valor de i queda fijo y se
    barajan los demas entre sus vecinos)."""
    z = (y - y.mean()) / y.std()
    lag = W @ z
    Ii = z * lag
    n = len(z)
    p = np.empty(n)
    for i in range(n):
        nb = np.nonzero(W[i])[0]
        others = np.delete(z, i)
        draws = np.array([rng.choice(others, len(nb), replace=False)
                          for _ in range(n_perm)])
        sim = z[i] * (draws * W[i, nb]).sum(1)
        # pseudo p unilateral en la direccion observada (convencion de GeoDa)
        extreme = np.sum(sim >= Ii[i]) if Ii[i] >= 0 else np.sum(sim <= Ii[i])
        p[i] = (1 + extreme) / (n_perm + 1)
    quad = np.where(z > 0, np.where(lag > 0, "HH", "HL"),
                    np.where(lag > 0, "LH", "LL"))
    cls = np.where(p < ALPHA, quad, "ns")
    return Ii, p, lag, cls


def bh(p):
    """q de Benjamini-Hochberg."""
    n = len(p)
    o = np.argsort(p)
    q = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(q, 1)
    return out


def main():
    banner("p12  AUTOCORRELACION ESPACIAL DE LOS RESIDUOS (diseno primario)")
    rng = np.random.default_rng(SEED)
    st = station_table()
    y = st.residual.values
    D = dist_matrix(st)
    print(f"  estaciones: {len(st)}  |  distancia al vecino mas cercano, mediana "
          f"{np.median(np.sort(D, 1)[:, 1]):.1f} km")

    sens = []
    for k in K_SENS:
        I, p, EI, _ = moran_test(y, knn_weights(D, k), rng)
        sens.append({"k": k, "moran_I": round(I, 4), "p": round(p, 4),
                     "expected_I": round(EI, 4)})
        print(f"  Moran global k={k:2d}: I = {I:+.3f}  (E[I] = {EI:+.4f}, p = {p:.4f})")
    prim = next(s for s in sens if s["k"] == K_PRIMARY)

    rings = []
    for lo, hi in zip(RINGS_KM[:-1], RINGS_KM[1:]):
        W = ring_weights(D, lo, hi)
        if W.sum() == 0:
            continue
        I, p, EI, env = moran_test(y, W, rng, n_perm=4999)
        pairs = int(W.sum() / 2)
        rings.append({"lo_km": lo, "hi_km": hi, "mid_km": (lo + hi) / 2,
                      "moran_I": round(I, 4), "p": round(p, 4), "pairs": pairs,
                      "env_lo": round(env[0], 4), "env_hi": round(env[1], 4)})
        print(f"  anillo {lo:3d}-{hi:3d} km: I = {I:+.3f}  p = {p:.4f}  pares = {pairs}")

    W = knn_weights(D, K_PRIMARY)
    Ii, p_loc, lag, cls = lisa(y, W, rng)
    st["local_I"], st["p_local"], st["lag"], st["cluster"] = Ii, p_loc, lag, cls
    st["z"] = (y - y.mean()) / y.std()
    st["q_local"] = bh(p_loc)
    counts = pd.Series(cls).value_counts().to_dict()
    n_fdr = int((st.q_local < ALPHA).sum())
    fdr_cls = st.loc[st.q_local < ALPHA, "cluster"].value_counts().to_dict()
    print(f"  LISA (k={K_PRIMARY}, p<{ALPHA}, sin ajustar): {counts}")
    print(f"  LISA tras Benjamini-Hochberg (q<{ALPHA}): {n_fdr} estaciones {fdr_cls}")
    print(f"  esperadas por azar sin ajustar: {ALPHA * len(st):.1f}")
    by_zone = (st[st.cluster != "ns"].groupby(["zone_label", "cluster"]).size()
               .rename("n").reset_index())
    print(by_zone.to_string(index=False))

    st.to_csv(TIDY / "lisa_stations.csv", index=False, float_format="%.6f")
    pd.DataFrame(rings).to_csv(TIDY / "moran_correlogram.csv", index=False)
    out = {"n_stations": int(len(st)), "k_primary": K_PRIMARY,
           "moran_I": prim["moran_I"], "moran_p": prim["p"],
           "moran_expected": prim["expected_I"], "sensitivity_k": sens,
           "correlogram": rings, "n_perm": N_PERM, "alpha": ALPHA,
           "lisa_counts": {c: int(counts.get(c, 0))
                           for c in ["HH", "LL", "HL", "LH", "ns"]},
           "lisa_significant": int((cls != "ns").sum()),
           "lisa_fdr_significant": n_fdr,
           "lisa_fdr_counts": {c: int(fdr_cls.get(c, 0))
                               for c in ["HH", "LL", "HL", "LH"]},
           "lisa_puno_hh": int(((st.cluster == "HH") &
                                (st.zone_label == "Bahia de Puno")).sum())}
    json.dump(out, open(MET / "spatial_autocorrelation.json", "w"), indent=2)
    print(f"\n  -> {MET / 'spatial_autocorrelation.json'}")
    print(f"  -> {TIDY / 'lisa_stations.csv'}, {TIDY / 'moran_correlogram.csv'}")


if __name__ == "__main__":
    main()
