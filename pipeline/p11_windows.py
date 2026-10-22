"""
p11_windows.py -- sensibilidad de la recuperacion a la ventana del match-up
(+-1, +-3, +-5, +-10 dias). No usa Earth Engine: lee matchups.csv.

Cada ventana define una muestra distinta (mas ancha = mas match-ups, otros
eventos), asi que su R2 no es comparable directamente. Por eso se evalua:
  * own      : cada ventana con su propia muestra;
  * common_1 : todas las ventanas sobre los mismos match-ups (sensor x evento)
               presentes a +-1 d;
  * common_3 : idem con los presentes a +-3 d.
En las muestras comunes las filas son las mismas, y la diferencia de R2
contra +-10 d lleva IC pareado por campanas. RF, 12 features, diseno primario.
Producto principal: hls; secundario: sr (bandpass_adjust + add_indices).

Salidas: results/tidy/window_sensitivity.csv
         results/metrics/window_sensitivity.json
"""
import json

import numpy as np
import pandas as pd

from p00_config import (load_matchups, FEATURES, MET, PRIMARY, PROC, TIDY, add_indices,
                        bandpass_adjust, banner, cluster_boot, cv_oof, metrics)

WINDOWS = [1, 3, 5, 10]
REF = 10


def load(product):
    m = load_matchups()
    m = m[(m["product"] == product) & m.secchi.notna()].copy()
    if product == "sr":
        m = bandpass_adjust(m)
    m = add_indices(m).dropna(subset=FEATURES)
    m["event"] = m.location + "|" + m.sampling_date
    m["key"] = m.sensor + "|" + m.event
    return m


def evaluate(d):
    d = d.reset_index(drop=True)
    y, p = cv_oof(d, FEATURES, PRIMARY)
    return d, y, p


def row(product, sample, w, d, y, p):
    m = metrics(y, p)
    lo, hi = cluster_boot(y, p, d.campaign.values)
    rlo, rhi = cluster_boot(y, p, d.campaign.values, stat="RMSE")
    return {"product": product, "sample": sample, "window_days": w,
            "n_matchups": int(len(d)), "n_S2": int((d.sensor == "S2").sum()),
            "n_LS": int((d.sensor == "LS").sum()),
            "n_sampling_dates": int(d.sampling_date.nunique()),
            "n_campaigns": int(d.campaign.nunique()),
            "median_n_img": float(d.n_img.median()),
            **m, "R2_lo": lo, "R2_hi": hi, "RMSE_lo": rlo, "RMSE_hi": rhi}


def main():
    banner("p11 -- SENSIBILIDAD A LA VENTANA DEL MATCH-UP")
    rows = []
    for product in ["hls", "sr"]:
        m = load(product)
        byw = {w: m[m.window == w] for w in WINDOWS}
        banner(f"producto {product}", "-")
        for w in WINDOWS:
            d, y, p = evaluate(byw[w])
            r = row(product, "own", w, d, y, p)
            rows.append(r)
            print(f"  own      +-{w:2d} d n={r['n_matchups']:4d} fechas={r['n_sampling_dates']:3d} "
                  f"camp={r['n_campaigns']} img={r['median_n_img']:.0f} "
                  f"R2={r['R2']:+.3f} [{r['R2_lo']:+.3f}, {r['R2_hi']:+.3f}] "
                  f"RMSE={r['RMSE']:.2f}")
        for base in [1, 3]:
            common = set(byw[base].key)
            for w in WINDOWS:
                if w < base:
                    continue
                common &= set(byw[w].key)
            res = {}
            for w in [x for x in WINDOWS if x >= base]:
                sub = byw[w][byw[w].key.isin(common)].sort_values("key")
                res[w] = evaluate(sub)
            dref, yref, pref = res[REF]
            for w, (d, y, p) in res.items():
                assert (d.key.values == dref.key.values).all()
                r = row(product, f"common_{base}", w, d, y, p)
                dr2 = round(r["R2"] - metrics(yref, pref)["R2"], 3)
                lo, hi = (0.0, 0.0) if w == REF else cluster_boot(
                    y, p, d.campaign.values, p2=pref)
                r.update({"dR2_vs_10d": dr2, "dR2_lo": lo, "dR2_hi": hi})
                rows.append(r)
                print(f"  common_{base} +-{w:2d} d n={r['n_matchups']:4d} "
                      f"R2={r['R2']:+.3f} [{r['R2_lo']:+.3f}, {r['R2_hi']:+.3f}] "
                      f"RMSE={r['RMSE']:.2f}  dR2 vs 10d={dr2:+.3f} [{lo:+.3f}, {hi:+.3f}]")

    out = pd.DataFrame(rows)
    out["sample_label"] = out["sample"].map(
        {"own": "Each window's own match-ups",
         "common_1": "Common match-ups (present at +-1 d)",
         "common_3": "Common match-ups (present at +-3 d)"})
    out.to_csv(TIDY / "window_sensitivity.csv", index=False)

    def recs(product, sample):
        s = out[(out["product"] == product) & (out["sample"] == sample)]
        return {f"pm{int(r.window_days)}d": {k: (v.item() if hasattr(v, "item") else v)
                                              for k, v in r.items()
                                              if k not in ("product", "sample", "sample_label")}
                for _, r in s.iterrows()}

    js = {"design": "RF (make_rf), 12 features, leave-one-campaign-out; R2 CI = "
                    "cluster bootstrap over campaigns; dR2 CI paired on identical rows",
          "windows_days": WINDOWS}
    for product in ["hls", "sr"]:
        js[product] = {s: recs(product, s) for s in ["own", "common_1", "common_3"]}
        own = out[(out["product"] == product) & (out["sample"] == "own")]
        js[product]["own_R2_range"] = [float(own.R2.min()), float(own.R2.max())]
    json.dump(js, open(MET / "window_sensitivity.json", "w"), indent=2)
    banner("SALIDAS")
    print(f"  {TIDY / 'window_sensitivity.csv'}\n  {MET / 'window_sensitivity.json'}")


if __name__ == "__main__":
    main()
