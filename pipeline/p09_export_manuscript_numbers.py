"""
p09_export_manuscript_numbers.py -- Fuente unica de verdad para el manuscrito.

Recoge TODOS los numeros que aparecen en el articulo desde los artefactos que
produce el pipeline y los escribe en dos formatos:

  results/metrics/manuscript_numbers.json  -- para consulta programatica
  manuscript/jglr/numbers.tex              -- macros LaTeX \\numXxx

La razon de existir de este script es el fallo que provoco toda la auditoria:
el manuscrito anterior mezclaba en la misma frase un R2 del subconjunto
solo-Sentinel-2 (n=274) con el RMSE del modelo combinado (n=812), y afirmaba
que el test incluia El Nino 2023-24 cuando no existe dato de 2023. Si el texto
usa \\numHeadlineRtwo en lugar de teclear "0.574", eso no puede volver a pasar.
"""
import json
import re

import pandas as pd

from p00_config import MET, ROOT, TIDY, banner

OUT_TEX = ROOT / "manuscript/jglr/numbers.tex"


def load_json(name):
    p = MET / name
    return json.load(open(p)) if p.exists() else {}


def load_tidy(name):
    p = TIDY / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def texify(key):
    """snake_case -> \\numCamelCase (LaTeX no admite digitos ni guiones bajos)."""
    parts = re.split(r"[_\W]+", key)
    digits = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
              "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}
    out = "num"
    for p in parts:
        if not p:
            continue
        p = "".join(digits.get(c, c) for c in p)
        out += p[0].upper() + p[1:]
    return out


# Politica de redondeo por tipo de cantidad. Sin esto el manuscrito acaba
# escribiendo "99.75% de exactitud" y "cobertura del 71.264%", que sugieren una
# precision que el dato no tiene (n=812).
ROUNDING = [
    # (predicado sobre la clave, numero de decimales)
    (lambda k: "_acc" in k or "_cov" in k or "_pct" in k or "_share" in k, 1),
    (lambda k: k.endswith("_r2") or k.endswith("_rtwo"), 3),
    (lambda k: k.startswith("local_r_"), 2),
    (lambda k: k.endswith("_p"), 3),
    (lambda k: "_maxpred" in k, 1),
    (lambda k: any(t in k for t in ("_rmse", "_mae", "_bias", "_width",
                                    "_slope")), 2),
    (lambda k: "_ratio" in k or k.startswith("ls_over_s2"), 1),
    (lambda k: "intercept" in k, 4),
    (lambda k: k.startswith("secchi_"), 1),
]


def fmt(v, key=""):
    if not isinstance(v, float):
        return str(v)
    k = key.lower()
    for pred, nd in ROUNDING:
        if pred(k):
            return f"{v:.{nd}f}"
    return f"{v:.3f}".rstrip("0").rstrip(".") if abs(v) < 1000 else f"{v:.0f}"


def main():
    banner("p09 -- EXPORTACION DE NUMEROS PARA EL MANUSCRITO")
    N = {}

    # --- dataset ------------------------------------------------------------
    ds = load_json("dataset_summary.json")
    if ds:
        N.update({
            "matchups_total": ds["n_matched"],
            "matchups_s2": ds["n_matched_s2"],
            "matchups_ls": ds["n_matched_ls"],
            "matchups_secchi": ds["n_analysis"],
            "stations": ds["n_stations"],
            "campaigns": ds["n_campaign_dates"],
            "independent_events": ds["n_independent_events"],
            "cross_sensor_pairs": ds["n_cross_sensor_pseudoreplicates"],
            "images_per_matchup_s2": ds["images_per_matchup_s2"],
            "images_per_matchup_ls": ds["images_per_matchup_ls"],
            "year_first": min(ds["years_present"]),
            "year_last": max(ds["years_present"]),
            "years_missing": ", ".join(str(y) for y in ds["years_missing_in_span"]),
            "month_first": min(ds["months_present"]),
            "month_last": max(ds["months_present"]),
            "secchi_min": ds["secchi_min"], "secchi_max": ds["secchi_max"],
            "secchi_mean": ds["secchi_mean"], "secchi_sd": ds["secchi_sd"],
            "secchi_unique_values": ds["secchi_n_unique_values"],
        })

    # --- armonizacion -------------------------------------------------------
    hb = load_tidy("harmonization_bands.csv")
    if not hb.empty:
        for _, r in hb.iterrows():
            N[f"roy_ratio_{r.band_label.lower()}"] = r.intercept_over_signal
            N[f"ls_over_s2_{r.band_label.lower()}"] = r.ls_over_s2
        N["roy_ratio_max"] = float(hb.intercept_over_signal.max())
        N["roy_intercept_min"] = float(hb.roy_intercept.min())
        N["roy_intercept_max"] = float(hb.roy_intercept.max())

    sep = load_tidy("sensor_separability.csv")
    if not sep.empty:
        N["sensor_acc_all"] = float(sep.loc[sep.feature_set == "all_12_features",
                                            "accuracy"].iloc[0] * 100)
        N["sensor_acc_visible"] = float(sep.loc[sep.feature_set == "visible_only",
                                                "accuracy"].iloc[0] * 100)
        N["sensor_acc_ratios"] = float(sep.loc[sep.feature_set == "ratios_only",
                                               "accuracy"].iloc[0] * 100)
        N["sensor_acc_chance"] = float(sep.chance.iloc[0] * 100)

    lc = load_tidy("harmonization_local_coefficients.csv")
    if not lc.empty:
        for _, r in lc.iterrows():
            N[f"local_r_{r.band_label.lower()}"] = r.pearson_r
        N["paired_events"] = int(lc.n_paired.iloc[0])

    # --- benchmark ----------------------------------------------------------
    bm = load_tidy("benchmark_models.csv")
    if not bm.empty:
        m = {"artefact": "classical_broken", "classical": "classical_bounded",
             "linear": "linear", "rf": "rf"}
        for fam, key in m.items():
            sub = bm[bm.family == fam]
            if sub.empty:
                continue
            N[f"{key}_r2"] = float(sub.R2.iloc[0])
            N[f"{key}_rmse"] = float(sub.RMSE.iloc[0])
        art = bm[bm.family == "artefact"]
        if not art.empty:
            N["classical_broken_maxpred"] = float(art.max_prediction_m.iloc[0])

    # --- validacion ---------------------------------------------------------
    nm = load_tidy("null_models.csv")
    for _, r in nm.iterrows():
        N[f"null_{r.null_model}_r2"] = float(r.R2)
    # rejilla: nulo X bajo folds bloqueados por Y -> \numNullgrid<Y><X>RTwo
    ng = load_tidy("null_models_grid.csv")
    for _, r in ng.iterrows():
        N[f"nullgrid_{r.blocked_by}_{r.null_model}_r2"] = float(r.R2)
    vj = load_json("validation.json")
    for k, v in vj.get("campaign_stats", {}).items():
        N[k] = v
    # cuanto R2 separa el bloqueo por fecha (primario) del bloqueo por anio
    # (= campana entera): acota la dependencia intra-campana residual
    vh0 = load_tidy("validation_hierarchy.csv").set_index("case")
    N["date_vs_year_gap_r2"] = float(vh0.loc["by_campaign_date", "R2"]
                                     - vh0.loc["by_year", "R2"])

    vh = load_tidy("validation_hierarchy.csv")
    for case, key in [("random_kfold", "random"), ("by_station", "station"),
                      ("by_campaign_date", "campaign"), ("by_year", "year"),
                      ("one_record_per_event", "one_event"),
                      ("BAHIA PUNO", "zoneout_puno"),
                      ("LAGO MENOR", "zoneout_menor"),
                      ("LAGO MAYOR", "zoneout_mayor"),
                      ("visible_only", "visible"), ("ratios_only", "ratios"),
                      ("local_reharm", "local_reharm"),
                      ("S2_only", "s2_only"), ("LS_only", "ls_only")]:
        sub = vh[vh.case == case]
        if sub.empty:
            continue
        N[f"{key}_r2"] = float(sub.R2.iloc[0])
        N[f"{key}_rmse"] = float(sub.RMSE.iloc[0])
        N[f"{key}_n"] = int(sub.n.iloc[0])

    # titular: el diseno primario, para que no se pueda mezclar con otro
    camp = vh[vh.case == "by_campaign_date"]
    if not camp.empty:
        N["headline_r2"] = float(camp.R2.iloc[0])
        N["headline_rmse"] = float(camp.RMSE.iloc[0])
        N["headline_mae"] = float(camp.MAE.iloc[0])
        N["headline_bias"] = float(camp.bias.iloc[0])
        N["headline_n"] = int(camp.n.iloc[0])

    ez = load_tidy("error_by_zone.csv")
    for _, r in ez.iterrows():
        k = r.zone_label.lower().replace(" ", "_").replace("í", "i")
        N[f"zone_{k}_r2"] = float(r.R2)
        N[f"zone_{k}_rmse"] = float(r.RMSE)
        N[f"zone_{k}_bias"] = float(r.bias)
        N[f"zone_{k}_n"] = int(r.n)

    # --- temporal -----------------------------------------------------------
    th = load_tidy("temporal_holdout.csv")
    if not th.empty:
        r = th.iloc[-1]
        N.update({"temporal_r2": float(r.R2), "temporal_rmse": float(r.RMSE),
                  "temporal_n": int(r.n), "temporal_cut": int(r.cut_year),
                  "temporal_overlap_pct": float(r.station_overlap_pct),
                  "temporal_shared": int(r.test_stations_also_in_train),
                  "temporal_test_stations": int(r.test_stations)})

    tr = load_tidy("annual_trends.csv")
    if not tr.empty:
        for _, r in tr.drop_duplicates("zone_label").iterrows():
            k = r.zone_label.lower().replace(" ", "_").replace("í", "i")
            N[f"trend_{k}_slope"] = float(r.sen_slope_m_per_yr)
            N[f"trend_{k}_p"] = float(r.p_value)

    ts = load_tidy("trend_start_year_sensitivity.csv")
    if not ts.empty:
        N["trend_flips"] = int(ts.significant_at_005.sum())
        # cada zona y anio inicial: el texto cita el caso 2013 y no debe teclearlo
        for _, r in ts.iterrows():
            k = r.zone_label.lower().replace(" ", "_").replace("í", "i")
            N[f"trend_{k}_from_{int(r.start_year)}_slope"] = float(r.sen_slope_m_per_yr)
            N[f"trend_{k}_from_{int(r.start_year)}_p"] = float(r.p_value)
        N["trend_combinations"] = int(len(ts))

    # --- incertidumbre ------------------------------------------------------
    cc = load_tidy("conformal_coverage.csv")
    for _, r in cc.iterrows():
        lvl = int(round(r.nominal * 100))
        N[f"conformal_cov_{lvl}"] = float(r.empirical_coverage * 100)
        N[f"conformal_width_{lvl}"] = float(r.mean_width_m)
    if not cc.empty:
        N["conformal_max_error"] = float(cc.calibration_error.abs().max() * 100)

    iv = load_tidy("conformal_intervals.csv")
    if not iv.empty:
        N["conformal_outside"] = int((~iv.covered.astype(bool)).sum())
    wb = load_tidy("interval_width_by_secchi.csv")
    if not wb.empty:
        N["conformal_cov_clearest"] = float(wb.coverage.iloc[-1] * 100)
        N["conformal_cov_turbidest"] = float(wb.coverage.iloc[0] * 100)

    # --- interpretabilidad --------------------------------------------------
    ip = load_json("interpretability.json")
    if ip.get("share_by_band_group"):
        for k, v in ip["share_by_band_group"].items():
            key = k.lower().replace("/", "_").replace(" ", "_")
            N[f"shap_share_{key}"] = float(v * 100)
    rt = load_tidy("retrievability.csv")
    for _, r in rt.iterrows():
        N[f"retr_{r.variable}_r2"] = float(r.R2)
        N[f"retr_{r.variable}_n"] = int(r.n)

    # --- seleccion de modelo ------------------------------------------------
    ms = load_json("model_selection.json")
    if ms:
        N["tuning_gain_r2"] = ms.get("gain_R2")
        N["seed_noise_sd"] = ms.get("seed_noise_sd_R2")
        N["seed_noise_range"] = ms.get("seed_noise_range_R2")
        if ms.get("best"):
            N["best_model_r2"] = ms["best"].get("R2")
            N["best_model_label"] = ms["best"].get("label")

    # --- sensibilidad a la ventana del match-up (p11, requiere GEE) ---------
    ws = load_json("window_sensitivity.json")
    if ws:
        for w in ws.get("windows", []):
            d = w["window_days"]
            N[f"window_{d}_r2"] = float(w["R2"])
            N[f"window_{d}_rmse"] = float(w["RMSE"])
            N[f"window_{d}_n"] = int(w["n_matchups"])
            N[f"window_{d}_campaigns"] = int(w["n_campaigns"])
        for k in ("primary_n", "recovered_n", "recovered_pct", "reflectance_r_min"):
            if k in ws:
                N[f"window_{k}"] = ws[k]

    # --- autocorrelacion espacial de los residuos (p12) ---------------------
    sa = load_json("spatial_autocorrelation.json")
    if sa:
        ks = sa["sensitivity_k"]
        N["spatial_n"] = int(sa["n_stations"])
        N["spatial_k"] = int(sa["k_primary"])
        N["spatial_nperm"] = int(sa["n_perm"])
        N["spatial_moran_i"] = float(sa["moran_I"])
        N["spatial_moran_p"] = float(sa["moran_p"])
        N["spatial_moran_i_min"] = float(min(s["moran_I"] for s in ks))
        N["spatial_moran_i_max"] = float(max(s["moran_I"] for s in ks))
        N["spatial_moran_min_p"] = float(min(s["p"] for s in ks))
        N["spatial_ring_min_p"] = float(min(r["p"] for r in sa["correlogram"]))
        N["spatial_ring_max_km"] = int(max(r["hi_km"] for r in sa["correlogram"]))
        N["spatial_lisa_sig"] = int(sa["lisa_significant"])
        N["spatial_lisa_expected"] = round(sa["alpha"] * sa["n_stations"], 1)
        N["spatial_lisa_fdr"] = int(sa["lisa_fdr_significant"])
        N["spatial_lisa_puno_hh"] = int(sa["lisa_puno_hh"])
        N["spatial_lisa_hh"] = int(sa["lisa_counts"]["HH"])

    # --- escritura ----------------------------------------------------------
    json.dump(N, open(MET / "manuscript_numbers.json", "w"), indent=2,
              default=str)

    lines = [
        "% numbers.tex -- generado por pipeline/p09_export_manuscript_numbers.py",
        "% NO EDITAR A MANO. Regenerar con:  python pipeline/p09_export_manuscript_numbers.py",
        "% Uso en el texto:  \\numHeadlineRtwo  en lugar de teclear 0.574",
        "",
    ]
    for k, v in sorted(N.items()):
        if v is None:
            continue
        lines.append(f"\\newcommand{{\\{texify(k)}}}{{{fmt(v, k)}}}")
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"  {len(N)} numeros exportados")
    print(f"  {MET / 'manuscript_numbers.json'}")
    print(f"  {OUT_TEX}")
    print("\n  Titular del articulo (diseno primario, bloqueo por campana):")
    print(f"    R2   = {N.get('headline_r2')}")
    print(f"    RMSE = {N.get('headline_rmse')} m")
    print(f"    MAE  = {N.get('headline_mae')} m")
    print(f"    n    = {N.get('headline_n')}")
    missing = [k for k in ("headline_r2", "conformal_cov_90", "sensor_acc_all",
                           "paired_events") if k not in N]
    if missing:
        print(f"\n  AVISO: faltan claves ({', '.join(missing)}); "
              f"ejecuta antes los scripts p01-p08 que las producen.")


if __name__ == "__main__":
    main()
