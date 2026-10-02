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

from p00_config import MET, PROC, ROOT, TIDY, banner

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
    (lambda k: k.endswith("_spearman"), 2),
    (lambda k: k.endswith("_bal"), 1),
    (lambda k: k.endswith("_auc"), 2),
    (lambda k: k.startswith("shore_rho"), 2),
    (lambda k: k.startswith("shore_p"), 3),
    (lambda k: k.startswith("hls_shift") or k.startswith("ls_red_nonpos_pct"), 0),
    (lambda k: k.startswith("paired_ratio") or k.startswith("multilake_vis"), 2),
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
            "secchi_half_metre_pct": 100 * ds["secchi_frac_half_metre_multiples"],
        })

    # --- armonizacion (HLS, p02) ---------------------------------------------
    hb = load_tidy("harmonization_bands.csv")
    if not hb.empty:
        for _, r in hb.iterrows():
            k = r.band_label.lower()
            N[f"hls_shift_{k}"] = float(r.hls_shift_pct)
            N[f"paired_ratio_{k}"] = float(r.paired_ls_over_s2_hls)
            N[f"ls_native_{k}"] = float(r.native_ls_median)
            N[f"s2_hls_{k}"] = float(r.s2_hls_median)
        N["hls_shift_absmax"] = float(hb.hls_shift_pct.abs().max())
        N["hls_intercept_absmax"] = float(hb.hls_intercept.abs().max())
        N["paired_events"] = int(hb.n_paired.iloc[0])

    sep = load_tidy("sensor_separability.csv")
    if not sep.empty:
        for _, r in sep.iterrows():
            N[f"sep_{r.test}_bal"] = float(r.balanced_accuracy * 100)
            N[f"sep_{r.test}_acc"] = float(r.accuracy * 100)
            N[f"sep_{r.test}_auc"] = float(r.auc)
            N[f"sep_{r.test}_n"] = int(r.n)
        N["sensor_acc_chance"] = float(sep.loc[sep.test == "all_hls", "chance"].iloc[0] * 100)

    lc = load_tidy("harmonization_local_coefficients.csv")
    if not lc.empty:
        for _, r in lc.iterrows():
            N[f"local_r_{r.band_label.lower()}"] = r.pearson_r

    # rojo nativo de Landsat <= 0 sobre agua clara (motiva el piso de los ratios)
    ad = pd.read_csv(PROC / "analysis_dataset.csv") if (PROC / "analysis_dataset.csv").exists() else pd.DataFrame()
    if not ad.empty:
        ls = ad[ad.sensor == "LS"]
        N["ls_red_nonpos_n"] = int((ls.B4 <= 0).sum())
        N["secchi_matchups_ls"] = int(len(ls))
        N["secchi_matchups_s2"] = int((ad.sensor == "S2").sum())
        N["sampling_dates"] = int(ad.campaign_date.nunique())
        N["matchup_years"] = int(ad.year.nunique())
        N["ls_red_nonpos_pct"] = float((ls.B4 <= 0).mean() * 100)

    # --- benchmark ----------------------------------------------------------
    bm = load_tidy("benchmark_models.csv")
    if not bm.empty:
        m = {"artefact": "classical_broken", "classical": "classical_bounded",
             "kloiber": "kloiber", "linear": "linear", "rf": "rf"}
        for fam, key in m.items():
            sub = bm[bm.family == fam]
            if sub.empty:
                continue
            N[f"{key}_r2"] = float(sub.R2.iloc[0])
            N[f"{key}_rmse"] = float(sub.RMSE.iloc[0])
            N[f"{key}_mae"] = float(sub.MAE.iloc[0])

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
    # el sesgo sale de las predicciones sin redondear: error_by_zone.csv lo
    # guarda con 3 decimales y redondear otra vez daba 1.01 en vez de 1.02
    oof = load_tidy("oof_predictions.csv")
    if not oof.empty:
        for zl, b in oof.groupby("zone_label").residual.mean().items():
            k = zl.lower().replace(" ", "_").replace("í", "i")
            N[f"zone_{k}_bias"] = float(b)

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
            N[f"trend_{k}_years"] = int(r.n_years)

    # sensibilidad: la misma prueba incluyendo la campana de lluvias (dic 2012)
    tss = load_tidy("trend_season_sensitivity.csv")
    if not tss.empty:
        for _, r in tss[tss.series == "all_campaigns"].iterrows():
            k = r.zone_label.lower().replace(" ", "_").replace("í", "i")
            N[f"trend_{k}_all_slope"] = float(r.sen_slope_m_per_yr)
            N[f"trend_{k}_all_p"] = float(r.p_value)

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
        N[f"conformal_{r.method}_cov_{lvl}"] = float(r.empirical_coverage * 100)
        N[f"conformal_{r.method}_width_{lvl}"] = float(r.mean_width_m)
    for meth, g in cc.groupby("method"):
        N[f"conformal_{meth}_max_error"] = float(g.calibration_error.abs().max() * 100)

    iv = load_tidy("conformal_intervals.csv")
    for meth, g in iv.groupby("method"):
        N[f"conformal_{meth}_outside"] = int((~g.covered.astype(bool)).sum())
    wb = load_tidy("interval_width_by_secchi.csv")
    for (meth, by), g in wb.groupby(["method", "conditioned_on"]):
        k = f"conformal_{meth}_{by}"
        N[f"{k}_cov_min"] = float(g.coverage.min() * 100)
        N[f"{k}_cov_max"] = float(g.coverage.max() * 100)
        N[f"{k}_cov_clearest"] = float(g.coverage.iloc[-1] * 100)
        N[f"{k}_cov_turbidest"] = float(g.coverage.iloc[0] * 100)
    N["conformal_n"] = int(iv[iv.method == "split"].shape[0]) if not iv.empty else 0

    # --- interpretabilidad --------------------------------------------------
    ip = load_json("interpretability.json")
    if ip.get("share_by_band_group"):
        for k, v in ip["share_by_band_group"].items():
            key = k.lower().replace("/", "_").replace(" ", "_")
            N[f"shap_share_{key}"] = float(v * 100)
    imp = load_tidy("shap_importance.csv")
    if not imp.empty:
        N["shap_top_label"] = str(imp.label.iloc[0])
        N["shap_top_share"] = float(imp.share.iloc[0] * 100)
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
            sel = load_tidy("model_selection.csv").set_index("label")
            N["tuning_gain_rmse"] = round(float(sel.loc["RF por defecto", "RMSE"]
                                                - ms["best"].get("RMSE")), 2)

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
        # Landsat invertido frente a Landsat leido directamente de Collection 2
        if "inversion_n" in ws:
            def sci(x):
                m, e = f"{x:.1e}".split("e")
                return f"{m}\\times10^{{{int(e)}}}"
            N["inversion_n"] = int(ws["inversion_n"])
            N["inversion_median_abs_diff"] = sci(ws["inversion_median_abs_diff"])
            N["inversion_p95_abs_diff"] = sci(ws["inversion_p95_abs_diff"])
            N["inversion_r_floor"] = ("0.9999" if ws["inversion_r_min"] >= 0.9999
                                      else f"{ws['inversion_r_min']:.4f}")
            N["pairs_pm1d"] = int(ws["pairs_pm1d"])

    # --- autocorrelacion espacial de los residuos (p12) ---------------------
    sa = load_json("spatial_autocorrelation.json")
    if sa:
        ks = sa["sensitivity_k"]
        N["spatial_n"] = int(sa["n_stations"])
        sd = sa.get("shore_distance", {})
        if "all" in sd:
            N["shore_rho"] = float(sd["all"]["rho_residual"])
            N["shore_p"] = float(sd["all"]["p_residual"])
            N["shore_rho_abs"] = float(sd["all"]["rho_abs_residual"])
            N["shore_p_abs"] = float(sd["all"]["p_abs_residual"])
            N["shore_km_median"] = float(sd["all"]["shore_km_median"])
        if "Bahia de Puno" in sd:
            pu = sd["Bahia de Puno"]
            N["shore_rho_puno"] = float(pu["rho_residual"])
            N["shore_p_puno"] = float(pu["p_residual"])
            N["shore_n_puno"] = int(pu["n"])
            N["shore_km_median_puno"] = float(pu["shore_km_median"])
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

    # --- el diagnostico en otros lagos grandes (p13, requiere GEE) ----------
    ml = load_json("multilake_harmonization.json")
    if ml:
        lakes = pd.DataFrame(ml["lakes"])
        N["multilake_n"] = int(len(lakes))
        N["multilake_points"] = int(lakes.n_points.iloc[0])
        N["multilake_below"] = int(ml["lakes_below_both_visible"])
        N["multilake_spearman"] = float(ml["spearman_green_vs_green_gap"])
        vis = lakes[["blue_ratio", "green_ratio"]]
        N["multilake_vis_ratio_min"] = float(vis.min().min())
        N["multilake_vis_ratio_max"] = float(vis.max().max())

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
    missing = [k for k in ("headline_r2", "conformal_split_cov_90",
                           "sep_all_hls_bal", "paired_events") if k not in N]
    if missing:
        print(f"\n  AVISO: faltan claves ({', '.join(missing)}); "
              f"ejecuta antes los scripts p01-p08 que las producen.")


if __name__ == "__main__":
    main()
