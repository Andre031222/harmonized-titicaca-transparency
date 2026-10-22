"""
p13_export_numbers.py -- exporta cada cifra que cita el manuscrito.

Lee results/metrics/*.json y results/tidy/*.csv y escribe
manuscript/rsase/numbers.tex (una macro por cifra) y
results/metrics/manuscript_numbers.json. El texto no teclea numeros.

Las cifras del registro de campo completo (881 mediciones) solo se pueden
recalcular con el registro de IMARPE; si no esta, se reutilizan las del
ultimo manuscript_numbers.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from p00_config import FIELD_RAW, MET, ROOT, TIDY, banner, write_json

OUT_TEX = ROOT / "manuscript/rsase/numbers.tex"
OUT_SUPP = ROOT / "manuscript/rsase/supp_tables.tex"


def tex_num(x, d=3):
    return "--" if pd.isna(x) else f"{x:.{d}f}".replace("-", "$-$")


def supp_tables():
    """Tablas del material suplementario, escritas desde results/tidy."""
    ms = pd.read_csv(TIDY / "model_selection.csv").sort_values("R2", ascending=False)
    feats = {"all12": "all 12", "visible": "visible only", "ratios": "ratios only"}
    rows = []
    for r in ms.itertuples():
        tuned = "tuned" if r.tuned else "default"
        target = "$\\ln Z_{SD}$" if r.target != "Zsd" else "$Z_{SD}$"
        rows.append(f"{r.learner}, {tuned} & {target} & {feats.get(r.features, r.features)} & "
                    f"{tex_num(r.R2)} [{tex_num(r.R2_lo, 2)}, {tex_num(r.R2_hi, 2)}] & "
                    f"{tex_num(r.RMSE, 2)} & {tex_num(r.dR2_vs_rf_default)} "
                    f"[{tex_num(r.dR2_lo)}, {tex_num(r.dR2_hi)}] \\\\")
    t1 = ("\\begin{table}[htbp]\n\\caption{Model selection under nested cross-validation "
          "(outer: leave-one-campaign-out; inner: grouped by campaign; Optuna). "
          "$\\Delta R^2$ is relative to the untuned Random Forest, with paired "
          "campaign-bootstrap 95\\% intervals.}\\label{tab:S1}\n\\centering\\small\n"
          "\\begin{adjustbox}{max width=\\linewidth}\\begin{tabular}{@{}lllrrr@{}}\\toprule\n"
          "Model & Target & Features & $R^2$ [95\\% CI] & RMSE (m) & $\\Delta R^2$ [95\\% CI] \\\\\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{adjustbox}\n\\end{table}\n")
    w = pd.read_csv(TIDY / "window_sensitivity.csv")
    lab = {"own": "own sample", "common_1": "common to $\\pm1$ d", "common_3": "common to $\\pm3$ d"}
    prod = {"hls": "HLS v2.0", "sr": "Agency SR + HLS coefficients"}
    rows = []
    for r in w.itertuples():
        rows.append(f"{prod[r.product]} & {lab[r.sample]} & $\\pm${r.window_days} & {r.n_matchups} & "
                    f"{r.n_sampling_dates} & {tex_num(r.R2)} [{tex_num(r.R2_lo, 2)}, {tex_num(r.R2_hi, 2)}] & "
                    f"{tex_num(r.RMSE, 2)} & {tex_num(r.dR2_vs_10d)} \\\\")
    t2 = ("\\begin{table}[htbp]\n\\caption{Retrieval by match-up window under leave-one-campaign-out "
          "validation. The common samples evaluate every window on the same match-ups; "
          "$\\Delta R^2$ is relative to the $\\pm10$-day window on those match-ups.}\\label{tab:S2}\n"
          "\\centering\\small\n\\begin{adjustbox}{max width=\\linewidth}"
          "\\begin{tabular}{@{}llrrrrrr@{}}\\toprule\n"
          "Product & Sample & Window & Match-ups & Dates & $R^2$ [95\\% CI] & RMSE (m) & $\\Delta R^2$ \\\\\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{adjustbox}\n\\end{table}\n")
    rc = pd.read_csv(TIDY / "recalibrated_retrieval.csv")
    rc = rc[rc.group == "pooled"]
    rows = []
    for r in rc.itertuples():
        label = r.variant_label.replace("->", "$\\rightarrow$")
        rows.append(f"{label} & {tex_num(r.R2)} [{tex_num(r.R2_lo, 2)}, {tex_num(r.R2_hi, 2)}] & "
                    f"{tex_num(r.dR2)} [{tex_num(r.dR2_lo)}, {tex_num(r.dR2_hi)}] & "
                    f"{tex_num(r.ba_paired, 2)} \\\\")
    t3 = ("\\begin{table}[htbp]\n\\caption{Retrieval after cross-sensor recalibration fitted within each "
          "training fold (leave-one-campaign-out). $\\Delta R^2$ is relative to the raw HLS baseline, with "
          "paired campaign-bootstrap 95\\% intervals; BA is the balanced accuracy of the sensor classifier "
          "on paired events after the recalibration.}\\label{tab:S3}\n\\centering\\small\n"
          "\\begin{adjustbox}{max width=\\linewidth}\\begin{tabular}{@{}lrrr@{}}\\toprule\n"
          "Variant & $R^2$ [95\\% CI] & $\\Delta R^2$ [95\\% CI] & BA \\\\\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{adjustbox}\n\\end{table}\n")
    to = pd.read_csv(TIDY / "toa_separability.csv").set_index("lake")
    gl = pd.read_csv(TIDY / "glint_separability.csv")
    g40 = gl[(gl.subset != "all") & (gl.threshold_deg == 40)].set_index("lake")
    ml = (pd.read_csv(TIDY / "multilake_hls_separability.csv").set_index("lake")
          .sort_values("s2_green_median"))
    rows = []
    for lake, r in ml.iterrows():
        t = to.loc[lake] if lake in to.index else None
        g = g40.loc[lake] if lake in g40.index else None
        low = tex_num(g.balanced_accuracy, 2) if g is not None and bool(g.tested) else "--"
        rows.append(f"{lake} & {r.s2_green_median:.4f} & {tex_num(r.balanced_accuracy, 2)} & "
                    f"{tex_num(t.hls_sr_bacc, 2) if t is not None else '--'} & "
                    f"{tex_num(t.toa_raw_bacc, 2) if t is not None else '--'} & {low} \\\\")
    t4 = ("\\begin{table}[htbp]\n\\caption{Same-day separability (balanced accuracy) of the HLS sensors by "
          "lake, ordered by median Sentinel-2 green reflectance. HLS: all pairs; HLS and TOA (subset): the "
          "same subset of pairs in surface reflectance and at the top of the atmosphere; low glint: pairs "
          "with a glint angle above 40\\textdegree{} for both sensors.}\\label{tab:S4}\n\\centering\\small\n"
          "\\begin{tabular}{@{}lrrrrr@{}}\\toprule\n"
          "Lake & Green & HLS & HLS (subset) & TOA (subset) & Low glint \\\\\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n\\end{table}\n")
    OUT_SUPP.write_text("%% Generado por pipeline/p13_export_numbers.py -- no editar a mano\n"
                        + t1 + "\n" + t2 + "\n" + t3 + "\n" + t4)


def J(name):
    return json.load(open(MET / f"{name}.json"))


def f(x, d=3, sign=False):
    """Numero con signo menos tipografico, valido dentro y fuera de modo matematico."""
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", "\\ensuremath{-}")


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def pval(p):
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def main():
    banner("p13 -- CIFRAS DEL MANUSCRITO")
    N = {}
    ds, va, be = J("dataset_summary"), J("validation"), J("benchmark")
    sp, te, un, it = J("spatial"), J("temporal"), J("uncertainty"), J("interpretability")
    ha, wi = J("harmonization"), J("window_sensitivity")

    # --- registro y dataset -------------------------------------------------
    prev = json.load(open(MET / "manuscript_numbers.json")) if (MET / "manuscript_numbers.json").exists() else {}
    if FIELD_RAW.exists():
        fr = pd.read_csv(FIELD_RAW)
        N["numFieldN"] = f"{len(fr)}"
        N["numFieldSecchi"] = f"{fr.Transparencia_m.notna().sum()}"
    else:
        N["numFieldN"], N["numFieldSecchi"] = prev["numFieldN"], prev["numFieldSecchi"]
    sid = pd.read_csv(TIDY / "station_id_spread.csv")
    N["numStationIds"] = f"{len(sid)}"
    N["numIdsMoving"] = f"{int((sid.max_spread_km > 10).sum())}"
    N["numIdsMaxKm"] = f"{sid.max_spread_km.max():.0f}"
    N["numMatch"] = f"{ds['n_analysis']}"
    N["numMatchSTwo"] = f"{ds['n_s2']}"
    N["numMatchLS"] = f"{ds['n_ls']}"
    N["numQcExcluded"] = f"{ds['n_qc_excluded']}"
    N["numLoc"] = f"{ds['n_locations']}"
    N["numDates"] = f"{ds['n_sampling_dates']}"
    N["numCampaigns"] = f"{ds['n_campaigns']}"
    N["numPairedEvents"] = f"{ds['n_paired_events']}"
    N["numEvents"] = f"{ds['n_events']}"
    yrs = [int(c) for c in ds["campaigns"]]
    N["numYearFirst"], N["numYearLast"] = f"{min(yrs)}", f"{max(yrs)}"
    miss = [y for y in range(min(yrs), max(yrs) + 1) if y not in yrs]
    N["numYearsMissing"] = ", ".join(map(str, miss[:-1])) + f" or {miss[-1]}"
    N["numSecchiHalfPct"] = pct(ds["secchi_frac_half_metre_multiples"], 0)
    N["numDatesSingleZone"] = f"{va['sampling_dates_single_zone']}"

    # --- validacion -----------------------------------------------------------
    h = va["headline"]
    N["numHeadR"], N["numHeadRlo"], N["numHeadRhi"] = f(h["R2"], 2), f(h["R2_lo"], 2), f(h["R2_hi"], 2)
    N["numHeadRmse"], N["numHeadBias"] = f(h["RMSE"], 2), f(h["bias"], 2)
    vd = {r["step"]: r for r in va["variance_decomposition"]}
    N["numVdZone"] = f(vd["zone"]["R2_added"], 2)
    N["numVdCampaignTotal"] = f(vd["+ campaign"]["R2_added"] + vd["+ zone x campaign"]["R2_added"], 2)
    g = va["null_grid"]
    short = {"random": "Random", "location": "Location", "sampling_date": "Date",
             "spatial_block": "Block", "campaign": "Campaign", "campaign_block": "CampBlock"}
    nulls = {"zone": "Zone", "zone+location": "Loc", "zone+sampling_date": "Date",
             "zone+campaign": "Camp"}
    for (dk, ds_) in short.items():
        for (nk, ns) in nulls.items():
            N[f"Null{ns}{ds_}"] = f(g[f"{dk}|{nk}"], 2)
    N["numNullCampLoc"] = f(g["location|zone+campaign"], 2)
    N["numNullCampDate"] = f(g["sampling_date|zone+campaign"], 2)
    N["numNullDateDate"] = f(g["sampling_date|zone+sampling_date"], 2)
    N["numNullLocCamp"] = f(g["campaign|zone+location"], 2)
    hier = {r["design"]: r for r in va["hierarchy"] if r["setting"] == "design"}
    for k, m in [("random", "Random"), ("location", "Location"), ("sampling_date", "Date"),
                 ("spatial_block", "Block"), ("campaign_block", "CampBlock")]:
        N[f"numRf{m}"] = f(hier[k]["R2"], 2)
    for k, m in [("location", "Location"), ("sampling_date", "Date")]:
        N[f"numInfl{m}"] = f(hier[k]["R2_minus_primary"], 2)
        N[f"numInfl{m}Lo"], N[f"numInfl{m}Hi"] = f(hier[k]["diff_lo"], 2), f(hier[k]["diff_hi"], 2)
    N["numCampBlockCost"] = f(-hier["campaign_block"]["R2_minus_primary"], 2)
    sub = {r["label"]: r for r in va["hierarchy"] if r["setting"] == "subset"}
    N["numSTwoOnlyR"] = f(sub["Sentinel-2 only"]["R2"], 2)
    N["numLsOnlyR"] = f(sub["Landsat only"]["R2"], 2)
    N["numSrRouteR"] = f(sub["Agency SR + HLS bandpass coefficients"]["R2"], 2)
    zo = [r for r in va["hierarchy"] if r["setting"] == "zone_out"]
    lo_, hi_ = min(r["R2"] for r in zo), max(r["R2"] for r in zo)
    N["numZoneOutRange"] = f"{f(lo_, 2)} to {f(hi_, 2)}"
    t = va["temporal_holdout"]
    N["numTemporalR"], N["numTemporalBias"] = f(t["R2"], 2), f(t["bias"], 2)
    N["numTemporalShared"] = f"{t['pct_shared']:.0f}"
    ez = {r["zona"]: r for r in va["error_by_zone"]}
    N["numBiasPuno"] = f(ez["BAHIA PUNO"]["bias"], 2, sign=True)
    ec = {r["campaign"]: r for r in va["error_by_campaign"]}
    N["numBiasTwoTwo"] = f(ec["2022"]["bias"], 2)
    oof = pd.read_csv(TIDY / "oof_predictions.csv")
    N["numPredSlope"] = f(np.polyfit(oof.secchi, oof.pred, 1)[0], 2)

    # --- benchmark y seleccion de modelos --------------------------------------
    P = be["pooled"]
    N["numLinR"], N["numTwoBandR"], N["numKloiberR"] = (f(P["linear"]["R2"], 2),
                                                       f(P["two_band"]["R2"], 2),
                                                       f(P["kloiber"]["R2"], 2))
    cmp_ = pd.read_csv(TIDY / "model_comparison_ci.csv")
    rl = cmp_[(cmp_.subset == "pooled") & (cmp_.comparison == "rf - linear")].iloc[0]
    N["numRfMinusLin"] = f(rl.dR2, 3)
    N["numRfMinusLinLo"], N["numRfMinusLinHi"] = f(rl.dR2_lo, 2), f(rl.dR2_hi, 2)
    ms = pd.read_csv(TIDY / "model_selection.csv")
    best = ms.loc[ms.R2.idxmax()]
    N["numNConfigs"] = f"{len(ms)}"
    N["numBestConfigR"] = f(best.R2, 2)
    N["numBestConfigDelta"] = f(best.dR2_vs_rf_default, 3, sign=True)
    N["numBestConfigDeltaLo"], N["numBestConfigDeltaHi"] = f(best.dR2_lo, 3), f(best.dR2_hi, 3)
    N["numTuningSentence"] = (f"the best of {len(ms)} configurations, a tuned {best.learner}, "
                              f"reached $R^2={f(best.R2, 2)}$, {f(best.dR2_vs_rf_default, 3, sign=True)} "
                              f"[{f(best.dR2_lo, 3)}, {f(best.dR2_hi, 3)}] relative to the Random Forest")

    # --- ventanas -------------------------------------------------------------
    w1 = wi["hls"]["common_1"]["pm1d"]
    N["numWinCommonN"] = f"{w1['n_matchups']}"
    N["numWinOneDelta"] = f(-w1["dR2_vs_10d"], 2)
    N["numWinOneDeltaLo"], N["numWinOneDeltaHi"] = f(-w1["dR2_hi"], 2), f(-w1["dR2_lo"], 2)

    # --- espacial ---------------------------------------------------------------
    mg = sp["moran_global"]
    N["numMoranI"], N["numMoranP"] = f(mg["residual"]["I"], 2), pval(mg["residual"]["p_two_sided"])
    N["numMoranIzd"], N["numMoranPzd"] = f(mg["zone_demeaned"]["I"], 2), pval(mg["zone_demeaned"]["p_two_sided"])
    N["numMoranIrange"] = f"{mg['I_range_over_k'][0]:.2f}--{mg['I_range_over_k'][1]:.2f}"
    N["numMoranIstation"] = f(mg["station_id_obsolete"]["I"], 2)
    N["numMoranPstation"] = pval(mg["station_id_obsolete"]["p_two_sided"])
    N["numVarioRange"] = f"{sp['variogram']['residual']['range_km']:.0f}"
    N["numVarioNugSill"] = f(sp["variogram"]["residual"]["nugget_sill_ratio"], 2)
    sh = sp["shore_distance"]
    lw = sh["spearman"]["Lake-wide"]
    N["numShoreRho"], N["numShoreP"] = f(lw["rho_signed"], 2), pval(lw["p_signed"])
    N["numShoreRhoZd"] = f(lw["rho_zone_demeaned"], 2)
    N["numShoreClarityP"] = pval(sh["ols_residual_on_zone_and_log_shore"]["signed_plus_secchi_mean"]["p"])

    # --- tendencias ----------------------------------------------------------------
    tr = te["trends"]
    for z, m in [("Bahia de Puno", "Puno"), ("Lago Mayor", "Mayor")]:
        r = tr["dry|raw"][z]
        N[f"numTrend{m}Slope"] = f(r["sen_slope_m_per_yr"], 2, sign=True)
        N[f"numTrend{m}P"] = pval(r["p_exact"])
        N[f"numTrend{m}LocP"] = pval(tr["dry|location"][z]["p_exact"])

    # --- incertidumbre ---------------------------------------------------------------
    mar = un["marginal"]
    for k, m in [("split", "Split"), ("mondrian", "Mondrian"), ("cqr", "Cqr")]:
        N[f"numCov{m}"] = pct(mar[k]["90"]["coverage"])
        N[f"numWidth{m}"] = f(mar[k]["90"]["mean_width_m"], 1)
    N["numConfWidthNineZero"] = f(mar["split"]["90"]["mean_width_m"], 1)
    cp = [v["coverage"] for v in un["coverage90_by_predicted_bin"]["split"].values()]
    N["numCovPredMin"], N["numCovPredMax"] = pct(min(cp), 0), pct(max(cp), 0)
    ex = un["measured_extremes"]
    clear = ex[">12.5"]["coverage_90"]
    N["numCovClearSplit"] = pct(min(clear.values()), 0)
    N["numCovClearMondrian"] = pct(max(clear.values()), 0)
    turb = ex["<5"]["coverage_90"]
    N["numCovTurbidMin"], N["numCovTurbidMax"] = pct(min(turb.values()), 0), pct(max(turb.values()), 0)
    N["numBiasClear"] = f(ex[">12.5"]["mean_bias_pred_minus_obs"], 1)
    N["numBiasTurbid"] = f(ex["<5"]["mean_bias_pred_minus_obs"], 1)
    cr = un["coverage90_campaign_range"]["split"]
    N["numCovCampMin"], N["numCovCampMax"] = pct(cr[0], 0), pct(cr[1], 0)

    # --- armonizacion ------------------------------------------------------------------
    T = ha["tests"]
    def ba(key):
        return T[key]["balanced_accuracy"]
    N["numSepHlsAll"] = f(ba("hls|all|campaign|all12|none"), 2)
    N["numSepHlsAllLo"] = f(T["hls|all|campaign|all12|none"]["ba_lo"], 2)
    N["numSepHlsAllHi"] = f(T["hls|all|campaign|all12|none"]["ba_hi"], 2)
    N["numSepHlsAllAuc"] = f(T["hls|all|campaign|all12|none"]["auc"], 2)
    N["numSepHlsPaired"] = f(ba("hls|paired|campaign|all12|none"), 2)
    k3 = "hls|pm3d|campaign|all12|none"
    N["numSepHlsPmThree"] = f(ba(k3), 2)
    N["numSepHlsPmThreeLo"], N["numSepHlsPmThreeHi"] = f(T[k3]["ba_lo"], 2), f(T[k3]["ba_hi"], 2)
    N["numSepHlsVis"] = f(ba("hls|all|campaign|visible|none"), 2)
    N["numSepSrAll"] = f(ba("sr|all|campaign|all12|none"), 2)
    N["numSepSrRawAll"] = f(ba("sr_raw|all|campaign|all12|none"), 2)
    N["numSepOlsPmThree"] = f(ba("hls|pm3d|campaign|all12|ols"), 2)
    N["numSepRmaPmThree"] = f(ba("hls|pm3d|campaign|all12|rma"), 2)
    N["numSepQmPmThree"] = f(ba("hls|pm3d|campaign|all12|qm_bands"), 2)
    N["numSepQmPaired"] = f(ba("hls|paired|campaign|all12|qm_bands"), 2)
    N["numSepQmFeatPmThree"] = f(ba("hls|pm3d|campaign|all12|qm_features"), 2)
    c = ha["counts"]["hls"]
    N["numPairedHls"] = f"{c['n_paired_events']}"
    N["numPmThreeHls"] = f"{c['n_pairs_pm3d_single_image']}"
    N["numHlsAllN"] = f"{c['n_all']}"
    ctrl = [x for x in ha["controls"] if x["route"] == "hls" and x["sample"] == "all"
            and x["control"] == "dates"]
    N["numCtrlHlsMean"] = f(np.mean([x["ba_mean"] for x in ctrl]), 2)
    bands = pd.DataFrame(ha["bands"])
    def rr(route, band):
        s = bands[(bands.route == route) & (bands["sample"].str.startswith("paired"))
                  & (bands.band == band)]
        return f(float(s.pearson_r.iloc[0]), 2)
    N["numRGreenHls"], N["numRGreenSr"] = rr("hls", "B3"), rr("sr", "B3")
    nir = bands[(bands.route == "hls") & (bands["sample"] == "all_w10") & (bands.band == "B8")]
    N["numNirNegPct"] = f"{float(nir.pct_s2_le0.iloc[0]):.0f}"
    shift = {r["band"]: r for r in ha["bandpass_shift_sr"]}
    N["numShiftBlue"] = f(shift["B2"]["median_shift_pct"], 0)
    N["numShiftSwirOne"] = f(shift["B11"]["median_shift_pct"], 0)

    # --- multilagos --------------------------------------------------------------------
    ml = J("multilake_hls")
    mls = pd.read_csv(TIDY / "multilake_hls_separability.csv")
    N["numLakesTested"] = f"{ml['n_lakes_tested']}"
    N["numLakesN"] = f"{mls.lake.str.replace(' Mayor| Menor', '', regex=True).nunique()}"
    N["numMlPairs"] = f"{ml['n_pairs_total']:,}".replace(",", "{,}")
    N["numMlDates"] = f"{ml['n_dates_total']:,}".replace(",", "{,}")
    N["numMlBaMin"], N["numMlBaMax"] = f(ml["bacc_range"][0], 2), f(ml["bacc_range"][1], 2)
    ctrl = pd.concat([mls.control_LS_bacc, mls.control_S2_bacc])
    N["numMlCtrlMin"], N["numMlCtrlMax"] = f(ctrl.min(), 2), f(ctrl.max(), 2)
    N["numMlPooled"] = f(ml["pooled_lake_date"]["balanced_accuracy"], 2)
    N["numMlLeaveLake"] = f(ml["pooled_leave_lake_out"]["balanced_accuracy"], 2)
    sp_ = ml["spearman_vs_s2_green"]
    N["numMlRho"] = f(sp_["balanced_accuracy_vs_clarity"]["rho"], 2)
    N["numMlRhoGreenRatio"] = f(sp_["ratio_B3_vs_clarity"]["rho"], 2)
    cvt = ml["clear_vs_turbid"]
    N["numMlClearMed"], N["numMlTurbidMed"] = f(cvt["clear"]["bacc_median"], 2), f(cvt["turbid"]["bacc_median"], 2)
    tt = mls.set_index("lake")
    N["numMlTiticacaMayor"] = f(tt.loc["Titicaca Mayor", "balanced_accuracy"], 2)
    N["numMlTiticacaMenor"] = f(tt.loc["Titicaca Menor", "balanced_accuracy"], 2)
    clear = mls.s2_green_median < cvt["green_split"]
    N["numMlRatioGreenClear"] = f(mls.loc[clear, "ratio_B3"].median(), 2)
    N["numMlRatioGreenTurbid"] = f(mls.loc[~clear, "ratio_B3"].median(), 2)
    from scipy.stats import spearmanr
    rg = spearmanr(mls.balanced_accuracy, mls.s2_vza_median, nan_policy="omit")
    N["numMlRhoVza"], N["numMlPVza"] = f(rg.statistic, 2), f(rg.pvalue, 2)
    N["numMlVzaSTwo"] = f"{mls.s2_vza_median.median():.1f}"
    N["numMlVzaLS"] = f"{mls.ls_vza_median.median():.1f}"

    # --- recalibracion, mapas, TOA y glint ------------------------------------------
    rc = J("recalibrated_retrieval")["variants"]
    qm = rc["qm_s2ls_paired"] if "qm_s2ls_paired" in rc else next(
        v for k, v in rc.items() if k.startswith("qm") and "paired" in k)
    N["numRecalQmR"] = f(qm["R2"], 2)
    N["numRecalQmBa"] = f(qm["ba_paired"], 2)
    dr = [(k, v.get("dR2"), v.get("dR2_lo"), v.get("dR2_hi")) for k, v in rc.items()
          if v.get("group") == "pooled" and k != "baseline" and v.get("method") not in (None, "", "sensor_feature")]
    N["numRecalDrMin"] = f(min(x[1] for x in dr if x[1] is not None), 2)
    N["numRecalDrMax"] = f(max(x[1] for x in dr if x[1] is not None), 2)
    N["numRecalN"] = f"{len(dr)}"
    lm = J("lake_maps")
    dfp = lm["difference_S2_minus_LS"]["pooled"]
    N["numMapDiffMedian"] = f(dfp["median"], 2, sign=True)
    N["numMapDiffIqrLo"], N["numMapDiffIqrHi"] = f(dfp["iqr"][0], 2), f(dfp["iqr"][-1], 2)
    N["numMapDiffGtOne"] = f"{dfp['pct_abs_gt_1m']:.0f}"
    N["numMapDiffGtHalf"] = f"{dfp['pct_abs_gt_marginal_halfwidth90']:.2f}"
    N["numMapR"] = f(lm["pearson_r_LS_vs_S2_pooled_maps"], 2)
    N["numMapHalfWidth"] = f(lm["halfwidth90_m"]["marginal_split"], 1)
    fw = lm["field_2024"]["metrics"]["without_2024"]
    N["numMapLsR"], N["numMapLsRmse"] = f(fw["LS_pooled"]["R2"], 2), f(fw["LS_pooled"]["RMSE"], 2)
    N["numMapStwoR"], N["numMapStwoRmse"] = f(fw["S2_pooled"]["R2"], 2), f(fw["S2_pooled"]["RMSE"], 2)
    N["numMapFieldN"] = f"{fw['LS_pooled']['n']}"
    N["numMapQcPct"] = f"{lm['coverage']['LS']['pct_of_lake_pixels_pass_qc']:.0f}"
    zd = lm["zones"]["LAGO MAYOR"]["diff_pooled"]
    N["numMapDiffMayor"] = f(zd["mean"], 2, sign=True)
    to = J("toa")
    pl = to["paired_per_lake"]
    N["numToaMed"], N["numToaSrMed"] = f(pl["median_bacc"]["toa_raw"], 2), f(pl["median_bacc"]["hls_sr"], 2)
    N["numToaMin"], N["numToaMax"] = f(pl["range_bacc"]["toa_raw"][0], 2), f(pl["range_bacc"]["toa_raw"][1], 2)
    N["numToaLower"] = f"{pl['n_lakes_toa_lower']}"
    N["numToaP"] = f"{pl['wilcoxon_toa_vs_sr_p']:.4f}"
    N["numToaPairs"] = f"{to['n_pairs_with_toa']:,}".replace(",", "{,}")
    N["numToaRho"] = f(to["spearman_vs_hls_s2_green"]["toa_raw_bacc"]["rho"], 2)
    N["numToaDeltaRho"] = f(to["spearman_vs_hls_s2_green"]["delta_bacc_toa_minus_sr"]["rho"], 2)
    N["numToaDt"] = f"{to['acquisition_geometry']['dt_min_median_all']:.0f}"
    br = to["band_ratio_median_over_lakes"]
    N["numToaRatioGreen"], N["numSrRatioGreen"] = f(br["toa"]["B3"], 2), f(br["hls_sr"]["B3"], 2)
    N["numToaRatioBlue"], N["numSrRatioBlue"] = f(br["toa"]["B2"], 2), f(br["hls_sr"]["B2"], 2)
    gl = J("glint")
    N["numGlintAll"] = f(gl["pooled"]["all"]["balanced_accuracy"], 2)
    N["numGlintThirty"] = f(gl["pooled"]["glint_gt_30"]["balanced_accuracy"], 2)
    N["numGlintForty"] = f(gl["pooled"]["glint_gt_40"]["balanced_accuracy"], 2)
    N["numGlintFortyPct"] = f"{100 * gl['glint_distribution']['frac_pairs_both_gt40']:.0f}"
    N["numGlintRho"] = f(gl["spearman_per_lake"]["bacc_vs_abs_glint_diff_median"]["rho"], 2)
    N["numGlintRhoP"] = f(gl["spearman_per_lake"]["bacc_vs_abs_glint_diff_median"]["p"], 2)

    # --- escribir ---------------------------------------------------------------------
    lines = ["%% Generado por pipeline/p13_export_numbers.py -- no editar a mano"]
    for k in sorted(N):
        lines.append(f"\\newcommand{{\\{k}}}{{{N[k]}}}")
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text("\n".join(lines) + "\n")
    write_json(N, MET / "manuscript_numbers.json")
    supp_tables()
    pend = [k for k, v in N.items() if v == "PENDING"]
    print(f"  {len(N)} macros -> {OUT_TEX}" + (f"  (pendientes: {pend})" if pend else ""))


if __name__ == "__main__":
    main()
