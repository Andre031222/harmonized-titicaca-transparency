"""
==============================================================================
18_results_and_figures.py
Modelo de produccion final + figuras del manuscrito + tendencias temporales.
Todo sobre datos REALES (match-ups S2+Landsat). Sin datos sinteticos.
==============================================================================
"""
import pandas as pd, numpy as np, json, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

ROOT=Path(__file__).resolve().parent.parent
PROC=ROOT/"data/processed"; MET=ROOT/"results/metrics"; FIG=ROOT/"results/figures"; MOD=ROOT/"results/models"
for d in (MET,FIG,MOD): d.mkdir(parents=True,exist_ok=True)

plt.rcParams.update({"font.size":10,"axes.grid":True,"grid.alpha":0.25,
                     "figure.dpi":120,"savefig.dpi":600,
                     "axes.axisbelow":True,"font.family":"DejaVu Sans"})
Zc={"BAHIA PUNO":"#d1495b","LAGO MENOR":"#edae49","LAGO MAYOR":"#2e86ab"}

def prep(d):
    e=1e-9
    d["NDWI"]=(d.B3-d.B8)/(d.B3+d.B8+e); d["NDTI"]=(d.B4-d.B3)/(d.B4+d.B3+e)
    d["B2_B3"]=d.B2/(d.B3+e); d["B4_B3"]=d.B4/(d.B3+e); d["B3_B2"]=d.B3/(d.B2+e); d["B2_B4"]=d.B2/(d.B4+e)
    return d
F=["B2","B3","B4","B8","B11","B12","NDWI","NDTI","B2_B3","B4_B3","B3_B2","B2_B4"]
Fl={"B2":"Blue","B3":"Green","B4":"Red","B8":"NIR","B11":"SWIR1","B12":"SWIR2",
    "NDWI":"NDWI","NDTI":"NDTI","B2_B3":"Blue/Green","B4_B3":"Red/Green",
    "B3_B2":"Green/Blue","B2_B4":"Blue/Red"}

def load():
    fr=[]
    for f,s in [("matchups_s2.csv","S2"),("matchups_ls.csv","LS")]:
        d=pd.read_csv(PROC/f); d=d[d.matched==1].copy(); d["sensor"]=s; fr.append(d)
    df=prep(pd.concat(fr,ignore_index=True)); df["year"]=pd.to_datetime(df.date).dt.year
    return df

def mk_test(x):
    """Mann-Kendall + Sen slope. Devuelve (tau, p, slope)."""
    x=np.asarray(x,float); n=len(x)
    if n<5: return np.nan,np.nan,np.nan
    s=sum(np.sign(x[j]-x[i]) for i in range(n-1) for j in range(i+1,n))
    var=n*(n-1)*(2*n+5)/18
    z=(s-np.sign(s))/np.sqrt(var) if var>0 else 0
    from math import erf,sqrt
    p=2*(1-0.5*(1+erf(abs(z)/sqrt(2))))
    slopes=[(x[j]-x[i])/(j-i) for i in range(n-1) for j in range(i+1,n)]
    return s/(0.5*n*(n-1)), p, np.median(slopes)

def main():
    df=load(); s=df[df.secchi.notna()].dropna(subset=F)
    X=s[F].values; y=s.secchi.values; g=s.station.values
    print(f"[final] {len(s)} match-ups con Secchi | {s.station.nunique()} estaciones | {s.year.min()}-{s.year.max()}")

    # --- predicciones out-of-fold (honestas) para el scatter ---
    gkf=GroupKFold(5); oof=np.zeros(len(y))
    for tr,te in gkf.split(X,y,g):
        sc=RobustScaler().fit(X[tr]); m=RandomForestRegressor(n_estimators=500,max_depth=12,min_samples_leaf=3,random_state=42,n_jobs=-1)
        m.fit(sc.transform(X[tr]),y[tr]); oof[te]=np.clip(m.predict(sc.transform(X[te])),0.05,None)
    R2=r2_score(y,oof); RMSE=np.sqrt(mean_squared_error(y,oof)); MAE=mean_absolute_error(y,oof)
    print(f"[final] OOF R2={R2:.3f} RMSE={RMSE:.2f}m MAE={MAE:.2f}m")

    # --- modelo de produccion final (todos los datos) + guardar ---
    scaler=RobustScaler().fit(X)
    model=RandomForestRegressor(n_estimators=600,max_depth=12,min_samples_leaf=3,random_state=42,n_jobs=-1).fit(scaler.transform(X),y)
    joblib.dump({"model":model,"scaler":scaler,"features":F},MOD/"secchi_rf_production.joblib")
    json.dump({"R2_oof":R2,"RMSE_oof":RMSE,"MAE_oof":MAE,"n":len(y),"n_stations":int(s.station.nunique()),
               "years":[int(s.year.min()),int(s.year.max())]},open(MET/"secchi_final_metrics.json","w"),indent=2)

    # ===== FIG: scatter observado vs predicho =====
    fig,ax=plt.subplots(figsize=(5.2,5))
    for z in ["LAGO MAYOR","LAGO MENOR","BAHIA PUNO"]:
        mk=s.zona.values==z
        ax.scatter(y[mk],oof[mk],s=20,alpha=0.6,color=Zc.get(z,"#666"),edgecolor="none",label=z.title())
    lim=[0,max(y.max(),oof.max())*1.05]; ax.plot(lim,lim,"k--",lw=1,zorder=1)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel("In-situ Secchi depth (m)"); ax.set_ylabel("Predicted Secchi depth (m)")
    ax.set_title(f"Random Forest — out-of-fold\n$R^2$={R2:.2f}, RMSE={RMSE:.2f} m, n={len(y)}")
    ax.legend(frameon=False,fontsize=8,loc="upper left"); fig.tight_layout()
    fig.savefig(FIG/"fig_scatter_secchi.png"); plt.close(fig)

    # ===== FIG: SHAP =====
    try:
        import shap
        ex=shap.TreeExplainer(model); sv=ex.shap_values(scaler.transform(X))
        imp=pd.Series(np.abs(sv).mean(0),index=[Fl[f] for f in F]).sort_values()
        fig,ax=plt.subplots(figsize=(5.5,4.2)); imp.plot.barh(ax=ax,color="#2e86ab")
        ax.set_xlabel("Mean |SHAP value|  (impact on predicted transparency)")
        ax.set_title("Feature importance (SHAP) — water transparency"); fig.tight_layout()
        fig.savefig(FIG/"fig_shap_importance.png"); plt.close(fig)
        imp.sort_values(ascending=False).to_csv(MET/"secchi_shap_final.csv")
    except Exception as e: print("SHAP:",repr(e)[:60])

    # ===== FIG: distribucion por zona y sensor =====
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    for z in ["BAHIA PUNO","LAGO MENOR","LAGO MAYOR"]:
        axes[0].hist(s[s.zona==z].secchi,bins=20,alpha=0.6,color=Zc[z],label=z.title())
    axes[0].set_xlabel("Secchi depth (m)"); axes[0].set_ylabel("n match-ups"); axes[0].legend(frameon=False,fontsize=8)
    axes[0].set_title("Transparency distribution by zone")
    for sen,c in [("S2","#2e86ab"),("LS","#d1495b")]:
        axes[1].hist(s[s.sensor==sen].secchi,bins=20,alpha=0.6,color=c,label=f"{sen} (n={sum(s.sensor==sen)})")
    axes[1].set_xlabel("Secchi depth (m)"); axes[1].legend(frameon=False,fontsize=8); axes[1].set_title("By sensor")
    fig.tight_layout(); fig.savefig(FIG/"fig_distribution.png"); plt.close(fig)

    # ===== Tendencias temporales (Mann-Kendall) sobre transparencia in-situ real por zona =====
    j=pd.read_csv(ROOT/"data/external_validation/jairo_titicaca_wq.csv")
    j.columns=[c.replace("\xf1","n").replace("�","n") for c in j.columns]
    j=j.rename(columns={"Ano":"year","Transparencia_m":"secchi","Zona":"zona"})
    j=j.dropna(subset=["secchi","year"])
    trend_rows=[]
    fig,ax=plt.subplots(figsize=(8.4,4.2))
    for z in ["BAHIA PUNO","LAGO MENOR","LAGO MAYOR"]:
        ann=j[j.zona==z].groupby("year").secchi.median()
        if len(ann)>=5:
            tau,p,sl=mk_test(ann.values)
            trend_rows.append({"zona":z,"n_years":len(ann),"tau":tau,"p":p,"sen_slope_m_yr":sl})
            ax.plot(ann.index,ann.values,"-o",color=Zc[z],label=f"{z.title()}\nSen={sl:+.2f} m/yr, p={p:.2f}",ms=5,lw=1.8)
    ax.set_xlabel("Year"); ax.set_ylabel("Median Secchi depth (m)")
    ax.set_title("In-situ transparency trends by zone (2011–2024)")
    ax.margins(y=0.15)
    ax.legend(frameon=False,fontsize=8,loc="center left",bbox_to_anchor=(1.01,0.5))
    fig.subplots_adjust(right=0.74)
    fig.savefig(FIG/"fig_trends.png",bbox_inches="tight"); plt.close(fig)
    pd.DataFrame(trend_rows).to_csv(MET/"transparency_trends.csv",index=False)
    print("\n=== Tendencias temporales (Mann-Kendall, transparencia in-situ por zona) ===")
    for r in trend_rows:
        sig="SIGNIFICATIVA" if r["p"]<0.05 else "no significativa"
        print(f"  {r['zona']:12s}: Sen={r['sen_slope_m_yr']:+.3f} m/yr, p={r['p']:.3f} ({sig}, {r['n_years']} anos)")

    print(f"\nFiguras en {FIG}, modelo en {MOD}")

if __name__=="__main__":
    main()
