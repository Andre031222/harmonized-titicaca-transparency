"""
==============================================================================
23_figures_q1.py
Figuras nuevas de nivel Q1:
 (A) fig_validation_hierarchy.png : jerarquia de validacion espacial
     (random ~ estacion >> leave-one-zone-out) + hold-out temporal.
     Materializa la contribucion metodologica (envelope de transferibilidad).
 (B) fig_uncertainty.png : intervalos de prediccion conformal (90%) sobre el
     scatter OOF + curva de cobertura empirica vs nominal.
Lee results/metrics/validation_q1.csv y recomputa la incertidumbre conformal.
==============================================================================
"""
import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

ROOT=Path(__file__).resolve().parent.parent
PROC=ROOT/"data/processed"; MET=ROOT/"results/metrics"; FIG=ROOT/"results/figures"
FIG.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({"font.size":10,"axes.grid":True,"grid.alpha":0.25,
                     "figure.dpi":120,"savefig.dpi":600,"axes.axisbelow":True,"font.family":"DejaVu Sans"})

F=["B2","B3","B4","B8","B11","B12","NDWI","NDTI","B2_B3","B4_B3","B3_B2","B2_B4"]
def prep(d):
    e=1e-9
    d["NDWI"]=(d.B3-d.B8)/(d.B3+d.B8+e); d["NDTI"]=(d.B4-d.B3)/(d.B4+d.B3+e)
    d["B2_B3"]=d.B2/(d.B3+e); d["B4_B3"]=d.B4/(d.B3+e); d["B3_B2"]=d.B3/(d.B2+e); d["B2_B4"]=d.B2/(d.B4+e)
    return d
def load():
    fr=[]
    for f,s in [("matchups_s2.csv","S2"),("matchups_ls.csv","LS")]:
        d=pd.read_csv(PROC/f); d=d[d.matched==1].copy(); d["sensor"]=s; fr.append(d)
    df=prep(pd.concat(fr,ignore_index=True))
    return df[df.secchi.notna()].dropna(subset=F).reset_index(drop=True)
def rf(): return RandomForestRegressor(n_estimators=500,max_depth=12,min_samples_leaf=3,random_state=42,n_jobs=-1)

# ---------- (A) jerarquia de validacion ----------
v=pd.read_csv(MET/"validation_q1.csv")
def get(exp,case): r=v[(v.experiment==exp)&(v.case==case)]; return float(r.R2.iloc[0]),float(r.RMSE.iloc[0])
bars=[("Random\nK-fold",*get("spatial_hierarchy","random_kfold"),"#2e86ab"),
      ("Station-wise\nGroupKFold",*get("spatial_hierarchy","groupkfold_station"),"#2e86ab"),
      ("Temporal\nhold-out",*get("temporal_holdout","train<=2019_test>=2022"),"#edae49"),
      ("Leave-out\nL. Menor",*get("spatial_hierarchy","leave_out_LAGO MENOR"),"#d1495b"),
      ("Leave-out\nBah. Puno",*get("spatial_hierarchy","leave_out_BAHIA PUNO"),"#d1495b"),
      ("Leave-out\nL. Mayor",*get("spatial_hierarchy","leave_out_LAGO MAYOR"),"#d1495b")]
fig,ax=plt.subplots(figsize=(7.2,4.2))
xs=np.arange(len(bars)); r2=[b[1] for b in bars]; cols=[b[3] for b in bars]
ax.bar(xs,r2,color=cols,edgecolor="black",lw=0.6,width=0.7)
for x,b in zip(xs,bars): ax.text(x,max(b[1],0)+0.02,f"{b[1]:.2f}",ha="center",va="bottom",fontsize=9)
ax.axhline(0,color="k",lw=0.8); ax.set_xticks(xs); ax.set_xticklabels([b[0] for b in bars],fontsize=8.5)
ax.set_ylabel("$R^2$ (out-of-fold)"); ax.set_ylim(-0.15,0.75)
ax.set_title("Validation hierarchy: interpolation within sampled optical regimes\nis robust, but spatial extrapolation to an unseen zone collapses")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#2e86ab",label="Within-distribution (no leakage)"),
                   Patch(color="#edae49",label="Unseen years (temporal)"),
                   Patch(color="#d1495b",label="Unseen zone (extrapolation)")],
          frameon=False,fontsize=8,loc="upper right")
fig.tight_layout(); fig.savefig(FIG/"fig_validation_hierarchy.png"); plt.close(fig)
print("[A] fig_validation_hierarchy.png")

# ---------- (B) incertidumbre conformal ----------
s=load(); X=s[F].values; y=s.secchi.values; g=s.station.values; n=len(y)
oof=np.full(n,np.nan); clo=np.full(n,np.nan); chi=np.full(n,np.nan)
ALPHA=0.10
for tr,te in GroupKFold(5).split(X,y,g):
    sc=RobustScaler().fit(X[tr]); Xtr=sc.transform(X[tr]); Xte=sc.transform(X[te])
    rng=np.random.RandomState(42); perm=rng.permutation(len(tr))
    cal=perm[:max(20,len(tr)//5)]; fit=perm[len(cal):]
    m=rf(); m.fit(Xtr[fit],y[tr][fit]); p=m.predict(Xte); oof[te]=np.clip(p,0.05,None)
    q=np.quantile(np.abs(y[tr][cal]-m.predict(Xtr[cal])),1-ALPHA)
    clo[te]=np.clip(p-q,0.05,None); chi[te]=p+q
R2=r2_score(y,oof); RMSE=float(np.sqrt(mean_squared_error(y,oof)))
order=np.argsort(y); cov=float(np.mean((y>=clo)&(y<=chi)))

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,4.4))
# scatter con barras de PI
yerr=np.vstack([oof-clo,chi-oof])
ax1.errorbar(y,oof,yerr=yerr,fmt="o",ms=3.5,alpha=0.35,elinewidth=0.5,color="#2e86ab",ecolor="#9bbfd4",capsize=0)
lim=[0,max(y.max(),chi.max())*1.02]; ax1.plot(lim,lim,"k--",lw=1)
ax1.set_xlim(lim); ax1.set_ylim(lim); ax1.set_aspect("equal")
ax1.set_xlabel("In-situ Secchi depth (m)"); ax1.set_ylabel("Predicted Secchi depth (m)")
ax1.set_title(f"Conformal 90% prediction intervals\n$R^2$={R2:.2f}, RMSE={RMSE:.2f} m, coverage={cov*100:.0f}%")
# curva de calibracion de cobertura (nominal vs empirica) recomputando q por nivel
levels=np.array([0.5,0.6,0.7,0.8,0.9,0.95]); emp=[]
for lv in levels:
    clo2=np.full(n,np.nan); chi2=np.full(n,np.nan)
    for tr,te in GroupKFold(5).split(X,y,g):
        sc=RobustScaler().fit(X[tr]); Xtr=sc.transform(X[tr]); Xte=sc.transform(X[te])
        rng=np.random.RandomState(42); perm=rng.permutation(len(tr))
        cal=perm[:max(20,len(tr)//5)]; fit=perm[len(cal):]
        m=rf(); m.fit(Xtr[fit],y[tr][fit]); p=m.predict(Xte)
        q=np.quantile(np.abs(y[tr][cal]-m.predict(Xtr[cal])),lv)
        clo2[te]=p-q; chi2[te]=p+q
    emp.append(np.mean((y>=clo2)&(y<=chi2)))
ax2.plot([0.4,1],[0.4,1],"k--",lw=1,label="ideal")
ax2.plot(levels,emp,"o-",color="#d1495b",lw=1.8,ms=6,label="conformal (empirical)")
ax2.set_xlabel("Nominal coverage"); ax2.set_ylabel("Empirical coverage")
ax2.set_title("Prediction-interval calibration"); ax2.set_xlim(0.45,1); ax2.set_ylim(0.45,1)
ax2.legend(frameon=False,fontsize=9); ax2.set_aspect("equal")
fig.tight_layout(); fig.savefig(FIG/"fig_uncertainty.png"); plt.close(fig)
print(f"[B] fig_uncertainty.png  (coverage90={cov*100:.1f}%)")
print("Figuras Q1 generadas en", FIG)
