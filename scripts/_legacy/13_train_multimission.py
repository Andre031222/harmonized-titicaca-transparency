"""
==============================================================================
13_train_multimission.py
Modelo HONESTO multi-mision (Sentinel-2 + Landsat 8/9 armonizado) de
transparencia (Secchi) en el Lago Titicaca, + temperatura superficial (termico).

Validacion: GroupKFold por estacion (sin fuga espacial), scaler solo-train.
Comparaciones: S2-only / LS-only / combinado; ML vs algoritmo clasico.
SHAP sobre el mejor modelo. Salidas en results/.
==============================================================================
"""
import pandas as pd, numpy as np, json
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
try:
    from xgboost import XGBRegressor; HAS_XGB=True
except Exception: HAS_XGB=False

ROOT=Path(__file__).resolve().parent.parent
PROC=ROOT/"data/processed"; MET=ROOT/"results/metrics"; FIG=ROOT/"results/figures"
MET.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)

COMMON=["B2","B3","B4","B8","B11","B12"]
def add_idx(d):
    e=1e-9
    d["NDWI"]=(d.B3-d.B8)/(d.B3+d.B8+e); d["NDTI"]=(d.B4-d.B3)/(d.B4+d.B3+e)
    d["B2_B3"]=d.B2/(d.B3+e); d["B4_B3"]=d.B4/(d.B3+e); d["B3_B2"]=d.B3/(d.B2+e)
    d["B2_B4"]=d.B2/(d.B4+e)
    return d
FEATS=COMMON+["NDWI","NDTI","B2_B3","B4_B3","B3_B2","B2_B4"]

def load():
    s2=pd.read_csv(PROC/"matchups_s2.csv"); s2=s2[s2.matched==1].copy(); s2["sensor"]="S2"
    frames=[s2]
    lp=PROC/"matchups_ls.csv"
    if lp.exists():
        ls=pd.read_csv(lp); ls=ls[ls.matched==1].copy(); ls["sensor"]="LS"
        frames.append(ls)
    df=pd.concat(frames,ignore_index=True)
    return add_idx(df)

def cv(model_fn,X,y,g,log=False,k=5):
    gkf=GroupKFold(n_splits=k); yt,yp=[],[]
    for tr,te in gkf.split(X,y,g):
        sc=RobustScaler().fit(X[tr]); m=model_fn()
        m.fit(sc.transform(X[tr]), np.log1p(y[tr]) if log else y[tr])
        p=m.predict(sc.transform(X[te])); p=np.expm1(p) if log else p
        yt.append(y[te]); yp.append(p)
    yt=np.concatenate(yt); yp=np.concatenate(yp)
    return dict(R2=r2_score(yt,yp),RMSE=float(np.sqrt(mean_squared_error(yt,yp))),
                MAE=float(mean_absolute_error(yt,yp)),n=len(yt)),yt,yp

def rf(): return RandomForestRegressor(n_estimators=500,max_depth=12,min_samples_leaf=3,random_state=42,n_jobs=-1)
def xgb(): return XGBRegressor(n_estimators=600,max_depth=5,learning_rate=0.03,subsample=0.8,colsample_bytree=0.8,random_state=42)

def main():
    df=load()
    print(f"[multi] total matchups: {len(df)} | S2={sum(df.sensor=='S2')} LS={sum(df.sensor=='LS')}")
    sec=df[df.secchi.notna()].dropna(subset=FEATS)
    print(f"[secchi] n={len(sec)} | estaciones={sec.station.nunique()} | "
          f"S2={sum(sec.sensor=='S2')} LS={sum(sec.sensor=='LS')}")
    X=sec[FEATS].values; y=sec.secchi.values; g=sec.station.values
    res={}
    res["Classical_B2B3"],_,_=cv(lambda:LinearRegression(),sec[["B2_B3"]].values,y,g,log=True)
    res["Linear_full"],_,_=cv(lambda:LinearRegression(),X,y,g,log=True)
    res["RF_combined"],yt,yp=cv(rf,X,y,g)
    if HAS_XGB: res["XGB_combined"],_,_=cv(xgb,X,y,g)
    # por sensor
    for s in ["S2","LS"]:
        sub=sec[sec.sensor==s]
        if sub.station.nunique()>=5 and len(sub)>=40:
            res[f"RF_{s}_only"],_,_=cv(rf,sub[FEATS].values,sub.secchi.values,sub.station.values)
    print("\n=== Transparencia (Secchi) multi-mision ===")
    for k,v in res.items(): print(f"  {k:20s} R2={v['R2']:+.3f} RMSE={v['RMSE']:.2f}m MAE={v['MAE']:.2f} n={v['n']}")
    pd.DataFrame(res).T.to_csv(MET/"secchi_multimission_metrics.csv")

    # temperatura superficial (termico Landsat)
    if "lst_C" in df.columns:
        tmp=df[(df.lst_C.notna())&(df.temp_insitu.notna())]
        if len(tmp)>=20:
            r=np.corrcoef(tmp.lst_C,tmp.temp_insitu)[0,1]
            rmse=float(np.sqrt(mean_squared_error(tmp.temp_insitu,tmp.lst_C)))
            print(f"\n=== Temperatura superficial (LST termico vs in-situ) ===")
            print(f"  n={len(tmp)} Pearson r={r:.3f} RMSE={rmse:.2f}C bias={float((tmp.lst_C-tmp.temp_insitu).mean()):+.2f}C")
            json.dump({"n":len(tmp),"pearson_r":r,"rmse_C":rmse},open(MET/"lst_validation.json","w"),indent=2)

    # SHAP del RF combinado (sobre todo el set)
    try:
        import shap
        sc=RobustScaler().fit(X); m=rf(); m.fit(sc.transform(X),y)
        ex=shap.TreeExplainer(m); sv=ex.shap_values(sc.transform(X))
        imp=pd.Series(np.abs(sv).mean(0),index=FEATS).sort_values(ascending=False)
        print("\n=== SHAP importancia (Secchi, RF combinado) ===")
        for f,val in imp.items(): print(f"  {f:8s} {val:.3f}")
        imp.to_csv(MET/"secchi_shap_importance.csv")
    except Exception as e:
        print("  (SHAP omitido:",repr(e)[:60],")")

    # scatter
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(5,5))
        ax.scatter(yt,yp,s=16,alpha=0.5,edgecolor="none")
        lim=[0,max(yt.max(),yp.max())*1.05]; ax.plot(lim,lim,"k--",lw=1)
        ax.set_xlabel("Secchi in-situ (m)"); ax.set_ylabel("Secchi predicho (m)")
        ax.set_title(f"RF multi-mision  R2={res['RF_combined']['R2']:.2f}  n={res['RF_combined']['n']}")
        ax.set_xlim(lim); ax.set_ylim(lim); fig.tight_layout()
        fig.savefig(FIG/"secchi_multimission_scatter.png",dpi=150)
    except Exception as e: print("  (sin figura:",repr(e)[:50],")")

if __name__=="__main__":
    main()
